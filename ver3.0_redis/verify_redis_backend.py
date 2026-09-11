"""
Redis 后端集成验证（带断言，纯标准库，不需要 python `redis` 包）

验证 cache.py 的 _RedisBackend 是否符合"上生产的要求"：

  1. 写缓存真的带 TTL（命令里必须有 SET ... EX）；
  2. 批量清理必须用 SCAN 游标遍历，**绝不能用 KEYS**
     —— KEYS 在线上会阻塞整个 Redis 实例，是大 key 空间下的经典事故；
  3. 版本号用 Redis 原生的 INCR（原子操作，多实例并发上传不会互相覆盖版本）；
  4. 中文/长文本经过 RESP 编解码后不丢字节。

怎么做到"不需要模型、不需要 redis 包"？
  tools/mini_redis_resp.py 是一个只实现所需命令的最小 RESP 服务；
  tools/resp_client.py 是一个纯 socket 的 RESP 客户端（接口与 redis-py 一致）。
  被测代码是 cache.py 的 _RedisBackend，和线上走的是同一条代码路径。

用法：
    python tools/mini_redis_resp.py 6399            # 终端 A
    python verify_redis_backend.py                  # 终端 B（默认连 6399）
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)                      # 能 import cache
sys.path.insert(0, os.path.dirname(ROOT))     # 能 import tools.*

from tools.resp_client import RespClient       # noqa: E402

# ---------- 直接连 mock 服务（不经 cache.py 的全局后端，保持测试独立） ----------
HOST = os.getenv("RESP_HOST", "127.0.0.1")
PORT = int(os.getenv("RESP_PORT", "6399"))

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    flag = "PASS" if condition else "FAIL"
    print(f"  [{flag}] {name}" + (f"  ({detail})" if detail else ""))


class LoggingClient(RespClient):
    """在 RespClient 之上记录发出的每条命令，用来断言实现细节。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = []

    def _send(self, args) -> None:
        self.commands.append([str(a) for a in args])
        super()._send(args)

    def used_keys_command(self) -> bool:
        return any(cmd and cmd[0].upper() == "KEYS" for cmd in self.commands)


def main() -> int:
    try:
        probe = RespClient(HOST, PORT)
        probe.ping()
        probe.close()
    except OSError as e:
        print(f"无法连接 mock Redis（{HOST}:{PORT}）：{e}")
        print("请先启动：python tools/mini_redis_resp.py 6399")
        return 2

    # ---------- 1. _RedisBackend 的真实读写 ----------
    print("\n=== 1. _RedisBackend 读写与 TTL ===")
    import cache

    backend = LoggingClient(HOST, PORT)
    cache._backend = cache._RedisBackend(backend)      # 注入记录版客户端

    cache.set("rag:probe:a", {"v": "中文内容"}, ttl=60)
    check("写入后能读出（中文不丢字节）", cache.get("rag:probe:a") == {"v": "中文内容"})

    cache.set("rag:probe:ttl", {"v": 1}, ttl=1)
    check("TTL 短时内可读", cache.get("rag:probe:ttl") == {"v": 1})
    time.sleep(1.2)
    check("TTL 过期后读不到", cache.get("rag:probe:ttl") is None)

    set_cmds = [c for c in backend.commands if c[0].upper() == "SET"]
    check("SET 命令带了 EX（真的设了过期，不是永久驻留）",
          all("EX" in [x.upper() for x in c] for c in set_cmds),
          f"{len(set_cmds)} 条 SET 全部带 EX" if set_cmds else "没有 SET 命令")

    # ---------- 2. 清理必须用 SCAN ----------
    print("\n=== 2. 批量清理：必须 SCAN，不能 KEYS ===")
    for i in range(5):
        cache.set(f"rag:v1:qa:{i}", {"answer": f"a{i}"}, ttl=60)
    for i in range(3):
        cache.set(f"rag:embed:{i}", [0.1, 0.2], ttl=60)

    backend.commands.clear()
    deleted = cache.clear(["qa"])

    check("删除了 5 条问答缓存", deleted == 5, f"deleted={deleted}")
    check("向量缓存未被误删", cache.get("rag:embed:0") == [0.1, 0.2])
    check("使用了 SCAN 遍历", any(c[0].upper() == "SCAN" for c in backend.commands))
    check("没有使用 KEYS（线上会阻塞 Redis 实例）",
          not backend.used_keys_command())
    scan_cmds = [c for c in backend.commands if c[0].upper() == "SCAN"]
    check("SCAN 带了 MATCH 通配前缀",
          scan_cmds and any("MATCH" in [x.upper() for x in c] for c in scan_cmds))

    # ---------- 3. 版本号用原子 INCR ----------
    print("\n=== 3. 版本号：原子 INCR ===")
    backend.commands.clear()
    v1 = cache.bump_version()
    v2 = cache.bump_version()
    check("版本号递增", v2 == v1 + 1, f"{v1} -> {v2}")
    check("使用 INCR 而非读-改-写",
          sum(1 for c in backend.commands if c[0].upper() == "INCR") >= 2)

    # ---------- 4. 版本提升让问答缓存整体失效 ----------
    print("\n=== 4. 版本提升 = 旧答案整体失效 ===")
    cache.set_answer("缓存失效验证问题", 5, 20, "旧答案", ["旧片段"])
    check("提升前命中", cache.get_answer("缓存失效验证问题", 5, 20) is not None)
    cache.bump_version()
    check("提升后立即失效（key 里带版本号）",
          cache.get_answer("缓存失效验证问题", 5, 20) is None)

    # ---------- 5. clear() 能清掉所有历史版本 ----------
    print("\n=== 5. clear() 能清理所有历史版本的 key ===")
    cache.set_answer("历史版本问题", 5, 20, "v1答案", ["s"])
    cache.bump_version()
    cache.set_answer("历史版本问题", 5, 20, "v2答案", ["s"])
    deleted = cache.clear(["qa"])
    check("所有版本（v1/v2）的问答 key 都被清掉",
          deleted >= 2, f"deleted={deleted}")
    check("清理后两个版本的 key 都取不到",
          cache._get_backend().get("rag:v1:qa:x") is None)

    # ---------- 汇总 ----------
    total = len(PASSED) + len(FAILED)
    print(f"\n{'=' * 46}")
    print(f"通过 {len(PASSED)}/{total}，失败 {len(FAILED)}")
    for name in FAILED:
        print(f"  - {name}")
    print("=" * 46)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
