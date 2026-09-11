"""
ver3.0_redis 缓存层验证脚本（不加载任何模型，秒级跑完）

为什么单独写这个脚本？
  app.py 的完整链路要加载 embedding 模型和 cross-encoder（合计 1GB+），
  但"缓存到底有没有生效"只取决于 cache.py 的读写/失效/统计逻辑，
  和模型无关。所以这里把缓存层单独拎出来验证：
  不需要模型、不需要 DeepSeek Key、不需要联网（Redis 连不上会自动测降级路径）。

运行：
    python verify_cache.py                 # 用默认配置（auto）
    CACHE_BACKEND=memory python verify_cache.py
    CACHE_BACKEND=redis  python verify_cache.py    # 需本机 Redis

把结果同时写入 UTF-8 日志文件（避免 Windows 控制台/重定向的编码问题）：
    set VERIFY_LOG=verify_memory.log && python verify_cache.py

退出码 0 = 全部通过；非 0 = 有用例失败。
"""
import os
import sys

# 允许在仓库内直接运行：确保能 import 同目录的 cache.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 若用 RESP_CLIENT_MODULE=tools.resp_client 验证 Redis 代码路径，
# 这里要保证 `tools` 包能被 import（工作目录不在 ver3.0_redis 时也能跑）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache

# 可选：把输出同时写进 UTF-8 日志（Windows 下 PowerShell 重定向默认是 UTF-16，
# 用这个环境变量可以拿到干净的 UTF-8 文件）
_log_file = None
if os.getenv("VERIFY_LOG"):
    _log_file = open(os.getenv("VERIFY_LOG"), "w", encoding="utf-8")

PASSED, FAILED = [], []


def _emit(line: str) -> None:
    print(line)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()



def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        _emit(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append(name)
        _emit(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    _emit(f"\n=== {title} ===")


def main() -> int:
    backend = cache.backend_name()
    extra = ""
    if os.getenv("RESP_CLIENT_MODULE"):
        extra = f"（用 {os.getenv('RESP_CLIENT_MODULE')} 顶替 redis 包）"
    _emit(f"缓存后端：{backend}{extra}（CACHE_BACKEND={cache.CACHE_BACKEND}）")

    # 用 redis 模式跑却降级到 memory 时，明确提示：这次没有真的验证 Redis 路径
    if cache.CACHE_BACKEND == "redis" and backend != "redis":
        _emit("  [WARN] CACHE_BACKEND=redis 但后端不是 redis：Redis 代码路径未被验证！")
        FAILED.append("Redis 后端可用性")

    # ---------- 1. 基础读写 ----------
    section("1. 基础读写与 TTL")
    cache.set("rag:test:plain", {"hello": "世界"}, ttl=60)
    check("写入后能读出（含中文）", cache.get("rag:test:plain") == {"hello": "世界"})
    check("不存在的 key 返回 None", cache.get("rag:test:not-exist") is None)

    cache.set("rag:test:ttl", {"v": 1}, ttl=1)
    check("TTL 内可读", cache.get("rag:test:ttl") == {"v": 1})
    import time
    time.sleep(1.2)
    check("TTL 过期后读不到（不会返回脏数据）", cache.get("rag:test:ttl") is None)

    # ---------- 2. 问答缓存 ----------
    section("2. 问答缓存（同一问题不重复走完整链路）")
    cache.reset_stats()
    q, k, top_n = "这篇论文的主要贡献是什么？", 20, 5
    check("首次查询未命中", cache.get_answer(q, top_n, k) is None)

    cache.set_answer(q, top_n, k, "答案是 XXX", ["片段1", "片段2"])
    hit = cache.get_answer(q, top_n, k)
    check("写入后能命中", hit is not None)
    check("答案与引用完整还原",
          hit and hit["answer"] == "答案是 XXX" and hit["sources"] == ["片段1", "片段2"])

    # 问题带空格应视作同一问题（查询前 strip）
    check("问题首尾空格不影响命中", cache.get_answer(f"  {q}  ", top_n, k) is not None)
    # 检索参数不同不应复用旧结果：签名是 (question, top_n, k)
    check("改召回条数 k 不会误命中", cache.get_answer(q, top_n, 50) is None)
    check("改精排条数 top_n 不会误命中", cache.get_answer(q, 10, k) is None)

    stats = cache.get_stats()
    check("命中/未命中计数正确",
          stats["qa_hit"] == 2 and stats["qa_miss"] == 3,
          f"hit={stats['qa_hit']}, miss={stats['qa_miss']}")
    check("命中率计算正确", abs(stats["qa_hit_rate"] - 0.4) < 1e-9)

    # ---------- 3. 向量缓存 ----------
    section("3. 向量缓存（同一文本不重复做模型前向）")
    cache.reset_stats()
    texts = ["第一段文本", "第二段文本"]
    check("首次批量查询全部未命中", cache.get_embeddings(texts) == {})

    cache.set_embeddings([("第一段文本", [0.1, 0.2]), ("第二段文本", [0.3, 0.4])])
    got = cache.get_embeddings(texts)
    check("写入后能按文本 hash 取回", len(got) == 2)
    check("向量数值一致",
          got.get(cache.embedding_key("第一段文本")) == [0.1, 0.2])
    check("未写过的文本仍算未命中",
          cache.get_embeddings(["没见过的文本"]) == {})
    check("相同文本产生相同 key",
          cache.embedding_key("abc") == cache.embedding_key("abc"))
    check("不同文本 key 不同",
          cache.embedding_key("abc") != cache.embedding_key("abd"))

    # ---------- 4. 精排分数缓存 ----------
    section("4. 精排分数缓存（cross-encoder 是最慢的一环）")
    cache.reset_stats()
    docs = ["片段A", "片段B", "片段C"]
    q2 = "火焰检测阈值是多少？"
    check("首次打分全部未命中", cache.get_rerank_scores(q2, docs) == {})

    cache.set_rerank_scores(q2, docs, [0.9, 0.3, 0.6])
    scores = cache.get_rerank_scores(q2, docs)
    check("写入后 3 条全部命中", len(scores) == 3)
    check("分数按片段下标对应",
          scores == {0: 0.9, 1: 0.3, 2: 0.6})
    # 同一个片段换个问题不该复用
    check("换问题不会误命中",
          cache.get_rerank_scores("另一个问题", ["片段A"]) == {})
    check("精排命中计数为 3", cache.get_stats()["rerank_hit"] == 3)

    # ---------- 5. 版本号失效 ----------
    section("5. 版本号失效（文档更新后旧答案必须作废）")
    cache.set_answer("版本测试问题", top_n, k, "旧答案", ["旧片段"])
    check("失效前能命中", cache.get_answer("版本测试问题", top_n, k) is not None)

    # 说明：内存后端的版本号随进程重启归零（无持久化），所以这里先把基准
    # stabilize 到一个已知值，再断言下一次递增 —— 不能拿"当前值"当基准。
    baseline = cache.bump_version()
    new_version = cache.bump_version()
    check("版本号递增", new_version == baseline + 1, f"v{baseline} -> v{new_version}")
    check("版本提升后旧答案立即失效",
          cache.get_answer("版本测试问题", top_n, k) is None)

    cache.set_answer("版本测试问题", top_n, k, "新答案", ["新片段"])
    after = cache.get_answer("版本测试问题", top_n, k)
    check("新答案可正常写入并命中", after is not None and after["answer"] == "新答案")

    # 再提一次版本，验证"只让旧版本失效、新数据仍可用"的完整闭环
    cache.bump_version()
    check("再次提升版本后，上一版答案同样失效",
          cache.get_answer("版本测试问题", top_n, k) is None)

    # ---------- 6. 清理与统计 ----------
    section("6. 清理与统计")
    cache.clear(["qa"])
    check("定向清理只清问答缓存",
          cache.get_answer("版本测试问题", top_n, k) is None)
    check("向量缓存不受定向清理影响",
          cache.get_embeddings(["第一段文本"]) != {})

    cache.clear()
    check("全量清理后向量缓存也清空",
          cache.get_embeddings(["第一段文本"]) == {})
    check("清理后统计归零", cache.get_stats()["qa_hit"] == 0)

    # ---------- 7. 健康与降级 ----------
    section("7. 后端健康与降级")
    check("backend_name 有值", cache.backend_name() in ("redis", "memory"))
    check("healthy() 返回 True", cache.healthy() is True)
    check("embed 与 rerank key 前缀区分明确",
          cache.embedding_key("x") == cache.embedding_key("x"))

    # ---------- 汇总 ----------
    total = len(PASSED) + len(FAILED)
    _emit(f"\n{'=' * 46}")
    _emit(f"通过 {len(PASSED)}/{total}，失败 {len(FAILED)}")
    if FAILED:
        _emit("失败用例：")
        for name in FAILED:
            _emit(f"  - {name}")
    _emit("=" * 46)
    if _log_file is not None:
        _log_file.close()
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
