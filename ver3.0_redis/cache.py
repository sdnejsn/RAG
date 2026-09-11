"""
ver3.0_redis —— 缓存层（Redis 优先，自动降级本地内存）

设计目标（对应 README 里"缓存"一节的判据）：
  1. 同一个问题不要重复走完整条 RAG 链路（召回 -> 精排 -> 调 DeepSeek）；
  2. 同一段文本不要重复计算 embedding 向量 / 不要重复跑 cross-encoder 打分；
  3. 缓存丢了系统只会"变慢"，不会"变错" —— 所以缓存只是加速层，
     任何一次缓存异常都直接回落到真实计算（fail-open）。

三层结构：
  ┌─────────────┬──────────────────────────┬──────────────────────────┐
  │ 层           │ 存什么                    │ 典型收益                  │
  ├─────────────┼──────────────────────────┼──────────────────────────┤
  │ QA 缓存      │ 问题 -> (答案, 引用片段)   │ 跳过整条链路，秒回         │
  │ embedding 缓存│ 文本 hash -> 向量         │ 跳过模型前向，省 CPU/GPU  │
  │ rerank 缓存  │ (问题,片段) hash -> 分数   │ 跳过 cross-encoder 打分   │
  └─────────────┴──────────────────────────┴──────────────────────────┘

后端选择：
  - CACHE_BACKEND=auto（默认）：能连上 Redis 就用 Redis，连不上自动用本地内存；
  - CACHE_BACKEND=redis：强制 Redis，连不上直接抛错（生产环境用，避免静默降级）；
  - CACHE_BACKEND=memory：只用本地内存（单进程、无外部依赖，适合本地开发演示）。

失效策略：
  - 每条记录都带 TTL（默认 answer 1 小时、embedding 24 小时、rerank 1 小时）；
  - 知识库变化（上传新文档）时调用 bump_version()，让所有旧答案立即作废 —— 
    这是缓存系统里最容易被忽略、也最容易出错的一环：文档更新了但缓存没清，
    用户就会拿到基于旧文档的答案。
"""
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

# ---------- 配置（.env 可覆盖） ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
CACHE_BACKEND = os.getenv("CACHE_BACKEND", "auto").lower()      # auto / redis / memory
ANSWER_TTL = int(os.getenv("CACHE_ANSWER_TTL", 3600))           # 问答结果：1 小时
EMBEDDING_TTL = int(os.getenv("CACHE_EMBEDDING_TTL", 86400))    # 向量：24 小时
RERANK_TTL = int(os.getenv("CACHE_RERANK_TTL", 3600))           # 精排分数：1 小时
VERSION_KEY = "rag:cache_version"

# ---------- 缓存统计（进程内计数器，够用且零成本） ----------
_stats = {"qa_hit": 0, "qa_miss": 0,
          "embed_hit": 0, "embed_miss": 0,
          "rerank_hit": 0, "rerank_miss": 0}
_stats_lock = threading.Lock()


def _bump_stat(key: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[key] = _stats.get(key, 0) + n


def reset_stats() -> None:
    with _stats_lock:
        for k in _stats:
            _stats[k] = 0


def get_stats() -> Dict[str, Any]:
    with _stats_lock:
        snapshot = dict(_stats)
    for name in ("qa", "embed", "rerank"):
        hit, miss = snapshot[f"{name}_hit"], snapshot[f"{name}_miss"]
        total = hit + miss
        snapshot[f"{name}_hit_rate"] = round(hit / total, 4) if total else 0.0
    snapshot["backend"] = backend_name()
    snapshot["asked_pipeline_saved"] = snapshot["qa_hit"]      # 省下的 DeepSeek 调用次数
    return snapshot


# ---------- 本地内存后端（Redis 不可用时的降级方案） ----------
class _MemoryBackend:
    """带 TTL 的进程内字典。单进程可用；多进程/多实例下各自独立，仅作兜底。"""

    name = "memory"

    def __init__(self) -> None:
        self._data: Dict[str, tuple] = {}      # key -> (value, expire_at)
        # 必须是可重入锁：incr() 内部会调用 self.get()，普通 Lock 会自死锁
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expire_at = item
            if expire_at and expire_at < time.time():   # 惰性过期
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            self._data[key] = (value, time.time() + ttl if ttl else 0)

    def delete_prefix(self, prefix: str) -> int:
        """支持 glob 前缀（* 通配），这样不清版本号也能删掉所有历史版本的 key。"""
        import fnmatch
        with self._lock:
            keys = [k for k in self._data if fnmatch.fnmatchcase(k, f"{prefix}*")]
            for k in keys:
                self._data.pop(k, None)
            return len(keys)

    def incr(self, key: str) -> int:
        # 这里必须直接操作 _data，不能用 self.get()：
        # 虽然 RLock 可重入不会死锁，但 get() 会把过期的版本号当成 0 重新计数，
        # 导致版本号回退（回退 = 命中已被作废的旧缓存）。
        with self._lock:
            current = int(self._data.get(key, ("0", 0))[0]) + 1
            self._data[key] = (str(current), 0)     # 版本号不设过期
            return current

    def ping(self) -> bool:
        return True


# ---------- Redis 后端 ----------
class _RedisBackend:
    name = "redis"

    def __init__(self, client) -> None:
        self._client = client

    def get(self, key: str) -> Optional[str]:
        return self._client.get(key)

    def set(self, key: str, value: str, ttl: int) -> None:
        self._client.set(key, value, ex=ttl or None)

    def delete_prefix(self, prefix: str) -> int:
        """用 SCAN 删除，绝不用 KEYS（KEYS 会阻塞整个 Redis 实例）。"""
        deleted = 0
        for key in self._client.scan_iter(match=f"{prefix}*", count=500):
            deleted += self._client.delete(key)
        return deleted

    def incr(self, key: str) -> int:
        return int(self._client.incr(key))

    def ping(self) -> bool:
        return bool(self._client.ping())


_backend = None
_backend_lock = threading.Lock()


def _connect_redis():
    """连接 Redis。

    默认用官方 `redis` 包。如果设置了 RESP_CLIENT_MODULE，则改用该模块的 from_url()：
    这是给"没装 redis 包但要验证 Redis 代码路径"准备的测试钩子
    （见 tools/resp_client.py + tools/mini_redis_resp.py），生产环境不要设置它。
    """
    module_name = os.getenv("RESP_CLIENT_MODULE", "redis")
    module = __import__(module_name, fromlist=["from_url"])
    kwargs = dict(decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
    try:
        # 强制走 RESP2：redis-py 5+ 默认用 RESP3 并在连接时发送 HELLO，
        # 而 Redis 6 以下（含 Windows 版 redis 5.0.x）不认识 HELLO，
        # 会直接抛 ResponseError: unknown command `HELLO`，导致整个 Redis 后端失效。
        client = module.from_url(REDIS_URL, protocol=2, **kwargs)
    except TypeError:
        # 极简 RESP 客户端（tools/resp_client.py）没有 protocol 参数，退回原调用方式
        client = module.from_url(REDIS_URL, **kwargs)
    client.ping()
    return _RedisBackend(client)


def _get_backend():
    """按配置选后端，只初始化一次。auto 模式下 Redis 连不上会安静降级。"""
    global _backend
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is not None:
            return _backend

        if CACHE_BACKEND == "memory":
            _backend = _MemoryBackend()
            print("[cache] CACHE_BACKEND=memory，使用进程内内存缓存")
            return _backend

        try:
            _backend = _connect_redis()
            print(f"[cache] 已连接 Redis：{REDIS_URL}（TTL: 问答 {ANSWER_TTL}s / 向量 {EMBEDDING_TTL}s）")
        except Exception as e:
            if CACHE_BACKEND == "redis":
                raise RuntimeError(f"CACHE_BACKEND=redis 但连接失败：{e}") from e
            _backend = _MemoryBackend()
            print(f"[cache] Redis 不可用（{type(e).__name__}: {e}），已降级为进程内内存缓存；"
                  f"问答结果仍会缓存，但重启即失效、多进程不共享。")
        return _backend


def backend_name() -> str:
    try:
        return _get_backend().name
    except Exception:
        return "unavailable"


def healthy() -> bool:
    try:
        return _get_backend().ping()
    except Exception:
        return False


# ---------- 版本号：知识库变更时整体失效 ----------
def get_version() -> int:
    """当前缓存版本。key 里带版本号 = 一次 incr 让所有旧缓存失效，比逐个删 key 可靠。"""
    try:
        return int(_get_backend().get(VERSION_KEY) or 1)
    except Exception:
        return 1


def bump_version() -> int:
    """上传/删除文档后调用：旧答案立即作废，避免"文档已更新、答案还是旧的"。"""
    try:
        version = _get_backend().incr(VERSION_KEY)
        print(f"[cache] 知识库已变更，缓存版本提升到 v{version}，旧缓存全部失效")
        return version
    except Exception as e:
        print(f"[cache] 版本号提升失败（忽略，不影响主流程）：{e}")
        return 0


# ---------- 通用读写 ----------
def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(raw: Optional[str]) -> Optional[Any]:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def get(key: str) -> Optional[Any]:
    try:
        return _loads(_get_backend().get(key))
    except Exception:
        return None        # fail-open：缓存读失败当作未命中，走真实计算


def set(key: str, value: Any, ttl: int) -> None:
    try:
        _get_backend().set(key, _dumps(value), ttl)
    except Exception:
        pass               # fail-open：写失败不影响返回结果


def make_key(kind: str, *parts: str) -> str:
    """key 格式：rag:v{版本}:{类型}:{内容hash}。带版本号，便于整体失效。"""
    raw = "||".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"rag:v{get_version()}:{kind}:{digest}"


# ---------- 业务封装：① 问答缓存 ----------
def get_answer(question: str, top_n: int, k: int) -> Optional[Dict[str, Any]]:
    """按『问题原文 + 检索参数 + 缓存版本』查缓存。

    带缓存版本是刻意的：上传新文档后版本号 +1，旧答案虽然还在存储里，
    但用新版本号算出的 key 已经不同，等于立即失效（比逐个删 key 可靠，
    也不会出现"漏删"）。参数进 key，改检索/精排参数不会命中旧口径的结果。
    """
    key = make_key("qa", question.strip(), str(k), str(top_n))
    hit = get(key)
    if hit is None:
        _bump_stat("qa_miss")
        return None
    _bump_stat("qa_hit")
    hit["cache_key"] = key
    return hit


def set_answer(question: str, top_n: int, k: int,
               answer: str, sources: List[str]) -> None:
    key = make_key("qa", question.strip(), str(k), str(top_n))
    set(key, {"answer": answer, "sources": sources,
              "ts": time.time(), "question": question.strip()}, ANSWER_TTL)


# ---------- 业务封装：② embedding 缓存 ----------
def get_embeddings(texts: List[str]) -> Dict[str, List[float]]:
    """批量查向量缓存，返回 {文本hash: 向量}（未命中的不在返回值里）。"""
    result: Dict[str, List[float]] = {}
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        value = get(f"rag:embed:{digest}")
        if value is None:
            _bump_stat("embed_miss")
        else:
            _bump_stat("embed_hit")
            result[digest] = value
    return result


def set_embeddings(pairs: List[tuple]) -> None:
    """pairs: [(文本, 向量), ...]。文本 hash 与模型名绑定，换模型不会命中旧向量。"""
    for text, vector in pairs:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        set(f"rag:embed:{digest}", list(vector), EMBEDDING_TTL)


def embedding_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- 业务封装：③ rerank 分数缓存 ----------
def get_rerank_scores(question: str, docs: List[str]) -> Dict[int, float]:
    """按 (问题, 片段) 查精排分数，返回 {片段下标: 分数}。"""
    scores: Dict[int, float] = {}
    for i, doc in enumerate(docs):
        digest = hashlib.sha256(f"{question}||{doc}".encode("utf-8")).hexdigest()
        value = get(f"rag:rerank:{digest}")
        if value is None:
            _bump_stat("rerank_miss")
        else:
            _bump_stat("rerank_hit")
            scores[i] = float(value)
    return scores


def set_rerank_scores(question: str, docs: List[str], scores: List[float]) -> None:
    for doc, score in zip(docs, scores):
        digest = hashlib.sha256(f"{question}||{doc}".encode("utf-8")).hexdigest()
        set(f"rag:rerank:{digest}", float(score), RERANK_TTL)


# ---------- 运维接口 ----------
def clear(kinds: Optional[List[str]] = None) -> int:
    """清空缓存。kinds 为空表示全部；传 ['qa'] / ['embed'] / ['rerank'] 可定向清理。

    问答/精排的 key 带缓存版本号，所以这里用通配符前缀（rag:v*:qa:）
    把**所有历史版本**的记录一并删掉，避免旧版本 key 变成删不掉的垃圾数据。
    向量缓存的 key 不含版本号——向量只取决于文本和模型，和知识库内容无关，
    所以上传新文档不该把它清掉（清了只是白白重算一遍）。
    """
    kinds = kinds or ["qa", "embed", "rerank"]
    prefixes = {"qa": "rag:v*:qa:", "embed": "rag:embed:", "rerank": "rag:rerank:"}
    deleted = 0
    try:
        for kind in kinds:
            prefix = prefixes.get(kind)
            if prefix:
                deleted += _get_backend().delete_prefix(prefix)
    except Exception as e:
        print(f"[cache] 清理失败：{e}")
    reset_stats()
    return deleted
