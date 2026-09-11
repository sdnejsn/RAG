"""
ver3.0_redis —— LangChain 版 RAG 核心管道（在 ver2.0_LangChain 基础上加入缓存层）

与 ver2.0_LangChain 的差异只有"缓存"这一块，管道本身逐行不变：
  ver2.0_LangChain：问题 -> 召回20 -> 精排5 -> 调 DeepSeek      （每次都全跑一遍）
  ver3.0_redis：问题 -> 【问答缓存命中？】是 -> 直接返回
                              └ 否 -> 召回20【向量缓存】-> 精排5【分数缓存】
                                      -> 调 DeepSeek -> 写入问答缓存

三种缓存各自解决一件事：
  问答缓存：同一个问题不重复走完整链路（省 DeepSeek 调用、省 5~15 秒等待）
  向量缓存：同一段文本不重复做模型前向（上传重复内容、重复提问时收益明显）
  精排缓存：(问题,片段) 的 cross-encoder 分数不重复算（cross-encoder 是最慢的一环）

对应关系（原 ver1.0/main.py）：
  load_pdf() / 打开 .md           -> PyMuPDFLoader 或直接读文本
  split_by_chunk_size()           -> split_fixed_width()（保留原切法以保证评测口径）
  SentenceTransformer 嵌入         -> HuggingFaceEmbeddings（底层仍是 sentence-transformers）
  chromadb 客户端/集合/入库        -> langchain_chroma.Chroma + store.add_texts()
  retrieve()                     -> store.as_retriever(search_kwargs={"k": 20})
  rerank()                       -> make_retriever() 内用 HuggingFaceCrossEncoder.score()
  OpenAI(base_url=deepseek)       -> ChatDeepSeek
  手拼 prompt / 解析返回 content  -> PromptTemplate + StrOutputParser，整条链用 LCEL
"""
import os
import inspect
import math
import time
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

# langchain-community 已被官方标记为 sunset（正迁移到独立包），导入时会打印
# DeprecationWarning；当前版本功能不受影响，这里按消息内容屏蔽该噪音。
warnings.filterwarnings("ignore", message=".*langchain-community is being sunset.*")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import cache

load_dotenv()

# ---------- 常量 ----------
EMBED_MODEL = "../text2vec-base-chinese"
RERANK_MODEL = "../mmarco-mMiniLMv2-L12-H384-v1"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "default"
# 默认文档：本目录下的 doc.pdf（注意不是 ".doc.pdf"，那个前导点是笔误，
# 会去找一个隐藏文件，直接回车就会报 "is not a valid file or url"）
DEFAULT_SOURCE = "doc.pdf"

# 检索参数提到模块级：它们要参与缓存 key 计算（改参数不应命中旧口径的答案）
RETRIEVE_K = 20     # 向量召回条数
RERANK_TOP_N = 5    # 精排后保留条数

# 语义缓存开关（默认关闭，见 SemanticCache 注释）
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE", "0") == "1"
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.95"))


# ---------- 1. 嵌入模型（原 embed_chunk）+ 向量缓存 ----------

def _embed_documents_cached(raw_embed_documents, texts: List[str]) -> List[List[float]]:
    """批量嵌入 + 向量缓存：命中的文本不送进模型，只算未命中的。

    这是缓存与"模型单例"最大的区别：模型单例保证模型只加载一次，
    但每次请求的文本仍然要过一遍前向计算；向量缓存则是连计算都省了。
    """
    if not texts:
        return []

    cached = cache.get_embeddings(texts)                    # {文本hash: 向量}
    miss_texts = [t for t in texts if cache.embedding_key(t) not in cached]

    fresh: Dict[str, List[float]] = {}
    if miss_texts:
        vectors = raw_embed_documents(miss_texts)
        fresh = {cache.embedding_key(t): v for t, v in zip(miss_texts, vectors)}
        cache.set_embeddings(list(zip(miss_texts, vectors)))

    # 按原始顺序拼回：缓存命中 + 新计算
    merged = {**cached, **fresh}
    return [merged[cache.embedding_key(t)] for t in texts]


def _embed_query_cached(raw_embed_query, text: str) -> List[float]:
    """单条查询向量 + 缓存（同一问题重复提问时直接复用）。"""
    key = cache.embedding_key(text)
    cached = cache.get_embeddings([text]).get(key)
    if cached is not None:
        return cached
    vector = raw_embed_query(text)
    cache.set_embeddings([(text, vector)])
    return vector


class CachedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """带向量缓存的 HuggingFaceEmbeddings。

    为什么用子类覆写，而不是给实例贴方法（原来的写法）？
    langchain-huggingface 1.x 起 HuggingFaceEmbeddings 改成了 pydantic BaseModel，
    `embeddings.embed_documents = xxx` 会被 pydantic 的 __setattr__ 拦下并抛
    ValueError: object has no field "embed_documents"。
    改成子类覆写后，Chroma 与 retriever 拿到的仍然是同一个 embedding_function，
    缓存照常生效，也不再依赖 monkey patch 这种脆弱写法。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return _embed_documents_cached(super().embed_documents, texts)

    def embed_query(self, text: str) -> List[float]:
        return _embed_query_cached(super().embed_query, text)


embeddings = CachedHuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},   # 对应 normalize_embeddings=True
)


def _chroma_persist_kwargs() -> dict:
    """兼容 langchain-chroma 新旧版本：
    旧版（<1.0）持久化参数是 persist_directory=，新版（>=1.0）改成了 path=。"""
    key = "path" if "path" in inspect.signature(Chroma.__init__).parameters \
        else "persist_directory"
    return {key: CHROMA_DB_PATH}


# ---------- 2. 文档加载与切块 ----------
def load_raw_text(path: str) -> str:
    """读 PDF / Markdown，返回整篇纯文本（行为与原 load_pdf / 直接 read 一致）。"""
    # 先把"文件不存在"挑出来给人话提示：PyMuPDFLoader 对不存在的路径只会抛
    # ValueError: File path xxx is not a valid file or url，看不出是路径写错了还是格式不支持。
    if not path.lower().startswith(("http://", "https://")) and not os.path.isfile(path):
        here = os.path.dirname(os.path.abspath(__file__))
        raise FileNotFoundError(
            f"找不到文档：{path!r}（当前目录：{os.getcwd()}）\n"
            f"  - 若用默认值，请确认 {os.path.join(here, DEFAULT_SOURCE)} 存在；\n"
            f"  - 也可以直接输入绝对路径，例如 {os.path.join(here, 'doc.pdf')}"
        )
    if path.lower().endswith(".pdf"):
        return "".join(p.page_content for p in PyMuPDFLoader(path).load())
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_fixed_width(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """按固定长度切块，与原 split_by_chunk_size 逐字符等价。

    为什么不换 RecursiveCharacterTextSplitter？
    LangChain 自带的分割器更"聪明"（按句子/段落边界切），但会改变切块边界，
    使手标的 true_indices 失效，Hit Rate/MRR 无法和旧版对比。
    因此这里保留原切法——也说明一件事：切块本身很简单，LangChain 真正省的是
    嵌入/入库/检索/重排/生成这些环节，而不是这 8 行 while。
    """
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------- 3. 向量库（原 chromadb 客户端/集合/入库/加载全部被 Chroma 封装） ----------
def load_store() -> Chroma:
    """打开（不存在则创建）持久化向量库。"""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        **_chroma_persist_kwargs(),
    )


def build_index(source_file: str, force: bool = False) -> Chroma:
    """重建或追加索引：切块 -> 嵌入 -> 入库。"""
    if force:
        try:
            load_store().delete_collection()
        except Exception:
            pass  # 集合不存在时忽略
    store = load_store()
    chunks = split_fixed_width(load_raw_text(source_file))
    if chunks:
        store.add_texts(  # 一次批量嵌入 + 入库；嵌入过程自动走向量缓存
            chunks,
            metadatas=[{"index": i, "source": os.path.basename(source_file)}
                       for i in range(len(chunks))],
        )
    print(f"已索引 {len(chunks)} 个文本块 -> {CHROMA_DB_PATH}")
    return store


def get_all_chunks_in_order(store: Chroma) -> List[str]:
    """按 metadata['index'] 还原入库顺序（对应原 get_all_chunks_from_existed_collection）。"""
    data = store.get(include=["documents", "metadatas"])
    pairs = sorted(zip(data["metadatas"], data["documents"]),
                   key=lambda md_doc: md_doc[0].get("index", 0))
    return [doc for _, doc in pairs]


# ---------- 4 + 5. 检索 + 重排（原 retrieve() + rerank()）+ 精排分数缓存 ----------
_cross_encoder = None


def _get_cross_encoder():
    """懒加载 cross-encoder（模型 ~470MB，避免模块导入/建库时白白加载）。"""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    return _cross_encoder


def rerank_cached(query: str, docs: List[str]) -> List[float]:
    """cross-encoder 打分 + 分数缓存。

    为什么这里特别值得缓存？cross-encoder 要把 (问题, 片段) 成对送进模型，
    召回 20 条就要前向 20 次，是整个链路里最慢的一步（本地 CPU 上百毫秒级）。
    缓存键是 (问题, 片段) 的内容哈希，所以同一问题换文档、同一文档换问题都能复用。
    """
    if not docs:
        return []

    hitting = cache.get_rerank_scores(query, docs)          # {下标: 分数}
    miss_idx = [i for i in range(len(docs)) if i not in hitting]

    fresh: Dict[int, float] = {}
    if miss_idx:
        pairs = [(query, docs[i]) for i in miss_idx]
        scores = [float(s) for s in _get_cross_encoder().score(pairs)]
        fresh = dict(zip(miss_idx, scores))
        cache.set_rerank_scores(query, [docs[i] for i in miss_idx], scores)

    merged = {**hitting, **fresh}
    return [merged[i] for i in range(len(docs))]


def make_retriever(store: Chroma, k: int = RETRIEVE_K, top_n: int = RERANK_TOP_N):
    """召回 top-k，再用 cross-encoder 精排到 top_n（等价原 retrieve(20) + rerank(5)）。

    说明：LangChain 1.x 起，ContextualCompressionRetriever / CrossEncoderReranker
    被移出 langchain 主包，所以这里直接用 HuggingFaceCrossEncoder.score() 精排，
    和原版 rerank() 逐行等价。ver3.0_redis 的关键差异是打分改为走 rerank_cached()。
    """
    base = store.as_retriever(search_kwargs={"k": k})

    def _retrieve_and_rerank(query: str):
        docs = base.invoke(query)
        scores = rerank_cached(query, [d.page_content for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]

    return RunnableLambda(_retrieve_and_rerank)


# ---------- 6. 生成（原 generate()） ----------
prompt = PromptTemplate.from_template(
    "你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。\n"
    "用户问题: {question}\n相关片段:\n{context}\n请基于上述内容作答，不要编造信息。"
)
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)   # 原 OpenAI(base_url=...)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def make_chain(retriever):
    """LCEL 一条链：检索 -> 拼上下文 -> prompt -> DeepSeek -> 纯文本。

    输入两种形态，都返回模型答案字符串：
      - 传字符串：链内自己召回（等价原来的写法，适合一次性调用）；
      - 传 {"question", "context"}：**跳过召回**，直接用给好的 context。

    为什么要支持第二种？query() 需要先把召回结果拿出来做向量缓存/精排缓存，
    再调模型。如果这里无条件再召回一次，就等于：
      1. 白跑一遍检索 + 精排（慢一倍）；
      2. 更糟的是——传进去的是 {"question","context"} 字典，retriever 会把这个
         字典当成 query 去 embed_query，直接抛
         AttributeError: 'dict' object has no attribute 'encode'。
    所以召回改成"按需触发"：给了 context 就不再召回。
    """
    def _context_and_question(payload):
        if isinstance(payload, dict):
            return {"question": payload["question"], "context": payload["context"]}
        docs = retriever.invoke(payload)          # 只有传字符串时才召回
        return {"question": payload, "context": format_docs(docs)}

    return (
        RunnableLambda(_context_and_question)
        | prompt
        | llm
        | StrOutputParser()
    )


# ============================================================
# 语义缓存（选配，默认关闭）：把"问法不同但意思相同"的问题也救回来
# ============================================================
class SemanticCache:
    """用嵌入向量做近似问题匹配。

    精确缓存（question 原文当 key）能挡住"重复问同一句话"，但挡不住
    「这篇论文的主要贡献是什么」和「论文主要贡献有哪些」这种换了个说法的问法。
    语义缓存就是把历史问题的向量存下来，新问题先算向量、比余弦相似度，
    超过阈值就复用旧答案。

    为什么默认关闭？
      1. 阈值调高（0.95+）收益有限，调低（0.85）会误命中、答非所问，
         而 RAG 场景下"答错"比"慢一点"严重得多；
      2. 近似匹配天然有误判风险——这正是"缓存丢了系统应该变慢，
         而不是变错"要警惕的地方：缓存可以加速，但绝不能改变正确答案。

    打开方式：.env 里设 SEMANTIC_CACHE=1、SEMANTIC_THRESHOLD=0.95。
    这里用手写向量比较，不引入 Redis 向量检索，保持依赖最小。
    """

    def __init__(self, max_size: int = 256, threshold: float = SEMANTIC_THRESHOLD):
        self.max_size = max_size
        self.threshold = threshold
        self._entries: "OrderedDict[str, Tuple[List[float], Dict[str, Any]]]" = OrderedDict()

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def lookup(self, vector: List[float]) -> Optional[Tuple[Dict[str, Any], float]]:
        best, best_score = None, 0.0
        for _, (vec, payload) in self._entries.items():
            score = self._cosine(vector, vec)
            if score > best_score:
                best, best_score = payload, score
        if best is not None and best_score >= self.threshold:
            return best, best_score
        return None

    def store(self, question: str, vector: List[float], payload: Dict[str, Any]) -> None:
        self._entries[question] = (vector, payload)
        self._entries.move_to_end(question)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)      # 简单 LRU

    def clear(self) -> None:
        self._entries.clear()


_semantic_cache = SemanticCache()


# ---------- 7. 带缓存的问答入口（ver3.0_redis 新增的核心函数） ----------
def query(question: str, retriever, chain,
          use_cache: bool = True,
          k: int = RETRIEVE_K, top_n: int = RERANK_TOP_N) -> Dict[str, Any]:
    """一次问答的完整入口：先查缓存，未命中再走 RAG 链路并回写缓存。

    返回 dict（接口/前端直接消费）：
      answer / sources / cache_hit / cache_type / elapsed_ms
    """
    started = time.perf_counter()

    if use_cache:
        # ① 精确缓存：问题原文一致
        exact = cache.get_answer(question, top_n, k)
        if exact is not None:
            return {"answer": exact["answer"], "sources": exact["sources"],
                    "cache_hit": True, "cache_type": "exact",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}

        # ② 语义缓存（选配）：问题不同但语义相近
        if SEMANTIC_CACHE_ENABLED:
            vector = _embed_query_cached(question)
            similar = _semantic_cache.lookup(vector)
            if similar is not None:
                payload, score = similar
                return {"answer": payload["answer"], "sources": payload["sources"],
                        "cache_hit": True, "cache_type": f"semantic({score:.3f})",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}

    # ③ 真实链路：这里召回一次，然后把结果连同问题一起交给 chain。
    #    注意必须传 dict —— chain 收到 context 后就不再自己召回，
    #    否则会重复检索，并且把整个 dict 当 query 送进 embedding 而报错。
    sources = retriever.invoke(question)
    answer = chain.invoke({"question": question, "context": format_docs(sources)})
    source_texts = [d.page_content for d in sources]

    if use_cache:
        cache.set_answer(question, top_n, k, answer, source_texts)
        if SEMANTIC_CACHE_ENABLED:
            _semantic_cache.store(question, _embed_query_cached(question),
                                  {"answer": answer, "sources": source_texts})

    return {"answer": answer, "sources": source_texts,
            "cache_hit": False, "cache_type": "miss",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


def invalidate_caches() -> int:
    """文档变更后调用：问答/精排缓存全清 + 语义缓存清空 + 版本号提升。"""
    _semantic_cache.clear()
    deleted = cache.clear(["qa", "rerank"])
    cache.bump_version()
    return deleted


# ---------- 8. 评测（Hit Rate / MRR）----------
def evaluate_retrieval(questions, true_indices, chunks, reranked_results):
    """
    questions: List[str]
    true_indices: List[int]  每个问题对应的标准答案块索引
    chunks: List[str]        按入库顺序的全文块
    reranked_results: List[List[str]]  已重排序的结果，和 questions 一一对应
    返回 (hit_rate, mrr)
    """
    hits = 0
    reciprocal_ranks = []

    for q, true_idx, reranked_chunks in zip(questions, true_indices, reranked_results):
        true_chunk = chunks[true_idx]
        found = False
        rank = 0

        for i, chunk in enumerate(reranked_chunks, start=1):
            if chunk == true_chunk:
                found = True
                rank = i
                break

        if found:
            hits += 1
            reciprocal_ranks.append(1 / rank)
            print(f"[命中] {q} -> 排名第 {rank}")
        else:
            reciprocal_ranks.append(0)
            print(f"[未命中] {q}")

    hit_rate = hits / len(questions) if questions else 0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    return hit_rate, mrr


def save_chunks_to_file(chunks: List[str], output_file: str = "./chunks_preview.md") -> None:
    """把文本块和索引保存到 Markdown 文件，方便人工标注标准答案。"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 文本块预览\n\n")
        for idx, chunk in enumerate(chunks):
            f.write(f"## Chunk {idx}\n\n")
            f.write(chunk)
            f.write("\n\n---\n\n")
    print(f"已保存到 {output_file}")


# ============================================================
# 离线主流程（原 main.py 的 __main__）
# ============================================================
if __name__ == "__main__":
    rebuild = input("是否重新构建向量库？(y/N): ").strip().lower() == "y"
    source = input(f"文档路径（默认 {DEFAULT_SOURCE}）: ").strip() or DEFAULT_SOURCE

    store = build_index(source, force=rebuild) if rebuild else load_store()
    chunks = get_all_chunks_in_order(store)
    print(f"向量库中共 {len(chunks)} 个文本块。")
    save_chunks_to_file(chunks)

    # 测试问题与标准答案索引：与旧版完全一致，可直接对比 Hit Rate / MRR
    test_questions = [
        "该系统在硬件层面由哪两个主要节点构成？它们分别负责哪些具体功能？",
        "系统依据什么算法和置信度区间来区分家庭成员与陌生人？具体划分标准是什么？",
        "火焰检测模块采用了哪两种手段来抑制误报？其置信度阈值设为多少？",
        "烟雾传感器的报警触发条件是什么？该阈值是如何通过实验确定的？",
        "温度传感器的报警阈值是多少？该值的设定依据是什么？",
        "ESP32-CAM 模块最终选用了哪种图像分辨率？为何不选用更高或更低的分辨率？",
        "系统对人脸图像、火焰检测图像和传感器数据分别设定了怎样的上传间隔？这样设计有何考虑？",
        "在对已知人员的人脸识别测试中，是否出现过将本人误判为陌生人的情况？统计结果如何？",
        "在抗干扰测试中，系统面对阳光和台灯直射时的表现如何？是否产生了误报？",
        "系统后端数据库包含哪些数据表？请分别说明其存储内容。",
        "作者对于任课老师和同学的情感是什么样的",
        "人脸识别误报率是多少？",
    ]
    true_indices = [1, 120, 153, 144, 144, 133, 173, 177, 181, 161, 226, 179]

    # 第一次跑：精排分数全部未命中（真实计算）；再跑同一批就是缓存命中
    retriever = make_retriever(store)
    print(f"\n缓存后端：{cache.backend_name()}")

    t0 = time.perf_counter()
    reranked_results = [
        [d.page_content for d in retriever.invoke(q)] for q in test_questions
    ]
    t1 = time.perf_counter()
    print(f"[第一次] 检索+精排耗时 {t1 - t0:.2f}s")

    t0 = time.perf_counter()
    _ = [[d.page_content for d in retriever.invoke(q)] for q in test_questions]
    t1 = time.perf_counter()
    print(f"[第二次] 检索+精排耗时 {t1 - t0:.2f}s（精排分数命中缓存）")

    hit_rate, mrr = evaluate_retrieval(test_questions, true_indices,
                                       chunks, reranked_results)
    print(f"\nHit Rate: {hit_rate:.2%}")
    print(f"MRR: {mrr:.2f}")

    print(f"\n缓存统计：{cache.get_stats()}")
不过，其实我感觉我现在做的都是一些小项目，代码量都不高，大多在 10 个文件以内，每个文件的代码量也就几百行
    if input("是否调用 DeepSeek 生成一段示例回答？(y/N): ").strip().lower() == "y":
        chain = make_chain(retriever)
        q = test_questions[10]
        first = query(q, retriever, chain)      # 未命中 -> 调 DeepSeek
        print(f"\n问题：{q}\n最终回答：\n{first['answer']}")
        print(f"（cache_hit={first['cache_hit']}, 耗时 {first['elapsed_ms']}ms）")
        second = query(q, retriever, chain)     # 命中 -> 秒回
        print(f"再问一次：cache_hit={second['cache_hit']}, 耗时 {second['elapsed_ms']}ms")
