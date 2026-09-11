"""
ver4.0 —— LangChain 重构版 RAG 核心管道

对应关系（原 ver3.0/main.py）：
  load_pdf() / 打开 .md           -> PyMuPDFLoader 或直接读文本
  split_by_chunk_size()           -> split_fixed_width()（保留原切法以保证评测口径，
                                     原因见该函数注释）
  SentenceTransformer 嵌入         -> HuggingFaceEmbeddings（底层仍是 sentence-transformers）
  chromadb 客户端/集合/入库        -> langchain_chroma.Chroma + store.add_texts()
  get_all_chunks_from_existed...  -> get_all_chunks_in_order()（按 metadata["index"] 排序，
                                     不再把 id 字符串转 int，ver3 用 uuid 后 int(id) 会崩的坑消失）
  retrieve()                     -> store.as_retriever(search_kwargs={"k": 20})
  rerank()                       -> make_retriever() 内用 HuggingFaceCrossEncoder.score()
                                     精排（LangChain 1.x 已把压缩检索器移出主包，
                                     见 make_retriever 注释）
  OpenAI(base_url=deepseek)       -> ChatDeepSeek
  手拼 prompt / 解析返回 content  -> PromptTemplate + StrOutputParser，整条链用 LCEL
"""
import os
import inspect
import warnings
from typing import List
from dotenv import load_dotenv

# langchain-community 已被官方标记为 sunset（正迁移到独立包），导入时会打印
# DeprecationWarning；当前版本功能不受影响，这里按消息内容屏蔽该噪音（见 README）。
warnings.filterwarnings("ignore", message=".*langchain-community is being sunset.*")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# ---------- 常量 ----------
EMBED_MODEL = "../text2vec-base-chinese"
RERANK_MODEL = "../mmarco-mMiniLMv2-L12-H384-v1"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "default"
DEFAULT_SOURCE = ".doc.pdf"

# ---------- 1. 嵌入模型（原 embed_chunk） ----------
embeddings = HuggingFaceEmbeddings(
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
    if path.lower().endswith(".pdf"):
        return "".join(p.page_content for p in PyMuPDFLoader(path).load())
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_fixed_width(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """按固定长度切块，与原 split_by_chunk_size 逐字符等价。

    为什么不换 RecursiveCharacterTextSplitter？
    LangChain 自带的分割器更"聪明"（按句子/段落边界切），但会改变切块边界，
    使 ver2/ver3 里手标的 true_indices 失效，Hit Rate/MRR 无法和旧版对比。
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
    """重建或追加索引：切块 -> 嵌入 -> 入库（原来 4 个函数合并成 1 个调用）。"""
    if force:
        try:
            load_store().delete_collection()
        except Exception:
            pass  # 集合不存在时忽略，对应原 try/except pass
    store = load_store()
    chunks = split_fixed_width(load_raw_text(source_file))
    if chunks:
        store.add_texts(  # 一次批量嵌入 + 入库，替代原来逐条 add 的循环
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


# ---------- 4 + 5. 检索 + 重排（原 retrieve() + rerank() 合并成一条 retriever） ----------
_cross_encoder = None


def _get_cross_encoder():
    """懒加载 cross-encoder（模型 ~470MB，避免模块导入/建库时白白加载）。"""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    return _cross_encoder


def make_retriever(store: Chroma, k: int = 20, top_n: int = 5):
    """召回 top-k，再用 cross-encoder 精排到 top_n（等价原 retrieve(20) + rerank(5)）。

    说明：LangChain 1.x 起，ContextualCompressionRetriever / CrossEncoderReranker
    被移出 langchain 主包（旧实现只存在于遗留的 langchain_classic 里），
    所以这里直接用 HuggingFaceCrossEncoder.score() 精排，和你的原版 rerank()
    逐行等价。返回的 RunnableLambda 既能被 make_chain 里的 LCEL 管道调用，
    也能直接 .invoke(query) 拿到精排后的 Document 列表。
    """
    base = store.as_retriever(search_kwargs={"k": k})

    def _retrieve_and_rerank(query: str):
        docs = base.invoke(query)
        scores = _get_cross_encoder().score([(query, d.page_content) for d in docs])
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
    """LCEL 一条链：检索 -> 拼上下文 -> prompt -> DeepSeek -> 纯文本。"""
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ---------- 7. 评测（Hit Rate / MRR）----------
# 说明：这是你自己的评测脚本，LangChain 不负责这部分，保留手写。
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

    # 测试问题与标准答案索引：与 ver3.0 完全一致，可直接对比 Hit Rate / MRR
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

    # 批量检索 + 重排（原来是 retrieve() + rerank() 两个函数两段循环）
    retriever = make_retriever(store)
    reranked_results = [
        [d.page_content for d in retriever.invoke(q)] for q in test_questions
    ]

    # 评估效果（口径与 ver3 相同）
    hit_rate, mrr = evaluate_retrieval(test_questions, true_indices,
                                       chunks, reranked_results)
    print(f"\nHit Rate: {hit_rate:.2%}")
    print(f"MRR: {mrr:.2f}")

    # 选一个问题调用 DeepSeek 生成最终回答（原来是手写 generate()；
    # 需要联网且 .env 里有 DEEPSEEK_API_KEY，故默认不执行，和 ver3 一致）
    if input("是否调用 DeepSeek 生成一段示例回答？(y/N): ").strip().lower() == "y":
        chain = make_chain(retriever)
        q = test_questions[10]
        print(f"\n问题：{q}\n最终回答：\n{chain.invoke(q)}")
