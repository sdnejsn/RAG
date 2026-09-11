"""
ver3.0_redis Web 服务 —— 在 ver2.0_LangChain（LangChain 版）基础上加入缓存层

接口对照：
  POST /upload         与 ver2.0_LangChain 一致，额外做一件事：上传成功后失效旧缓存
  POST /ask            与 ver2.0_LangChain 一致，额外返回 cache_hit / cache_type / elapsed_ms
  GET  /cache/stats    新增：查看缓存后端、命中率、当前缓存版本
  POST /cache/clear    新增：手动清空缓存（可选按类型清）
  GET  /health         新增：轻量健康检查
  GET  /               前端页面

设计要点：
  1. 缓存是"加速层"不是"数据源"：清掉缓存系统只会变慢，答案不会变错；
  2. 上传文档会 bump 缓存版本，避免"文档更新了、答案还是旧的"；
  3. 任何缓存异常都 fail-open（读不到就当未命中，照常走真实链路）。
"""
import os
import tempfile
import warnings
from typing import List, Optional

# 屏蔽 langchain-community sunset 的 DeprecationWarning（见 rag_langchain.py 顶部注释）
warnings.filterwarnings("ignore", message=".*langchain-community is being sunset.*")

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader

import cache
from rag_langchain import (
    prompt, llm, load_store, make_retriever, make_chain,
    split_fixed_width, query as rag_query, invalidate_caches,
    RETRIEVE_K, RERANK_TOP_N,
)

app = FastAPI(title="RAG Web Service (LangChain + Cache)")

# 全局对象，启动时懒加载（避免重复加载本地模型）
# 注意：这是"模型/连接复用"，不是缓存——缓存见 cache.py 与 rag_langchain.query()
_store = None
_retriever = None
_chain = None

FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")


def get_store():
    global _store
    if _store is None:
        _store = load_store()
    return _store


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = make_retriever(get_store())     # 召回 RETRIEVE_K + 精排 RERANK_TOP_N
    return _retriever


def get_chain():
    """LCEL 链（检索 -> prompt -> DeepSeek -> 文本）。只构造一次，复用同一实例。"""
    global _chain
    if _chain is None:
        _chain = make_chain(get_retriever())
    return _chain


class AskRequest(BaseModel):
    query: str
    use_cache: bool = True       # 可传 false 强制走真实链路（便于对比耗时）


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    cache_hit: bool = False
    cache_type: str = "miss"
    elapsed_ms: float = 0.0


class UploadResponse(BaseModel):
    success: bool
    message: str
    chunk_count: int = 0
    cache_invalidated: bool = False


class CacheClearRequest(BaseModel):
    kinds: Optional[List[str]] = None    # 空 = 全部；可传 ["qa","embed","rerank"]


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传 PDF 或 Markdown 文件，解析后加入向量库，并失效受影响的缓存。"""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    file_bytes = await file.read()

    try:
        if ext == ".pdf":
            # 先写入临时文件再交给 PyMuPDFLoader
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                text = "".join(p.page_content for p in PyMuPDFLoader(tmp_path).load())
            finally:
                os.unlink(tmp_path)
        elif ext in (".md", ".markdown", ".txt"):
            text = file_bytes.decode("utf-8")
        else:
            return UploadResponse(success=False, message=f"不支持的文件类型: {ext}，请上传 PDF 或 Markdown 文件")

        if not text.strip():
            return UploadResponse(success=False, message="文件内容为空")

        chunks = split_fixed_width(text)
        if not chunks:
            return UploadResponse(success=False, message="分块后无内容")

        # 向量化 + 入库；嵌入会走向量缓存（重复上传同一文件不会重复算向量）
        get_store().add_texts(
            chunks,
            metadatas=[{"source": filename} for _ in chunks],
        )

        # 知识库变了：旧答案必须作废，否则用户会拿到基于旧文档的回答
        invalidated = invalidate_caches()

        return UploadResponse(
            success=True,
            message=(f"成功解析文件 {filename}，共 {len(chunks)} 个文本块已加入向量库"
                     f"（已失效 {invalidated} 条问答/精排缓存）"),
            chunk_count=len(chunks),
            cache_invalidated=True,
        )
    except Exception as e:
        return UploadResponse(success=False, message=f"上传失败: {str(e)}")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """接收问题，先查缓存，未命中再走"检索 -> 精排 -> 生成"。"""
    q = request.query.strip()
    if not q:
        return AskResponse(answer="问题不能为空", sources=[], cache_type="empty")

    result = rag_query(
        q,
        retriever=get_retriever(),
        chain=get_chain(),
        use_cache=request.use_cache,
        k=RETRIEVE_K,
        top_n=RERANK_TOP_N,
    )
    return AskResponse(**result)


@app.get("/cache/stats")
def cache_stats():
    """查看缓存运行状态：后端类型、命中率、当前版本号。"""
    return {
        "backend": cache.backend_name(),
        "healthy": cache.healthy(),
        "version": cache.get_version(),
        "ttl_seconds": {
            "answer": cache.ANSWER_TTL,
            "embedding": cache.EMBEDDING_TTL,
            "rerank": cache.RERANK_TTL,
        },
        "stats": cache.get_stats(),
    }


@app.post("/cache/clear")
def cache_clear(request: CacheClearRequest):
    """手动清空缓存。kinds 为空清全部；传 ['qa'] 可只清问答缓存。"""
    deleted = cache.clear(request.kinds)
    return {"success": True, "deleted": deleted, "kinds": request.kinds or ["qa", "embed", "rerank"]}


@app.get("/health")
def health():
    return {"status": "ok", "cache_backend": cache.backend_name(), "cache_healthy": cache.healthy()}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """返回前端页面。"""
    with open(FRONTEND, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
