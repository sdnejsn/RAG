"""
ver4.0 Web 服务：接口与 ver3.0/app.py 完全一致，内部换成 LangChain 组件。

改动点：
  retrieve() + rerank()  -> get_retriever()（召回20 + cross-encoder 精排5）
  generate()             -> (prompt | llm | StrOutputParser()).invoke({question, context})
  embed + add_embeddings -> store.add_texts()
FastAPI 与前端本身不属于 LangChain 范畴，保留手写。
"""
import os
import tempfile
import warnings
from typing import List

# 屏蔽 langchain-community sunset 的 DeprecationWarning（见 rag_langchain.py 顶部注释）
warnings.filterwarnings("ignore", message=".*langchain-community is being sunset.*")

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader

from rag_langchain import (
    prompt, llm, load_store, make_retriever,
    split_fixed_width, format_docs,
)

app = FastAPI(title="RAG Web Service (LangChain)")

# 全局对象，启动时懒加载（避免重复加载本地模型）
_store = None
_retriever = None


def get_store():
    global _store
    if _store is None:
        _store = load_store()
    return _store


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = make_retriever(get_store())   # 原 retrieve() + rerank()
    return _retriever


def ask(query: str):
    """检索(20) + 重排(5) + 生成，返回 (答案, 引用原文列表)。"""
    sources = get_retriever().invoke(query)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"question": query, "context": format_docs(sources)}   # 原 generate()
    )
    return answer, [d.page_content for d in sources]


class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

class UploadResponse(BaseModel):
    success: bool
    message: str
    chunk_count: int = 0


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传 PDF 或 Markdown 文件，解析后加入向量库。"""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    # 读取文件内容
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

        # 分块 + 嵌入 + 入库（原 embed_chunk + add_embeddings 两步合并）
        chunks = split_fixed_width(text)
        if not chunks:
            return UploadResponse(success=False, message="分块后无内容")

        get_store().add_texts(
            chunks,
            metadatas=[{"source": filename} for _ in chunks],
        )

        return UploadResponse(
            success=True,
            message=f"成功解析文件 {filename}，共 {len(chunks)} 个文本块已加入向量库",
            chunk_count=len(chunks),
        )
    except Exception as e:
        return UploadResponse(success=False, message=f"上传失败: {str(e)}")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """接收问题，返回答案和引用来源。"""
    answer, sources = ask(request.query)
    return AskResponse(answer=answer, sources=sources)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """返回前端页面。"""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
