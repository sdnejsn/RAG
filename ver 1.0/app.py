import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# 复用你项目里的函数
from main import (
    retrieve, rerank, generate, load_existing_collection,
    load_pdf, split_by_chunk_size, embed_chunk, add_embeddings
)

app = FastAPI(title="RAG Web Service")

# 全局集合，启动时加载
collection = load_existing_collection()

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
            # 先写入临时文件再用 PyMuPDF 读取
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                text = load_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)
        elif ext in (".md", ".markdown", ".txt"):
            text = file_bytes.decode("utf-8")
        else:
            return UploadResponse(success=False, message=f"不支持的文件类型: {ext}，请上传 PDF 或 Markdown 文件")

        if not text.strip():
            return UploadResponse(success=False, message="文件内容为空")

        # 分块 + 嵌入
        chunks = split_by_chunk_size(text)
        if not chunks:
            return UploadResponse(success=False, message="分块后无内容")

        embeddings = [embed_chunk(c) for c in chunks]
        add_embeddings(chunks, embeddings, source_file=filename)

        return UploadResponse(
            success=True,
            message=f"成功解析文件 {filename}，共 {len(chunks)} 个文本块已加入向量库",
            chunk_count=len(chunks)
        )
    except Exception as e:
        return UploadResponse(success=False, message=f"上传失败: {str(e)}")

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """接收问题，返回答案和引用来源。"""
    query = request.query
    # 1. 检索
    retrieved = retrieve([query], top_k=10)
    # 2. 重排序
    reranked = rerank([query], retrieved, top_k=5)
    # 3. 生成回答
    answer = generate(query, reranked[0])
    # 4. 返回答案 + 引用
    return AskResponse(answer=answer, sources=reranked[0])

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """返回前端页面。"""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)