# RAG 文档问答助手（PDF / Markdown）

基于 **检索增强生成（RAG）** 的智能文档问答系统。支持上传 PDF 或 Markdown 文档，将其内容分块、向量化后存入本地向量数据库（ChromaDB），然后通过 **DeepSeek 大模型** 对用户问题给出基于文档内容的精准回答，并附带引用来源。

---

## ✨ 特性

- 📄 支持 PDF、Markdown（`.md`、`.markdown`）和纯文本（`.txt`）文件上传
- 🧩 自动分块（可配置块大小与重叠）并生成向量
- 💾 使用 ChromaDB 持久化存储向量，重启不丢失
- 🔍 两阶段检索：初步召回（向量相似度）+ 重排序（Cross-Encoder）提升精度
- 🤖 调用 DeepSeek API 生成自然语言回答
- 🌐 简洁的 Web 界面（FastAPI + HTML/CSS/JS），开箱即用

---

## 📊 离线评估结果

在包含 **12 个模糊化提问**（不直接引用原文）的测试集上，基于项目内置文档进行了检索效果评估：

- **Hit Rate@5**：**75.00%**（Top-5 内命中正确答案的比例）
- **MRR**（Mean Reciprocal Rank）：**0.41**

该评估展示了系统在真实场景下（问题表述与原文不完全一致）的检索有效性，重排序模型显著提升了相关文档的排名。

---

## 🛠️ 技术栈

- **后端框架**：FastAPI + Uvicorn
- **向量数据库**：ChromaDB（持久化）
- **嵌入模型**：[text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)（中文语义向量）
- **重排序模型**：[mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)（Cross-Encoder）
- **PDF 解析**：PyMuPDF（fitz）
- **大模型 API**：DeepSeek Chat

---


### 3. 下载模型文件

本项目需要两个预训练模型，请将它们下载到项目根目录下的指定文件夹：

- **嵌入模型**（中文）  
  下载自：[shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)  
  将整个模型文件夹放入项目根目录，并重命名为 `text2vec-base-chinese`（或保持原名，但需在 `main.py` 中修改加载路径）。

- **重排序模型**（多语言，支持中文）  
  下载自：[cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)  
  将整个模型文件夹放入项目根目录，并重命名为 `mmarco-mMiniLMv2-L12-H384-v1`。

### 4. 配置 DeepSeek API Key

在项目根目录创建 `.env` 文件，写入你的 API Key：

```
DEEPSEEK_API_KEY=sk-你的实际密钥
```

> 你可以从 [DeepSeek 官网](https://platform.deepseek.com/) 获取 API Key。

---

## 🚀 运行

启动 Web 服务：

```bash
python app.py
```

服务将在 `http://localhost:8000` 启动。打开浏览器访问该地址即可使用。

---

## 📁 项目文件结构

```
.
├── app.py                 # FastAPI 主服务，包含 /upload 和 /ask 接口
├── main.py                # 核心 RAG 逻辑：加载、分块、嵌入、检索、重排序、生成
├── index.html             # 前端交互界面
├── text2vec-base-chinese/ # 中文嵌入模型（需自行下载）
├── mmarco-mMiniLMv2-L12-H384-v1/ # 重排序模型（需自行下载）
├── chroma_db/             # ChromaDB 持久化数据（自动生成）
├── .env                   # 环境变量（API Key）
└── README.md
```

---

## ⚙️ 高级配置

- **分块参数**：可在 `main.py` 的 `split_by_chunk_size` 函数中调整 `chunk_size` 和 `overlap`。
- **检索数量**：`/ask` 接口中 `top_k` 默认为 10，可在 `app.py` 中修改。
- **重排序保留数量**：`rerank` 函数的 `top_k` 参数（目前为 5）。
