# RAG 文档问答助手（PDF / Markdown）

一个基于 **检索增强生成（RAG，Retrieval-Augmented Generation）** 的中文文档问答系统：上传 PDF / Markdown / TXT 文档后，系统自动完成 *分块 → 向量化 → 入库 → 检索 → 重排序 → 大模型生成* 的完整流程，最终基于文档内容回答用户问题，并附上引用原文片段。

本仓库包含该系统的 **两个实现版本**，可对照学习 RAG 管道的两种写法：

| 目录 | 定位 | 核心特点 |
|---|---|---|
| [`ver 1.0`](<ver 1.0/README.md>) | 手写 RAG 管道 | 不依赖任何 RAG 框架，用 PyMuPDF / sentence-transformers / ChromaDB / OpenAI SDK 逐环节手写，**便于理解 RAG 原理** |
| [`ver 2.0 ( LangChain优化)`](<ver 2.0 ( LangChain优化)/README.md>) | LangChain 重构版 | 用 **LangChain 1.x 官方组件** 重写同一套管道（嵌入 / 向量库 / 检索 / 精排 / LCEL 生成），代码更简洁，接口与 ver 1.0 完全一致 |

> 💡 版本目录内代码注释中出现的 `ver3.0` / `ver4.0` 等编号为迭代过程中的历史命名，仓库中请以顶层目录 `ver 1.0` / `ver 2.0` 为准。

---

## ✨ 功能特性

- 📄 支持 **PDF、Markdown（`.md` / `.markdown`）、纯文本（`.txt`）** 文件上传
- 🧩 自动分块（默认固定宽度 300 字、重叠 50，可配置）并生成语义向量
- 💾 使用 **ChromaDB 本地持久化** 存储向量，重启不丢失，可增量追加文档
- 🔍 **两阶段检索**：向量相似度初步召回（Top-20）→ Cross-Encoder 精排（Top-5），提升精度
- 🤖 调用 **DeepSeek API** 基于检索片段生成回答，并返回引用来源
- 🌐 简洁的 **Web 界面**（FastAPI + 原生 HTML/CSS/JS），开箱即用
- 📊 内置**离线评测脚本**，输出 Hit Rate / MRR 两项检索指标，方便迭代对比

---

## 🏗️ 系统架构与工作流程

```
                         ┌────────────────────────────────────────────┐
                         │              文档上传 / 离线建库              │
                         └────────────────────────────────────────────┘
   PDF / MD / TXT
        │  ① 文档加载（PyMuPDF）
        ▼
   原始文本 ──► ② 固定宽度切块（300 字 / 重叠 50）
        │  ③ 嵌入：text2vec-base-chinese（本地，中文语义向量，归一化）
        ▼
   ChromaDB 持久化向量库（本地 sqlite，重启不丢）
                              ▲
  用户提问 ──► ④ 语义召回 Top-20      │  ⑤ Cross-Encoder 精排 Top-5
        │                        ────────────────► 相关文档片段
        ▼                                            │
  ⑥ DeepSeek（deepseek-chat）根据「问题 + 片段」生成回答 ──► 答案 + 引用原文
```

核心代码路径为：**加载 → 切块 → 嵌入 → 存储 → 召回 → 重排 → 生成 → 评测**，两个版本在该流程上的对应实现见下方对照表。

---

## 🔄 两版本实现对照（ver 1.0 → ver 2.0）

ver 2.0 用 LangChain 官方组件封装了 ver 1.0 中手写的各个环节，具体替换关系如下：

| ver 1.0 手写代码 | ver 2.0 LangChain 写法 |
|---|---|
| `load_pdf()`（fitz 手写遍历） | `PyMuPDFLoader`（langchain-community） |
| `split_by_chunk_size()` 固定切块 | `split_fixed_width()`（保留原切法，保证评测口径一致） |
| `SentenceTransformer` + `embed_chunk()` | `HuggingFaceEmbeddings(normalize_embeddings=True)` |
| `chromadb.PersistentClient` + 集合管理 | `langchain_chroma.Chroma` |
| `save_embeddings_persistent()` 逐条入库 | `store.add_texts()` 一次批量入库 |
| 按字符串 id 排序恢复顺序 | 按 `metadata["index"]` 排序，不再依赖 id |
| `retrieve()`（手写 query 向量检索） | `store.as_retriever(search_kwargs={"k": 20})` |
| `rerank()`（手写打分排序） | `HuggingFaceCrossEncoder.score()` 精排 |
| `OpenAI(base_url=deepseek)` + 手拼 Prompt | `ChatDeepSeek` + `PromptTemplate` + `StrOutputParser`，整链使用 LCEL |
| `__main__` 串行编排 | LCEL 链式调用 |

> ver 2.0 面向 **LangChain 1.x**：`ContextualCompressionRetriever` / `CrossEncoderReranker` 已移出主包，故重排直接用 `HuggingFaceCrossEncoder.score()` 实现，逻辑与原版逐行等价，详见该目录 README。

---

## 🚀 快速开始

### 0. 准备（两个版本通用）

1. **下载本地模型**（embedding 与重排模型由两个版本共享，放在**仓库根目录**）：

   - 中文嵌入模型 [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese) → 放到仓库根目录，文件夹名保持 `text2vec-base-chinese`
   - 多语言重排模型 [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) → 放到仓库根目录，文件夹名保持 `mmarco-mMiniLMv2-L12-H384-v1`

   （代码中以 `../text2vec-base-chinese`、`../mmarco-mMiniLMv2-L12-H384-v1` 相对引用，即从版本目录向上找到根目录下的模型。）

2. **配置 DeepSeek API Key**：在要运行的版本目录下创建 `.env` 文件：

   ```
   DEEPSEEK_API_KEY=sk-你的实际密钥
   ```

   > 可从 [DeepSeek 开放平台](https://platform.deepseek.com/) 获取。

### 1. 运行 ver 1.0（手写版）

```bash
cd "ver 1.0"
pip install fastapi uvicorn python-dotenv openai sentence-transformers chromadb pymupdf
python app.py        # 打开 http://127.0.0.1:8000
```

### 2. 运行 ver 2.0（LangChain 版）

```bash
cd "ver 2.0 ( LangChain优化)"
pip install -r requirements.txt
python app.py        # 打开 http://127.0.0.1:8000

# 或运行离线评测（需本地文档与已标注的标准答案索引，见下）
python rag_langchain.py
```

> 两个版本的向量库各自生成在各自目录下的 `chroma_db/`，相互独立、互不影响。更多配置细节见各目录内 README：[ver 1.0](<ver 1.0/README.md>) / [ver 2.0](<ver 2.0 ( LangChain优化)/README.md>)。

---

## 📁 目录结构

```
RAG/
├── README.md                         # 本总览文档
│
├── ver 1.0/                          # ── 版本一：手写 RAG 管道
│   ├── app.py                        #    FastAPI 主服务：/upload、/ask、前端页面
│   ├── main.py                       #    核心 RAG 逻辑 + 离线评测主流程
│   ├── index.html                    #    前端交互界面（单文件）
│   └── README.md                     #    版本说明（安装 / 使用 / 高级配置）
│
└── ver 2.0 ( LangChain优化)/         # ── 版本二：LangChain 重构版
    ├── app.py                        #    FastAPI Web 服务（接口与 ver 1.0 一致）
    ├── rag_langchain.py              #    LangChain 核心管道 + 离线评测主流程
    ├── index.html                    #    前端交互界面
    ├── requirements.txt              #    Python 依赖清单
    └── README.md                     #    版本说明（对照表 / 注意事项）

# 以下目录/文件不随仓库提交，需自行准备或由程序自动生成：
text2vec-base-chinese/                # 中文嵌入模型（自行下载，仓库根目录）
mmarco-mMiniLMv2-L12-H384-v1/         # Cross-Encoder 重排模型（自行下载，仓库根目录）
*/chroma_db/                          # ChromaDB 向量库（运行后自动生成）
.env                                  # DeepSeek API Key（自行创建，勿提交）
doc.pdf                               # 内置评测文档（自行准备）
chunks_preview.md                     # 分块预览（自动生成，便于人工标注）
```

---

## 📊 离线评测结果

在包含 **12 个模糊化提问**（问题表述不直接引用原文，贴近真实提问习惯）的测试集上，对内置文档进行检索效果评估：

| 指标 | 结果 | 说明 |
|---|---|---|
| **Hit Rate@5** | **75.00%** | Top-5 内命中标准答案块的比例 |
| **MRR** | **0.41** | 平均倒数排名，衡量答案排位质量 |

该评测展示了系统在"提问与原文表述不一致"的真实场景下的检索有效性，其中 Cross-Encoder 重排序显著提升了相关文档的排名。ver 2.0 使用与 ver 1.0 完全相同的测试集与人工标注索引，可直接对比两个版本的 Hit Rate / MRR。

> ⚠️ 评测依赖本地 `doc.pdf`（一篇硬件安防系统毕业设计文档，**未包含在仓库中**）以及人工标注的标准答案块索引（`true_indices`）。若需复现，请自备文档，先运行脚本生成 `chunks_preview.md` 后人工标注索引。

---

## 🔧 常用可调参数

- **分块参数**：`chunk_size`（默认 300）、`overlap`（默认 50）——ver 1.0 在 `main.py` 的 `split_by_chunk_size`，ver 2.0 在 `rag_langchain.py` 的 `split_fixed_width`
- **召回数量**：ver 1.0 为 `/ask` 中 `retrieve(..., top_k=10)`；ver 2.0 为 `make_retriever(store, k=20)`
- **精排保留数量**：ver 1.0 `rerank(..., top_k=5)`；ver 2.0 `make_retriever(store, ..., top_n=5)`

---

## ⚠️ 上传 GitHub 前的注意事项

1. **切勿提交密钥与大数据文件**。`.env` 含 DeepSeek API Key，务必排除；两个本地模型（合计约 1 GB+）、`chroma_db/`、`doc.pdf`、`chunks_preview.md`、`__pycache__/` 均不应入库。建议在仓库根目录添加 `.gitignore`：

   ```gitignore
   .env
   __pycache__/
   *.pyc
   text2vec-base-chinese/
   mmarco-mMiniLMv2-L12-H384-v1/
   **/chroma_db/
   *.pdf
   chunks_preview.md
   ```

2. **目录名含空格与括号**（如 `ver 2.0 ( LangChain优化)`）：Git / GitHub 支持，但命令行操作需加引号；部分脚本与 CI 工具可能不便，介意的话可重命名目录（代码内路径均为相对引用，重命名不影响运行，仅需同步各目录 README 中的路径说明）。

3. **开源许可**：仓库目前未包含 LICENSE 文件。公开前建议明确许可协议（如 MIT / Apache-2.0），并补一个 LICENSE 文件。

---

## 📄 License

本仓库暂未指定开源许可协议，仅供学习交流使用。作者保留所有权利。
