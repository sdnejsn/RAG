# ver2.0 —— LangChain 精简版 RAG

用 LangChain 官方组件重构 ver1.0 的 RAG 管道（文档加载 → 切块 → 嵌入 → 向量库 →
检索 → cross-encoder 重排 → DeepSeek 生成），评估（Hit Rate / MRR）与
FastAPI Web 接口。

## 文件说明

| 文件 | 作用 |
|---|---|
| `rag_langchain.py` | 核心管道 + 离线评测主流程（对应 ver3.0/main.py） |
| `app.py` | FastAPI Web 服务，接口与 ver3.0/app.py 一致（对应 ver3.0/app.py） |
| `index.html` | 前端页面（与 ver3.0 相同） |
| `requirements.txt` | 依赖清单 |

## 与原版对照（ver1.0/main.py → ver2.0/rag_langchain.py）

| ver1.0 手写代码 | ver2.0 LangChain 写法 |
|---|---|
| `load_pdf()`（fitz 手写遍历） | `PyMuPDFLoader` |
| `split_by_chunk_size()` 固定切块 | 保留 `split_fixed_width()`（见下方"为什么"） |
| `SentenceTransformer` + `embed_chunk()` | `HuggingFaceEmbeddings(normalize_embeddings=True)` |
| `chromadb.PersistentClient` + 集合管理 | `langchain_chroma.Chroma` |
| `save_embeddings_persistent()` 逐条入库 | `store.add_texts()` 一次批量入库 |
| `get_all_chunks_from_existed_collection()` 按 id 排序 | 按 `metadata["index"]` 排序，不再依赖字符串 id |
| `retrieve()`（手写 query_embeddings） | `store.as_retriever(search_kwargs={"k": 20})` |
| `rerank()`（手写打分排序） | `make_retriever()` 内用 `HuggingFaceCrossEncoder.score()` 精排 |
| `OpenAI(base_url=deepseek)` + 手拼 prompt + 取 content | `ChatDeepSeek` + `PromptTemplate` + `StrOutputParser` |
| `__main__` 串行编排 | LCEL 链 / 少量顺序调用 |

## 安装与运行

```bash
cd ver4.0
pip install -r requirements.txt

# 1) 离线评测
python rag_langchain.py

# 2) Web 服务（另开终端）
python app.py                 # 打开 http://127.0.0.1:8000
```

索引库生成在 `ver2.0/chroma_db/`，与 ver1.0 的库相互独立，互不影响。

## 注意事项

1. **切块方式与原版保持一致（固定 300/50）**：ver1 里 Hit Rate / MRR 的
   `true_indices` 是按"固定宽度切块"手标的。若换成 LangChain 更聪明的
   `RecursiveCharacterTextSplitter`，切块边界会变、旧的标准答案索引全部失效，
   需要重新人工标注后才能比较。日常使用（`/ask`、上传）想换分割器只需改
   `split_fixed_width` 这一个函数。
2. **LangChain 只封装了管道，底层没变**：嵌入/重排仍跑你本地的
   sentence-transformers 模型，chroma 仍是本目录的 sqlite 库，模型占用与效果不变。
3. **评测与 Web 层不属于 LangChain 范畴**：`evaluate_retrieval`（Hit Rate/MRR）、
   `app.py`、`index.html` 是业务代码，LangChain（或 LangServe）不负责这些，保留手写。
4. **版本兼容说明**：本代码面向 langchain 1.x。LangChain 1.0
   起，`langchain.retrievers` 里的 `ContextualCompressionRetriever` /
   `CrossEncoderReranker` 被移出主包（旧实现只存在于遗留的 `langchain_classic`），
   因此重排改用 `HuggingFaceCrossEncoder.score()` 直接实现，逻辑与原版一致。
   另外 `langchain-community` 已被官方标记 sunset，当前仍可用；导入它的
   DeprecationWarning 已在代码里按消息屏蔽。未来它停用后，`PyMuPDFLoader` 可换
   独立包 `langchain-pymupdf`，cross-encoder 换对应的独立集成即可。
