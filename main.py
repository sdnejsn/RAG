import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import fitz  # PyMuPDF

# ============================================================
# 1. 文档加载与分割
# ============================================================
def load_pdf(file_path: str) -> str:
    """读取 PDF，返回纯文本。"""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_by_blank_line(doc_file: str) -> List[str]:
    """读取文档，按空行分割成文本块。"""
    with open(doc_file, 'r', encoding='utf-8') as file:
        content = file.read()
    return content.split("\n\n")

def split_by_chunk_size(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """按固定长度切块，带重叠。"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ============================================================
# 2. 嵌入模型（文本 → 向量）
# ============================================================
embedding_model = SentenceTransformer("./text2vec-base-chinese")

def embed_chunk(chunk: str) -> List[float]:
    """把单个文本块转成向量。"""
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


# ============================================================
# 3. 向量数据库存储
# ============================================================
CHROMA_DB_PATH = "./chroma_db"

chromadb_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

import uuid

def save_embeddings_persistent(chunks: List[str], embeddings: List[List[float]], metadatas: List[dict] = None) -> None:
    """使用持久化客户端保存向量。"""
    ids = [str(uuid.uuid4()) for _ in chunks]
    chromadb_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

def add_embeddings(chunks: List[str], embeddings: List[List[float]], source_file: str = "") -> None:
    """向已有的向量库追加新文档块。"""
    metadatas = [{"source": source_file} for _ in chunks]
    save_embeddings_persistent(chunks, embeddings, metadatas)

def load_existing_collection():
    """加载已有的向量数据库。"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name="default")

def get_all_chunks_from_existed_collection(collection):
    """从 ChromaDB 集合中提取所有文档，并按 id 升序恢复原始顺序。"""
    data = collection.get(include=["documents"])
    ids = data["ids"]
    docs = data["documents"]

    # 因为 ids 是字符串，转成 int 排序，再取对应文档
    pairs = sorted(zip(ids, docs), key=lambda x: int(x[0]))
    chunks = [doc for _, doc in pairs]
    return chunks

# ============================================================
# 4. 检索（初步召回）
# ============================================================
def retrieve(queries: List[str], top_k: int) -> List[List[str]]:
    """根据问题列表，返回每个问题的 top_k 个相关文本块。"""
    all_retrieved = []
    for q in queries:
        query_embedding = embed_chunk(q)
        results = chromadb_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        all_retrieved.append(results['documents'][0])
    return all_retrieved


# ============================================================
# 5. 重排序（精排）
# ============================================================
cross_encoder = CrossEncoder('./mmarco-mMiniLMv2-L12-H384-v1')

def rerank(queries: List[str], retrieved_results: List[List[str]], top_k: int) -> List[List[str]]:
    """对每个问题及其候选块重排序，返回每个问题最相关的 top_k 个文本块。"""
    all_reranked = []
    for q, chunks in zip(queries, retrieved_results):
        pairs = [(q, chunk) for chunk in chunks]
        scores = cross_encoder.predict(pairs)

        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        top_chunks = [chunk for chunk, _ in scored_chunks][:top_k]
        all_reranked.append(top_chunks)
    return all_reranked


# ============================================================
# 6. 生成回答（调用 DeepSeek API）
# ============================================================
load_dotenv() # 这是一个函数，作用是读取你项目根目录下的 .env 文件，把里面的密钥加载到环境变量里。
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), # os.getenv 是 Python 用来读取环境变量的函数。
    base_url="https://api.deepseek.com"
)

def generate(query: str, chunks: List[str]) -> str:
    """把问题和相关片段打包，调用大模型生成答案。"""
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。
    用户问题: {query}
    相关片段:
    {"\n\n".join(chunks)}
    请基于上述内容作答，不要编造信息。"""
    print(f"{prompt}\n\n---\n")
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


# ============================================================
# 7. 评估，计算Hit Rate和MRR
# ============================================================
def evaluate_retrieval(questions, true_indices, chunks, reranked_results):
    """
    questions: List[str]
    true_indices: List[int]  每个问题对应的标准答案块索引
    chunks: List[str]
    reranked_results: List[List[str]]  已经重排序后的结果，和 questions 一一对应
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
# 主流程
# ============================================================
if __name__ == "__main__":
    rebuild = input("是否重新构建向量库？(y/N): ").strip().lower()

    if rebuild == "y": # 建立向量数据库
        # 删除旧集合，重新创建
        try:
            chromadb_client.delete_collection("default")
        except:
            pass
        chromadb_collection = chromadb_client.get_or_create_collection(name="default")
        # 读取文档
        path = "./doc.pdf"
        if path.lower().endswith(".pdf"):
            raw_text = load_pdf(path)
            chunks = split_by_chunk_size(raw_text)
        else:
            chunks = split_by_blank_line(path)
        # 生成向量
        embeddings = [embed_chunk(chunk) for chunk in chunks]
        # 存入向量库
        save_embeddings_persistent(chunks, embeddings)
        print(f"已构建向量库，共 {len(chunks)} 个文本块。")
        save_chunks_to_file(chunks)
        print("已保存chunks到当前文件夹")
    else:
        # 使用已有向量库
        chromadb_collection = load_existing_collection()
        chunks = get_all_chunks_from_existed_collection(chromadb_collection)
        print(f"使用已有向量库，共 {len(chunks)} 个文本块。")
        save_chunks_to_file(chunks)
        print("已保存chunks到当前文件夹")

    # 定义测试问题（模糊化表述，不直接引用原文）
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
        "人脸识别误报率是多少？"
    ]

    # 对应的答案所在段落号（0-based，与上述问题一一对应）
    true_indices = [
        1,  # 段落1：双节点分布式架构（ESP32-CAM入口，ESP32 Uno室内）
        120,  # 段落120：置信度划分表（>0.6已知，0.5~0.6待确认，<0.5陌生人）
        153,  # 段落153：人脸预检机制 + 置信度阈值提升至0.8
        144,  # 段落144：烟雾阈值1500，通过基线与激励实验确定
        144,  # 段落144：温度阈值1400，同样基于实验数据
        133,  # 段落133：SVGA（800×600）分辨率，权衡内存与识别精度
        173,  # 段落173：人脸0.5s、火焰0.5s、传感器5s，平衡实时性与负载
        177,  # 段落177：已知人员测试中未出现误判为陌生人的帧
        181,  # 段落181：阳光和台灯干扰测试中误报次数为零
        161,  # 段落161：三张表（access_logs, sensor_data, fire_alerts）
        226,
        179
    ]

    # 批量检索 + 重排序
    retrieved_results = retrieve(test_questions, 20)
    reranked_results = rerank(test_questions, retrieved_results, 5)

    # 评估效果
    hit_rate, mrr = evaluate_retrieval(test_questions, true_indices, chunks, reranked_results)
    print(f"\nHit Rate: {hit_rate:.2%}")
    print(f"MRR: {mrr:.2f}")

    # 选一个问题调用DeepSeek生成最终回答
    # index_of_question_to_deepseek=10
    # query = test_questions[index_of_question_to_deepseek]
    # answer = generate(query, reranked_results[index_of_question_to_deepseek])
    # print(f"\n问题：{query}")
    # print(f"最终回答：\n{answer}")
