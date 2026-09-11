# ver3.0_redis —— RAG + 缓存层（Redis）

在 [`ver2.0_LangChain`](<../ver2.0_LangChain/README.md>) 的基础上，**只加了"缓存"这一层**：
RAG 管道（加载 → 切块 → 嵌入 → 入库 → 召回 → 精排 → 生成）逐行未变，接口与前端保持向后兼容，
新增的是"同一个问题、同一段文本，不要重复算第二遍"的能力。

> 本版本对应迭代计划里的两个主题：**缓存（本次完成）** 与 **消息队列（预留，见文末）**。

---

## 一、为什么 RAG 特别需要缓存

RAG 的一次问答要串行做四件重活：

| 环节 | 成本 | 是否可复用 |
|---|---|---|
| 问题向量化 | 模型前向一次，几十毫秒 | ✅ 同一个问题完全可复用 |
| 向量召回 | Chroma 查一次 | ✅ 结果确定性，可复用 |
| Cross-Encoder 精排 | **召回 20 条就要前向 20 次**，本地 CPU 上百毫秒级 | ✅ (问题, 片段) 组合可复用 |
| DeepSeek 生成 | **5~15 秒**，且**按 token 计费** | ✅ 同一问题答案可复用 |

前三个环节是"算力"，最后一个是"钱 + 时间"。而真实使用中，用户重复提问、多人问同一个问题、
刷新页面重问，都是非常常见的——这些请求**全部在重算一模一样的东西**。

ver2.0_LangChain 里的"模型懒加载 + 全局单例"解决的是"模型不要重复加载"，
但**每次请求的文本依然要完整走一遍前向计算**。这正是"模型单例 ≠ 缓存"的地方：

| | 模型单例（ver2.0_LangChain 已有） | 缓存（ver3.0_redis 新增） |
|---|---|---|
| 解决什么 | 同一进程内不重复**加载模型** | 同一输入不重复**计算结果** |
| 丢掉会怎样 | 变慢（重新加载 1GB 模型） | 变慢（重算一遍），**不会变错** |
| 跨进程/多实例 | 每个进程各一份 | Redis 下全局共享 |

---

## 二、三层缓存设计

```
用户提问
   │
   ├─① 问答缓存（问题原文 + 检索参数 + 缓存版本  →  答案 + 引用片段）
   │      命中 ──► 直接返回（不查库、不精排、不调 DeepSeek）
   │      未命中
   ▼
   ② 向量缓存（文本 hash → 向量）──► 只把未命中的文本送进 embedding 模型
   ▼
   ③ 精排分数缓存（(问题,片段) hash → 分数）──► 只对未算过的组合跑 cross-encoder
   ▼
   DeepSeek 生成  ──►  回写 ①
```

三层各自独立生效，任意一层命中都能省掉对应环节，测试与观察也互相不干扰。

### ① 问答缓存 —— 省时间、省钱

- **Key**：`rag:v{版本}:qa:{sha256(问题.strip() + k + top_n)}`
- **Value**：`{answer, sources, ts, question}`
- **TTL**：默认 1 小时（`CACHE_ANSWER_TTL`）
- 问题会先 `strip()`，避免"末尾多个空格就当成新问题"；
- **检索参数进 key**：改了 `k` 或 `top_n` 不会命中旧口径的答案——否则调参后拿到旧结果，会误以为参数没生效。

### ② 向量缓存 —— 省算力

包装 `HuggingFaceEmbeddings` 的 `embed_documents` / `embed_query`，**只把未命中的文本送进模型**：

```python
cached = cache.get_embeddings(texts)                       # 命中的直接拿
miss_texts = [t for t in texts if key(t) not in cached]    # 未命中的才算
```

- **Key**：`rag:embed:{sha256(文本)}`（**不含缓存版本号**）
- **TTL**：默认 24 小时（`CACHE_EMBEDDING_TTL`）
- 为什么不含版本号：向量只取决于"文本 + 模型"，与知识库内容无关。上传新文档时清掉向量缓存
  纯属浪费——同一段文本下次还要重算。**只有换 embedding 模型时才需要清**。

### ③ 精排分数缓存 —— 省最慢的一环

cross-encoder 要把 `(问题, 片段)` 成对送进模型，召回 20 条就是 20 次前向。缓存键是二者内容哈希，
所以"同一问题换文档"、"同一文档换问题"都能部分复用。

- **Key**：`rag:rerank:{sha256(问题 + "||" + 片段)}`
- **TTL**：默认 1 小时（`CACHE_RERANK_TTL`）

---

## 三、最容易写错的两件事，以及本版本的解法

### 1. 失效：文档更新了，答案还是旧的

上传新文档后，向量库变了，但缓存里还躺着**基于旧文档生成的答案**——
用户会拿到一个"看起来很正常但是错的"回答。这是缓存系统里最典型的事故。

本版本的解法是**版本号**（`cache.py` 的 `bump_version()`）：

- 所有问答/精排 key 都带 `v{版本}`；上传成功后版本号 +1；
- 旧 key 瞬间"再也拼不出来"，等于全部失效——**比逐个删 key 更可靠**，不存在漏删；
- `clear()` 用通配前缀 `rag:v*:qa:` 清理**所有历史版本**，避免旧 key 变成删不掉的垃圾。

对照 `verify_cache.py` 第 5 节，这部分有 5 条专门用例覆盖。

### 2. 边界：缓存只能让系统"变慢"，不能让它"变错"

这是判断一个缓存写得对不对的标准。落实到代码里：

- 任何缓存读写异常都 **fail-open**：读不到就当未命中，照常走真实链路，绝不因为缓存挂了而报错；
- 答案缓存**不是唯一数据源**：清空缓存后重新提问，答案应该完全一致；
- **语义缓存默认关闭**。它能让"换个说法问同一问题"也命中，但近似匹配天然会误命中，
  阈值调低（如 0.85）时很容易答非所问。在 RAG 场景下，"答错"比"慢几秒"严重得多，
  所以默认不开，要开请显式设置：

  ```bash
  SEMANTIC_CACHE=1
  SEMANTIC_THRESHOLD=0.95    # 阈值越接近 1 越保守
  ```

---

## 四、降级策略：没有 Redis 也能跑

缓存是加速层，**不该成为单点依赖**。`cache.py` 按配置选后端：

| `CACHE_BACKEND` | 行为 | 适用 |
|---|---|---|
| `auto`（默认） | 能连 Redis 就用 Redis，连不上自动用进程内内存 | 开发/演示 |
| `redis` | 强制 Redis，连不上**直接报错** | 生产（避免静默降级后多实例缓存不共享） |
| `memory` | 只用进程内内存，零外部依赖 | 单机/离线调试 |

> ⚠️ 内存后端的局限要说清楚：**只在单进程内有效，重启即失效，多实例之间不共享**。
> 它保证"没有 Redis 时功能不受影响"，但拿不到 Redis 那种跨进程/跨实例的收益。

---

## 五、接口变化（相对 ver2.0_LangChain）

| 接口 | 变化 |
|---|---|
| `POST /upload` | 响应新增 `cache_invalidated`；成功后自动提升缓存版本 |
| `POST /ask` | 响应新增 `cache_hit` / `cache_type` / `elapsed_ms`；请求可传 `use_cache: false` 强制重算 |
| `GET /cache/stats` | **新增**：后端类型、命中率、TTL、当前缓存版本 |
| `POST /cache/clear` | **新增**：清空缓存，可传 `{"kinds": ["qa"]}` 定向清理 |
| `GET /health` | **新增**：轻量健康检查 |

前端的"提问"旁边多了一个 **跳过缓存** 勾选框：勾上走完整链路，不勾走缓存，
方便你亲眼对比"命中缓存"与"真实计算"的耗时差（答案区会显示徽标与毫秒数）。

---

## 六、安装与运行

```bash
cd ver3.0_redis
pip install -r requirements.txt

# 1) 配置（与 ver2.0_LangChain 相同的 DEEPSEEK_API_KEY，加缓存相关项）
copy .env .env        # Windows；Linux/macOS 用 cp

# 2) 启动 Redis（可选，没有也能跑，会自动降级为内存缓存）
redis-server                  # 本机默认 127.0.0.1:6379

# 3) 启动服务
python app.py                 # http://127.0.0.1:8000
```

模型文件（`../text2vec-base-chinese`、`../mmarco-mMiniLMv2-L12-H384-v1`）与 ver2.0_LangChain 共用，
放在仓库根目录即可，见顶层 README。

### 验证缓存层（不需要模型、不需要联网）

```bash
python verify_cache.py                       # 默认 auto
CACHE_BACKEND=memory python verify_cache.py   # 只测内存后端
CACHE_BACKEND=redis  python verify_cache.py   # 只测真实 Redis
```

Windows 下想拿到干净的 UTF-8 日志（PowerShell 重定向默认是 UTF-16）：

```powershell
$env:VERIFY_LOG="verify_memory.log"; python verify_cache.py
```

覆盖 7 组共 35 条用例：基础读写/TTL、问答缓存、向量缓存、精排缓存、**版本号失效**、
定向清理与统计、后端健康。退出码 0 表示全部通过。

实测结果（本机）：

| 后端 | 结果 |
|---|---|
| `memory`（进程内内存） | **通过 35/35，失败 0** |
| `redis`（真实 RESP 协议往返） | **通过 35/35，失败 0** |

> 注意：`CACHE_BACKEND=redis` 时如果连不上 Redis，脚本会**直接报错而不是静默降级**，
> 并在结果里标记 `Redis 后端可用性` 失败 —— 避免"以为测了 Redis，其实测的是内存"。

### 验证 Redis 后端的关键实现细节

`verify_redis_backend.py` 会断言几条"上生产才看得出来"的细节：

```bash
python tools/mini_redis_resp.py 6399    # 终端 A：最小 RESP 服务（纯标准库）
python verify_redis_backend.py          # 终端 B
```

它验证：

1. **写缓存真的带 TTL** —— 命令里必须有 `SET ... EX`，否则数据永久驻留；
2. **批量清理必须用 `SCAN` 游标遍历，绝不能用 `KEYS`** ——
   `KEYS` 在大 key 空间下会阻塞整个 Redis 实例，是线上经典事故；
3. **版本号用原生 `INCR`** —— 原子操作，多实例并发上传不会互相覆盖版本号；
4. 中文/长文本经过 RESP 编解码后不丢字节；
5. `clear()` 能清掉**所有历史版本**的 key（不只是当前版本）。

实测结果：**通过 15/15，失败 0**。

> 为什么不用 python `redis` 包跑这两个脚本？
> `tools/resp_client.py` 是一个纯标准库的 RESP 客户端，接口与 redis-py 一致，
> 配合 `tools/mini_redis_resp.py` 就能在没有 `redis` 包的环境下真实跑通
> `cache.py` 的 `_RedisBackend`（协议是真的，只有客户端实现是极简版）。
> 装了 `redis` 包时，直接 `pip install -r requirements.txt` 后用 `CACHE_BACKEND=redis` 跑即可，
> 不需要 `RESP_CLIENT_MODULE` 这个测试钩子（默认就是官方包）。


### 观察缓存效果

启动服务后重复问同一个问题，或在页面上点 **刷新** 看缓存统计：

```bash
curl http://127.0.0.1:8000/cache/stats
```

预期：第一次 `cache_hit=false`（耗时实打实），第二次 `cache_hit=true, cache_type="exact"`（毫秒级返回）。

---

## 七、相对 ver2.0_LangChain 的代码改动

| 文件 | 说明 |
|---|---|
| `cache.py` | **新增**。三种缓存的读写封装 + Redis/内存双后端 + 版本号失效 + 命中统计 |
| `rag_langchain.py` | 嵌入方法包一层向量缓存；`rerank()` → `rerank_cached()`；新增 `query()`（带缓存的问答入口）、`invalidate_caches()`、`SemanticCache` |
| `app.py` | `/ask` 改走 `query()`；`/upload` 成功后失效缓存；新增 3 个运维接口；前端路径改到 `templates/` |
| `templates/index.html` | 新增缓存命中徽标、耗时、跳过缓存开关、缓存统计面板 |
| `verify_cache.py` | **新增**。缓存层验证脚本（35 条用例，不依赖模型） |
| `verify_redis_backend.py` | **新增**。Redis 后端关键实现断言（TTL / SCAN / INCR，15 条用例） |
| `tools/resp_client.py`、`tools/mini_redis_resp.py` | **新增**。纯标准库的最小 RESP 客户端/服务，用于在没有 `redis` 包时验证 Redis 路径 |
| `requirements.txt` | 新增 `redis>=5.0`（不装也能跑，会自动降级 memory） |
| `.env.example`、`.gitignore` | **新增**。缓存配置说明 / ver3.0_redis 局部忽略规则 |

**没有改的**：切块逻辑、Chroma 用法、LCEL 链、评测口径 —— 所以 Hit Rate / MRR 与
ver2.0_LangChain / ver3.0_redis 可直接对比（缓存不影响检索结果，只影响要不要重算）。

---

## 八、下一步：消息队列（尚未实现）

缓存解决的是"**同样的活不要干第二遍**"，但它不解决另外两个问题：

1. **上传阻塞**：`/upload` 现在仍在请求内同步完成解析、切块、嵌入、入库，
   一本大 PDF 用户要等几十秒；
2. **削峰**：多人同时上传时，请求会堆积在 FastAPI 进程里，把 CPU 占满。

消息队列解决的是"**把必须马上做的事，改成排队慢慢做**"。改造方向：

```
现在：/upload ──► 解析+嵌入+入库（同步，用户等待）──► 返回

改造：/upload ──► 存文件 + 往队列扔一条消息 ──► 立刻返回 task_id
                 （Redis Stream / RabbitMQ）
                        │
                  worker 进程消费 ──► 解析+嵌入+入库 ──► 更新任务状态
                        │
       前端拿 task_id 轮询 GET /status/{task_id}
```

届时缓存层需要配合的一点：**入库完成后再 `bump_version()`**——
现在的失效点在"收到文件"时，改成异步后应挪到"worker 入库成功"时，
否则任务还在排队、用户就已经拿不到旧答案了（旧答案其实仍然正确）。

> LLM 缓存这类"边界"要格外小心：它必须可随时清空且不改变正确答案，
> 这也是本版本坚持"缓存只是加速层、fail-open、版本号失效"的原因。
