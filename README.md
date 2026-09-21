# 📚 知问 ZhiWen · 基于大模型的 RAG 知识问答系统

> 面向高校招生咨询场景的检索增强问答系统：抓取真实招生问答建库，混合检索召回相关片段，交由大模型流式生成回答。

## 一、项目简介

让大模型基于**自己的知识文档**回答问题。核心做法：先把文档切块并向量化存入向量库；用户提问时用「向量语义 + BM25 关键词」双路检索召回相关片段，RRF 融合后连同问题一起交给大模型，由模型"带着参考资料"作答——既降低幻觉，又能回答私有 / 实时知识。

**技术亮点**

- **混合检索**：bge-small-zh 语义向量 + jieba 中文分词 BM25 双路召回，RRF 倒数排名融合，兼顾语义理解与精确关键词匹配
- **手写核心组件**：NumPy 余弦相似度向量库、BM25 索引、RRF 融合均为手写实现，不依赖 LangChain 等框架黑盒
- **真实场景落地**：爬虫抓取招生网 2158 条真实咨询问答（2020–2026，约 17.5 万字），端到端验证
- **时间近优先**：同一问题存在跨年份不同答复时，自动优先参考最新回复
- **流式输出**：大模型打字机式逐字返回，体验接近 ChatGPT
- **隐私脱敏**：参考依据中的学生姓名自动打码（张\*\*），原始数据不动
- OpenAI 兼容接口，DeepSeek、通义千问、智谱、本地 Ollama 一键切换
- 命令行 + Streamlit 网页双入口，网页端带对话历史

## 二、目录结构

```
rag-system/
├── config.py          # 所有配置：模型、API Key、切块/混合检索参数
├── embedder.py        # 文本向量化（local 本地模型 / api 云端 / debug 兜底）
├── retriever.py       # 手写向量库：存储 + 余弦相似度检索
├── bm25.py            # 手写 BM25（jieba 中文分词，无 jieba 时回退字符级）
├── hybrid.py          # 混合检索：向量 + BM25，RRF / 加权融合
├── ingest.py          # 阶段一：读文档 -> 切块 -> 向量化+BM25 -> 入库（支持 --append 增量）
├── qa.py              # 阶段二：问答主流程（命令行版 + 流式生成器）
├── app.py             # 阶段二：Streamlit 网页版
├── eval.py            # 检索效果评测：纯向量 vs 混合召回率对比
├── eval_report.html   # 评测结果可视化图表（浏览器打开）
├── crawl_ndky.py      # 招生网咨询爬虫（--until-date 按日期停止）
├── parse_qa.py        # 原始问答文本 -> 知识库 + 结构化 JSON + 评测集
├── raw_qa_3years.txt  # 爬虫抓取的 2159 条原始问答（2020-08~2026-05）
├── qa_dataset.json    # 2158 条结构化问答
├── eval_questions.json# 19 条人工标注评测集
├── requirements.txt
├── data/
│   ├── ndky_qa_kb.md   # 知识库（2158 条问答，约 17.5 万字）
│   └── ndky_intro.md   # 学校基本信息（校区、专业、招生电话等）
└── index/             # 向量索引 + BM25 索引 + manifest（ingest 后生成）
```

## 三、快速开始

### 第 1 步：安装依赖

```bash
pip install -r requirements.txt
```

### 第 2 步：构建知识库索引

```bash
python ingest.py            # 全量重建（改了切块/embedding 后用）
python ingest.py --append   # 增量追加（只处理新增文件，见第六节）
```

默认 `EMBED_MODE = "local"`，使用 `BAAI/bge-small-zh-v1.5` 中文语义模型（约 100MB，首次运行自动下载，之后离线可用）。国内网络下载慢可设镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python ingest.py
```

### 第 3 步：配置大模型

在 `config.py` 填入 OpenAI 兼容的 API Key（以 DeepSeek 为例，便宜且中文好）：

```python
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_API_KEY  = "sk-你的key"
LLM_MODEL    = "deepseek-chat"
```

未配置 Key 时系统自动降级为只展示检索结果，不报错。

### 第 4 步：问答

```bash
# 命令行
python qa.py "宿舍是几人间？"

# 网页版（默认 http://localhost:8501）
streamlit run app.py
```

## 四、配置说明（config.py）

| 配置项 | 说明 | 示例 |
|---|---|---|
| `LLM_BASE_URL` | OpenAI 兼容接口地址 | DeepSeek: `https://api.deepseek.com/v1`；Ollama: `http://localhost:11434/v1` |
| `LLM_MODEL` | 模型名 | `deepseek-chat` / `qwen-plus` / `glm-4-flash` / `qwen2.5:7b` |
| `EMBED_MODE` | `local` 本地模型（推荐）/ `api` 云端 / `debug` 零依赖兜底 | 推荐 `local` |
| `CHUNK_SIZE` | 切块字符数（问答文档按咨询边界切，此值为打包上限） | 500 |
| `TOP_K` | 召回条数 | 4 |
| `HYBRID_FUSION` | `rrf` 倒数排名融合（默认）/ `weighted` 加权融合 | `rrf` |

## 五、系统工作原理

```
【离线索引】 文档 -> 按咨询边界切块 -> [bge 向量化 + jieba BM25 建索引] -> 向量库

【在线问答】 问题 -> 向量化 -> 向量检索 Top-K + BM25 检索 Top-K
              -> RRF 融合 -> 按咨询日期近优先重排 -> 拼 Prompt -> LLM 流式生成
```

### 5.1 混合检索（Hybrid Search）

| 检索方式 | 擅长 | 短板 |
|---|---|---|
| 向量语义检索（bge-small-zh） | 理解同义、近义、语义相关 | 对精确关键词（编号、术语）不敏感 |
| BM25 关键词检索（jieba 分词） | 精确匹配关键词、术语、编号 | 不理解语义，同义词召回差 |

**RRF 融合**：`score = Σ 1/(k + rank)`，k=60，无需调参，业界最常用。关闭混合检索把 `USE_HYBRID` 改为 `False` 即回退纯向量。

### 5.2 检索效果评测（19 条人工标注问题，845 块索引）

| 指标 | 纯向量检索 | 混合检索 | 提升 |
|---|---|---|---|
| Top-1 召回率 | 57.9% | 57.9% | — |
| Top-2 召回率 | 63.2% | **68.4%** | +5.2 pp |
| Top-4 召回率 | 63.2% | **89.5%** | **+26.3 个百分点** |

> 数据由 `python eval.py eval_questions.json` 真实跑出，可复现；可视化见 `eval_report.html`（浏览器打开）。
>
> **关键发现**：真实语义向量（bge）下纯向量 Top-1 已达 57.9%，但 Top-4 停在 63.2%；加入 BM25 精确匹配并 RRF 融合后 Top-4 提升到 89.5%。典型案例："怎么知道自己是几班的"，纯向量漏检，BM25 通过"几班"关键词成功召回——说明关键词检索能有效弥补语义检索在精确词项上的漏检。

## 六、知识库更新

- **新增一批问答**：把新文件（如 `data/raw_qa_2027.txt`）放进 `data/`，跑 `python ingest.py --append`。系统通过 `index/manifest.json` 记录每个文件的大小和修改时间，只对新增/变化的文件切块向量化，旧索引不动。
- **改了切块策略或 embedding 模型**：跑 `python ingest.py`（不带参数）全量重建。
- **重新抓取招生网**：`python crawl_ndky.py --until-date 2026-09-01 --pages 300 --output raw_qa_new.txt`，再用 `parse_qa.py` 解析入库。

## 七、部署

### 本机访问

```bash
streamlit run app.py
# 浏览器打开 http://localhost:8501
```

### 局域网访问（同 WiFi 手机/其他电脑可开）

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

启动日志会打印三个地址：
- `Local URL`：本机访问
- `Network URL`：局域网访问（如 `http://10.x.x.x:8501`），手机连同一 WiFi 即可打开
- `External URL`：外网地址（需路由器端口映射，谨慎开放）

### 后台常驻（Windows）

```powershell
Start-Process python -ArgumentList "-m","streamlit","run","app.py","--server.headless","true" -WindowStyle Hidden
```

### 安全注意

- `config.py` 里的 API Key 是明文，**传到 GitHub 前务必改回占位符或用环境变量**：`$env:LLM_API_KEY="sk-..."`（代码已支持环境变量优先）。
- 页面底部已加免责声明：AI 回答仅供参考，最终以招生办官方回复为准。

## 八、进阶改造方向

1. ~~混合检索~~ ✅ 向量 + BM25 + RRF
2. ~~真实语义 Embedding~~ ✅ bge-small-zh
3. ~~中文分词~~ ✅ jieba
4. ~~流式输出 / 时间近优先 / 隐私脱敏 / 增量更新~~ ✅
5. **重排序（Rerank）**：召回后用 bge-reranker 精排，进一步提升 Top-1
6. **多轮对话**：用 session 管理上下文，支持追问
7. **RAGAS 评测**：接入忠实度、答案相关性等自动化指标
8. **查询改写**：用大模型把口语化问题改写成更适合检索的形式
