# -*- coding: utf-8 -*-
"""RAG 系统全局配置。所有可调参数集中在这里，改完即生效。"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 〇、项目信息
# ============================================================
PROJECT_NAME = "知问 ZhiWen"          # 系统名（想改名只改这一处）
PROJECT_SLOGAN = "基于大模型的 RAG 知识问答系统（混合检索版）"

# ============================================================
# 一、大模型 LLM 配置（OpenAI 兼容接口，可任意切换厂商）
# ============================================================
# 常见厂商的 base_url 示例：
#   DeepSeek    https://api.deepseek.com/v1            model: deepseek-chat
#   通义千问     https://dashscope.aliyuncs.com/compatible-mode/v1   model: qwen-plus
#   智谱 GLM    https://open.bigmodel.cn/api/paas/v4   model: glm-4-flash
#   硅基流动     https://api.siliconflow.cn/v1          model: deepseek-ai/DeepSeek-V3
#   本地 Ollama  http://localhost:11434/v1             model: qwen2.5:7b
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-在这里填你的APIKey")
LLM_MODEL = "deepseek-chat"

# ============================================================
# 二、Embedding 向量化配置（三种模式）
# ============================================================
#   debug : 纯 numpy 哈希向量，零依赖、无需联网，先跑通全流程用
#   local : sentence-transformers 本地模型（免费、离线、中文效果好，推荐）
#   api   : OpenAI 兼容向量接口（如硅基流动 BAAI/bge-m3）
EMBED_MODE = "local"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"      # local 模式模型名（中文小模型，约100MB，离线可用）
EMBED_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", LLM_API_KEY)
EMBED_API_MODEL = "BAAI/bge-m3"

# ============================================================
# 三、切块与检索参数
# ============================================================
CHUNK_SIZE = 500          # 每个文本块约多少字符
CHUNK_OVERLAP = 80        # 相邻块重叠多少字符，避免切断语义
TOP_K = 4                 # 检索召回的最相关块数量
SCORE_THRESHOLD = 0.0     # 最低相似度阈值（可过滤无关命中）

# ============================================================
# 四、混合检索（向量语义 + BM25 关键词）
# ============================================================
USE_HYBRID = True         # True=混合检索；False=仅向量检索（旧行为）
HYBRID_FUSION = "rrf"     # "rrf" 倒数排名融合（推荐，无需调参）/ "weighted" 加权融合
RRF_K = 60                 # RRF 的 k 参数，经典值 60
VECTOR_WEIGHT = 0.5        # weighted 模式下向量检索权重
BM25_WEIGHT = 0.5          # weighted 模式下 BM25 检索权重
BM25_K1 = 1.5              # BM25 的 k1，经典值 1.5
BM25_B = 0.75              # BM25 的 b，经典值 0.75

# ============================================================
# 五、存储路径
# ============================================================
DATA_DIR = os.path.join(BASE_DIR, "data")      # 放原始知识文档（txt/md/pdf）
INDEX_PATH = os.path.join(BASE_DIR, "index")   # 向量索引 + BM25 索引落盘目录
