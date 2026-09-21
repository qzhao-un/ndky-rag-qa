# -*- coding: utf-8 -*-
"""文本向量化模块：三种 Embedding 实现，统一 embed(texts) -> list[list[float]] 接口。

debug : 纯 numpy 哈希向量（零依赖，演示流程用）
local : sentence-transformers 本地模型（推荐，免费离线）
api   : OpenAI 兼容向量接口（如硅基流动 BAAI/bge-m3）
"""

import hashlib
import numpy as np

DIM = 256  # debug 模式向量维度


class _DebugEmbedder:
    """纯 numpy 字符哈希向量。不依赖任何第三方库，仅用于跑通流程。"""

    def embed(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(DIM, dtype=np.float32)
            for ch in t:
                if ch.strip():
                    idx = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16) % DIM
                    v[idx] += 1.0
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            vecs.append(v.tolist())
        return vecs


class _LocalEmbedder:
    """sentence-transformers 本地模型，首次运行会自动下载模型。"""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class _ApiEmbedder:
    """OpenAI 兼容向量接口。"""

    def __init__(self, api_url, api_key, model):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.model = model

    def embed(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def build_embedder(mode, model_name="", api_url="", api_key="", api_model=""):
    """按配置构建 embedder 实例。"""
    mode = (mode or "debug").lower()
    if mode == "local":
        return _LocalEmbedder(model_name or "BAAI/bge-small-zh-v1.5")
    if mode == "api":
        return _ApiEmbedder(api_url, api_key, api_model)
    return _DebugEmbedder()
