# -*- coding: utf-8 -*-
"""手写向量库：基于 numpy 的余弦相似度检索。
不依赖任何第三方向量数据库，方便理解 RAG 核心原理；
索引以 chunks.json（文本）+ vectors.npy（向量矩阵）落盘，可随时重建。
"""

import json
import os

import numpy as np


class VectorStore:
    def __init__(self, path):
        self.path = path
        self.chunks = []     # 文本块内容
        self.metas = []      # 来源元信息（文件名等）
        self.vectors = None  # 向量矩阵 N x D
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        cpath = os.path.join(self.path, "chunks.json")
        vpath = os.path.join(self.path, "vectors.npy")
        if os.path.exists(cpath) and os.path.exists(vpath):
            with open(cpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            self.metas = data.get("metas", [""] * len(self.chunks))
            self.vectors = np.load(vpath)
            print(f"[VectorStore] 已加载索引：{len(self.chunks)} 个文本块")

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "metas": self.metas},
                      f, ensure_ascii=False, indent=2)
        np.save(os.path.join(self.path, "vectors.npy"), self.vectors)
        print(f"[VectorStore] 索引已保存到：{self.path}（{len(self.chunks)} 块）")

    # ---------- 写入 ----------
    def add(self, chunks, metas, embedder):
        """传入文本块列表、元信息列表、embedder，自动向量化并追加。"""
        vectors = np.array(embedder.embed(chunks), dtype=np.float32)
        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)
        self.metas.extend(metas)

    # ---------- 检索 ----------
    def search(self, query_vec, top_k=4):
        """余弦相似度检索，返回 [(score, text, meta), ...] 降序。"""
        if self.vectors is None or len(self.vectors) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        scores = (self.vectors @ q) / (np.linalg.norm(self.vectors, axis=1) * qn + 1e-9)
        idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.chunks[i], self.metas[i]) for i in idx]
