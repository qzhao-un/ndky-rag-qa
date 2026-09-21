# -*- coding: utf-8 -*-
"""混合检索器：向量语义检索 + BM25 关键词检索，双路召回后融合。

支持两种融合策略：
  - rrf      ：Reciprocal Rank Fusion（倒数排名融合），无需调参，默认推荐
  - weighted ：归一化后加权线性融合，可通过 vector_weight / bm25_weight 调权

RRF 公式：score(d) = Σ_{retriever} 1 / (k + rank(d))，k 通常取 60。
"""

from bm25 import BM25Index
from retriever import VectorStore


class HybridRetriever:
    def __init__(self, vector_store, bm25_index, fusion="rrf", rrf_k=60,
                 vector_weight=0.5, bm25_weight=0.5):
        self.vec = vector_store
        self.bm25 = bm25_index
        self.fusion = (fusion or "rrf").lower()
        self.rrf_k = rrf_k
        self.vec_w = vector_weight
        self.bm25_w = bm25_weight

    def search(self, question, query_vec, top_k=4,
               vector_top_k=None, bm25_top_k=None):
        """混合检索。
        返回 (final_hits, vec_hits, bm25_hits)，后两者用于调试对比。
        """
        vk = vector_top_k or max(top_k * 2, 8)
        bk = bm25_top_k or max(top_k * 2, 8)

        vec_hits = self.vec.search(query_vec, top_k=vk)
        bm25_hits = self.bm25.search(question, top_k=bk)

        if self.fusion == "weighted":
            merged = self._weighted_fusion(vec_hits, bm25_hits)
        else:
            merged = self._rrf_fusion(vec_hits, bm25_hits)

        merged.sort(key=lambda x: -x[0])
        return merged[:top_k], vec_hits, bm25_hits

    # ---------- RRF 融合 ----------
    @staticmethod
    def _key(text, meta):
        return (text, meta)

    def _rrf_fusion(self, vec_hits, bm25_hits):
        scores = {}
        meta_map = {}
        for rank, (_, text, meta) in enumerate(vec_hits):
            k = self._key(text, meta)
            scores[k] = scores.get(k, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            meta_map[k] = meta
        for rank, (_, text, meta) in enumerate(bm25_hits):
            k = self._key(text, meta)
            scores[k] = scores.get(k, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            meta_map[k] = meta
        return [(s, text, meta_map[(text, meta)]) for (text, meta), s in scores.items()]

    # ---------- 加权融合 ----------
    def _weighted_fusion(self, vec_hits, bm25_hits):
        def norm(hits):
            if not hits:
                return {}
            mx = max(h[0] for h in hits)
            mx = mx if mx > 0 else 1.0
            return {self._key(h[1], h[2]): h[0] / mx for h in hits}

        vs = norm(vec_hits)
        bs = norm(bm25_hits)
        all_keys = set(vs) | set(bs)
        meta_map = {self._key(h[1], h[2]): h[2] for h in vec_hits + bm25_hits}
        return [
            (self.vec_w * vs.get(k, 0.0) + self.bm25_w * bs.get(k, 0.0), k[0], meta_map[k])
            for k in all_keys
        ]
