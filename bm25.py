# -*- coding: utf-8 -*-
"""手写 BM25 关键词检索器。

中文分词优先使用 jieba（按词切分，"录取通知书"→"录取/通知书"）；
jieba 不可用时回退到字符级 unigram+bigram 兜底。

BM25 经典公式（k1=1.5, b=0.75）：
  score(D, Q) = Σ_{q∈Q} IDF(q) * f(q,D)*(k1+1) / (f(q,D) + k1*(1 - b + b*|D|/avgdl))
  IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
"""

import json
import math
import os
from collections import Counter

try:
    import jieba
    jieba.initialize()
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


def _char_tokens(text):
    """字符级 unigram + bigram 兜底分词。"""
    tokens = []
    for i, ch in enumerate(text):
        if not ch.strip():
            continue
        tokens.append(ch)
        if i + 1 < len(text) and text[i + 1].strip():
            tokens.append(ch + text[i + 1])
    return tokens


def tokenize(text):
    """分词：优先 jieba 中文分词，失败回退字符级。"""
    text = (text or "").lower().strip()
    if not text:
        return []
    if _HAS_JIEBA:
        return [w for w in jieba.lcut(text) if w.strip()]
    return _char_tokens(text)


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.metas = []
        self.doc_tokens = []        # 每个文档的 token 列表
        self.doc_freq = Counter()   # 每个 token 出现在多少个文档
        self.avgdl = 0.0
        self.idf = {}
        self._fitted = False

    # ---------- 构建 ----------
    def fit(self, chunks, metas):
        """用全量文本块构建 BM25 索引。"""
        self.chunks = list(chunks)
        self.metas = list(metas)
        self.doc_tokens = [tokenize(c) for c in chunks]
        n = len(self.doc_tokens)
        if n == 0:
            self._fitted = True
            return
        total_len = 0
        for toks in self.doc_tokens:
            total_len += len(toks)
            for t in set(toks):
                self.doc_freq[t] += 1
        self.avgdl = total_len / n
        for t, df in self.doc_freq.items():
            self.idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        self._fitted = True

    # ---------- 检索 ----------
    def search(self, query, top_k=4):
        """BM25 检索，返回 [(score, text, meta), ...] 降序。"""
        if not self._fitted or not self.chunks:
            return []
        q_tokens = set(tokenize(query))
        scores = []
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for qt in q_tokens:
                idf = self.idf.get(qt)
                if idf is None:
                    continue
                f = tf.get(qt, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                score += idf * f * (self.k1 + 1) / denom
            scores.append((score, i))
        scores.sort(key=lambda x: -x[0])
        return [(float(s), self.chunks[i], self.metas[i]) for s, i in scores[:top_k]]

    # ---------- 持久化 ----------
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        data = {
            "k1": self.k1, "b": self.b,
            "chunks": self.chunks, "metas": self.metas,
            "doc_tokens": self.doc_tokens,
            "doc_freq": dict(self.doc_freq),
            "avgdl": self.avgdl, "idf": self.idf,
        }
        with open(os.path.join(path, "bm25.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[BM25] 索引已保存到：{path}（{len(self.chunks)} 块）")

    def load(self, path):
        fpath = os.path.join(path, "bm25.json")
        if not os.path.exists(fpath):
            return False
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        self.chunks = data["chunks"]
        self.metas = data["metas"]
        self.doc_tokens = data["doc_tokens"]
        self.doc_freq = Counter(data["doc_freq"])
        self.avgdl = data["avgdl"]
        self.idf = data["idf"]
        self._fitted = True
        print(f"[BM25] 已加载索引：{len(self.chunks)} 个文本块")
        return True
