# -*- coding: utf-8 -*-
"""检索评测（标准 IR 口径）：纯向量 vs 混合检索。

评测集 eval_holdout.json：独立改写的问题 + gold 咨询编号（知识库不含改写问题，无泄漏）。
判定：检索返回的块中若包含 gold 咨询（块内 "## 咨询N" 标题）即相关。

指标：
  Recall@K = gold 出现在 Top-K 的问题比例
  MRR      = gold 首次出现位置倒数的平均（候选池 CANDIDATE_N 内，未命中记0）

用法：python eval.py eval_holdout.json
"""

import json
import os
import re
import sys

from bm25 import BM25Index
from config import (
    BM25_B, BM25_K1, BM25_WEIGHT, EMBED_API_KEY, EMBED_API_MODEL, EMBED_API_URL,
    EMBED_MODE, EMBED_MODEL_NAME, HYBRID_FUSION, INDEX_PATH, RRF_K,
    VECTOR_WEIGHT,
)
from embedder import build_embedder
from hybrid import HybridRetriever
from retriever import VectorStore

CANDIDATE_N = 10          # MRR 计算的候选池大小
REPORT_KS = (1, 2, 4)     # 报告 Recall@K 的 K 值
_CONS_RE = re.compile(r"##\s*(咨询\s*\d+)")


def chunk_consult_ids(text):
    """提取一个块里包含的所有咨询编号（去空格），如 ['咨询21','咨询22']。"""
    return [re.sub(r"\s+", "", m.group(1)) for m in _CONS_RE.finditer(text)]


def rank_of_gold(hits, gold_id):
    """返回 gold 首次出现的 1-based 排名；候选池内未出现返回 None。"""
    for rank, hit in enumerate(hits, 1):
        if gold_id in chunk_consult_ids(hit[1]):
            return rank
    return None


def evaluate(search_fn, holdout):
    """跑评测，返回汇总指标和每条详情。"""
    details = []
    for c in holdout:
        hits = search_fn(c["question"], CANDIDATE_N)
        rank = rank_of_gold(hits, c["gold_id"])
        row = {"id": c["id"], "topic": c["topic"], "question": c["question"],
               "gold": c["gold_id"], "rank": rank}
        for k in REPORT_KS:
            row[f"recall@{k}"] = rank is not None and rank <= k
        details.append(row)
    n = len(details)
    metrics = {}
    for k in REPORT_KS:
        metrics[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in details) / n * 100
    metrics["mrr"] = sum((1.0 / r["rank"] if r["rank"] else 0.0) for r in details) / n * 100
    return metrics, details


def print_table(name, metrics, details):
    print(f"\n========== {name} ==========")
    for c in details:
        rk = c["rank"] if c["rank"] else "-"
        mark = "✓" if c["rank"] and c["rank"] <= 4 else "✗"
        print(f"  {mark} #{c['id']:>2} [{c['topic']}] gold={c['gold']:<7} rank={rk:<3} {c['question'][:34]}")
    print("  Recall@1={:.1f}%  Recall@2={:.1f}%  Recall@4={:.1f}%  MRR={:.1f}%".format(
        metrics["recall@1"], metrics["recall@2"], metrics["recall@4"], metrics["mrr"]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval_holdout.json"
    with open(path, encoding="utf-8") as f:
        holdout = json.load(f)
    print(f"[eval] 评测集 {path}：{len(holdout)} 条（独立改写，无泄漏）")

    embedder = build_embedder(
        EMBED_MODE, EMBED_MODEL_NAME, EMBED_API_URL, EMBED_API_KEY, EMBED_API_MODEL,
    )
    store = VectorStore(INDEX_PATH)
    bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
    bm25.load(INDEX_PATH)
    hybrid = HybridRetriever(
        store, bm25, fusion=HYBRID_FUSION, rrf_k=RRF_K,
        vector_weight=VECTOR_WEIGHT, bm25_weight=BM25_WEIGHT,
    )

    def vec_search(q, k):
        return store.search(embedder.embed([q])[0], top_k=k)

    def hyb_search(q, k):
        final, _, _ = hybrid.search(q, embedder.embed([q])[0], top_k=k)
        return final

    vec_m, vec_d = evaluate(vec_search, holdout)
    hyb_m, hyb_d = evaluate(hyb_search, holdout)
    print_table("纯向量检索", vec_m, vec_d)
    print_table("混合检索（向量+BM25+RRF）", hyb_m, hyb_d)

    print("\n========== 可写入简历的结论 ==========")
    print("纯向量 : Recall@1={:.1f}% Recall@2={:.1f}% Recall@4={:.1f}% MRR={:.1f}%".format(
        vec_m["recall@1"], vec_m["recall@2"], vec_m["recall@4"], vec_m["mrr"]))
    print("混合   : Recall@1={:.1f}% Recall@2={:.1f}% Recall@4={:.1f}% MRR={:.1f}%".format(
        hyb_m["recall@1"], hyb_m["recall@2"], hyb_m["recall@4"], hyb_m["mrr"]))
    print(f"Recall@4 混合相对纯向量变化：{hyb_m['recall@4'] - vec_m['recall@4']:+.1f} 个百分点")
    print(f"MRR 混合相对纯向量变化：{hyb_m['mrr'] - vec_m['mrr']:+.1f} 个百分点")


if __name__ == "__main__":
    main()
