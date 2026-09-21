# -*- coding: utf-8 -*-
"""RAG 检索效果评测脚本：对比「纯向量检索」vs「混合检索」的 Top-K 召回率。

用法：
  python eval.py                  # 使用内置示例评测集
  python eval.py my_questions.json  # 使用自定义评测集（JSON 数组）

评测集格式（JSON 数组）：
  [
    {"question": "RAG 的全称是什么？", "answer_keywords": ["检索增强生成"]},
    {"question": "文本切块一般多大？", "answer_keywords": ["200", "800"]}
  ]

命中标准：召回的 Top-K 个文本块中，至少有一个块包含该问题的全部 answer_keywords。
召回率 = 命中问题数 / 总问题数。

【重要】示例文档仅 2 个文本块，评测区分度有限；建议替换为真实业务文档
（10+ 块）后再跑，得到的召回率数据才可写入简历。
"""

import json
import os
import sys

from bm25 import BM25Index
from config import (
    BM25_B, BM25_K1, BM25_WEIGHT, EMBED_API_KEY, EMBED_API_MODEL, EMBED_API_URL,
    EMBED_MODE, EMBED_MODEL_NAME, HYBRID_FUSION, INDEX_PATH, RRF_K, TOP_K,
    VECTOR_WEIGHT,
)
from embedder import build_embedder
from hybrid import HybridRetriever
from retriever import VectorStore

# 内置示例评测集（基于 data/sample.md，用户可替换为自己的真实问题）
DEFAULT_EVAL_SET = [
    {"question": "RAG 的全称是什么？", "answer_keywords": ["检索增强生成"]},
    {"question": "大模型为什么会产生幻觉？", "answer_keywords": ["幻觉"]},
    {"question": "文本切块一般切多大？", "answer_keywords": ["200", "800"]},
    {"question": "向量检索用什么衡量文本相似度？", "answer_keywords": ["余弦相似度"]},
    {"question": "RAG 和微调相比有什么优势？", "answer_keywords": ["微调"]},
]


def load_eval_set(path=None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_EVAL_SET


def is_hit(retrieved_texts, keywords):
    """判断召回结果中是否至少有一个块包含全部关键词。"""
    if not keywords:
        return True
    for text in retrieved_texts:
        if all(kw in text for kw in keywords):
            return True
    return False


def evaluate(name, search_fn, eval_set, k_list):
    """跑评测，返回 {k: 召回率} 和每个问题的详情。"""
    results = {k: [] for k in k_list}  # k -> [bool, ...]
    details = []
    for item in eval_set:
        q = item["question"]
        kws = item.get("answer_keywords", [])
        hits = search_fn(q, max(k_list))
        texts = [h[1] for h in hits]
        row = {"question": q, "keywords": kws}
        for k in k_list:
            hit = is_hit(texts[:k], kws)
            results[k].append(hit)
            row[f"hit@{k}"] = hit
        details.append(row)
    rates = {k: sum(v) / len(v) * 100 if v else 0 for k, v in results.items()}
    return rates, details


def main():
    eval_path = sys.argv[1] if len(sys.argv) > 1 else None
    eval_set = load_eval_set(eval_path)
    print(f"[eval] 评测集：{eval_path or '内置示例'}，共 {len(eval_set)} 个问题")

    embedder = build_embedder(
        EMBED_MODE, EMBED_MODEL_NAME, EMBED_API_URL, EMBED_API_KEY, EMBED_API_MODEL,
    )
    store = VectorStore(INDEX_PATH)
    if len(store.chunks) == 0:
        print("索引为空，请先运行：python ingest.py")
        return

    bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
    bm25.load(INDEX_PATH)
    hybrid = HybridRetriever(
        store, bm25, fusion=HYBRID_FUSION, rrf_k=RRF_K,
        vector_weight=VECTOR_WEIGHT, bm25_weight=BM25_WEIGHT,
    )

    k_list = [1, 2, min(TOP_K, len(store.chunks))]
    k_list = sorted(set(k_list))

    # 纯向量检索
    def vec_search(q, k):
        return store.search(embedder.embed([q])[0], top_k=k)

    # 混合检索
    def hybrid_search(q, k):
        final, _, _ = hybrid.search(q, embedder.embed([q])[0], top_k=k)
        return final

    print("\n========== 评测结果 ==========")
    vec_rates, vec_details = evaluate("纯向量", vec_search, eval_set, k_list)
    hyb_rates, hyb_details = evaluate("混合检索", hybrid_search, eval_set, k_list)

    header = f"{'问题':<28}" + "".join(f"{'向量@'+str(k):>10}" for k in k_list) + "".join(f"{'混合@'+str(k):>10}" for k in k_list)
    print(header)
    print("-" * len(header))
    for i, item in enumerate(eval_set):
        q = item["question"][:26]
        row = f"{q:<28}"
        for k in k_list:
            row += f"{'✓' if vec_details[i][f'hit@{k}'] else '✗':>10}"
        for k in k_list:
            row += f"{'✓' if hyb_details[i][f'hit@{k}'] else '✗':>10}"
        print(row)

    print("-" * len(header))
    rate_row = f"{'召回率(%)':<28}"
    for k in k_list:
        rate_row += f"{vec_rates[k]:>10.1f}"
    for k in k_list:
        rate_row += f"{hyb_rates[k]:>10.1f}"
    print(rate_row)

    print("\n========== 可写入简历的结论 ==========")
    best_k = k_list[-1]
    print(f"纯向量检索 Top-{best_k} 召回率：{vec_rates[best_k]:.1f}%")
    print(f"混合检索 Top-{best_k} 召回率：{hyb_rates[best_k]:.1f}%")
    diff = hyb_rates[best_k] - vec_rates[best_k]
    if diff > 0:
        print(f"混合检索相比纯向量提升：{diff:.1f} 个百分点")
    elif diff < 0:
        print(f"混合检索相比纯向量下降：{abs(diff):.1f} 个百分点（建议调参或扩充评测集）")
    else:
        print("两者持平（示例文档过小，建议替换为真实文档后重跑）")
    print("\n提示：替换为真实业务文档（10+ 块）和真实问题后，以上数据可直接写入简历。")


if __name__ == "__main__":
    main()
