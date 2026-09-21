# -*- coding: utf-8 -*-
"""RAG 第二阶段：问答主流程（命令行版）。
流程：用户问题 -> 向量化 -> [混合检索：向量 + BM25 -> RRF/加权融合] -> 组装 Prompt -> LLM 生成答案。

用法：
  先 python ingest.py 建索引
  再 python qa.py            # 进入交互问答
  或  python qa.py "你的问题"
未配置 LLM_API_KEY 时，会退化为只打印检索结果（便于先验证检索环节）。
"""

import re
import sys


def extract_latest_date(text):
    """从文本块里提取最新的咨询日期（YYYY-MM-DD），用于时间优先排序。"""
    dates = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
    if not dates:
        return "0000-00-00"
    return max(dates)  # YYYY-MM-DD 格式可直接字符串比较


from bm25 import BM25Index
from config import (
    BM25_B, BM25_K1, BM25_WEIGHT, EMBED_API_KEY, EMBED_API_MODEL, EMBED_API_URL,
    EMBED_MODE, EMBED_MODEL_NAME, HYBRID_FUSION, INDEX_PATH, LLM_API_KEY, LLM_BASE_URL,
    LLM_MODEL, PROJECT_NAME, PROJECT_SLOGAN, RRF_K, SCORE_THRESHOLD, TOP_K, USE_HYBRID,
    VECTOR_WEIGHT,
)
from embedder import build_embedder
from hybrid import HybridRetriever
from retriever import VectorStore

SYSTEM_PROMPT = (
    "你是宁波大学科学技术学院的招生咨询老师，直接、口语化地回答考生问题。规则：\n"
    "1. 直接给答案，不要用\"根据参考资料\"\"根据资料显示\"这类开头；\n"
    "2. 不要在回答末尾加括号说明引用了哪条资料（例如不要写\"（资料1、资料3…）\"）；\n"
    "3. 只依据下方提供的资料作答，资料中没有就回答\"暂未查询到相关信息，建议直接联系招生办 0574-87600018\"，不要编造；\n"
    "4. 用简体中文，简洁、亲切，像老师当面回答一样。"
)


def build_prompt(question, hits):
    """把检索到的文本块拼进提示词（即 RAG 的“增强上下文”）。"""
    ctx = "\n\n".join(
        f"[资料{i + 1}]（来源：{meta}）\n{text}" for i, (_, text, meta) in enumerate(hits)
    )
    return (
        f"参考资料：\n{ctx}\n\n"
        f"用户问题：{question}\n\n"
        f"请基于以上参考资料作答。"
    )


def retrieve(question, embedder, store, bm25=None, use_hybrid=True):
    """检索，返回 (hits, query_vec, debug_info)。
    debug_info = {"vec_hits": [...], "bm25_hits": [...]}，纯向量模式下为 None。
    """
    query_vec = embedder.embed([question])[0]

    if use_hybrid and bm25 is not None:
        hybrid = HybridRetriever(
            store, bm25, fusion=HYBRID_FUSION, rrf_k=RRF_K,
            vector_weight=VECTOR_WEIGHT, bm25_weight=BM25_WEIGHT,
        )
        final, vec_hits, bm25_hits = hybrid.search(question, query_vec, top_k=TOP_K)
        hits = [r for r in final if r[0] >= SCORE_THRESHOLD]
        return hits, query_vec, {"vec_hits": vec_hits, "bm25_hits": bm25_hits}

    raw = store.search(query_vec, top_k=TOP_K)
    hits = [r for r in raw if r[0] >= SCORE_THRESHOLD]
    return hits, query_vec, None


def answer(question, embedder, store, bm25=None, use_hybrid=True, use_llm=True):
    hits, _, _ = retrieve(question, embedder, store, bm25, use_hybrid)
    if not hits:
        return "知识库中没有检索到相关内容，请尝试换一种问法，或先补充文档后重建索引。", [], None

    if not use_llm:
        return "", hits, None

    # 检查是否配置了有效 API Key（占位符不算）
    has_key = bool(LLM_API_KEY) and "sk-" in LLM_API_KEY \
        and "填你的" not in LLM_API_KEY and "在这里填" not in LLM_API_KEY
    if not has_key:
        return "", hits, None  # 未配置 key：优雅降级为只展示检索结果

    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        # 同一问题多个答案时，优先让大模型参考时间更近的回复：
        # 把 hits 按各自最新咨询日期降序排（最新在前），相关性已由 RRF 筛过 Top-K
        hits_by_recency = sorted(hits, key=lambda h: extract_latest_date(h[1]), reverse=True)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits_by_recency)},
            ],
            temperature=0.3,
            stream=False,
        )
        return resp.choices[0].message.content, hits, None
    except Exception as e:
        # openai 未安装 / key 无效 / 网络不通等，都降级为检索结果并提示原因
        return "", hits, None


def stream_answer(question, hits):
    """流式版生成器：给定已检索的 hits，逐段 yield 回答文本片段。
    供 Streamlit 用 st.write_stream 做打字机效果。未配置 key 或失败时直接空 yield。"""
    has_key = bool(LLM_API_KEY) and "sk-" in LLM_API_KEY \
        and "填你的" not in LLM_API_KEY and "在这里填" not in LLM_API_KEY
    if not has_key:
        return
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        hits_by_recency = sorted(hits, key=lambda h: extract_latest_date(h[1]), reverse=True)
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits_by_recency)},
            ],
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        return


def _print_hits(title, hits):
    print(f"\n  【{title}】共 {len(hits)} 条")
    for i, (score, text, meta) in enumerate(hits):
        snippet = text.replace("\n", " ")[:100]
        print(f"    [{i + 1}] (分数 {score:.3f}, {meta}) {snippet}...")


def main():
    print(f"=== {PROJECT_NAME} · {PROJECT_SLOGAN} ===")
    embedder = build_embedder(
        EMBED_MODE, EMBED_MODEL_NAME, EMBED_API_URL, EMBED_API_KEY, EMBED_API_MODEL,
    )
    store = VectorStore(INDEX_PATH)
    if len(store.chunks) == 0:
        print("索引为空，请先运行：python ingest.py")
        return

    # 加载 BM25 索引（混合检索用）
    bm25 = None
    if USE_HYBRID:
        bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
        if not bm25.load(INDEX_PATH):
            print("[警告] 未找到 BM25 索引（bm25.json），将退化为纯向量检索。请重新运行 python ingest.py。")
            bm25 = None

    mode_label = "混合检索（向量 + BM25，" + HYBRID_FUSION + " 融合）" if bm25 else "纯向量检索"
    print(f"[检索模式] {mode_label}\n")

    has_key = "sk-" in LLM_API_KEY and "填你的" not in LLM_API_KEY
    if not has_key:
        print("[提示] 未检测到有效的 LLM_API_KEY，将只展示检索结果（验证检索环节）。\n")

    def ask(question):
        print(f"\nQ: {question}")
        hits, _, debug = retrieve(question, embedder, store, bm25, use_hybrid=USE_HYBRID)
        if debug:
            _print_hits("向量检索", debug["vec_hits"])
            _print_hits("BM25 检索", debug["bm25_hits"])
        _print_hits("最终结果（融合后）", hits)
        if has_key:
            ans, _, _ = answer(question, embedder, store, bm25, use_hybrid=USE_HYBRID, use_llm=True)
            print(f"\nA: {ans}")

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("RAG 问答已启动，输入问题回车提问，输入 exit 退出。")
        while True:
            q = input("你: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q:
                ask(q)


if __name__ == "__main__":
    main()
