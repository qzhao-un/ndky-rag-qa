# -*- coding: utf-8 -*-
"""NDKY 招生咨询问答系统 —— Streamlit 网页界面。
用法：streamlit run app.py
需先 python ingest.py 建好索引，并在 config.py 配置 LLM_API_KEY。
"""

import re
from datetime import datetime

import streamlit as st

from bm25 import BM25Index
from config import (
    BM25_B, BM25_K1, HYBRID_FUSION, INDEX_PATH, RRF_K, TOP_K, USE_HYBRID,
)
from hybrid import HybridRetriever
from qa import answer, retrieve, stream_answer
from retriever import VectorStore

APP_TITLE = "NDKY 招生咨询"


def _clean_body(body: str) -> str:
    """清理切块残留：去掉单独一行只有"咨"或"询"的残字（切块把"咨询"切断时产生）。"""
    lines = body.split("\n")
    cleaned = [ln for ln in lines if ln.strip() not in ("咨", "询", "咨 ", "询 ")]
    return "\n".join(cleaned).strip()


def parse_chunk(text: str):
    """把一个检索到的文本块拆成若干 (meta_line, qa_body)。
    meta_line 形如 "咨询 830 · 张** · 2023-06-23 16:02:51"（小字淡化展示）；
    qa_body 是问/答正文（正常展示）。姓名只留姓，其余打 *。
    兼容 chunk 被截断、缺少右括号、"咨询"二字被切断等情况。"""
    # 宽松匹配：咨询 NNN（姓名，日期...）或 咨询 NNN（姓名，2023（截断无右括号）
    # 姓名 1-4 个汉字（含单字名，如"杨"），含 · 分隔符（复姓/少数民族）
    pattern = r'(?:##\s*)?咨询\s*(\d+)\s*（([\u4e00-\u9fa5·]{1,4})，(\s*\d{4}[^）\n]*)）?'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return [("", _clean_body(text))]
    blocks = []
    if matches[0].start() > 0:
        tail = text[:matches[0].start()].strip()
        if tail:
            blocks.append(("", _clean_body(tail)))
    for i, m in enumerate(matches):
        num, name, date = m.group(1), m.group(2), (m.group(3) or "").strip()
        masked = name[0] + "*" * (len(name) - 1) if len(name) > 1 else name
        meta = f"咨询 {num} · {masked}" + (f" · {date}" if date else "")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _clean_body(text[m.end():end].strip())
        blocks.append((meta, body))
    return blocks

st.set_page_config(page_title=APP_TITLE, page_icon="🏫", layout="wide")
st.title(f"🏫 {APP_TITLE}")
st.caption("宁波大学科学技术学院招生咨询智能问答 · 基于大模型 RAG")


# ---------- 初始化（带缓存，避免每次刷新重复加载模型/索引） ----------
@st.cache_resource
def load():
    from config import (
        EMBED_API_KEY, EMBED_API_MODEL, EMBED_API_URL, EMBED_MODE, EMBED_MODEL_NAME,
    )
    from embedder import build_embedder
    embedder = build_embedder(EMBED_MODE, EMBED_MODEL_NAME, EMBED_API_URL,
                               EMBED_API_KEY, EMBED_API_MODEL)
    store = VectorStore(INDEX_PATH)
    bm25 = None
    if USE_HYBRID:
        bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
        if not bm25.load(INDEX_PATH):
            bm25 = None
    return embedder, store, bm25


embedder, store, bm25 = load()

# ---------- 对话历史（问题记录） ----------
if "history" not in st.session_state:
    st.session_state.history = []   # [{"q":..., "a":..., "hits":[...]}]
if "selected" not in st.session_state:
    st.session_state.selected = None  # 当前展示的那条记录，None=空

with st.sidebar:
    st.subheader("问题记录")
    if not st.session_state.history:
        st.caption("暂无记录，提问后会显示在这里")
    else:
        for idx in range(len(st.session_state.history) - 1, -1, -1):
            rec = st.session_state.history[idx]
            q = rec["q"]
            label = q[:18] + ("…" if len(q) > 18 else "")
            ts = rec.get("t", "")
            if ts:
                label = f"{label}  ·  {ts}"
            if st.button(label, key=f"hist_{idx}", use_container_width=True):
                st.session_state.selected = rec
        if st.button("清空记录", use_container_width=True):
            st.session_state.history = []
            st.session_state.selected = None
            st.rerun()

def render_refs(hits):
    """渲染参考依据：判断是否"查不到"，查不到则不展示；否则按片段展开。"""
    if not hits:
        return
    for i, (score, text, meta) in enumerate(hits):
        st.markdown(f"**片段 {i + 1}**")
        for meta_line, body in parse_chunk(text):
            st.write(body)
            if meta_line:
                st.caption(meta_line)
        st.divider()


def is_no_answer(answer_text):
    a = answer_text or ""
    return any(k in a for k in ["暂未查询到", "暂时没有", "资料里", "未找到", "建议直接联系招生办"])


# ---------- 问答区 ----------
question = st.text_input("请输入你的问题：", placeholder="例如：宿舍几人间？录取通知书什么时候发？")

if question:
    with st.spinner("正在检索..."):
        try:
            hits, _, _ = retrieve(question, embedder, store, bm25, use_hybrid=USE_HYBRID)
        except Exception as e:
            st.error(f"检索失败：{e}")
            hits = []

    st.markdown(f"**问：** {question}")
    # 流式生成回答（打字机效果）
    answer_text = st.write_stream(stream_answer(question, hits)) or ""
    rec = {"q": question, "a": answer_text, "hits": hits,
           "t": datetime.now().strftime("%m-%d %H:%M")}
    st.session_state.history.append(rec)
    st.session_state.selected = rec

    if hits and not is_no_answer(answer_text):
        with st.expander("参考依据（点击展开）", expanded=False):
            render_refs(hits)
    st.stop()  # 本次提问已完整展示，不重复走下方历史回看块

# ---------- 展示历史记录 ----------
sel = st.session_state.selected
if sel:
    st.markdown(f"**问：** {sel['q']}")
    if sel["a"]:
        st.markdown(sel["a"])
    else:
        st.warning("未能生成回答，请检查 API Key 配置或网络。")
    if sel["hits"] and not is_no_answer(sel["a"]):
        with st.expander("参考依据（点击展开）", expanded=False):
            render_refs(sel["hits"])
else:
    st.info("在上方输入问题开始咨询。")

st.divider()
st.caption("本回答由 AI 基于历史招生咨询自动生成，仅供参考，最终以招生办官方回复为准（电话 0574-87600018）。")
