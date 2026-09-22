# -*- coding: utf-8 -*-
"""NDKY 招生顾问 Agent —— Streamlit 网页界面。
用法：streamlit run app.py
需先 python ingest.py 建好索引，并在环境变量 LLM_API_KEY 配置大模型 Key。
"""

from datetime import datetime

import streamlit as st
import os

# Streamlit Cloud Secrets 同步到环境变量（本地开发用环境变量，云端用 Secrets）
if "LLM_API_KEY" in st.secrets:
    os.environ["LLM_API_KEY"] = st.secrets["LLM_API_KEY"]

from bm25 import BM25Index
from config import (
    BM25_B, BM25_K1, INDEX_PATH, USE_HYBRID,
)
from retriever import VectorStore
from agent import KnowledgeSearchTool, run_agent, AGENT_SYSTEM_PROMPT

APP_TITLE = "NDKY 招生咨询"

st.set_page_config(page_title=APP_TITLE, page_icon="🏫", layout="wide")
st.title(f"🏫 {APP_TITLE}")
st.caption("宁波大学科学技术学院招生顾问 Agent · 知识库问答 + 历年分数 + 志愿分档")


# ---------- 初始化（缓存，不重复加载模型/索引） ----------
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
    tool = KnowledgeSearchTool(embedder, store, bm25, use_hybrid=USE_HYBRID)
    return tool


tool = load()

# ---------- Agent 会话状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
if "history" not in st.session_state:
    st.session_state.history = []   # [{"q","a","trace","t"}]
if "selected" not in st.session_state:
    st.session_state.selected = None

with st.sidebar:
    st.subheader("问题记录")
    # 报考信息画像卡
    st.markdown("**🧑 我的报考信息**")
    up_province = st.text_input("省份", value=st.session_state.get("up_province", "浙江"),
                                 key="up_province")
    up_score = st.number_input("高考分数", value=float(st.session_state.get("up_score", 0)),
                                step=1.0, key="up_score")
    up_major = st.text_input("意向专业", value=st.session_state.get("up_major", ""),
                              key="up_major")
    st.caption("填写后，Agent 回答会自动参考这些信息")
    st.divider()
    if not st.session_state.history:
        st.caption("暂无记录，提问后会显示在这里")
    else:
        for idx in range(len(st.session_state.history) - 1, -1, -1):
            rec = st.session_state.history[idx]
            label = rec["q"][:18] + ("…" if len(rec["q"]) > 18 else "")
            ts = rec.get("t", "")
            if ts:
                label = f"{label}  ·  {ts}"
            if st.button(label, key=f"hist_{idx}", use_container_width=True):
                st.session_state.selected = rec
        if st.button("清空记录", use_container_width=True):
            st.session_state.history = []
            st.session_state.selected = None
            st.session_state.messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
            st.rerun()


# ---------- 提问前：清空输入框 ----------
if st.session_state.pop("_need_clear", False):
    st.session_state.q_input = ""

# ---------- 展示全部历史对话（从上到下，新问题在下面） ----------
for rec in st.session_state.history:
    st.markdown(f"**问：** {rec['q']}")
    if rec.get("trace"):
        with st.expander("✨ Agent 执行过程", expanded=False):
            for line in rec["trace"]:
                st.caption(line)
    if rec["a"]:
        st.markdown(rec["a"])
    else:
        st.warning("未能生成回答，请检查 API Key 配置或网络。")
    if rec.get("sources"):
        with st.expander("📚 数据来源", expanded=False):
            for name, url in rec["sources"]:
                st.markdown(f"- [{name}]({url})")
    st.divider()

st.caption("本回答由 AI 基于历年招生咨询与录取数据自动生成，仅供参考，最终以招生办官方回复为准（电话 0574-87600018）。")

# ---------- 底部输入框（chat 风格） ----------
question = st.chat_input("请输入你的问题，回车发送…", key="q_input")

if question:
    # 把考生画像注入 system prompt，让 Agent 每轮都知道"谁在问"
    profile = (f"\n\n【当前考生信息】省份={up_province}；"
               f"分数={int(up_score) if up_score else '未填'}；"
               f"意向专业={up_major or '未指定'}。"
               f"考生追问时自动结合这些信息，不要让他重复说。")
    st.session_state.messages[0]["content"] = AGENT_SYSTEM_PROMPT + profile

    st.markdown(f"**问：** {question}")
    think_box = st.expander("✨ Agent 执行过程", expanded=False)
    think_lines = []
    answer_chunks = []
    sources = []
    try:
        for kind, payload in run_agent(st.session_state.messages, question, tool):
            if kind == "think":
                think_lines.append(payload)
                with think_box:
                    for line in think_lines:
                        st.caption(line)
            elif kind == "sources":
                sources = payload
            else:
                answer_chunks.append(payload)
    except Exception as e:
        st.error(f"生成失败：{e}")

    def _gen():
        for c in answer_chunks:
            yield c

    answer_text = st.write_stream(_gen()) or ""
    if sources:
        with st.expander("📚 数据来源", expanded=False):
            for name, url in sources:
                st.markdown(f"- [{name}]({url})")
    rec = {
        "q": question, "a": answer_text, "trace": list(think_lines),
        "sources": sources, "t": datetime.now().strftime("%m-%d %H:%M"),
    }
    st.session_state.history.append(rec)
    st.session_state.selected = rec
    st.session_state._need_clear = True   # 下次渲染前清空输入框
    st.rerun()
