# -*- coding: utf-8 -*-
"""RAG 第一阶段：构建知识库索引。
流程：读取 data/ 下文档 -> 文本切块 -> 向量化 -> 写入 index/ 向量库。

用法：python ingest.py
依赖：见 requirements.txt；首次以 local 模式运行会自动下载 embedding 模型。
"""

import os
import re

from config import (
    BM25_B, BM25_K1, CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBED_API_KEY,
    EMBED_API_MODEL, EMBED_API_URL, EMBED_MODE, EMBED_MODEL_NAME, INDEX_PATH,
    MAX_TOKENS, USE_HYBRID,
)
from bm25 import BM25Index
from embedder import build_embedder
from retriever import VectorStore


# ---------- 1. 读取文档 ----------
def read_documents(data_dir):
    """读取 data 目录下所有 .txt/.md/.pdf，返回 [(text, filename), ...]"""
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".pdf":
            text = _read_pdf(fpath)
        elif ext in (".txt", ".md", ".markdown"):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            continue
        if text.strip():
            docs.append((text, fname))
    if not docs:
        print(f"[ingest] data 目录为空或没有支持的文件，请先放入 txt/md/pdf：{data_dir}")
    return docs


def _read_pdf(fpath):
    try:
        from pypdf import PdfReader
        reader = PdfReader(fpath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"[ingest] 读取 PDF 失败（{fpath}）：{e}")
        return ""


# ---------- 2. 文本切块 ----------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, token_counter=None):
    """切块策略：
    - 问答文档（含 "## 咨询 NNN" 标记）：按咨询边界切，每条咨询完整不切断；
      把多条短咨询累积打包，同时受【字符数 chunk_size】和【token 数 MAX_TOKENS】双约束，
      确保每个块不超 bge 的 512 token 上限。
    - 其他文档（如学校简介）：回退到普通字符切块 + 重叠。
    返回 (chunks, oversize_count)，oversize_count 为单条自身超限的块数（无法在咨询边界内切小）。
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return [], 0

    # 用前瞻断言按咨询标题切分，保留标题本身在每个 block 开头
    parts = re.split(r"(?=##\s*咨询\s*\d+)", text)
    blocks = [p.strip() for p in parts if p.strip()]

    # 没有咨询标记 → 普通字符切块（适用于学校简介等非问答文档）
    if len(blocks) <= 1:
        chunks, step = [], max(1, chunk_size - overlap)
        for i in range(0, len(text), step):
            chunks.append(text[i:i + chunk_size])
        return chunks, 0

    def fits(cand):
        if len(cand) > chunk_size:
            return False
        if token_counter and token_counter(cand) > MAX_TOKENS:
            return False
        return True

    # 问答文档：逐条累积打包，不切断任何一条咨询
    chunks, buf = [], ""
    oversize = 0
    for block in blocks:
        if not buf:
            buf = block
        elif fits(buf + "\n\n" + block):
            buf += "\n\n" + block
        else:
            chunks.append(buf)
            buf = block
            # 单条咨询自身就超限（无法在不破坏完整性的前提下切小）
            if token_counter and token_counter(block) > MAX_TOKENS:
                oversize += 1
    if buf:
        chunks.append(buf)
        if token_counter and token_counter(buf) > MAX_TOKENS and len(chunks) == 1:
            oversize += 1
    return chunks, oversize


# ---------- 3. 主流程 ----------
def main():
    import argparse, json as _json
    parser = argparse.ArgumentParser(description="构建/更新 RAG 索引")
    parser.add_argument("--append", action="store_true",
                        help="增量追加：保留已有索引，只处理新增或内容变化的文件")
    args = parser.parse_args()

    print(f"[ingest] Embedding 模式：{EMBED_MODE}（{'增量追加' if args.append else '全量重建'}）")
    embedder = build_embedder(
        EMBED_MODE, EMBED_MODEL_NAME, EMBED_API_URL, EMBED_API_KEY, EMBED_API_MODEL,
    )

    # token 计数：本地模型用其 tokenizer，其他模式拿不到就回退到字符数估算
    def make_counter():
        try:
            tok = embedder.model.tokenizer
            return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
        except Exception:
            return lambda s: int(len(s) * 1.3)  # 中文粗略 1 字符≈1.3 token

    token_counter = make_counter()

    # 读 manifest（记录每个源文件上次处理时的大小/mtime，用于增量判断）
    manifest_path = os.path.join(INDEX_PATH, "manifest.json")
    manifest = {}
    if args.append and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = _json.load(f)

    # 非增量模式：清空旧索引，从头建
    if not args.append:
        for fn in ("chunks.json", "vectors.npy", "bm25.json"):
            p = os.path.join(INDEX_PATH, fn)
            if os.path.exists(p):
                os.remove(p)

    store = VectorStore(INDEX_PATH)  # 载入已有索引（增量模式）或空库（全量模式）
    docs = read_documents(DATA_DIR)
    if not docs:
        return

    total_chunks = 0
    new_manifest = {}
    for text, fname in docs:
        fpath = os.path.join(DATA_DIR, fname)
        size = os.path.getsize(fpath)
        mtime = os.path.getmtime(fpath)
        # 增量模式下，未变化的文件跳过
        prev = manifest.get(fname)
        if args.append and prev and prev.get("size") == size and prev.get("mtime") == mtime:
            new_manifest[fname] = prev
            continue
        chunks, oversize = split_text(text, token_counter=token_counter)
        metas = [fname] * len(chunks)
        store.add(chunks, metas, embedder)
        total_chunks += len(chunks)
        new_manifest[fname] = {"size": size, "mtime": mtime, "chunks": len(chunks)}
        tag = f"（{oversize} 条单咨询超 {MAX_TOKENS}token，bge 将截断尾部）" if oversize else ""
        print(f"  - {fname}: {len(chunks)} 块{'（新增）' if args.append else ''}{tag}")

    if args.append and total_chunks == 0:
        print("[ingest] 没有新增或变化的文件，索引未改动。")
        return

    store.save()

    # ---------- 构建 BM25 关键词索引（混合检索用，始终全量重建） ----------
    if USE_HYBRID:
        bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
        bm25.fit(store.chunks, store.metas)
        bm25.save(INDEX_PATH)

    with open(manifest_path, "w", encoding="utf-8") as f:
        _json.dump(new_manifest, f, ensure_ascii=False, indent=2)

    print(f"[ingest] 完成，本次写入 {total_chunks} 个文本块，索引共 {len(store.chunks)} 块。")
    print(f"[ingest] 后续：python qa.py  或  streamlit run app.py")


if __name__ == "__main__":
    main()
