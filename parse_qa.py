# -*- coding: utf-8 -*-
"""解析招生网咨询原始文本，生成：
  1. data/ndky_qa_kb.md   —— RAG 知识库文档（50 条问答）
  2. qa_dataset.json       —— 结构化原始数据（50 条）
  3. eval_questions.json   —— 评测集（20 条，问题 + 答案关键词）

用法：python parse_qa.py
"""

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(BASE_DIR, "raw_qa_3years.txt")
DATA_DIR = os.path.join(BASE_DIR, "data")


def parse_raw(raw_file):
    """按空行分块，每块第一行匹配'名字 日期 时间'，后续行依次为问题、回答。"""
    with open(raw_file, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = re.split(r"\n\s*\n", text.strip())
    items = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r"^(\S+)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})$", lines[0])
        if not m:
            continue
        name, time = m.group(1), m.group(2)
        question = lines[1]
        answer = "\n".join(lines[2:])
        items.append({"name": name, "time": time, "question": question, "answer": answer})
    return items


def build_knowledge_base(items, out_path):
    """生成知识库 Markdown：每条问答一个章节。"""
    lines = ["# 宁波大学科学技术学院招生咨询知识库", ""]
    lines.append(f"> 共 {len(items)} 条真实咨询问答，数据来源：宁波大学科学技术学院招生网在线咨询栏目。")
    lines.append("")
    for i, item in enumerate(items, 1):
        lines.append(f"## 咨询 {i}（{item['name']}，{item['time']}）")
        lines.append("")
        lines.append(f"**问：** {item['question']}")
        lines.append("")
        lines.append(f"**答：** {item['answer']}")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[parse] 知识库已生成：{out_path}（{len(items)} 条）")


def build_eval_set(items, out_path):
    """从 50 条中精选 20 条作为评测集，手动标注答案关键词。"""
    # 按问题关键词匹配，选取覆盖不同主题的 20 条
    selected_keywords = [
        ("宿舍床位", ["200cm", "90cm", "慈溪校区"]),
        ("专升本考生入学后不允许转专业", ["不允许转专业"]),
        ("上下铺床位长度都是2米", ["2米"]),
        ("床上用品", ["90cm", "200cm", "随身携带"]),
        ("研究生需自行考", ["自行考"]),
        ("录取通知书里并没有银行卡", ["没有银行卡"]),
        ("请查看录取通知书", ["录取通知书"]),
        ("科院通APP即可查看班级", ["科院通", "班级"]),
        ("我校有硕士点", ["硕士点"]),
        ("寝室为六人寝", ["六人寝"]),
        ("应该是可以的", ["可以的"]),
        ("专业调整政策", ["专业调整", "0574-87600018"]),
        ("无权操作退档", ["无权操作退档", "自动放弃"]),
        ("电话畅通", ["电话畅通", "EMS"]),
        ("朱尼亚塔学院", ["朱尼亚塔学院", "住宿费"]),
        ("已取消三位一体招生", ["取消三位一体"]),
        ("翻看《学生手册》或咨询教务部", ["学生手册", "教务部"]),
        ("《2025级新生报到手册》", ["报到手册"]),
        ("迎新系统还未开放", ["未开放"]),
        ("结果已出，可在招生网查询", ["结果已出", "招生网查询"]),
    ]
    eval_set = []
    used = set()
    for kw, answer_keywords in selected_keywords:
        for i, item in enumerate(items):
            if i in used:
                continue
            if kw in item["answer"]:
                eval_set.append({
                    "question": item["question"],
                    "answer_keywords": answer_keywords,
                    "_source": f"咨询{i+1}",
                })
                used.add(i)
                break
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)
    print(f"[parse] 评测集已生成：{out_path}（{len(eval_set)} 条）")
    if len(eval_set) < 20:
        print(f"  警告：仅匹配到 {len(eval_set)} 条，部分关键词未命中，可检查 build_eval_set 的关键词表或输入数据")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="解析招生网咨询原始文本，生成知识库和评测集")
    parser.add_argument("--input", default=RAW_FILE, help=f"输入原始文本文件（默认 {RAW_FILE}）")
    parser.add_argument("--kb-output", default=os.path.join(DATA_DIR, "ndky_qa_kb.md"),
                        help="知识库 Markdown 输出路径")
    parser.add_argument("--eval-output", default=os.path.join(BASE_DIR, "eval_questions.json"),
                        help="评测集 JSON 输出路径")
    args = parser.parse_args()

    items = parse_raw(args.input)
    print(f"[parse] 解析到 {len(items)} 条咨询问答")

    os.makedirs(DATA_DIR, exist_ok=True)
    build_knowledge_base(items, args.kb_output)

    with open(os.path.join(BASE_DIR, "qa_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[parse] 结构化数据已生成：qa_dataset.json（{len(items)} 条）")

    build_eval_set(items, args.eval_output)


if __name__ == "__main__":
    main()
