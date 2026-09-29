# -*- coding: utf-8 -*-
"""构造无泄漏评测集：从2158条FAQ里分层挑40条，再用LLM改写成同义问题。

产出 eval_holdout.json：
  [{"id":1,"topic":"宿舍","question":"<改写问题>","gold_id":"咨询123",
    "gold_answer":"<原答案，供人工核对>"}]

评测问题是独立改写的，知识库不含改写文本，无泄漏；
gold 通过切块里的"## 咨询N"标题定位。
"""

import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(BASE, "data", "ndky_qa_kb.md")

# 主题 -> 在【答案】里用于筛选的关键词（命中即候选）
TOPIC_RULES = [
    ("宿舍床位", ["几人间", "六人", "六人寝", "上下铺", "床位尺寸", "床的尺寸"]),
    ("床上用品", ["床上用品", "自带被子", "被褥", "统一发放"]),
    ("转专业", ["转专业", "专业调整", "不允许转", "申请转"]),
    ("保研考研", ["保研", "硕士点", "研究生", "推免"]),
    ("录取通知书", ["录取通知书", "通知书", "EMS", "银行卡"]),
    ("学号班级", ["学号", "几班", "班级", "科院通"]),
    ("学费缴费", ["学费", "缴费", "住宿费", "收费"]),
    ("录取结果", ["录取结果", "录取查询", "结果已出", "录取状态"]),
    ("三位一体", ["三位一体", "三位一体招生"]),
    ("报到入学", ["报到", "迎新", "入学", "注册"]),
    ("校区位置", ["校区", "慈溪", "周巷", "地址"]),
    ("退档复读", ["退档", "复读", "放弃", "档案"]),
    ("团组织关系", ["团籍", "团支部", "团组织关系", "团员"]),
    ("中外合作", ["中外合作", "中美合作", "朱尼亚塔", "合作办学"]),
    ("电脑网络", ["宽带", "网络", "电脑配置", "校园网"]),
]

# 每个主题最多选几条，控制总量约40
PER_TOPIC = 3


def parse_kb(path):
    """解析知识库为 {咨询编号: {"q":..,"a":..}}，保持出现顺序。"""
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"(?=##\s*咨询\s*\d+)", text)
    items = {}
    for p in parts:
        m = re.match(r"##\s*(咨询\s*\d+)", p.strip())
        if not m:
            continue
        cid = re.sub(r"\s+", "", m.group(1))
        qm = re.search(r"\*\*问：\*\*\s*(.+?)(?:\n\s*\n|\n\*\*答)", p, re.DOTALL)
        am = re.search(r"\*\*答：\*\*\s*(.+)", p, re.DOTALL)
        q = qm.group(1).strip() if qm else ""
        a = am.group(1).strip() if am else ""
        items[cid] = {"q": q, "a": a}
    return items


def select(items):
    """按主题规则分层挑选，答案明确（不过长）的优先。"""
    chosen = []
    seen = set()
    for topic, kws in TOPIC_RULES:
        cnt = 0
        for cid, it in items.items():
            if cnt >= PER_TOPIC:
                break
            if cid in seen:
                continue
            ans = it["a"]
            # 答案要含主题关键词、长度适中（明确），问题别太长
            if any(k in ans for k in kws) and 5 <= len(ans) <= 220 and 4 <= len(it["q"]) <= 60:
                chosen.append({"topic": topic, "gold_id": cid, "orig_q": it["q"], "gold_answer": ans})
                seen.add(cid)
                cnt += 1
    return chosen


def main():
    items = parse_kb(KB)
    print(f"解析到 {len(items)} 条FAQ")
    chosen = select(items)
    print(f"分层选出 {len(chosen)} 条：")
    for c in chosen:
        print(f"  [{c['topic']}] {c['gold_id']} | 问:{c['orig_q'][:30]} | 答:{c['gold_answer'][:30]}")
    json.dump(chosen, open(os.path.join(BASE, "_holdout_selected.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
