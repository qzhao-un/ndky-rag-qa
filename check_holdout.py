# -*- coding: utf-8 -*-
"""校验评测集无泄漏，并剔除数据本身答非所问/残句的条目，重排id。"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
kb = open(os.path.join(BASE, "data", "ndky_qa_kb.md"), encoding="utf-8").read()
hold = json.load(open(os.path.join(BASE, "eval_holdout.json"), encoding="utf-8"))

# 1) 泄漏校验：改写问题完整串不得出现在知识库
leak = [c["id"] for c in hold if c["question"] in kb]
print("完整问题泄漏数:", len(leak), leak)

# 2) 连续相同字数粗检（>=10 视为照抄风险）
def max_common(a, b, n=10):
    for i in range(len(a) - n + 1):
        if a[i:i+n] in b:
            return a[i:i+n]
    return None
risk = []
for c in hold:
    m = max_common(c["question"], c["gold_answer"], 12)  # 和答案比
    m2 = max_common(c["question"], c["orig_question"], 12)  # 和原问题比
    if m2:
        risk.append((c["id"], m2))
print("与原问题连续12字相同（照抄风险）:", risk)

# 3) 剔除答非所问/残句
DROP = {"咨询183", "咨询300", "咨询1324"}
kept = [c for c in hold if c["gold_id"] not in DROP]
for i, c in enumerate(kept, 1):
    c["id"] = i
json.dump(kept, open(os.path.join(BASE, "eval_holdout.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"剔除 {len(hold)-len(kept)} 条，最终 {len(kept)} 条")
