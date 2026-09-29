# -*- coding: utf-8 -*-
"""用 LLM 把选出的FAQ原问题改写成口语化同义问题（无泄漏评测query）。

输入 _holdout_selected.json，输出 eval_holdout.json。
改写要求：同义、换说法、不照抄、像真实考生提问。
"""

import json
import os
import re

from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

BASE = os.path.dirname(os.path.abspath(__file__))

INSTRUCT = """你在为一个高校招生问答系统构造检索评测题。
我会给你若干考生原始问题（带编号）。请把每条改写成【意思完全相同、但字面不同】的口语化问题，
用来测试检索系统对"换个说法"的鲁棒性。

硬性要求：
1. 语义、诉求、关键限定（省份/专业/年级/分数/专升本等）必须和原问题一致，不能增删信息；
2. 必须换同义词、语序或句式，改写后与原句不得有连续 8 个字相同，严禁整句照抄；
3. 长度和原问题相近，口语自然，像真实考生或家长打字问的；
4. 不要加"你好/请问/谢谢"之外的解释，只输出问题本身。

示例：
原：入校后可以转专业吗
改：进了学校之后还能不能换专业呀
原：怎么知道自己是几班的
改：去哪里能查到我被分到哪个班了
原：福建省本科批的录取有结果了吗
改：想问下福建本科批次的录取出来没有

只输出 JSON 数组，每个元素 {"id": 编号, "rewrite": "改写问题"}，不要输出其他内容。"""


def call_llm(batch):
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    numbered = "\n".join(f"{i+1}. {c['orig_q']}" for i, c in enumerate(batch))
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": INSTRUCT},
            {"role": "user", "content": numbered},
        ],
        temperature=0.5,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    data = json.loads(m.group(0) if m else raw)
    return {int(d["id"]): d["rewrite"] for d in data}


def main():
    selected = json.load(open(os.path.join(BASE, "_holdout_selected.json"), encoding="utf-8"))
    print(f"待改写 {len(selected)} 条，调 LLM ...")
    mapping = call_llm(selected)
    out = []
    for i, c in enumerate(selected):
        rw = mapping.get(i + 1)
        if not rw:
            print(f"  缺少第{i+1}条改写，跳过")
            continue
        out.append({
            "id": len(out) + 1,
            "topic": c["topic"],
            "question": rw,
            "gold_id": c["gold_id"],
            "orig_question": c["orig_q"],
            "gold_answer": c["gold_answer"],
        })
    json.dump(out, open(os.path.join(BASE, "eval_holdout.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"完成，写入 eval_holdout.json 共 {len(out)} 条：")
    for c in out:
        print(f"  {c['id']:>2} [{c['topic']}] {c['question']}  =>{c['gold_id']}")


if __name__ == "__main__":
    main()
