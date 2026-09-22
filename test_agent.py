# -*- coding: utf-8 -*-
"""招生顾问 Agent 测试用例。
跑法：
  $env:LLM_API_KEY="sk-..."
  python test_agent.py
"""

import os
import re
import sys

# 确保用国内 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from agent import build_tool, run_agent, AGENT_SYSTEM_PROMPT


CASES = [
    {
        "id": 1,
        "name": "知识库问答：宿舍",
        "input": "宿舍是几人间？",
        "expect_tools": ["search_knowledge"],
        "expect_keywords": ["六", "宿舍", "寝"],
    },
    {
        "id": 2,
        "name": "分数线查询：计算机最低分",
        "input": "浙江的计算机科学与技术去年最低多少分？",
        "expect_tools": ["query_admission_score"],
        "expect_keywords": ["507", "2025"],
    },
    {
        "id": 3,
        "name": "志愿分档：560分",
        "input": "我浙江560分，能报什么专业？",
        "expect_tools": ["recommend_majors"],
        "expect_keywords": ["保底", "稳妥"],
    },
    {
        "id": 4,
        "name": "学校硬事实：校区",
        "input": "学校在哪个城市？有几个校区？",
        "expect_tools": ["school_info"],
        "expect_keywords": ["宁波", "慈溪"],
    },
    {
        "id": 5,
        "name": "多轮追问：带上文",
        "input": ["我浙江560分报计算机怎么样？", "那软件工程呢？"],
        "expect_tools_chain": [
            ["recommend_majors", "query_admission_score"],
            ["query_admission_score"],
        ],
        "expect_keywords_chain": [
            ["560"],
            ["软件工程"],
        ],
    },
]


def run_single(tool, messages, question):
    """跑一轮，返回 (调用的工具列表, 完整回答, 所有 think 轨迹)。"""
    tools_called = []
    answer_parts = []
    thinks = []
    for kind, payload in run_agent(messages, question, tool):
        if kind == "think":
            thinks.append(payload)
            m = re.search(r"(🔍|📊|🎯|🏫)\s*(\S+)", payload)
            if m:
                tools_called.append(payload.split("：")[0].split(" ")[-1].strip())
        elif kind == "answer":
            answer_parts.append(payload)
    return tools_called, "".join(answer_parts), thinks


def main():
    print("=" * 60)
    print("招生顾问 Agent · 测试用例")
    print("=" * 60)
    tool = build_tool()
    passed = 0
    total = len(CASES)

    for case in CASES:
        print(f"\n【用例 {case['id']}】{case['name']}")
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        if isinstance(case["input"], str):
            # 单轮
            tc, ans, thinks = run_single(tool, messages, case["input"])
            print(f"  输入: {case['input']}")
            print(f"  调用工具: {tc}")
            print(f"  回答: {ans[:150]}...")
            ok = any(k in ans for k in case["expect_keywords"])
            print(f"  预期关键词 {case['expect_keywords']} -> {'✓ 通过' if ok else '✗ 未命中'}")
            if ok:
                passed += 1
        else:
            # 多轮
            ok = True
            for i, q in enumerate(case["input"]):
                print(f"  第{i+1}轮输入: {q}")
                tc, ans, thinks = run_single(tool, messages, q)
                print(f"    调用工具: {tc}")
                print(f"    回答: {ans[:120]}...")
                if not any(k in ans for k in case["expect_keywords_chain"][i]):
                    print(f"    ✗ 未命中预期关键词 {case['expect_keywords_chain'][i]}")
                    ok = False
            print(f"  多轮结果: {'✓ 通过' if ok else '✗ 失败'}")
            if ok:
                passed += 1

    print("\n" + "=" * 60)
    print(f"通过 {passed}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
