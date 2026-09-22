# -*- coding: utf-8 -*-
"""录取分数查询工具：基于 crawl_scores.py 抓到的 data/scores.json。

供 Agent 调用：根据专业名（和省份）查历年录取最低分/位次。
"""

import json
import os

_SCORES_PATH = os.path.join("data", "scores.json")
_CACHE = None


def load_scores():
    global _CACHE
    if _CACHE is None:
        with open(_SCORES_PATH, encoding="utf-8") as f:
            _CACHE = json.load(f)
    return _CACHE


def query_admission_score(major, province="浙江"):
    """模糊查询某专业在指定省份的历年录取最低分。

    返回给 LLM 看的简洁文本。
    """
    rows = load_scores()
    # 专业模糊匹配：用户说"计算机" -> 匹配 major 含"计算机"
    matched = [r for r in rows if major in r["major"]]
    if not matched:
        # 尝试去掉"学/类"等字的粗匹配
        matched = [r for r in rows if r["major"].replace("学", "") in major or major in r["major"]]
    if not matched:
        return f"（未在历年录取数据中找到专业「{major}」，请确认专业全称）"

    # 优先匹配指定省份；该省没有再给全部省份（但限制条数）
    prov_rows = [r for r in matched if r["province"] == province]
    if not prov_rows:
        prov_rows = matched
        prov_note = f"（{province}未收录，以下为各省数据）"
    else:
        prov_note = ""

    # 按年份降序，同专业同名合并
    prov_rows.sort(key=lambda r: (-r["year"], r["major"]))
    lines = [f"专业「{major}」历年录取分数{prov_note}（最高/最低/平均）:"]
    seen = set()
    for r in prov_rows[:12]:
        key = (r["year"], r["major"], r["province"], r["subject"])
        if key in seen:
            continue
        seen.add(key)
        rank = f"，位次{int(r['rank'])}" if r.get("rank") else ""
        hi = f"最高{r['max']} / " if r.get("max") is not None else ""
        lo = f"最低{r['min']}" if r.get("min") is not None else ""
        avg = f" / 平均{r['avg']}" if r.get("avg") is not None else ""
        lines.append(f"{r['year']}年 {r['province']} {r['major']}（{r['subject']}）"
                     f"{hi}{lo}{avg}{rank}")
    return "\n".join(lines)


# Agent 工具的 function calling schema
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_admission_score",
        "description": (
            "查询宁波大学科学技术学院历年分省分专业录取最低分和位次。"
            "当考生问\"某专业多少分能上/历年分数线/录取分数/多少位次\"时调用。"
            "需要知道考生所在省份和专业名；考生没说省份时默认按浙江。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "major": {"type": "string", "description": "要查询的专业名，如 计算机科学与技术 / 汉语言文学"},
                "province": {"type": "string", "description": "省份，如 浙江/安徽/江苏；考生没说就传 浙江"},
            },
            "required": ["major"],
        },
    },
}
