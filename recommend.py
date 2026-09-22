# -*- coding: utf-8 -*-
"""志愿规划工具：输入考生分数/位次，对比历年录取分，给出 冲/稳/保 分档建议。"""

from admission import load_scores


def recommend_majors(province, score, rank=None):
    """根据考生分数（及位次），对比该省最近一年各专业录取最低分，分档推荐。

    province: 省份（默认浙江）
    score: 考生高考分数
    rank: 考生位次（可选；浙江有位次数据时优先参考）
    """
    rows = load_scores()
    # 只取该省最近一年（年份最大）的普通类
    years = sorted({r["year"] for r in rows if r["province"] == province and r["category"] == "普通类"}, reverse=True)
    if not years:
        return f"（暂无 {province} 的录取数据）"
    latest = years[0]
    cand = [r for r in rows if r["province"] == province and r["year"] == latest and r["category"] == "普通类"]
    # 同专业同名去重（取最低分那条）
    best = {}
    for r in cand:
        k = (r["major"], r["subject"])
        if k not in best or (r["min"] or 999) < (best[k]["min"] or 999):
            best[k] = r
    cand = list(best.values())

    buckets = {"保底": [], "稳妥": [], "冲刺": [], "偏险": []}
    for r in cand:
        lo = r["min"]
        if lo is None:
            continue
        gap = score - lo  # 正：考生分高于最低分；负：低于
        if gap >= 15:
            buckets["保底"].append(r)
        elif gap >= -10:
            buckets["稳妥"].append(r)
        elif gap >= -30:
            buckets["冲刺"].append(r)
        else:
            buckets["偏险"].append(r)

    lines = [f"根据你 {score} 分（{province}），对比 {latest} 年各专业最低分："]
    label_map = [
        ("保底", f"相对稳妥（比最低分高15分以上）"),
        ("稳妥", f"把握较大（接近最低分上下10分）"),
        ("冲刺", f"有风险（比最低分低10~30分）"),
        ("偏险", f"风险较大（比最低分低30分以上）"),
    ]
    for key, label in label_map:
        arr = sorted(buckets[key], key=lambda r: r["min"], reverse=True)
        if not arr:
            continue
        names = "、".join(f"{r['major']}({r['subject']} 最低{r['min']})" for r in arr[:8])
        lines.append(f"【{key}】{label}：{names}")
    lines.append(
        f"\n注：以上按 {latest} 年最低分粗分档，仅供参考。"
        "历史最低分≠当年录取概率，实际录取受当年招生计划、考生位次、专业热度和选科要求影响，"
        "建议结合一分一段表和当年计划综合判断，最终以招生办为准。"
    )
    return "\n".join(lines)


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recommend_majors",
        "description": (
            "根据考生高考分数，对比宁波大学科学技术学院该省最近一年各专业录取最低分，"
            "把专业分成 保底/稳妥/冲刺/偏险 四档，给志愿填报参考。"
            "当考生说'我XX分能上什么专业/帮我看看志愿/冲稳保推荐'时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "province": {"type": "string", "description": "省份，考生没说就传 浙江"},
                "score": {"type": "number", "description": "考生高考总分"},
                "rank": {"type": "number", "description": "考生位次（可选，知道就填）"},
            },
            "required": ["score"],
        },
    },
}
