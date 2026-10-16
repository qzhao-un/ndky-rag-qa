# -*- coding: utf-8 -*-
"""学费查询工具：基于 2025 年招生章程的官方收费标准（硬事实，不依赖召回）。

供 Agent 调用：根据专业名返回学费/住宿费。确定性数字不交给 LLM 计算，避免编造。
- 普通高考本科按专业分四档：18000 / 20000 / 24700 / 27700 元/学年
- 广告学（中外合作办学）：36000 元/学年（前 3 学年）
- 住宿费：六人间 1800、四人间 3000 元/学年
"""

import re

# ---------- 住宿费 ----------
DORM_FEE = "六人间 1800 元/学年，四人间 3000 元/学年"

# ---------- 五年制专业 ----------
FIVE_YEAR = {"临床医学", "建筑学"}

# ---------- 学费档位（普通高考本科，元/学年）----------
TUITION = {
    18000: [
        "电子商务", "法学", "翻译", "商务英语", "日语", "工程管理",
        "应用化学", "生物技术", "社会体育指导与管理",
    ],
    20000: [
        "电子信息工程", "环境工程", "智能制造工程", "机械电子工程",
        "电气工程及其自动化", "工业设计", "视觉传达设计", "产品设计",
        "动画", "航空服务艺术与管理",
    ],
    24700: [
        "工商管理", "会计学", "物流管理", "旅游管理", "金融学",
        "国际经济与贸易", "汉语言文学", "英语", "数字经济",
    ],
    27700: [
        "计算机科学与技术", "软件工程", "人工智能", "机械设计制造及其自动化",
        "机器人工程", "土木工程", "建筑学", "环境设计", "工艺美术", "临床医学",
    ],
}

# 中外合作办学（特殊）
SINO_FOREIGN_FEE = 36000


def _build_index():
    idx = {}
    for fee, majors in TUITION.items():
        for m in majors:
            idx[m] = fee
    return idx


_MAJOR_FEE = _build_index()


def _normalize(name):
    """剥离括号后缀（专升本/单独考试/学前教育特色班/三年等），回到基础专业名。"""
    name = re.sub(r"[（(].*?[）)]", "", name or "")
    return name.strip()


def _all_tuition():
    """返回全部专业学费档位（用于'各专业学费/收费标准'这类总览问题）。"""
    lines = ["宁波大学科学技术学院 2025 年学费标准（普通高考本科，元/学年）："]
    for fee in sorted(TUITION):
        names = "、".join(
            m + ("（五年制）" if m in FIVE_YEAR else "") for m in TUITION[fee]
        )
        lines.append(f"· {fee} 元/学年：{names}")
    lines.append(
        "· 广告学（中外合作办学）：36000 元/学年（前 3 学年）；"
        "第 4 学年赴美朱尼亚塔学院，按当年美方标准（参考约 37200 美元/年）。"
    )
    lines.append(f"住宿费：{DORM_FEE}。")
    lines.append("最终以浙江省价格主管部门批准或报备为准。")
    return "\n".join(lines)


def query_tuition(major=""):
    """查某专业学费/住宿费，返回给 LLM 的简洁文本。"""
    key = (major or "").strip()

    # 无具体专业，或问的是总览
    if not key or key in ("学费", "各专业", "所有专业", "全部", "收费标准"):
        return _all_tuition()

    # 住宿费
    if any(k in key for k in ("住宿费", "宿舍", "住宿")):
        return f"住宿费标准：{DORM_FEE}（2025 年标准，最终以价格主管部门批准为准）。"

    norm = _normalize(key)

    # 广告学（中外合作办学）
    if "广告" in norm:
        return (
            "广告学（中外合作办学）学费：前 3 学年在国内 36000 元/学年；"
            "第 4 学年原则上赴美国朱尼亚塔学院，按当年美方标准（参考约 37200 美元/年）。"
            "该专业入校后不能转专业，外语单科成绩不低于 95 分。"
        )

    # 精确匹配
    fee = _MAJOR_FEE.get(norm)
    matched_name = norm if fee is not None else None

    # 模糊匹配：用户说"计算机" -> 计算机科学与技术；取专业名最长的，最精确
    if fee is None:
        cand = [m for m in _MAJOR_FEE if (norm in m) or (m in norm)]
        if cand:
            matched_name = max(cand, key=len)
            fee = _MAJOR_FEE[matched_name]

    if fee is None:
        return (
            f"（未在 2025 年学费标准中找到专业「{major}」，请确认专业全称；"
            "也可咨询招生办 0574-87600018）"
        )

    years = "（五年制）" if matched_name in FIVE_YEAR else ""
    return (
        f"{matched_name}{years}学费为 {fee} 元/学年（2025 年普通高考本科口径）；"
        f"住宿费{DORM_FEE}。最终以浙江省价格主管部门批准或报备为准。"
    )


# Agent 工具的 function calling schema
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_tuition",
        "description": (
            "查询宁波大学科学技术学院各专业学费和住宿费标准（2025 年招生章程官方口径）。"
            "当考生问\"某专业学费多少/住宿费多少/收费标准/一年多少钱\"时调用。"
            "问住宿费或全部专业时可不传专业。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "major": {
                    "type": "string",
                    "description": "专业名，如 计算机科学与技术 / 临床医学 / 广告学；问住宿费或全部标准可留空",
                },
            },
            "required": [],
        },
    },
}
