# -*- coding: utf-8 -*-
"""学校基本信息工具：硬编码学校事实，不靠 RAG 召回（避免答错校区）。"""

# 学校硬事实（来自学校官网/招生网公开信息；不确定的不要写）
SCHOOL_FACTS = """
宁波大学科学技术学院（简称"宁大科院"）基本信息：
- 办学性质：全日制普通本科高校，独立学院
- 国标代码：13277；浙江省代码：0094
- 主校区：浙江省宁波市慈溪市白沙路街道文蔚路521号
- 周巷校区：浙江省慈溪市周巷镇和通路588号
- 招生办电话：0574-87600018
- 招生网：https://zs.ndky.edu.cn
- 学校官网：https://www.ndky.edu.cn
""".strip()


def school_info(topic=""):
    """返回学校基本信息。topic 只是用来让 LLM 觉得参数化了，实际返回全套事实。"""
    return SCHOOL_FACTS


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "school_info",
        "description": (
            "查询宁波大学科学技术学院的基本事实：办学性质、校区位置（主校区在宁波慈溪，另有周巷校区）、"
            "招生办电话、官网等。当考生问\"学校在哪/有几个校区/学校什么性质/介绍一下学校\"时调用，"
            "这些是硬事实，不要靠知识库猜测。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "考生想问的方面，如 校区/性质/电话"}
            },
            "required": [],
        },
    },
}
