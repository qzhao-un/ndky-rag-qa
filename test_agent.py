# -*- coding: utf-8 -*-
"""招生顾问 Agent 测试（pytest）。

分层：
- 单元测试：不依赖 LLM、不联网，验证工具函数、数据、解析器、索引，CI 上跑。
- 集成测试：端到端调用 LLM，需 LLM_API_KEY，标记 integration，CI 默认跳过。

本地跑：pip install -r requirements-dev.txt 后 pytest -v
只跑单元：pytest -v -m "not integration"
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from agent import (  # noqa: E402
    AGENT_SYSTEM_PROMPT,
    build_tool,
    parse_dsml_tool_calls,
    run_agent,
)
from admission import load_scores, query_admission_score  # noqa: E402
from recommend import recommend_majors  # noqa: E402
from school import school_info  # noqa: E402
from config import CHUNK_SIZE  # noqa: E402

# 标签全部拆开构造，避免源码里直接出现完整标记
LT = "<"
DSML = LT + "||DSML||" + ">"
INVOKE_OPEN = LT + 'invoke name="{}">'
INVOKE_CLOSE = LT + "/invoke>"
PARAM_OPEN = LT + 'parameter name="{}" string="{}">'
PARAM_CLOSE = LT + "/parameter" + ">"
PARAM = PARAM_OPEN + "{}" + PARAM_CLOSE


class TestModuleImports:
    def test_agent_importable(self):
        import agent
        assert hasattr(agent, "re")

    def test_system_prompt_nonempty(self):
        assert AGENT_SYSTEM_PROMPT and len(AGENT_SYSTEM_PROMPT) > 50


class TestDsmlParser:
    def test_no_dsml_returns_empty(self):
        assert parse_dsml_tool_calls("普通回答") == []
        assert parse_dsml_tool_calls("") == []
        assert parse_dsml_tool_calls(None) == []

    def test_parse_single_call(self):
        content = (
            DSML + INVOKE_OPEN.format("search_knowledge")
            + PARAM.format("query", "true", "宿舍几人间") + INVOKE_CLOSE
        )
        calls = parse_dsml_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_knowledge"
        assert calls[0]["arguments"]["query"] == "宿舍几人间"

    def test_parse_numeric_param(self):
        body = PARAM.format("score", "false", "560") + PARAM.format("province", "true", "浙江")
        content = DSML + INVOKE_OPEN.format("recommend_majors") + body + INVOKE_CLOSE
        calls = parse_dsml_tool_calls(content)
        assert calls[0]["arguments"]["score"] == 560
        assert calls[0]["arguments"]["province"] == "浙江"


class TestScoresData:
    def test_scores_nonempty(self):
        rows = load_scores()
        assert isinstance(rows, list)
        assert len(rows) > 1000

    def test_required_fields(self):
        required = {"major", "province", "year", "min", "category"}
        for r in load_scores()[:50]:
            assert required.issubset(r.keys())

    def test_years_covered(self):
        assert {2023, 2024, 2025}.issubset({r["year"] for r in load_scores()})


class TestAdmissionTool:
    def test_query_computer(self):
        result = query_admission_score("计算机科学与技术", "浙江")
        assert "2025" in result
        assert "507" in result

    def test_returns_full_stats(self):
        result = query_admission_score("计算机科学与技术", "浙江")
        assert "最高" in result
        assert "最低" in result

    def test_unknown_major(self):
        result = query_admission_score("不存在的奇葩专业xyz", "浙江")
        assert "未" in result or "确认" in result


class TestRecommendTool:
    def test_recommend_560(self):
        result = recommend_majors("浙江", 560)
        assert "保底" in result or "稳妥" in result

    def test_recommend_low_score(self):
        result = recommend_majors("浙江", 400)
        assert "偏险" in result or "冲刺" in result

    def test_unknown_province(self):
        assert "暂无" in recommend_majors("火星省", 560)

    def test_has_risk_disclaimer(self):
        result = recommend_majors("浙江", 560)
        assert "参考" in result or "以招生办为准" in result


class TestSchoolTool:
    def test_correct_main_campus(self):
        result = school_info()
        assert "白沙路" in result
        assert "文蔚路521号" in result

    def test_has_second_campus(self):
        assert "周巷" in school_info()

    def test_no_wrong_location(self):
        assert "杭州湾" not in school_info()

    def test_has_phone_and_code(self):
        result = school_info()
        assert "0574-87600018" in result
        assert "13277" in result


class TestIndex:
    @pytest.mark.xfail(reason="P1: 切块500字符超 bge 512token，待降到350并重建索引")
    def test_chunk_size_safe(self):
        assert CHUNK_SIZE <= 400

    def test_index_files_exist(self):
        assert os.path.exists(os.path.join(ROOT, "index", "vectors.npy"))
        assert os.path.exists(os.path.join(ROOT, "index", "chunks.json"))


# ============ 二、集成测试（依赖 LLM，CI 跳过）============

integration = pytest.mark.integration


@pytest.fixture(scope="module")
def live_tool():
    if not os.environ.get("LLM_API_KEY"):
        pytest.skip("未配置 LLM_API_KEY，跳过端到端测试")
    return build_tool()


def _run_once(tool, question):
    answer = []
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for kind, payload in run_agent(messages, question, tool):
        if kind == "answer":
            answer.append(payload)
    return "".join(answer)


@integration
class TestEndToEnd:
    def test_dorm(self, live_tool):
        ans = _run_once(live_tool, "宿舍是几人间？")
        assert any(k in ans for k in ["六", "寝", "宿舍"])

    def test_score_query(self, live_tool):
        ans = _run_once(live_tool, "浙江的计算机科学与技术去年最低多少分？")
        assert "507" in ans

    def test_recommend(self, live_tool):
        ans = _run_once(live_tool, "我浙江560分，能报什么专业？")
        assert "保底" in ans or "稳妥" in ans

    def test_school(self, live_tool):
        ans = _run_once(live_tool, "学校在哪个城市？")
        assert "慈溪" in ans

    def test_multiturn(self, live_tool):
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        gen1 = run_agent(messages, "我浙江560分报计算机怎么样？", live_tool)
        "".join(p for k, p in gen1 if k == "answer")
        gen2 = run_agent(messages, "那软件工程呢？", live_tool)
        a2 = "".join(p for k, p in gen2 if k == "answer")
        assert "软件工程" in a2