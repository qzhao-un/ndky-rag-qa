# -*- coding: utf-8 -*-
"""知问 ZhiWen · 招生顾问 Agent（阶段 1）。

在原有 RAG 之上升级：把"一次性检索 -> 一次性生成"升级为 Agent 自主循环。
- 工具 search_knowledge(query)：检索历史招生咨询知识库
- Agent 自己决定：要不要查、用什么关键词查、要不要换个词再查一轮、信息够了再作答
- 手写 ReAct 循环 + DeepSeek 原生 function calling，不依赖 LangChain / LangGraph

用法：
  先设好 key： $env:LLM_API_KEY="sk-..."
  再： python agent.py "宿舍是几人间？"
       python agent.py            # 交互模式
"""

import json
import sys

from bm25 import BM25Index
from config import (
    BM25_B, BM25_K1, EMBED_API_KEY, EMBED_API_MODEL, EMBED_API_URL, EMBED_MODE,
    EMBED_MODEL_NAME, INDEX_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, PROJECT_NAME,
    PROJECT_SLOGAN, USE_HYBRID,
)
from embedder import build_embedder
from qa import extract_latest_date, retrieve
from retriever import VectorStore
from admission import TOOL_SCHEMA as SCORE_TOOL, query_admission_score
from recommend import TOOL_SCHEMA as RECOMMEND_TOOL, recommend_majors
from school import TOOL_SCHEMA as SCHOOL_TOOL, school_info

# ---------- Agent 系统提示 ----------
AGENT_SYSTEM_PROMPT = (
    "你是宁波大学科学技术学院的招生顾问，负责解答考生关于学校招生的问题。\n"
    "你是一个会自主规划任务的 Agent，手上有四个工具：\n"
    "  1) search_knowledge：搜索学校历年真实招生咨询记录（政策、宿舍、转专业、学费、专业介绍等）；\n"
    "  2) query_admission_score：查询历年分省分专业录取最高/最低/平均分和位次；\n"
    "  3) recommend_majors：输入考生分数，把各专业分成 保底/稳妥/冲刺/偏险 四档；\n"
    "  4) school_info：学校硬事实（校区位置、办学性质、招生办电话等，不依赖召回）。\n\n"
    "【任务路由】拿到问题后，先判断意图，再决定调哪个/哪几个工具：\n"
    "  · 政策/宿舍/转专业/学费/专业介绍 -> search_knowledge\n"
    "  · 某专业历年分数/最高分/最低分/位次 -> query_admission_score\n"
    "  · 我XX分能上什么专业/冲稳保 -> recommend_majors\n"
    "  · 学校在哪/几个校区/什么性质 -> school_info\n"
    "  · 纯问候/闲聊 -> 直接回答，不用工具\n\n"
    "【多工具串联】一个问题可能需要多个工具：例如\"我560分报计算机，学费多少\"，"
    "要先 recommend_majors 看档位，再 query_admission_score 看计算机分数，再 search_knowledge 查学费，最后整合回答。"
    "每调完一个工具都判断：信息够了吗？不够就接着调下一个；够了就直接作答。\n\n"
    "【硬性要求】凡是涉及学校具体事实，必须真正调用工具去查；"
    "绝对不要只在文字里说\"稍等我查一下\"却不真正调用，也不要凭空编造。\n"
    "【格式硬约束】工具调用必须通过 function calling 协议（tool_calls 字段）发起，"
    "绝对不要在正文里输出 <||DSML||>、invoke name=、parameter name= 这类工具调用 XML 或标记，"
    "那些是系统内部协议，不是给用户看的。\n\n"
    "回答规则：\n"
    "- 直接、口语化，像老师当面回答，不要说\"根据参考资料\"这类开头；\n"
    "- 【严谨】不要说\"稳上/基本没悬念/100%能录\"这类打包票的话；"
    "冲稳保只是基于历史最低分的粗分档，要加一句\"历史分数仅供参考，实际录取受当年招生计划、位次、专业热度影响\"；\n"
    "- 【简短肯定词直接执行】当你上一句提议了某个动作（如\"要不要我帮你看看能冲哪些专业/查学费\"），"
    "用户回\"可以/好/行/要/嗯/查吧/看看/是的/OK\"等肯定词时，不要再反问，直接去执行那个动作（调对应工具）并给出结果；\n"
    "- 【多轮记忆】考生追问（如\"那软件工程呢\"\"那学费呢\"）时，自动结合上文已聊到的省份、分数、专业，"
    "把这些上下文填进工具参数，不要让用户重新说一遍；\n"
    "- 同一问题多年答案不同时，优先参考时间更新的说法；\n"
    "- 【硬规则·不调工具不许答】凡是问\"这个分数能不能报/稳不稳/够不够/有希望吗\"，"
    "必须先调用 recommend_majors 或 query_admission_score 拿到真实分数数据，"
    "再基于数据给结论；绝对不许不调工具就直接反问或空泛回答；\n"
    "- 【先给结论再追问】回答时先正面给判断（比如\"540分报机械比较稳，比去年最低分高30分\"），"
    "不要只反问\"要不要我再查XX\"就结束；追问建议放最后一句；\n"
    "- 不要反问\"你想了解哪方面\"让用户重述；查不到就说\"这一项暂时没查到，建议直接联系招生办 0574-87600018\"；\n"
    "- 问分数/位次/冲稳保用对应工具查完再答；查不到再建议打招生办电话，不要编造；\n"
    "- 用简体中文，简洁亲切。"
)

# ---------- 工具定义（OpenAI function calling schema）----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "搜索宁波大学科学技术学院历年真实招生咨询知识库，返回相关的问答片段。"
                "涉及招生政策、专业介绍、宿舍、转专业、学费、录取、档案、报到等校内问题时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索的关键词或短句，聚焦一个点，例如\"宿舍几人间\"\"转专业要求\"",
                    }
                },
                "required": ["query"],
            },
        }
    }
]
TOOLS.append(SCORE_TOOL)
TOOLS.append(RECOMMEND_TOOL)
TOOLS.append(SCHOOL_TOOL)

TOOL_DESC = "可用工具：search_knowledge / query_admission_score"


class KnowledgeSearchTool:
    """把现有混合检索包装成 Agent 可调用的工具。"""

    def __init__(self, embedder, store, bm25, use_hybrid=True):
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.use_hybrid = use_hybrid

    def search(self, query):
        """执行检索，返回精简后的文本（给 LLM 看，控制 token）。"""
        hits, _, _ = retrieve(
            query, self.embedder, self.store, self.bm25, use_hybrid=self.use_hybrid
        )
        if not hits:
            return f"（用关键词「{query}」未检索到相关咨询记录）"
        # 时间近优先：最新的排在前面
        hits = sorted(hits, key=lambda h: extract_latest_date(h[1]), reverse=True)
        parts = []
        for i, (score, text, meta) in enumerate(hits[:4]):
            snippet = text.strip().replace("\n", " ")[:350]  # 每条截短，控制上下文
            parts.append(f"[片段{i+1}]({meta}) {snippet}")
        return "\n".join(parts)


def _has_valid_key():
    return bool(LLM_API_KEY) and "sk-" in LLM_API_KEY \
        and "填你的" not in LLM_API_KEY and "在这里填" not in LLM_API_KEY


def parse_dsml_tool_calls(content):
    """兼容 DeepSeek DSML 工具调用：模型偶尔把调用写在正文里而非 tool_calls 字段。"""
    if not content or "DSML" not in content:
        return []
    calls = []
    pattern = re.compile(
        r'invoke\s+name="([^"]+)"(.*?)(?:</.*?invoke>|$)',
        re.DOTALL
    )
    for match in pattern.finditer(content):
        name = match.group(1)
        body = match.group(2)
        args = {}
        param_pattern = re.compile(
            r'parameter\s+name="([^"]+)"\s+string="([^"]+)">(.*?)</.*?parameter>',
            re.DOTALL
        )
        for pm in param_pattern.finditer(body):
            key = pm.group(1)
            is_string = pm.group(2) == "true"
            value = pm.group(3).strip()
            if is_string:
                args[key] = value
            else:
                try:
                    args[key] = json.loads(value)
                except json.JSONDecodeError:
                    args[key] = value
        calls.append({"name": name, "arguments": args})
    return calls


def run_agent(messages, question, tool, max_steps=4):
    """Agent 主循环（生成器）。

    messages：会话历史列表，第一项应为 system，其后是之前干净的 [user, assistant] 问答对。
              本函数会把本轮问答收敛成一对干净消息追加进 messages（ReAct 中间过程不长期保留）。

    yield 的事件：
      ("think", text)   —— Agent 的思考/行动说明（供前端展示 ReAct 轨迹）
      ("answer", chunk) —— 最终回答的流式片段
    """
    if not _has_valid_key():
        yield ("answer", "[未配置 LLM_API_KEY，已降级] 请设置 key 后再使用 Agent。")
        return

    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    # 本轮起始位置（append user 之后），结束后把 [起点:] 收敛成干净问答对
    messages.append({"role": "user", "content": question})
    start_idx = len(messages) - 1
    full_answer = ""
    sources = []  # 本轮引用的数据来源，回答结束后抛给前端展示

    def _stream_final(extra_user_note=None):
        nonlocal full_answer
        if extra_user_note is not None:
            messages.append({"role": "user", "content": extra_user_note})
        s = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.3, stream=True,
        )
        for chunk in s:
            if chunk.choices and chunk.choices[0].delta.content:
                full_answer += chunk.choices[0].delta.content
                yield ("answer", chunk.choices[0].delta.content)
        if not full_answer.strip():
            # 模型空回答兜底，避免屏幕空白
            fallback = "抱歉，暂时没查到相关信息，建议直接联系招生办 0574-87600018 确认。"
            full_answer = fallback
            yield ("answer", fallback)

    try:
        for step in range(max_steps):
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                stream=False,
            )
            msg = resp.choices[0].message

            # 统一成 [{id, name, arguments(dict)}]，兼容标准 tool_calls 和 DeepSeek DSML 正文调用
            tool_calls = []
            for tc in (msg.tool_calls or []):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
            if not tool_calls:
                for i, c in enumerate(parse_dsml_tool_calls(msg.content or "")):
                    tool_calls.append({"id": f"dsml_{i}", "name": c["name"], "arguments": c["arguments"]})

            if not tool_calls:
                # msg.content 就是模型第一次生成的最终回答，直接流式吐出，不再重调 LLM
                messages.append({"role": "assistant", "content": msg.content or ""})
                yield ("sources", list(dict.fromkeys(sources)))
                if msg.content and msg.content.strip():
                    full_answer = msg.content
                    # 模拟打字机效果，按片段吐出
                    chunk = 40
                    for i in range(0, len(msg.content), chunk):
                        yield ("answer", msg.content[i:i + chunk])
                else:
                    yield from _stream_final()
                return

            # 有工具调用：记录思考，逐个执行，把结果喂回下一轮
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                args = tc["arguments"]
                fname = tc["name"]
                if fname == "search_knowledge":
                    q = args.get("query", question)
                    yield ("think", f"🔍 检索知识库：「{q}」")
                    result = tool.search(q)
                    yield ("think", "   ↳ 已返回相关片段")
                    sources.append(("招生网 · 历年招生咨询", "https://zs.ndky.edu.cn"))
                elif fname == "query_admission_score":
                    major = args.get("major", question)
                    prov = args.get("province", "浙江")
                    yield ("think", f"📊 查录取分数：{prov} · {major}")
                    result = query_admission_score(major, prov)
                    yield ("think", "   ↳ 已返回历年分数")
                    sources.append(("招生网 · 历年录取分数", "https://zs.ndky.edu.cn/history/index.jhtml"))
                elif fname == "recommend_majors":
                    score = args.get("score", 0)
                    prov = args.get("province", "浙江")
                    yield ("think", f"🎯 志愿分档：{prov} · {score} 分")
                    result = recommend_majors(prov, score, args.get("rank"))
                    yield ("think", "   ↳ 已返回冲稳保分档")
                    sources.append(("招生网 · 历年录取分数", "https://zs.ndky.edu.cn/history/index.jhtml"))
                elif fname == "school_info":
                    yield ("think", "🏫 查学校基本信息")
                    result = school_info(args.get("topic", ""))
                    yield ("think", "   ↳ 已返回学校事实")
                    sources.append(("学校官网", "https://www.ndky.edu.cn"))
                else:
                    result = "（未知工具）"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        # 超过最大步数：基于已有信息总结
        yield ("think", "检索轮次已达上限，基于已有信息作答。")
        yield ("sources", list(dict.fromkeys(sources)))
        yield from _stream_final("请基于上面已检索到的信息，直接回答我最初的问题。")
    finally:
        # 把本轮 ReAct 中间过程收敛成一对干净的 [user, assistant]，保持历史轻量
        messages[start_idx:] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": full_answer},
        ]
        # 简单控制历史长度：只保留 system + 最近 6 轮问答对（12 条）
        if len(messages) > 13:
            messages[1:13] = []  # 删除中间旧轮，保留 system 与最近几轮


def build_tool():
    """加载索引和检索组件，返回 KnowledgeSearchTool。"""
    embedder = build_embedder(
        EMBED_MODE, EMBED_MODEL_NAME, EMBED_API_URL, EMBED_API_KEY, EMBED_API_MODEL,
    )
    store = VectorStore(INDEX_PATH)
    if len(store.chunks) == 0:
        print("索引为空，请先运行：python ingest.py")
        sys.exit(1)
    bm25 = None
    if USE_HYBRID:
        bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
        if not bm25.load(INDEX_PATH):
            print("[警告] 未找到 BM25 索引，退化为纯向量检索。")
            bm25 = None
    return KnowledgeSearchTool(embedder, store, bm25, use_hybrid=USE_HYBRID)


def main():
    print(f"=== {PROJECT_NAME} · 招生顾问 Agent ===")
    if not _has_valid_key():
        print("[提示] 未检测到有效 LLM_API_KEY，请先：$env:LLM_API_KEY=\"sk-...\"")
    tool = build_tool()
    # 会话历史：system 固定开头，之后逐轮追加干净的问答对，实现多轮记忆
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    def ask(question):
        print("Agent:", end="", flush=True)
        for kind, payload in run_agent(messages, question, tool):
            if kind == "think":
                print(f"\n  {payload}", flush=True)
            else:
                print(payload, end="", flush=True)
        print()

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("Agent 已启动，输入问题回车，exit 退出。支持追问，例如先问宿舍再问'那转专业呢'。")
        while True:
            q = input("你: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q:
                ask(q)


if __name__ == "__main__":
    main()
