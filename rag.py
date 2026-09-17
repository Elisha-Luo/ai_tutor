# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# RAG 的第二步：生成与引用层
#
# 【和 retriever.py 的分工】
#   retriever.py —— 找资料。把用户问题变成「最相关的几段」，纯本地、不调模型。
#   rag.py       —— 用资料。把「问题 + 那几段资料」交给模型，让它照着回答，
#                   并且要求它说清楚每句话出自哪一段（citation）。
#
# 「先检索、再照着检索结果回答」——这就是 RAG 的核心思路。
# =====================================================================
# 【这一层真正的工作量在哪：不在「调模型」，而在「不信模型」】
#
# 模型非常擅长一本正经地编——不光编内容，还会编来源。
# 所以这个文件里绝大部分代码不是「发给模型什么」，而是「拿回答案之后逐条核对」：
#
#   · 它引用的 source + heading，真的在我这次给它的片段里吗？
#   · 它说 decision 是 answer，那 citation 呢？空的就直接否掉。
#   · 它返回的压根不是 JSON 呢？抛异常了呢？
#
# 任何一条对不上，就安全降级成 insufficient_evidence。
# 【宁可说「证据不足」，也绝不放一条编的回答过去。】
#
# 为什么这件事值得花这么多代码：RAG 相比普通对话最大的卖点是「有据可查」。
# 如果引用可以随口编，那这个卖点就归零了——用户看到一个像模像样的来源，
# 反而会更相信一段编出来的内容，比没有引用还危险。
# =====================================================================

import json      # 解析模型返回的 JSON


# ===================== 三种结论 =====================

DECISION_ANSWER = "answer"                        # 资料能完整支持，给出回答
DECISION_REFUSE = "refuse"                        # 资料里完全没有相关内容
DECISION_INSUFFICIENT = "insufficient_evidence"   # 资料沾边，但不足以完整回答

VALID_DECISIONS = {DECISION_ANSWER, DECISION_REFUSE, DECISION_INSUFFICIENT}


# ===================== 兜底文案 =====================
# 【为什么这两条一律用固定文字，不用模型给的那段】
# 因为 refuse 和 insufficient_evidence 本身就是「模型不可信」时的安全网。
# 把对外话术再交回给模型，等于把安全网又交回给它——它可能在这里继续夹带私货。
# 固定文案还有第二个好处：行为确定、可测试。

REFUSE_TEXT = (
    "资料里没有和这个问题相关的内容，我不能凭猜想回答。"
    "如果你需要，可以换个问法，或者补充说明你想了解的是哪一部分。"
)

INSUFFICIENT_TEXT = (
    "资料里提到了相关的话题，但信息不够完整，我不确定能不能准确回答这个问题。"
    "为了不给你一个没有依据的说法，这里先不展开——建议查阅更完整的资料，或向老师确认。"
)


# ===================== 给模型的提示词 =====================

SYSTEM_PROMPT = """你是一名严谨的 AI 助教。你只能依据用户提供的「资料片段」回答问题，不许使用任何资料之外的知识。

你必须判断当前属于下面哪一种情况，并如实填写：

1. answer —— 资料片段能够完整支持一个回答
2. insufficient_evidence —— 资料片段涉及了相关话题，但缺少关键信息，不足以完整回答。这时不要猜测、不要脑补，如实返回这个结论
3. refuse —— 资料片段和用户的问题完全无关，没有任何可用内容

引用规则（非常严格，务必遵守）：

- decision 为 answer 时，citations 至少要有 1 项
- 每一项 citation 的 source 和 heading，必须与上面资料片段里「来源：」「标题：」后面写的文字完全一致（一个字、一个标点都不能改），只能从提供的资料片段里选
- 绝对不许自己编造文件名或标题，也不许引用没有出现在本次资料片段里的内容
- decision 不是 answer 时，citations 必须是空数组 []

输出要求：

- 只输出一个 JSON 对象，不要输出 Markdown 代码块（不要用 ``` 包裹），不要任何解释文字、前后缀
- JSON 格式必须严格如下：

{"decision": "answer", "answer": "给用户看的中文回答", "citations": [{"source": "文件名", "heading": "二级标题"}]}

- answer 字段是要直接给用户看的中文回答，语气自然、准确、简洁
"""


def _build_user_prompt(question, chunks):
    """把资料片段和用户问题拼成这一次要发给模型的内容。

    【为什么用字符串拼接，不用 .format()】
    这段提示词里含有 JSON 的 { }。一旦用 .format()，Python 会把花括号当成
    占位符，直接抛 KeyError。evaluate.py 里是靠把 { 写成 {{ 才躲过去的，
    写法很丑而且容易改错。所以这里一律用 + 拼接，绝不对含 JSON 的提示词 format()。
    """
    blocks = []                                    # 每段资料拼成一小块文字
    for i, c in enumerate(chunks, 1):              # 从 1 开始编号，方便模型引用
        blocks.append(
            "[" + str(i) + "]\n"
            "来源：" + str(c.get("source", "")) + "\n"
            "标题：" + str(c.get("heading", "")) + "\n"
            "正文：" + str(c.get("text", "")).strip()
        )

    material = "\n\n".join(blocks)                 # 各段之间空一行

    return (
        "下面是本次可用的资料片段。\n\n"
        "======== 资料片段开始 ========\n"
        + material +
        "\n======== 资料片段结束 ========\n\n"
        "用户的问题：\n" + str(question) + "\n\n"
        "请严格按照 system 里的规则判断 decision，并只输出那一个 JSON 对象。"
    )


# ===================== 小工具 =====================

def _parse_json(text):
    """把模型返回的文字解析成 Python 字典。

    模型经常不听话，会给 JSON 套一层 ```json 代码块——先把那层壳剥掉再解析。

    【行为是确定的，两种结果二选一，没有中间态】
      · 能剥出合法 JSON  → 返回字典
      · 剥不出合法 JSON  → 抛异常，由调用方统一降级成 insufficient_evidence
    无论哪种，都不会让程序崩掉。
    """
    if not isinstance(text, str):                  # 模型没返回文字（比如 None）
        raise ValueError("模型没有返回文字")

    t = text.strip()

    if t.startswith("```"):                        # 开头是三个反引号，说明套了代码块
        t = t.split("\n", 1)[1] if "\n" in t else ""    # 去掉第一行（``` 或 ```json）
        if t.rstrip().endswith("```"):             # 结尾也有三个反引号
            t = t.rstrip()[:-3]                    # 就把结尾那三个也去掉

    return json.loads(t.strip())


def _normalize_citations(raw, allowed):
    """校验模型给的 citations，返回干净的一份；只要有一处不合法就返回 None。

    allowed 是这次【真实传给模型】的片段集合，元素是 (source, heading)。
    这就是所谓「白名单」——引用只能从这个集合里挑，别的一律不认。

    【为什么一条坏引用就否掉整份，而不是把坏的那条删掉】
    因为一条引用是编的，说明这个回答整体就不可信了——它可能连内容也是编的。
    只删掉坏引用、留下回答，等于把一段没有依据的话包装成「有引用」的样子，
    那比直接拒绝更危险。
    """
    if not isinstance(raw, list):                  # 不是列表，格式就不对
        return None

    out = []
    seen = set()                                   # 去重用，避免同一段被引用两遍

    for item in raw:
        if not isinstance(item, dict):
            return None

        src = item.get("source")
        head = item.get("heading")
        if not isinstance(src, str) or not isinstance(head, str):
            return None

        key = (src.strip(), head.strip())

        if key not in allowed:
            # 【最关键的一行】引用了这次没给它的来源——要么文件名是编的，
            # 要么标题是编的，要么引的是别的项目的资料。一律不认。
            return None

        if key not in seen:
            seen.add(key)
            out.append({"source": key[0], "heading": key[1]})

    return out


def _degrade():
    """安全降级：任何一条校验没过，都回到这个结果。

    【为什么不把模型的原始错误信息告诉用户】
    原始错误可能包含内部细节、堆栈、甚至请求内容。对用户没有意义，
    还可能泄露不该露的东西。用户只需要知道「这次没能给出可靠回答」就够了。
    """
    return {
        "decision": DECISION_INSUFFICIENT,
        "answer": INSUFFICIENT_TEXT,
        "citations": [],
    }


# ===================== 对外的主函数 =====================

def generate_answer(question, chunks, client, model):
    """把「问题 + 已检索到的片段」交给模型，返回一个**可检查**的结果。

    参数：
        question —— 用户问题（字符串）
        chunks   —— retriever.py 返回的检索片段列表，每项含 source / heading / text
        client   —— 外部传入的模型客户端（本模块不自己建客户端，也不碰密钥）
        model    —— 外部传入的模型名

    返回（格式固定，永远是这三个键）：
        {
          "decision": "answer" | "refuse" | "insufficient_evidence",
          "answer":   "给用户看的中文回答",
          "citations": [{"source": "文件名", "heading": "二级标题"}]
        }

    【为什么 client 和 model 要从外面传进来，而不是在这里读环境变量自己建】
    这样这个模块就完全不碰密钥、不依赖网络库，测试时塞一个假的客户端进去
    就能把每条分支都跑一遍。职责单一，也更好测。
    """

    # ---------- 第一道关：根本没有资料，直接拒答，【不调用模型】 ----------
    # 没有资料却还要问模型，等于逼着它凭记忆回答——那正是我们要避免的事。
    if not chunks:
        return {
            "decision": DECISION_REFUSE,
            "answer": REFUSE_TEXT,
            "citations": [],
        }

    # 这一批片段的「白名单」。后面所有引用都要拿它来对照。
    allowed = set()
    for c in chunks:
        allowed.add((str(c.get("source", "")).strip(),
                     str(c.get("heading", "")).strip()))

    # ---------- 第二道关：调用模型（任何异常都不外泄） ----------
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(question, chunks)},
            ],
        )
        raw = response.choices[0].message.content
    except Exception:
        # 网络抖了、模型炸了、回复结构不对……用户都不需要知道细节。
        # 注意这里【没有】把异常对象存下来、也没有往 answer 里塞 str(e)。
        return _degrade()

    # ---------- 第三道关：解析 JSON ----------
    try:
        data = _parse_json(raw)
    except Exception:
        return _degrade()

    if not isinstance(data, dict):
        return _degrade()

    # ---------- 第四道关：decision 必须是那三个值之一 ----------
    decision = data.get("decision")
    if decision not in VALID_DECISIONS:
        return _degrade()

    # ---------- 第五道关：citations 必须全部来自本次传入的片段 ----------
    citations = _normalize_citations(data.get("citations", []), allowed)
    if citations is None:
        return _degrade()

    # ---------- 第六道关：分情况裁决 ----------
    if decision == DECISION_ANSWER:
        text = data.get("answer")

        # 回答内容不能是空的——说好要回答，却没给内容，属于格式不合格
        if not isinstance(text, str) or not text.strip():
            return _degrade()

        # 【answer 必须至少有一条引用】没有引用 = 说完话不给出处 = 不可查证。
        # 这正是 RAG 存在的意义，缺了就否掉，不留情面。
        if not citations:
            return _degrade()

        return {
            "decision": DECISION_ANSWER,
            "answer": text.strip(),
            "citations": citations,
        }

    # 走到这里只剩 refuse 和 insufficient_evidence 两种。
    # 【它们必须没有引用】没给答案却给了出处，本身自相矛盾；
    # 更麻烦的是，这种「无效却看着有据」的输出最容易骗到用户。
    if citations:
        return _degrade()

    return {
        "decision": decision,
        "answer": REFUSE_TEXT if decision == DECISION_REFUSE else INSUFFICIENT_TEXT,
        "citations": [],
    }
