# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# rag.py（生成与引用层）的自动化测试
#
# 【原则】这些测试完全离线运行：不联网、不调用真实 DeepSeek、不需要密钥。
# 做法是把模型客户端换成一个假的——它返回什么，完全由每条测试自己决定。
# 这样就能把「模型不听话」的各种情况（编来源、漏引用、返回坏 JSON、直接报错）
# 一条条造出来，看看 rag.py 到底接不接得住。
#
# 运行方式（在 ai_tutor 文件夹里）：
#     python -m unittest test_rag -v
# =====================================================================

import os          # 拼路径（结构检查那条测试要读源码文件）
import json        # 拼造假的模型回复
import unittest    # Python 自带的测试框架

import rag         # 被测对象


# ===================== 假的模型客户端 =====================
# 这一组类的唯一作用，就是冒充真的 client。结构故意和 test_app.py 里那套保持一致，
# 因为它们冒充的是同一个接口：client.chat.completions.create(...)

class FakeMessage:
    def __init__(self, content):
        self.content = content              # 真货也是这个结构：response.choices[0].message.content

class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)

class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]

class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner                  # 回头去主人那里记一笔「我被调用过」
    def create(self, model, messages, **kwargs):
        self.owner.calls.append({"model": model, "messages": messages})   # 记下调用参数，方便断言
        return FakeResponse(self.owner.reply)

class FakeChat:
    def __init__(self, owner):
        self.completions = FakeCompletions(owner)

class FakeClient:
    def __init__(self):
        self.calls = []                     # 每次被调用都记在这里
        self.reply = "{}"                   # 默认回复；每条测试会改成自己需要的
        self.chat = FakeChat(self)


# ===================== 测试用的资料片段 =====================
# 故意用两份【不同文件、不同标题】的片段，这样才测得出「引用了没给它的来源」。

CHUNK_SINCE_FOR = {
    "source": "grammar_present_perfect.md",
    "heading": "since 和 for 的区别",
    "text": "since 后面接时间点，例如 since 2015；for 后面接时间段，例如 for ten years。",
}

CHUNK_TIME = {
    "source": "course_faq.md",
    "heading": "每天需要花多少时间",
    "text": "建议每天 20～30 分钟，关键是每天都做。",
}

CHUNKS = [CHUNK_SINCE_FOR, CHUNK_TIME]

MODEL = "test-model"          # 只是个名字，会被原样传给假客户端


# ===================== 测试基类 =====================

class RagTestCase(unittest.TestCase):

    def setUp(self):
        self.fake = FakeClient()

    def ask(self, question="since 和 for 有什么区别？", chunks=None, reply=None):
        """跑一次 generate_answer。reply 可以传字符串，也可以传 dict（自动转 JSON）。"""
        if chunks is None:
            chunks = [dict(c) for c in CHUNKS]        # 复制一份，避免测试之间互相污染
        if reply is not None:
            if isinstance(reply, str):
                self.fake.reply = reply
            else:
                self.fake.reply = json.dumps(reply, ensure_ascii=False)
        return rag.generate_answer(question, chunks, self.fake, MODEL)

    @staticmethod
    def answer_json(text="这是回答。", citations=None):
        """造一个格式正确的 answer 回复。"""
        if citations is None:
            citations = [{"source": CHUNK_SINCE_FOR["source"],
                          "heading": CHUNK_SINCE_FOR["heading"]}]
        return {"decision": "answer", "answer": text, "citations": citations}


# ===================== 1. 没有检索结果时不调用模型 =====================

class TestEmptyChunks(RagTestCase):
    """【本轮最关键的行为变化】chunks 为空时【也要】调用模型。

    以前「没检索到资料」等于直接拒答、不花钱。现在不行了 ——
    因为「检索不到」和「这是个正常的英语问题」完全是两回事：
    用户问「this 和 that 有什么区别」，知识库里没有，但那是个完全正当的问题，
    应该用通用知识回答（general_answer），而不是被拒答。

    【这个类的旧名字叫 TestNoChunks，里面的测试断言「不调用模型」——
      那些断言本轮已经全部作废，所以整个类重写了。】
    """

    def test_empty_chunks_still_calls_the_model(self):
        """【核心】chunks 为空时，仍然要调用模型。"""
        self.fake.reply = json.dumps(
            {"decision": "general_answer", "answer": "通用回答", "citations": []},
            ensure_ascii=False)
        self.ask(chunks=[])
        self.assertEqual(len(self.fake.calls), 1, "chunks 为空时居然没调用模型")

    def test_empty_chunks_can_return_general_answer(self):
        """【核心】chunks 为空时，可以返回 general_answer。"""
        r = self.ask(chunks=[], reply={"decision": "general_answer",
                                       "answer": "这是通用知识回答", "citations": []})
        self.assertEqual(r["decision"], "general_answer")
        self.assertIn("通用知识回答", r["answer"])
        self.assertEqual(r["citations"], [])

    def test_empty_chunks_can_still_return_refuse(self):
        """chunks 为空时也可以返回 refuse（比如问题超出英语学习范围）。"""
        r = self.ask(chunks=[], reply={"decision": "refuse", "answer": "x", "citations": []})
        self.assertEqual(r["decision"], "refuse")

    def test_none_chunks_does_not_crash(self):
        """chunks 传 None 也不能崩 —— 按空处理，照样调用模型。"""
        self.fake.reply = json.dumps(
            {"decision": "general_answer", "answer": "通用回答", "citations": []},
            ensure_ascii=False)
        r = rag.generate_answer("随便问", None, self.fake, MODEL)
        self.assertEqual(r["decision"], "general_answer")
        self.assertEqual(len(self.fake.calls), 1)

    def test_empty_chunks_cannot_produce_answer(self):
        """【核心】chunks 为空时不能返回 answer。

        没有片段可引用，任何引用都必然不在白名单里 —— 会被安全降级。
        这条保证「answer 一定有真实出处」这个承诺不会因为空检索而破掉。
        """
        r = self.ask(chunks=[], reply={
            "decision": "answer", "answer": "x",
            "citations": [{"source": "编的.md", "heading": "编的"}]})
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_empty_chunks_without_citations_is_refused(self):
        """chunks 为空时，模型说 answer 却给不出引用 —— 同样降级。"""
        r = self.ask(chunks=[], reply={"decision": "answer", "answer": "x", "citations": []})
        self.assertEqual(r["decision"], "insufficient_evidence")


# ===================== 2. 正常回答与合法引用 =====================

class TestValidAnswer(RagTestCase):

    def test_answer_with_valid_citation_passes(self):
        """模型给出合法回答 + 合法引用 → 原样通过。"""
        r = self.ask(reply=self.answer_json("since 接时间点，for 接时间段。"))
        self.assertEqual(r["decision"], "answer")
        self.assertIn("since", r["answer"])
        self.assertEqual(r["citations"],
                         [{"source": CHUNK_SINCE_FOR["source"],
                           "heading": CHUNK_SINCE_FOR["heading"]}])

    def test_answer_with_multiple_valid_citations_passes(self):
        """同时引用两个真实存在的片段，是允许的。"""
        r = self.ask(reply=self.answer_json(citations=[
            {"source": CHUNK_SINCE_FOR["source"], "heading": CHUNK_SINCE_FOR["heading"]},
            {"source": CHUNK_TIME["source"], "heading": CHUNK_TIME["heading"]},
        ]))
        self.assertEqual(r["decision"], "answer")
        self.assertEqual(len(r["citations"]), 2)

    def test_duplicate_citations_are_deduped(self):
        """同一条引用重复两次，会被去重成一条。"""
        same = {"source": CHUNK_SINCE_FOR["source"], "heading": CHUNK_SINCE_FOR["heading"]}
        r = self.ask(reply=self.answer_json(citations=[same, dict(same)]))
        self.assertEqual(r["decision"], "answer")
        self.assertEqual(len(r["citations"]), 1)

    def test_citation_whitespace_is_tolerated(self):
        """引用前后多打了空格，属于手滑，不该判为编造。"""
        r = self.ask(reply=self.answer_json(citations=[
            {"source": "  " + CHUNK_SINCE_FOR["source"] + " ",
             "heading": CHUNK_SINCE_FOR["heading"] + "  "},
        ]))
        self.assertEqual(r["decision"], "answer")

    def test_model_receives_question_and_chunk_text(self):
        """确认发给模型的内容里，确实有用户问题和片段正文。"""
        self.ask(question="这是个特定的问题标记")
        sent = self.fake.calls[0]
        self.assertEqual(sent["model"], MODEL)             # 模型名原样透传
        user_text = sent["messages"][1]["content"]
        self.assertIn("这是个特定的问题标记", user_text)
        self.assertIn(CHUNK_SINCE_FOR["text"], user_text)  # 片段正文也带上了
        self.assertIn(CHUNK_SINCE_FOR["heading"], user_text)


# ===================== 3. 编造来源会被拒绝 =====================

class TestFabricatedCitations(RagTestCase):

    def test_fabricated_source_is_rejected(self):
        """【核心】模型编了一个不存在的文件名 → 拒绝，降级为 insufficient_evidence。"""
        r = self.ask(reply=self.answer_json(citations=[
            {"source": "grammar_past_tense.md", "heading": CHUNK_SINCE_FOR["heading"]},
        ]))
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_fabricated_heading_is_rejected(self):
        """【核心】文件名是真的，但标题是编的（本次没传这段）→ 拒绝。"""
        r = self.ask(reply=self.answer_json(citations=[
            {"source": CHUNK_SINCE_FOR["source"], "heading": "一个根本不存在的标题"},
        ]))
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_citation_from_real_file_but_not_passed_is_rejected(self):
        """看起来像真的、但这次没传给它 → 照样拒绝。白名单只管「本次给了什么」。"""
        other = {"source": "vocabulary_study_method.md", "heading": "间隔重复"}
        r = self.ask(reply=self.answer_json(citations=[other]))
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_one_bad_citation_rejects_the_whole_answer(self):
        """一条真 + 一条假：整份都否掉，不做「只删坏的那条」的处理。"""
        r = self.ask(reply=self.answer_json(citations=[
            {"source": CHUNK_SINCE_FOR["source"], "heading": CHUNK_SINCE_FOR["heading"]},
            {"source": "编造的文件.md", "heading": "编造的标题"},
        ]))
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_citation_with_wrong_type_is_rejected(self):
        """citations 里的字段类型不对（不是字符串）→ 拒绝。"""
        r = self.ask(reply={"decision": "answer", "answer": "回答",
                            "citations": [{"source": 123, "heading": None}]})
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_citations_not_a_list_is_rejected(self):
        """citations 根本不是列表 → 拒绝。"""
        r = self.ask(reply={"decision": "answer", "answer": "回答",
                            "citations": "grammar_present_perfect.md"})
        self.assertEqual(r["decision"], "insufficient_evidence")


# ===================== 4. answer 没有引用会被拒绝 =====================

class TestAnswerWithoutCitations(RagTestCase):

    def test_answer_with_empty_citation_list_is_rejected(self):
        """【核心】说好了回答，却一条引用都不给 → 拒绝。没有出处的回答不可查证。"""
        r = self.ask(reply={"decision": "answer", "answer": "这是没有出处的回答", "citations": []})
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_answer_with_missing_citations_key_is_rejected(self):
        """citations 这个键干脆没给 → 同样拒绝。"""
        r = self.ask(reply={"decision": "answer", "answer": "没有 citations 字段"})
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_answer_with_blank_text_is_rejected(self):
        """decision 说是 answer，但正文是空的 → 拒绝。"""
        r = self.ask(reply=self.answer_json(text="   "))
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_answer_with_non_string_text_is_rejected(self):
        """正文不是字符串 → 拒绝。"""
        r = self.ask(reply={"decision": "answer", "answer": ["a", "b"],
                            "citations": [{"source": CHUNK_SINCE_FOR["source"],
                                           "heading": CHUNK_SINCE_FOR["heading"]}]})
        self.assertEqual(r["decision"], "insufficient_evidence")


# ===================== 5. refuse / insufficient_evidence 的引用必须为空 =====================

class TestRefuseAndInsufficient(RagTestCase):

    def test_refuse_without_citations_passes(self):
        """干净的拒答 → 通过。"""
        r = self.ask(reply={"decision": "refuse", "answer": "资料里没有", "citations": []})
        self.assertEqual(r["decision"], "refuse")
        self.assertEqual(r["citations"], [])

    def test_refuse_with_citations_is_rejected(self):
        """【核心】拒答却带着引用 → 自相矛盾，拒绝。"""
        r = self.ask(reply={"decision": "refuse", "answer": "资料里没有",
                            "citations": [{"source": CHUNK_SINCE_FOR["source"],
                                           "heading": CHUNK_SINCE_FOR["heading"]}]})
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_insufficient_without_citations_passes(self):
        """干净的「证据不足」→ 通过。"""
        r = self.ask(reply={"decision": "insufficient_evidence", "answer": "不够完整", "citations": []})
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_insufficient_with_citations_is_rejected(self):
        """「证据不足」却带着引用 → 拒绝（降级后仍然是 insufficient，但引用必须清空）。"""
        r = self.ask(reply={"decision": "insufficient_evidence", "answer": "不够完整",
                            "citations": [{"source": CHUNK_SINCE_FOR["source"],
                                           "heading": CHUNK_SINCE_FOR["heading"]}]})
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_unknown_decision_is_rejected(self):
        """decision 是四个合法值之外的任意东西 → 拒绝。"""
        for bad in ["maybe", "ANSWER", "", None, 1, "answer "]:
            r = self.ask(reply={"decision": bad, "answer": "x", "citations": []})
            self.assertEqual(r["decision"], "insufficient_evidence",
                             "decision=" + repr(bad) + " 居然被放过了")


# ===================== 5b. 提示词里 refuse / 证据不足 的边界 =====================
#
# 【为什么单开一组只测「提示词文字」】
# trap-one-on-one-tutoring 翻车不是代码 bug，而是「话没说清楚」：
# 资料讲了答疑机制、没讲一对一辅导，模型在 refuse 和 insufficient_evidence
# 之间选错了。模型到底怎么判，只能靠真实评测（--live）去验；
# 这里能守住的是——这段边界说明【真的发给模型了】，而且没被改回原来那句含糊的话。

class TestDecisionBoundaryPrompt(RagTestCase):
    """检查发给模型的 system 提示词里，两条规则的边界写清楚没有。"""

    def system_prompt(self):
        """跑一次，取回真正发给模型的 system 提示词。"""
        self.ask()
        first = self.fake.calls[0]["messages"][0]
        self.assertEqual(first["role"], "system", "第一条消息必须是 system")
        return first["content"]

    def test_boundary_is_stated_from_both_sides(self):
        """两条规则里都要写清分界线：资料提没提到相关的东西。"""
        p = self.system_prompt()
        self.assertIn("提到了与这个问题直接相关的服务、机制或相近事实", p)
        self.assertIn("一个字都没有", p)

    def test_one_on_one_tutoring_example_is_present(self):
        """一对一辅导这个具体例子必须在提示词里 —— 它就是翻车的那道题。"""
        p = self.system_prompt()
        self.assertIn("一对一辅导", p)
        self.assertIn("在学习群里提问", p)
        self.assertIn("在下次答疑时集中处理", p)

    def test_example_forbids_both_wrong_answers(self):
        """例子要同时点明两个错法：答「有的」，和说成「完全无关」。"""
        p = self.system_prompt()
        self.assertIn("不能顺着问题回答「有的」", p)
        self.assertIn("也不该说成「跟资料完全无关」", p)

    def test_refuse_side_also_has_a_concrete_example(self):
        """refuse 那一侧也要有例子，否则容易反向倒向「证据不足」。"""
        p = self.system_prompt()
        self.assertIn("这个课程多少钱", p)
        self.assertIn("完全没有", p)

    def test_disclaimer_line_does_not_count_as_mentioning(self):
        """【防止误伤 refuse-price】免责声明里出现「价格」不算「提到」。

        course_faq.md 顶部有一句「本文不包含价格、退费、证书信息」，
        问价格时这句很可能被检索到。如果模型把「出现过这个词」当成「提到过」，
        价格题就会从 refuse 翻成 insufficient_evidence。
        """
        p = self.system_prompt()
        self.assertIn("本文不包含价格、退费、证书信息", p)
        self.assertIn("那【不算】提到", p)

    def test_old_ambiguous_wording_is_gone(self):
        """原来那句「业务事实，而资料里没有记录」正是把这道题带偏的原因，不能留。"""
        self.assertNotIn("询问课程的业务事实，而资料里没有记录", self.system_prompt())


# ===================== 5c. 短期对话上下文 =====================
#
# 分成两组：
#   · TestRecentContextBudget        —— 纯函数，管「带几条、每条多长、总共多长、顺序」
#   · TestContextInThePrompt         —— 管「拼进提示词之后，安全边界还在不在」
#
# 【最要紧的一条】上下文【不是资料】。
# 历史里带着上一轮回答的「资料来源：xxx.md · yyy」，如果模型从那里挑来源就能通过校验，
# 引用白名单就形同虚设 —— 这是本轮最危险的一处，必须有测试钉住。

class TestRecentContextBudget(unittest.TestCase):
    """只测 rag.build_recent_context 这个纯函数：四条闸，一条一条来。"""

    @staticmethod
    def msg(role, text):
        return {"role": role, "content": text}

    def build(self, messages):
        return rag.build_recent_context(messages)

    # ---------- 上限本身 ----------

    def test_shipped_limits_are_the_documented_ones(self):
        """三个上限是写死的约定，不能被悄悄改掉（文档和测试都按这个数写）。"""
        self.assertEqual(rag.RECENT_MESSAGE_LIMIT, 6)
        self.assertEqual(rag.MAX_HISTORY_MESSAGE_CHARS, 1200)
        self.assertEqual(rag.MAX_HISTORY_TOTAL_CHARS, 4000)

    # ---------- 空输入 ----------

    def test_no_history_gives_an_empty_context(self):
        """None 和空列表都算「本次没有上下文」，不能崩，也不能拼出空壳。"""
        for empty in (None, []):
            self.assertEqual(self.build(empty), [], "输入=" + repr(empty))

    # ---------- 闸一：条数 ----------

    def test_at_most_six_messages_and_the_newest_win(self):
        """10 条历史只留 6 条，而且是【最新】那 6 条 —— 不是最早的。"""
        history = [self.msg("user", "第" + str(i) + "条") for i in range(10)]
        out = self.build(history)

        self.assertEqual(len(out), 6)
        self.assertEqual([e["text"] for e in out],
                         ["第4条", "第5条", "第6条", "第7条", "第8条", "第9条"])

    def test_output_is_in_chronological_order(self):
        """从最新那头挑选，但交出去时必须翻回「先问后答」的正常顺序。"""
        history = [self.msg("user", "问题一"), self.msg("assistant", "回答一"),
                   self.msg("user", "问题二"), self.msg("assistant", "回答二")]
        out = self.build(history)

        self.assertEqual([e["role"] for e in out], ["用户", "助手", "用户", "助手"])
        self.assertEqual([e["text"] for e in out], ["问题一", "回答一", "问题二", "回答二"])

    # ---------- 闸二：单条长度 ----------

    def test_a_single_message_is_truncated_to_the_cap(self):
        """单条超长 → 截断，并且末尾留一个省略号说明「这里被切了」。"""
        out = self.build([self.msg("assistant", "甲" * 3000)])

        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]["text"]), rag.MAX_HISTORY_MESSAGE_CHARS)
        self.assertTrue(out[0]["text"].endswith("…"))

    def test_a_huge_message_cannot_eat_the_whole_budget(self):
        """【核心】一条超长历史不能把 4000 的预算吃光。

        因为单条先被截到 1200（< 4000），所以它最多只能占掉一小部分，
        更老的正常消息仍然进得来 —— 这正是「不能让一条超长历史占满预算」。
        """
        history = [self.msg("user", "老问题"),
                   self.msg("assistant", "老回答"),
                   self.msg("assistant", "乙" * 10000)]        # 最新这条超大
        out = self.build(history)

        self.assertEqual([e["text"] for e in out],
                         ["老问题", "老回答", "乙" * 1199 + "…"])

    # ---------- 闸三：总量 ----------

    def test_total_budget_is_respected(self):
        """6 条 × 1000 字 = 6000，超过 4000 → 只留下装得下的那几条。"""
        history = [self.msg("user", "丙" * 1000) for _ in range(6)]
        out = self.build(history)

        total = sum(len(e["text"]) for e in out)
        self.assertLessEqual(total, rag.MAX_HISTORY_TOTAL_CHARS)
        self.assertEqual(len(out), 4, "1000 × 4 = 4000 刚好装满，第 5 条进不来")

    def test_the_oldest_are_the_ones_dropped(self):
        """预算不够时先丢最老的 —— 靠「刚才那句」的时候，最近的最有用。"""
        history = [self.msg("user", "标记" + str(i) + "-" + "丁" * 995) for i in range(6)]
        out = self.build(history)

        kept = [e["text"][:3] for e in out]
        self.assertEqual(kept, ["标记2", "标记3", "标记4", "标记5"], "丢的不是最老的")

    # ---------- 脏数据 ----------

    def test_unknown_roles_and_malformed_entries_are_dropped(self):
        """role 不认识、内容不是字符串、空内容 —— 一律丢掉，绝不猜、绝不塞空壳。"""
        out = self.build([
            self.msg("system", "系统消息不该进来"),
            self.msg("user", "正常消息"),
            {"role": "user"},                            # 缺 content
            {"role": "user", "content": None},            # content 不是字符串
            {"role": "user", "content": "   "},           # 只有空白
            "我压根不是字典",
            {"content": "没有 role"},
            self.msg("assistant", "正常回答"),
        ])

        self.assertEqual([e["text"] for e in out], ["正常消息", "正常回答"])

    def test_whitespace_is_stripped_from_the_edges(self):
        out = self.build([self.msg("user", "  前后有空格  ")])
        self.assertEqual(out[0]["text"], "前后有空格")

    def test_building_twice_gives_the_same_answer(self):
        """纯函数：同样的输入必须给同样的输出（app.py 靠这个算日志数字）。"""
        history = [self.msg("user", "问题"), self.msg("assistant", "回答")]
        self.assertEqual(self.build(history), self.build(history))


class TestContextInThePrompt(RagTestCase):
    """上下文拼进提示词之后：该带进去的带进去了，不该越的界一步没越。"""

    def ask_with_context(self, recent, question="再给一个例子", reply=None, chunks=None):
        """走【带上下文】的那个入口。"""
        if reply is not None:
            self.fake.reply = reply if isinstance(reply, str) else json.dumps(
                reply, ensure_ascii=False)
        if chunks is None:
            chunks = [dict(c) for c in CHUNKS]
        return rag.generate_answer_with_context(question, chunks, recent, self.fake, MODEL)

    def sent_user_text(self, index=0):
        """取出发给模型的 user 消息正文。"""
        return self.fake.calls[index]["messages"][1]["content"]

    @staticmethod
    def msg(role, text):
        return {"role": role, "content": text}

    # ---------- 上下文确实进去了 ----------

    def test_recent_history_reaches_the_model(self):
        """【核心】上一轮的原话必须出现在这次发给模型的内容里。"""
        self.ask_with_context([self.msg("user", "帮我把这句改简单一点"),
                               self.msg("assistant", "改成这样：……")])

        sent = self.sent_user_text()
        self.assertIn("帮我把这句改简单一点", sent)
        self.assertIn("改成这样：……", sent)
        self.assertIn("最近上下文", sent, "没有标出哪一段是上下文")

    def test_context_comes_before_the_current_question(self):
        """顺序：上下文 → 资料 → 当前问题。当前问题必须紧贴着指令，不能被埋掉。"""
        self.ask_with_context([self.msg("user", "上一轮的话")], question="再给一个例子")

        sent = self.sent_user_text()
        self.assertLess(sent.index("上一轮的话"), sent.index("再给一个例子"))

    def test_only_one_model_call_even_with_context(self):
        """【核心】有上下文时仍然只调用模型一次 —— 不是「先理解指代再回答」两次。"""
        self.ask_with_context([self.msg("user", "上一轮的话")])
        self.assertEqual(len(self.fake.calls), 1)

    def test_no_context_still_works(self):
        """没有历史时，提示词里不该出现上下文那一段。"""
        self.ask_with_context(None)
        self.assertNotIn("最近上下文", self.sent_user_text())

    # ---------- 向后兼容 ----------

    def test_old_entry_point_sends_no_history(self):
        """【核心】老入口 generate_answer 保持原样：不带上下文。

        评测器走的就是它 —— 判分必须只针对「问题 + 片段」，
        掺进历史会让同一个问题在不同上下文里得出不同结果，评分不可复现。
        """
        rag.generate_answer("再给一个例子", [dict(c) for c in CHUNKS], self.fake, MODEL)

        sent = self.sent_user_text()
        self.assertNotIn("最近上下文", sent)

    def test_all_entry_points_never_add_a_second_model_call(self):
        """挨个调用三个入口，一共只有三次模型调用 —— 一个不多。

        （第四个入口 generate_answer_with_context_and_diagnostics 由本类
          下面那几条测试单独覆盖，也各自断言过「只调一次」。）
        """
        rag.generate_answer("问题一", [dict(c) for c in CHUNKS], self.fake, MODEL)
        rag.generate_answer_with_context(
            "问题二", [dict(c) for c in CHUNKS], [self.msg("user", "历史")], self.fake, MODEL)
        rag.generate_answer_with_diagnostics("问题三", [dict(c) for c in CHUNKS], self.fake, MODEL)

        self.assertEqual(len(self.fake.calls), 3)

    # ---------- 安全边界：上下文绝不能被当成资料 ----------

    def test_a_source_name_seen_in_history_cannot_be_cited(self):
        """【本轮最关键的一条】历史里出现过的来源名，不算数。

        构造：上一轮回答正文里提到了 grammar_present_perfect.md，
        本次检索【什么都没捞到】（白名单是空的），模型却引用了那个来源。
        正确行为：白名单拦下 → 安全降级 → 一条引用都不许留下。
        """
        history = [self.msg("assistant",
                            "上一轮我引用的是 grammar_present_perfect.md 的「基本结构」一节")]

        result = self.ask_with_context(
            history, chunks=[],                      # 本次白名单为空
            reply={"decision": "answer", "answer": "顺手引一个来源。",
                   "citations": [{"source": "grammar_present_perfect.md",
                                  "heading": "基本结构"}]})

        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(result["citations"], [], "历史里的来源被当成合法引用了")

    def test_context_never_leaks_into_citations_even_when_chunks_are_real(self):
        """本次有真片段时也一样：引历史里的标题 → 依然被拦。"""
        history = [self.msg("assistant", "资料来源：course_faq.md · 以前的标题")]

        result = self.ask_with_context(
            history,
            reply={"decision": "answer", "answer": "回答。",
                   "citations": [{"source": "course_faq.md", "heading": "以前的标题"}]})

        self.assertEqual(result["decision"], "insufficient_evidence",
                         "引了历史里的标题（不是本次片段）却没被拦下")
        self.assertEqual(result["citations"], [])

    def test_return_shape_is_still_exactly_three_keys(self):
        """带上下文之后，返回结构一个键都不许多。"""
        result = self.ask_with_context([self.msg("user", "历史")])
        self.assertEqual(set(result.keys()), {"decision", "answer", "citations"})

    def test_diagnostics_entry_point_is_unaffected(self):
        """评测器入口的返回值还是二元组，且不掺历史。"""
        result, code = rag.generate_answer_with_diagnostics(
            "问题", [dict(c) for c in CHUNKS], self.fake, MODEL)

        self.assertEqual(set(result.keys()), {"decision", "answer", "citations"})
        self.assertIn(code, rag.DIAGNOSTIC_CODES)

    # ---------- 带上下文的诊断入口（网页写日志用） ----------

    def test_context_diagnostics_entry_returns_a_pair(self):
        """回二元组，第一个元素仍然是那三个键。"""
        self.fake.reply = json.dumps(
            {"decision": "insufficient_evidence", "answer": "信息不够完整。", "citations": []},
            ensure_ascii=False)

        result, code = rag.generate_answer_with_context_and_diagnostics(
            "问题", [dict(c) for c in CHUNKS], [self.msg("user", "历史")], self.fake, MODEL)

        self.assertEqual(set(result.keys()), {"decision", "answer", "citations"})
        self.assertIn(code, rag.DIAGNOSTIC_CODES)
        self.assertEqual(code, rag.DIAG_OK)

    def test_context_diagnostics_entry_reuses_the_context_mechanism(self):
        """它复用同一个实现，所以上下文照样进提示词，而且只有一次模型调用。"""
        rag.generate_answer_with_context_and_diagnostics(
            "再给一个例子", [dict(c) for c in CHUNKS], [self.msg("user", "上一轮的话")],
            self.fake, MODEL)

        self.assertEqual(len(self.fake.calls), 1, "诊断入口不该多调一次模型")
        self.assertIn("上一轮的话", self.sent_user_text())

    def test_context_diagnostics_agrees_with_the_plain_context_entry(self):
        """同一份模型回复，两个入口的判断必须完全一致（它们共用同一个实现）。"""
        reply = {"decision": "insufficient_evidence", "answer": "信息不够。", "citations": []}
        plain = self.ask_with_context([self.msg("user", "历史")], reply=reply)

        result, code = rag.generate_answer_with_context_and_diagnostics(
            "再给一个例子", [dict(c) for c in CHUNKS], [self.msg("user", "历史")],
            self.fake, MODEL)

        self.assertEqual(result["decision"], plain["decision"])
        self.assertEqual(result["answer"], plain["answer"])
        self.assertEqual(code, rag.DIAG_OK)

    def test_context_diagnostics_reports_degradation_honestly(self):
        """【核心】被降级时 decision 和普通入口一样，但标签说出了真相。"""
        self.fake.reply = "这不是 JSON"

        result, code = rag.generate_answer_with_context_and_diagnostics(
            "问题", [dict(c) for c in CHUNKS], [self.msg("user", "历史")], self.fake, MODEL)

        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(code, rag.DIAG_INVALID_JSON,
                         "降级了却报成 ok —— 那日志就又看不出根因了")


# ===================== 6. 安全降级 =====================

class TestSafeDegradation(RagTestCase):

    def test_bad_json_degrades_safely(self):
        """模型返回一段根本不是 JSON 的文字 → 安全降级，不崩。"""
        r = self.ask(reply="我觉得这个问题应该这样回答：since 接时间点。")
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_empty_reply_degrades_safely(self):
        """模型返回空字符串 → 安全降级。"""
        r = self.ask(reply="")
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_none_reply_degrades_safely(self):
        """模型返回 None（比如接口抽风）→ 安全降级。"""
        self.fake.reply = None          # 直接设成 None，模拟接口返回空
        r = self.ask()
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_json_array_instead_of_object_degrades_safely(self):
        """返回的是 JSON，但是个数组而不是对象 → 安全降级。"""
        r = self.ask(reply="[1, 2, 3]")
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_model_exception_degrades_safely(self):
        """模型直接抛异常 → 安全降级，不崩。"""
        def boom(model, messages, **kwargs):
            raise RuntimeError("模拟网络故障")
        self.fake.chat.completions.create = boom

        r = self.ask()
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_model_exception_does_not_leak_error_text(self):
        """【核心】模型的原始错误信息绝不能出现在给用户看的内容里。"""
        secret = "内部细节-不该外露-9f3a"

        def boom(model, messages, **kwargs):
            raise RuntimeError(secret)
        self.fake.chat.completions.create = boom

        r = self.ask()
        self.assertNotIn(secret, r["answer"], "原始异常信息泄露到 answer 里了")
        self.assertNotIn(secret, json.dumps(r, ensure_ascii=False))


# ===================== 7. Markdown 代码块包裹 =====================

class TestMarkdownFence(RagTestCase):

    def test_fenced_json_is_parsed(self):
        """模型用 ```json 把 JSON 包起来时，能正确剥壳解析。（行为确定：解析）"""
        inner = json.dumps(self.answer_json("包在代码块里的回答"), ensure_ascii=False)
        r = self.ask(reply="```json\n" + inner + "\n```")
        self.assertEqual(r["decision"], "answer")
        self.assertIn("包在代码块里的回答", r["answer"])

    def test_fenced_json_without_language_tag_is_parsed(self):
        """不带 json 标记的 ``` 也一样处理。"""
        inner = json.dumps(self.answer_json("没有语言标记"), ensure_ascii=False)
        r = self.ask(reply="```\n" + inner + "\n```")
        self.assertEqual(r["decision"], "answer")

    def test_fenced_garbage_degrades_safely(self):
        """代码块里装的不是合法 JSON → 安全降级。（行为确定：拒绝）"""
        r = self.ask(reply="```json\n这不是 JSON\n```")
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_unterminated_fence_degrades_safely(self):
        """只有开头的 ``` 没有结尾 → 安全降级，不崩。"""
        r = self.ask(reply="```json\n{\"decision\": \"answer\"")
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_fence_handling_is_deterministic(self):
        """同一份输入跑两次，结果必须完全一样。"""
        inner = json.dumps(self.answer_json("确定性检查"), ensure_ascii=False)
        reply = "```json\n" + inner + "\n```"
        first = self.ask(reply=reply)
        second = self.ask(reply=reply)
        self.assertEqual(first, second)


# ===================== 8. 返回格式的结构性保证 =====================

class TestReturnShape(RagTestCase):

    def _varied_results(self):
        """把各种输入都跑一遍，收集结果，用来做整体性的结构检查。"""
        cases = [
            ([], None),
            (CHUNKS, self.answer_json()),
            (CHUNKS, {"decision": "refuse", "answer": "x", "citations": []}),
            (CHUNKS, {"decision": "insufficient_evidence", "answer": "x", "citations": []}),
            (CHUNKS, {"decision": "answer", "answer": "x",
                      "citations": [{"source": "编的.md", "heading": "编的"}]}),
            (CHUNKS, "不是 JSON"),
            (CHUNKS, {"decision": "answer", "answer": "x", "citations": []}),
        ]
        out = []
        for chunks, reply in cases:
            out.append(self.ask(chunks=chunks, reply=reply))
        return out

    def test_shape_is_always_the_same(self):
        """不管什么情况，返回的永远是那三个键，一个不多一个不少。"""
        for r in self._varied_results():
            self.assertEqual(set(r.keys()), {"decision", "answer", "citations"},
                             "返回的键不对：" + str(sorted(r.keys())))

    def test_decision_is_always_one_of_the_three(self):
        """decision 永远只能是那三个值之一。"""
        for r in self._varied_results():
            self.assertIn(r["decision"], rag.VALID_DECISIONS)

    def test_answer_is_always_a_non_empty_string(self):
        """answer 永远是非空字符串（给用户看的东西不能是空的）。"""
        for r in self._varied_results():
            self.assertIsInstance(r["answer"], str)
            self.assertTrue(r["answer"].strip())

    def test_refuse_and_insufficient_never_carry_citations(self):
        """【不变量】refuse / insufficient_evidence 的 citations 永远是空的。"""
        for r in self._varied_results():
            if r["decision"] != "answer":
                self.assertEqual(r["citations"], [],
                                 r["decision"] + " 居然带了引用")

    def test_answer_always_carries_at_least_one_citation(self):
        """【不变量】反过来：只要是 answer，就一定有引用。"""
        for r in self._varied_results():
            if r["decision"] == "answer":
                self.assertGreaterEqual(len(r["citations"]), 1)

    def test_citations_always_reference_passed_chunks(self):
        """【不变量】所有引用都必须落在本次传入的片段里。"""
        allowed = {(c["source"], c["heading"]) for c in CHUNKS}
        for r in self._varied_results():
            for c in r["citations"]:
                self.assertIn((c["source"], c["heading"]), allowed)
                self.assertEqual(set(c.keys()), {"source", "heading"})

    def test_results_are_json_serializable(self):
        """结果必须能直接 json.dumps——以后要存库、要传给网页用。"""
        for r in self._varied_results():
            json.dumps(r, ensure_ascii=False)

    def test_rag_module_is_pure_local(self):
        """【结构性保证】rag.py 不能引入网络或模型库——客户端一律从外面传进来。"""
        banned = ["requests", "urllib", "httpx", "aiohttp", "socket", "openai", "flask"]
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()

        for line in source.splitlines():
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                for bad in banned:
                    self.assertNotIn(bad, line, "rag.py 引入了不该引入的库：" + line)

    def test_rag_module_does_not_touch_api_keys(self):
        """【结构性保证】rag.py 不读取、不出现任何密钥相关代码。"""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()

        for token in ["API_KEY", "api_key", "DEEPSEEK_API_KEY", "environ"]:
            self.assertNotIn(token, source, "rag.py 里出现了密钥相关代码：" + token)


# ===================== 9. 诊断标签（只给后厨看的「退菜原因单」）=====================
#
# 【为什么单独测这一组】
# 一次降级如果只说「insufficient_evidence」，看不出是哪一步拦下的——
# 是没检索到资料？API 挂了？返回的不是 JSON？还是模型编了个来源？
# 修 API 和改提示词完全是两件事，所以必须能区分。
#
# 但诊断标签【只给后厨看】：它绝不能混进用户看到的回答里，
# 也绝不能带上异常原文、密钥、提示词或模型的完整原始回答。

class TestDiagnostics(RagTestCase):

    def run_diag(self, chunks=None, reply=None, question="这是问题"):
        """跑一次带诊断的入口，返回 (结果, 诊断标签)。"""
        if chunks is None:
            chunks = [dict(c) for c in CHUNKS]
        if reply is not None:
            if isinstance(reply, str):
                self.fake.reply = reply
            else:
                self.fake.reply = json.dumps(reply, ensure_ascii=False)
        return rag.generate_answer_with_diagnostics(question, chunks, self.fake, MODEL)

    # ---------- 每一种降级原因都要能被区分出来 ----------

    def test_ok_when_answer_passes_all_checks(self):
        r, code = self.run_diag(reply=self.answer_json())
        self.assertEqual(code, rag.DIAG_OK)
        self.assertEqual(r["decision"], "answer")

    def test_ok_when_clean_refuse(self):
        r, code = self.run_diag(reply={"decision": "refuse", "answer": "没有", "citations": []})
        self.assertEqual(code, rag.DIAG_OK)
        self.assertEqual(r["decision"], "refuse")

    def test_empty_chunks_now_uses_the_model(self):
        """【本轮改动】chunks 为空不再是一条「不调模型」的捷径 —— 照样走模型。"""
        self.fake.reply = json.dumps(
            {"decision": "general_answer", "answer": "通用回答", "citations": []},
            ensure_ascii=False)
        r, code = self.run_diag(chunks=[])
        self.assertEqual(code, rag.DIAG_OK)
        self.assertEqual(r["decision"], "general_answer")
        self.assertEqual(len(self.fake.calls), 1)

    def test_clean_general_answer_is_ok(self):
        """干净的 general_answer → 标签是 ok。"""
        r, code = self.run_diag(reply={"decision": "general_answer",
                                       "answer": "通用回答", "citations": []})
        self.assertEqual(code, rag.DIAG_OK)
        self.assertEqual(r["decision"], "general_answer")

    def test_general_answer_with_citations_is_flagged(self):
        """【核心】general_answer 却带了引用 → 有专门的诊断标签。"""
        r, code = self.run_diag(reply={
            "decision": "general_answer", "answer": "x",
            "citations": [{"source": CHUNK_SINCE_FOR["source"],
                           "heading": CHUNK_SINCE_FOR["heading"]}]})
        self.assertEqual(code, rag.DIAG_CITATIONS_ON_GENERAL_ANSWER)
        self.assertEqual(r["decision"], "insufficient_evidence")
        self.assertEqual(r["citations"], [])

    def test_api_or_response_error(self):
        """模型直接抛异常 → api_or_response_error（不是别的标签）。"""
        def boom(model, messages, **kwargs):
            raise RuntimeError("模拟网络故障")
        self.fake.chat.completions.create = boom

        r, code = self.run_diag()
        self.assertEqual(code, rag.DIAG_API_OR_RESPONSE_ERROR)
        self.assertEqual(r["decision"], "insufficient_evidence")

    def test_response_shape_error_is_also_api_or_response_error(self):
        """响应对象结构不对（取不到 choices）→ 同样算 api_or_response_error。"""
        def bad(model, messages, **kwargs):
            return object()          # 没有 .choices 属性
        self.fake.chat.completions.create = bad

        r, code = self.run_diag()
        self.assertEqual(code, rag.DIAG_API_OR_RESPONSE_ERROR)

    def test_invalid_json(self):
        """返回一段根本不是 JSON 的文字。"""
        r, code = self.run_diag(reply="我觉得应该这样回答：since 接时间点。")
        self.assertEqual(code, rag.DIAG_INVALID_JSON)

    def test_response_not_object(self):
        """JSON 合法，但是个数组而不是对象。"""
        r, code = self.run_diag(reply="[1, 2, 3]")
        self.assertEqual(code, rag.DIAG_RESPONSE_NOT_OBJECT)

    def test_invalid_decision(self):
        """decision 不在四个合法值里。"""
        r, code = self.run_diag(reply={"decision": "maybe", "answer": "x", "citations": []})
        self.assertEqual(code, rag.DIAG_INVALID_DECISION)

    def test_invalid_citations_fabricated_source(self):
        """【核心】模型编了一个不存在的文件名 —— 要能和别的降级区分开。"""
        r, code = self.run_diag(reply=self.answer_json(citations=[
            {"source": "根本没有这个文件.md", "heading": CHUNK_SINCE_FOR["heading"]},
        ]))
        self.assertEqual(code, rag.DIAG_INVALID_CITATIONS)

    def test_invalid_citations_fabricated_heading(self):
        """文件名是真的，但标题是编的 —— 同样算 invalid_citations。"""
        r, code = self.run_diag(reply=self.answer_json(citations=[
            {"source": CHUNK_SINCE_FOR["source"], "heading": "编造的标题"},
        ]))
        self.assertEqual(code, rag.DIAG_INVALID_CITATIONS)

    def test_empty_answer(self):
        """说好要回答，正文却是空的。"""
        r, code = self.run_diag(reply=self.answer_json(text="   "))
        self.assertEqual(code, rag.DIAG_EMPTY_ANSWER)

    def test_missing_citations(self):
        """说好要回答，却一条引用都不给。"""
        r, code = self.run_diag(reply={"decision": "answer", "answer": "有内容但没出处",
                                       "citations": []})
        self.assertEqual(code, rag.DIAG_MISSING_CITATIONS)

    def test_citations_on_non_answer(self):
        """拒答却带着引用 —— 自相矛盾，单独一个标签。"""
        r, code = self.run_diag(reply={
            "decision": "refuse", "answer": "没有",
            "citations": [{"source": CHUNK_SINCE_FOR["source"],
                           "heading": CHUNK_SINCE_FOR["heading"]}]})
        self.assertEqual(code, rag.DIAG_CITATIONS_ON_NON_ANSWER)

    # ---------- 标签本身的安全性与完整性 ----------

    def test_every_scenario_produces_a_known_code(self):
        """所有能想到的输入，产出的标签都必须在固定枚举里。"""
        scenarios = [
            ([], None),
            (CHUNKS, self.answer_json()),
            (CHUNKS, {"decision": "refuse", "answer": "x", "citations": []}),
            (CHUNKS, {"decision": "insufficient_evidence", "answer": "x", "citations": []}),
            (CHUNKS, "不是 JSON"),
            (CHUNKS, "[1,2,3]"),
            (CHUNKS, {"decision": "??", "answer": "x", "citations": []}),
            (CHUNKS, {"decision": "answer", "answer": "x", "citations": []}),
            (CHUNKS, {"decision": "answer", "answer": "  ", "citations": [
                {"source": CHUNK_SINCE_FOR["source"], "heading": CHUNK_SINCE_FOR["heading"]}]}),
            (CHUNKS, {"decision": "answer", "answer": "x", "citations": [
                {"source": "假的.md", "heading": "假的"}]}),
        ]
        for chunks, reply in scenarios:
            _r, code = self.run_diag(chunks=chunks, reply=reply)
            self.assertIn(code, rag.DIAGNOSTIC_CODES,
                          "产出了枚举之外的标签：" + repr(code))

    def test_diagnostic_is_never_inside_the_result(self):
        """【核心】诊断标签绝不能混进结果字典里。"""
        for chunks, reply in [([], None), (CHUNKS, "不是 JSON"), (CHUNKS, self.answer_json())]:
            r, code = self.run_diag(chunks=chunks, reply=reply)
            self.assertNotIn("diagnostic_code", r)
            self.assertNotIn("diagnostic", r)
            self.assertNotIn(code, json.dumps(r, ensure_ascii=False))

    def test_diagnostic_carries_no_content_from_the_reply(self):
        """【核心】标签里不能带上模型原始输出的任何片段。"""
        marker = "ZZTOP-SECRET-MARKER-42"
        _r, code = self.run_diag(reply=marker + " 这不是 JSON，只是随便一段话")
        self.assertNotIn(marker, code)
        self.assertEqual(code, rag.DIAG_INVALID_JSON)

    def test_diagnostic_carries_no_exception_text(self):
        """【核心】标签里不能带上异常原文。"""
        marker = "内部细节-不该外露-8c1f"

        def boom(model, messages, **kwargs):
            raise RuntimeError(marker)
        self.fake.chat.completions.create = boom

        _r, code = self.run_diag()
        self.assertNotIn(marker, code)
        self.assertEqual(code, rag.DIAG_API_OR_RESPONSE_ERROR)

    def test_codes_are_short_fixed_tokens(self):
        """标签必须都是短的、写死的小写标识，不是句子。"""
        for code in rag.DIAGNOSTIC_CODES:
            self.assertIsInstance(code, str)
            self.assertLessEqual(len(code), 32, code)
            self.assertTrue(code.replace("_", "").isalnum(), code)
            self.assertEqual(code, code.lower(), code)


# ===================== 10. generate_answer 的返回格式恒定不变 =====================

class TestPublicShapeIsUnchanged(RagTestCase):

    def _results(self):
        """把各种情况都跑一遍，收集 generate_answer 的返回值。"""
        scenarios = [
            ([], None),
            (CHUNKS, self.answer_json()),
            (CHUNKS, {"decision": "refuse", "answer": "x", "citations": []}),
            (CHUNKS, {"decision": "insufficient_evidence", "answer": "x", "citations": []}),
            (CHUNKS, "不是 JSON"),
            (CHUNKS, "[1,2,3]"),
            (CHUNKS, {"decision": "??", "answer": "x", "citations": []}),
            (CHUNKS, {"decision": "answer", "answer": "x", "citations": []}),
        ]
        out = []
        for chunks, reply in scenarios:
            out.append(self.ask(chunks=chunks, reply=reply))
        return out

    def test_generate_answer_returns_exactly_three_keys(self):
        """【核心】对外返回恒定只有 decision / answer / citations 三个键。"""
        for r in self._results():
            self.assertEqual(set(r.keys()), {"decision", "answer", "citations"},
                             "返回的键不对：" + str(sorted(r.keys())))

    def test_generate_answer_is_json_serializable(self):
        """返回值必须能直接序列化——网页和结果文件都要用它。"""
        for r in self._results():
            json.dumps(r, ensure_ascii=False)

    def test_generate_answer_has_no_diagnostic_leak(self):
        """对外返回值里不能有任何诊断相关的痕迹。"""
        for r in self._results():
            blob = json.dumps(r, ensure_ascii=False)
            for code in rag.DIAGNOSTIC_CODES:
                self.assertNotIn(code, blob, "诊断标签泄漏进了对外结果：" + code)

    def test_diagnostics_entry_point_returns_a_pair(self):
        """带诊断的入口返回的是二元组 (结果, 标签)。"""
        out = rag.generate_answer_with_diagnostics(
            "问题", [dict(c) for c in CHUNKS], self.fake, MODEL)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        result, code = out
        self.assertEqual(set(result.keys()), {"decision", "answer", "citations"})
        self.assertIn(code, rag.DIAGNOSTIC_CODES)

    def test_both_entry_points_agree(self):
        """同一份输入，两个入口给出的结果字典必须完全一致。"""
        self.fake.reply = self.answer_json()
        plain = rag.generate_answer("问题", [dict(c) for c in CHUNKS], self.fake, MODEL)

        self.fake.reply = self.answer_json()
        with_diag, _code = rag.generate_answer_with_diagnostics(
            "问题", [dict(c) for c in CHUNKS], self.fake, MODEL)

        self.assertEqual(plain, with_diag)


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_rag.py 也能跑
