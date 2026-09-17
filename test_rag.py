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

class TestNoChunks(RagTestCase):

    def test_empty_chunks_returns_refuse(self):
        """chunks 为空 → 拒答。"""
        r = self.ask(chunks=[])
        self.assertEqual(r["decision"], "refuse")

    def test_empty_chunks_does_not_call_model(self):
        """【核心】chunks 为空时，绝不能调用模型——没有资料就没什么可问的。"""
        self.ask(chunks=[])
        self.assertEqual(len(self.fake.calls), 0, "没有资料却调用了模型")

    def test_none_chunks_returns_refuse(self):
        """chunks 传 None 也不能崩，同样拒答。"""
        r = rag.generate_answer("随便问", None, self.fake, MODEL)
        self.assertEqual(r["decision"], "refuse")
        self.assertEqual(len(self.fake.calls), 0)

    def test_empty_chunks_result_has_no_citations(self):
        """拒答结果里不能有引用。"""
        r = self.ask(chunks=[])
        self.assertEqual(r["citations"], [])


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
        """decision 是三个合法值之外的任意东西 → 拒绝。"""
        for bad in ["maybe", "ANSWER", "", None, 1, "answer "]:
            r = self.ask(reply={"decision": bad, "answer": "x", "citations": []})
            self.assertEqual(r["decision"], "insufficient_evidence",
                             "decision=" + repr(bad) + " 居然被放过了")


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


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_rag.py 也能跑
