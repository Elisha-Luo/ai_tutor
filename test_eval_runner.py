# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# 评测器（evals/run_rag_eval.py）的自动化测试
#
# 【原则】这些测试完全离线运行：不联网、不调用真实 DeepSeek、不需要密钥。
# 做法有两个：
#   · 模型客户端换成假的（跟 test_rag.py 里那套一样）
#   · 检索函数也换成假的，这样每一道题「会捞到哪些资料」由测试自己说了算
#
# 运行方式（在 ai_tutor 文件夹里）：
#     python -m unittest test_eval_runner -v
# =====================================================================

import os          # 拼路径
import sys         # 把项目根目录加进模块搜索路径
import io          # 用来「接住」被测试的打印输出，别把测试日志刷屏
import json        # 造假的模型回复、校验结果文件
import inspect     # 检查 run_dry 的函数签名（这是「dry-run 不可能调模型」的结构性证据）
import tempfile    # 存结果文件的测试要用临时目录，别污染真项目
import contextlib  # 临时把标准输出/报错重定向走
import subprocess  # 起一个真进程来复现 GBK 控制台问题（进程内测不出来）
import unittest    # Python 自带的测试框架

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from evals import run_rag_eval as runner      # 被测对象


# ===================== 假的模型客户端 =====================

class _Msg:
    def __init__(self, content):
        self.content = content

class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)

class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]

class _Completions:
    def __init__(self, owner):
        self.owner = owner
    def create(self, model, messages, **kwargs):
        self.owner.calls.append({"model": model, "messages": messages})
        if self.owner.raise_error:
            raise self.owner.raise_error
        reply = self.owner.replies.pop(0) if self.owner.replies else "{}"
        return _Resp(reply)

class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)

class FakeClient:
    """按顺序吐出预设好的回复。用完了就返回 "{}"（会被 rag 判为格式不合格）。"""
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []
        self.raise_error = None
        self.chat = _Chat(self)


# ===================== 测试用的题库与检索 =====================
# 故意做小：4 道题覆盖四种分类，每条断言都能一眼看懂为什么。

CASES = [
    {"id": "a1", "question": "q-answer", "category": "answer",
     "expected_behavior": "answer", "expected_sources": ["grammar_present_perfect.md"]},
    {"id": "x1", "question": "q-cross", "category": "cross_source",
     "expected_behavior": "answer",
     "expected_sources": ["course_faq.md", "vocabulary_study_method.md"]},
    {"id": "r1", "question": "q-refuse", "category": "refuse",
     "expected_behavior": "refuse", "expected_sources": []},
    {"id": "t1", "question": "q-trap", "category": "trap_refuse",
     "expected_behavior": "refuse", "expected_sources": []},
]

# 每道题「检索会捞到什么」。
# 【注意两条拒答题也捞到了东西】这才是真实的陷阱：检索必然命中，但资料答不了。
RETRIEVAL = {
    "q-answer": ["grammar_present_perfect.md"],
    "q-cross": ["course_faq.md", "vocabulary_study_method.md"],
    "q-refuse": ["course_faq.md"],
    "q-trap": ["grammar_present_perfect.md"],
}


def fake_retrieve(question, top_k=3):
    """假检索：按上面的表返回片段，heading 统一是 "H-<文件名>"。"""
    return [{"source": s, "heading": "H-" + s, "text": "正文-" + s}
            for s in RETRIEVAL.get(question, [])]


def retrieve_with_distractor(question, top_k=3):
    """检索「多捞了一份」的情况——专门用来测「引对了检索结果、却引错了期望来源」。

    为什么需要它：
      rag.py 的引用白名单对照的是「本次检索到了什么」，
      而评测层对照的是「点菜单上期望引用什么」。这是两件不同的事。
      只有让检索多捞一份、模型再引错那一份，才能触发评测层的
      「answer 且来源不匹配」——也就是 rag.py 白名单抓不到的那种错。
    """
    if question == "q-answer":
        return [{"source": "grammar_present_perfect.md",
                 "heading": "H-grammar_present_perfect.md", "text": "正文"},
                {"source": "course_faq.md",
                 "heading": "H-course_faq.md", "text": "正文"}]     # 多捞的干扰项
    return fake_retrieve(question, top_k)


def h(source):
    """拼出这个来源对应的 heading，省得每次手写。"""
    return "H-" + source


def reply(decision, answer="这是回答。", citations=None):
    """造一个 rag.py 会接受的模型回复。"""
    return json.dumps({"decision": decision, "answer": answer,
                       "citations": citations or []}, ensure_ascii=False)


def cite(*sources):
    """按来源名拼出 citations 数组。"""
    return [{"source": s, "heading": h(s)} for s in sources]


# ===================== 1. 该答题的精确来源评分 =====================

class TestScoreAnswer(unittest.TestCase):

    def case(self):
        return CASES[0]          # a1：期望 grammar_present_perfect.md

    def result(self, decision, citations):
        return {"decision": decision, "answer": "x", "citations": citations}

    def test_exact_source_passes_strict_and_safe(self):
        """引用来源与期望完全一致 → strict 和 safe 都过。"""
        s = runner.score_case(self.case(), self.result("answer", cite("grammar_present_perfect.md")))
        self.assertTrue(s["strict_pass"])
        self.assertTrue(s["safe_pass"])
        self.assertIsNone(s["failure_reason"])

    def test_wrong_source_fails_and_is_unsafe(self):
        """【核心】引用了别的资料 → 不算过，而且是不安全的（它答错了依据）。"""
        s = runner.score_case(self.case(), self.result("answer", cite("course_faq.md")))
        self.assertFalse(s["strict_pass"])
        self.assertFalse(s["safe_pass"])
        self.assertIsNotNone(s["failure_reason"])

    def test_extra_source_fails(self):
        """多引了一份 → 也不算「完全一致」，判不过。"""
        s = runner.score_case(self.case(), self.result("answer", cite(
            "grammar_present_perfect.md", "course_faq.md")))
        self.assertFalse(s["strict_pass"])

    def test_not_answered_is_safe_but_not_strict(self):
        """该答却没答（模型说拒答）→ 不危险，但不通过。"""
        s = runner.score_case(self.case(), self.result("refuse", []))
        self.assertFalse(s["strict_pass"])
        self.assertTrue(s["safe_pass"])

    def test_passing_answer_still_flagged_for_human_review(self):
        """【核心】就算严格通过，也必须标记「回答内容需人工复核」。"""
        s = runner.score_case(self.case(), self.result("answer", cite("grammar_present_perfect.md")))
        self.assertTrue(s["strict_pass"])
        self.assertIsNotNone(s["needs_human_review"])
        self.assertIn("人工复核", s["needs_human_review"])


# ===================== 2. 跨来源题必须同时引用两份 =====================

class TestScoreCrossSource(unittest.TestCase):

    def case(self):
        return CASES[1]          # x1：期望 course_faq + vocabulary

    def result(self, sources):
        return {"decision": "answer", "answer": "x", "citations": cite(*sources)}

    def test_citing_both_passes(self):
        """两份都引 → 通过。"""
        s = runner.score_case(self.case(), self.result(
            ["course_faq.md", "vocabulary_study_method.md"]))
        self.assertTrue(s["strict_pass"])

    def test_citing_both_in_reverse_order_also_passes(self):
        """顺序反过来不影响——比的是集合，不是顺序。"""
        s = runner.score_case(self.case(), self.result(
            ["vocabulary_study_method.md", "course_faq.md"]))
        self.assertTrue(s["strict_pass"])

    def test_citing_only_one_fails(self):
        """【核心】只引一份 → 不过。这正是「该引两份只引一份」的典型错误。"""
        s = runner.score_case(self.case(), self.result(["course_faq.md"]))
        self.assertFalse(s["strict_pass"])
        self.assertFalse(s["safe_pass"])

    def test_citing_one_extra_fails(self):
        """两份都引了、但另外多引一份 → 仍然判不过（要求完全一致）。"""
        s = runner.score_case(self.case(), self.result(
            ["course_faq.md", "vocabulary_study_method.md", "grammar_present_perfect.md"]))
        self.assertFalse(s["strict_pass"])


# ===================== 3. 严格拒答 与 安全拒答 的区别 =====================

class TestScoreRefuse(unittest.TestCase):

    def case(self):
        return CASES[2]          # r1：该拒答

    def result(self, decision, citations=None):
        return {"decision": decision, "answer": "x", "citations": citations or []}

    def test_explicit_refuse_passes_both(self):
        """明确拒答 → strict 和 safe 都过。"""
        s = runner.score_case(self.case(), self.result("refuse"))
        self.assertTrue(s["strict_pass"])
        self.assertTrue(s["safe_pass"])
        self.assertIsNone(s["failure_reason"])

    def test_insufficient_is_safe_but_not_strict(self):
        """【核心区别】说「证据不足」而不是「资料里没有」→ 安全，但不算严格通过。"""
        s = runner.score_case(self.case(), self.result("insufficient_evidence"))
        self.assertFalse(s["strict_pass"])
        self.assertTrue(s["safe_pass"])
        self.assertIsNotNone(s["failure_reason"])

    def test_answer_on_refuse_question_is_unsafe(self):
        """【最严重】该拒答却硬答 → 不安全，必须进失败清单。"""
        s = runner.score_case(self.case(), self.result("answer", cite("course_faq.md")))
        self.assertFalse(s["strict_pass"])
        self.assertFalse(s["safe_pass"])
        self.assertIn("强行作答", s["failure_reason"])

    def test_trap_question_uses_the_same_rule(self):
        """陷阱题和普通拒答题走同一套规则。"""
        s = runner.score_case(CASES[3], self.result("answer", cite("grammar_present_perfect.md")))
        self.assertFalse(s["safe_pass"])


# ===================== 4. dry-run 不调用模型 =====================

class TestDryRun(unittest.TestCase):

    def test_run_dry_has_no_client_parameter(self):
        """【结构性证据】run_dry 的签名里根本没有 client —— 它不可能调模型。"""
        params = list(inspect.signature(runner.run_dry).parameters)
        self.assertNotIn("client", params, "run_dry 居然接受 client 参数")
        self.assertNotIn("model", params, "run_dry 居然接受 model 参数")

    def test_dry_run_produces_no_model_decisions(self):
        """dry-run 的报告里，任何一题都不该出现模型给出的 decision。"""
        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)
        self.assertEqual(report["mode"], "dry-run")
        for item in report["retrieval"]["per_case"]:
            self.assertNotIn("decision", item)

    def test_dry_run_runs_retrieval_for_every_case(self):
        """每一道题都要真的跑一遍检索。"""
        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)
        self.assertEqual(len(report["retrieval"]["per_case"]), len(CASES))

    def test_dry_run_flags_empty_retrieval(self):
        """检索为空的题要被单独标出来（那些题会走拒答，不会调模型）。"""
        def empty_retrieve(question, top_k=3):
            return []
        report = runner.run_dry(CASES, retrieve_fn=empty_retrieve)
        self.assertEqual(sorted(report["retrieval"]["empty"]), ["a1", "r1", "t1", "x1"])

    def test_dry_run_record_shape_is_ok(self):
        """dry-run 也要验证「结果记录」的字段齐不齐、能不能序列化。"""
        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)
        self.assertTrue(report["record_shape_ok"], report["record_shape_problems"])
        self.assertEqual(report["record_shape_problems"], [])

    def test_dry_run_writes_no_files(self):
        """【核心】dry-run 不产生任何结果文件。"""
        original = runner.RESULTS_DIR
        with tempfile.TemporaryDirectory() as td:
            runner.RESULTS_DIR = os.path.join(td, "should_not_exist")
            try:
                runner.run_dry(CASES, retrieve_fn=fake_retrieve)
            finally:
                runner.RESULTS_DIR = original
            self.assertFalse(os.path.exists(os.path.join(td, "should_not_exist")))

    def test_dry_run_reports_case_problems(self):
        """题库本身有毛病时，dry-run 要报出来。"""
        bad = [{"id": "x", "question": "q", "category": "answer",
                "expected_behavior": "answer", "expected_sources": []}]
        report = runner.run_dry(bad, retrieve_fn=fake_retrieve)
        self.assertTrue(report["case_problems"])

    def test_dry_run_reports_environment_without_leaking_key(self):
        """环境信息只报告「密钥有没有」，绝不出现密钥内容。"""
        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)
        env = report["environment"]
        self.assertIn("has_api_key", env)
        self.assertIsInstance(env["has_api_key"], bool)
        # 整个报告里不该出现任何像密钥的东西
        blob = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("sk-", blob)


# ===================== 5. live 模式的执行与汇总 =====================

class TestLiveRun(unittest.TestCase):

    def good_client(self):
        """一个「全部答对」的假客户端。"""
        return FakeClient([
            reply("answer", citations=cite("grammar_present_perfect.md")),      # a1
            reply("answer", citations=cite("course_faq.md",
                                           "vocabulary_study_method.md")),      # x1
            reply("refuse", citations=[]),                                     # r1
            reply("refuse", citations=[]),                                     # t1
        ])

    def test_all_correct_run_summarizes_as_all_pass(self):
        """全部答对 → 严格通过 4、安全通过 4、失败 0。"""
        client = self.good_client()
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)

        self.assertEqual(report["mode"], "live")
        self.assertEqual(report["total"], 4)
        self.assertEqual(report["counts"]["strict_pass"], 4)
        self.assertEqual(report["counts"]["safe_pass"], 4)
        self.assertEqual(report["counts"]["failed"], 0)
        self.assertEqual(report["failures"], [])

    def test_model_is_called_once_per_case(self):
        """每题正好调用一次模型。"""
        client = self.good_client()
        runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)
        self.assertEqual(len(client.calls), len(CASES))

    def test_failed_cases_land_in_the_failure_list(self):
        """【核心】不安全的输出必须出现在失败清单里。"""
        client = FakeClient([
            reply("answer", citations=cite("course_faq.md")),   # a1 引了「检索到但期望之外」的来源 → 失败
            reply("answer", citations=cite("course_faq.md")),   # x1 只引一份 → 失败
            reply("answer", citations=cite("course_faq.md")),   # r1 该拒却硬答 → 失败
            reply("refuse", citations=[]),                      # t1 正常拒答
        ])
        # 这里必须用带干扰项的检索：a1 要多捞一份 course_faq.md，
        # 否则 rag.py 的引用白名单会先一步拦下它，测不到评测层这一关。
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=retrieve_with_distractor)

        self.assertEqual(report["counts"]["failed"], 3)
        failed_ids = {f["id"] for f in report["failures"]}
        self.assertEqual(failed_ids, {"a1", "x1", "r1"})
        for f in report["failures"]:
            self.assertTrue(f["reason"], "失败清单里必须写明原因")

    def test_by_category_counts_are_correct(self):
        """分类统计要能对上。"""
        client = self.good_client()
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)
        self.assertEqual(report["by_category"]["answer"]["total"], 1)
        self.assertEqual(report["by_category"]["cross_source"]["total"], 1)
        self.assertEqual(report["by_category"]["refuse"]["total"], 1)
        self.assertEqual(report["by_category"]["trap_refuse"]["total"], 1)
        self.assertEqual(report["by_category"]["refuse"]["strict_pass"], 1)

    def test_every_record_has_all_required_fields(self):
        """【结果文件结构】每条记录都要有全部必需字段。"""
        client = self.good_client()
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)
        for rec in report["cases"]:
            for key in runner.REQUIRED_RECORD_KEYS:
                self.assertIn(key, rec, str(rec.get("id")) + " 缺少字段 " + key)

    def test_records_capture_retrieval_and_elapsed(self):
        """记录里要能看到检索到了什么、以及耗时。"""
        client = self.good_client()
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)
        rec = report["cases"][0]
        self.assertEqual(rec["retrieved_sources"], ["grammar_present_perfect.md"])
        self.assertIsInstance(rec["elapsed_ms"], int)
        self.assertGreaterEqual(rec["elapsed_ms"], 0)

    def test_expected_fields_are_carried_through(self):
        """点菜单上的期望值要原样带进记录，方便日后复盘。"""
        client = self.good_client()
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)
        rec = report["cases"][1]
        self.assertEqual(rec["expected_behavior"], "answer")
        self.assertEqual(sorted(rec["expected_sources"]),
                         ["course_faq.md", "vocabulary_study_method.md"])
        self.assertEqual(rec["category"], "cross_source")

    def test_report_is_json_serializable(self):
        """整份报告必须能直接写进 JSON 文件。"""
        client = self.good_client()
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)
        json.dumps(report, ensure_ascii=False)

    def test_model_exception_does_not_leak_or_crash(self):
        """【核心】模型抛异常时：不崩，而且异常原文绝不进报告。"""
        secret = "内部细节-不该外露-7b2e"
        client = FakeClient()
        client.raise_error = RuntimeError(secret)

        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)

        blob = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(secret, blob, "原始异常信息泄露进报告了")
        self.assertEqual(report["total"], 4)          # 没有中途崩掉

    def test_limit_is_respected(self):
        """--limit 能只跑前几题。"""
        client = FakeClient([reply("answer", citations=cite("grammar_present_perfect.md"))])
        report = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=fake_retrieve)
        self.assertEqual(report["total"], 1)


# ===================== 6. 结果文件 =====================

class TestResultFile(unittest.TestCase):

    def build_report(self):
        client = FakeClient([reply("refuse"), reply("refuse")])
        return runner.run_live(CASES[2:], client, "fake-model", retrieve_fn=fake_retrieve)

    def test_save_results_writes_a_timestamped_json_file(self):
        """存出来的文件名带时间戳、带模式后缀，内容能读回来。"""
        report = self.build_report()
        original = runner.RESULTS_DIR
        with tempfile.TemporaryDirectory() as td:
            runner.RESULTS_DIR = td
            try:
                path = runner.save_results(report)
            finally:
                runner.RESULTS_DIR = original

            self.assertTrue(os.path.exists(path))
            name = os.path.basename(path)
            self.assertTrue(name.endswith("-live.json"), name)
            self.assertEqual(len(name.split("-")[0]), 8)       # 8 位日期
            self.assertEqual(len(name.split("-")[1]), 6)       # 6 位时间

            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["mode"], "live")
            self.assertEqual(loaded["total"], len(report["cases"]))
            self.assertIn("counts", loaded)
            self.assertIn("failures", loaded)
            self.assertIn("cases", loaded)

    def test_save_results_creates_the_directory(self):
        """结果目录不存在时会自动创建。"""
        report = self.build_report()
        original = runner.RESULTS_DIR
        with tempfile.TemporaryDirectory() as td:
            runner.RESULTS_DIR = os.path.join(td, "results")
            try:
                path = runner.save_results(report)
            finally:
                runner.RESULTS_DIR = original
            self.assertTrue(os.path.exists(path))


# ===================== 7. 题库校验 =====================

class TestValidateCases(unittest.TestCase):

    def test_the_real_case_bank_is_clean(self):
        """【真实题库】evals/rag_cases.json 必须干干净净：33 道题、无问题。"""
        cases = runner.load_cases()
        self.assertEqual(len(cases), 33)
        self.assertEqual(runner.validate_cases(cases), [])

    def test_duplicate_ids_are_detected(self):
        bad = [dict(CASES[0]), dict(CASES[0])]
        self.assertTrue(any("重复" in p for p in runner.validate_cases(bad)))

    def test_refuse_case_with_sources_is_detected(self):
        bad = [{"id": "r", "question": "q", "category": "refuse",
                "expected_behavior": "refuse", "expected_sources": ["a.md"]}]
        self.assertTrue(any("空数组" in p for p in runner.validate_cases(bad)))

    def test_answer_case_without_sources_is_detected(self):
        bad = [{"id": "a", "question": "q", "category": "answer",
                "expected_behavior": "answer", "expected_sources": []}]
        self.assertTrue(any("expected_sources" in p for p in runner.validate_cases(bad)))

    def test_missing_field_is_detected(self):
        bad = [{"id": "a", "question": "q"}]
        problems = runner.validate_cases(bad)
        self.assertTrue(any("缺少字段" in p for p in problems))

    def test_bad_expected_behavior_is_detected(self):
        bad = [{"id": "a", "question": "q", "category": "answer",
                "expected_behavior": "maybe", "expected_sources": ["a.md"]}]
        self.assertTrue(any("expected_behavior" in p for p in runner.validate_cases(bad)))

    def test_empty_bank_is_detected(self):
        self.assertTrue(runner.validate_cases([]))


# ===================== 8. 命令行参数 =====================

class TestCommandLine(unittest.TestCase):

    def test_dry_run_needs_no_api_key(self):
        """dry-run 在没有密钥的情况下也必须能跑完。"""
        original = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            # 把打印输出接住——dry-run 会打一整份报告，不接住会把测试日志刷爆
            with contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["--dry-run", "--limit", "2"])
        finally:
            if original is not None:
                os.environ["DEEPSEEK_API_KEY"] = original
        self.assertEqual(code, 0)

    def test_live_and_dry_run_are_mutually_exclusive(self):
        """--live 和 --dry-run 不能同时给。"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            with self.assertRaises(SystemExit):
                runner.main(["--live", "--dry-run"])

    def test_default_mode_is_dry_run(self):
        """不给任何模式参数时，默认走 dry-run —— 绝不默认花钱。"""
        original = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["--limit", "1"])
        finally:
            if original is not None:
                os.environ["DEEPSEEK_API_KEY"] = original
        self.assertEqual(code, 0, "不带参数时应该走 dry-run 并且不报错")


class TestSummaryLists(unittest.TestCase):

    def test_safe_misses_are_listed_separately_from_failures(self):
        """【核心】「安全但没达标」要和「不安全」分开列，不能混在一起。"""
        client = FakeClient([
            reply("answer", citations=cite("grammar_present_perfect.md")),      # a1 正常
            reply("answer", citations=cite("course_faq.md",
                                           "vocabulary_study_method.md")),      # x1 正常
            reply("insufficient_evidence", citations=[]),                       # r1 安全但没达标
            reply("refuse", citations=[]),                                     # t1 正常
        ])
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)

        self.assertEqual(report["counts"]["failed"], 0)                 # 没出事故
        self.assertEqual(report["counts"]["strict_pass"], 3)            # 但有一题没达标
        self.assertEqual(report["counts"]["safe_pass"], 4)
        self.assertEqual([m["id"] for m in report["safe_misses"]], ["r1"])
        self.assertEqual(report["failures"], [])


# ===================== 9. Windows GBK 控制台兼容性 =====================
#
# 【这一类测试和别的不一样】
# 前面的测试都是「在同一个进程里调用函数」。但 GBK 这个 bug 偏偏在进程内测不出来——
# 因为测试进程自己的标准输出编码是 UTF-8，✓ 打印得好好的。
# 真实故障发生在【另一个进程】里：那个进程的 stdout 被设成了 GBK，
# 而 GBK 字符集里没有 ✓ ✗ ⚠。
#
# 所以这里必须用 subprocess 起一个真进程，并强制 PYTHONIOENCODING=gbk，
# 才能复现、也才能真正守住这个坑。

class TestWindowsConsoleCompatibility(unittest.TestCase):

    def run_dry_in_gbk_console(self, extra_args=()):
        """在一个 PYTHONIOENCODING=gbk 的子进程里跑 dry-run，返回 (退出码, 输出文字)。"""
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "gbk"        # 强制子进程的 stdout 用 GBK
        env.pop("DEEPSEEK_API_KEY", None)      # dry-run 不需要密钥，去掉它更能验证「不依赖密钥」

        result = subprocess.run(
            [sys.executable, "-m", "evals.run_rag_eval", "--dry-run"] + list(extra_args),
            cwd=BASE, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120,
        )

        # 子进程吐出来的是 GBK 字节，用 gbk 解回来——解不开就说明它吐了 GBK 装不下的东西
        out = result.stdout.decode("gbk", errors="replace")
        err = result.stderr.decode("gbk", errors="replace")
        return result.returncode, out + err

    def test_dry_run_survives_a_gbk_console(self):
        """【本轮核心回归】默认 GBK 控制台下，--dry-run --limit 1 必须退出码 0。"""
        code, text = self.run_dry_in_gbk_console(["--limit", "1"])
        self.assertEqual(code, 0, "GBK 控制台下 dry-run 失败了：\n" + text)

    def test_full_dry_run_survives_a_gbk_console(self):
        """不加 --limit 跑全部 33 题，同样不能在 GBK 控制台下崩掉。"""
        code, text = self.run_dry_in_gbk_console()
        self.assertEqual(code, 0, "GBK 控制台下完整 dry-run 失败了：\n" + text)

    def test_no_encoding_error_in_output(self):
        """输出里绝不能出现编码错误的痕迹。"""
        code, text = self.run_dry_in_gbk_console(["--limit", "1"])
        self.assertNotIn("UnicodeEncodeError", text)
        self.assertNotIn("Traceback", text)
        self.assertNotIn("codec can't encode", text)

    def test_dry_run_completes_the_whole_report(self):
        """要跑到最后一行，而不是中途崩掉——所以结尾那句总结必须出现。"""
        code, text = self.run_dry_in_gbk_console(["--limit", "1"])
        self.assertIn("dry-run 结束", text)
        self.assertIn("没有调用任何模型", text)

    def test_output_uses_ascii_markers_instead_of_symbols(self):
        """输出里应该用 [OK] / [FAIL] / [WARN] 这类纯 ASCII 标记。"""
        code, text = self.run_dry_in_gbk_console(["--limit", "1"])
        self.assertIn("[OK]", text)
        self.assertNotIn("✓", text)
        self.assertNotIn("✗", text)

    def test_gbk_run_does_not_create_results_dir(self):
        """【核心】GBK 下跑 dry-run 也不许创建 evals/results。"""
        results_dir = runner.RESULTS_DIR
        existed_before = os.path.isdir(results_dir)

        code, text = self.run_dry_in_gbk_console(["--limit", "1"])
        self.assertEqual(code, 0)

        if not existed_before:
            self.assertFalse(os.path.isdir(results_dir),
                             "dry-run 不该创建 results/ 目录")

    def test_gbk_run_calls_no_model(self):
        """【核心】dry-run 不调用模型：整场跑完，且从不创建结果文件。

        注意这条测试【不依赖密钥在不在】——评测器现在会读项目根目录的 .env，
        所以子进程里密钥可能是「已设置」也可能是「未设置」。
        两种情况下 dry-run 都必须能跑完，这正是它不调模型的体现。
        """
        code, text = self.run_dry_in_gbk_console(["--limit", "1"])
        self.assertEqual(code, 0)
        # 跑到最后一行，说明整场流程走完了，而不是中途崩掉
        self.assertIn("dry-run 结束", text)
        self.assertIn("没有调用任何模型", text)
        # 【安全】输出里绝不能回显密钥内容
        self.assertNotIn("sk-", text)

    def test_safety_net_is_in_place(self):
        """harden_console 必须存在，而且能在当前进程里安全调用（幂等、不抛异常）。"""
        self.assertTrue(callable(runner.harden_console))
        runner.harden_console()          # 调两次也不该出问题
        runner.harden_console()


# ===================== 10. .env 配置加载（网页与评测器共用一份）=====================
#
# 【这一组在防什么】
# 以前只有 app.py 会读 .env，评测器不读。于是密钥只写在 .env 里的人：
#   网页 → 能正常用
#   评测器 → 报告「密钥未设置」，--live 直接被拦下
# 同一份配置、两个入口两种结论。这一组测试就是钉死「两边读的是同一份」。

class TestEnvFileIntegration(unittest.TestCase):

    SECRET = "sk-inttest-SECRET-VALUE-abcdef987654"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        # 保护真实的环境变量：先摘掉，测完还原，免得污染别的测试
        self.saved_key = os.environ.get("DEEPSEEK_API_KEY")
        os.environ.pop("DEEPSEEK_API_KEY", None)
        self.addCleanup(self._restore_key)

        # 造一个只含密钥的临时 .env
        self.env_path = os.path.join(self.tmp.name, ".env")
        with open(self.env_path, "w", encoding="utf-8") as f:
            f.write("DEEPSEEK_API_KEY=" + self.SECRET + "\n")

        # 让评测器去读这个临时 .env，而不是项目里真的那个
        self.saved_env_path = runner.ENV_PATH
        runner.ENV_PATH = self.env_path
        self.addCleanup(self._restore_env_path)

    def _restore_key(self):
        if self.saved_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = self.saved_key

    def _restore_env_path(self):
        runner.ENV_PATH = self.saved_env_path

    def test_dry_run_sees_key_that_lives_only_in_env_file(self):
        """【本轮核心】密钥只在 .env 里、进程环境变量里没有时，dry-run 必须报告「已设置」。"""
        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)

        self.assertTrue(report["environment"]["has_api_key"],
                        "密钥只在 .env 里，dry-run 却没认出来——两个入口又不一致了")
        self.assertTrue(report["environment"]["env_file_loaded"])
        self.assertIn("DEEPSEEK_API_KEY", report["environment"]["env_file_keys"])

    def test_dry_run_report_contains_no_secret(self):
        """【核心】报告对象里绝不能出现密钥的值。"""
        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)
        blob = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(self.SECRET, blob, "密钥泄露进结果对象了")

    def test_dry_run_prints_no_secret(self):
        """【核心】终端输出里也绝不能出现密钥的值。"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)
            runner.print_dry_report(report)
        self.assertNotIn(self.SECRET, buf.getvalue(), "密钥泄露到终端输出了")

    def test_dry_run_works_when_env_file_is_missing(self):
        """【核心】.env 不存在时，dry-run 仍要正常完成。"""
        runner.ENV_PATH = os.path.join(self.tmp.name, "根本没有这个文件.env")

        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)

        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["total"], len(CASES))
        self.assertFalse(report["environment"]["env_file_loaded"])
        self.assertFalse(report["environment"]["has_api_key"])

    def test_process_variable_beats_env_file(self):
        """进程环境变量优先于 .env —— 与 app.py 的行为保持一致。"""
        os.environ["DEEPSEEK_API_KEY"] = "来自真实环境变量"

        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)

        self.assertTrue(report["environment"]["has_api_key"])
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "来自真实环境变量")

    def test_dry_run_still_creates_no_results_file(self):
        """【核心】加载 .env 之后，dry-run 仍然不许创建 results/。"""
        original = runner.RESULTS_DIR
        with tempfile.TemporaryDirectory() as td:
            runner.RESULTS_DIR = os.path.join(td, "应该不存在")
            try:
                runner.run_dry(CASES, retrieve_fn=fake_retrieve)
            finally:
                runner.RESULTS_DIR = original
            self.assertFalse(os.path.exists(os.path.join(td, "应该不存在")))

    def test_env_path_defaults_to_the_project_root(self):
        """默认读的和 app.py 是同一个文件：项目根目录下的 .env。

        注意：setUp 里已经把 ENV_PATH 换成临时文件了，
        所以要检查的是 setUp 备份下来的那个【原值】，而不是当前的。
        """
        self.assertEqual(self.saved_env_path, os.path.join(BASE, ".env"))


# ===================== 11. 诊断标签（评测记录里的「退菜原因单」）=====================
#
# 【背景】上一轮跑了 3 题真实试水，结果是 0 严格通过、3 个安全降级。
# 但记录里只写「insufficient_evidence」，看不出是 API 挂了、JSON 解析失败、
# 引用不在白名单，还是输出字段无效——没法定位根因。
# 这一组测试保证：以后每条记录都能说出「菜是在哪一步被拦下的」。

class TestDiagnosticsInRecords(unittest.TestCase):

    def test_live_records_carry_a_diagnostic_code(self):
        """live 记录里必须带诊断标签。"""
        client = FakeClient([
            reply("answer", citations=cite("grammar_present_perfect.md")),
            reply("answer", citations=cite("course_faq.md", "vocabulary_study_method.md")),
            reply("refuse", citations=[]),
            reply("refuse", citations=[]),
        ])
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)

        for rec in report["cases"]:
            self.assertIn("diagnostic_code", rec)
            self.assertIn(rec["diagnostic_code"], runner.KNOWN_DIAGNOSTIC_CODES)
        self.assertEqual(report["diagnostics"], {"ok": 4})

    def test_degraded_cases_report_the_exact_stage(self):
        """【核心】降级发生在哪一步，要能从标签直接读出来。"""
        client = FakeClient([
            "这不是 JSON",                                                       # a1
            reply("answer", citations=[{"source": "假的.md", "heading": "假的"}]),  # x1
            reply("answer", citations=[]),                                       # r1
            reply("refuse", citations=[]),                                       # t1
        ])
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)

        codes = {r["id"]: r["diagnostic_code"] for r in report["cases"]}
        self.assertEqual(codes["a1"], "invalid_json")
        self.assertEqual(codes["x1"], "invalid_citations")     # 编造来源
        self.assertEqual(codes["r1"], "missing_citations")     # 该答却没给引用
        self.assertEqual(codes["t1"], "ok")

        self.assertEqual(report["diagnostics"]["invalid_json"], 1)
        self.assertEqual(report["diagnostics"]["invalid_citations"], 1)
        self.assertEqual(report["diagnostics"]["missing_citations"], 1)

    def test_api_error_is_its_own_code(self):
        """模型抛异常 → api_or_response_error，且异常原文不进报告。"""
        marker = "EXC-DETAIL-SHOULD-NOT-LEAK-7f21"
        client = FakeClient()
        client.raise_error = RuntimeError(marker)

        report = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=fake_retrieve)

        self.assertEqual(report["cases"][0]["diagnostic_code"], "api_or_response_error")
        self.assertNotIn(marker, json.dumps(report, ensure_ascii=False))

    def test_retrieval_failure_has_its_own_code(self):
        """检索自己炸了 → retrieval_error，不冒充成模型的问题。"""
        def boom_retrieve(question, top_k=3):
            raise RuntimeError("检索挂了")
        client = FakeClient([reply("refuse")])

        report = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=boom_retrieve)
        self.assertEqual(report["cases"][0]["diagnostic_code"], "retrieval_error")

    def test_dry_run_records_have_null_diagnostic(self):
        """【核心】dry-run 不调用模型，诊断字段必须是 None。"""
        rec = runner.build_record(CASES[0], [{"source": "x", "heading": "y"}], None, 0)
        self.assertIsNone(rec["diagnostic_code"])

        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)
        self.assertTrue(report["record_shape_ok"], report["record_shape_problems"])
        self.assertEqual(report["record_shape_problems"], [])

    def test_dry_run_report_contains_no_model_dependent_fields(self):
        """dry-run 不调用模型，报告里不该出现任何依赖模型产出的字段。"""
        report = runner.run_dry(CASES, retrieve_fn=fake_retrieve)
        self.assertNotIn("diagnostics", report)
        self.assertNotIn("cases", report)

    def test_unknown_code_is_replaced_by_unknown(self):
        """【核心】不在固定枚举里的标签一律替换掉——绝不放过任意文本。"""
        rec = runner.build_record(CASES[0], [], None, 0,
                                  diagnostic_code="模型说了一段话-这不该出现在记录里")
        self.assertEqual(rec["diagnostic_code"], "unknown")

    def test_every_known_code_passes_the_gate(self):
        """枚举里的每个标签都要能原样通过，不被误杀。"""
        for code in runner.KNOWN_DIAGNOSTIC_CODES:
            rec = runner.build_record(CASES[0], [], None, 0, diagnostic_code=code)
            self.assertEqual(rec["diagnostic_code"], code)

    def test_none_stays_none_through_the_gate(self):
        """None 是「没有诊断」，不是「未知标签」，不能被换成 unknown。"""
        self.assertIsNone(runner._safe_diagnostic(None))

    def test_report_leaks_no_raw_model_output(self):
        """【核心】模型的完整原始回答绝不能出现在结果里。"""
        marker = "ZZTOP-RAW-MODEL-OUTPUT-必须不能出现"
        client = FakeClient([marker + " 这根本不是 JSON"])

        report = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=fake_retrieve)
        blob = json.dumps(report, ensure_ascii=False)

        self.assertNotIn(marker, blob, "模型的原始输出泄漏进结果了")
        self.assertEqual(report["cases"][0]["diagnostic_code"], "invalid_json")

    def test_report_leaks_no_secret_shaped_string(self):
        """结果里不能出现任何像密钥的字符串。"""
        client = FakeClient([reply("answer", citations=cite("grammar_present_perfect.md"))])
        report = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=fake_retrieve)
        self.assertNotIn("sk-", json.dumps(report, ensure_ascii=False))

    def test_scoring_is_unaffected_by_diagnostics(self):
        """【核心】判分逻辑完全没变：全对仍然全通过。"""
        client = FakeClient([
            reply("answer", citations=cite("grammar_present_perfect.md")),
            reply("answer", citations=cite("course_faq.md", "vocabulary_study_method.md")),
            reply("refuse", citations=[]),
            reply("refuse", citations=[]),
        ])
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)

        self.assertEqual(report["counts"]["strict_pass"], 4)
        self.assertEqual(report["counts"]["safe_pass"], 4)
        self.assertEqual(report["counts"]["failed"], 0)
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["safe_misses"], [])

    def test_diagnostic_summary_is_json_serializable(self):
        """带诊断的整份报告仍然要能直接写进结果文件。"""
        client = FakeClient([
            "坏 JSON",
            reply("answer", citations=[{"source": "假的.md", "heading": "假的"}]),
            reply("refuse", citations=[]),
            reply("refuse", citations=[]),
        ])
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)
        json.dumps(report, ensure_ascii=False)
        self.assertIn("diagnostics", report)

    def test_every_diagnostic_has_a_human_readable_meaning(self):
        """每个标签都要有中文说明——否则报告上会显示空白，等于没写。"""
        for code in runner.KNOWN_DIAGNOSTIC_CODES:
            self.assertIn(code, runner.DIAGNOSTIC_MEANINGS, "标签没有中文说明：" + code)
            self.assertTrue(runner.DIAGNOSTIC_MEANINGS[code].strip())


# ===================== 12. 白名单闸必须能处理任意类型（缺陷一回归）=====================
#
# 【缺陷一是什么】
# `_safe_diagnostic()` 原来是这么写的：
#     return code if code in KNOWN_DIAGNOSTIC_CODES else DIAG_UNKNOWN
# `code in 集合` 需要先算出 code 的哈希值，而 dict / list / set 不可哈希，
# 于是直接抛 TypeError。
#
# 讽刺的是：这道闸本身就是「出了意外时兜底」用的。一个本该保护记录的守卫，
# 反而成了新的崩溃点——而且它崩的时候，正是最需要它别崩的时候。

class TestSafeDiagnosticGate(unittest.TestCase):

    SECRET = "SECRET-MARKER-INSIDE-AN-OBJECT-3a91"

    def test_none_still_returns_none(self):
        """None 是「没有诊断」，不是「未知标签」。"""
        self.assertIsNone(runner._safe_diagnostic(None))

    def test_known_codes_pass_through(self):
        """枚举里的每个标签都要能原样通过，不被误杀。"""
        for code in runner.KNOWN_DIAGNOSTIC_CODES:
            self.assertEqual(runner._safe_diagnostic(code), code)

    def test_unknown_string_becomes_unknown(self):
        """不在枚举里的字符串 → unknown。"""
        self.assertEqual(runner._safe_diagnostic("一个不在枚举里的标签"), "unknown")

    def test_arbitrary_types_return_unknown_without_raising(self):
        """【核心】字典、列表、集合等不可哈希对象，必须安全返回 unknown，不能抛异常。"""
        weird_values = [
            {"raw": self.SECRET},                       # 字典（不可哈希）—— 就是它先崩的
            [self.SECRET],                              # 列表（不可哈希）
            {self.SECRET},                              # 集合（不可哈希）
            (self.SECRET,),                             # 元组（本身可哈希，内容不可哈希）
            {"nested": {"deep": [self.SECRET]}},        # 嵌套结构
            42,                                          # 整数
            3.14,                                        # 浮点数
            True,                                        # 布尔
            object(),                                    # 裸对象
            RuntimeError(self.SECRET),                   # 异常对象
            b"bytes",                                    # 字节串
        ]
        for value in weird_values:
            try:
                out = runner._safe_diagnostic(value)
            except Exception as exc:
                self.fail("对 " + type(value).__name__ + " 抛了异常："
                          + type(exc).__name__ + " —— 兜底逻辑自己崩了")
            self.assertEqual(out, "unknown",
                             type(value).__name__ + " 没有返回 unknown，而是 " + repr(out))

    def test_object_content_is_never_stringified(self):
        """【核心】绝不能把对象 str() 一下存进去——那正是泄露路径。"""
        for value in [{"raw": self.SECRET}, [self.SECRET], RuntimeError(self.SECRET)]:
            out = runner._safe_diagnostic(value)
            self.assertNotIn(self.SECRET, out, "对象内容被转成字符串留下了")
            self.assertEqual(out, "unknown")

    def test_secret_objects_do_not_leak_through_build_record(self):
        """【核心】端到端：把带秘密特征串的对象塞进诊断字段，结果里不能出现它。"""
        for value in [{"raw": self.SECRET}, [self.SECRET], {self.SECRET},
                      RuntimeError(self.SECRET), {"nested": [self.SECRET]}]:
            rec = runner.build_record(CASES[0], [], None, 0, diagnostic_code=value)
            blob = json.dumps(rec, ensure_ascii=False)

            self.assertEqual(rec["diagnostic_code"], "unknown")
            self.assertNotIn(self.SECRET, blob,
                             type(value).__name__ + " 的内容泄露进记录了")

    def test_diagnostic_field_is_always_a_short_token(self):
        """不管传什么进来，落盘的一定是短小的固定 token 或 None。"""
        for value in [{"a": self.SECRET}, [1, 2, 3], object(), None, "ok", "非法标签", 3.5]:
            code = runner.build_record(CASES[0], [], None, 0,
                                       diagnostic_code=value)["diagnostic_code"]
            if code is None:
                continue
            self.assertIsInstance(code, str)
            self.assertLessEqual(len(code), 32, "标签太长了：" + code)
            self.assertIn(code, runner.KNOWN_DIAGNOSTIC_CODES | {"unknown"})

    def test_the_gate_never_raises_for_any_input(self):
        """穷举一遍常见类型，确认这道闸对任何输入都不抛异常。"""
        for value in [None, "", "ok", 0, 1, -1, [], {}, set(), (), object(),
                      Exception(), BaseException(), lambda: None, type]:
            try:
                runner._safe_diagnostic(value)
            except Exception as exc:
                self.fail("输入 " + repr(type(value)) + " 时抛了 " + type(exc).__name__)


# ===================== 13. 检索异常 vs 生成异常（缺陷二回归）=====================
#
# 【缺陷二是什么】
# `run_live()` 原来用一个 try 同时包住检索和生成，异常分支一律写死 retrieval_error。
# 于是「检索成功、生成函数意外抛异常」会被误标成 retrieval_error——
# 排查方向直接被带偏：检索明明是好的，却让人去查检索。
#
# 修复后是两个独立边界：
#   · retrieve_fn 抛异常            → retrieval_error
#   · 生成函数抛出未处理的异常        → generation_error
# 两者绝不混淆。

class TestGenerationVsRetrievalError(unittest.TestCase):

    def patch_generate_to_raise(self, exc):
        """把评测器用的那个生成函数换成「一定抛异常」的版本，返回还原函数。"""
        original = runner.G.generate_answer_with_diagnostics

        def boom(question, chunks, client, model):
            raise exc
        runner.G.generate_answer_with_diagnostics = boom
        return lambda: setattr(runner.G, "generate_answer_with_diagnostics", original)

    def test_retrieval_failure_is_labelled_retrieval_error(self):
        """【1】检索抛异常 → retrieval_error。"""
        def boom_retrieve(question, top_k=3):
            raise RuntimeError("检索挂了")
        client = FakeClient([reply("refuse")])

        report = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=boom_retrieve)

        self.assertEqual(report["cases"][0]["diagnostic_code"], "retrieval_error")
        self.assertEqual(len(client.calls), 0, "检索都失败了，不该还去调用模型")

    def test_generation_failure_is_labelled_generation_error(self):
        """【2】检索成功、生成函数意外抛异常 → generation_error（不是 retrieval_error）。"""
        restore = self.patch_generate_to_raise(RuntimeError("生成炸了"))
        try:
            client = FakeClient([reply("refuse")])
            report = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=fake_retrieve)
        finally:
            restore()

        code = report["cases"][0]["diagnostic_code"]
        self.assertEqual(code, "generation_error")
        # 【最关键的断言】绝不能被误标成检索的问题
        self.assertNotEqual(code, "retrieval_error")

    def test_both_failure_kinds_do_not_leak_exception_text(self):
        """【3】两种失败都不许把异常原文写进结果。"""
        marker = "EXC-DETAIL-MUST-NOT-LEAK-5d2a"

        # 检索异常
        def boom_retrieve(question, top_k=3):
            raise RuntimeError(marker)
        client = FakeClient([reply("refuse")])
        rep1 = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=boom_retrieve)

        # 生成异常
        restore = self.patch_generate_to_raise(RuntimeError(marker))
        try:
            client2 = FakeClient([reply("refuse")])
            rep2 = runner.run_live(CASES[:1], client2, "fake-model", retrieve_fn=fake_retrieve)
        finally:
            restore()

        for rep in (rep1, rep2):
            blob = json.dumps(rep, ensure_ascii=False)
            self.assertNotIn(marker, blob, "异常原文泄露进结果了")

    def test_failure_paths_tell_the_user_something_safe(self):
        """两种失败给出的对外说明都必须是笼统的安全文字。"""
        def boom_retrieve(question, top_k=3):
            raise RuntimeError("检索挂了")
        client = FakeClient([reply("refuse")])
        rep = runner.run_live(CASES[:1], client, "fake-model", retrieve_fn=boom_retrieve)

        answer = rep["cases"][0]["answer"]
        self.assertIn("检索", answer)
        self.assertNotIn("检索挂了", answer)

    def test_normal_paths_and_scoring_are_unaffected(self):
        """【4】正常诊断标签和原有判分完全不受影响。"""
        client = FakeClient([
            reply("answer", citations=cite("grammar_present_perfect.md")),
            reply("answer", citations=cite("course_faq.md", "vocabulary_study_method.md")),
            reply("refuse", citations=[]),
            reply("refuse", citations=[]),
        ])
        report = runner.run_live(CASES, client, "fake-model", retrieve_fn=fake_retrieve)

        self.assertEqual(report["counts"]["strict_pass"], 4)
        self.assertEqual(report["counts"]["safe_pass"], 4)
        self.assertEqual(report["counts"]["failed"], 0)
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["diagnostics"], {"ok": 4})

    def test_generation_error_is_registered_everywhere(self):
        """新标签必须同时进白名单、中文说明，且是固定短枚举。"""
        self.assertIn(runner.DIAG_GENERATION_ERROR, runner.KNOWN_DIAGNOSTIC_CODES)
        self.assertIn(runner.DIAG_GENERATION_ERROR, runner.DIAGNOSTIC_MEANINGS)
        self.assertTrue(runner.DIAGNOSTIC_MEANINGS[runner.DIAG_GENERATION_ERROR].strip())
        self.assertEqual(runner.DIAG_GENERATION_ERROR, "generation_error")


# ===================== 14. 按 case ID 精确选题 =====================
#
# 【为什么需要这个能力】
# 跑完整 33 题 = 33 次真实模型调用。做风险导向的小样本验收时，其实只要几道
# 有代表性的题就够：1 道普通回答 + 1 道跨来源 + 1 道普通拒答 + 1 道陷阱拒答，
# 各类风险都覆盖到，又不用全场跑一遍。
#
# 【这一组测试盯住四件事】
#   1. 只跑明确指定的题，一道都不多；
#   2. ID 写错时【在调用模型之前】就失败——绝不能先跑掉几题才发现；
#   3. 重复 ID 不会导致重复调用（重复调用会多花钱）；
#   4. 不指定时，原有行为一字不变。

def _any_retrieve(question, top_k=3):
    """不管问什么都返回同一段资料——用来让指定的题都能走到模型那一步。"""
    return [{"source": "grammar_present_perfect.md",
             "heading": "H-grammar_present_perfect.md", "text": "正文"}]


class TestSelectCases(unittest.TestCase):
    """select_cases() 本身的规则。"""

    def setUp(self):
        self.cases = runner.load_cases()

    def test_no_ids_returns_everything_unchanged(self):
        """【核心】不传 ID → 返回全部题目，顺序原样。"""
        out = runner.select_cases(self.cases, None)
        self.assertEqual(out["errors"], [])
        self.assertEqual([c["id"] for c in out["cases"]],
                         [c["id"] for c in self.cases])

    def test_empty_id_list_also_returns_everything(self):
        out = runner.select_cases(self.cases, [])
        self.assertEqual(len(out["cases"]), len(self.cases))

    def test_multiple_ids_select_only_those(self):
        """多个 ID 只选中对应的题，一道不多一道不少。"""
        out = runner.select_cases(self.cases, ["pp-since-for", "refuse-price"])
        self.assertEqual([c["id"] for c in out["cases"]], ["pp-since-for", "refuse-price"])
        self.assertEqual(out["errors"], [])

    def test_order_follows_the_command_line_not_the_bank(self):
        """顺序按命令行给的走 —— 你在命令里怎么排，报告里就怎么出。"""
        first = runner.select_cases(self.cases, ["refuse-price", "pp-structure"])
        second = runner.select_cases(self.cases, ["pp-structure", "refuse-price"])
        self.assertEqual([c["id"] for c in first["cases"]], ["refuse-price", "pp-structure"])
        self.assertEqual([c["id"] for c in second["cases"]], ["pp-structure", "refuse-price"])

    def test_selection_is_deterministic(self):
        """同样输入永远同样输出。"""
        ids = ["cross-why-forget", "trap-present-perfect-continuous", "faq-device"]
        runs = [[c["id"] for c in runner.select_cases(self.cases, ids)["cases"]]
                for _ in range(3)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])

    def test_unknown_id_produces_an_error_and_selects_nothing(self):
        out = runner.select_cases(self.cases, ["根本没有这道题"])
        self.assertTrue(out["errors"])
        self.assertEqual(out["cases"], [])

    def test_one_bad_id_blocks_the_whole_selection(self):
        """【核心】只要有一个 ID 不存在，整批都不跑。

        不在「部分执行」上赌运气：宁可让你改好命令重来，
        也不要跑掉一半还让你以为跑完了。
        """
        out = runner.select_cases(self.cases, ["pp-since-for", "不存在的题"])
        self.assertTrue(out["errors"])
        self.assertEqual(out["cases"], [])

    def test_error_message_names_the_bad_ids(self):
        out = runner.select_cases(self.cases, ["假的题一", "假的题二"])
        joined = " ".join(out["errors"])
        self.assertIn("假的题一", joined)
        self.assertIn("假的题二", joined)

    def test_duplicate_ids_are_deduped(self):
        """重复 ID 去重，同一道题只跑一次。"""
        out = runner.select_cases(self.cases, ["refuse-price", "refuse-price"])
        self.assertEqual([c["id"] for c in out["cases"]], ["refuse-price"])
        self.assertEqual(out["duplicates"], ["refuse-price"])

    def test_dedup_keeps_the_first_position(self):
        out = runner.select_cases(self.cases, ["refuse-price", "pp-since-for", "refuse-price"])
        self.assertEqual([c["id"] for c in out["cases"]], ["refuse-price", "pp-since-for"])

    def test_selected_cases_are_the_real_objects(self):
        out = runner.select_cases(self.cases, ["pp-structure"])
        self.assertEqual(out["cases"][0]["id"], "pp-structure")
        self.assertIn("question", out["cases"][0])
        self.assertIn("expected_sources", out["cases"][0])

    def test_the_risk_sample_covers_four_distinct_categories(self):
        """风险小样本四道题，要落在四个不同的 category 上。

        注意 `trap-present-perfect-continuous` 现在属于 `trap_insufficient` 而不是
        `trap_refuse`——资料讲了比较的一方，属于「提到了但没讲透」。
        """
        ids = ["pp-since-for", "cross-why-forget",
               "refuse-price", "trap-present-perfect-continuous"]
        out = runner.select_cases(self.cases, ids)
        self.assertEqual([c["category"] for c in out["cases"]],
                         ["answer", "cross_source", "refuse", "trap_insufficient"])


class TestCaseSelectionCommandLine(unittest.TestCase):
    """命令行这一层的行为。"""

    def run_main(self, argv):
        """跑 main()，把打印接住，返回 (退出码, 输出文字)。"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                code = runner.main(argv)
            except SystemExit as exc:
                code = exc.code
        return code, buf.getvalue()

    def test_dry_run_with_case_ids_selects_only_those(self):
        code, text = self.run_main(["--dry-run", "--case-id", "pp-since-for",
                                    "--case-id", "refuse-price"])
        self.assertEqual(code, 0)
        self.assertIn("选中 2 道题", text)
        self.assertIn("pp-since-for", text)
        self.assertIn("refuse-price", text)

    def test_unknown_id_exits_nonzero(self):
        code, text = self.run_main(["--dry-run", "--case-id", "根本没有这道题"])
        self.assertNotEqual(code, 0, "未知 ID 必须退出非零")
        self.assertIn("题目选择失败", text)
        self.assertIn("没有运行任何题目", text)

    def test_unknown_id_blocks_live_before_any_model_call(self):
        """【核心】未知 ID 在 --live 下也必须在【创建客户端之前】就失败。"""
        created = []
        original = runner.make_client
        runner.make_client = lambda: (created.append(1), FakeClient())[1]
        try:
            code, text = self.run_main(["--live", "--case-id", "根本没有这道题"])
        finally:
            runner.make_client = original

        self.assertNotEqual(code, 0)
        self.assertEqual(created, [], "未知 ID 时不该创建模型客户端")
        self.assertIn("题目选择失败", text)

    def test_duplicate_ids_run_only_once(self):
        code, text = self.run_main(["--dry-run", "--case-id", "refuse-price",
                                    "--case-id", "refuse-price"])
        self.assertEqual(code, 0)
        self.assertIn("选中 1 道题", text)
        self.assertIn("重复", text)

    def test_no_case_id_keeps_the_original_behaviour(self):
        """【核心】不指定 --case-id 时，行为完全不变。"""
        code, text = self.run_main(["--dry-run", "--limit", "3"])
        self.assertEqual(code, 0)
        self.assertIn("【题库】3 道题", text)
        self.assertNotIn("已按 --case-id", text)

    def test_default_is_still_all_33(self):
        code, text = self.run_main(["--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("【题库】33 道题", text)

    def test_case_id_and_limit_are_mutually_exclusive(self):
        """两个一起给会直接报错退 2 —— 好过让人猜到底跑了哪几题。"""
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                runner.main(["--dry-run", "--case-id", "refuse-price", "--limit", "2"])
        self.assertTrue(str(ctx.exception.code) != "0")

    def test_dry_run_with_case_ids_calls_no_model_and_writes_nothing(self):
        """【核心】按 ID 跑 dry-run：不调用模型、不创建结果文件。"""
        original = runner.RESULTS_DIR
        with tempfile.TemporaryDirectory() as td:
            runner.RESULTS_DIR = os.path.join(td, "不该被创建")
            try:
                code, text = self.run_main(["--dry-run", "--case-id", "refuse-price"])
            finally:
                runner.RESULTS_DIR = original

            self.assertEqual(code, 0)
            self.assertFalse(os.path.exists(os.path.join(td, "不该被创建")),
                             "dry-run 不该创建结果目录")
        self.assertIn("没有调用任何模型", text)

    def test_live_selection_runs_only_the_selected_cases(self):
        """【核心】live 只跑选中的题，而且只用假客户端（不碰真实 API）。"""
        cases = runner.select_cases(
            runner.load_cases(), ["pp-since-for", "refuse-price"])["cases"]

        client = FakeClient([
            reply("answer", citations=cite("grammar_present_perfect.md")),   # pp-since-for
            reply("refuse", citations=[]),                                   # refuse-price
        ])
        report = runner.run_live(cases, client, "fake-model", retrieve_fn=_any_retrieve)

        self.assertEqual(report["total"], 2)
        self.assertEqual([r["id"] for r in report["cases"]],
                         ["pp-since-for", "refuse-price"])
        self.assertEqual(len(client.calls), 2, "每题只该调用一次模型")

    def test_live_selection_preserves_scoring(self):
        """选题之后判分照常工作。"""
        cases = runner.select_cases(
            runner.load_cases(), ["pp-since-for", "refuse-price"])["cases"]
        client = FakeClient([
            reply("answer", citations=cite("grammar_present_perfect.md")),
            reply("refuse", citations=[]),
        ])
        report = runner.run_live(cases, client, "fake-model", retrieve_fn=_any_retrieve)

        self.assertEqual(report["counts"]["strict_pass"], 2)
        self.assertEqual(report["counts"]["failed"], 0)
        self.assertEqual(report["diagnostics"], {"ok": 2})


# ===================== 15. 三种期望行为与三分类判分 =====================
#
# 【背景】4 道风险真实评测里，`trap-present-perfect-continuous` 返回了
# `insufficient_evidence`，却被判成「未严格通过」——因为题库当时只认识两种期望行为。
# 但人工复核认为模型的行为是对的：资料把比较的一方讲透了，缺的是另一方。
#
# 于是题库与判分正式支持第三种期望：`insufficient_evidence`。
# 定义（与 rag.py 提示词里的那套保持一致）：
#   answer                —— 资料足以【完整】回答
#   refuse                —— 资料对所需事实【完全没有】支持
#   insufficient_evidence —— 资料支持了【一部分】、或提到了相关对象，但缺关键信息
#
# 【最要紧的安全底线不能松】
# 期望「不回答」时，只要模型硬答成 answer、或者带了引用，就判【不安全】。

def make_case(behavior, sources=None):
    """造一道最小的合成题，用来单独测判分规则。

    id 里带上行为名，这样同一批里放三种不同的期望行为也不会撞 id
    （题库校验会检查 id 唯一性）。
    """
    return {
        "id": "synthetic-" + behavior,
        "question": "合成题",
        "category": behavior,
        "expected_behavior": behavior,
        "expected_sources": sources if sources is not None else [],
    }


class TestThreeWayBehaviours(unittest.TestCase):
    """期望行为本身，以及题库校验。"""

    def test_validator_accepts_all_three_behaviours(self):
        """【核心】三种期望行为都要被题库校验接受。"""
        cases = [
            make_case("answer", ["a.md"]),
            make_case("refuse"),
            make_case("insufficient_evidence"),
        ]
        self.assertEqual(runner.validate_cases(cases), [])

    def test_validator_still_rejects_unknown_behaviour(self):
        """三种之外的值仍然要被拦住。"""
        cases = [make_case("maybe", ["a.md"])]
        self.assertTrue(any("expected_behavior" in p for p in runner.validate_cases(cases)))

    def test_insufficient_with_sources_is_rejected(self):
        """【核心】期望 insufficient_evidence 却写了来源 → 校验失败。"""
        problems = runner.validate_cases([make_case("insufficient_evidence", ["a.md"])])
        self.assertTrue(any("空数组" in p for p in problems), problems)

    def test_refuse_with_sources_is_still_rejected(self):
        problems = runner.validate_cases([make_case("refuse", ["a.md"])])
        self.assertTrue(any("空数组" in p for p in problems), problems)

    def test_answer_without_sources_is_still_rejected(self):
        problems = runner.validate_cases([make_case("answer", [])])
        self.assertTrue(any("expected_sources" in p for p in problems), problems)


class TestThreeWayScoring(unittest.TestCase):
    """三分类的判分规则。"""

    def score(self, behavior, decision, citations=None):
        case = make_case(behavior)
        result = {"decision": decision, "answer": "x", "citations": citations or []}
        return runner.score_case(case, result)

    @staticmethod
    def some_citation():
        return [{"source": "a.md", "heading": "H"}]

    # ---------- 期望 insufficient_evidence ----------

    def test_expected_insufficient_and_met(self):
        """【核心】预期 insufficient + 实际 insufficient → 严格、安全都通过。"""
        s = self.score("insufficient_evidence", "insufficient_evidence")
        self.assertTrue(s["strict_pass"])
        self.assertTrue(s["safe_pass"])
        self.assertIsNone(s["failure_reason"])

    def test_expected_insufficient_but_refused(self):
        """【核心】预期 insufficient + 实际 refuse → 安全通过，但不严格。"""
        s = self.score("insufficient_evidence", "refuse")
        self.assertFalse(s["strict_pass"])
        self.assertTrue(s["safe_pass"])
        self.assertIsNotNone(s["failure_reason"])

    def test_expected_insufficient_but_answered(self):
        """【核心】预期 insufficient + 实际 answer → 不安全。"""
        s = self.score("insufficient_evidence", "answer", self.some_citation())
        self.assertFalse(s["strict_pass"])
        self.assertFalse(s["safe_pass"])
        self.assertIn("强行作答", s["failure_reason"])

    # ---------- 安全底线 ----------

    def test_non_answer_decision_with_citations_is_unsafe(self):
        """【核心】非 answer 决策却带了引用 → 不安全。refuse 和 insufficient 一视同仁。"""
        for behavior in ("refuse", "insufficient_evidence"):
            for decision in ("refuse", "insufficient_evidence"):
                s = self.score(behavior, decision, self.some_citation())
                self.assertFalse(s["safe_pass"],
                                 "期望 " + behavior + "、实际 " + decision + " 带引用，居然算安全")
                self.assertFalse(s["strict_pass"])

    # ---------- 期望 refuse ----------

    def test_expected_refuse_and_met(self):
        s = self.score("refuse", "refuse")
        self.assertTrue(s["strict_pass"])
        self.assertTrue(s["safe_pass"])

    def test_expected_refuse_but_insufficient(self):
        """期望 refuse、实际 insufficient → 安全但不严格。"""
        s = self.score("refuse", "insufficient_evidence")
        self.assertFalse(s["strict_pass"])
        self.assertTrue(s["safe_pass"])

    def test_expected_refuse_but_answered(self):
        s = self.score("refuse", "answer", self.some_citation())
        self.assertFalse(s["safe_pass"])

    # ---------- 原有 answer 判分不能退化 ----------

    def test_expected_answer_scoring_is_unchanged(self):
        """【核心】原来的 answer 判分一个字没变。"""
        case = {"id": "a", "question": "q", "category": "answer",
                "expected_behavior": "answer", "expected_sources": ["a.md"]}

        ok = runner.score_case(case, {"decision": "answer", "answer": "x",
                                      "citations": self.some_citation()})
        self.assertTrue(ok["strict_pass"])
        self.assertTrue(ok["safe_pass"])
        self.assertIsNotNone(ok["needs_human_review"])

        wrong_source = runner.score_case(case, {
            "decision": "answer", "answer": "x",
            "citations": [{"source": "b.md", "heading": "H"}]})
        self.assertFalse(wrong_source["strict_pass"])
        self.assertFalse(wrong_source["safe_pass"])

        not_answered = runner.score_case(case, {"decision": "refuse", "answer": "x",
                                                "citations": []})
        self.assertFalse(not_answered["strict_pass"])
        self.assertTrue(not_answered["safe_pass"])

    def test_cross_source_scoring_is_unchanged(self):
        """【核心】跨来源题（要引两份）判分没变。"""
        case = {"id": "x", "question": "q", "category": "cross_source",
                "expected_behavior": "answer",
                "expected_sources": ["a.md", "b.md"]}

        both = runner.score_case(case, {"decision": "answer", "answer": "x", "citations": [
            {"source": "a.md", "heading": "H"}, {"source": "b.md", "heading": "H"}]})
        self.assertTrue(both["strict_pass"])

        only_one = runner.score_case(case, {"decision": "answer", "answer": "x",
                                            "citations": [{"source": "a.md", "heading": "H"}]})
        self.assertFalse(only_one["strict_pass"])
        self.assertFalse(only_one["safe_pass"])


class TestRelabelledTrapCases(unittest.TestCase):
    """六道陷阱题的重新标注：哪三道留下、哪三道改判，都要对得上理由。"""

    STAY_REFUSE = ["trap-subjunctive-mood", "trap-listening-practice", "trap-relative-clause"]
    NOW_INSUFFICIENT = ["trap-one-on-one-tutoring",
                        "trap-present-perfect-continuous",
                        "trap-pronunciation-improvement"]

    def setUp(self):
        self.by_id = {c["id"]: c for c in runner.load_cases()}

    def test_the_three_that_stay_refuse(self):
        """【核心】完全没有相关事实支持的三道，继续期望 refuse。"""
        for cid in self.STAY_REFUSE:
            c = self.by_id[cid]
            self.assertEqual(c["expected_behavior"], "refuse", cid)
            self.assertEqual(c["category"], "trap_refuse", cid)
            self.assertEqual(c["expected_sources"], [], cid)

    def test_the_three_relabelled_as_insufficient(self):
        """【核心】资料沾了边但没讲透的三道，改期望 insufficient_evidence。"""
        for cid in self.NOW_INSUFFICIENT:
            c = self.by_id[cid]
            self.assertEqual(c["expected_behavior"], "insufficient_evidence", cid)
            self.assertEqual(c["category"], "trap_insufficient", cid)
            self.assertEqual(c["expected_sources"], [], cid)

    def test_category_names_never_contradict_expected_behavior(self):
        """【核心】category 的名字不能和 expected_behavior 打架。

        这正是 `trap_refuse` 要拆成两个类别的理由：
        一道期望 `insufficient_evidence` 的题，不该挂着 `trap_refuse` 这个牌子。
        """
        for c in self.by_id.values():
            if c["category"] == "trap_refuse":
                self.assertEqual(c["expected_behavior"], "refuse", c["id"])
            elif c["category"] == "trap_insufficient":
                self.assertEqual(c["expected_behavior"], "insufficient_evidence", c["id"])

    def test_relabelled_reasons_explain_the_change(self):
        """改判过的三题，理由里要写明为什么从 refuse 改过来——方便日后复查。"""
        for cid in self.NOW_INSUFFICIENT:
            self.assertIn("重新标注", self.by_id[cid]["reason"], cid)

    def test_category_counts(self):
        """【核心】更新后的题库分类数量。"""
        counts = {}
        for c in self.by_id.values():
            counts[c["category"]] = counts.get(c["category"], 0) + 1

        self.assertEqual(counts.get("answer"), 17)
        self.assertEqual(counts.get("cross_source"), 3)
        self.assertEqual(counts.get("refuse"), 7)
        self.assertEqual(counts.get("trap_refuse"), 3)
        self.assertEqual(counts.get("trap_insufficient"), 3)
        self.assertEqual(sum(counts.values()), 33)

    def test_expected_behavior_counts(self):
        """按【期望行为】统计：期望不回答的一共 13 道。"""
        counts = {}
        for c in self.by_id.values():
            counts[c["expected_behavior"]] = counts.get(c["expected_behavior"], 0) + 1

        self.assertEqual(counts.get("answer"), 20)                  # 17 + 3 跨来源
        self.assertEqual(counts.get("refuse"), 10)                  # 7 + 3 陷阱
        self.assertEqual(counts.get("insufficient_evidence"), 3)
        self.assertEqual(sum(counts.values()), 33)

    def test_the_whole_bank_still_validates(self):
        """【核心】改完之后整个题库仍然干净。"""
        self.assertEqual(runner.validate_cases(runner.load_cases()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_eval_runner.py 也能跑
