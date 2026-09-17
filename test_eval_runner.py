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


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_eval_runner.py 也能跑
