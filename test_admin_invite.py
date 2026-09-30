# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# admin_invite.py（本地发码 / 作废脚本）的自动化测试
#
# 【原则】完全离线：不联网、不碰数据库、不需要任何密钥。
# 做法是把脚本的 I/O 全部注入假的（令牌输入、码输入、确认、打印、HTTP），
# 于是可以完整地驱动一遍「生成 → 展示 → 确认 → 发送 → 解读状态」的流程。
#
# 【这一组最要紧的三条】
#   ① 状态未知时【绝不】声称成功，也【绝不】自动生成第二张码
#   ② 明文码只在【发送之前】展示一次，而且要提示「先复制、别急着转发」
#   ③ 令牌永远来自隐藏输入，不出现在命令行参数里
#
# 运行方式（在 ai_tutor 文件夹里）：
#     python -m unittest test_admin_invite -v
# =====================================================================

import os
import sys
import unittest

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import admin_invite   # 被测对象


class ScriptHarness(unittest.TestCase):
    """把脚本的 I/O 换掉，只测它的决策逻辑。"""

    TOKEN = "test-admin-token-not-a-real-one"

    def run_script(self, action, *, answers=(), secrets_=(), confirms=(),
                   http_result=(200, {"result": "created"}), base_url="https://example.invalid"):
        """跑一次 main()，把所有输入输出都换成假的。

        answers/secrets_/confirms 是按顺序喂进去的回答；
        返回 (退出码, 打印出来的所有文字, 发出去的请求列表)。
        """
        printed = []
        secrets_iter = iter(list(secrets_))
        confirm_iter = iter(list(confirms))
        sent = []

        def ask_secret(_label):
            return next(secrets_iter, "")

        def ask_confirm(_label):
            return next(confirm_iter, False)

        def post(base_url_arg, path, fields, token, timeout=30):
            sent.append({"base_url": base_url_arg, "path": path,
                         "fields": dict(fields), "token": token})
            return http_result

        code = admin_invite.main(
            [action, "--base-url", base_url],
            ask_token=lambda _label: self.TOKEN,
            ask_secret=ask_secret,
            ask_confirm=ask_confirm,
            say=printed.append,
            post=post)

        return code, "\n".join(printed), sent


# ===================== 1. 正常发码 =====================

class TestRegisterFlow(ScriptHarness):

    def test_it_generates_a_code_shows_it_once_then_sends_it(self):
        """【核心】先本地生成 → 展示一次 → 确认后才发送。"""
        code, text, sent = self.run_script(
            "new", confirms=[False, True],       # 不是重试；确认发送
            http_result=(200, {"result": "created"}))

        self.assertEqual(code, 0)
        self.assertEqual(len(sent), 1, "应该只发一次请求")
        self.assertEqual(sent[0]["path"], "/admin/invites")

        the_code = sent[0]["fields"]["code"]
        self.assertGreaterEqual(len(the_code), 24, "本地生成的码太短了")
        # 明文在输出里【只出现一次】
        self.assertEqual(text.count(the_code), 1, "邀请码被打印了不止一次")
        self.assertIn("待登记邀请码", text)
        self.assertIn("只显示这一次", text)

    def test_it_warns_the_operator_before_sending(self):
        """【核心】发送之前必须提醒：先安全复制；确认成功前不要转发。"""
        _code, text, _sent = self.run_script("new", confirms=[False, True])

        self.assertIn("安全复制", text)
        self.assertIn("不要", text)
        self.assertIn("转发", text)

    def test_the_token_is_taken_from_hidden_input_not_argv(self):
        """【核心】令牌只能来自隐藏输入；命令行里不许出现它。"""
        import contextlib
        import io

        _code, text, sent = self.run_script("new", confirms=[False, True])

        self.assertEqual(sent[0]["token"], self.TOKEN)          # 走的是注入的隐藏输入
        self.assertNotIn(self.TOKEN, text, "令牌被打出来了")

        # main() 的参数里也不能有令牌这个选项（传了就直接退出）
        # 【为什么要把 stderr 接住】argparse 会往 stderr 打一句用法提示，
        # 那是预期的输出，但混在测试结果里很吵。
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                admin_invite.main(["new", "--base-url", "https://x.invalid",
                                   "--token", "不该有这种参数"])

    def test_it_sends_the_code_in_the_body_and_the_token_in_the_header(self):
        """码在请求体、令牌在头 —— 都不进 URL（进了 URL 就会落到访问日志里）。"""
        _code, _text, sent = self.run_script("new", confirms=[False, True])

        self.assertIn("code", sent[0]["fields"])
        self.assertNotIn("token", sent[0]["fields"])
        self.assertNotIn(self.TOKEN, sent[0]["path"])

    def test_cancelling_before_send_sends_nothing(self):
        """在发送前取消 → 一个请求都不发，并告诉操作者码还在他手里。"""
        code, text, sent = self.run_script("new", confirms=[False, False])

        self.assertEqual(sent, [], "取消之后居然还是发了请求")
        self.assertEqual(code, 0)
        self.assertIn("什么都没有发送", text)
        self.assertIn("重试", text, "没告诉操作者怎么接着做")

    def test_an_empty_token_does_nothing(self):
        printed = []
        code = admin_invite.main(
            ["new", "--base-url", "https://x.invalid"],
            ask_token=lambda _l: "   ",
            ask_secret=lambda _l: "",
            ask_confirm=lambda _l: True,
            say=printed.append,
            post=lambda *a, **k: self.fail("不该发出任何请求"))
        self.assertEqual(code, 4)
        self.assertIn("没有输入令牌", "\n".join(printed))


# ===================== 2. 重试：同一张码 =====================

class TestRetryFlow(ScriptHarness):

    def test_a_retry_reuses_the_same_code_via_hidden_input(self):
        """【核心】重试时用隐藏输入收旧码，绝不生成第二张。"""
        code, _text, sent = self.run_script(
            "new",
            confirms=[True, True],                     # 是重试；确认发送
            secrets_=["the-code-from-before"],
            http_result=(200, {"result": "already_registered"}))

        self.assertEqual(code, 0)
        self.assertEqual(sent[0]["fields"]["code"], "the-code-from-before")

    def test_a_retry_never_generates_a_new_code(self):
        """【核心】重试路径里不许调用生成器 —— 否则会悄悄多出一张码。"""
        calls = []
        real_generate = admin_invite.generate_code

        def spy_generate():
            calls.append(1)
            return real_generate()

        admin_invite.generate_code = spy_generate
        self.addCleanup(setattr, admin_invite, "generate_code", real_generate)

        self.run_script("new", confirms=[True, True], secrets_=["old-code"],
                        http_result=(200, {"result": "already_registered"}))

        self.assertEqual(calls, [], "重试路径居然又生成了一张码")

    def test_an_empty_retry_code_does_nothing(self):
        code, text, sent = self.run_script("new", confirms=[True], secrets_=[""])

        self.assertEqual(sent, [])
        self.assertEqual(code, 4)
        self.assertIn("没有输入邀请码", text)


# ===================== 3. 各种结果的说法 =====================

class TestOutcomes(ScriptHarness):
    """服务端返回的每一种状态，脚本都要给出正确的下一步。"""

    def test_created_and_already_registered_are_success(self):
        for result in ("created", "already_registered"):
            code, text, _sent = self.run_script(
                "new", confirms=[False, True], http_result=(200, {"result": result}))
            with self.subTest(result=result):
                self.assertEqual(code, 0)
                self.assertIn("✅", text)
                self.assertIn("可以发给用户", text.replace("可以发给用户了", "可以发给用户"))

    def test_bound_and_revoked_codes_are_not_success(self):
        """【核心】已绑定 / 已作废 → 明说「不是新码」，并要求生成一张新的。"""
        for result, keyword in (("already_bound", "已经被某位学习者用掉"),
                                ("revoked", "已经被作废")):
            code, text, _sent = self.run_script(
                "new", confirms=[False, True], http_result=(200, {"result": result}))
            with self.subTest(result=result):
                self.assertEqual(code, 2, "不该当成成功")
                self.assertIn(keyword, text)
                self.assertIn("新的", text)
                self.assertNotIn("✅", text)

    def test_401_and_404_are_unknown_with_a_hint_but_no_claim(self):
        """【核心】401/404 只给「可能原因」，绝不断言登没登记 —— 哪怕响应带我们的标记。

        带标记只说明"这个响应来自我们的进程"，**不说明数据库有没有被写过**。
        """
        for status, payload in ((401, {"error": "unauthorized"}),
                                (404, {"error": "not_found"}),
                                (401, None),              # 代理回的，没有我们的标记
                                (404, "<html>404</html>")):
            code, text, _sent = self.run_script(
                "new", confirms=[False, True], http_result=(status, payload))
            with self.subTest(status=status, payload=str(payload)[:16]):
                self.assertEqual(code, 3, "非 200 一律是「状态未知」")
                self.assertIn("状态未知", text)
                self.assertIn("同一张码", text)
                self.assertNotIn("没有被登记", text, "不该断言没被登记")
                self.assertNotIn("✅", text, "不该报成功")

    def test_the_hint_mentions_the_token_only_as_a_possible_cause(self):
        """401 的提示可以说「多半是令牌问题」，但必须写明那**只是排查方向**。"""
        _code, text, _sent = self.run_script(
            "new", confirms=[False, True],
            http_result=(401, {"error": "unauthorized"}))

        self.assertIn("ADMIN_MINT_TOKEN", text)          # 排查方向
        self.assertIn("不是结论", text)                   # 明确它不是结论
        self.assertIn("无法确认", text)

    def test_a_bad_request_does_not_claim_success(self):
        code, text, _sent = self.run_script(
            "new", confirms=[False, True], http_result=(400, {"error": "bad_request"}))

        self.assertEqual(code, 3)
        self.assertNotIn("✅", text)
        self.assertIn("状态未知", text)
        self.assertNotIn("没有被登记", text)


# ===================== 4. 状态未知：这一组最要紧 =====================

class TestUnknownOutcome(ScriptHarness):
    """【核心】网络挂了 / 响应读不懂时：绝不声称成功，也绝不自动换一张码。"""

    def test_a_network_failure_says_unknown_and_tells_you_to_retry_the_same_code(self):
        code, text, sent = self.run_script(
            "new", confirms=[False, True], http_result=(None, None))

        self.assertEqual(code, 3, "状态未知不该返回成功码")
        self.assertIn("状态未知", text)
        self.assertIn("同一张码", text, "没告诉操作者用同一张码重试")
        self.assertIn("不要", text, "没有明确劝阻再生成一张")
        self.assertNotIn("✅", text)
        self.assertEqual(len(sent), 1, "只该发那一次请求")

    def test_a_500_with_our_own_marker_is_still_unknown(self):
        """【核心】**这一条是本轮修的那个判定。**

        500 + {"error": "internal_error"} 看起来像"我们自己回的、所以没写库"——
        **这个推理是错的**：完全可能是事务已经 COMMIT、之后才在别处出错并返回 500。
        所以绝不能说"没有被登记"，也绝不能说成功，只能按状态未知、用同一张码重试。
        """
        code, text, _sent = self.run_script(
            "new", confirms=[False, True],
            http_result=(500, {"error": "internal_error"}))

        self.assertEqual(code, 3, "500 必须是「状态未知」")
        self.assertIn("状态未知", text)
        self.assertIn("同一张码", text)
        self.assertIn("无法确认", text)
        self.assertNotIn("没有被登记", text, "绝不能声称没有登记")
        self.assertNotIn("✅", text, "绝不能声称成功")

    def test_every_5xx_variant_is_unknown(self):
        """所有 5xx（带标记 / 不带标记 / 网关页面）一律状态未知。"""
        for status in (500, 502, 503, 504):
            for payload in ({"error": "internal_error"}, None,
                            {"error": "bad_gateway"}, "<html>502</html>"):
                code, text, _sent = self.run_script(
                    "new", confirms=[False, True], http_result=(status, payload))
                with self.subTest(status=status, payload=str(payload)[:16]):
                    self.assertEqual(code, 3)
                    self.assertIn("状态未知", text)
                    self.assertIn("同一张码", text)
                    self.assertNotIn("没有被登记", text)

    def test_an_unparseable_body_is_unknown(self):
        code, text, _sent = self.run_script(
            "new", confirms=[False, True], http_result=(200, None))

        self.assertEqual(code, 3)
        self.assertIn("状态未知", text)

    def test_an_unexpected_result_value_is_unknown(self):
        code, text, _sent = self.run_script(
            "new", confirms=[False, True], http_result=(200, {"result": "???"}))

        self.assertEqual(code, 3)
        self.assertIn("状态未知", text)


# ===================== 5. 撤销 =====================

class TestRevokeFlow(ScriptHarness):

    def test_revoke_sends_the_code_with_an_explicit_confirmation(self):
        code, _text, sent = self.run_script(
            "revoke", secrets_=["code-to-kill"], confirms=[True],
            http_result=(200, {"result": "revoked", "sessions_dropped": 2}))

        self.assertEqual(code, 0)
        self.assertEqual(sent[0]["path"], "/admin/invites/revoke")
        self.assertEqual(sent[0]["fields"]["code"], "code-to-kill")
        self.assertEqual(sent[0]["fields"]["confirm"], "yes", "服务端要求的显式确认没带上")

    def test_revoke_explains_what_it_will_do_before_asking(self):
        """【核心】撤销是不可逆的：动手之前要把后果说清楚。"""
        _code, text, _sent = self.run_script(
            "revoke", secrets_=["c"], confirms=[True],
            http_result=(200, {"result": "revoked"}))

        self.assertIn("断开", text, "没说会断开已绑定的会话")
        self.assertIn("不可撤销", text)
        self.assertIn("不会被删除", text, "没说清数据不会被删（避免误解）")

    def test_cancelling_revoke_sends_nothing(self):
        code, text, sent = self.run_script("revoke", secrets_=["c"], confirms=[False])

        self.assertEqual(sent, [])
        self.assertEqual(code, 0)
        self.assertIn("什么都没有改动", text)

    def test_revoke_unknown_code_says_nothing_changed(self):
        """200 + not_found 是【服务端明确给的结果】，所以可以照实说。"""
        code, text, _sent = self.run_script(
            "revoke", secrets_=["nope"], confirms=[True],
            http_result=(200, {"result": "not_found"}))

        self.assertEqual(code, 2)
        self.assertIn("没有找到", text)
        self.assertIn("什么都没有改动", text)

    def test_revoke_network_failure_is_not_reported_as_success(self):
        code, text, _sent = self.run_script(
            "revoke", secrets_=["c"], confirms=[True], http_result=(None, None))

        self.assertEqual(code, 3)
        self.assertIn("状态未知", text)
        self.assertIn("不要", text)

    def test_revoke_never_claims_the_outcome_on_a_non_200(self):
        """【核心】撤销侧同样：非 200 一律状态未知，绝不声称作废了或没作废。"""
        for status, payload in ((500, {"error": "internal_error"}),
                                (401, {"error": "unauthorized"}),
                                (404, {"error": "not_found"}),
                                (400, {"error": "bad_request"}),
                                (500, "<html>502</html>")):
            code, text, _sent = self.run_script(
                "revoke", secrets_=["c"], confirms=[True], http_result=(status, payload))
            with self.subTest(status=status, payload=str(payload)[:16]):
                self.assertEqual(code, 3)
                self.assertIn("状态未知", text)
                self.assertIn("同一个码", text)
                self.assertNotIn("✅", text)
                self.assertNotIn("什么都没有改动", text, "不该断言服务端没动过")


# ===================== 4b. 「没有发送」和「状态未知」必须分得清 =====================

class TestNotSentVersusUnknown(ScriptHarness):
    """【核心】两条界线不能混：
    · 请求【根本没发出去】（本地校验没过 / 操作者取消）→ 可以明确说"没有发送"
    · 请求【发出去了】（哪怕没收到响应）→ 只能说"状态未知"
    """

    def test_a_cancelled_send_says_clearly_that_nothing_was_sent(self):
        code, text, sent = self.run_script("new", confirms=[False, False])

        self.assertEqual(sent, [])
        self.assertEqual(code, 0)
        self.assertIn("什么都没有发送", text)
        self.assertNotIn("状态未知", text, "压根没发，不该说状态未知")

    def test_an_invalid_url_says_nothing_was_sent(self):
        printed = []
        asked = []

        code = admin_invite.main(
            ["new", "--base-url", "http://insecure.example"],
            ask_token=lambda label: asked.append(label) or "x",
            ask_secret=lambda _l: "",
            ask_confirm=lambda _l: True,
            say=printed.append,
            post=lambda *a, **k: self.fail("不该发请求"))

        self.assertEqual(code, 4)
        text = "\n".join(printed)
        self.assertIn("什么都没有发送", text)
        self.assertNotIn("状态未知", text)


# ===================== 6. 脚本自己的安全约定 =====================

class TestScriptSafety(unittest.TestCase):

    def test_the_script_never_writes_secrets_to_disk(self):
        """【结构性保证】脚本不许写文件 —— 令牌和码都不落盘。

        【为什么用正则】以前这里查的是字面量 "open("，结果把
        `urllib.request.urlopen(` 也算成「写文件」了 —— 那是**读**响应，是必须的。
        所以要区分「open(」和「urlopen(」。
        """
        import re

        with open(os.path.join(BASE, "admin_invite.py"), encoding="utf-8") as f:
            source = f.read()

        # 真正的 open( ：前面既不能是 url（urlopen 是读响应），也不能是点（opener.open 是发请求）
        self.assertIsNone(re.search(r"(?<![\w.])open\(", source),
                          "脚本里出现了打开文件的调用（不该有任何文件读写）")
        for banned in ("json.dump", "logging.", ".write(", "csv."):
            self.assertNotIn(banned, source, "脚本里出现了写文件的迹象：" + banned)


# 【这一条要用有状态的假 I/O，所以放在 ScriptHarness 那一组里】
class TestNoSecretsInUrl(ScriptHarness):

    def test_the_token_and_code_never_land_in_a_url(self):
        """【核心】请求 URL 里不能出现令牌（URL 会进访问日志、浏览器历史）。"""
        _code, _text, sent = self.run_script("new", confirms=[False, True])

        self.assertEqual(sent[0]["path"], "/admin/invites")
        self.assertNotIn(self.TOKEN, sent[0]["path"])
        self.assertNotIn(sent[0]["fields"]["code"], sent[0]["path"])
        self.assertEqual(sent[0]["base_url"], "https://example.invalid")


# ===================== 7. 目标地址必须在要令牌之前就验干净 =====================

class TestBaseUrlValidation(unittest.TestCase):
    """【核心】令牌是要发给这个地址的 —— 地址不对，令牌就等于送人。"""

    GOOD = "https://ai-tutor-demo.up.railway.app"

    def test_a_plain_https_url_is_accepted(self):
        normalized, problem = admin_invite.validate_base_url(self.GOOD)

        self.assertEqual(normalized, self.GOOD)
        self.assertEqual(problem, "")

    def test_a_trailing_slash_is_normalized_away(self):
        normalized, _ = admin_invite.validate_base_url(self.GOOD + "/")
        self.assertEqual(normalized, self.GOOD)

    def test_an_explicit_port_is_kept_and_shown(self):
        normalized, _ = admin_invite.validate_base_url("https://example.com:8443")
        self.assertEqual(normalized, "https://example.com:8443")

    def test_http_is_rejected(self):
        for raw in ("http://example.com", "HTTP://example.com", "ftp://example.com",
                    "//example.com", "example.com"):
            with self.subTest(raw=raw):
                normalized, problem = admin_invite.validate_base_url(raw)
                self.assertIsNone(normalized, "明文/缺协议的地址被放行了")
                self.assertTrue(problem)

    def test_misleading_forms_are_rejected(self):
        """用户名密码 / 查询串 / 片段 / 路径 —— 都会让人看错真正的目标。"""
        bad = ("https://user:pass@example.com",
               "https://user@example.com",
               "https://example.com/?x=1",
               "https://example.com?x=1",
               "https://example.com/#frag",
               "https://example.com/admin",
               "https://exam ple.com",
               "https://localhost",
               "https://",
               "   ")
        for raw in bad:
            with self.subTest(raw=raw):
                normalized, problem = admin_invite.validate_base_url(raw)
                self.assertIsNone(normalized, "这种形式不该被接受：" + raw)
                self.assertTrue(problem, "拒绝了但没给原因")

    def test_no_token_is_asked_for_before_the_url_is_valid(self):
        """【核心】地址不合法时：不许索取令牌、不许生成码、不许发请求。"""
        asked = []
        printed = []

        code = admin_invite.main(
            ["new", "--base-url", "http://insecure.example"],
            ask_token=lambda label: asked.append(label) or "不该被问到",
            ask_secret=lambda _l: "x",
            ask_confirm=lambda _l: True,
            say=printed.append,
            post=lambda *a, **k: self.fail("地址非法却还是发了请求"))

        self.assertEqual(code, 4)
        self.assertEqual(asked, [], "地址还没验就先问了令牌")
        self.assertIn("不合法", "\n".join(printed))

    def test_the_final_target_is_shown_before_the_token_is_asked(self):
        """【核心】必须把最终域名清楚地打给操作者看。"""
        order = []
        printed = []

        admin_invite.main(
            ["new", "--base-url", self.GOOD],
            ask_token=lambda label: order.append("token") or "",
            ask_secret=lambda _l: "",
            ask_confirm=lambda _l: True,
            say=printed.append,
            post=lambda *a, **k: (200, {"result": "created"}))

        text = "\n".join(printed)
        self.assertIn("ai-tutor-demo.up.railway.app", text, "没有把目标域名打出来")
        self.assertIn("令牌只会发往这个域名", text)
        self.assertTrue(order, "根本没问令牌，测试前提不成立")


# ===================== 8. 重定向：绝不把令牌转走 =====================

class _FakeResponse:
    """冒充一次正常的 HTTP 响应（测试用，不联网）。"""

    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _RecordingOpener:
    """记录收到的每一个请求，并按脚本要求回一个状态（不联网）。"""

    def __init__(self, status, body=b"", location=None):
        self.status = status
        self.body = body
        self.location = location
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        if 300 <= self.status < 400:
            headers = {"Location": self.location} if self.location else {}
            raise admin_invite.urllib.error.HTTPError(
                request.full_url, self.status, "redirect", headers, None)
        return _FakeResponse(self.status, self.body.decode("utf-8"))


class TestRedirectSafety(unittest.TestCase):
    """【核心】urllib 默认会跟随 3xx，而它的重定向处理会【复制自定义请求头】——
    也就是说 `X-Admin-Token` 会被原样发到新地址去。所以必须禁止跟随。"""

    def test_the_real_opener_refuses_to_follow_any_redirect(self):
        opener = admin_invite.build_opener()

        handlers = [h for h in opener.handlers
                    if isinstance(h, admin_invite.NoRedirectHandler)]
        self.assertTrue(handlers, "opener 里没有装「不跟随重定向」的处理器")

        request = admin_invite.urllib.request.Request("https://good.example/admin/invites")
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                self.assertIsNone(
                    handlers[0].redirect_request(request, None, code, "x", {}, "https://evil.example/"),
                    "处理器居然同意跟随跳转（那会把令牌转走）")

    def test_a_cross_host_redirect_is_blocked_and_never_succeeds(self):
        opener = _RecordingOpener(302, location="https://evil.example/admin/invites")

        status, payload = admin_invite.http_post(
            "https://good.example", "/admin/invites", {"code": "c"}, "token", opener=opener)
        outcome = admin_invite.classify_register(status, payload)

        self.assertEqual(outcome, admin_invite.OUT_REDIRECT_BLOCKED)
        # 【只有一次请求】重定向目标从来没有被请求过 —— 令牌也就没被转过去
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(opener.requests[0].full_url, "https://good.example/admin/invites")

    def test_a_same_host_redirect_is_also_blocked(self):
        """同域跳转同样不跟随：我们自己的服务端不该回 3xx，回了就说明有问题。"""
        opener = _RecordingOpener(307, location="https://good.example/other")

        status, payload = admin_invite.http_post(
            "https://good.example", "/admin/invites", {"code": "c"}, "token", opener=opener)
        outcome = admin_invite.classify_register(status, payload)

        self.assertEqual(outcome, admin_invite.OUT_REDIRECT_BLOCKED)
        self.assertEqual(len(opener.requests), 1)

    def test_a_redirect_result_never_says_success_and_tells_you_what_to_do(self):
        for result in (admin_invite.OUT_REDIRECT_BLOCKED,):
            text, code = admin_invite.register_message(result, "c")
            with self.subTest(result=result):
                self.assertEqual(code, 3, "3xx 绝不能被当成成功")
                self.assertNotIn("✅", text)
                self.assertIn("令牌没有被转发", text)
                self.assertIn("状态未知", text)
                self.assertIn("同一张码", text)

    def test_the_token_only_ever_goes_to_the_validated_host(self):
        """端到端（离线）：整个流程只发出一个请求，令牌只出现在它里面。"""
        opener = _RecordingOpener(302, location="https://evil.example/admin/invites")
        printed = []
        # 【注意第一个确认是「这是重试吗？」】答 True 会走重试路径、去要旧码，
        # 那样根本走不到发送。所以这里要按顺序给：不是重试 → 确认发送。
        confirms = iter([False, True])

        code = admin_invite.main(
            ["new", "--base-url", "https://good.example"],
            ask_token=lambda _l: "TOKEN-VALUE-supersecret",
            ask_secret=lambda _l: "",
            ask_confirm=lambda _l: next(confirms, False),
            say=printed.append,
            post=lambda base, path, fields, token, timeout=30: admin_invite.http_post(
                base, path, fields, token, opener=opener))

        self.assertEqual(code, 3)
        self.assertEqual(len(opener.requests), 1)
        sent = opener.requests[0]
        self.assertEqual(sent.full_url, "https://good.example/admin/invites")
        self.assertEqual(sent.get_header("X-admin-token"), "TOKEN-VALUE-supersecret")
        # 跳转目标那个域名，从来没有收到过任何东西
        self.assertNotIn("evil.example", "\n".join(
            r.full_url for r in opener.requests))


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_admin_invite.py 也能跑
