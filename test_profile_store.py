# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# profile_store.py（学习档案存储层）的自动化测试
#
# 【原则】完全离线：不联网、不调模型、不需要任何密钥。
# 测试自己建一个临时数据库文件，跑完就删 —— 绝不会碰到你本地的 chat.db。
#
# 【这一层测什么】它不认识 Flask，所以这里也不起网页：
# 直接喂参数、直接查表。网页那一层的测试在 test_app.py 里。
#
# 运行方式（在 ai_tutor 文件夹里）：
#     python -m unittest test_profile_store -v
# =====================================================================

import io          # 抓命令行的输出
import os          # 拼路径、设环境变量
import contextlib  # 把命令行的打印重定向到内存里
import sqlite3     # 直接查表，验证数据真的写对了
import tempfile    # 临时文件夹，测试用独立的数据库
import threading   # 并发兑换邀请码那一条要用
import unittest    # Python 自带的测试框架

import profile_store   # 被测对象


PEPPER = "unit-test-pepper-不要用在真实环境"      # 测试专用的 pepper，和真实密钥无关


class StoreTestCase(unittest.TestCase):
    """所有测试的公共部分：建一个临时库、开好表、提供几个小工具。"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "store.db")
        self.conn = self._open()
        profile_store.ensure_schema(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def _open(self):
        """开一个连接。**必须开外键**，否则 CASCADE 不会生效。"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ---------- 小工具 ----------

    def make_invite(self, code="test-invite-code", status=None):
        """登记一张邀请码。

        【走真实的 create_invite，而不是自己拼 SQL】
        自己拼 SQL 会让测试绕过产品代码那条路 —— 万一 create_invite 里
        写错了（比如存了明文），测试反而发现不了。所以这里必须走真函数。
        """
        if status is None or status == profile_store.INVITE_ACTIVE:
            self.assertTrue(profile_store.create_invite(self.conn, code, PEPPER))
        else:
            # 需要造「已作废」这种状态时，先正常建出来，再改状态
            self.assertTrue(profile_store.create_invite(self.conn, code, PEPPER))
            with self.conn:
                self.conn.execute("UPDATE invites SET status = ?", (status,))

    def count(self, table):
        return self.conn.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]

    def create_messages_table(self):
        """建一张最小的 messages 表。

        【为什么测试要自己建】messages / api_usage 属于 app.py 那一层，
        profile_store 从不管它们（这是刻意的分层）。但有几条测试要证明
        「作废 / 清空都【没有】碰聊天记录」，那就得先有一张 messages 表可比对。
        """
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  session_id TEXT NOT NULL, role TEXT NOT NULL,"
                "  content TEXT NOT NULL, created_at TEXT NOT NULL)")

    def add_message(self, session_id, content="一句话"):
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, "user", content, "2026-01-01T00:00:00"))

    def redeem(self, code, conn=None):
        return profile_store.redeem_invite(conn or self.conn, code, PEPPER)


# ===================== 1. 邀请码摘要 =====================

class TestInviteDigest(StoreTestCase):

    def test_same_code_gives_the_same_digest(self):
        """同一个码永远算出同一个摘要 —— 这正是「能按摘要直接查」的前提。"""
        self.assertEqual(profile_store.digest_invite_code("abc", PEPPER),
                         profile_store.digest_invite_code("abc", PEPPER))

    def test_different_codes_give_different_digests(self):
        self.assertNotEqual(profile_store.digest_invite_code("abc", PEPPER),
                            profile_store.digest_invite_code("abd", PEPPER))

    def test_different_pepper_gives_a_different_digest(self):
        """换了 pepper，摘要就全变了 —— 所以换 pepper 会让已发出的码失效。"""
        self.assertNotEqual(profile_store.digest_invite_code("abc", PEPPER),
                            profile_store.digest_invite_code("abc", "另一个 pepper"))

    def test_digest_is_not_the_plaintext(self):
        """摘要里绝不能含有邀请码本身。"""
        code = "super-secret-invite"
        digest = profile_store.digest_invite_code(code, PEPPER)
        self.assertNotIn(code, digest)
        self.assertEqual(len(digest), 64, "HMAC-SHA256 的十六进制结果是 64 个字符")

    def test_generated_codes_are_random_and_long(self):
        """生成器要给出高熵的码：每次都不同，而且足够长。"""
        codes = {profile_store.generate_invite_code() for _ in range(20)}
        self.assertEqual(len(codes), 20, "生成的邀请码重复了")
        for code in codes:
            self.assertGreaterEqual(len(code), 24)


# ===================== 2. 兑换：四种结果 =====================

class TestRedeem(StoreTestCase):

    def test_unknown_code_changes_nothing(self):
        """【核心安全断言】未知邀请码绝不能「先建个号再说」。"""
        outcome, learner_id = self.redeem("我根本不存在")

        self.assertEqual(outcome, profile_store.REDEEM_INVALID)
        self.assertIsNone(learner_id)
        self.assertEqual(self.count("learners"), 0, "未知邀请码创建了学习者")

    def test_active_code_creates_exactly_one_learner(self):
        self.make_invite("good-code")
        outcome, learner_id = self.redeem("good-code")

        self.assertEqual(outcome, profile_store.REDEEM_NEW)
        self.assertIsNotNone(learner_id)
        self.assertEqual(self.count("learners"), 1)

        # 邀请码要变成「已绑定」，并指向那个学习者
        status, bound = self.conn.execute(
            "SELECT status, learner_id FROM invites").fetchone()
        self.assertEqual(status, profile_store.INVITE_BOUND)
        self.assertEqual(bound, learner_id)

    def test_using_the_same_code_again_returns_the_same_learner(self):
        """换设备、清 cookie 之后重新输码 → 回到【原来看那个人】，不新建。"""
        self.make_invite("good-code")
        _outcome, first = self.redeem("good-code")
        outcome, second = self.redeem("good-code")

        self.assertEqual(outcome, profile_store.REDEEM_EXISTING)
        self.assertEqual(first, second)
        self.assertEqual(self.count("learners"), 1, "重复使用同一个码又建了一个学习者")

    def test_revoked_code_is_refused(self):
        self.make_invite("dead-code", status=profile_store.INVITE_REVOKED)
        outcome, learner_id = self.redeem("dead-code")

        self.assertEqual(outcome, profile_store.REDEEM_REVOKED)
        self.assertIsNone(learner_id)
        self.assertEqual(self.count("learners"), 0)

    def test_concurrent_redeem_creates_only_one_learner(self):
        """【核心】两个人同时用同一个码 → 只能建出一个学习者。

        靠的是 redeem_invite 里的 BEGIN IMMEDIATE：它立刻拿写锁，
        把并发的两次兑换串起来。少了它，两边都会读到「这个码还没被用过」。
        """
        self.make_invite("shared-code")
        results = []
        guard = threading.Lock()

        def worker():
            conn = self._open()
            try:
                outcome = profile_store.redeem_invite(conn, "shared-code", PEPPER)
            finally:
                conn.close()
            with guard:
                results.append(outcome)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(self.count("learners"), 1, "并发兑换建出了不止一个学习者")
        self.assertEqual(len({r[1] for r in results}), 1, "两人拿到了不同的 learner_id")


# ===================== 3. 邀请码不能明文进库 =====================

class TestInviteIsNotStoredInPlaintext(StoreTestCase):

    def test_the_plaintext_appears_nowhere_in_the_invites_table(self):
        """【核心安全断言】数据库里只有摘要，没有明文。"""
        code = "plaintext-must-not-appear-9f3a"
        self.make_invite(code)

        dump = " ".join(str(row) for row in
                        self.conn.execute("SELECT * FROM invites").fetchall())

        self.assertNotIn(code, dump, "邀请码明文被存进数据库了")

    def test_only_the_digest_is_stored_and_it_matches_the_pepper(self):
        self.make_invite("my-code")
        digest = self.conn.execute("SELECT code_digest FROM invites").fetchone()[0]

        self.assertEqual(digest, profile_store.digest_invite_code("my-code", PEPPER))
        self.assertNotEqual(digest, "my-code")


# ===================== 3b. 登记与作废 =====================

class TestCreateAndRevoke(StoreTestCase):

    def test_create_then_redeem_round_trip(self):
        """走真实路径：登记一张码 → 用它换到一个学习者。"""
        self.assertTrue(profile_store.create_invite(self.conn, "round-trip", PEPPER))
        outcome, learner = self.redeem("round-trip")

        self.assertEqual(outcome, profile_store.REDEEM_NEW)
        self.assertIsNotNone(learner)

    def test_creating_the_same_code_twice_is_refused(self):
        """同一个码登记两次 → 第二次返回 False，库里仍然只有一行。"""
        self.assertTrue(profile_store.create_invite(self.conn, "dup", PEPPER))
        self.assertFalse(profile_store.create_invite(self.conn, "dup", PEPPER))
        self.assertEqual(self.count("invites"), 1)

    def test_create_stores_only_the_digest(self):
        """登记之后，明文不该出现在库里任何地方。"""
        profile_store.create_invite(self.conn, "only-digest-please", PEPPER)

        dump = " ".join(str(r) for r in self.conn.execute("SELECT * FROM invites"))
        self.assertNotIn("only-digest-please", dump)

    def test_revoke_marks_the_code_and_stops_redemption(self):
        profile_store.create_invite(self.conn, "to-revoke", PEPPER)
        result = profile_store.revoke_invite(self.conn, "to-revoke", PEPPER)

        self.assertTrue(result["revoked"])
        outcome, learner = self.redeem("to-revoke")
        self.assertEqual(outcome, profile_store.REDEEM_REVOKED)
        self.assertIsNone(learner)
        self.assertEqual(self.count("learners"), 0)

    def test_revoking_an_unknown_code_reports_false(self):
        result = profile_store.revoke_invite(self.conn, "从没登记过", PEPPER)

        self.assertFalse(result["revoked"])
        self.assertIsNone(result["learner_id"])
        self.assertEqual(result["sessions_dropped"], 0)

    def test_a_bound_code_that_is_later_revoked_is_refused(self):
        """【顺序是刻意的】已绑定的码被作废之后 → 拒绝（不是「回到原来那个人」）。

        设计文档里兑换的判定顺序是：
            查不到 → revoked → 已绑定 → 未使用且有效
        也就是说 **revoked 排在「已绑定」前面**，作废就是作废，谁也进不来。

        【这个取舍要说清楚】代价是：用户如果清掉了 cookie，而那张码又被作废了，
        他就再也回不到自己的档案了。换来的是「作废」这个动作是真的能止血的 ——
        码泄露时，作废必须能立刻让所有拿着它的人失效，而不是只挡住没进来过的人。
        真要找回档案，得等以后的「补发邀请码」功能（设计文档里提到，本轮没做）。
        """
        profile_store.create_invite(self.conn, "bound-then-revoked", PEPPER)
        _o, first = self.redeem("bound-then-revoked")
        self.assertIsNotNone(first)
        profile_store.revoke_invite(self.conn, "bound-then-revoked", PEPPER)

        outcome, second = self.redeem("bound-then-revoked")
        self.assertEqual(outcome, profile_store.REDEEM_REVOKED)
        self.assertIsNone(second)
        self.assertEqual(self.count("learners"), 1, "学习者不该因为码被作废而消失")


# ===================== 3c. 作废的真实语义 =====================
#
# 【这一组守什么】「作废」如果只改一个状态字段，那是**假的作废**：
# 已经进来的浏览器照样能一直看档案。所以这里必须证明两件事：
#   ① 作废会真的断开已绑定的会话（现有设备立刻失去访问）
#   ② 数据没有被销毁，本人补发新码后能拿回原来的档案

class TestRevokeSemantics(StoreTestCase):

    def bound_learner(self, code="c1", session="device-1"):
        """造一个「已经输过码、并且保存了偏好、还开着一个会话」的学习者。"""
        profile_store.create_invite(self.conn, code, PEPPER)
        _o, learner = self.redeem(code)
        profile_store.link_session(self.conn, learner, session)
        profile_store.save_preferences(self.conn, learner, "b2", False,
                                       "en_only", "brief", goal_code="exam")
        return learner

    def test_revoke_actually_cuts_off_the_existing_device(self):
        """【核心】作废之后，已绑定设备的会话立刻查不到 learner —— 这才叫失效。"""
        learner = self.bound_learner()
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "device-1"), learner)

        result = profile_store.revoke_invite(self.conn, "c1", PEPPER)

        self.assertTrue(result["revoked"])
        self.assertEqual(result["learner_id"], learner)
        self.assertEqual(result["sessions_dropped"], 1)
        self.assertIsNone(profile_store.learner_id_for_session(self.conn, "device-1"),
                          "作废之后设备仍然能进 —— 那是假作废")

    def test_revoke_does_not_destroy_any_data(self):
        """【核心】断开通道 ≠ 销毁数据：学习者、偏好、聊天记录都还在。"""
        learner = self.bound_learner()
        self.create_messages_table()
        with self.conn:
            self.add_message("device-1", "我之前问过的话")
        before_prefs = profile_store.get_preferences(self.conn, learner)

        profile_store.revoke_invite(self.conn, "c1", PEPPER)

        self.assertEqual(self.count("learners"), 1, "学习者被删了")
        self.assertEqual(profile_store.get_preferences(self.conn, learner), before_prefs,
                         "偏好被动了")
        self.assertEqual(self.count("messages"), 1, "聊天记录被删了")

    def test_revoking_can_leave_sessions_alone_when_asked(self):
        """有 invalidate_sessions=False 这条路 —— 用于「只是不想再发这张码」的场合。"""
        learner = self.bound_learner()

        result = profile_store.revoke_invite(self.conn, "c1", PEPPER, invalidate_sessions=False)

        self.assertTrue(result["revoked"])
        self.assertEqual(result["sessions_dropped"], 0)
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "device-1"), learner,
                         "说不清会话，结果还是被断了")

    def test_revoking_an_unused_code_affects_nobody(self):
        """没人用过的码被作废 → 没有学习者受影响，不报错。"""
        profile_store.create_invite(self.conn, "never-used", PEPPER)
        result = profile_store.revoke_invite(self.conn, "never-used", PEPPER)

        self.assertTrue(result["revoked"])
        self.assertIsNone(result["learner_id"])
        self.assertEqual(result["sessions_dropped"], 0)

    def test_revoking_one_code_does_not_touch_another_learner(self):
        """【核心隔离断言】作废 A 的码，绝不能波及 B 的设备。"""
        learner_a = self.bound_learner("code-a", "device-a")
        learner_b = self.bound_learner("code-b", "device-b")

        profile_store.revoke_invite(self.conn, "code-a", PEPPER)

        self.assertIsNone(profile_store.learner_id_for_session(self.conn, "device-a"))
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "device-b"), learner_b,
                         "作废 A 的码把 B 也踢下线了")
        self.assertIsNotNone(profile_store.get_preferences(self.conn, learner_a))

    # ---------- 补发原语（⚠️ 已停用，没有对外入口，见 TestReissueIsDisabled） ----------
    #
    # 下面这几条测的是 issue_replacement_invite 这个**零件本身**的行为，
    # 不是「产品现在能帮用户找回档案」—— 那条路已经关了（没有身份核验）。
    # 留着它们是为了：将来有了安全的核验流程，这块逻辑已经有测试兜着。

    def test_a_replacement_invite_brings_the_user_back_to_the_same_profile(self):
        """（零件行为）作废 → 补发 → 输新码 → 回到**原来那份档案**。

        ⚠️ 目前没有任何命令行或网页入口能走到这里，见 TestReissueIsDisabled。
        """
        learner = self.bound_learner()

        profile_store.revoke_invite(self.conn, "c1", PEPPER)          # 作废（含断会话）
        new_code = profile_store.issue_replacement_invite(self.conn, learner, PEPPER)
        self.assertIsNotNone(new_code)

        # 用户在【另一台设备】上输新码
        outcome, back = profile_store.redeem_invite(self.conn, new_code, PEPPER)

        self.assertEqual(outcome, profile_store.REDEEM_EXISTING, "补发的码应该回到原学习者")
        self.assertEqual(back, learner)
        self.assertEqual(self.count("learners"), 1, "补发时又建了一个新学习者")
        prefs = profile_store.get_preferences(self.conn, learner)
        self.assertEqual(prefs["level_code"], "b2", "档案没能拿回来")
        self.assertEqual(prefs["goal_code"], "exam")

    def test_a_replacement_invite_is_stored_as_a_digest_too(self):
        learner = self.bound_learner()
        new_code = profile_store.issue_replacement_invite(self.conn, learner, PEPPER)

        dump = " ".join(str(r) for r in self.conn.execute("SELECT * FROM invites"))
        self.assertNotIn(new_code, dump, "补发的码被明文存下来了")

    def test_reissue_for_an_unknown_learner_is_refused(self):
        """不能凭空给一个不存在的学习者补发。"""
        self.assertIsNone(profile_store.issue_replacement_invite(self.conn, 999, PEPPER))

    def test_a_replacement_invite_can_be_revoked_in_turn(self):
        """补发的码同样可以被作废（不然就等于留了一条永久通道）。"""
        learner = self.bound_learner()
        new_code = profile_store.issue_replacement_invite(self.conn, learner, PEPPER)

        result = profile_store.revoke_invite(self.conn, new_code, PEPPER)

        self.assertTrue(result["revoked"])
        outcome, _ = self.redeem(new_code)
        self.assertEqual(outcome, profile_store.REDEEM_REVOKED)


# ===================== 3d. 清空学习档案（只清偏好）=====================

class TestClearPreferences(StoreTestCase):

    def make_learner_with_profile(self, code="c1", session="device-1"):
        profile_store.create_invite(self.conn, code, PEPPER)
        _o, learner = self.redeem(code)
        profile_store.link_session(self.conn, learner, session)
        profile_store.save_preferences(self.conn, learner, "b2", False,
                                       "en_only", "brief", goal_code="work",
                                       focus_code="speaking")
        return learner

    def test_clear_removes_only_the_preferences(self):
        """【核心】清完之后：偏好没了，学习者/邀请码/会话/聊天记录全都在。"""
        learner = self.make_learner_with_profile()
        self.create_messages_table()
        with self.conn:
            self.add_message("device-1")

        self.assertTrue(profile_store.clear_preferences(self.conn, learner))

        self.assertIsNone(profile_store.get_preferences(self.conn, learner), "偏好没清干净")
        self.assertEqual(self.count("learners"), 1, "学习者被删了 —— 这只该清偏好")
        self.assertEqual(self.count("invites"), 1, "邀请码被牵连了")
        self.assertEqual(self.count("learner_sessions"), 1, "会话绑定被牵连了")
        self.assertEqual(self.count("messages"), 1, "聊天记录被删了")
        # 同一张码仍然能进（清空学习档案不该让用户失去入口）
        outcome, again = self.redeem("c1")
        self.assertEqual(outcome, profile_store.REDEEM_EXISTING)
        self.assertEqual(again, learner)

    def test_clearing_twice_reports_false_the_second_time(self):
        """清空是幂等的：第二次没有东西可清，返回 False（不是报错）。"""
        learner = self.make_learner_with_profile()
        self.assertTrue(profile_store.clear_preferences(self.conn, learner))
        self.assertFalse(profile_store.clear_preferences(self.conn, learner))

    def test_clearing_without_a_learner_does_nothing(self):
        self.assertFalse(profile_store.clear_preferences(self.conn, None))

    def test_clearing_one_learner_never_touches_another(self):
        """【核心隔离断言】A 清空自己的档案，B 的档案一个字都不能变。"""
        learner_a = self.make_learner_with_profile("code-a", "device-a")
        learner_b = self.make_learner_with_profile("code-b", "device-b")
        b_before = profile_store.get_preferences(self.conn, learner_b)

        profile_store.clear_preferences(self.conn, learner_a)

        self.assertIsNone(profile_store.get_preferences(self.conn, learner_a))
        self.assertEqual(profile_store.get_preferences(self.conn, learner_b), b_before,
                         "清空 A 的时候把 B 的档案也清了")

    def test_the_user_can_save_a_new_profile_after_clearing(self):
        """清空不是封禁：他还能重新填一份。"""
        learner = self.make_learner_with_profile()
        profile_store.clear_preferences(self.conn, learner)

        self.assertTrue(profile_store.save_preferences(
            self.conn, learner, "a2", False, "zh_pair", "detailed"))

        prefs = profile_store.get_preferences(self.conn, learner)
        self.assertEqual(prefs["level_code"], "a2")
        self.assertIsNone(prefs["goal_code"], "旧的目标不该残留")


# ===================== 3f. 清空全部个人数据 =====================
#
# 【这一组要证明什么】
#   ① 该删的一条不漏：多设备、以及【已经被作废过的旧会话】的聊天记录
#   ② 不该碰的一条不动：别的学习者、匿名访客、全站 api_usage
#   ③ 同一事务：中途失败必须整体回滚，不能留下删了一半的状态

class TestClearAllData(StoreTestCase):

    def setUp(self):
        super().setUp()
        self.create_messages_table()
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS api_usage (day TEXT PRIMARY KEY, used INTEGER)")

    def add_quota(self, day="2026-01-01", used=7):
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO api_usage (day, used) VALUES (?, ?)",
                              (day, used))

    def make_learner(self, code="c1"):
        profile_store.create_invite(self.conn, code, PEPPER)
        _o, learner = self.redeem(code)
        return learner

    def chat(self, session_id, text="一句话"):
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, "user", text, "2026-01-01T00:00:00"))

    # ---------- 该删的一条不漏 ----------

    def test_it_deletes_everything_in_scope(self):
        """【核心】聊天记录、偏好、会话绑定、学习者身份全删，邀请码作废。"""
        learner = self.make_learner()
        profile_store.link_session(self.conn, learner, "device-1")
        profile_store.save_preferences(self.conn, learner, "b2", False, "en_only", "brief")
        self.chat("device-1")

        stats = profile_store.clear_all_data(self.conn, learner)

        self.assertEqual(stats["sessions"], 1)
        self.assertEqual(stats["messages"], 1)
        self.assertEqual(stats["preferences"], 1)
        self.assertEqual(stats["invites"], 1)
        self.assertEqual(stats["learners"], 1)

        self.assertEqual(self.count("messages"), 0)
        self.assertEqual(self.count("learner_preferences"), 0)
        self.assertEqual(self.count("learner_sessions"), 0)
        self.assertEqual(self.count("learners"), 0)

    def test_it_covers_every_device(self):
        """【核心】同一个学习者名下多个设备 → 每一台的聊天记录都要删。"""
        learner = self.make_learner()
        for name in ("phone", "laptop", "tablet"):
            profile_store.link_session(self.conn, learner, name)
            self.chat(name, name + " 上问的话")

        stats = profile_store.clear_all_data(self.conn, learner)

        self.assertEqual(stats["sessions"], 3)
        self.assertEqual(stats["messages"], 3)
        self.assertEqual(self.count("messages"), 0)

    def test_it_covers_sessions_that_were_revoked_earlier(self):
        """【核心】这一条是修正过的关键场景。

        用户先被作废过一次（某个设备失去访问权），过一阵才来要求清空全部数据。
        旧实现作废时把绑定那一行【删掉】了，于是这个会话和学习的关联也没了 ——
        清空的时候根本找不到它的聊天记录，用户以为删干净了，其实还躺在库里。
        现在作废只打标记，所以关联还在，这条测试就是钉住这一点。
        """
        learner = self.make_learner()
        profile_store.link_session(self.conn, learner, "old-device")
        self.chat("old-device", "作废之前问的话")

        # 作废：old-device 失去访问权，但那一行的关联必须留着
        profile_store.revoke_invite(self.conn, "c1", PEPPER)
        self.assertIsNone(profile_store.learner_id_for_session(self.conn, "old-device"))
        self.assertEqual(profile_store.session_ids_for_learner(self.conn, learner),
                         ["old-device"], "作废把会话关联也弄丢了")

        # 他后来在另一台设备上重新进来（用补发码的机制，这里直接绑）
        profile_store.link_session(self.conn, learner, "new-device")
        self.chat("new-device", "后来问的话")

        stats = profile_store.clear_all_data(self.conn, learner)

        self.assertEqual(stats["sessions"], 2, "没有把已失效的旧会话算进来")
        self.assertEqual(stats["messages"], 2, "旧会话的聊天记录被漏掉了")
        self.assertEqual(self.count("messages"), 0)

    def test_revoking_marks_instead_of_deleting_the_row(self):
        """作废是打标记：行还在、访问权没了 —— 这正是上面那条能成立的前提。"""
        learner = self.make_learner()
        profile_store.link_session(self.conn, learner, "device-1")

        profile_store.revoke_invite(self.conn, "c1", PEPPER)

        self.assertEqual(self.count("learner_sessions"), 1, "作废把行删了")
        row = self.conn.execute(
            "SELECT learner_id, revoked_at FROM learner_sessions WHERE session_id = 'device-1'"
        ).fetchone()
        self.assertEqual(row[0], learner)
        self.assertIsNotNone(row[1], "没有打上失效标记")
        self.assertIsNone(profile_store.learner_id_for_session(self.conn, "device-1"))

    def test_the_invite_is_revoked_not_deleted(self):
        """邀请码改成 revoked，不删行 —— 那是「谁用过」的痕迹。"""
        learner = self.make_learner()
        profile_store.clear_all_data(self.conn, learner)

        row = self.conn.execute("SELECT status, learner_id FROM invites").fetchone()
        self.assertEqual(row[0], profile_store.INVITE_REVOKED)
        self.assertIsNone(row[1], "作废后不该再指向已经删掉的学习者")
        self.assertEqual(self.count("invites"), 1)

    def test_a_revoked_invite_cannot_be_used_after_the_wipe(self):
        learner = self.make_learner()
        profile_store.clear_all_data(self.conn, learner)

        outcome, new_learner = self.redeem("c1")
        self.assertEqual(outcome, profile_store.REDEEM_REVOKED)
        self.assertIsNone(new_learner)
        self.assertEqual(self.count("learners"), 0)

    # ---------- 不该碰的一条不动 ----------

    def test_another_learner_is_untouched(self):
        """【核心隔离断言】删 A 的全部数据，B 的一切原封不动。"""
        learner_a = self.make_learner("code-a")
        learner_b = self.make_learner("code-b")
        profile_store.link_session(self.conn, learner_a, "device-a")
        profile_store.link_session(self.conn, learner_b, "device-b")
        profile_store.save_preferences(self.conn, learner_b, "c1", False, "en_advanced", "detailed")
        self.chat("device-a", "A 的话")
        self.chat("device-b", "B 的话")

        profile_store.clear_all_data(self.conn, learner_a)

        self.assertEqual(self.count("learners"), 1)
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "device-b"), learner_b)
        self.assertIsNotNone(profile_store.get_preferences(self.conn, learner_b))
        self.assertEqual(self.count("messages"), 1, "把 B 的聊天记录也删了")
        self.assertEqual(
            self.conn.execute("SELECT content FROM messages").fetchone()[0], "B 的话")

    def test_anonymous_visitors_are_untouched(self):
        """匿名访客的聊天记录没有主人 —— 删某个学习者时绝不能顺手带走。"""
        learner = self.make_learner()
        profile_store.link_session(self.conn, learner, "device-1")
        self.chat("device-1", "学习者的记录")
        self.chat("someone-without-a-learner", "匿名访客的记录")

        profile_store.clear_all_data(self.conn, learner)

        self.assertEqual(self.count("messages"), 1)
        self.assertEqual(
            self.conn.execute("SELECT content FROM messages").fetchone()[0], "匿名访客的记录")

    def test_the_global_quota_is_never_touched(self):
        """【核心】不能清空全站额度 —— 那是全站共用的计数器，不属于任何个人。"""
        self.add_quota(used=7)
        learner = self.make_learner()
        profile_store.link_session(self.conn, learner, "device-1")

        profile_store.clear_all_data(self.conn, learner)

        self.assertEqual(
            self.conn.execute("SELECT used FROM api_usage WHERE day='2026-01-01'").fetchone()[0],
            7, "全站额度被清空了")

    # ---------- 事务边界 ----------

    def test_a_failure_halfway_rolls_everything_back(self):
        """【核心】第三步失败（用触发器故意让它炸）→ 前两步必须一起回滚。

        【为什么值得这么测】「删了一半」是最糟的结果：用户以为删干净了，
        实际聊天记录还在，而偏好已经没了 —— 数据处于谁都说不清的状态。
        """
        learner = self.make_learner()
        profile_store.link_session(self.conn, learner, "device-1")
        profile_store.save_preferences(self.conn, learner, "b2", False, "en_only", "brief")
        self.chat("device-1")

        # 装一个「删偏好就报错」的触发器，把失败点精确地放在③
        with self.conn:
            self.conn.execute(
                "CREATE TRIGGER boom BEFORE DELETE ON learner_preferences "
                "BEGIN SELECT RAISE(ABORT, '演示用的故障'); END")

        with self.assertRaises(sqlite3.IntegrityError):
            profile_store.clear_all_data(self.conn, learner)

        # ①②已经执行过了，但必须被整体撤销
        self.assertEqual(self.count("messages"), 1, "聊天记录被删了却没回滚")
        self.assertEqual(self.count("learner_preferences"), 1, "偏好被删了却没回滚")
        self.assertEqual(self.count("learner_sessions"), 1)
        self.assertEqual(self.count("learners"), 1)
        self.assertEqual(
            self.conn.execute("SELECT status FROM invites").fetchone()[0],
            profile_store.INVITE_BOUND, "邀请码被作废了却没回滚")

    def test_clearing_an_unknown_learner_changes_nothing(self):
        """编号不存在 → 什么都不做，也不报错（更不能顺手把额度清了）。"""
        self.add_quota(used=7)
        stats = profile_store.clear_all_data(self.conn, 999)

        self.assertEqual(stats["learners"], 0)
        self.assertEqual(stats["messages"], 0)
        self.assertEqual(
            self.conn.execute("SELECT used FROM api_usage").fetchone()[0], 7,
            "清一个不存在的学习者居然动了全站额度")

    def test_clearing_none_is_a_no_op(self):
        self.assertIsNone(profile_store.clear_all_data(self.conn, None))

    def test_the_learner_can_start_over_with_a_brand_new_learner(self):
        """清空之后可以用一张全新的码重新开始 —— 但那是另一个人，不是原来的档案。"""
        old = self.make_learner("old-code")
        profile_store.clear_all_data(self.conn, old)

        profile_store.create_invite(self.conn, "fresh-code", PEPPER)
        outcome, fresh = self.redeem("fresh-code")

        self.assertEqual(outcome, profile_store.REDEEM_NEW)
        self.assertNotEqual(fresh, old)
        self.assertEqual(self.count("learners"), 1)


# ===================== 3g. 失效绑定的恢复规则 =====================

class TestRevokedBindingRules(StoreTestCase):

    def test_the_same_learner_can_rebind_a_revoked_session(self):
        """本人拿新码回来 → 清掉失效标记，回到同一份档案。

        （这正是「作废旧码 → 发新码 → 绑到同一个 learner」那条补救路径要用的机制。）
        """
        profile_store.create_invite(self.conn, "c1", PEPPER)
        _o, learner = self.redeem("c1")
        profile_store.link_session(self.conn, learner, "device-1")
        profile_store.revoke_invite(self.conn, "c1", PEPPER)
        self.assertIsNone(profile_store.learner_id_for_session(self.conn, "device-1"))

        self.assertEqual(profile_store.link_session(self.conn, learner, "device-1"), learner)

        self.assertEqual(profile_store.learner_id_for_session(self.conn, "device-1"), learner)
        self.assertEqual(self.count("learner_sessions"), 1, "恢复时插出了第二行")

    def test_another_learner_cannot_take_over_a_revoked_session(self):
        """【核心】失效的会话仍然属于原来那个人，别人不能用一张码把它认领走。"""
        profile_store.create_invite(self.conn, "code-a", PEPPER)
        profile_store.create_invite(self.conn, "code-b", PEPPER)
        _o, learner_a = self.redeem("code-a")
        _o, learner_b = self.redeem("code-b")
        profile_store.link_session(self.conn, learner_a, "device-1")
        profile_store.revoke_invite(self.conn, "code-a", PEPPER)

        self.assertIsNone(profile_store.link_session(self.conn, learner_b, "device-1"))

        row = self.conn.execute(
            "SELECT learner_id, revoked_at FROM learner_sessions").fetchone()
        self.assertEqual(row[0], learner_a, "会话被改绑给别人了")
        self.assertIsNotNone(row[1], "失效标记被清掉了")

    def test_an_active_binding_still_cannot_be_rebound(self):
        """有效绑定照旧不许改绑（这条是原有的安全规则，不能因为这次改动松掉）。"""
        profile_store.create_invite(self.conn, "code-a", PEPPER)
        profile_store.create_invite(self.conn, "code-b", PEPPER)
        _o, learner_a = self.redeem("code-a")
        _o, learner_b = self.redeem("code-b")
        profile_store.link_session(self.conn, learner_a, "device-1")

        self.assertEqual(profile_store.link_session(self.conn, learner_b, "device-1"), learner_a)


# ===================== 3h. 老库迁移 =====================

class TestSessionTableMigration(StoreTestCase):

    def test_an_old_table_gets_the_new_column(self):
        """老库里的 learner_sessions 没有 revoked_at → 启动时自动补上。

        【为什么必须测】CREATE TABLE IF NOT EXISTS 对已存在的表什么都不做，
        所以老库不会自己多出这一列 —— 少了迁移，线上升级后作废功能会直接报错。
        """
        import importlib
        importlib.reload(profile_store)

        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS learner_sessions")
            self.conn.execute(                      # 老结构：没有 revoked_at
                "CREATE TABLE learner_sessions ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  learner_id INTEGER NOT NULL,"
                "  session_id TEXT NOT NULL UNIQUE,"
                "  created_at TEXT NOT NULL,"
                "  last_seen_at TEXT NOT NULL)")

        profile_store.ensure_schema(self.conn)

        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(learner_sessions)")}
        self.assertIn("revoked_at", columns)

    def test_existing_bindings_stay_active_after_the_migration(self):
        """迁移之后，老数据里那些绑定仍然是「有效」的（NULL = 有效）。"""
        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS learner_sessions")
            self.conn.execute(
                "CREATE TABLE learner_sessions ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  learner_id INTEGER NOT NULL,"
                "  session_id TEXT NOT NULL UNIQUE,"
                "  created_at TEXT NOT NULL,"
                "  last_seen_at TEXT NOT NULL)")
        profile_store.create_invite(self.conn, "c1", PEPPER)
        _o, learner = self.redeem("c1")
        with self.conn:
            self.conn.execute(
                "INSERT INTO learner_sessions (learner_id, session_id, created_at, last_seen_at)"
                " VALUES (?, 'old-row', '2026-01-01T00:00:00', '2026-01-01T00:00:00')",
                (learner,))

        profile_store.ensure_schema(self.conn)          # 迁移

        self.assertEqual(profile_store.learner_id_for_session(self.conn, "old-row"), learner,
                         "老绑定在迁移后失效了")


# ===================== 3i. 兑换 + 绑定：要么都成，要么都不做 =====================
#
# 【修的是什么】旧流程「先 redeem_invite()（自己提交）+ 再 link_session()」有个半截状态：
# 第二步可能失败（会话已属于别人 → 拒绝改绑），而第一步已经落库 ——
# 结果是邀请码被白白消耗、多出一个谁都进不去的孤儿学习者，网页还报「成功」。

class TestRedeemAndBind(StoreTestCase):

    def setUp(self):
        super().setUp()
        self.create_messages_table()

    def invite_state(self, code):
        """查一张码的现状：(status, learner_id)。"""
        return self.conn.execute(
            "SELECT status, learner_id FROM invites WHERE code_digest = ?",
            (profile_store.digest_invite_code(code, PEPPER),)).fetchone()

    # ---------- 正常路径 ----------

    def test_redeem_and_bind_succeeds_together(self):
        profile_store.create_invite(self.conn, "c1", PEPPER)
        outcome, learner = profile_store.redeem_and_bind(self.conn, "c1", PEPPER, "device-1")

        self.assertEqual(outcome, profile_store.REDEEM_NEW)
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "device-1"), learner)
        self.assertEqual(self.count("learners"), 1)

    def test_redeem_and_bind_keeps_the_learner_on_re_entry(self):
        """同一个码第二次用（换设备）→ 回到原学习者，不新建。"""
        profile_store.create_invite(self.conn, "c1", PEPPER)
        _o, learner = profile_store.redeem_and_bind(self.conn, "c1", PEPPER, "device-1")
        outcome, again = profile_store.redeem_and_bind(self.conn, "c1", PEPPER, "device-2")

        self.assertEqual(outcome, profile_store.REDEEM_EXISTING)
        self.assertEqual(again, learner)
        self.assertEqual(self.count("learners"), 1)

    # ---------- 拒绝路径：必须整体回滚 ----------

    def test_binding_to_an_occupied_session_refuses_and_rolls_back(self):
        """【核心】会话已经有效绑着 A，再来兑 B 的码 → 拒绝，而且 B 的码【没被消耗】。"""
        profile_store.create_invite(self.conn, "code-a", PEPPER)
        profile_store.create_invite(self.conn, "code-b", PEPPER)
        _o, learner_a = profile_store.redeem_and_bind(self.conn, "code-a", PEPPER, "device-1")
        learners_before = self.count("learners")

        outcome, learner = profile_store.redeem_and_bind(self.conn, "code-b", PEPPER, "device-1")

        self.assertEqual(outcome, profile_store.REDEEM_BIND_REFUSED)
        self.assertIsNone(learner)
        # ① 会话仍然属于 A
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "device-1"), learner_a)
        # ② B 的码原封不动 —— 还能给别人用
        self.assertEqual(self.invite_state("code-b"),
                         (profile_store.INVITE_ACTIVE, None), "B 的码被白白消耗了")
        # ③ 没有多出孤儿学习者
        self.assertEqual(self.count("learners"), learners_before)

    def test_a_refused_code_still_works_on_a_clean_browser(self):
        """被拒绝之后，那张码给一个干净的会话仍然能用（证明回滚是干净的）。"""
        profile_store.create_invite(self.conn, "code-a", PEPPER)
        profile_store.create_invite(self.conn, "code-b", PEPPER)
        profile_store.redeem_and_bind(self.conn, "code-a", PEPPER, "device-1")
        profile_store.redeem_and_bind(self.conn, "code-b", PEPPER, "device-1")   # 被拒

        outcome, learner = profile_store.redeem_and_bind(self.conn, "code-b", PEPPER, "device-2")

        self.assertEqual(outcome, profile_store.REDEEM_NEW)
        self.assertIsNotNone(learner)

    def test_an_invalid_code_leaves_nothing_behind(self):
        outcome, learner = profile_store.redeem_and_bind(self.conn, "不存在的码", PEPPER, "device-1")

        self.assertEqual(outcome, profile_store.REDEEM_INVALID)
        self.assertIsNone(learner)
        self.assertEqual(self.count("learners"), 0)
        self.assertEqual(self.count("learner_sessions"), 0)

    def test_a_revoked_code_leaves_nothing_behind(self):
        profile_store.create_invite(self.conn, "c1", PEPPER)
        profile_store.revoke_invite(self.conn, "c1", PEPPER)

        outcome, learner = profile_store.redeem_and_bind(self.conn, "c1", PEPPER, "device-1")

        self.assertEqual(outcome, profile_store.REDEEM_REVOKED)
        self.assertIsNone(learner)
        self.assertEqual(self.count("learner_sessions"), 0)

    # ---------- 绑定那一步自己炸了：兑换也必须回滚 ----------

    def test_a_failure_during_binding_rolls_the_redemption_back(self):
        """【核心】用触发器让「插入会话绑定」那一步报错 →

        邀请码必须回到 active、学习者不能留下、异常照常往外抛（让调用方知道失败了）。
        """
        profile_store.create_invite(self.conn, "c1", PEPPER)
        with self.conn:
            self.conn.execute(
                "CREATE TRIGGER boom BEFORE INSERT ON learner_sessions "
                "BEGIN SELECT RAISE(ABORT, '演示用的故障'); END")

        with self.assertRaises(sqlite3.IntegrityError):
            profile_store.redeem_and_bind(self.conn, "c1", PEPPER, "device-1")

        self.assertEqual(self.invite_state("c1"),
                         (profile_store.INVITE_ACTIVE, None), "兑换没有被回滚")
        self.assertEqual(self.count("learners"), 0, "留下了孤儿学习者")
        self.assertEqual(self.count("learner_sessions"), 0)

    # ---------- 失效会话的恢复规则（走 redeem_and_bind 这条路） ----------

    def test_a_revoked_session_bound_to_the_same_learner_is_restored(self):
        """失效会话 + 同一个学习者的新码 → 恢复那一行，不算拒绝。"""
        profile_store.create_invite(self.conn, "c1", PEPPER)
        _o, learner = profile_store.redeem_and_bind(self.conn, "c1", PEPPER, "device-1")
        profile_store.revoke_invite(self.conn, "c1", PEPPER)
        new_code = profile_store.issue_replacement_invite(self.conn, learner, PEPPER)

        outcome, back = profile_store.redeem_and_bind(self.conn, new_code, PEPPER, "device-1")

        self.assertEqual(outcome, profile_store.REDEEM_EXISTING)
        self.assertEqual(back, learner)
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "device-1"), learner)

    def test_a_revoked_session_bound_to_someone_else_is_refused_and_rolled_back(self):
        """失效会话属于 A，却拿 B 的【新】码来兑 → 拒绝 + 回滚（B 的码留着）。"""
        profile_store.create_invite(self.conn, "code-a", PEPPER)
        profile_store.create_invite(self.conn, "code-b", PEPPER)
        _o, learner_a = profile_store.redeem_and_bind(self.conn, "code-a", PEPPER, "device-1")
        profile_store.revoke_invite(self.conn, "code-a", PEPPER)

        outcome, learner = profile_store.redeem_and_bind(self.conn, "code-b", PEPPER, "device-1")

        self.assertEqual(outcome, profile_store.REDEEM_BIND_REFUSED)
        self.assertIsNone(learner)
        self.assertEqual(self.invite_state("code-b"),
                         (profile_store.INVITE_ACTIVE, None), "B 的码被消耗了")
        self.assertEqual(self.count("learners"), 1, "多出了学习者")

        # A 的旧关联仍然留着（以后清空 A 的数据要靠它），而且没有被恢复成有效
        row = self.conn.execute(
            "SELECT learner_id, revoked_at FROM learner_sessions WHERE session_id = 'device-1'"
        ).fetchone()
        self.assertEqual(row[0], learner_a)
        self.assertIsNotNone(row[1])
        self.assertIsNone(profile_store.learner_id_for_session(self.conn, "device-1"))


# ===================== 3j. 会话失效的判定 =====================

class TestSessionIsRevoked(StoreTestCase):

    def test_an_unbound_session_is_not_revoked(self):
        self.assertFalse(profile_store.session_is_revoked(self.conn, "从没见过"))

    def test_an_active_session_is_not_revoked(self):
        profile_store.create_invite(self.conn, "c1", PEPPER)
        _o, learner = profile_store.redeem_invite(self.conn, "c1", PEPPER)
        profile_store.link_session(self.conn, learner, "device-1")

        self.assertFalse(profile_store.session_is_revoked(self.conn, "device-1"))

    def test_a_revoked_session_is_reported_as_revoked(self):
        profile_store.create_invite(self.conn, "c1", PEPPER)
        _o, learner = profile_store.redeem_invite(self.conn, "c1", PEPPER)
        profile_store.link_session(self.conn, learner, "device-1")
        profile_store.revoke_invite(self.conn, "c1", PEPPER)

        self.assertTrue(profile_store.session_is_revoked(self.conn, "device-1"))


# ===================== 3k. 登记「外部生成」的邀请码（幂等）=====================
#
# 【这一组守什么】线上发码走 HTTPS，响应可能丢。操作者手里有码，正确的做法是
# 拿同一张码重试 —— 所以「同一个码重复登记」必须是**幂等**的，而且
# 「数据库有别的完整性问题」绝不能被误报成「已经登记好了」。

class TestRegisterGeneratedInvite(StoreTestCase):

    def state(self, code):
        return self.conn.execute(
            "SELECT status, learner_id FROM invites WHERE code_digest = ?",
            (profile_store.digest_invite_code(code, PEPPER),)).fetchone()

    def test_a_fresh_code_is_registered(self):
        outcome = profile_store.register_generated_invite(self.conn, "fresh-code", PEPPER)

        self.assertEqual(outcome, profile_store.REGISTER_CREATED)
        self.assertEqual(self.state("fresh-code"), (profile_store.INVITE_ACTIVE, None))

    def test_registering_only_stores_the_digest(self):
        """【核心安全断言】明文不进库。"""
        profile_store.register_generated_invite(self.conn, "only-digest-xyz", PEPPER)

        dump = " ".join(str(r) for r in self.conn.execute("SELECT * FROM invites"))
        self.assertNotIn("only-digest-xyz", dump)

    def test_registering_the_same_code_again_is_idempotent(self):
        """【核心】同一张码重复登记 → 不产生第二行，返回「已登记过」。"""
        profile_store.register_generated_invite(self.conn, "same-code", PEPPER)
        outcome = profile_store.register_generated_invite(self.conn, "same-code", PEPPER)

        self.assertEqual(outcome, profile_store.REGISTER_ALREADY_REGISTERED)
        self.assertEqual(self.count("invites"), 1, "重复登记产生了第二行")

    def test_a_bound_code_reports_bound_not_registered(self):
        """已经被人用掉的码 → 必须如实说「已绑定」，不能报成功。"""
        profile_store.create_invite(self.conn, "c1", PEPPER)
        profile_store.redeem_invite(self.conn, "c1", PEPPER)

        outcome = profile_store.register_generated_invite(self.conn, "c1", PEPPER)

        self.assertEqual(outcome, profile_store.REGISTER_ALREADY_BOUND)
        self.assertEqual(self.count("invites"), 1)

    def test_a_revoked_code_reports_revoked_not_registered(self):
        """作废过的码不"复活"，也不能被当成新登记成功。"""
        profile_store.create_invite(self.conn, "c1", PEPPER)
        profile_store.revoke_invite(self.conn, "c1", PEPPER)

        outcome = profile_store.register_generated_invite(self.conn, "c1", PEPPER)

        self.assertEqual(outcome, profile_store.REGISTER_REVOKED)
        self.assertEqual(self.state("c1"), (profile_store.INVITE_REVOKED, None))

    def test_other_integrity_errors_are_not_reported_as_already_registered(self):
        """【核心】完整性错误 ≠ 已登记。

        用触发器制造一个「跟唯一约束无关」的完整性错误：
        这时库里的那一行【并不存在】，如果我们一律按「已登记」回，
        就等于告诉操作者"你那张码已经好了"，而它其实根本不在库里。
        """
        with self.conn:
            self.conn.execute(
                "CREATE TRIGGER boom BEFORE INSERT ON invites "
                "BEGIN SELECT RAISE(ABORT, '演示用的故障'); END")

        outcome = profile_store.register_generated_invite(self.conn, "never-inserted", PEPPER)

        self.assertEqual(outcome, profile_store.REGISTER_REJECTED)
        self.assertIsNone(self.state("never-inserted"), "居然真的插进去了")
        self.assertEqual(self.count("invites"), 0)

    def test_every_outcome_is_in_the_known_set(self):
        """返回值只能是那几个约定的状态之一（防止以后加了分支忘了登记）。"""
        profile_store.register_generated_invite(self.conn, "c1", PEPPER)
        self.assertIn(profile_store.REGISTER_CREATED, profile_store.VALID_REGISTER_OUTCOMES)
        self.assertIn(profile_store.register_generated_invite(self.conn, "c1", PEPPER),
                      profile_store.VALID_REGISTER_OUTCOMES)


# ===================== 3e. 补发命令已停用 =====================
#
# 【为什么要有这一组】「按编号补发」在没有身份核验的前提下是个危险入口：
# 编号是自增的，猜都猜得到 —— 能用编号补发，等于谁都能拿到别人的档案。
# 所以命令被停用。停用不是一个口头约定，而要有测试守着：
# 将来谁想「顺手把它打开」，会先看到这条测试红掉。

class TestReissueIsDisabled(StoreTestCase):

    def run_cli(self, argv):
        """跑一次命令行，把输出和退出码拿回来。

        【为什么要设这两个环境变量】_main 会读 INVITE_CODE_PEPPER 和 CHAT_DB_PATH。
        不设的话它会先在「缺 pepper」那一关退出 —— 那样测出来的是别的东西，
        不是「reissue 被停用」。
        """
        saved = {k: os.environ.get(k) for k in ("INVITE_CODE_PEPPER", "CHAT_DB_PATH")}
        os.environ["INVITE_CODE_PEPPER"] = PEPPER
        os.environ["CHAT_DB_PATH"] = self.db_path

        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                code = profile_store._main(argv)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        return code, buffer.getvalue()

    def test_reissue_refuses_and_explains_why(self):
        """【核心】reissue 被拒绝，而且要说明原因（不是含糊地报错）。"""
        self.make_invite("c1")
        _o, learner = self.redeem("c1")

        code, output = self.run_cli(["reissue", str(learner)])

        self.assertEqual(code, 1, "reissue 居然成功了")
        self.assertIn("已停用", output)
        self.assertIn("身份核验", output, "没有说明停用原因")

    def test_reissue_creates_no_invite_at_all(self):
        """【核心】拒绝的时候绝不能顺手建出一张码 —— 那等于门还开着。"""
        self.make_invite("c1")
        _o, learner = self.redeem("c1")
        before = self.count("invites")

        self.run_cli(["reissue", str(learner)])

        self.assertEqual(self.count("invites"), before, "reissue 被拒的同时还是建了码")

    def test_the_help_text_does_not_offer_reissue(self):
        """用法说明里不能把 reissue 列成一个可用命令。

        【注意不能传空参数】不带参数时命令行的默认动作是「生成一张新码」
        （那是刻意的，方便直接跑），所以这里要传一个看不懂的命令才会走到用法说明。
        """
        code, output = self.run_cli(["这是个不存在的命令"])

        self.assertEqual(code, 1)
        self.assertIn("new-invite", output)
        self.assertIn("revoke", output)
        self.assertNotIn("reissue <编号>", output, "用法里又把 reissue 当成可用命令列出来了")

    def test_revoke_no_longer_points_at_reissue_as_a_solution(self):
        """作废的提示里不能再写「补发一张就能回来」—— 那条路现在是断的。"""
        self.make_invite("c1")
        _o, learner = self.redeem("c1")

        code, output = self.run_cli(["revoke", "c1"])

        self.assertEqual(code, 0)
        self.assertIn("已作废", output)
        self.assertIn("无法补回", output, "作废提示没有说明「补不回来」")
        self.assertNotIn("要让本人重新获得访问权", output, "还在承诺可以恢复")

    def test_the_primitive_still_works_but_is_not_exposed(self):
        """函数本身没被删（将来有身份核验后要用），但**没有任何命令暴露它**。

        这条测试的作用是把这个状态写下来：能力留着，入口关着。
        """
        self.make_invite("c1")
        _o, learner = self.redeem("c1")

        # 直接调还能用（它是给未来的安全流程准备的零件）
        self.assertIsNotNone(profile_store.issue_replacement_invite(self.conn, learner, PEPPER))

        # 但命令行里没有一条路能走到它
        code, _output = self.run_cli(["reissue", str(learner)])
        self.assertEqual(code, 1)


# ===================== 4. 会话绑定：绝不能改绑 =====================

class TestSessionBinding(StoreTestCase):

    def test_first_link_creates_the_binding(self):
        self.make_invite("c1")
        _o, learner = self.redeem("c1")

        bound = profile_store.link_session(self.conn, learner, "session-A")

        self.assertEqual(bound, learner)
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "session-A"), learner)

    def test_an_already_bound_session_cannot_be_moved_to_another_learner(self):
        """【本轮最要紧的一条安全断言】

        如果一个已经属于 A 的 session 能被改绑到 B，
        那 B 只要再走一次邀请码流程，就能看到 A 的对话和档案。
        UNIQUE(session_id) + 「已绑过就保留原绑定」两条合起来堵住这条路。
        """
        self.make_invite("c1")
        self.make_invite("c2")
        _o, learner_a = self.redeem("c1")
        _o, learner_b = self.redeem("c2")

        profile_store.link_session(self.conn, learner_a, "shared-session")
        bound = profile_store.link_session(self.conn, learner_b, "shared-session")   # 试图改绑

        self.assertEqual(bound, learner_a, "session 被改绑到另一个人身上了")
        self.assertEqual(profile_store.learner_id_for_session(self.conn, "shared-session"),
                         learner_a)
        self.assertEqual(self.count("learner_sessions"), 1)

    def test_linking_the_same_session_twice_is_idempotent(self):
        self.make_invite("c1")
        _o, learner = self.redeem("c1")
        profile_store.link_session(self.conn, learner, "s")
        profile_store.link_session(self.conn, learner, "s")

        self.assertEqual(self.count("learner_sessions"), 1)

    def test_unknown_session_has_no_learner(self):
        """匿名访客：查不到绑定，返回 None（而不是报错、也不是随便给一个）。"""
        self.assertIsNone(profile_store.learner_id_for_session(self.conn, "没见过"))

    def test_one_learner_can_have_several_sessions(self):
        """换设备是正常的：一个学习者可以挂多个会话。"""
        self.make_invite("c1")
        _o, learner = self.redeem("c1")
        profile_store.link_session(self.conn, learner, "手机")
        profile_store.link_session(self.conn, learner, "电脑")

        self.assertEqual(self.count("learner_sessions"), 2)


# ===================== 5. 偏好：读 / 写 / 白名单 =====================

class TestPreferences(StoreTestCase):

    def make_learner(self):
        self.make_invite("c1")
        _o, learner = self.redeem("c1")
        return learner

    def test_no_preferences_yet_returns_none(self):
        """没填过 → None（而不是一套默认值）。

        【为什么这个区分重要】档案页要靠它显示「你还没设置过」。
        如果这里悄悄返回默认值，页面就只能说「这是你的设置」—— 那是假话。
        """
        learner = self.make_learner()
        self.assertIsNone(profile_store.get_preferences(self.conn, learner))
        self.assertIsNone(profile_store.get_preferences(self.conn, None))

    def test_save_then_read_back(self):
        learner = self.make_learner()
        ok = profile_store.save_preferences(
            self.conn, learner,
            level_code="b1", level_uncertain=False,
            language_mode="zh_pair", length_mode="brief",
            goal_code="exam", focus_code="writing")

        self.assertTrue(ok)
        prefs = profile_store.get_preferences(self.conn, learner)
        self.assertEqual(prefs["level_code"], "b1")
        self.assertEqual(prefs["level_uncertain"], 0, "布尔值要归一成 0/1")
        self.assertEqual(prefs["language_mode"], "zh_pair")
        self.assertEqual(prefs["length_mode"], "brief")
        self.assertEqual(prefs["goal_code"], "exam")
        self.assertEqual(prefs["focus_code"], "writing")

    def test_saving_again_overwrites_the_same_row(self):
        """【核心】改档案是【覆盖同一行】，不是再插一行。"""
        learner = self.make_learner()
        profile_store.save_preferences(self.conn, learner, "b1", False, "zh_pair", "brief")
        profile_store.save_preferences(self.conn, learner, "c1", True, "en_only", "detailed")

        self.assertEqual(self.count("learner_preferences"), 1, "改档案时插出了第二行")
        prefs = profile_store.get_preferences(self.conn, learner)
        self.assertEqual(prefs["level_code"], "c1")
        self.assertEqual(prefs["language_mode"], "en_only")
        self.assertEqual(prefs["level_uncertain"], 1)

    def test_goal_and_focus_are_optional(self):
        learner = self.make_learner()
        profile_store.save_preferences(self.conn, learner, "b1", False, "zh_pair", "normal")

        prefs = profile_store.get_preferences(self.conn, learner)
        self.assertIsNone(prefs["goal_code"])
        self.assertIsNone(prefs["focus_code"])

    def test_every_whitelist_is_enforced(self):
        """【核心安全断言】白名单之外的代号一律拒绝，而且什么都不写。

        这条挡的是「用户自己造一个值塞进数据库」——
        档案会拼进提示词，不能有任何用户可控的自由文本。
        """
        learner = self.make_learner()
        bad_cases = [
            {"level_code": "z9"},
            {"language_mode": "hhh"},
            {"length_mode": "超级详细"},
            {"goal_code": "随便写的目标"},
            {"focus_code": "everything"},
        ]
        for bad in bad_cases:
            args = {"level_code": "b1", "level_uncertain": False,
                    "language_mode": "zh_pair", "length_mode": "normal"}
            args.update(bad)
            with self.subTest(bad=bad):
                self.assertFalse(
                    profile_store.save_preferences(self.conn, learner, **args),
                    "非法值被放行了：" + repr(bad))

        self.assertEqual(self.count("learner_preferences"), 0, "非法输入居然写了库")

    def test_every_documented_code_is_accepted(self):
        """白名单里的每一个值都必须真的能存进去（防止拼错代号）。"""
        learner = self.make_learner()
        for level in profile_store.VALID_LEVELS:
            for lang in profile_store.VALID_LANGUAGE_MODES:
                for length in profile_store.VALID_LENGTH_MODES:
                    self.assertTrue(profile_store.save_preferences(
                        self.conn, learner, level, False, lang, length))
        for goal in profile_store.VALID_GOALS:
            self.assertTrue(profile_store.save_preferences(
                self.conn, learner, "b1", False, "zh_pair", "normal", goal_code=goal))
        for focus in profile_store.VALID_FOCUS:
            self.assertTrue(profile_store.save_preferences(
                self.conn, learner, "b1", False, "zh_pair", "normal", focus_code=focus))


# ===================== 6. 级联删除与持久化 =====================

class TestCascadeAndPersistence(StoreTestCase):

    def test_deleting_a_learner_cleans_up_children(self):
        """【核心】删掉学习者 → 他的偏好和会话绑定必须一起消失。

        这条只有在 PRAGMA foreign_keys=ON 时才成立 ——
        关着的话 ON DELETE CASCADE 只是一句注释，删完会留下孤儿数据。
        """
        self.make_invite("c1")
        _o, learner = self.redeem("c1")
        profile_store.link_session(self.conn, learner, "s1")
        profile_store.save_preferences(self.conn, learner, "b1", False, "zh_pair", "normal")

        with self.conn:
            self.conn.execute("DELETE FROM learners WHERE id = ?", (learner,))

        self.assertEqual(self.count("learners"), 0)
        self.assertEqual(self.count("learner_sessions"), 0, "会话绑定变成了孤儿数据")
        self.assertEqual(self.count("learner_preferences"), 0, "偏好变成了孤儿数据")

    def test_deleting_a_learner_keeps_the_invite_but_clears_the_link(self):
        """邀请码本身不该跟着消失（它是「谁用过」的线索），只把指向清空。"""
        self.make_invite("c1")
        _o, learner = self.redeem("c1")

        with self.conn:
            self.conn.execute("DELETE FROM learners WHERE id = ?", (learner,))

        row = self.conn.execute("SELECT status, learner_id FROM invites").fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row[1], "邀请码应该断开指向，而不是被删掉")

    def test_data_survives_reopening_the_database(self):
        """【核心】关掉连接、重新打开（模拟应用重启）→ 档案还在。"""
        self.make_invite("c1")
        _o, learner = self.redeem("c1")
        profile_store.link_session(self.conn, learner, "session-X")
        profile_store.save_preferences(self.conn, learner, "b2", False, "en_only", "detailed",
                                       goal_code="work", focus_code="speaking")
        self.conn.close()

        self.conn = self._open()                       # 相当于重启后重新连上

        self.assertEqual(profile_store.learner_id_for_session(self.conn, "session-X"), learner)
        prefs = profile_store.get_preferences(self.conn, learner)
        self.assertEqual(prefs["level_code"], "b2")
        self.assertEqual(prefs["goal_code"], "work")
        self.assertEqual(prefs["focus_code"], "speaking")


# ===================== 7. 结构性保证 =====================

class TestStructuralGuarantees(StoreTestCase):

    def test_schema_creation_is_idempotent(self):
        """重复建表不能出错（每次应用启动都会调一次）。"""
        for _ in range(3):
            profile_store.ensure_schema(self.conn)

        names = {row[0] for row in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertLessEqual({"invites", "learners", "learner_sessions",
                              "learner_preferences"}, names)

    def test_profiling_schema_does_not_touch_the_existing_tables(self):
        """【结构性保证】这一轮只加新表，绝不碰 messages / api_usage。"""
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
                "  role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)")
        before = self.conn.execute("SELECT sql FROM sqlite_master WHERE name='messages'").fetchone()

        profile_store.ensure_schema(self.conn)

        after = self.conn.execute("SELECT sql FROM sqlite_master WHERE name='messages'").fetchone()
        self.assertEqual(before, after, "档案层把 messages 表改动了")

    def test_module_does_not_import_flask_or_network_libraries(self):
        """【结构性保证】存储层不认识 Flask，也不联网 —— 所以才能单独测。"""
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "profile_store.py"), encoding="utf-8") as f:
            source = f.read()

        for forbidden in ("import flask", "from flask", "import requests",
                          "import openai", "urllib"):
            self.assertNotIn(forbidden, source.lower(),
                             "profile_store.py 里不该出现 " + forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
