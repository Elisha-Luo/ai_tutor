# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# ai_tutor 网页的自动化测试
#
# 【最重要的一条原则】这些测试绝不调用真实的 DeepSeek，也绝不真的去读知识库。
# 做法是把两样东西都换成假的：
#   · app 模块里的 client  —— 所有模型请求都走假货
#   · retriever.retrieve   —— 所有检索都走假货
# 这样测试跑起来又快又花钱为零，断网也能跑。
#
# 【为什么检索也必须打桩】
# 不打的话，每个测试都会真的去读 knowledge_base、跑一遍 BM25 检索。
# 那样测试就不再是「只测网页这一层」，还会被检索结果的变化牵连——
# 你调一下检索权重，网页测试就红了，但它其实什么都没坏。
#
# 运行方式（在 ai_tutor 文件夹里）：
#     python -m unittest test_app -v
# =====================================================================

import os                                  # 设环境变量、拼路径
import re                                  # 从页面里抠出常量（等待状态的客户端断言）
import sys                                 # 把当前文件夹加进模块搜索路径
import json                                # 假模型要返回 JSON，这里负责拼
import atexit                              # 程序结束时顺手清理临时文件
import shutil                              # 删除临时文件夹用
import sqlite3                             # 测试里要直接查数据库，验证数据真的写进去了
import subprocess                          # 用 node 跑页面里那段纯函数（本机没有就跳过）
import tempfile                            # 建临时文件夹，让测试用独立的数据库文件
import time                                # 假上游要「挂住」一段时间
import unittest                            # Python 自带的测试框架，不用额外安装
import http.server                         # 本机假上游：只是一个本地 HTTP 服务
import importlib                           # 用来「重新加载模块」，模拟重启 Flask
import threading                           # 并发测每日额度时要起多个线程一起抢
import http.client                         # 真实浏览器那一组：把请求转发给本机 Flask
import urllib.parse                        # 同上：拼/拆探针回传的那串查询参数
import html as html_module                 # 同上：DOM 转出来的文本要先反转义再当 JSON 读

# =====================================================================
# 【顺序极其重要】下面这几行必须在 import app 之前执行，原因有两个：
#
#   1. app.py 一被导入就会读 DEEPSEEK_API_KEY 和 FLASK_SECRET_KEY，
#      读不到会直接报错退出，测试根本跑不起来。
#
#   2. app.py 一被导入还会【建数据库表】。如果不先把路径指走，
#      它就会在你项目文件夹里建出一个真实的 chat.db ——
#      跑个测试不该动到真实数据，哪怕只是建个空表。
#
# 所以：先把数据库路径指向一个临时文件夹，再去导入。
# =====================================================================
_IMPORT_TMPDIR = tempfile.mkdtemp(prefix="ai_tutor_test_")               # 建一个系统临时文件夹
atexit.register(shutil.rmtree, _IMPORT_TMPDIR, ignore_errors=True)      # 程序退出时自动删掉它

os.environ["CHAT_DB_PATH"] = os.path.join(_IMPORT_TMPDIR, "import_time.db")   # 导入时的建表动作也走临时文件
os.environ["DEEPSEEK_API_KEY"] = "test-key-not-a-real-key"                    # 假的密钥，绝不可能调通真实服务
os.environ["FLASK_SECRET_KEY"] = "test-secret-not-a-real-key"                 # 假的 cookie 签名密钥
os.environ["INVITE_CODE_PEPPER"] = "test-pepper-not-a-real-pepper"            # 假的 invitation pepper，只用于测试

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 保证能找到同文件夹下的 app.py
import env_utils                                                # 常量（示例值等）在这里定义
import model_client                                             # 真实的有界客户端：假上游那一组要用它
import app as tutor                                             # 导入被测对象


# ===================== 假检索：返回固定的一段资料 =====================
#
# source 和 heading 必须和下面假回复里的引用【逐字一致】，
# 否则 rag.py 的引用白名单会把它们拦下——那测的就是白名单，不是网页了。

FAKE_CHUNKS = [
    {"source": "grammar_present_perfect.md",
     "heading": "基本结构",
     "text": "主语 + have / has + 动词的过去分词"},
]

RETRIEVAL_CALLS = []        # 每次假检索被调用都记在这里，供测试断言「到底检索了没有、用的什么问题」


def fake_retrieve(question, top_k=3):
    """假的检索函数：记一笔调用，然后返回上面那一段固定资料。"""
    RETRIEVAL_CALLS.append({"question": question, "top_k": top_k})
    return [dict(c) for c in FAKE_CHUNKS]                  # 复制一份，免得测试互相改到同一份数据


def rag_reply(decision="answer", answer="这是 AI 的回答。", citations=None):
    """造一个 rag.py 会接受的模型回复。

    【和以前最大的不同】rag.py 要求模型输出 JSON，不再是一段自由文本。
    所以假客户端的默认回复必须是一个格式正确的 JSON 字符串，
    否则 rag 会判成「invalid_json」然后安全降级——那样大部分测试都会
    莫名其妙地拿到拒答，却看不出是夹具的问题。
    """
    if citations is None:
        citations = [{"source": "grammar_present_perfect.md", "heading": "基本结构"}]
    return json.dumps({"decision": decision, "answer": answer, "citations": citations},
                      ensure_ascii=False)


# ===================== 假的 DeepSeek 客户端 =====================
# 这一组类的唯一作用，就是冒充真的 client，让测试永远不会发出真实网络请求。

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
        reply = self.owner.replies.pop(0) if self.owner.replies else self.owner.reply
        if isinstance(reply, BaseException):
            raise reply
        return FakeResponse(reply)

class FakeChat:
    def __init__(self, owner):
        self.completions = FakeCompletions(owner)

class FakeClient:
    def __init__(self):
        self.calls = []                     # 每次被调用都记在这里，用来证明「调了几次」「传了什么」
        self.reply = rag_reply()            # 默认返回一个合法的 RAG 结果
        self.replies = []                    # 可选：逐次返回不同草稿，离线测一次改写
        self.rewrite_timeouts = []
        self.chat = FakeChat(self)

    def with_timeout(self, timeout):
        self.rewrite_timeouts.append(timeout)
        return self


# ===================== 测试基类 =====================

class ChatTestCase(unittest.TestCase):

    def setUp(self):
        """每个测试方法跑之前都会执行一次，负责把环境恢复干净。"""
        global tutor

        # 建一个临时文件夹，测试用的数据库就放这儿。
        # 这样绝不会碰到你本地那个真实的 chat.db。
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test.db")
        os.environ["CHAT_DB_PATH"] = self.db_path          # 告诉 app：数据库用这个临时文件

        # 重新加载 app 模块。这会重新执行 app.py：
        #   - 用上面那个临时数据库路径重新建表
        #   - 模块级状态（那把锁、Flask 应用对象）全部重置成干净的
        # 相当于每次测试都在一个全新的进程里跑。
        tutor = importlib.reload(tutor)

        # ---- 打桩 1：把真客户端换掉 ----
        self.fake = FakeClient()
        tutor.client = self.fake                           # 【打桩】从此不可能调真 API

        # ---- 打桩 2：把真检索换掉 ----
        # 【为什么必须还原】retriever 是个模块级的单例，改了它会影响
        # 同一个进程里后面跑的其它测试（比如评测器的 dry-run 会用真的 retrieve）。
        # 所以备份原函数，测试结束再放回去。
        self._real_retrieve = tutor.retriever.retrieve
        tutor.retriever.retrieve = fake_retrieve
        self.addCleanup(self._restore_retrieve)

        RETRIEVAL_CALLS.clear()
        tutor.app.config["TESTING"] = True

    def _restore_retrieve(self):
        tutor.retriever.retrieve = self._real_retrieve

    def tearDown(self):
        self.tmpdir.cleanup()                              # 删掉临时文件夹和里面的数据库

    # ---------- 几个小工具 ----------

    def all_rows(self):
        """把数据库里的所有记录读出来，用来验证「到底写了什么」。"""
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                "SELECT session_id, role, content FROM messages ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

    def session_id_of(self, client):
        """查出某个测试客户端对应的会话 ID（从它自己的对话记录里反推）。"""
        with client.session_transaction() as sess:          # Flask 提供的测试接口，能直接看会话内容
            return sess.get("sid")

    def ask(self, client, question="随便问一句"):
        """模拟一次完整的提问，返回响应对象。"""
        return client.post("/", data={"question": question})

    def use_fake_upstream(self, mode, timeout=10):
        """把 app 的客户端换成【真的】BoundedModelClient，指向本机假上游。

        【和 FakeClient 的区别】FakeClient 是纯打桩，用来测网页自己那一层；
        这个用的是真实代码路径（真的起子进程、真的发 HTTP、真的有 deadline），
        用来测「超时 / 上游报错 / 卡住」这些只有真链路才答得准的事。
        """
        upstream = FakeUpstream(mode)
        self.addCleanup(upstream.stop)
        tutor.client = model_client.BoundedModelClient(
            api_key="test-key-not-a-real-key", base_url=upstream.base_url,
            timeout=timeout)
        return upstream

    # ---------- 学习档案相关的小工具 ----------

    def make_invite(self, code="test-invite-code", status=None):
        """登记一张邀请码（走产品代码里的 create_invite，不自己拼 SQL）。"""
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertTrue(tutor.profile_store.create_invite(
                conn, code, tutor.INVITE_CODE_PEPPER))
            if status is not None and status != tutor.profile_store.INVITE_ACTIVE:
                with conn:
                    conn.execute("UPDATE invites SET status = ?", (status,))
        finally:
            conn.close()

    def csrf_token(self, client, path="/invite"):
        """从页面里把 CSRF token 抠出来（模拟浏览器随表单一起提交）。

        【为什么要从页面里抠，而不是直接读 session】
        读 session 能绕过「页面到底有没有真的把它渲染出来」这件事。
        从 HTML 里抠，顺带证明了模板里确实带上了这个隐藏字段。
        """
        html = client.get(path).get_data(as_text=True)
        marker = 'name="csrf_token" value="'
        start = html.index(marker) + len(marker)
        return html[start:html.index('"', start)]

    def enter_invite(self, client, code):
        """走一遍「输邀请码」的完整流程。"""
        return client.post("/invite", data={
            "invite_code": code,
            "csrf_token": self.csrf_token(client, "/invite"),
        })

    def save_profile(self, client, **fields):
        """提交档案表单。没指定的字段用一组合法的默认值。"""
        data = {
            "csrf_token": self.csrf_token(client, "/profile"),
            "level_code": "b1",
            "language_mode": "zh_pair",
            "length_mode": "normal",
        }
        data.update(fields)
        return client.post("/profile", data=data)

    def learner_id_of(self, client):
        """查出这个测试浏览器【当前有效】地属于哪个 learner（从数据库反查）。

        【为什么必须带 revoked_at IS NULL】这和 app.py 的 _learner_id_for 是同一条规则：
        作废之后绑定那一行还在（关联要留着），但访问权已经没了。
        少了这个条件，「作废后设备失去访问权」的测试就会误报通过。
        """
        sid = self.session_id_of(client)
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT learner_id FROM learner_sessions "
                "WHERE session_id = ? AND revoked_at IS NULL", (sid,)
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def session_row(self, client):
        """把这个浏览器的会话绑定行原样取出来：(learner_id, revoked_at)。

        【和 learner_id_of 的区别】那个回答「现在归谁」，这个回答「库里那一行长什么样」——
        测「作废只打标记、不删行」时需要看原始状态。
        """
        sid = self.session_id_of(client)
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                "SELECT learner_id, revoked_at FROM learner_sessions WHERE session_id = ?",
                (sid,)
            ).fetchone()
        finally:
            conn.close()

    def messages_for_session(self, client):
        """这个浏览器【当前】会话的聊天记录条数（直接查库，不经过页面）。"""
        return self.messages_for_session_id(self.session_id_of(client))

    def messages_for_session_id(self, sid):
        """指定 session_id 的聊天记录条数。

        【为什么要单独一个】会话轮换之后，浏览器手里的 id 和旧数据的 id 不一样了 ——
        要验证「旧数据还在库里」，就得能按【旧的】id 去查。
        """
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
            ).fetchone()[0]
        finally:
            conn.close()

    def table_count(self, table):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
        finally:
            conn.close()

    def profile_row(self, learner_id):
        """直接查库，看档案到底存了什么。"""
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                "SELECT level_code, level_uncertain, language_mode, length_mode, "
                "goal_code, focus_code FROM learner_preferences WHERE learner_id = ?",
                (learner_id,),
            ).fetchone()
        finally:
            conn.close()

    def connect(self):
        """开一个到测试库的连接（外键打开，和 app 里一致）。"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def revoke(self, code):
        """作废一张邀请码（走产品代码，不自己拼 SQL）。"""
        conn = self.connect()
        try:
            return tutor.profile_store.revoke_invite(conn, code, tutor.INVITE_CODE_PEPPER)
        finally:
            conn.close()

    def reissue(self, learner_id):
        """给一个学习者补发新码。"""
        conn = self.connect()
        try:
            return tutor.profile_store.issue_replacement_invite(
                conn, learner_id, tutor.INVITE_CODE_PEPPER)
        finally:
            conn.close()

    def use_empty_retrieval(self):
        """把检索换成「永远找不到资料」，用来测拒答路径。

        注意这里照样要记一笔调用——不然就分不清「没检索」和「检索了但没找到」，
        而这两件事的处理完全不同。
        """
        def empty(question, top_k=3):
            RETRIEVAL_CALLS.append({"question": question, "top_k": top_k})
            return []
        tutor.retriever.retrieve = empty


# ===================== 1. 会话隔离 =====================

class TestSessionIsolation(ChatTestCase):

    def test_two_sessions_do_not_leak(self):
        """两个独立会话，各自只能看到自己的对话，绝不串话。"""
        c1 = tutor.app.test_client()       # 第一个「浏览器」
        c2 = tutor.app.test_client()       # 第二个「浏览器」，cookie 是分开的

        self.ask(c1, "会话一的秘密问题")
        self.ask(c2, "会话二的问题")

        html1 = c1.get("/").get_data(as_text=True)
        html2 = c2.get("/").get_data(as_text=True)

        # 各看各的
        self.assertIn("会话一的秘密问题", html1)
        self.assertIn("会话二的问题", html2)

        # 【核心断言】绝不能看到对方的
        self.assertNotIn("会话二的问题", html1)
        self.assertNotIn("会话一的秘密问题", html2)

    def test_each_session_gets_a_different_id(self):
        """两个浏览器拿到的会话 ID 必须不一样，否则就谈不上隔离。"""
        c1 = tutor.app.test_client()
        c2 = tutor.app.test_client()
        c1.get("/")                        # 各访问一次，触发会话 ID 的生成
        c2.get("/")

        sid1 = self.session_id_of(c1)
        sid2 = self.session_id_of(c2)

        self.assertIsNotNone(sid1)
        self.assertIsNotNone(sid2)
        self.assertNotEqual(sid1, sid2)    # 不一样才安全

    def test_database_rows_are_partitioned_by_session(self):
        """数据库层面也要确认：两个人的记录挂在不同的 session_id 下。"""
        c1 = tutor.app.test_client()
        c2 = tutor.app.test_client()
        self.ask(c1, "甲的问题")
        self.ask(c2, "乙的问题")

        rows = self.all_rows()
        self.assertEqual(len(rows), 4)                      # 两个人 × 一问一答 = 4 条

        sids = {r[0] for r in rows}                         # 取出所有出现过的 session_id
        self.assertEqual(len(sids), 2)                      # 恰好两个，说明没有混在一起

        # 甲的问题只属于甲的会话
        jia = [r for r in rows if r[2] == "甲的问题"]
        self.assertEqual(len(jia), 1)
        self.assertNotEqual(jia[0][0], "")                  # 确实挂了会话 ID


# ===================== 2. 持久化（重启不丢）=====================

class TestPersistence(ChatTestCase):

    def test_history_survives_app_recreation(self):
        """模拟重启 Flask：模块重新加载后，同一个浏览器仍能看到自己的历史。"""
        global tutor                        # 这个测试会重新赋值 tutor，所以 global 必须写在函数最前面
        c = tutor.app.test_client()
        self.ask(c, "重启之前问的问题")

        cookie = c.get_cookie("session")                    # 浏览器手里那个签名过的 cookie
        self.assertIsNotNone(cookie, "提交之后应该拿到会话 cookie")

        # ---- 模拟「重启服务器」----
        # 重新加载模块 = 进程里的所有状态清空：锁是新的、Flask 应用对象是新的。
        # 唯一活下来的是磁盘上的数据库文件，以及浏览器手里的 cookie。
        tutor = importlib.reload(tutor)
        tutor.client = FakeClient()                         # 新模块也要重新打桩
        tutor.retriever.retrieve = fake_retrieve

        c2 = tutor.app.test_client()                        # 全新的应用实例
        c2.set_cookie("session", cookie.value)              # 但用的是同一个浏览器 cookie

        html = c2.get("/").get_data(as_text=True)
        self.assertIn("重启之前问的问题", html)               # 历史还在

    def test_history_read_from_database_not_memory(self):
        """确认历史真的是从数据库读的，而不是留在某个内存变量里。"""
        c = tutor.app.test_client()
        self.ask(c, "这句话应该落进数据库")

        sid = self.session_id_of(c)
        rows = [r for r in self.all_rows() if r[0] == sid]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][2], "这句话应该落进数据库")


# ===================== 3. 空问题不进数据库 =====================

class TestEmptyInput(ChatTestCase):

    def test_empty_question_writes_nothing(self):
        """空问题、纯空格、纯换行，都不该写进数据库，也不该调用 AI。"""
        c = tutor.app.test_client()
        c.get("/")                                          # 先访问一次，建立会话

        self.ask(c, "")
        self.ask(c, "   ")
        self.ask(c, "\n\t  ")

        self.assertEqual(self.all_rows(), [], "空问题不该在数据库里留下任何记录")
        self.assertEqual(len(self.fake.calls), 0, "空问题不该调用 AI")
        self.assertEqual(len(RETRIEVAL_CALLS), 0, "空问题也不该去检索")

    def test_empty_question_does_not_block_later_questions(self):
        """空问题不能把锁占住——之后正常提问还得能用。"""
        c = tutor.app.test_client()
        self.ask(c, "   ")                                  # 先来一个空的
        r = self.ask(c, "之后正常的问题")                     # 再来一个正常的

        self.assertEqual(r.status_code, 302)                # 正常走了 POST-Redirect-GET
        self.assertEqual(len(self.all_rows()), 2)           # 一问一答正好两条
        self.assertIn("这是 AI 的回答。", self.all_rows()[1][2])


# ===================== 4. 原有行为没被破坏 =====================

class TestExistingBehaviour(ChatTestCase):

    def test_post_redirect_get_still_works(self):
        """成功后仍然是 302 跳转，不是直接渲染——F5 才不会重复提问。"""
        c = tutor.app.test_client()
        r = self.ask(c, "随便问一句")
        self.assertEqual(r.status_code, 302)

    def test_duplicate_request_is_blocked(self):
        """上一个请求还在处理时，第二次提问被挡住：不调 AI，也不写数据库。"""
        c = tutor.app.test_client()
        c.get("/")

        tutor._lock.acquire()                               # 假装「上一个问题还在问 AI」
        try:
            self.ask(c, "这是重复按回车发出的")
        finally:
            tutor._lock.release()                           # 记得还回去，否则后面的测试都会卡住

        self.assertEqual(len(self.fake.calls), 0, "重复请求不该调用 AI")
        self.assertEqual(self.all_rows(), [], "重复请求不该写数据库")

    def test_lock_is_released_after_ai_error(self):
        """模型服务失败返回可重试错误，释放锁，不存成教学回答。"""
        c = tutor.app.test_client()
        c.get("/")

        def boom(model, messages, **kwargs):                # 让假的 AI 故意炸一次
            raise RuntimeError("模拟网络故障")
        self.fake.chat.completions.create = boom

        r = self.ask(c, "这一问会失败")

        self.assertEqual(r.status_code, 503)
        self.assertFalse(tutor._lock.locked(), "出错后锁必须被释放")
        self.assertEqual(self.all_rows(), [])

    def test_no_api_key_in_database(self):
        """数据库里绝不能出现 API 密钥。"""
        c = tutor.app.test_client()
        self.ask(c, "随便问一句")

        conn = sqlite3.connect(self.db_path)
        try:
            dump = " ".join(str(r) for r in conn.execute("SELECT * FROM messages").fetchall())
        finally:
            conn.close()
        self.assertNotIn(tutor.API_KEY, dump)


# ===================== 5. RAG 接入网页后的完整链路 =====================

class TestRagPipeline(ChatTestCase):

    def test_question_goes_through_retrieval_then_model(self):
        """【核心】问题必须先检索、再交给模型，顺序不能反。"""
        c = tutor.app.test_client()
        self.ask(c, "现在完成时的句子结构是怎样的？")

        self.assertEqual(len(RETRIEVAL_CALLS), 1, "应该先走一次检索")
        self.assertEqual(RETRIEVAL_CALLS[0]["question"], "现在完成时的句子结构是怎样的？")
        self.assertEqual(RETRIEVAL_CALLS[0]["top_k"], tutor.RETRIEVE_TOP_K)
        self.assertEqual(len(self.fake.calls), 1, "检索之后才调用模型")

    def test_answer_is_saved_with_source_and_heading(self):
        """【核心】正常答案要连正文和来源一起存进数据库。"""
        c = tutor.app.test_client()
        self.ask(c, "随便问一句")

        rows = self.all_rows()
        self.assertEqual(len(rows), 2)

        saved = rows[1][2]                                  # assistant 那条
        self.assertIn("这是 AI 的回答。", saved)
        self.assertIn("资料来源：", saved)
        self.assertIn("- grammar_present_perfect.md · 基本结构", saved)

    def test_page_shows_source_file_and_heading(self):
        """【核心】页面上要能看到文件名和二级标题。"""
        c = tutor.app.test_client()
        self.ask(c, "随便问一句")

        html = c.get("/").get_data(as_text=True)
        # 用带全角冒号的写法，才能确认页面上真的列出了来源，
        # 而不是只命中副标题里那四个字
        self.assertIn("资料来源：", html)
        self.assertIn("grammar_present_perfect.md", html)
        self.assertIn("基本结构", html)

    def test_empty_retrieval_still_calls_the_model(self):
        """【本轮核心改动】没检索到资料时【仍然】调用模型。

        因为「资料里没有」和「这不是个正当的英语问题」完全是两回事。
        模型会判断：正常的英语问题用 general_answer，超范围的用 refuse。
        """
        self.use_empty_retrieval()
        self.fake.reply = rag_reply(decision="general_answer",
                                    answer="这是通用知识回答。", citations=[])
        c = tutor.app.test_client()
        self.ask(c, "this 和 that 有什么区别？")

        self.assertEqual(len(self.fake.calls), 1, "chunks 为空时也该调用模型")
        self.assertEqual(len(RETRIEVAL_CALLS), 1, "检索还是要跑一次的")

        rows = self.all_rows()
        self.assertEqual(len(rows), 2, "回答要如实记进历史")
        self.assertIn("通用知识回答", rows[1][2])

    def test_general_answer_shows_the_marker_and_no_sources(self):
        """【核心】general_answer 要显示「AI 通用知识回答」，且【绝不出】资料来源。"""
        self.fake.reply = rag_reply(decision="general_answer",
                                    answer="this 和 that 的区别是……", citations=[])
        c = tutor.app.test_client()
        self.ask(c, "this 和 that 有什么区别？")

        saved = self.all_rows()[1][2]
        self.assertIn("AI 通用知识回答", saved)
        self.assertNotIn("资料来源", saved)

        html = c.get("/").get_data(as_text=True)
        self.assertIn("AI 通用知识回答", html)
        self.assertNotIn("资料来源：", html)

    def test_general_answer_with_citations_is_degraded_not_displayed(self):
        """【核心】general_answer 如果带了引用，会安全降级 —— 引用不能显示出来。"""
        self.fake.reply = rag_reply(
            decision="general_answer", answer="通用回答",
            citations=[{"source": "grammar_present_perfect.md", "heading": "基本结构"}])
        c = tutor.app.test_client()
        self.ask(c, "this 和 that 有什么区别？")

        saved = self.all_rows()[1][2]
        self.assertNotIn("AI 通用知识回答", saved)
        self.assertNotIn("基本结构", saved)

    def test_refusal_never_shows_a_sources_section(self):
        """【核心】拒答绝不能伪造「资料来源」。

        顺带钉住一件事：拒答时给用户看的【不是】模型那段话，
        而是 rag.py 写死的固定文案。理由是不回答的路径本身就是「模型不可信」时的
        安全网，把对外话术交回给模型，等于把安全网又交回给它。
        """
        self.fake.reply = rag_reply(decision="refuse",
                                    answer="模型随便写的一段话", citations=[])
        c = tutor.app.test_client()
        self.ask(c, "课程多少钱？")

        saved = self.all_rows()[1][2]
        self.assertEqual(saved, tutor.rag.REFUSE_TEXT)
        self.assertNotIn("模型随便写的一段话", saved, "拒答不该采用模型那段话")
        self.assertNotIn("资料来源：", saved)

        # 【为什么查「资料来源：」而不是「资料来源」】
        # 页面副标题里本来就有「资料来源」四个字（"显示资料来源"），
        # 直接查那四个字永远会命中，等于没查。
        # 格式化出来的那一节用的是全角冒号「资料来源：」，副标题用的是分号，
        # 用带冒号的完整写法才能精确区分「页面上真的列出了来源」。
        html = c.get("/").get_data(as_text=True)
        self.assertNotIn("资料来源：", html)

    def test_insufficient_evidence_never_shows_a_sources_section(self):
        """【核心】证据不足同样不显示来源，也同样是固定文案。"""
        self.fake.reply = rag_reply(decision="insufficient_evidence",
                                    answer="模型随便写的一段话", citations=[])
        c = tutor.app.test_client()
        self.ask(c, "现在完成进行时和现在完成时有什么区别？")

        saved = self.all_rows()[1][2]
        self.assertEqual(saved, tutor.rag.INSUFFICIENT_TEXT)
        self.assertNotIn("模型随便写的一段话", saved)
        self.assertNotIn("资料来源", saved)

    def test_fabricated_citation_never_reaches_the_page_or_the_database(self):
        """【核心】模型编造来源时，编造的内容绝不能出现在页面或数据库里。

        这是整条链路上最要紧的一关，所以两个出口都要查。
        """
        self.fake.reply = rag_reply(citations=[
            {"source": "编造出来的文件.md", "heading": "编造出来的标题"},
        ])
        c = tutor.app.test_client()
        self.ask(c, "随便问一句")

        dump = " ".join(str(r) for r in self.all_rows())
        self.assertNotIn("编造", dump, "编造的来源进数据库了")

        html = c.get("/").get_data(as_text=True)
        self.assertNotIn("编造", html, "编造的来源显示在页面上了")

        # 安全降级的结果应该是「不回答」，而不是带着假来源的答案
        self.assertNotIn("资料来源", dump)

    def test_bad_json_degrades_safely(self):
        """模型返回一坨不是 JSON 的文字 → 安全降级，网页照常可用。"""
        self.fake.reply = "我觉得应该这样回答：主语 + have + 过去分词。"
        c = tutor.app.test_client()
        r = self.ask(c, "随便问一句")

        self.assertEqual(r.status_code, 302)
        dump = " ".join(str(r2) for r2 in self.all_rows())
        self.assertNotIn("我觉得应该这样回答", dump, "坏 JSON 的原文被当成回答存下来了")
        self.assertNotIn("资料来源", dump)

    def test_api_exception_does_not_leak_or_save_a_teaching_answer(self):
        """服务故障不伪装成证据不足；保留输入且不泄露内部细节。"""
        marker = "内部细节-接口炸了-7c1e"

        def boom(model, messages, **kwargs):
            raise RuntimeError(marker)
        self.fake.chat.completions.create = boom

        c = tutor.app.test_client()
        response = self.ask(c, "随便问一句")

        rows = self.all_rows()
        self.assertEqual(rows, [], "服务故障不能污染对话上下文")
        dump = " ".join(str(r) for r in rows)
        self.assertNotIn(marker, dump, "异常原文泄露进数据库了")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 503)
        self.assertIn(tutor.MODEL_UNAVAILABLE_MESSAGE, html)
        self.assertIn('value="随便问一句"', html)
        self.assertNotIn(marker, html, "异常原文泄露到页面了")
        self.assertNotIn("资料来源", dump)

    def test_retrieval_exception_does_not_leak_and_writes_nothing(self):
        """【核心】检索自己炸了：不泄露异常原文，也【不写数据库】。

        为什么和「没找到资料」区别对待：检索出错时我们并不知道资料里到底
        有没有答案，写一句「资料里没有」就是撒谎。宁可不记，也不要记错的。
        """
        marker = "内部细节-检索炸了-9a2f"

        def boom(question, top_k=3):
            raise RuntimeError(marker)
        tutor.retriever.retrieve = boom

        c = tutor.app.test_client()
        r = self.ask(c, "随便问一句")
        html = r.get_data(as_text=True)

        self.assertEqual(r.status_code, 200)                # 不白屏
        self.assertNotIn(marker, html, "异常原文泄露到页面了")
        self.assertIn(tutor.SAFE_ERROR_MESSAGE, html)       # 只给一句固定的通俗提示
        self.assertEqual(self.all_rows(), [], "检索异常时不该写数据库")
        self.assertFalse(tutor._lock.locked(), "出错后锁必须被释放")
        self.assertEqual(len(self.fake.calls), 0, "检索都失败了，不该再去调模型")


# ===================== 5c-1. 作文明确字数的可见校验 =====================

class TestEssayWordRangeWarning(ChatTestCase):
    """真实网页流程 + 假模型：字数补救至多一次，不合格稿不交付。"""

    TASK = "Write an English opinion essay of 140–190 words about teenagers teaching adults."

    def test_slightly_long_second_draft_can_drop_one_optional_sentence(self):
        import writing_limits
        from test_writing_limits import SECOND_DRAFT_FROM_LIVE_RETEST

        first = "first " * 225
        second = SECOND_DRAFT_FROM_LIVE_RETEST
        self.fake.replies = [
            rag_reply(decision="general_answer", answer=first, citations=[]),
            rag_reply(decision="general_answer", answer=second, citations=[]),
        ]
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            response = self.ask(tutor.app.test_client(), self.TASK)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(self.fake.calls), 2)
        saved = self.all_rows()[1][2]
        self.assertNotIn(first.strip(), saved)
        self.assertNotIn("Word-count warning", saved)
        self.assertNotIn("字数提醒", saved)
        self.assertIn("Technology is the clearest example.", saved)
        self.assertIn("Environmental awareness is another area.", saved)
        self.assertIn("A third area is communication.", saved)
        expected = writing_limits.trim_optional_essay_sentence(
            second, writing_limits.WordRangeCheck(140, 190, 201))
        self.assertIn(expected, saved)
        self.assertTrue(140 <= writing_limits.count_english_words(expected) <= 190)
        events = "\n".join(captured.output)
        self.assertIn("essay_rewrite_sentence_trimmed", events)
        self.assertIn("essay_rewrite_accepted", events)
        self.assertNotIn("Teenagers are often dismissed", events)

    def test_over_limit_draft_is_rejected_without_saving_it(self):
        draft = "word " * 225
        self.fake.reply = rag_reply(decision="general_answer", answer=draft, citations=[])
        c = tutor.app.test_client()
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            response = self.ask(c, self.TASK)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(self.fake.calls), 2, "改写最多只可补调用一次")
        self.assertEqual(self.all_rows(), [], "不合格稿不能写入历史")
        html = response.get_data(as_text=True)
        self.assertIn("这次未能生成符合 140–190 英文词要求的作文", html)
        self.assertIn(self.TASK, html, "问题应保留在输入框，便于用户自己决定是否重试")
        self.assertNotIn(draft.strip(), html, "不合格稿不能伪装成成品")
        self.assertEqual(self.fake.rewrite_timeouts and len(self.fake.rewrite_timeouts), 1)
        events = "\n".join(captured.output)
        self.assertIn("essay_rewrite_rejected", events)
        self.assertIn("stage=word_range", events)
        self.assertIn("rewrite_word_actual=225", events)
        self.assertIn("rewrite_identical=True", events)
        self.assertNotIn(draft.strip(), events, "日志不能出现作文正文")

    def test_valid_second_draft_replaces_first_and_uses_two_quota_units(self):
        first, second = "first " * 225, "revised " * 170
        self.fake.replies = [
            rag_reply(decision="general_answer", answer=first, citations=[]),
            rag_reply(decision="general_answer", answer=second, citations=[]),
        ]
        c = tutor.app.test_client()
        self.assertEqual(self.ask(c, self.TASK).status_code, 302)
        saved = self.all_rows()
        self.assertEqual(len(saved), 2)
        self.assertIn(second.strip(), saved[1][2])
        self.assertNotIn(first.strip(), saved[1][2])
        self.assertNotIn("字数提醒", saved[1][2])
        self.assertEqual(len(self.fake.calls), 2)
        revision_message = self.fake.calls[1]["messages"][1]["content"]
        self.assertIn("Draft to edit", revision_message)
        self.assertIn("225 English words", revision_message)
        self.assertIn("Cut at least 35 words", revision_message)
        self.assertIn("Aim for 150 words", revision_message)
        self.assertIn(first.strip(), revision_message)
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("SELECT used FROM api_usage").fetchone()[0], 2)
        finally:
            conn.close()

    def test_no_second_quota_rejects_overlength_draft(self):
        tutor.DAILY_API_LIMIT = 1
        self.fake.reply = rag_reply(
            decision="general_answer", answer="word " * 225, citations=[])
        c = tutor.app.test_client()
        self.ask(c, self.TASK)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.all_rows(), [])
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("SELECT used FROM api_usage").fetchone()[0], 1)
        finally:
            conn.close()

    def test_invalid_second_result_never_replaces_valid_first(self):
        first = "original " * 225
        self.fake.replies = [
            rag_reply(decision="general_answer", answer=first, citations=[]),
            rag_reply(decision="general_answer", answer="revised " * 170,
                      citations=[{"source": "forged.md", "heading": "not supplied"}]),
        ]
        self.ask(tutor.app.test_client(), self.TASK)
        self.assertEqual(self.all_rows(), [])
        self.assertEqual(len(self.fake.calls), 2)

    def test_citation_supported_answer_is_not_rewritten(self):
        self.fake.reply = rag_reply(decision="answer", answer="word " * 225)
        response = self.ask(tutor.app.test_client(), self.TASK)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.all_rows(), [])
        self.assertNotIn("资料来源：", response.get_data(as_text=True))

    def test_second_model_error_rejects_first_overlength_draft(self):
        first = "original " * 225
        self.fake.replies = [
            rag_reply(decision="general_answer", answer=first, citations=[]),
            RuntimeError("private model failure"),
        ]
        response = self.ask(tutor.app.test_client(), self.TASK)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.all_rows(), [])
        self.assertNotIn(first.strip(), response.get_data(as_text=True))
        self.assertEqual(len(self.fake.calls), 2)

    def test_no_time_for_rewrite_rejects_overlength_draft(self):
        tutor.MODEL_REQUEST_TIMEOUT_SECONDS = 5
        self.fake.reply = rag_reply(
            decision="general_answer", answer="word " * 225, citations=[])
        self.ask(tutor.app.test_client(), self.TASK)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.all_rows(), [])

    def test_in_range_draft_has_no_warning(self):
        self.fake.reply = rag_reply(
            decision="general_answer", answer="word " * 189, citations=[])
        c = tutor.app.test_client()
        self.ask(c, self.TASK)

        self.assertEqual(len(self.fake.calls), 1)
        saved = self.all_rows()[1][2]
        self.assertNotIn("字数提醒", saved)
        self.assertNotIn("Word-count warning", saved)

    def test_non_essay_and_refusal_are_not_relabelled_as_bad_essays(self):
        c = tutor.app.test_client()
        self.fake.reply = rag_reply(
            decision="general_answer", answer="word " * 225, citations=[])
        self.ask(c, "What does 140–190 words mean in an essay task?")
        self.assertNotIn("字数提醒", self.all_rows()[1][2])

        self.fake.reply = rag_reply(decision="refuse", answer="ignored", citations=[])
        self.ask(c, self.TASK)
        self.assertEqual(self.all_rows()[3][2], tutor.rag.REFUSE_TEXT)
        self.assertEqual(len(self.fake.calls), 2)

    def test_english_profile_receives_english_failure(self):
        self.make_invite("essay-range-profile")
        c = tutor.app.test_client()
        self.enter_invite(c, "essay-range-profile")
        self.save_profile(c, language_mode="en_only")
        self.fake.reply = rag_reply(
            decision="general_answer", answer="word " * 225, citations=[])
        response = self.ask(c, self.TASK)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.all_rows(), [])
        html = response.get_data(as_text=True)
        self.assertIn("I could not produce an essay within your requested", html)
        self.assertNotIn("字数提醒", html)


# ===================== 5d. 短期对话上下文（网页层）=====================
#
# 【这一组在测什么】网页把【本会话最近的几条消息】带给模型这条链路上，
# 三件事必须同时成立：
#   1. 带对了 —— 最近的进去、更老的出来、顺序是正的、不超过三条上限
#   2. 只带自己的 —— A 会话的历史绝不会出现在 B 会话的上下文里
#   3. 只当上下文 —— 历史里的来源名【不能】变成引用，仍然只认本次检索到的片段
#
# 【解析提示词而不是直接调 rag.build_recent_context】
# 下面那个 context_entries() 解析的是【真正发给模型的那串字】。
# 所以它同时证明了两件事：「上限生效」和「app.py 确实把历史传下去了」——
# 只调纯函数的话，app.py 万一忘了传，测试照样绿。

class TestShortTermContext(ChatTestCase):

    # ---------- 小工具 ----------

    def ask_turn(self, client, question, answer):
        """问一轮，并让这一轮的回答带上可辨认的标记。"""
        self.fake.reply = rag_reply(answer=answer)
        return self.ask(client, question)

    def user_text(self, index=-1):
        """取出发给模型的 user 消息正文（-1 = 最后一次调用）。"""
        return self.fake.calls[index]["messages"][1]["content"]

    def context_entries(self, index=-1):
        """把提示词里那段『最近上下文』解析成 [(role, 正文), ...]。

        正文里的换行（比如回答带的『资料来源：』那一节）会归到上一条，
        不会被算成新的一条，也不会被漏掉 —— 预算断言必须算上它们。
        """
        sent = self.user_text(index)
        open_mark = "======== 最近上下文开始 ========"
        close_mark = "======== 最近上下文结束 ========"

        start = sent.index(open_mark) + len(open_mark)
        end = sent.index(close_mark)

        entries = []
        for line in sent[start:end].strip("\n").split("\n"):
            for prefix, role in (("用户：", "用户"), ("助手：", "助手")):
                if line.startswith(prefix):
                    entries.append([role, line[len(prefix):]])
                    break
            else:
                if entries:                                 # 续行：接到上一条后面
                    entries[-1][1] += "\n" + line
        return [tuple(e) for e in entries]

    # ---------- 一、带对了 ----------

    def test_first_turn_has_no_context(self):
        """第一轮没有历史可带 —— 提示词里不该凭空出现一个空的上下文区。"""
        c = tutor.app.test_client()
        self.ask_turn(c, "第一轮的问题", "第一轮的回答")

        sent = self.user_text()
        self.assertIn("第一轮的问题", sent)
        self.assertNotIn("最近上下文", sent, "第一轮不该有上下文区")

    def test_previous_turn_reaches_the_model(self):
        """【核心】上一轮的问和答都要进上下文 —— 只带问题不够，
        「为什么这样改」要靠上一轮的回答才答得出来。

        【助手那条为什么不是「就是回答原文」】
        数据库里存的是【格式化之后】的整段文字，带「资料来源」那一节。
        所以上下文里也一定带着它 —— 这恰恰是下面
        test_a_source_name_from_history_cannot_be_cited 必须存在的原因：
        历史里躺着一个长得跟真引用一模一样的来源块，绝不能让它变成引用。
        """
        c = tutor.app.test_client()
        self.ask_turn(c, "第一轮的问题", "第一轮的回答")
        self.ask_turn(c, "再给一个例子", "第二轮的回答")

        entries = self.context_entries()
        self.assertEqual([role for role, _ in entries], ["用户", "助手"])
        self.assertEqual(entries[0][1], "第一轮的问题")
        self.assertTrue(entries[1][1].startswith("第一轮的回答"))
        self.assertIn("资料来源：", entries[1][1], "存进历史的应该是格式化后的整段文字")

    def test_context_is_in_chronological_order(self):
        """【核心】交出去的一定是「先问后答」，不能倒着来。"""
        c = tutor.app.test_client()
        for i in range(3):
            self.ask_turn(c, "问题" + str(i), "回答" + str(i))

        roles = [role for role, _ in self.context_entries()]
        self.assertEqual(roles[:2], ["用户", "助手"])
        self.assertEqual(roles, ["用户", "助手"] * 2, "顺序或条数不对")

    def test_at_most_six_messages_reach_the_model(self):
        """【核心】最多 6 条，而且留下的是【最新】那 6 条。"""
        c = tutor.app.test_client()
        for i in range(5):                                  # 5 轮 = 库里 10 条
            self.ask_turn(c, "问题" + str(i), "回答" + str(i))

        self.ask_turn(c, "第六轮的问题", "第六轮的回答")        # 第 6 轮时再看上下文
        entries = self.context_entries()

        self.assertEqual(len(entries), tutor.rag.RECENT_MESSAGE_LIMIT)
        block = " ".join(text for _role, text in entries)
        for i in (2, 3, 4):                                 # 最近 6 条 = 第 2~4 轮
            self.assertIn("问题" + str(i), block)
        for i in (0, 1):                                    # 更老的必须已经出去
            self.assertNotIn("问题" + str(i), block)
            self.assertNotIn("回答" + str(i), block)

    def test_single_message_stays_within_the_per_message_cap(self):
        """【核心】单条历史不超过 1200 字 —— 一条超长回答不能挤爆上下文。"""
        c = tutor.app.test_client()
        self.ask_turn(c, "第一轮的问题", "乙" * 3000)          # 回答远超 1200
        self.ask_turn(c, "再给一个例子", "第二轮的短回答")

        for role, body in self.context_entries():
            self.assertLessEqual(len(body), tutor.rag.MAX_HISTORY_MESSAGE_CHARS,
                                 role + " 那一条超长了")

    def test_total_context_stays_within_the_total_budget(self):
        """【核心】所有历史加起来不超过 4000 字。"""
        c = tutor.app.test_client()
        for i in range(4):
            self.ask_turn(c, "问题" + str(i), "答" * 1500)     # 每轮都被截到 1200
        self.ask_turn(c, "最后一个问题", "ok")

        entries = self.context_entries()
        total = sum(len(body) for _role, body in entries)

        self.assertLessEqual(total, tutor.rag.MAX_HISTORY_TOTAL_CHARS,
                             "上下文总长度超预算：" + str(total))
        self.assertLessEqual(len(entries), tutor.rag.RECENT_MESSAGE_LIMIT)

    # ---------- 二、只带自己的 ----------

    def test_a_session_never_sees_another_sessions_history(self):
        """【核心】A 会话的历史绝不能出现在 B 会话的上下文里。"""
        a = tutor.app.test_client()
        b = tutor.app.test_client()

        self.ask_turn(a, "甲会话的秘密问题", "甲会话的秘密回答")
        self.ask_turn(b, "乙会话的问题", "乙会话的回答")
        self.ask_turn(b, "乙会话的追问", "乙会话的第二次回答")

        sent_b = self.user_text()                           # B 的最后一次调用
        self.assertNotIn("甲会话的秘密问题", sent_b, "B 的上下文里混进了 A 的问题")
        self.assertNotIn("甲会话的秘密回答", sent_b, "B 的上下文里混进了 A 的回答")
        self.assertIn("乙会话的问题", sent_b, "B 自己的历史反而没带进去")

        self.ask_turn(a, "甲会话的追问", "甲会话的第二次回答")
        sent_a = self.user_text()                           # A 的最后一次调用
        self.assertNotIn("乙会话的问题", sent_a, "A 的上下文里混进了 B 的历史")
        self.assertIn("甲会话的秘密问题", sent_a)

    # ---------- 三、只当上下文 ----------

    def test_current_question_is_not_counted_as_history(self):
        """【核心】当前问题还没存库，所以它只该出现在『当前问题』那一处。

        一旦取历史的时机放错（比如先存后取），当前问题就会被当成上下文
        再喂一遍，模型会以为用户在重复问同一件事。
        """
        c = tutor.app.test_client()
        self.ask_turn(c, "第一轮的问题", "第一轮的回答")
        self.ask_turn(c, "再给一个例子", "第二轮的回答")

        sent = self.user_text()
        self.assertEqual(sent.count("再给一个例子"), 1, "当前问题被出现了一次以上")

        context_text = " ".join(text for _role, text in self.context_entries())
        self.assertNotIn("再给一个例子", context_text, "当前问题被塞进上下文了")

    def test_retriever_only_ever_receives_the_current_question(self):
        """【核心】检索只拿当前问题，绝不把历史拼进查询。

        检索决定了「这次能引用什么」。要是把历史也拼进去，
        命中的片段会跟着上一轮漂移，引用就不可控了。
        """
        c = tutor.app.test_client()
        self.ask_turn(c, "第一轮的问题", "第一轮独特回答")
        self.ask_turn(c, "第二轮的问题", "第二轮独特回答")

        self.assertEqual([call["question"] for call in RETRIEVAL_CALLS],
                         ["第一轮的问题", "第二轮的问题"],
                         "检索收到的不是「每轮各自的问题」")
        for call in RETRIEVAL_CALLS:
            self.assertNotIn("独特回答", call["question"], "回答正文被拼进检索查询了")

    def test_a_source_name_from_history_cannot_be_cited(self):
        """【本轮最关键的一条】历史里出现过的来源不算数 —— 引用只认本次片段。

        构造：第一轮的回答正文里提到了 grammar_present_perfect.md；
        第二轮【什么都没检索到】（白名单是空的），模型却引用了那个来源。
        正确行为：白名单拦下 → 安全降级 → 页面和数据库都不能出现那段回答。
        """
        c = tutor.app.test_client()

        # 第一轮：正文里提到一个来源名（general_answer 不带引用，但正文可以提到）
        self.fake.reply = rag_reply(
            decision="general_answer",
            answer="我上次是参考 grammar_present_perfect.md 的「基本结构」讲的",
            citations=[])
        self.ask(c, "第一轮的问题")

        # 第二轮：检索为空，模型从上一轮「记得的」来源里挑了一个来引
        self.use_empty_retrieval()
        self.fake.reply = rag_reply(answer="顺手引一个来源。", citations=[
            {"source": "grammar_present_perfect.md", "heading": "基本结构"}])
        self.ask(c, "再给一个例子")

        rows = self.all_rows()
        self.assertEqual(len(rows), 4)
        second_answer = rows[3][2]                          # 第二轮的 assistant 那条

        self.assertEqual(second_answer, tutor.rag.INSUFFICIENT_TEXT,
                         "引了历史里的来源却当成合法回答存下来了")
        self.assertNotIn("顺手引一个", second_answer)
        self.assertNotIn("资料来源", second_answer)

        html = c.get("/").get_data(as_text=True)
        self.assertNotIn("顺手引一个", html)
        self.assertNotIn("资料来源：", html)

    def test_model_is_called_exactly_once_per_question(self):
        """【核心】有上下文时仍然一次提问 = 一次模型调用。

        上下文是【拼进那一次请求】的，不是「先调一次理解指代、再调一次回答」。
        """
        c = tutor.app.test_client()
        self.ask_turn(c, "问题一", "回答一")
        self.ask_turn(c, "问题二", "回答二")
        self.ask_turn(c, "问题三", "回答三")

        self.assertEqual(len(self.fake.calls), 3, "每轮只该有一次模型调用")

    # ---------- 四、日志与持久化 ----------

    def test_logs_never_contain_history_text(self):
        """【核心】上下文的正文一个字都不许进日志 —— 只记条数和字符数。"""
        c = tutor.app.test_client()

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask_turn(c, "特征历史问题-ZZT-1a", "特征历史回答-ZZT-2b")
            self.ask_turn(c, "再给一个例子", "第二轮的回答")

        blob = "\n".join(captured.output)
        self.assertNotIn("特征历史问题-ZZT-1a", blob, "历史问题正文进了日志")
        self.assertNotIn("特征历史回答-ZZT-2b", blob, "历史回答正文进了日志")
        # 该记的统计量要有，否则「没记」和「没泄露」就分不出来了
        self.assertIn("context_messages=", blob)
        self.assertIn("context_chars=", blob)

    def test_context_survives_an_app_restart(self):
        """【核心】重启后，同一个 cookie 仍能从 SQLite 里取到最近上下文。

        上下文没有存在内存里、也没有单独的会话状态 —— 它就是数据库里的最近几条。
        所以「重启后还认得上一轮」是自动成立的，这条测试把它钉住。
        """
        global tutor
        c = tutor.app.test_client()
        self.ask_turn(c, "重启前的第一轮", "重启前的回答")
        cookie = c.get_cookie("session")
        self.assertIsNotNone(cookie)

        # ---- 模拟重启 ----
        tutor = importlib.reload(tutor)
        tutor.client = FakeClient()
        tutor.retriever.retrieve = fake_retrieve
        self.fake = tutor.client                            # 后面继续用新模块的假客户端

        c2 = tutor.app.test_client()
        c2.set_cookie("session", cookie.value)              # 同一个浏览器
        self.ask_turn(c2, "重启后的追问", "重启后的回答")

        entries = self.context_entries()
        self.assertEqual([role for role, _ in entries], ["用户", "助手"])
        self.assertEqual(entries[0][1], "重启前的第一轮")
        self.assertTrue(entries[1][1].startswith("重启前的回答"))


# ===================== 5e. 网页路径的诊断标签（只进日志）=====================
#
# 【为什么需要这一组】
# 两轮真实上下文验收里，两次最终的 decision 都是 insufficient_evidence，
# 但日志看不出这到底是「模型就是这么判的」，还是「某一关校验没过、被降级下来的」——
# 前者要动提示词，后者要查模型的输出格式或引用，排查方向完全相反。
#
# 这一组把那条区分钉住：decision 照旧对外（页面 + 数据库），
# diagnostic_code 只进安全日志，且永远是固定短枚举。

class TestContextDiagnostics(ChatTestCase):

    def logs_of(self, question="随便问一句"):
        """问一次，把这一轮打的日志和测试客户端一起拿回来。"""
        c = tutor.app.test_client()
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, question)
        return "\n".join(captured.output), c

    # ---------- 六种情况要能分辨 ----------

    def test_a_healthy_answer_logs_ok(self):
        """正常回答 → ok。免得「ok」被误读成「有问题」。"""
        blob, _c = self.logs_of()

        self.assertIn("decision=answer", blob)
        self.assertIn("diagnostic_code=ok", blob)

    def test_valid_insufficient_evidence_logs_ok(self):
        """【核心】模型主动判 insufficient_evidence 且校验通过 → 仍然是 ok。

        这条正好对应两轮真实验收的现象：decision 是 insufficient_evidence，
        但日志里必须能看出【模型就是这么说】，而不是被降级了。
        """
        self.fake.reply = rag_reply(decision="insufficient_evidence",
                                    answer="信息不够完整。", citations=[])
        blob, _c = self.logs_of()

        self.assertIn("decision=insufficient_evidence", blob)
        self.assertIn("diagnostic_code=ok", blob, "模型主动判的被误报成降级")

    def test_invalid_json_is_distinguishable_from_a_real_decision(self):
        """【核心】坏 JSON 降级后 decision 一模一样，但标签必须是 invalid_json。"""
        self.fake.reply = "我觉得应该这样回答：主语 + have + 过去分词。"
        blob, _c = self.logs_of()

        self.assertIn("decision=insufficient_evidence", blob)
        self.assertIn("diagnostic_code=invalid_json", blob)
        self.assertNotIn("diagnostic_code=ok", blob, "降级了却报成 ok")
        self.assertNotIn("我觉得应该这样回答", self.all_rows()[1][2],
                         "坏 JSON 的原文被当成回答存下来了")

    def test_fabricated_citation_logs_invalid_citations(self):
        """编造来源被拦下 → invalid_citations。"""
        self.fake.reply = rag_reply(citations=[{"source": "编造.md", "heading": "编造标题"}])
        blob, _c = self.logs_of()

        self.assertIn("diagnostic_code=invalid_citations", blob)

    def test_general_answer_with_citations_logs_its_own_code(self):
        """通用知识回答却带了引用 → citations_on_general_answer。"""
        self.fake.reply = rag_reply(decision="general_answer", answer="通用回答",
                                    citations=[{"source": "grammar_present_perfect.md",
                                                "heading": "基本结构"}])
        blob, _c = self.logs_of()

        self.assertIn("diagnostic_code=citations_on_general_answer", blob)

    def test_api_error_logs_api_or_response_error(self):
        """接口层就没成功 → api_or_response_error，且不泄露异常原文。"""
        def boom(model, messages, **kwargs):
            raise RuntimeError("接口炸了")
        self.fake.chat.completions.create = boom

        blob, _c = self.logs_of()

        self.assertIn("diagnostic_code=api_or_response_error", blob)
        self.assertNotIn("RuntimeError", blob)
        self.assertNotIn("接口炸了", blob)

    # ---------- 标签只能进日志 ----------

    def test_diagnostic_code_never_reaches_the_page_or_the_database(self):
        """【核心】诊断标签是后厨的东西，页面和数据库都不能出现。"""
        self.fake.reply = "这不是 JSON"
        blob, c = self.logs_of()

        self.assertIn("diagnostic_code=invalid_json", blob)      # 日志里有

        dump = " ".join(str(r) for r in self.all_rows())
        self.assertNotIn("invalid_json", dump, "诊断标签进了数据库")
        self.assertNotIn("diagnostic", dump)

        html = c.get("/").get_data(as_text=True)
        self.assertNotIn("invalid_json", html, "诊断标签进了页面")
        self.assertNotIn("diagnostic", html)

    def test_public_result_still_has_exactly_three_keys(self):
        """【核心】对外结果一个键都不许多 —— 页面/数据库拿到的仍是三键。"""
        self.fake.reply = "这不是 JSON"
        c = tutor.app.test_client()
        self.ask(c, "随便问一句")

        rows = self.all_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][2], tutor.rag.INSUFFICIENT_TEXT)

    # ---------- 上下文与调用次数不受影响 ----------

    def test_context_is_still_used_and_counted(self):
        """带上下文时诊断入口复用同一实现：上下文进提示词，两轮只有两次调用。"""
        c = tutor.app.test_client()
        self.fake.reply = rag_reply(answer="第一轮的回答")
        self.ask(c, "第一轮的问题")

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, "再给一个例子")
        blob = "\n".join(captured.output)

        self.assertIn("context_messages=2", blob)
        self.assertIn("context_chars=", blob)
        self.assertEqual(len(self.fake.calls), 2, "两轮只该有两次模型调用")

        sent = self.fake.calls[-1]["messages"][1]["content"]
        self.assertIn("第一轮的问题", sent, "上下文没有带进去")

    # ---------- 日志里不能有任何正文 ----------

    def test_logs_never_contain_any_content(self):
        """【核心】模型原文、问题、回答、上下文、异常原文、密钥，一个都不许进日志。"""
        marker_q = "特征问题-ZZD-1a"
        marker_ctx = "特征上下文-ZZD-3c"
        marker_exc = "特征异常-ZZD-4d"

        c = tutor.app.test_client()
        self.fake.reply = rag_reply(answer=marker_ctx)          # 第一轮的回答会进第二轮的上下文
        self.ask(c, "第一轮的问题")

        def boom(model, messages, **kwargs):
            raise RuntimeError(marker_exc)
        self.fake.chat.completions.create = boom

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, marker_q)
        blob = "\n".join(captured.output)

        self.assertIn("diagnostic_code=api_or_response_error", blob)   # 该记的记了

        for marker in (marker_q, marker_ctx, marker_exc,
                       tutor.API_KEY, tutor.SECRET_KEY):
            self.assertNotIn(marker, blob, "日志里出现了不该有的内容")


# ===================== 5f. 邀请码进入 =====================
#
# 【这一组守什么】邀请码 = 访问某个学习者全部数据的凭证，和密钥同级。
# 所以四件事必须钉死：
#   1. 正确的码能进；错的码绝不能进（更不能「先建个号再说」）
#   2. 码永远不出现在 URL、页面或日志里
#   3. 尝试要限速（防暴力枚举）
#   4. 输码这个动作要有 CSRF 保护

class TestInviteEntry(ChatTestCase):

    def test_correct_invite_lets_the_user_in(self):
        """【核心】正确的邀请码 → 跳回首页，并且这个浏览器已经绑上学习者。"""
        self.make_invite("good-code")
        c = tutor.app.test_client()

        r = self.enter_invite(c, "good-code")

        self.assertEqual(r.status_code, 302)                     # PRG
        self.assertIsNotNone(self.learner_id_of(c), "输对了码却没有绑定学习者")
        self.assertEqual(self.table_count("learners"), 1)
        self.assertEqual(self.table_count("learner_sessions"), 1)

    def test_wrong_invite_creates_nothing(self):
        """【核心安全断言】错误的邀请码：不建学习者、不绑会话、只给一句提示。"""
        c = tutor.app.test_client()
        r = self.enter_invite(c, "我瞎编的码")

        self.assertEqual(r.status_code, 200)
        self.assertIn(tutor.INVITE_INVALID_MESSAGE, r.get_data(as_text=True))
        self.assertEqual(self.table_count("learners"), 0, "错误的码建出了学习者")
        self.assertEqual(self.table_count("learner_sessions"), 0)
        self.assertIsNone(self.learner_id_of(c))

    def test_revoked_invite_is_refused_with_its_own_message(self):
        """被作废的码要给一句不同的话（用户才知道该找管理员换一个）。"""
        self.make_invite("dead", status=tutor.profile_store.INVITE_REVOKED)
        c = tutor.app.test_client()
        r = self.enter_invite(c, "dead")

        self.assertIn(tutor.INVITE_REVOKED_MESSAGE, r.get_data(as_text=True))
        self.assertEqual(self.table_count("learners"), 0)

    def test_the_code_never_appears_in_the_page_or_the_url(self):
        """【核心安全断言】邀请码不进 URL，也不回显到页面上。

        URL 会被浏览器历史、代理和服务器访问日志记下来。
        """
        code = "secret-code-4f2a"
        self.make_invite(code)
        c = tutor.app.test_client()

        r = self.enter_invite(c, code)
        self.assertNotIn(code, r.headers.get("Location", ""), "邀请码出现在跳转地址里")

        html = c.get("/").get_data(as_text=True)
        self.assertNotIn(code, html)

        wrong = self.enter_invite(c, "另一个瞎编的码")
        self.assertNotIn("另一个瞎编的码", wrong.get_data(as_text=True), "输入框回显了邀请码")

    def test_neither_the_code_nor_its_digest_nor_the_pepper_reach_the_logs(self):
        """【核心安全断言】明文码、摘要、pepper，一个都不许进日志。"""
        code = "log-secret-code-7b1c"
        self.make_invite(code)
        c = tutor.app.test_client()

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.enter_invite(c, code)                    # 成功那条路径
            self.enter_invite(c, "不存在的码")             # 失败那条路径
        blob = "\n".join(captured.output)

        digest = tutor.profile_store.digest_invite_code(code, tutor.INVITE_CODE_PEPPER)
        self.assertNotIn(code, blob, "邀请码明文进了日志")
        self.assertNotIn(digest, blob, "邀请码摘要进了日志")
        self.assertNotIn(tutor.INVITE_CODE_PEPPER, blob, "pepper 进了日志")

        # 该记的还是要记，否则「没记」和「没泄露」分不清
        self.assertIn("invite_accepted", blob)
        self.assertIn("invite_rejected", blob)

    def test_invite_attempts_are_rate_limited(self):
        """【核心】连续输错到上限后 → 被挡住，而且始终没有创建任何学习者。"""
        c = tutor.app.test_client()

        for i in range(tutor.INVITE_MAX_FAILURES):
            r = self.enter_invite(c, "错的码" + str(i))
            self.assertIn(tutor.INVITE_INVALID_MESSAGE, r.get_data(as_text=True),
                          "第 " + str(i + 1) + " 次就不给「无效」提示了？")

        blocked = self.enter_invite(c, "再试一次")
        self.assertIn(tutor.INVITE_RATE_MESSAGE, blocked.get_data(as_text=True))

        # 限速期间就算输对了也不放行 —— 否则限速形同虚设
        self.make_invite("good-code")
        still_blocked = self.enter_invite(c, "good-code")
        self.assertIn(tutor.INVITE_RATE_MESSAGE, still_blocked.get_data(as_text=True))
        self.assertEqual(self.table_count("learners"), 0)

    def test_a_successful_entry_is_not_counted_as_a_failure(self):
        """输对的人不该被自己之前的失败次数连累到「锁死」——
        这里验证的是：成功的路径不写失败记录。"""
        self.make_invite("good-code")
        c = tutor.app.test_client()
        self.enter_invite(c, "good-code")

        self.assertEqual(tutor._invite_failures, {}, "成功的尝试被记成了失败")

    def test_invite_requires_a_csrf_token(self):
        """【核心】没有 token（或 token 不对）→ 拒绝，且什么也不建。"""
        self.make_invite("good-code")
        c = tutor.app.test_client()
        c.get("/invite")                                   # 先建立会话，但【不】取 token

        for data in ({"invite_code": "good-code"},                     # 完全没有 token
                     {"invite_code": "good-code", "csrf_token": "伪造的"}):   # 伪造的 token
            r = c.post("/invite", data=data)
            with self.subTest(data=data):
                self.assertIn(tutor.CSRF_MESSAGE, r.get_data(as_text=True))

        self.assertEqual(self.table_count("learners"), 0, "CSRF 没拦住，居然建了学习者")


# ===================== 5g. 档案页与用户隔离 =====================

class TestProfilePage(ChatTestCase):

    def bind(self, code="c1"):
        """建一张邀请码，并让这个浏览器绑上去。返回测试客户端。"""
        self.make_invite(code)
        c = tutor.app.test_client()
        self.enter_invite(c, code)
        return c

    # ---------- 访问控制 ----------

    def test_an_anonymous_visitor_cannot_open_the_profile_page(self):
        """【核心安全断言】没绑定学习档案的人，档案页不对他开放。"""
        c = tutor.app.test_client()
        r = c.get("/profile")

        self.assertEqual(r.status_code, 302)
        self.assertIn("/invite", r.headers["Location"], "应该被引导去输邀请码")
        self.assertNotIn("我的学习档案", r.get_data(as_text=True))

    def test_a_bound_user_can_open_the_profile_page(self):
        c = self.bind()
        r = c.get("/profile")

        self.assertEqual(r.status_code, 200)
        self.assertIn("我的学习档案", r.get_data(as_text=True))

    def test_the_page_says_so_when_nothing_is_set_yet(self):
        """还没填过 → 页面必须如实说「你还没有设置过」，而不是假装这是用户设的。"""
        c = self.bind()
        html = c.get("/profile").get_data(as_text=True)

        self.assertIn("还没有设置过", html)

    # ---------- 填 / 改 ----------

    def test_a_user_can_save_their_profile(self):
        c = self.bind()
        r = self.save_profile(c, level_code="b2", language_mode="en_only",
                              length_mode="brief", goal_code="exam", focus_code="writing")

        self.assertEqual(r.status_code, 302)               # PRG
        row = self.profile_row(self.learner_id_of(c))
        self.assertEqual(row, ("b2", 0, "en_only", "brief", "exam", "writing"))

    def test_a_user_can_modify_their_profile(self):
        """【核心】改档案是覆盖同一行，不是又插一行。"""
        c = self.bind()
        self.save_profile(c, level_code="a2", language_mode="zh_pair", length_mode="normal")
        self.save_profile(c, level_code="c1", language_mode="en_advanced", length_mode="detailed")

        learner = self.learner_id_of(c)
        self.assertEqual(self.table_count("learner_preferences"), 1, "改档案时插出了第二行")
        row = self.profile_row(learner)
        self.assertEqual(row[0], "c1")
        self.assertEqual(row[2], "en_advanced")
        self.assertEqual(row[3], "detailed")

        html = c.get("/profile").get_data(as_text=True)
        self.assertIn('value="c1" selected', html, "页面没有回显改后的水平")

    def test_the_uncertain_flag_is_stored(self):
        c = self.bind()
        self.save_profile(c, level_uncertain="1")
        self.assertEqual(self.profile_row(self.learner_id_of(c))[1], 1)

        self.save_profile(c)                               # 不勾 = 0
        self.assertEqual(self.profile_row(self.learner_id_of(c))[1], 0)

    def test_an_invalid_option_is_rejected_without_writing_anything(self):
        """【核心安全断言】用户自己造一个选项值 → 拒绝，什么都不写。"""
        c = self.bind()
        r = self.save_profile(c, level_code="z9")

        self.assertIn(tutor.PROFILE_INVALID_MESSAGE, r.get_data(as_text=True))
        self.assertEqual(self.table_count("learner_preferences"), 0, "非法选项被写库了")

    def test_saving_the_profile_requires_a_csrf_token(self):
        """【核心】没有 token → 拒绝，档案一个字都没改。"""
        c = self.bind()
        self.save_profile(c, level_code="b2")              # 先存一版

        r = c.post("/profile", data={"level_code": "c1", "language_mode": "en_only",
                                     "length_mode": "detailed"})     # 不带 token

        self.assertIn(tutor.CSRF_MESSAGE, r.get_data(as_text=True))
        self.assertEqual(self.profile_row(self.learner_id_of(c))[0], "b2", "档案被改了")

    # ---------- 用户之间严格隔离 ----------

    def test_two_users_profiles_are_isolated(self):
        """【核心安全断言】A 看不到 B 的档案，也改不了 B 的档案。"""
        a = self.bind("code-a")
        b = self.bind("code-b")

        self.save_profile(a, level_code="a1", language_mode="zh_pair",
                          length_mode="brief", goal_code="exam", focus_code="writing")
        self.save_profile(b, level_code="c1", language_mode="en_advanced",
                          length_mode="detailed", goal_code="work", focus_code="speaking")

        learner_a = self.learner_id_of(a)
        learner_b = self.learner_id_of(b)
        self.assertNotEqual(learner_a, learner_b, "两个浏览器拿到了同一个 learner_id")

        # 数据库层面：各写各的行
        self.assertEqual(self.profile_row(learner_a),
                         ("a1", 0, "zh_pair", "brief", "exam", "writing"))
        self.assertEqual(self.profile_row(learner_b),
                         ("c1", 0, "en_advanced", "detailed", "work", "speaking"))

        # 页面层面：A 的档案页里不能出现 B 的选择
        html_a = a.get("/profile").get_data(as_text=True)
        self.assertIn('value="a1" selected', html_a)
        self.assertNotIn('value="c1" selected', html_a, "A 的页面里出现了 B 的水平")
        self.assertNotIn('value="en_advanced" selected', html_a, "A 的页面里出现了 B 的语言偏好")

        html_b = b.get("/profile").get_data(as_text=True)
        self.assertIn('value="c1" selected', html_b)
        self.assertNotIn('value="a1" selected', html_b)

    def test_a_cannot_touch_bs_profile_even_with_the_right_csrf_token(self):
        """【核心安全断言】CSRF token 只能证明「这个表单是我自己提交的」，
        **绝不能**证明「我有权改这个 learner」。

        这里的机制是：路由压根不接受「改哪个 learner」这个参数 ——
        learner_id 只从服务端会话反查。所以 A 拿着自己的合法 token，
        也只能改到自己那一行。
        """
        a = self.bind("code-a")
        b = self.bind("code-b")
        self.save_profile(b, level_code="c1", language_mode="en_only", length_mode="detailed")

        # A 用【自己的】合法 token 提交，并试图在表单里塞一个 learner_id
        a.post("/profile", data={
            "csrf_token": self.csrf_token(a, "/profile"),
            "learner_id": str(self.learner_id_of(b)),      # 试图指向 B
            "level_code": "a1", "language_mode": "zh_pair", "length_mode": "brief",
        })

        self.assertEqual(self.profile_row(self.learner_id_of(b)),
                         ("c1", 0, "en_only", "detailed", None, None),
                         "B 的档案被 A 改掉了")

    def test_an_invite_holder_cannot_take_over_an_existing_session(self):
        """【核心安全断言】已经绑过 A 的浏览器，再输一次 B 的邀请码也不能改绑。

        否则 B 只要拿到自己的邀请码，就能把 A 的浏览器会话抢过来。
        """
        self.make_invite("code-a")
        self.make_invite("code-b")
        c = tutor.app.test_client()
        self.enter_invite(c, "code-a")
        learner_a = self.learner_id_of(c)

        self.enter_invite(c, "code-b")                     # 同一个浏览器再输一张码

        self.assertEqual(self.learner_id_of(c), learner_a, "会话被改绑到另一个学习者了")
        self.assertEqual(self.table_count("learner_sessions"), 1)

    # ---------- 重启后仍在 ----------

    def test_the_profile_survives_an_app_restart(self):
        """【核心】重启应用后，同一个 cookie 仍能读到自己的档案。"""
        global tutor
        c = self.bind("code-a")
        self.save_profile(c, level_code="b2", language_mode="en_advanced",
                          length_mode="brief", goal_code="work", focus_code="listening")
        cookie = c.get_cookie("session")
        self.assertIsNotNone(cookie)

        # ---- 模拟重启 ----
        tutor = importlib.reload(tutor)
        tutor.client = FakeClient()
        tutor.retriever.retrieve = fake_retrieve

        c2 = tutor.app.test_client()
        c2.set_cookie("session", cookie.value)             # 同一个浏览器

        html = c2.get("/profile").get_data(as_text=True)
        self.assertIn('value="b2" selected', html)
        self.assertIn('value="en_advanced" selected', html)
        self.assertIn('value="brief" selected', html)
        self.assertIn('value="listening" selected', html)
        self.assertIn("你设置过档案", html)


# ===================== 5h. 档案真的进了提示词 =====================

class TestLengthModeGating(ChatTestCase):
    """「详细」和「精简」这两档，必须**真的送到模型面前**，而且不能互相串味。

    【为什么单独测这一件事】
    rag.py 只管把档案代号翻成人话；「这一档到底有没有生效」要看
    app.py → rag 这一路拼出来的提示词。所以这里走网页全流程，检查
    假客户端**实际收到**的那段 user 消息。

    【能证明什么、不能证明什么】
    能证明：档案里选的档位进了提示词，而且「详细」才带那条补搭配的口径。
    证明不了：模型照不照做 —— 假客户端不懂教学（见 test_rag.TestAnswerContentGuidance）。
    """

    def prompt_text(self, index=-1):
        """取出发给模型的 user 消息正文。"""
        return self.fake.calls[index]["messages"][1]["content"]

    def system_text(self):
        return self.fake.calls[-1]["messages"][0]["content"]

    def bind(self, code="gate1"):
        self.make_invite(code)
        c = tutor.app.test_client()
        self.enter_invite(c, code)
        return c

    def test_detailed_mode_carries_the_extension_wording(self):
        """选了「详细」→ 提示词里的那一行要带上「补搭配或近义表达」这个口径。"""
        c = self.bind()
        self.save_profile(c, length_mode="detailed")
        self.ask(c, "affect 和 effect 有什么区别？")

        sent = self.prompt_text()
        self.assertIn("详细", sent)
        self.assertIn("搭配或近义表达", sent)
        self.assertIn("不要为了长而长", sent)

    def test_brief_mode_does_not_carry_the_extension_wording(self):
        """【护住 B03】选了「精简」→ 不许把「补搭配」那条口径带进这次请求。"""
        c = self.bind()
        self.save_profile(c, length_mode="brief")
        self.ask(c, "affect 和 effect 有什么区别？")

        sent = self.prompt_text()
        self.assertIn("精简", sent)
        self.assertNotIn("搭配或近义表达", sent)

    def test_the_brief_exemption_is_decided_in_the_system_prompt(self):
        """「精简就不加延伸」这条规则常驻 system 提示词 —— 模型每道题都看得到。"""
        c = self.bind()
        self.save_profile(c, length_mode="brief")
        self.ask(c, "affect 和 effect 有什么区别？")

        system = self.system_text()
        self.assertIn("不要加延伸", system)


class TestProfileReachesTheModel(ChatTestCase):

    def prompt_text(self, index=-1):
        """取出发给模型的 user 消息正文。"""
        return self.fake.calls[index]["messages"][1]["content"]

    def bind(self, code="c1"):
        self.make_invite(code)
        c = tutor.app.test_client()
        self.enter_invite(c, code)
        return c

    def test_the_profile_is_sent_to_the_model(self):
        """【核心】档案里的每一项，都要能在发给模型的内容里找到对应的说法。"""
        c = self.bind()
        self.save_profile(c, level_code="c1", language_mode="en_advanced",
                          length_mode="brief", goal_code="work", focus_code="speaking")

        self.ask(c, "帮我改一下这句话")

        sent = self.prompt_text()
        self.assertIn("学习者档案", sent)
        self.assertIn("C1（高级）", sent)
        self.assertIn("全英文 + 高级表达", sent)
        self.assertIn("精简", sent)
        self.assertIn("工作 / 商务", sent)
        self.assertIn("口语", sent)

    def test_the_uncertain_level_is_flagged_to_the_model(self):
        """用户勾了「不确定」→ 提示词里要提醒模型「这个水平很可能不准」。"""
        c = self.bind()
        self.save_profile(c, level_code="c1", level_uncertain="1")
        self.ask(c, "随便问一句")

        self.assertIn("很可能不准", self.prompt_text())

    def test_an_anonymous_visitor_gets_no_profile_block_at_all(self):
        """【核心】匿名访客的提示词里【不该】出现档案那一段 ——
        他的体验和加这个功能之前完全一样。"""
        c = tutor.app.test_client()
        self.ask(c, "随便问一句")

        sent = self.prompt_text()
        self.assertNotIn("学习者档案", sent)
        self.assertIn("随便问一句", sent)

    def test_an_anonymous_visitor_still_gets_answers(self):
        """【核心】没有档案照样能正常回答：302、存库、照样检索、照样调模型。"""
        c = tutor.app.test_client()
        r = self.ask(c, "现在完成时的句子结构是怎样的？")

        self.assertEqual(r.status_code, 302)
        self.assertEqual(len(self.all_rows()), 2)
        self.assertEqual(len(RETRIEVAL_CALLS), 1)
        self.assertEqual(len(self.fake.calls), 1)

    def test_a_user_without_a_saved_profile_gets_the_defaults(self):
        """绑定了但还没填 → 用默认值回答（不能报错，也不能空着不谈）。"""
        c = self.bind()
        r = self.ask(c, "随便问一句")

        self.assertEqual(r.status_code, 302)
        self.assertIn("学习者档案", self.prompt_text())
        self.assertIn("中英对照", self.prompt_text())      # 默认语言模式

    def test_the_answer_language_rule_and_the_chosen_mode_both_reach_the_model(self):
        """【本轮新增】语言这件事有两半，都要出现在假客户端收到的东西里：

        ① system 消息里那条规则 —— 语言按档案偏好走、没有档案时默认中文；
        ② user 消息里这一档的具体偏好行。

        【能证明什么】两半确实都发出去了。
        【证明不了】模型会不会照它选语言 —— 假客户端不懂教学，它返回什么是测试写死的。
        """
        c_en = self.bind("lang-en")
        self.save_profile(c_en, language_mode="en_only")
        self.ask(c_en, "随便问一句")
        system = self.fake.calls[-1]["messages"][0]["content"]
        self.assertIn("语言按上面的档案语言偏好走", system)
        self.assertIn("默认用中文", system)
        self.assertIn("全英文", self.prompt_text())

        c_pair = self.bind("lang-pair")
        self.save_profile(c_pair, language_mode="zh_pair")
        self.ask(c_pair, "随便问一句")
        self.assertIn("中英对照", self.prompt_text())

    def test_the_bilingual_dedup_wording_reaches_both_messages(self):
        """【本轮新增】中英对照的新口径，两半都要在假客户端收到的东西里：

        ① user 消息里那一行语言偏好（英文内容照常给 + 不要整段复述）；
        ② system 消息里那条通用去重规则（单位是句子和短语）。

        【能证明什么】两条都确实发出去了。
        【证明不了】模型会不会照做 —— 假客户端不懂教学，它返回什么由测试写死。
        """
        c = self.bind("dedup-zh")
        self.save_profile(c, language_mode="zh_pair")
        self.ask(c, "帮我改一下这句话：I am afraid of fly.")

        user = " ".join(self.prompt_text().split())
        self.assertIn("中英对照", user)
        self.assertIn("不是重复", user)
        self.assertIn("不要先写一整段英文讲解", user)

        system = " ".join(self.fake.calls[-1]["messages"][0]["content"].split())
        self.assertIn("单位是句子和短语，不是整篇回答", system)
        self.assertIn("例句和搭配照常给", system)
        self.assertIn("一处都不能省", system)

    def test_the_profile_cannot_become_a_citation_source(self):
        """【核心安全断言】档案**不是资料**。

        模型如果顺着档案里提到的来源去引用，仍然会被白名单拦下 ——
        白名单只从本次检索到的片段建。
        """
        c = self.bind()
        self.save_profile(c, focus_code="writing")

        # 本次检索什么都没捞到（白名单是空的），模型却引用了一个来源
        self.use_empty_retrieval()
        self.fake.reply = rag_reply(answer="顺手引一个。", citations=[
            {"source": "grammar_present_perfect.md", "heading": "基本结构"}])
        self.ask(c, "帮我改一下这句话")

        rows = self.all_rows()
        self.assertEqual(rows[1][2], tutor.rag.INSUFFICIENT_TEXT)
        self.assertNotIn("资料来源", rows[1][2])

    def test_profile_values_never_leak_into_the_logs(self):
        """档案的内容也不该进日志 —— 日志只记统计量。"""
        c = self.bind()
        self.save_profile(c, goal_code="exam", focus_code="vocabulary")

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, "随便问一句")
        blob = "\n".join(captured.output)

        self.assertIn("diagnostic_code=", blob)            # 该记的还在
        self.assertNotIn("考试 / 升学", blob)                # 档案的中文说法不该出现
        self.assertNotIn("vocabulary", blob)

    def test_only_one_model_call_with_profile_and_context(self):
        """档案 + 上下文一起带上时，依然是一次提问 = 一次模型调用。"""
        c = self.bind()
        self.save_profile(c, level_code="b2")
        self.ask(c, "第一轮的问题")
        self.ask(c, "再给一个例子")

        self.assertEqual(len(self.fake.calls), 2)
        sent = self.prompt_text()
        self.assertIn("学习者档案", sent)
        self.assertIn("最近上下文", sent)


# ===================== 5i. 清空学习档案（只清偏好）=====================
#
# 【它和「清空全部个人数据」不是一回事】只清那五项偏好。
# 学习身份、邀请码、聊天记录、会话绑定都不动 —— 清完还能用同一张码继续用。

class TestClearProfile(ChatTestCase):

    def bind_with_profile(self, code="c1"):
        """建码 → 绑定 → 存一份偏好，返回测试客户端。"""
        self.make_invite(code)
        c = tutor.app.test_client()
        self.enter_invite(c, code)
        self.save_profile(c, level_code="c1", language_mode="en_advanced",
                          length_mode="brief", goal_code="work", focus_code="speaking")
        return c

    def clear(self, client, confirm="yes", csrf=None):
        """提交清空表单。默认带齐 token 和确认勾选。"""
        if csrf is None:
            csrf = self.csrf_token(client, "/profile")
        data = {"csrf_token": csrf}
        if confirm is not None:
            data["confirm"] = confirm
        return client.post("/profile/clear", data=data)

    # ---------- 正常路径 ----------

    def test_a_user_can_clear_their_own_profile(self):
        """【核心】清空之后：偏好没了，身份和邀请码还在。"""
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)

        r = self.clear(c)

        self.assertEqual(r.status_code, 302)               # PRG
        self.assertIsNone(self.profile_row(learner), "偏好没清掉")
        self.assertEqual(self.table_count("learners"), 1, "学习者被删了")
        self.assertEqual(self.table_count("invites"), 1, "邀请码被牵连了")
        self.assertEqual(self.table_count("learner_sessions"), 1, "会话绑定被牵连了")
        self.assertIsNotNone(self.learner_id_of(c), "清空之后自己反而进不去了")

    def test_the_page_says_not_set_after_clearing(self):
        """清完之后页面必须回到「还没有设置过」。"""
        c = self.bind_with_profile()
        self.assertIn("你设置过档案", c.get("/profile").get_data(as_text=True))

        self.clear(c)

        html = c.get("/profile").get_data(as_text=True)
        self.assertIn("还没有设置过", html)
        self.assertNotIn("你设置过档案", html)

    def test_the_next_answer_uses_the_defaults_not_the_old_profile(self):
        """【核心】清空之后，模型【绝不能】再收到旧偏好。"""
        c = self.bind_with_profile()
        self.clear(c)

        self.ask(c, "帮我改一下这句话")

        sent = self.fake.calls[-1]["messages"][1]["content"]
        self.assertIn("中英对照", sent, "没有退回默认的语言偏好")
        self.assertNotIn("en_advanced", sent)
        self.assertNotIn("全英文 + 高级表达", sent, "旧偏好还在提示词里")
        self.assertNotIn("工作 / 商务", sent, "旧的学习目标还在提示词里")
        self.assertNotIn("口语", sent, "旧的重点方向还在提示词里")

    def test_the_user_can_save_a_profile_again_after_clearing(self):
        c = self.bind_with_profile()
        self.clear(c)

        self.save_profile(c, level_code="a2", language_mode="zh_pair", length_mode="detailed")

        row = self.profile_row(self.learner_id_of(c))
        self.assertEqual(row[:4], ("a2", 0, "zh_pair", "detailed"))
        self.assertIsNone(row[4], "旧的目标残留了")
        self.assertIsNone(row[5], "旧的重点方向残留了")

    # ---------- 表单结构：确认必须由用户真的勾出来 ----------

    def test_the_checkbox_itself_carries_the_confirm_field(self):
        """【核心】confirm 必须挂在复选框上，不能是一个隐藏字段。

        隐藏字段的坑：`<input type="hidden" name="confirm" value="yes">` 会让浏览器
        **不管用户勾没勾都自动提交 confirm=yes** ——
        服务端那一关就永远通过，复选框只剩视觉效果，等于没有二次确认。
        """
        c = self.bind_with_profile()
        html = c.get("/profile").get_data(as_text=True)

        self.assertNotIn('type="hidden" name="confirm"', html,
                         "又出现了隐藏的 confirm 字段 —— 二次确认会被绕过")
        self.assertIn('name="confirm" value="yes"', html, "复选框没有带上 confirm 字段")

    def test_not_checking_the_box_is_really_submitted_as_absent(self):
        """模拟一次真实的浏览器提交：没勾复选框 → 表单里压根没有 confirm 这个键。

        上面的结构测试证明「字段挂在复选框上」；这条证明**服务端确实会因此拒绝**。
        """
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)

        # 浏览器在复选框未勾选时，不会提交这个字段 —— 只带 csrf_token
        r = c.post("/profile/clear",
                   data={"csrf_token": self.csrf_token(c, "/profile")})

        self.assertIn(tutor.CLEAR_CONFIRM_MESSAGE, r.get_data(as_text=True))
        self.assertIsNotNone(self.profile_row(learner), "没勾确认也把档案清了")

    def test_checking_the_box_with_a_valid_token_clears_it(self):
        """勾了 + CSRF 有效 → 清空成功（正向路径，和上面那条配成一对）。"""
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)

        r = c.post("/profile/clear", data={
            "csrf_token": self.csrf_token(c, "/profile"),
            "confirm": "yes",                       # 复选框勾上时浏览器会带上这个
        })

        self.assertEqual(r.status_code, 302)
        self.assertIsNone(self.profile_row(learner))

    # ---------- 两道闸门 ----------

    def test_clearing_needs_the_confirmation_checkbox(self):
        """【核心】没有勾确认 → 拒绝，档案一个字都没动。"""
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)

        r = self.clear(c, confirm=None)

        self.assertIn(tutor.CLEAR_CONFIRM_MESSAGE, r.get_data(as_text=True))
        self.assertIsNotNone(self.profile_row(learner), "没勾确认也把档案清了")

    def test_a_forged_confirmation_value_is_not_enough_without_csrf(self):
        """confirm=yes 谁都能写，真正管用的是 CSRF token。"""
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)

        r = c.post("/profile/clear", data={"confirm": "yes"})     # 有确认、没 token

        self.assertIn(tutor.CSRF_MESSAGE, r.get_data(as_text=True))
        self.assertIsNotNone(self.profile_row(learner), "没有 CSRF 也把档案清了")

    def test_an_anonymous_visitor_cannot_clear_anything(self):
        """没绑定的人连清空入口都到不了。"""
        c = tutor.app.test_client()
        c.get("/profile")                                  # 建立会话

        r = c.post("/profile/clear", data={"confirm": "yes"})

        self.assertEqual(r.status_code, 302)
        self.assertIn("/invite", r.headers["Location"])

    def test_a_cannot_clear_bs_profile(self):
        """【核心隔离断言】A 清空自己的档案，B 的档案一个字都不能变。"""
        a = self.bind_with_profile("code-a")
        b = self.bind_with_profile("code-b")
        learner_b = self.learner_id_of(b)
        b_before = self.profile_row(learner_b)

        # A 用【自己的】合法 token 提交，并在表单里塞上 B 的编号，试图指向 B
        a.post("/profile/clear", data={
            "csrf_token": self.csrf_token(a, "/profile"),
            "confirm": "yes",
            "learner_id": str(learner_b),
        })

        self.assertEqual(self.profile_row(learner_b), b_before, "B 的档案被 A 清掉了")

    def test_the_error_message_does_not_leak_internals(self):
        c = self.bind_with_profile()
        r = self.clear(c, confirm=None, csrf="伪造的 token")

        body = r.get_data(as_text=True)
        for leak in ("Traceback", "sqlite", "learner_id", tutor.INVITE_CODE_PEPPER):
            self.assertNotIn(leak, body)


# ===================== 5j. 作废邀请码之后的实际访问行为 =====================
#
# 【为什么必须有这一组】「作废」如果只是把状态字段改一下，
# 那它是**假的作废**：已经进来的设备照样能一直看档案。
# 这里从网页这一侧证明：作废之后，那台设备真的失去访问权；
# 而补发新码之后，本人真的能拿回原来那份档案。

class TestRevokedInviteAccess(ChatTestCase):

    def test_a_revoked_invite_cuts_off_the_bound_browser(self):
        """【核心】作废 → 已绑定的浏览器访问档案页时被引导回输码页。"""
        self.make_invite("c1")
        c = tutor.app.test_client()
        self.enter_invite(c, "c1")
        self.save_profile(c, level_code="c1", language_mode="en_only", length_mode="detailed")
        self.assertEqual(c.get("/profile").status_code, 200)     # 作废前：进得去

        result = self.revoke("c1")

        self.assertEqual(result["sessions_dropped"], 1)
        r = c.get("/profile")
        self.assertEqual(r.status_code, 302, "作废之后设备仍然能进档案页")
        self.assertIn("/invite", r.headers["Location"])
        self.assertIsNone(self.learner_id_of(c), "会话绑定没有被断开")

    def test_revoking_does_not_delete_the_profile_or_the_chat(self):
        """【核心】断开通道 ≠ 销毁数据：偏好和聊天记录都还在。"""
        self.make_invite("c1")
        c = tutor.app.test_client()
        self.enter_invite(c, "c1")
        self.save_profile(c, level_code="c1", language_mode="en_only", length_mode="detailed")
        learner = self.learner_id_of(c)
        self.ask(c, "作废之前问过的话")
        before = self.profile_row(learner)

        self.revoke("c1")

        self.assertEqual(self.profile_row(learner), before, "偏好被删了")
        self.assertEqual(self.table_count("messages"), 2, "聊天记录被删了")
        self.assertEqual(self.table_count("learners"), 1, "学习者被删了")

    def test_the_revoked_browser_no_longer_shows_its_old_history(self):
        """【行为变化，如实钉住】作废之后，那台浏览器连自己的旧聊天记录也不再显示。

        【为什么变了】作废会让它的会话失效，而失效的会话会被【轮换】成一个全新的
        session_id（见 app.py 的 _ensure_session_id）。新 id 下面什么都没有，
        所以页面显示「还没有对话」。

        【数据丢了吗 —— 没有，这一点很重要】
        旧消息仍然挂在【旧】session_id 下，而那一行 learner_sessions 还在
        （revoked_at 有值、learner_id 仍然指着他），所以以后「清空全部个人数据」
        时照样找得到、删得掉。页面看不见 ≠ 库里没有。
        """
        self.make_invite("c1")
        c = tutor.app.test_client()
        self.enter_invite(c, "c1")
        old_sid = self.session_id_of(c)
        self.ask(c, "作废之前问过的话")

        self.revoke("c1")

        html = c.get("/").get_data(as_text=True)
        self.assertNotIn("作废之前问过的话", html, "旧历史还在显示")
        self.assertIn("还没有对话", html)

        self.assertNotEqual(self.session_id_of(c), old_sid, "会话没有被轮换")
        self.assertEqual(self.messages_for_session_id(old_sid), 2,
                         "旧消息不该被删掉 —— 它们还要留给「清空全部数据」去删")

    def test_a_revoked_code_cannot_be_used_again(self):
        self.make_invite("c1")
        c = tutor.app.test_client()
        self.enter_invite(c, "c1")
        self.revoke("c1")

        fresh = tutor.app.test_client()
        r = self.enter_invite(fresh, "c1")

        self.assertIn(tutor.INVITE_REVOKED_MESSAGE, r.get_data(as_text=True))
        self.assertIsNone(self.learner_id_of(fresh))

    def test_the_user_gets_the_same_profile_back_with_a_replacement_invite(self):
        """【核心】恢复流程：作废 → 补发 → 在另一台设备输新码 → 档案回来了。"""
        self.make_invite("c1")
        c = tutor.app.test_client()
        self.enter_invite(c, "c1")
        self.save_profile(c, level_code="b2", language_mode="en_advanced",
                          length_mode="brief", goal_code="exam", focus_code="listening")
        learner = self.learner_id_of(c)

        self.revoke("c1")                                  # 码泄露了，先止血
        new_code = self.reissue(learner)                   # 补发一张给他
        self.assertIsNotNone(new_code)

        other_device = tutor.app.test_client()             # 另一台设备
        r = self.enter_invite(other_device, new_code)

        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.learner_id_of(other_device), learner, "没有回到原来那个学习者")
        self.assertEqual(self.table_count("learners"), 1, "补发时又建了一个学习者")

        html = other_device.get("/profile").get_data(as_text=True)
        self.assertIn('value="b2" selected', html)
        self.assertIn('value="en_advanced" selected', html)
        self.assertIn('value="listening" selected', html)

    def test_a_replacement_invite_is_never_shown_on_the_page(self):
        """补发的码同样不能回显到页面上。"""
        self.make_invite("c1")
        c = tutor.app.test_client()
        self.enter_invite(c, "c1")
        learner = self.learner_id_of(c)

        new_code = self.reissue(learner)
        html = c.get("/").get_data(as_text=True)

        self.assertNotIn(new_code, html)


# ===================== 5k. 文案的诚实性（防回归）=====================
#
# 【为什么文案也要有测试】这几句话是**对用户的承诺**。
# 说过头了，用户会基于一个假前提交出自己的数据；说少了，用户不知道自己在冒什么风险。
# 设计文档第 11 节把「不用于训练」该怎么说、不该怎么说写得很死，这里把那条线钉住 ——
# 免得以后有人好心把它「改得更好听一点」，又改回一句绝对承诺。

class TestPrivacyCopy(ChatTestCase):

    def bound_client(self):
        self.make_invite("c1")
        c = tutor.app.test_client()
        self.enter_invite(c, "c1")
        return c

    # ---------- 档案页 ----------

    def test_the_profile_page_does_not_promise_absolute_no_training(self):
        """【核心】不能绝对承诺「不用于训练」—— 生成回答要经过第三方。"""
        html = self.bound_client().get("/profile").get_data(as_text=True)

        self.assertNotIn("不用于训练", html, "又出现了那句绝对承诺")
        self.assertNotIn("绝不会被用于训练", html)
        self.assertNotIn("绝不用于训练", html)

    def test_the_profile_page_states_the_third_party_provider_honestly(self):
        """必须说清两件事：我们自己不拿来训练；但必要内容会发给第三方。"""
        html = self.bound_client().get("/profile").get_data(as_text=True)

        self.assertIn("不主动用你的内容训练自己的模型", html)
        self.assertIn("第三方模型服务商", html)
        self.assertIn("隐私政策", html, "没有把「对方怎么用我们管不了」说清楚")

    def test_the_profile_page_does_not_claim_it_is_tied_to_one_browser_session(self):
        """【核心】不能说「只和这一个浏览器会话绑定」——
        换设备后重新输同一张有效邀请码，是能回到同一份档案的。"""
        html = self.bound_client().get("/profile").get_data(as_text=True)

        self.assertNotIn("只和这一个浏览器会话绑定", html)
        self.assertIn("重新输入同一张有效邀请码", html)
        self.assertIn("回到同一份档案", html)

    def test_the_profile_page_says_the_code_is_a_credential(self):
        """谁拿到码谁就能看档案 —— 这一点必须说出来。"""
        html = self.bound_client().get("/profile").get_data(as_text=True)

        self.assertIn("谁拿到这张码，谁就能访问这份档案", html)

    def test_the_clear_section_explains_what_is_deleted_and_what_is_kept(self):
        """【核心】清空按钮必须写清删除范围，而且要把它和「清空全部个人数据」区分开。

        注意：这一块只在「已经设置过档案」时才渲染，所以这里要先存一份。
        """
        c = self.bound_client()
        self.save_profile(c)
        html = c.get("/profile").get_data(as_text=True)

        self.assertIn("会清掉的", html)
        self.assertIn("不会动的", html)
        self.assertIn("聊天记录", html, "没说明聊天记录会保留")
        # 【区分两者】这一段必须把「全清」删什么、不可撤销说清楚，
        # 而且【不能】再出现「那个功能还没有做」那类过期说法（它早就做完了）。
        self.assertIn("和上面那个「清空全部个人数据」区分开", html)
        self.assertIn("不可撤销", html)
        self.assertNotIn("那个功能还没有做", html, "又出现了「全清还没做」这句过期的话")

    def test_the_clear_section_is_only_shown_when_there_is_something_to_clear(self):
        """没设置过的人不该看到「清空学习档案」那个表单 —— 那只会让人困惑。

        【为什么查表单的 action 而不是查那句话】页面别的注释里也会提到「清空学习档案」这几个字，
        查字符串会误判。表单的 action="/profile/clear" 是它真的在页面上出现的证据。
        """
        c = self.bound_client()
        self.assertNotIn('action="/profile/clear"', c.get("/profile").get_data(as_text=True))

        self.save_profile(c)
        self.assertIn('action="/profile/clear"', c.get("/profile").get_data(as_text=True))

    # ---------- 邀请码页 ----------

    def test_the_invite_page_tells_the_user_to_keep_the_code(self):
        """发码时就要说清楚：这码是唯一凭证，丢了找不回来。"""
        html = tutor.app.test_client().get("/invite").get_data(as_text=True)

        self.assertIn("请把这张码保存好", html)
        self.assertIn("丢了我们也没法帮你查回来", html)

    def test_the_invite_page_does_not_claim_the_code_is_entered_only_once(self):
        """旧文案说「只输一次、之后靠会话记住你」—— 换设备时那句是误导。"""
        html = tutor.app.test_client().get("/invite").get_data(as_text=True)

        self.assertNotIn("只在这里输入一次", html)
        self.assertIn("重新输入同一张码", html)

    def test_the_invite_page_explains_the_revoked_session_behaviour(self):
        """【核心】会话失效/轮换的后果必须写在这个页面上，不能只写在代码注释里。

        用户会遇到两种情况：已经有身份了却输新码；上一个身份失效后历史不见了。
        两种都要提前说清楚，不能让人对着一个空白的对话区猜。
        """
        html = tutor.app.test_client().get("/invite").get_data(as_text=True)

        self.assertIn("已经绑定过一个学习档案", html, "没说清「已有身份」会怎样")
        self.assertIn("不会再显示", html, "没说清失效之后旧历史会不见")
        self.assertIn("数据仍在服务器上", html, "没说明数据其实还在")

    # ---------- 首页入口 ----------

    def test_the_index_page_offers_the_right_entry(self):
        """没绑定 → 引导去输码；已绑定 → 引导去档案页。"""
        fresh = tutor.app.test_client().get("/").get_data(as_text=True)
        self.assertIn("输入邀请码", fresh)

        bound = self.bound_client().get("/").get_data(as_text=True)
        self.assertIn("我的学习档案", bound)


# ===================== 5o. 管理入口：发码 / 撤销 =====================
#
# 【这一组守什么】这是唯一一个【带令牌的写入口】。四条红线：
#   ① 没配令牌（或配成示例值）→ 入口当作不存在
#   ② 令牌只从请求头读、用常数时间比较；错了什么都不写
#   ③ 码明文只在响应里出现，服务端日志里一个字都没有
#   ④ 只能登记【全新的未绑定码】—— 结构上不可能给已有学习者补发

class TestAdminInviteApi(ChatTestCase):

    TOKEN = "test-admin-token-not-a-real-one"

    def setUp(self):
        super().setUp()
        tutor.ADMIN_MINT_TOKEN = self.TOKEN
        tutor.ADMIN_ENABLED = True
        self.addCleanup(self._restore_admin)

    def _restore_admin(self):
        """还原成「未配置」——否则后面的测试会莫名其妙地"管理入口开着"。"""
        tutor.ADMIN_MINT_TOKEN = ""
        tutor.ADMIN_ENABLED = False

    def post(self, path, data, token=TOKEN):
        headers = {} if token is None else {tutor.ADMIN_TOKEN_HEADER: token}
        return tutor.app.test_client().post(path, data=data, headers=headers)

    def mint(self, code, token=TOKEN):
        return self.post("/admin/invites", {"code": code}, token=token)

    def invite_row(self, code):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                "SELECT status, learner_id FROM invites WHERE code_digest = ?",
                (tutor.profile_store.digest_invite_code(code, tutor.INVITE_CODE_PEPPER),)
            ).fetchone()
        finally:
            conn.close()

    # ---------- 入口的开 / 关 ----------

    def test_the_entry_is_closed_when_no_token_is_configured(self):
        """【核心】没配令牌 → 入口当作不存在（404），其它路由不受影响。"""
        tutor.ADMIN_ENABLED = False

        r = self.post("/admin/invites", {"code": "whatever"}, token="随便什么")

        self.assertEqual(r.status_code, 404)
        self.assertEqual(self.table_count("invites"), 0)
        # 正常页面照常工作
        self.assertEqual(tutor.app.test_client().get("/").status_code, 200)

    def test_the_example_token_keeps_the_entry_closed(self):
        """【核心】配成 .env.example 里那个公开示例值 = 没配 → 入口关闭。

        否则等于「任何人都能拿模板里那句公开的话给自己发码」。
        """
        tutor.ADMIN_MINT_TOKEN = env_utils.EXAMPLE_ADMIN_TOKEN
        tutor.ADMIN_ENABLED = False                    # app 启动时就是这么算的

        r = self.post("/admin/invites", {"code": "whatever"},
                      token=env_utils.EXAMPLE_ADMIN_TOKEN)

        self.assertEqual(r.status_code, 404)
        self.assertEqual(self.table_count("invites"), 0)

    # ---------- 授权 ----------

    def test_a_missing_or_wrong_token_writes_nothing(self):
        """【核心】令牌缺失 / 错误 → 401，且不写库。"""
        for token in (None, "错的令牌", self.TOKEN + "x", "带中文的伪造令牌"):
            r = self.mint("new-code-" + str(token), token=token)
            with self.subTest(token=str(token)[:8]):
                self.assertEqual(r.status_code, 401)

        self.assertEqual(self.table_count("invites"), 0, "授权失败却写了库")

    def test_the_token_is_read_from_the_header_not_the_body(self):
        """令牌放错地方（放请求体）→ 不认。"""
        r = self.post("/admin/invites",
                      {"code": "c1", "token": self.TOKEN}, token=None)

        self.assertEqual(r.status_code, 401)
        self.assertEqual(self.table_count("invites"), 0)

    # ---------- 登记 ----------

    def test_a_valid_request_registers_exactly_one_invite(self):
        """【核心】正常路径：200 + created，库里只有摘要。"""
        r = self.mint("fresh-invite-abc")

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"result": "created"})
        self.assertEqual(self.invite_row("fresh-invite-abc"),
                         (tutor.profile_store.INVITE_ACTIVE, None))

        conn = sqlite3.connect(self.db_path)
        try:
            dump = " ".join(str(row) for row in
                            conn.execute("SELECT * FROM invites"))
        finally:
            conn.close()
        self.assertNotIn("fresh-invite-abc", dump, "邀请码明文被存进数据库了")

    def test_the_same_code_twice_is_idempotent_and_does_not_duplicate(self):
        """【核心】同码重试 → 200 + already_registered，且库里仍然只有一行。"""
        self.assertEqual(self.mint("same").get_json(), {"result": "created"})

        r = self.mint("same")

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"result": "already_registered"})
        self.assertEqual(self.table_count("invites"), 1, "重复登记多出了一行")

    def test_a_bound_code_is_reported_as_bound(self):
        """已经被用掉的码 → 明确报 already_bound（不是成功）。"""
        self.make_invite("used-code")
        c = tutor.app.test_client()
        self.enter_invite(c, "used-code")              # 有人用它进来了

        r = self.mint("used-code")

        self.assertEqual(r.get_json(), {"result": "already_bound"})
        self.assertEqual(self.table_count("invites"), 1)

    def test_a_revoked_code_is_reported_as_revoked(self):
        self.make_invite("dead-code")
        self.revoke("dead-code")

        r = self.mint("dead-code")

        self.assertEqual(r.get_json(), {"result": "revoked"})

    def test_an_integrity_failure_is_not_reported_as_already_registered(self):
        """【核心】数据库别的完整性问题 → 500 internal_error，绝不报「已登记」。"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("CREATE TRIGGER boom BEFORE INSERT ON invites "
                             "BEGIN SELECT RAISE(ABORT, '演示用的故障'); END")
        finally:
            conn.close()

        r = self.mint("never-inserted")

        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.get_json(), {"error": "internal_error"})
        self.assertIsNone(self.invite_row("never-inserted"))

    # ---------- 请求体限制 ----------

    def test_it_cannot_be_used_to_reissue_for_an_existing_learner(self):
        """【核心】夹带 learner_id → 400。结构上不可能给已有学习者补发码。"""
        self.make_invite("code-a")
        c = tutor.app.test_client()
        self.enter_invite(c, "code-a")
        learner = self.learner_id_of(c)

        r = self.post("/admin/invites", {"code": "sneaky", "learner_id": str(learner)})

        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.table_count("invites"), 1, "多出了一张码")

    def test_a_missing_or_overlong_code_is_rejected(self):
        for data in ({}, {"code": ""}, {"code": "   "},
                     {"code": "x" * (tutor.ADMIN_MAX_CODE_CHARS + 1)}):
            r = self.post("/admin/invites", data)
            with self.subTest(data=str(data)[:20]):
                self.assertEqual(r.status_code, 400)
        self.assertEqual(self.table_count("invites"), 0)

    def test_an_oversized_body_is_rejected(self):
        r = self.post("/admin/invites", {"code": "c1", "padding": "x" * 5000})

        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.table_count("invites"), 0)

    # ---------- 响应与日志 ----------

    def test_the_response_is_never_cached(self):
        r = self.mint("cache-me")
        self.assertEqual(r.headers.get("Cache-Control"), "no-store")

    def test_no_secret_reaches_the_logs(self):
        """【核心】令牌、邀请码明文、摘要，一个都不许进日志。"""
        code = "log-check-code-9f2a"
        digest = tutor.profile_store.digest_invite_code(code, tutor.INVITE_CODE_PEPPER)

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.mint(code)
            self.mint(code)                            # 幂等那次也要查
            self.mint("另一个码", token="错的令牌")       # 授权失败那次也要查
        blob = "\n".join(captured.output)

        self.assertNotIn(code, blob, "邀请码明文进了日志")
        self.assertNotIn(digest, blob, "邀请码摘要进了日志")
        self.assertNotIn(self.TOKEN, blob, "管理令牌进了日志")
        # 该记的要记（只有状态，没有内容）
        self.assertIn("admin_invite_registered", blob)
        self.assertIn("stage=created", blob)

    def test_errors_do_not_echo_internals(self):
        r = self.mint("x" * 500)
        body = r.get_data(as_text=True)
        for leak in ("Traceback", "sqlite", "Trigger", "app.py", tutor.INVITE_CODE_PEPPER):
            self.assertNotIn(leak, body)

    # ---------- 撤销 ----------

    def test_revoke_needs_a_valid_token_and_a_confirmation(self):
        """【核心】撤销要有令牌 + 显式确认，缺一个都不写库。"""
        self.make_invite("code-a")

        no_token = self.post("/admin/invites/revoke",
                             {"code": "code-a", "confirm": "yes"}, token=None)
        no_confirm = self.post("/admin/invites/revoke", {"code": "code-a"})

        self.assertEqual(no_token.status_code, 401)
        self.assertEqual(no_confirm.status_code, 400)
        self.assertEqual(self.invite_row("code-a"),
                         (tutor.profile_store.INVITE_ACTIVE, None), "被改动了")

    def test_revoking_cuts_off_the_bound_sessions(self):
        """【核心】按明文码撤销 → 复用现有逻辑：码作废 + 已绑定会话失效。"""
        self.make_invite("code-a")
        c = tutor.app.test_client()
        self.enter_invite(c, "code-a")
        self.assertEqual(c.get("/profile").status_code, 200)

        r = self.post("/admin/invites/revoke", {"code": "code-a", "confirm": "yes"})

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["result"], "revoked")
        self.assertEqual(r.get_json()["sessions_dropped"], 1)
        self.assertEqual(self.invite_row("code-a")[0], tutor.profile_store.INVITE_REVOKED)
        # 那台浏览器现在进不了档案页了
        self.assertEqual(c.get("/profile").status_code, 302)
        self.assertIsNone(self.learner_id_of(c))

    def test_revoking_an_unknown_code_changes_nothing(self):
        r = self.post("/admin/invites/revoke", {"code": "从没发过", "confirm": "yes"})

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["result"], "not_found")

    def test_revoke_never_puts_the_code_in_the_logs(self):
        code = "revoke-log-check-1b"
        self.make_invite(code)

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.post("/admin/invites/revoke", {"code": code, "confirm": "yes"})
        blob = "\n".join(captured.output)

        self.assertNotIn(code, blob)
        self.assertIn("admin_invite_revoked", blob)


# ===================== 5m. 会话边界：失效会话与新邀请码 =====================
#
# 【这一组修的是什么】两个真实复现出来的问题：
#   ① 已失效的会话输入一张【新】邀请码 → 兑换提交了、绑定却被拒 → 浏览器还是匿名，
#      但那张码已经被消耗、还多出一个谁都进不去的孤儿学习者。
#   ② 已失效的会话继续匿名提问 → 新消息仍挂在旧 session_id 下，
#      而旧 id 仍然关联着原来那个学习者 → 以后清空他的数据时会把无关消息一起删掉。
#
# 两条的根都在同一处：**失效的会话必须被轮换成新的 session_id**。

class TestSessionRotation(ChatTestCase):

    def bind(self, code="code-a"):
        self.make_invite(code)
        c = tutor.app.test_client()
        self.enter_invite(c, code)
        return c

    def invite_error(self, response):
        """把响应里的提示文字取出来（判断是哪种拒绝）。"""
        html = response.get_data(as_text=True)
        for message in (tutor.INVITE_BOUND_ELSEWHERE_MESSAGE, tutor.INVITE_REVOKED_MESSAGE,
                        tutor.INVITE_INVALID_MESSAGE):
            if message in html:
                return message
        return ""

    # ---------- 场景 1：有效会话 + 别人的码 ----------

    def test_a_valid_session_cannot_take_over_another_invite(self):
        """【核心】已经绑着 A 的浏览器，输一张全新的 B 码 →

        A 的绑定不变、**B 的码没有被消耗**、**不会多出孤儿学习者**，
        而且要给用户一句明确的话。
        """
        a = self.bind("code-a")
        learner_a = self.learner_id_of(a)
        self.make_invite("code-b")
        learners_before = self.table_count("learners")

        r = self.enter_invite(a, "code-b")

        self.assertEqual(r.status_code, 200, "被拒绝时不该 302（那看起来像成功了）")
        self.assertEqual(self.invite_error(r), tutor.INVITE_BOUND_ELSEWHERE_MESSAGE)

        # A 那边什么都没变
        self.assertEqual(self.learner_id_of(a), learner_a)

        # B 的码还是可用的（没有被消耗）
        conn = sqlite3.connect(self.db_path)
        try:
            status, linked = conn.execute(
                "SELECT status, learner_id FROM invites WHERE code_digest = ?",
                (tutor.profile_store.digest_invite_code("code-b", tutor.INVITE_CODE_PEPPER),)
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(status, tutor.profile_store.INVITE_ACTIVE, "B 的码被白白消耗了")
        self.assertIsNone(linked, "B 的码被绑到了一个没人能进的孤儿学习者上")

        # 没有多出学习者
        self.assertEqual(self.table_count("learners"), learners_before)

        # B 的码换一个干净的浏览器仍然能用
        fresh = tutor.app.test_client()
        self.assertEqual(self.enter_invite(fresh, "code-b").status_code, 302)
        self.assertIsNotNone(self.learner_id_of(fresh))

    # ---------- 被拒绝时给出的指引必须真的可行 ----------

    def test_the_bound_elsewhere_message_does_not_give_wrong_advice(self):
        """【核心】这条提示改过两轮，原来给的三条路全是错的，都不能再出现。

        ❌ 「清空当前档案」—— 那只清偏好，**不会解绑**，照着做一遍还是被拒。
        ❌ 「换一个浏览器窗口」—— 同一个浏览器的新窗口**共享 cookie**，等于没换。
        ❌ 「做一次清空全部个人数据来换号」—— 那是在诱导用户为了换号去毁掉全部数据。
        """
        message = tutor.INVITE_BOUND_ELSEWHERE_MESSAGE

        for wrong in ("清空当前档案", "换一个浏览器窗口", "新窗口",
                      "两条路", "或者做一次清空全部个人数据"):
            with self.subTest(wrong=wrong):
                self.assertNotIn(wrong, message, "又出现了那条做不到 / 不该给的建议：" + wrong)

        # 【注意】「清空学习档案」这几个字是【允许】出现的 ——
        # 但只能以「它不能解除绑定」的形式出现（下面那条测试钉住了这一点）。

    def test_the_bound_elsewhere_message_says_it_cannot_be_switched_losslessly(self):
        """【核心】先把实话说在前面：这个浏览器不能无损换绑。

        指引只有一条真正可行的路 —— **换一个不共享 cookie 的环境**。
        （「清空全部个人数据」也能走到新身份，但那要毁掉全部数据，
         是删除决定，不是换号手段，见下一条。）
        """
        message = tutor.INVITE_BOUND_ELSEWHERE_MESSAGE

        self.assertIn("没法无损换成另一个身份", message)
        self.assertIn("不共享 cookie", message)
        self.assertIn("无痕", message, "没有给出一个具体可操作的例子")
        self.assertIn("不能解除绑定", message, "没堵掉「清空学习档案能换绑」这个误解")

    def test_the_message_never_advertises_the_full_wipe_as_a_way_to_switch(self):
        """【核心】全清只能被描述成一次不可撤销的删除，而且必须明确劝阻。

        这是上一版的问题：把「做一次清空全部个人数据」摆成换身份的两条路之一，
        等于诱导用户为了换号去毁掉自己的聊天记录。
        """
        message = tutor.INVITE_BOUND_ELSEWHERE_MESSAGE

        # 提到它就必须写全代价：删什么 + 作废邀请码 + 不可撤销
        self.assertIn("清空全部个人数据", message)
        self.assertIn("聊天记录", message, "没说明它会删掉聊天记录")
        self.assertIn("作废已有邀请码", message, "没说明它会作废邀请码")
        self.assertIn("不可撤销", message)
        self.assertIn("是另一回事", message, "没把它和「换身份」区分开")
        self.assertIn("不要为了换一个身份去点它", message, "没有明确劝阻")

    def test_the_advertised_route_actually_works(self):
        """【核心】照着提示做（换一个不共享 cookie 的环境），真的能换到新身份。

        用两个 test_client 模拟「两个不共享 cookie 的浏览器环境」：
        第一个已经绑了 A（被拒），第二个是干净的 → 同一张 B 的码立刻就通了。
        """
        a = self.bind("code-a")
        self.make_invite("code-b")

        # 第一个环境：被拒
        r = self.enter_invite(a, "code-b")
        self.assertEqual(self.invite_error(r), tutor.INVITE_BOUND_ELSEWHERE_MESSAGE)

        # 第二个环境（独立 cookie）—— 提示里推荐的那条路
        other = tutor.app.test_client()
        r = self.enter_invite(other, "code-b")

        self.assertEqual(r.status_code, 302, "按提示换了环境却还是进不去")
        self.assertIsNotNone(self.learner_id_of(other))
        # 而原来那个环境不受影响，仍然属于 A
        self.assertIsNotNone(self.learner_id_of(a))
        self.assertNotEqual(self.learner_id_of(a), self.learner_id_of(other))

    def test_a_full_wipe_does_unbind_but_is_not_the_advertised_route(self):
        """【核心】锁定事实：全清确实会解绑（所以它"能用"）——
        但提示里**不推荐**它，这条测试只记录事实，不代表那是建议的做法。
        """
        a = self.bind("code-a")
        self.make_invite("code-b")

        self.delete(a)                                   # 不可撤销：删掉全部数据

        self.assertIsNone(self.learner_id_of(a), "清空全部数据之后居然还绑着")
        self.assertEqual(self.table_count("messages"), 0)
        # 解绑之后，同一张码在这个环境里确实能用了
        self.assertEqual(self.enter_invite(a, "code-b").status_code, 302)

    # ---------- 场景 2：失效会话 + 全新的码 ----------

    def test_a_revoked_session_gets_a_new_id_and_can_use_a_new_invite(self):
        """【核心】已失效的会话输 B 码 → 先轮换出新 session_id，再成功绑到 B。"""
        a = self.bind("code-a")
        old_sid = self.session_id_of(a)
        self.ask(a, "A 那时候问的话")
        self.revoke("code-a")                          # A 的会话失效
        self.assertIsNone(self.learner_id_of(a))

        self.make_invite("code-b")
        learners_before = self.table_count("learners")
        r = self.enter_invite(a, "code-b")

        self.assertEqual(r.status_code, 302, "应当成功进入")
        self.assertNotEqual(self.session_id_of(a), old_sid, "失效会话没有被轮换")
        self.assertIsNotNone(self.learner_id_of(a), "没有绑到 B")
        self.assertEqual(self.table_count("learners"), learners_before + 1)

        # 【A 的旧关联必须还留在库里】—— 以后清空 A 的数据要靠它
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT learner_id, revoked_at FROM learner_sessions WHERE session_id = ?",
                (old_sid,)).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row, "A 的旧关联行被删了")
        self.assertIsNotNone(row[1], "A 的旧关联不该被恢复成有效")

        # 新身份在 A 的旧消息里看不到任何东西
        self.assertNotIn("A 那时候问的话", a.get("/").get_data(as_text=True))

    def test_a_revoked_session_can_still_take_its_own_learners_new_code(self):
        """失效会话 + 【同一个学习者】的新码（补发场景）→ 恢复，不是轮换。

        这条是刻意的例外：轮换是为了「不再属于原来那个人」，
        而这里来的正是原来那个人的新码，所以应该恢复原来的档案。
        """
        a = self.bind("code-a")
        self.ask(a, "恢复之前问的话")
        learner = self.learner_id_of(a)
        self.revoke("code-a")

        new_code = self.reissue(learner)               # 补发（产品里没有入口，测试直接调原语）
        r = self.enter_invite(a, new_code)

        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.learner_id_of(a), learner, "没有回到原来的学习者")

    # ---------- 场景 3：失效会话继续匿名提问 ----------

    def test_questions_after_revocation_use_a_brand_new_session_id(self):
        """【核心】作废之后匿名提问 → 消息落在【新】session_id 下，不再算到 A 头上。"""
        a = self.bind("code-a")
        old_sid = self.session_id_of(a)
        self.ask(a, "A 那时候问的话")
        self.revoke("code-a")

        self.ask(a, "作废之后匿名问的话")               # 现在它是匿名访客

        new_sid = self.session_id_of(a)
        self.assertNotEqual(new_sid, old_sid)
        self.assertEqual(self.messages_for_session_id(new_sid), 2, "新消息没有落在新会话下")
        self.assertEqual(self.messages_for_session_id(old_sid), 2, "旧会话的消息被动过")

    def test_clearing_learner_a_keeps_the_new_anonymous_messages(self):
        """【核心】之后清空 A 的全部数据 → 只删 A 的旧消息，新的匿名消息留着。

        【注意顺序】作废会把这位学习者【所有】会话都置为失效（不只当前这台）。
        所以「还能执行清空」的那台设备，必须在作废【之后】才拿到新码进来 ——
        这也正是现实中的顺序：旧设备被踢下线 → 用户在另一台设备上用补发码回来
        → 而那台旧设备还在被当成匿名浏览器用着。
        """
        a = self.bind("code-a")
        old_sid = self.session_id_of(a)
        self.ask(a, "A 那时候问的话")
        learner = self.learner_id_of(a)

        self.revoke("code-a")                          # a 失效（A 名下所有会话一起失效）

        new_code = self.reissue(learner)                # 补发一张，绑回同一个学习者
        device2 = tutor.app.test_client()
        self.enter_invite(device2, new_code)            # 这台设备现在是有效的
        self.assertEqual(self.learner_id_of(device2), learner)

        self.ask(a, "作废之后匿名问的话")                # a 已轮换 → 匿名访客

        self.delete(device2)                            # 从 device2 清空 A 的全部数据

        # A 的旧消息（在旧 session_id 下）必须删掉
        self.assertEqual(self.messages_for_session_id(old_sid), 0, "A 的旧消息没删掉")
        # 但 a 在作废之后匿名问的那条必须留着 —— 它已经不是 A 的话了
        self.assertEqual(self.messages_for_session_id(self.session_id_of(a)), 2,
                         "把无关的匿名消息也删了")
        self.assertIn("作废之后匿名问的话", a.get("/").get_data(as_text=True))

    def delete(self, client, confirm="yes", csrf=None):
        """提交「清空全部个人数据」表单（和 TestDeleteAllData 里那个同款）。"""
        if csrf is None:
            csrf = self.csrf_token(client, "/profile")
        data = {"csrf_token": csrf}
        if confirm is not None:
            data["confirm"] = confirm
        return client.post("/profile/delete", data=data)

    # ---------- 轮换本身 ----------

    def test_a_healthy_session_is_never_rotated(self):
        """正常会话不该被换掉 —— 换了就等于把用户的历史弄丢了。"""
        a = self.bind("code-a")
        sid = self.session_id_of(a)

        self.ask(a, "随便问一句")
        self.assertEqual(self.session_id_of(a), sid)

    def test_an_anonymous_session_is_never_rotated(self):
        """匿名访客（从来没绑过）也不该被换。"""
        c = tutor.app.test_client()
        self.ask(c, "匿名问一句")
        sid = self.session_id_of(c)

        self.ask(c, "再问一句")
        self.assertEqual(self.session_id_of(c), sid)

    def test_the_old_binding_row_is_never_deleted(self):
        """轮换只是换 id，绝不去删旧行 —— 删了就找不到旧聊天记录了。"""
        a = self.bind("code-a")
        old_sid = self.session_id_of(a)
        self.ask(a, "一句话")
        self.revoke("code-a")

        self.ask(a, "触发轮换")

        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM learner_sessions WHERE session_id = ?", (old_sid,)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 1, "旧绑定行被删了")


# ===================== 5n. 会话状态确认不了时：必须停下 =====================
#
# 【这一组修的是什么】会话失效检查（session_is_revoked）读库失败时，
# 早期版本会选择「当成没失效、继续用这个 sid」。那是个真窟窿：
# 如果这个 sid 其实已经被作废，提问会照常成功，新消息就写进了别人（已作废）的旧会话下，
# 以后清空那位学习者的数据时，这些无关消息会被一起删掉。
#
# 正确做法是【宁可停下，也不猜】：不提问、不兑换、不写任何东西，只给一句重试提示。

class TestSessionCheckFailure(ChatTestCase):

    BOOM = "内部细节-会话检查炸了-4f7c"

    def bind(self, code="code-a"):
        self.make_invite(code)
        c = tutor.app.test_client()
        self.enter_invite(c, code)
        return c

    def setUp(self):
        super().setUp()
        # 记下真的那个函数，并保证每个测试结束都还原。
        # 【为什么必须还原】ChatTestCase 每次 setUp 都会 reload app，但 profile_store
        # 是同一个模块对象 —— 改过的属性会留到后面的测试里，变成难查的连锁失败。
        self._real_session_is_revoked = tutor.profile_store.session_is_revoked
        self.addCleanup(setattr, tutor.profile_store, "session_is_revoked",
                        self._real_session_is_revoked)

    def break_the_session_check(self):
        """让 session_is_revoked() 抛异常，模拟数据库读不了。"""
        def boom(conn, session_id):
            raise sqlite3.OperationalError(self.BOOM)

        tutor.profile_store.session_is_revoked = boom

    def restore_the_session_check(self):
        """手动恢复正常（用来测「恢复之后还能用」）。"""
        tutor.profile_store.session_is_revoked = self._real_session_is_revoked

    def learner_sessions_for(self, client):
        """这个浏览器的会话绑定行（原样）。"""
        return self.session_row(client)

    # ---------- 提问 ----------

    def test_a_question_is_not_processed_when_the_session_cannot_be_verified(self):
        """【核心】确认不了会话状态 → 不调模型、不写库，只给一句重试提示。"""
        c = self.bind()
        self.ask(c, "之前正常问的一句")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.table_count("messages"), 2)

        self.break_the_session_check()
        r = self.ask(c, "这次不该被处理")

        self.assertNotEqual(r.status_code, 302, "居然当成成功处理了")
        self.assertEqual(r.status_code, 503)
        self.assertIn(tutor.SESSION_RETRY_MESSAGE, r.get_data(as_text=True))

        self.assertEqual(len(self.fake.calls), 1, "会话没确认还去调模型了")
        self.assertEqual(self.table_count("messages"), 2, "会话没确认还写了消息")

    def test_nothing_is_written_under_the_old_session_id(self):
        """【核心】不能把消息写进那个【未经确认】的旧 sid 下 —— 这正是原来的窟窿。"""
        c = self.bind()
        sid = self.session_id_of(c)
        self.ask(c, "之前正常问的一句")

        self.break_the_session_check()
        self.ask(c, "这次不该被处理")

        self.assertEqual(self.messages_for_session_id(sid), 2, "旧会话下多出了消息")

    # ---------- 邀请码 ----------

    def test_an_invite_is_not_redeemed_when_the_session_cannot_be_verified(self):
        """【核心】确认不了会话状态 → 邀请码不兑换：不消耗、不建学习者。"""
        c = self.bind("code-a")
        self.make_invite("code-b")
        learners_before = self.table_count("learners")

        # 【token 要在弄坏之前拿】GET /invite 自己也会走会话检查，
        # 坏了之后再取就只会拿到那张 503 提示页，取不到 token。
        csrf = self.csrf_token(c, "/invite")

        self.break_the_session_check()
        r = c.post("/invite", data={"invite_code": "code-b", "csrf_token": csrf})

        self.assertEqual(r.status_code, 503)
        self.assertIn(tutor.SESSION_RETRY_MESSAGE, r.get_data(as_text=True))

        conn = sqlite3.connect(self.db_path)
        try:
            status, linked = conn.execute(
                "SELECT status, learner_id FROM invites WHERE code_digest = ?",
                (tutor.profile_store.digest_invite_code("code-b", tutor.INVITE_CODE_PEPPER),)
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(status, tutor.profile_store.INVITE_ACTIVE, "码被消耗了")
        self.assertIsNone(linked)
        self.assertEqual(self.table_count("learners"), learners_before, "建出了学习者")
        self.assertEqual(self.table_count("learner_sessions"), 1, "会话绑定被动了")

    # ---------- 其它依赖会话身份的操作 ----------

    def test_the_profile_pages_stop_too(self):
        """档案页、清空、删除都依赖会话身份 —— 一样要停，而且什么都不改。"""
        c = self.bind()
        self.save_profile(c, level_code="b2")

        self.break_the_session_check()

        for path, data in (("/profile", {}),
                           ("/profile/clear", {"confirm": "yes"}),
                           ("/profile/delete", {"confirm": "yes"})):
            r = c.post(path, data=data) if data else c.get(path)
            with self.subTest(path=path):
                self.assertEqual(r.status_code, 503)
                self.assertIn(tutor.SESSION_RETRY_MESSAGE, r.get_data(as_text=True))

        # 偏好还在（清空没执行）
        self.assertIsNotNone(self.profile_row(self.learner_id_of(c)))

    # ---------- 不泄露 ----------

    def test_the_retry_page_leaks_nothing(self):
        """提示页里不能出现异常原文、堆栈、数据库细节或密钥。"""
        c = self.bind()
        self.break_the_session_check()

        body = self.ask(c, "随便问一句").get_data(as_text=True)

        for leak in (self.BOOM, "Traceback", "OperationalError", "sqlite",
                     tutor.API_KEY, tutor.INVITE_CODE_PEPPER):
            self.assertNotIn(leak, body, "提示页里泄露了：" + leak[:12])

    def test_the_failure_is_logged_without_content(self):
        """日志要记「发生了」，但只记异常类型，不记原文。"""
        c = self.bind()
        self.break_the_session_check()

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, "随便问一句")
        blob = "\n".join(captured.output)

        self.assertIn("session_check_failed", blob)
        self.assertIn("error_type=OperationalError", blob)      # 类型可以记
        self.assertIn("session_unverifiable", blob, "统一收尾那一处没有记日志")
        self.assertNotIn(self.BOOM, blob, "异常原文进了日志")

    # ---------- 恢复正常 ----------

    def test_it_works_again_once_the_check_recovers(self):
        """恢复之后一切照旧：能提问、能轮换。"""
        c = self.bind()
        self.break_the_session_check()
        self.assertEqual(self.ask(c, "这次失败").status_code, 503)

        self.restore_the_session_check()             # 手动恢复

        # 提问恢复正常
        r = self.ask(c, "恢复之后问的一句")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.table_count("messages"), 2)
        self.assertEqual(len(self.fake.calls), 1)

    def test_rotation_still_works_after_recovery(self):
        """【核心】恢复之后，作废 → 轮换这条链路仍然正常。"""
        c = self.bind("code-a")
        old_sid = self.session_id_of(c)
        self.ask(c, "作废之前问的话")

        self.break_the_session_check()
        self.assertEqual(self.ask(c, "失败的一次").status_code, 503)
        self.restore_the_session_check()

        self.revoke("code-a")                       # 现在真的作废
        self.ask(c, "触发轮换")

        self.assertNotEqual(self.session_id_of(c), old_sid, "恢复之后轮换不工作了")
        self.assertIsNone(self.learner_id_of(c))
        self.assertEqual(self.messages_for_session_id(old_sid), 2, "旧会话的消息被动过")

    # ---------- 健康检查不受影响 ----------

    def test_health_is_unaffected(self):
        """【核心】/health 不依赖会话身份，所以它不该被这个故障带下水。"""
        self.break_the_session_check()
        c = tutor.app.test_client()
        r = c.get("/health")

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"status": "ok"})


# ===================== 5l. 清空全部个人数据 =====================
#
# 【它和「清空学习档案」的区别】
#   清空学习档案 —— 只清偏好，人还在，聊天记录还在
#   清空全部数据 —— 把人删掉：聊天记录 + 偏好 + 会话绑定 + 学习者身份，并作废邀请码
#
# 【这一组要证明什么】
#   ① 该删的删干净（多设备、以及被作废过的旧会话）
#   ② 不该碰的不动（别的学习者、匿名访客、全站额度）
#   ③ 门要关严（CSRF + 真的勾了确认 + 匿名不能删）
#   ④ 删完之后旧身份和旧数据都回不来（包括并发提问也写不回来）

class TestDeleteAllData(ChatTestCase):

    def bind(self, code="c1"):
        self.make_invite(code)
        c = tutor.app.test_client()
        self.enter_invite(c, code)
        return c

    def bind_with_profile(self, code="c1"):
        c = self.bind(code)
        self.save_profile(c, level_code="c1", language_mode="en_only",
                          length_mode="brief", goal_code="work", focus_code="speaking")
        return c

    def delete(self, client, confirm="yes", csrf=None):
        """提交「清空全部个人数据」表单。"""
        if csrf is None:
            csrf = self.csrf_token(client, "/profile")
        data = {"csrf_token": csrf}
        if confirm is not None:
            data["confirm"] = confirm
        return client.post("/profile/delete", data=data)

    # ---------- 入口本身 ----------

    def test_the_page_has_a_prominent_wipe_entry(self):
        """【核心】档案页要有这个入口，而且要把删除范围一条条写出来。"""
        html = self.bind_with_profile().get("/profile").get_data(as_text=True)

        self.assertIn('action="/profile/delete"', html)
        for phrase in ("聊天记录", "学习偏好", "学习者身份", "会话绑定", "邀请码"):
            self.assertIn(phrase, html, "没有说明会删除：" + phrase)

    def test_the_page_says_what_it_cannot_delete(self):
        """【核心】不能承诺「物理擦除」——要写清第三方那边和备份不管用。"""
        html = self.bind_with_profile().get("/profile").get_data(as_text=True)

        self.assertIn("第三方模型服务商", html)
        self.assertIn("备份", html)

    def test_the_wipe_confirm_is_the_checkbox_not_a_hidden_field(self):
        """【核心】和清空档案同一条规矩：confirm 必须由复选框自己提交。"""
        html = self.bind_with_profile().get("/profile").get_data(as_text=True)

        self.assertNotIn('type="hidden" name="confirm"', html)
        self.assertIn('name="confirm" value="yes"', html)

    def test_the_two_destructive_actions_are_visually_distinct(self):
        """两个危险操作要用不同的容器，别让人分不清删的是哪一档。"""
        html = self.bind_with_profile().get("/profile").get_data(as_text=True)

        self.assertIn('class="wipe"', html)              # 清空全部个人数据
        self.assertIn('class="danger"', html)            # 只清空学习档案

    # ---------- 三道门 ----------

    def test_an_anonymous_visitor_cannot_delete_anything(self):
        """【核心】没绑定的人连删除入口都到不了，数据也不受影响。"""
        c = tutor.app.test_client()
        c.get("/profile")

        r = c.post("/profile/delete", data={"confirm": "yes"})

        self.assertEqual(r.status_code, 302)
        self.assertIn("/invite", r.headers["Location"])

    def test_deleting_requires_a_csrf_token(self):
        """【核心】没有 token（或 token 不对）→ 拒绝，什么都不能删。"""
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)

        for data in ({"confirm": "yes"},                                   # 完全没有 token
                     {"confirm": "yes", "csrf_token": "伪造的"}):            # 伪造的 token
            r = c.post("/profile/delete", data=data)
            with self.subTest(data=data):
                self.assertIn(tutor.CSRF_MESSAGE, r.get_data(as_text=True))

        self.assertIsNotNone(self.profile_row(learner), "CSRF 没拦住，数据被删了")

    def test_deleting_requires_the_user_to_really_check_the_box(self):
        """【核心】没勾确认 → 拒绝（模拟真实浏览器：那个字段压根不会提交）。"""
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)

        r = self.delete(c, confirm=None)

        self.assertIn(tutor.DELETE_CONFIRM_MESSAGE, r.get_data(as_text=True))
        self.assertIsNotNone(self.profile_row(learner), "没勾确认也把数据删了")

    # ---------- 删得干净 ----------

    def test_deleting_removes_everything_and_logs_the_browser_out(self):
        """【核心】删完之后：数据没了、这个浏览器退回未认领状态。"""
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)
        self.ask(c, "删除之前问过的话")
        self.assertEqual(self.table_count("messages"), 2)

        r = self.delete(c)

        self.assertEqual(r.status_code, 302)                       # PRG
        self.assertEqual(self.table_count("messages"), 0, "聊天记录没删干净")
        self.assertEqual(self.table_count("learner_preferences"), 0)
        self.assertEqual(self.table_count("learner_sessions"), 0)
        self.assertEqual(self.table_count("learners"), 0)
        self.assertEqual(
            self.db_fetchone("SELECT status FROM invites"), tutor.profile_store.INVITE_REVOKED)

        # 旧身份没了：档案页把人引导去输邀请码
        self.assertIsNone(self.learner_id_of(c))
        self.assertEqual(c.get("/profile").status_code, 302)
        self.assertIn("/invite", c.get("/profile").headers["Location"])

    def test_the_old_page_no_longer_shows_the_deleted_conversation(self):
        c = self.bind_with_profile()
        self.ask(c, "删除之前问过的话")

        self.delete(c)

        html = c.get("/").get_data(as_text=True)
        self.assertNotIn("删除之前问过的话", html, "页面上还能看到已删除的对话")

    def test_the_model_context_no_longer_contains_the_deleted_conversation(self):
        """【核心】删完之后再提问，旧内容不能作为上下文出现在提示词里。"""
        c = self.bind_with_profile()
        self.ask(c, "删除之前问过的话")
        self.delete(c)

        self.ask(c, "删除之后问的话")

        sent = self.fake.calls[-1]["messages"][1]["content"]
        self.assertNotIn("删除之前问过的话", sent, "已删除的对话又进了模型上下文")
        self.assertNotIn("学习者档案", sent, "档案已经被删了，不该再带上")

    def test_every_device_of_that_learner_loses_access(self):
        """【核心】同一个学习者在多台设备上 → 全部失效，全部聊天记录都删。"""
        a = self.bind_with_profile("code-a")
        b = tutor.app.test_client()
        self.enter_invite(b, "code-a")                 # 同一个码，另一台设备
        self.assertEqual(self.learner_id_of(b), self.learner_id_of(a))
        self.ask(a, "A 设备问的话")
        self.ask(b, "B 设备问的话")
        self.assertEqual(self.table_count("messages"), 4)
        self.assertEqual(self.table_count("learner_sessions"), 2)

        self.delete(a)

        self.assertEqual(self.table_count("messages"), 0, "有一台设备的记录漏掉了")
        self.assertEqual(self.table_count("learner_sessions"), 0)
        self.assertIsNone(self.learner_id_of(a))
        self.assertIsNone(self.learner_id_of(b), "另一台设备还留着旧身份")
        self.assertEqual(b.get("/profile").status_code, 302)

    def test_conversations_from_revoked_devices_are_deleted_too(self):
        """【核心】被作废过的旧设备的聊天记录，也不能漏。

        先作废（旧设备失去访问权），再在别的设备上清空全部数据 ——
        旧设备那些消息必须一起删掉。这依赖「作废只打标记、不删关联」那条修正。
        """
        a = self.bind_with_profile("code-a")
        self.ask(a, "作废之前问过的话")
        self.revoke("code-a")                          # a 失去访问权，但关联还在
        self.assertIsNone(self.learner_id_of(a))

        b = tutor.app.test_client()                    # 补发一张码，绑到同一个学习者
        learner = self.db_fetchone("SELECT learner_id FROM invites WHERE learner_id IS NOT NULL")
        new_code = self.reissue(learner)
        self.enter_invite(b, new_code)
        self.assertEqual(self.learner_id_of(b), learner)
        self.ask(b, "新设备问的话")
        self.assertEqual(self.table_count("messages"), 4)

        self.delete(b)                                 # 从新设备清空全部数据

        self.assertEqual(self.table_count("messages"), 0, "被作废设备的聊天记录漏掉了")

    def test_the_old_invite_code_can_never_be_used_again(self):
        """【核心】删除之后旧码不能再用 —— 没有重新进入的通道。"""
        c = self.bind_with_profile("code-a")
        self.delete(c)

        fresh = tutor.app.test_client()
        r = self.enter_invite(fresh, "code-a")

        self.assertIn(tutor.INVITE_REVOKED_MESSAGE, r.get_data(as_text=True))
        self.assertIsNone(self.learner_id_of(fresh))

    # ---------- 不该碰的不动 ----------

    def test_another_learner_is_untouched(self):
        """【核心隔离断言】删 A 的全部数据，B 的档案、聊天、身份全都在。"""
        a = self.bind_with_profile("code-a")
        b = self.bind_with_profile("code-b")
        learner_b = self.learner_id_of(b)
        self.ask(a, "A 问的话")
        self.ask(b, "B 问的话")

        self.delete(a)

        self.assertIsNotNone(self.profile_row(learner_b), "B 的档案被删了")
        self.assertEqual(self.learner_id_of(b), learner_b, "B 的身份被动了")
        self.assertEqual(b.get("/profile").status_code, 200)
        self.assertIn("B 问的话", b.get("/").get_data(as_text=True))
        self.assertEqual(self.table_count("messages"), 2, "把 B 的聊天记录也删了")

    def test_an_anonymous_visitors_history_is_untouched(self):
        """匿名访客的聊天记录没有主人，删某个学习者时不能顺手带走。"""
        c = self.bind_with_profile()
        visitor = tutor.app.test_client()
        self.ask(visitor, "匿名访客问的话")

        self.delete(c)

        self.assertIn("匿名访客问的话", visitor.get("/").get_data(as_text=True))
        self.assertEqual(self.table_count("messages"), 2)

    def test_the_global_quota_is_not_cleared(self):
        """【核心】全站每日额度是全站共用的，不属于任何个人，绝不能清。"""
        c = self.bind_with_profile()
        self.ask(c, "消耗一次额度")
        self.assertEqual(self.used_today(), 1)

        self.delete(c)

        self.assertEqual(self.used_today(), 1, "全站额度被清空了")

    def used_today(self):
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT used FROM api_usage WHERE day = ?",
                               (tutor._utc_day(),)).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def db_fetchone(self, sql):
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(sql).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    # ---------- 和正在进行的提问抢跑 ----------

    def test_deleting_while_a_question_is_being_processed_is_refused_not_raced(self):
        """【核心】有提问正在处理时，删除会被挡下（而不是和它抢着写）。"""
        c = self.bind_with_profile()
        learner = self.learner_id_of(c)
        tutor.DELETE_LOCK_TIMEOUT = 0.2                # 别真等 10 秒

        tutor._lock.acquire()                          # 假装「有个提问正在处理」
        try:
            r = self.delete(c)
        finally:
            tutor._lock.release()

        self.assertIn(tutor.DELETE_BUSY_MESSAGE, r.get_data(as_text=True))
        self.assertIsNotNone(self.profile_row(learner), "被拒绝的删除居然动了数据")

    def test_an_in_flight_question_cannot_resurrect_deleted_data(self):
        """【核心】删除必须等正在进行的提问彻底结束，旧请求不能把消息写回来。

        【要防的是什么】一个提问的处理是「检索 → 调模型 → 写库」。
        如果删除能插在「正在调模型」和「写库」之间，那么删除提交之后，
        那个旧请求会把消息【又写回数据库】—— 用户以为删干净了，其实没有。

        这里用一个会卡住的假客户端把时序钉死：
          ① 线程 1 提问 → 卡在「调模型」那一步（锁在它手里）
          ② 主线程发起删除 → 应该被锁挡住，还没提交
          ③ 放开那个卡住的调用 → 线程 1 走完并写库、释放锁
          ④ 删除这才拿到锁、把数据（连同刚写进来的那条）一起删掉
        """
        c = self.bind_with_profile()

        entered = threading.Event()          # 假客户端「我开始调模型了」
        release = threading.Event()          # 主线程说「你可以返回了」

        def slow_create(model, messages, **kwargs):
            entered.set()
            release.wait(timeout=5)
            return FakeResponse(rag_reply())

        self.fake.chat.completions.create = slow_create

        # 【CSRF token 要提前取】GET /profile 不拿那把锁，但在别的线程里发请求更难排查
        csrf = self.csrf_token(c, "/profile")

        chat_result, delete_result = {}, {}

        def run_chat():
            chat_result["r"] = c.post("/", data={"question": "删除前问的话"})

        def run_delete():
            delete_result["r"] = c.post("/profile/delete",
                                        data={"csrf_token": csrf, "confirm": "yes"})

        chat = threading.Thread(target=run_chat)
        chat.start()
        self.assertTrue(entered.wait(timeout=5), "假客户端没被调用，测试前提不成立")

        remover = threading.Thread(target=run_delete)
        remover.start()

        # 此刻：提问卡在模型调用里，删除应该还在等锁 ——
        # 它不可能已经提交（它要先拿到锁，而锁在提问手里）。
        self.assertTrue(remover.is_alive(), "删除没有等锁，直接跑完了")
        self.assertIsNotNone(self.profile_row(self.learner_id_of(c)),
                             "删除在提问结束之前就提交了")

        # 放开：提问走完 → 写库 → 释放锁 → 删除才拿到锁
        release.set()
        chat.join(timeout=5)
        remover.join(timeout=5)

        self.assertFalse(chat.is_alive())
        self.assertFalse(remover.is_alive())
        self.assertEqual(chat_result["r"].status_code, 302, "提问本身应该正常完成")
        self.assertEqual(delete_result["r"].status_code, 302)

        # 【核心断言】提问确实写了库，但那些记录随后被删除带走了 ——
        # 说明删除是在它之后提交的，没有「删完又被写回来」。
        self.assertEqual(self.table_count("messages"), 0, "已删除的数据被旧请求写回来了")
        self.assertEqual(self.table_count("learners"), 0)
        self.assertEqual(self.table_count("learner_sessions"), 0)


# ===================== 5p. general_answer 的英文标识 =====================
#
# 【要解决什么】选了「全英文」的人，正文是英文、标签却是中文「AI 通用知识回答」——
# 他读不懂，等于把「这条回答没有课程资料依据」这件事藏起来了，透明标识就失效了。
#
# 【边界】换的只是【标签本身】：默认/中文模式一个字不变；
# answer / refuse / insufficient_evidence 完全不经过这段逻辑；
# rag.py 里的固定兜底话术一个字不动。

class TestGeneralAnswerMarker(ChatTestCase):

    EN_MARKER = "AI general-knowledge answer — not based on course materials"
    ZH_MARKER = "AI 通用知识回答"

    def bind(self, code="c1"):
        self.make_invite(code)
        c = tutor.app.test_client()
        self.enter_invite(c, code)
        return c

    def general_answer_reply(self):
        return rag_reply(decision="general_answer", answer="直译是……", citations=[])

    # ---------- 纯函数层 ----------

    def test_the_marker_defaults_to_chinese(self):
        """【核心防回归】默认（没有档案 / 没填语言偏好）必须还是那句中文。"""
        for preferences in (None, {}, {"language_mode": "zh_pair"},
                            {"level_code": "b1"}):
            with self.subTest(preferences=preferences):
                self.assertEqual(tutor.general_answer_marker(preferences), self.ZH_MARKER)

    def test_english_modes_switch_the_marker(self):
        for mode in ("en_only", "en_advanced"):
            with self.subTest(mode=mode):
                self.assertEqual(tutor.general_answer_marker({"language_mode": mode}),
                                 self.EN_MARKER)

    def test_formatting_still_puts_the_marker_before_the_body(self):
        result = {"decision": "general_answer", "answer": "正文在这里。", "citations": []}

        out = tutor.format_answer_with_sources(result, self.EN_MARKER)

        self.assertTrue(out.startswith("【" + self.EN_MARKER + "】"))
        self.assertLess(out.index(self.EN_MARKER), out.index("正文在这里。"))

    def test_other_decisions_do_not_get_a_marker(self):
        """answer / refuse / insufficient 都不该出现任何标识（换不换语言都一样）。"""
        for decision, answer in (("answer", "有依据的回答。"),
                                 ("refuse", "拒答。"),
                                 ("insufficient_evidence", "证据不足。")):
            result = {"decision": decision, "answer": answer,
                      "citations": [{"source": "a.md", "heading": "甲"}]
                      if decision == "answer" else []}
            with self.subTest(decision=decision):
                out = tutor.format_answer_with_sources(result, self.EN_MARKER)
                self.assertNotIn(self.EN_MARKER, out)
                self.assertNotIn(self.ZH_MARKER, out)

    # ---------- 网页层 ----------

    def test_an_english_profile_gets_the_english_marker(self):
        """【核心】选了全英文 → 存进库里、显示在页面上的是英文标识，没有中文标识。"""
        c = self.bind()
        self.save_profile(c, language_mode="en_only")
        self.fake.reply = self.general_answer_reply()

        self.ask(c, "帮我改一下这句话")

        saved = self.all_rows()[1][2]
        self.assertIn(self.EN_MARKER, saved)
        self.assertNotIn(self.ZH_MARKER, saved, "英文模式下还留着中文标识")

        html = c.get("/").get_data(as_text=True)
        self.assertIn("general-knowledge answer", html)

    def test_an_advanced_english_profile_also_gets_it(self):
        c = self.bind()
        self.save_profile(c, language_mode="en_advanced")
        self.fake.reply = self.general_answer_reply()

        self.ask(c, "帮我改一下这句话")

        self.assertIn(self.EN_MARKER, self.all_rows()[1][2])

    def test_a_chinese_profile_keeps_the_chinese_marker(self):
        """【核心防回归】选「中英对照」的人看到的还是原来那句中文。"""
        c = self.bind()
        self.save_profile(c, language_mode="zh_pair")
        self.fake.reply = self.general_answer_reply()

        self.ask(c, "帮我改一下这句话")

        saved = self.all_rows()[1][2]
        self.assertIn(self.ZH_MARKER, saved)
        self.assertNotIn(self.EN_MARKER, saved)

    def test_an_anonymous_visitor_keeps_the_chinese_marker(self):
        """【核心防回归】匿名访客（没有档案）行为零变化。"""
        c = tutor.app.test_client()
        self.fake.reply = self.general_answer_reply()

        self.ask(c, "随便问一句")

        self.assertIn(self.ZH_MARKER, self.all_rows()[1][2])

    def test_the_english_marker_is_never_dropped_or_turned_into_a_citation(self):
        """【安全边界】换语言不等于可以不标：标识必须在，且不能变成"引用"。"""
        c = self.bind()
        self.save_profile(c, language_mode="en_only")
        self.fake.reply = self.general_answer_reply()

        self.ask(c, "帮我改一下这句话")

        saved = self.all_rows()[1][2]
        self.assertIn(self.EN_MARKER, saved, "标识被丢掉了")
        self.assertNotIn("资料来源", saved, "general_answer 绝不能有来源区")

    def test_rag_module_untouched_by_this_change(self):
        """【结构性保证】这次只改展示标签：rag.py 的固定兜底话术一个字不动。"""
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "rag.py"), encoding="utf-8") as f:
            source = f.read()

        self.assertIn("这个问题我帮不上忙。", source)        # REFUSE_TEXT 开头
        self.assertIn("资料里提到了相关的话题", source)      # INSUFFICIENT_TEXT 开头
        self.assertNotIn("general-knowledge answer", source,
                         "英文标签不该出现在 rag.py 里（那是展示层的事）")


# ===================== 6. 纯函数：格式化成显示文本 =====================

class TestFormatAnswerWithSources(unittest.TestCase):
    """单独测那个纯函数——不用起 Flask、不用连数据库、不用碰模型。"""

    def test_answer_with_citations_gets_a_sources_section(self):
        out = tutor.format_answer_with_sources({
            "decision": "answer",
            "answer": "主语 + have / has + 动词的过去分词。",
            "citations": [{"source": "grammar_present_perfect.md", "heading": "基本结构"}],
        })
        self.assertIn("主语 + have / has + 动词的过去分词。", out)
        self.assertIn("资料来源：", out)
        self.assertIn("- grammar_present_perfect.md · 基本结构", out)

    def test_multiple_citations_each_get_a_line(self):
        out = tutor.format_answer_with_sources({
            "decision": "answer", "answer": "回答。",
            "citations": [{"source": "a.md", "heading": "甲"},
                          {"source": "b.md", "heading": "乙"}],
        })
        self.assertIn("- a.md · 甲", out)
        self.assertIn("- b.md · 乙", out)

    def test_refuse_has_no_sources_section(self):
        out = tutor.format_answer_with_sources({
            "decision": "refuse", "answer": "资料里没有。", "citations": []})
        self.assertEqual(out, "资料里没有。")
        self.assertNotIn("资料来源", out)

    def test_insufficient_evidence_has_no_sources_section(self):
        out = tutor.format_answer_with_sources({
            "decision": "insufficient_evidence", "answer": "信息不够。", "citations": []})
        self.assertEqual(out, "信息不够。")
        self.assertNotIn("资料来源", out)

    def test_answer_without_citations_has_no_sources_section(self):
        """兜底：冒充 answer 却没有任何引用，也不能凭空造一个来源区。"""
        out = tutor.format_answer_with_sources({
            "decision": "answer", "answer": "没有出处的回答。", "citations": []})
        self.assertEqual(out, "没有出处的回答。")

    def test_malformed_citations_are_skipped_without_an_empty_header(self):
        """引用项全都缺文件名时，宁可不加这一节，也不要留个空标题。"""
        out = tutor.format_answer_with_sources({
            "decision": "answer", "answer": "回答。",
            "citations": [{"source": "", "heading": "标题"}, {"heading": "只有标题"}]})
        self.assertEqual(out, "回答。")
        self.assertNotIn("资料来源", out)

    def test_citation_without_heading_still_shows_the_file(self):
        """只有文件名、没有标题时，文件名照常显示。"""
        out = tutor.format_answer_with_sources({
            "decision": "answer", "answer": "回答。",
            "citations": [{"source": "a.md", "heading": ""}]})
        self.assertIn("- a.md", out)

    def test_empty_result_does_not_crash(self):
        """缺字段、空字典都不能让它崩。"""
        for bad in ({}, {"decision": "answer"}, {"answer": None, "citations": None}):
            out = tutor.format_answer_with_sources(bad)
            self.assertIsInstance(out, str)


# ===================== 7. 测试自身的安全保证 =====================

class TestHarnessSafety(ChatTestCase):

    def test_uses_fake_key_not_the_real_one(self):
        """确认测试用的是假密钥。万一哪天环境变量被子进程污染，这条会立刻失败。"""
        self.assertEqual(tutor.API_KEY, "test-key-not-a-real-key")

    def test_client_is_stubbed(self):
        """确认真的客户端已经被换成假的——这是「绝不调用真实 API」的结构性保证。"""
        self.assertIsInstance(tutor.client, FakeClient)

    def test_retriever_is_stubbed(self):
        """【核心】检索函数也必须被换掉。

        不换的话，每个测试都会真的去读 knowledge_base 跑 BM25，
        测试就不再只测网页这一层了。
        """
        self.assertIs(tutor.retriever.retrieve, fake_retrieve)

    def test_fake_client_actually_receives_calls(self):
        """反面验证：正常提问时假客户端确实被调用了，说明上面那些断言不是空转。"""
        c = tutor.app.test_client()
        self.ask(c, "验证假客户端有被调用")
        self.assertEqual(len(self.fake.calls), 1)

        # 发给模型的内容由 rag.py 自己拼：第一条是它的提示词，
        # 第二条带着「用户问题 + 本次检索到的资料」。
        sent = self.fake.calls[0]["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("验证假客户端有被调用", sent[1]["content"])
        self.assertIn("基本结构", sent[1]["content"])       # 检索到的资料也一起带上了

    def test_previous_turn_is_now_sent_as_context(self):
        """【本轮的行为变化】上一轮的话现在会作为【上下文】发给模型。

        以前是「只按当前问题检索、历史一律不发」，于是「再给一个例子」
        「为什么这样改」这类依赖上一轮的问题答不了。
        现在会把同一会话最近的几条带上 —— 但只当上下文，不当资料（详见
        TestShortTermContext 里的安全测试）。
        """
        c = tutor.app.test_client()
        self.ask(c, "第一轮问的话")
        self.ask(c, "第二轮问的话")

        sent = self.fake.calls[-1]                          # 最后一次调用
        contents = " ".join(m["content"] for m in sent["messages"])

        self.assertIn("第二轮问的话", contents)
        self.assertIn("第一轮问的话", contents, "上一轮没有被当成上下文带进去")
        self.assertIn("最近上下文", contents, "带进去了，但没标明这是上下文而不是资料")

    def test_history_is_stored_displayed_and_also_sent_as_context(self):
        """历史现在有三件事同时成立：存下来、显示出来、还被当成上下文带上。

        注意展示和上下文是【两条独立的路】：展示读全部记录，
        上下文只读最近几条。这里三件事一起钉住。
        """
        c = tutor.app.test_client()
        self.ask(c, "第一轮问的话")
        self.ask(c, "第二轮问的话")

        self.assertEqual(len(self.all_rows()), 4)           # 两问两答，一条不少
        html = c.get("/").get_data(as_text=True)
        self.assertIn("第一轮问的话", html)                   # 第一轮在页面上仍然看得到
        self.assertIn("第二轮问的话", html)

        sent = self.fake.calls[-1]
        contents = " ".join(m["content"] for m in sent["messages"])
        self.assertIn("第一轮问的话", contents)               # 同时也在上下文里


# ===================== 8. 生产化：配置读取 =====================
#
# 【为什么这些要单独测】配置读错是「最难查的一类故障」——
# 程序照常启动、页面照常能用，只是限流其实没开、上限其实不是你以为的那个数。
# 所以合法值、默认值、非法值三种情况都要有确定行为。

class TestConfigParsing(unittest.TestCase):

    def setUp(self):
        self._saved = {}                                # 备份要动的环境变量，测完还原

        # 【为什么这个类也要有自己的临时库】
        # 它会 importlib.reload(app)，而 app 在导入时会检查 CHAT_DB_PATH 的目录存不存在。
        # 如果沿用上一个测试类留下的路径，而那个临时目录已经在它的 tearDown 里被删掉，
        # reload 就会直接抛「数据库所在目录不存在」——
        # 这属于测试之间的隐式依赖（一个类的成败取决于谁排在它前面），必须切断。
        self.tmpdir = tempfile.TemporaryDirectory()
        self._saved["CHAT_DB_PATH"] = os.environ.get("CHAT_DB_PATH")
        os.environ["CHAT_DB_PATH"] = os.path.join(self.tmpdir.name, "config_test.db")

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmpdir.cleanup()

    def read(self, name, raw):
        self._saved.setdefault(name, os.environ.get(name))
        if raw is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = raw
        return tutor._read_positive_int(name, 123, minimum=1)

    def test_unset_uses_the_default(self):
        self.assertEqual(self.read("AITUTOR_TEST_CFG", None), 123)

    def test_blank_is_treated_as_unset(self):
        """空字符串也算没设置——部署平台上清空一个变量后常常留个空串。"""
        self.assertEqual(self.read("AITUTOR_TEST_CFG", "   "), 123)

    def test_valid_value_is_used(self):
        self.assertEqual(self.read("AITUTOR_TEST_CFG", "500"), 500)

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertEqual(self.read("AITUTOR_TEST_CFG", " 500 "), 500)

    def test_non_numeric_is_rejected_loudly(self):
        """写了个不是数字的值 → 启动就报错，不静默用默认值。"""
        with self.assertRaises(RuntimeError) as ctx:
            self.read("AITUTOR_TEST_CFG", "abc")
        self.assertIn("AITUTOR_TEST_CFG", str(ctx.exception))

    def test_zero_and_negative_are_rejected(self):
        for bad in ("0", "-1", "-100"):
            with self.assertRaises(RuntimeError, msg=bad):
                self.read("AITUTOR_TEST_CFG", bad)

    def test_maximum_is_enforced(self):
        self._saved.setdefault("AITUTOR_TEST_CFG", os.environ.get("AITUTOR_TEST_CFG"))
        os.environ["AITUTOR_TEST_CFG"] = "70000"
        with self.assertRaises(RuntimeError):
            tutor._read_positive_int("AITUTOR_TEST_CFG", 5000, minimum=1, maximum=65535)

    def test_shipped_defaults_are_the_documented_ones(self):
        """默认值必须和文档里写的一致，否则文档就是错的。"""
        self.assertEqual(tutor.MAX_QUESTION_LENGTH, 500)
        self.assertEqual(tutor.DAILY_API_LIMIT, 50)
        self.assertEqual(tutor.MAX_CONTENT_LENGTH, 64 * 1024)

    def test_debug_is_off_unless_explicitly_enabled(self):
        """【核心】调试模式默认必须是关的——它开到公网上等于让人在你服务器上执行代码。"""
        self.assertFalse(tutor.DEBUG)

    def test_debug_only_turns_on_for_explicit_truthy_values(self):
        for value, expected in [("1", True), ("true", True), ("TRUE", True),
                                ("yes", True), ("0", False), ("", False), ("no", False)]:
            self._saved.setdefault("FLASK_DEBUG", os.environ.get("FLASK_DEBUG"))
            os.environ["FLASK_DEBUG"] = value
            reloaded = importlib.reload(tutor)
            self.assertEqual(reloaded.DEBUG, expected, "FLASK_DEBUG=" + repr(value))
        # 还原成干净状态，并重新打桩，别把真客户端留给后面的测试
        self._saved.setdefault("FLASK_DEBUG", None)
        os.environ.pop("FLASK_DEBUG", None)
        importlib.reload(tutor)
        tutor.client = FakeClient()
        tutor.retriever.retrieve = fake_retrieve


# ===================== 9. 生产化：健康检查 =====================

class TestHealthEndpoint(ChatTestCase):

    def test_health_returns_200_and_fixed_json(self):
        """【核心】数据库正常时返回 200 和固定 JSON。"""
        c = tutor.app.test_client()
        r = c.get("/health")

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"status": "ok"})

    def test_health_returns_503_when_database_is_unreachable(self):
        """【核心】数据库不可用 → 503，而且响应体里不能有内部信息。"""
        tutor.DB_PATH = os.path.join(self.tmpdir.name, "没有这个目录", "x.db")

        c = tutor.app.test_client()
        r = c.get("/health")

        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.get_json(), {"status": "unhealthy"})

        body = r.get_data(as_text=True)
        self.assertNotIn("没有这个目录", body, "响应里泄露了数据库路径")
        self.assertNotIn("Traceback", body)
        self.assertNotIn("sqlite", body.lower())

    def test_health_does_not_call_the_model(self):
        c = tutor.app.test_client()
        c.get("/health")
        self.assertEqual(len(self.fake.calls), 0)

    def test_health_does_not_touch_the_retriever(self):
        c = tutor.app.test_client()
        c.get("/health")
        self.assertEqual(len(RETRIEVAL_CALLS), 0)

    def test_health_does_not_write_chat_rows(self):
        c = tutor.app.test_client()
        c.get("/health")
        self.assertEqual(self.all_rows(), [])

    def test_health_does_not_consume_quota(self):
        """【核心】探针会被频繁调用，绝不能吃掉用户的模型额度。"""
        for _ in range(5):
            tutor.app.test_client().get("/health")
        self.assertEqual(self.used_today(), 0)

    def test_health_does_not_create_a_session(self):
        """【核心】探针没有浏览器，不该在数据库里攒下一堆垃圾会话。"""
        c = tutor.app.test_client()
        c.get("/health")
        self.assertIsNone(self.session_id_of(c), "健康检查不该创建会话")

    # ---------- 小工具 ----------

    def used_today(self):
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT used FROM api_usage WHERE day = ?",
                               (tutor._utc_day(),)).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()


# ===================== 10. 生产化：输入保护 =====================

class TestInputProtection(ChatTestCase):

    def test_overlong_question_does_no_work_at_all(self):
        """【核心】超长问题：不检索、不调模型、不写库——三样都不能发生。"""
        tutor.MAX_QUESTION_LENGTH = 20
        c = tutor.app.test_client()

        r = self.ask(c, "问" * 21)

        self.assertEqual(r.status_code, 200)
        self.assertIn(tutor.TOO_LONG_MESSAGE, r.get_data(as_text=True))
        self.assertEqual(len(RETRIEVAL_CALLS), 0, "超长问题不该检索")
        self.assertEqual(len(self.fake.calls), 0, "超长问题不该调用模型")
        self.assertEqual(self.all_rows(), [], "超长问题不该写数据库")

    def test_overlong_question_does_not_consume_quota(self):
        tutor.MAX_QUESTION_LENGTH = 20
        c = tutor.app.test_client()
        self.ask(c, "问" * 21)
        self.assertEqual(self.used_today(), 0)

    def test_question_at_the_limit_is_allowed(self):
        """刚好等于上限的应当放行（边界是「超过」才拦）。"""
        tutor.MAX_QUESTION_LENGTH = 20
        c = tutor.app.test_client()
        self.ask(c, "问" * 20)
        self.assertEqual(len(RETRIEVAL_CALLS), 1)
        self.assertEqual(len(self.fake.calls), 1)

    def test_oversized_request_body_is_rejected_safely(self):
        """【核心】超大请求体要安全拒绝，不显示任何内部异常。"""
        c = tutor.app.test_client()
        c.get("/")

        r = c.post("/", data={"question": "啊" * 200000})   # 远超 64KB 的请求体上限

        self.assertEqual(r.status_code, 413)
        html = r.get_data(as_text=True)
        self.assertIn(tutor.TOO_LARGE_MESSAGE, html)
        self.assertNotIn("Traceback", html)
        self.assertNotIn("RequestEntityTooLarge", html)
        self.assertEqual(self.all_rows(), [], "超大请求不该写数据库")
        self.assertEqual(len(self.fake.calls), 0, "超大请求不该调用模型")

    def used_today(self):
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT used FROM api_usage WHERE day = ?",
                               (tutor._utc_day(),)).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()


# ===================== 11. 生产化：每日额度 =====================

class TestDailyQuota(ChatTestCase):

    def used_today(self):
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT used FROM api_usage WHERE day = ?",
                               (tutor._utc_day(),)).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def test_normal_question_consumes_exactly_one(self):
        c = tutor.app.test_client()
        self.ask(c, "随便问一句")
        self.assertEqual(self.used_today(), 1)
        self.assertEqual(len(self.fake.calls), 1)

    def test_requests_within_the_limit_reach_the_model(self):
        tutor.DAILY_API_LIMIT = 3
        c = tutor.app.test_client()
        for i in range(3):
            self.ask(c, "第 " + str(i) + " 个问题")
        self.assertEqual(len(self.fake.calls), 3)
        self.assertEqual(self.used_today(), 3)

    def test_requests_beyond_the_limit_do_not_reach_the_model(self):
        """【核心】额度用完后不再调用模型，给固定提示。"""
        tutor.DAILY_API_LIMIT = 2
        c = tutor.app.test_client()

        self.ask(c, "第一个")
        self.ask(c, "第二个")
        r = self.ask(c, "第三个")                        # 这一次应该被挡住

        self.assertEqual(len(self.fake.calls), 2, "超额后不该再调用模型")
        self.assertIn(tutor.QUOTA_MESSAGE, r.get_data(as_text=True))
        self.assertEqual(self.used_today(), 2, "被挡住的那次不该再加计数")

    def test_quota_rejection_is_not_counted_as_an_error(self):
        """额度用完是正常业务情况，不该被当成内部错误。"""
        tutor.DAILY_API_LIMIT = 1
        c = tutor.app.test_client()
        self.ask(c, "第一个")
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, "第二个")
        self.assertNotIn("internal_error", "\n".join(captured.output))

    def test_empty_question_does_not_consume_quota(self):
        c = tutor.app.test_client()
        self.ask(c, "")
        self.ask(c, "   ")
        self.assertEqual(self.used_today(), 0)

    def test_empty_retrieval_still_consumes_quota(self):
        """【本轮改动】没检索到资料【也会】调用模型，所以也要占额度。

        以前「检索为空」等于不花钱；现在它可能变成一次真实调用
        （去判断这是不是正常的英语问题），因此必须占额度。
        """
        self.use_empty_retrieval()
        self.fake.reply = rag_reply(decision="general_answer", answer="通用回答", citations=[])
        c = tutor.app.test_client()
        self.ask(c, "this 和 that 有什么区别？")

        self.assertEqual(len(self.fake.calls), 1, "chunks 为空时也该调用模型")
        self.assertEqual(self.used_today(), 1, "调了模型就该占额度")

    def test_retrieval_error_does_not_consume_quota(self):
        """【核心】检索本身出错时不调模型，也不该占额度。"""
        def boom(question, top_k=3):
            raise RuntimeError("检索挂了")
        tutor.retriever.retrieve = boom

        c = tutor.app.test_client()
        self.ask(c, "随便问一句")

        self.assertEqual(len(self.fake.calls), 0, "检索失败不该调模型")
        self.assertEqual(self.used_today(), 0, "没调模型就不该占额度")

    def test_model_failure_still_counts(self):
        """【核心】模型调用失败，这次仍然计入额度。

        理由：失败的这一次【已经真实发出过外部请求】了，风险已经产生。
        网络抖动、接口报错都不该变成「免费重试」的漏洞。
        """
        def boom(model, messages, **kwargs):
            raise RuntimeError("模拟故障")
        self.fake.chat.completions.create = boom

        c = tutor.app.test_client()
        self.ask(c, "这一问会失败")

        self.assertEqual(self.used_today(), 1, "失败的那次也必须计数")

    def test_quota_survives_an_app_restart(self):
        """【核心】额度存在数据库里，重启应用不会自动归零。"""
        global tutor
        tutor.DAILY_API_LIMIT = 1
        c = tutor.app.test_client()
        self.ask(c, "重启前用掉唯一一次")
        self.assertEqual(self.used_today(), 1)

        # ---- 模拟重启 ----
        tutor = importlib.reload(tutor)
        tutor.client = FakeClient()
        tutor.retriever.retrieve = fake_retrieve
        tutor.DAILY_API_LIMIT = 1

        c2 = tutor.app.test_client()
        self.ask(c2, "重启后还想再问")

        self.assertEqual(len(tutor.client.calls), 0, "额度应该已经用完了，不该再调模型")
        self.assertEqual(self.used_today(), 1)

    def test_quota_is_tracked_per_utc_day(self):
        """额度按 UTC 日期分组，昨天的用量不影响今天。"""
        yesterday = "2026-01-01"
        ok, used = tutor.reserve_api_call(1, day=yesterday)
        self.assertTrue(ok)
        self.assertEqual(used, 1)

        ok2, used2 = tutor.reserve_api_call(1, day=yesterday)
        self.assertFalse(ok2)

        ok3, _ = tutor.reserve_api_call(1, day="2026-01-02")     # 换一天，额度是新的
        self.assertTrue(ok3)

    def test_concurrent_reservations_never_exceed_the_limit(self):
        """【核心】多线程同时抢，拿到的总数绝不能超过上限。

        这正是要 BEGIN IMMEDIATE 的原因：如果实现是「先读再写」，
        两个线程都会读到「还有名额」，然后各加一次，就超了。
        """
        limit = 5
        results = []
        guard = threading.Lock()

        def worker():
            ok, _ = tutor.reserve_api_call(limit)
            with guard:
                results.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sum(results), limit, "并发时超额了")
        self.assertEqual(self.used_today_limited(limit), limit)

    def used_today_limited(self, limit):
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT used FROM api_usage WHERE day = ?",
                               (tutor._utc_day(),)).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def test_sequential_reservations_stop_exactly_at_the_limit(self):
        for i in range(4):
            ok, _ = tutor.reserve_api_call(3)
            self.assertEqual(ok, i < 3, "第 " + str(i + 1) + " 次的判定不对")


# ===================== 12. 生产化：日志不泄露 =====================

class TestLoggingSafety(ChatTestCase):

    def test_logs_never_contain_user_or_model_content(self):
        """【核心】问题正文、回答正文、密钥、session_id 都不能进日志。"""
        marker_q = "特征问题串-ZZTOP-8f3a"
        marker_a = "特征回答串-ZZTOP-4b7c"
        self.fake.reply = rag_reply(answer=marker_a)

        c = tutor.app.test_client()
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, marker_q)

        sid = self.session_id_of(c)
        blob = "\n".join(captured.output)

        self.assertNotIn(marker_q, blob, "问题正文进了日志")
        self.assertNotIn(marker_a, blob, "回答正文进了日志")
        self.assertNotIn(tutor.API_KEY, blob, "API 密钥进了日志")
        self.assertNotIn(tutor.SECRET_KEY, blob, "FLASK_SECRET_KEY 进了日志")
        self.assertIsNotNone(sid)
        self.assertNotIn(sid, blob, "原始 session_id 进了日志")

    def test_logs_never_contain_exception_text(self):
        """【核心】异常原文不能进日志，只记异常类型。"""
        marker = "特征异常串-ZZTOP-9d1e"

        def boom(model, messages, **kwargs):
            raise RuntimeError(marker)
        self.fake.chat.completions.create = boom

        c = tutor.app.test_client()
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, "随便问一句")

        blob = "\n".join(captured.output)
        self.assertNotIn(marker, blob)
        self.assertNotIn("RuntimeError", blob)           # 连类型都不必暴露给业务日志

    def test_rag_decision_is_logged_without_content(self):
        """该记的统计量要记：决策、引用数、耗时。"""
        c = tutor.app.test_client()
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            self.ask(c, "随便问一句")

        blob = "\n".join(captured.output)
        self.assertIn("question_received", blob)
        self.assertIn("question_length=", blob)
        self.assertIn("rag_decision", blob)
        self.assertIn("decision=answer", blob)
        self.assertIn("citation_count=", blob)
        self.assertIn("elapsed_ms=", blob)

    def test_field_whitelist_drops_unknown_field_names(self):
        """【核心】字段名不在白名单里就自动丢掉——这是结构性的防泄露，不靠人自觉。"""
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            tutor.log_event("synthetic_event", question="绝密内容", raw_answer="绝密回答",
                            decision="answer")

        line = "\n".join(captured.output)
        self.assertNotIn("绝密内容", line)
        self.assertNotIn("绝密回答", line)
        self.assertIn("decision=answer", line)

    def test_log_values_are_truncated_to_one_line(self):
        """就算有人塞了一长串带换行的东西进去，日志也不会被撑爆。"""
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            tutor.log_event("synthetic_event", decision="x" * 500 + "\n伪造的第二行")

        line = "\n".join(captured.output)
        self.assertNotIn("伪造的第二行", line)
        self.assertLess(len(line), 200)

    def test_startup_is_logged(self):
        """启动日志必须存在——而且它在模块级，线上 Gunicorn 也能打到。"""
        with self.assertLogs("ai_tutor", level="INFO") as captured:
            importlib.reload(tutor)
        self.assertIn("app_start", "\n".join(captured.output))
        tutor.client = FakeClient()
        tutor.retriever.retrieve = fake_retrieve


# ===================== 13. 部署产物 =====================

class TestDeploymentArtifacts(unittest.TestCase):
    """检查 requirements 和部署文档里该有的东西——它们容易写着写着就漏了。"""

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.dirname(os.path.abspath(__file__))

    def read(self, name):
        with open(os.path.join(self.root, name), encoding="utf-8") as f:
            return f.read()

    def test_gunicorn_is_a_declared_dependency(self):
        self.assertIn("gunicorn", self.read("requirements.txt").lower())

    def test_deployment_doc_exists(self):
        self.assertTrue(os.path.exists(os.path.join(self.root, "DEPLOYMENT.md")))

    def test_deployment_doc_covers_the_essentials(self):
        """部署文档必须写清单 worker、$PORT、/health、持久卷路径。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertIn("--workers 1", doc, "没写清单 worker")
        self.assertIn("$PORT", doc, "没写启动命令里的端口变量")
        self.assertIn("/health", doc, "没写健康检查路径")
        self.assertIn("/data/chat.db", doc, "没写持久卷上的数据库路径")

    def test_deployment_doc_lists_the_env_vars(self):
        doc = self.read("DEPLOYMENT.md")
        for name in ("DEEPSEEK_API_KEY", "FLASK_SECRET_KEY", "INVITE_CODE_PEPPER",
                     "CHAT_DB_PATH", "MAX_QUESTION_LENGTH", "DAILY_API_LIMIT"):
            self.assertIn(name, doc, "部署文档漏了环境变量 " + name)

    def test_doc_does_not_promise_that_pepper_can_be_rotated_losslessly(self):
        """【防回归】换 pepper 的说明不能暗示「重新发码就能恢复」。

        这一版没有无损轮换的流程：`new-invite` 会创建【全新学习者】，
        能接回旧档案的 `reissue` 已停用 —— 所以「给所有人重新发码」是个做不到的承诺。
        """
        doc = self.read("DEPLOYMENT.md")

        self.assertIn("「无损轮换 pepper」的流程", doc, "没写明本版本做不到无损轮换")
        self.assertIn("身份核验方案", doc, "没指向「先停下制定身份核验方案」")
        self.assertIn("不能恢复任何人的访问权", doc, "把导出留档说成了恢复办法")
        self.assertNotIn("给所有人重新发码", doc, "又出现了那句做不到的承诺")

        # 【统计留档不能被说成「对账」】只留 status 和时间字段，
        # 那就只能看分布 —— 既对不上人，也对不上具体哪张码。这句承诺必须是这个口径。
        self.assertIn("状态与时间的分布", doc, "没写明留档到底能看到什么")
        self.assertIn("不要把它当成恢复方案", doc, "没把「留档 ≠ 恢复」这条边界写死")
        self.assertNotIn("谁、什么时候、哪张码被用过", doc, "又出现了那句做不到的对账承诺")

    def test_readme_links_to_the_deployment_doc(self):
        self.assertIn("DEPLOYMENT.md", self.read("README.md"))

    def test_no_railway_config_files_were_added(self):
        """Railway 的旧 Config as Code 已弃用，本阶段不新增这类文件。"""
        for name in ("railway.toml", "railway.json"):
            self.assertFalse(os.path.exists(os.path.join(self.root, name)),
                             name + " 不该存在（部署参数写在 DEPLOYMENT.md 里）")

    # ---------- 平台事实的防回归检查 ----------
    #
    # 【为什么这些要写成测试】
    # 部署文档写错平台行为，比代码写错更难发现：它不会报错、不会崩，
    # 只会让人按错误的心智模型去运维。这两条都是被独立验收抓出来的错误，
    # 所以钉成测试，防止以后改写文档时又漂回去。

    def test_doc_says_health_check_is_not_continuous_monitoring(self):
        """【防回归】必须写清楚：健康检查不是持续监控，而是部署时的一次性验收。"""
        doc = self.read("DEPLOYMENT.md")

        self.assertIn("不是「持续监控」", doc, "没写明健康检查不是持续监控")
        self.assertIn("验收", doc, "没说明它其实是「部署前的验收」")

    def test_doc_says_restarts_are_governed_by_restart_policy(self):
        """【防回归】重启是另一套机制管的，不能和 /health 混为一谈。"""
        doc = self.read("DEPLOYMENT.md")

        self.assertIn("Restart Policy", doc, "没写清重启由 Restart Policy 管理")
        # 部署完成后 Railway 就不再访问这个端点——这一点必须写明
        self.assertIn("不再访问", doc, "没写清部署完成后就不再调用 /health")

    def test_doc_no_longer_claims_health_failures_restart_the_service(self):
        """【防回归】原来那句「连续失败就会重启服务」必须已经不在了。"""
        doc = self.read("DEPLOYMENT.md")

        self.assertNotIn("连续失败就会重启服务", doc, "原来那句错误说法还在")

        # 如果文中还提到「连续失败」这个概念，必须是在【否定】它
        # （现在它只出现在「常见误解」对照表里，带着 ❌ 标记）
        for para in doc.split("\n\n"):
            if "连续失败" in para:
                self.assertIn("❌", para, "提到「连续失败」却没有明确否定它")

    def test_doc_no_longer_points_at_volume_download(self):
        """【防回归】「在卷操作里下载」不是 Railway 的正式备份方式，不能写。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertNotIn("卷操作里下载", doc)

    def test_doc_does_not_present_a_same_volume_copy_as_a_backup(self):
        """【防回归】`cp` 到同一个卷，绝不能写成备份方案。

        文档里【允许】出现这条命令——但只能在「不要这么做」的警告里。
        所以这里查两件事：
          1. 原来那两句推荐语必须已经不在了；
          2. 凡出现这条命令的地方，必须同时有明确的否定措辞。
        """
        doc = self.read("DEPLOYMENT.md")

        self.assertNotIn("备份很简单", doc, "原来那句把 cp 当方案的话还在")
        self.assertNotIn("或者临时起一个能跑", doc, "原来那句把 cp 当方案的话还在")

        for para in doc.split("\n\n"):
            if "cp /data/chat.db" in para:
                self.assertIn("不要", para, "提到了 cp 却没有明确否定它")

    def test_doc_explains_a_same_volume_copy_is_not_a_disaster_backup(self):
        """【防回归】要讲明白为什么同卷拷贝不算备份——两份会一起没。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertIn("同一个卷里的一份拷贝", doc)
        self.assertIn("一起没", doc)

    def test_doc_describes_the_official_volume_backup_flow(self):
        """【防回归】备份要走 Railway 官方的卷备份流程，四要素缺一不可。"""
        doc = self.read("DEPLOYMENT.md")

        self.assertIn("Backups", doc, "没提 Backups 页面")
        self.assertIn("手动备份", doc, "没提手动创建备份")
        self.assertIn("恢复", doc, "没提怎么恢复")
        self.assertIn("计费", doc, "没提备份会产生存储费用")

    def test_doc_is_honest_that_no_backup_was_created(self):
        """【诚实性】没做过的事就要写没做过。

        【2026-10-12 更新】原来断言的是「没有实际创建任何备份」这句写作时的说明。
        后来实测发现：平台**拒绝**为非 Pro 工作区创建手动备份
        （BAD_USER_INPUT: ... only available for Pro workspaces），
        所以文档改成了更强的一版：「这个卷**没有任何备份**，也没有备份计划，
        而且平台**拒绝创建**」。
        断言跟着改成这组新措辞 —— 本意没变：没备份这件事必须白纸黑字写出来。
        """
        doc = self.read("DEPLOYMENT.md")
        self.assertIn("没有任何备份", doc, "没写「没有备份」")
        self.assertIn("拒绝创建", doc, "没写「平台拒绝创建」")

    # ---------- 健康检查的「请求次数」与「重新部署停机」（第二轮修正） ----------
    #
    # 【为什么同一个地方被改了两轮】
    # 第一轮修的是「不是持续监控」；但那一版又把「反复请求直到 2xx」写成了「调用一次」，
    # 还顺手加了一句「旧版本继续服务，用户不受影响」——后者没有官方依据，属于过度承诺。
    #
    # 部署文档里一句没根据的承诺，会让人对停机毫无准备，比不写还糟。
    # 所以在测试里同时钉住「该有的」和「不该有的」。

    def test_doc_says_health_check_is_retried_until_2xx(self):
        """【核心】健康检查是【反复请求】直到收到 2xx，不是只调用一次。"""
        doc = self.read("DEPLOYMENT.md")

        self.assertIn("反复请求", doc, "没写清楚健康检查会反复请求")
        self.assertIn("直到收到 2xx", doc, "没写清楚反复请求的终止条件")

    def test_doc_does_not_say_the_health_check_is_called_once(self):
        """【核心】不能写成「调用一次 /health」。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertNotIn("调用一次", doc,
                         "「调用一次」的说法回来了——它是错的，Railway 会反复请求")

    def test_doc_does_not_promise_the_old_version_keeps_serving(self):
        """【核心】不要对失败后的流量行为作没有官方依据的承诺。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertNotIn("旧版本继续服务", doc)
        self.assertNotIn("用户不受影响", doc)

    def test_doc_says_a_timed_out_deploy_is_marked_failed(self):
        """超时仍未成功 → 新部署被标记为失败。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertIn("标记为失败", doc)
        self.assertIn("超时", doc)

    def test_doc_warns_that_redeploying_causes_downtime(self):
        """【核心】必须写明：挂了持久卷的服务，重新部署会短暂停机。"""
        doc = self.read("DEPLOYMENT.md")

        self.assertIn("短暂停机", doc, "没写明重新部署会短暂停机")
        self.assertIn("不能保证零停机", doc, "没写明健康检查也不能保证零停机")

    def test_doc_explains_why_the_downtime_is_unavoidable(self):
        """还要解释原因：Railway 要避免两个部署同时挂载同一个卷。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertIn("两个部署同时挂载同一个卷", doc)

    def test_doc_frames_the_downtime_as_a_known_architecture_limit(self):
        """要把它说成「SQLite + 单卷架构的已知限制」，而不是配置失误。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertIn("已知限制", doc)
        self.assertIn("PostgreSQL", doc)                 # 指明了真正的出路


class TestInterfaceGuidance(ChatTestCase):
    """首页的入门引导必须清楚，且示例不能擅自替用户发问。"""

    def test_first_visit_shows_how_to_start(self):
        html = tutor.app.test_client().get("/").get_data(as_text=True)
        self.assertIn("把问题说出来就好", html)
        self.assertIn("还没有对话", html)
        self.assertIn("点选只会填入输入框，确认后再发送", html)
        self.assertIn('href="/invite"', html)

    def test_starters_are_not_submit_buttons(self):
        html = tutor.app.test_client().get("/").get_data(as_text=True)
        self.assertEqual(html.count('type="button" class="starter"'), 3)
        self.assertIn("input.value = starter.dataset.prompt", html)
        self.assertIn('id="ask-form"', html)

    def test_shared_stylesheet_is_served_without_auth(self):
        client = tutor.app.test_client()
        self.assertIn('/static/ui.css', client.get("/").get_data(as_text=True))
        self.assertIn('/static/ui.css', client.get("/invite").get_data(as_text=True))
        css = client.get("/static/ui.css")
        try:
            self.assertEqual(css.status_code, 200)
            self.assertIn("@media (max-width: 600px)", css.get_data(as_text=True))
        finally:
            css.close()


class TestModelOutageRecovery(ChatTestCase):
    """跨浏览器故障只占一次额度，失败不写回答，冷却结束后可恢复。"""

    def test_busy_then_timeout_then_cooldown_then_recovery(self):
        entered = threading.Event()
        release = threading.Event()
        responses = []
        original_create = self.fake.chat.completions.create

        def blocked(model, messages, **kwargs):
            self.fake.calls.append({"model": model, "messages": messages})
            entered.set()
            release.wait(3)
            raise TimeoutError("private-timeout-detail")

        self.fake.chat.completions.create = blocked
        request_thread = threading.Thread(target=lambda: responses.append(
            self.ask(tutor.app.test_client(), "第一个问题")))
        request_thread.start()
        try:
            self.assertTrue(entered.wait(2))
            second = self.ask(tutor.app.test_client(), "另一台设备的问题")
            self.assertIn(tutor.LOCKED_MESSAGE, second.get_data(as_text=True))
            self.assertEqual(len(self.fake.calls), 1)
        finally:
            release.set()
            request_thread.join(4)
        self.assertFalse(request_thread.is_alive())
        self.assertEqual(responses[0].status_code, 503)
        self.assertFalse(tutor._lock.locked())
        self.assertEqual(self.all_rows(), [])
        self.assertNotIn("private-timeout-detail", responses[0].get_data(as_text=True))

        cooldown = self.ask(tutor.app.test_client(), "再次尝试")
        self.assertEqual(cooldown.status_code, 503)
        self.assertIn(tutor.MODEL_UNAVAILABLE_MESSAGE, cooldown.get_data(as_text=True))
        self.assertEqual(len(self.fake.calls), 1)
        conn = sqlite3.connect(self.db_path)
        try:
            used = conn.execute("SELECT used FROM api_usage WHERE day = ?",
                                (tutor._utc_day(),)).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(used, 1, "繁忙/冷却期间不能再次扣额度")

        tutor._model_retry_after = 0.0  # 模拟冷却时间已经过去，无需真的等待。
        self.fake.chat.completions.create = original_create
        recovered = self.ask(tutor.app.test_client(), "现在可以回答")
        self.assertEqual(recovered.status_code, 302)
        self.assertEqual(len(self.all_rows()), 2)
        self.assertEqual(len(self.fake.calls), 2)

    def test_provider_error_keeps_existing_history_and_input(self):
        browser = tutor.app.test_client()
        self.ask(browser, "已经回答的问题")
        existing = self.all_rows()

        def failed(*args, **kwargs):
            raise RuntimeError("sensitive-provider-details")

        self.fake.chat.completions.create = failed
        failed_response = self.ask(browser, "需要稍后重试的问题")
        self.assertEqual(failed_response.status_code, 503)
        self.assertEqual(self.all_rows(), existing)
        html = failed_response.get_data(as_text=True)
        self.assertIn('value="需要稍后重试的问题"', html)
        self.assertNotIn("sensitive-provider-details", html)
        self.assertFalse(tutor._lock.locked())


# =====================================================================
# 等待状态、重试时机、问题文字不丢（客户端那一半）
#
# 【和上面那组的分工】「超时/503/冷却」在服务端早就测过了
# （test_busy_then_timeout_then_cooldown_then_recovery、
#  test_provider_error_keeps_existing_history_and_input）。
# 这一组补的是另一半：请求**再也回不来**时，页面会不会一直卡在「思考中」，
# 刷新之后那句问题还在不在，以及用户怎么知道能不能重试。
#
# 【能证明什么】服务端把预算 / 回显 / 用户消息条数真的交给了页面；
# 页面脚本能把「思考中」收回来、且**不会自己重发**；
# 「刷新后把问题放回输入框」「这次发送算不算上一次的重发」这两条规则的判定表
# （用 node 跑页面里同一段函数）；
# 「响应丢了、用户又发一次」不会变成两次模型调用（服务端那一半，用身份牌挡住 ——
#  而身份牌只在「是同一条问题」且「那份回答还在库里」时才作数，见下一组）；
# 以及用**本机假上游**跑通真实链路 —— 真实的 BoundedModelClient
# 打真实 HTTP 端口，超时/上游 503 都落到同一个可重试的 503 页。
#
# 【证明不了什么】
#   · 真实浏览器里的定时器行为（这一组只测下发内容、判定表、服务端行为；
#     浏览器那一半在下面 TestWaitingStateInRealBrowser 里，用本机 Chrome 跑）；
#   · 模型真实行为（这里用的是本机假上游，不联网、不用真密钥）。
# =====================================================================

class FakeUpstream:
    """本机假上游：一个只会按 mode 回答的 HTTP 服务。

    【为什么要有它】超时到底会不会真的杀掉子进程、上游 503 会不会真的映射成
    upstream_error —— 这些应该由**真实代码路径**回答，而不是靠打桩猜。
    它只监听 127.0.0.1 的随机端口，不出网、不碰任何真实密钥。

    mode：
      ok        —— 立刻回一个格式正确的回答
      slow      —— 先停一下再回答（「慢但还活着」）
      hang      —— 收下请求就不吭声，一直拖到客户端自己的 deadline 放弃（测服务端超时）
      status503 —— 立刻回 503（测上游报错）
      gate      —— 收下请求，等测试说「可以了」再回答。用来把请求【确定性地】卡在半路：
                   不依赖任何秒数，也不会因为机器慢就变成另一回事。
    """

    ANSWER = {"decision": "general_answer", "answer": "【假上游】假回答。", "citations": []}
    GATE_WAIT_SECONDS = 30          # gate 模式下，最多替测试举着请求这么久
    HANG_WAIT_SECONDS = 60          # hang 模式下，最多装死这么久

    def __init__(self, mode="ok"):
        self.mode = mode
        self.requests = 0
        self.arrived = threading.Event()    # 收到请求就亮：测试靠它对齐时序，不靠 sleep
        self.release = threading.Event()    # gate 模式：测试放行
        self.finished = threading.Event()   # 测试收工时让还在装死的线程立刻收手
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                outer.requests += 1
                outer.arrived.set()
                mode = outer.mode              # 测试中途可以改：上游「恢复」了
                if mode == "hang":
                    outer.finished.wait(FakeUpstream.HANG_WAIT_SECONDS)
                    return                     # 什么都不回：连接就这么挂着
                if mode == "gate":
                    outer.release.wait(FakeUpstream.GATE_WAIT_SECONDS)
                if mode == "status503":
                    body = json.dumps({"error": {"message": "upstream busy"}}).encode()
                    self.send_response(503)
                else:
                    if mode == "slow":
                        time.sleep(1.5)
                    body = json.dumps({
                        "id": "x", "object": "chat.completion", "created": 0,
                        "model": "deepseek-chat",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant",
                                                 "content": json.dumps(
                                                     FakeUpstream.ANSWER,
                                                     ensure_ascii=False)}}],
                    }).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass                                # 别把请求日志刷到测试输出里

        class QuietUpstreamServer(http.server.ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                pass                     # 客户端被 deadline 杀掉时连接会被重置，那是正常现象

        self.server = QuietUpstreamServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.server.server_address[1]

    def stop(self):
        self.finished.set()                # 先让装死/举着的线程收手，再关服务
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


# 【给模型子进程留出启动时间的超时值】
#
# 客户端每问一次都会【新起一个子进程】（Python 解释器 + import openai）。
# 这台机器上实测：光 import openai 就要 1.7～2.0 秒，加上解释器启动总共约 2.1 秒
# （空启动只要 0.14 秒，所以慢的就是那个 import）；端到端一次提问大约 1.9～3.2 秒。
#
# 结论：任何「小于冷启动时间」的 deadline，都会在子进程还没连上上游时把它杀掉。
# 那时候上游收到的请求数是 0，测试会报「0 != 1」——但它证明的不是「超时处理有问题」，
# 而是「这次的超时设得比启动还短」。所以凡是需要上游【真的收到】请求的用例，
# 都留足余量，而不是原来那个 2 秒。
#
# 这个值本身是「装死用例」的 deadline：测试要等满它，所以不能随便往大里写。
# 10 秒 ≈ 正常一次提问的 3～5 倍，机器偶尔忙一下也还站得住；
# 万一真的没连上，用例会明确报「上游一次请求都没收到」，不会含混成 0 != 1。
HUNG_UPSTREAM_TIMEOUT_SECONDS = 10

# 【真上游那一组的兜底超时】不是被测对象，够宽就行：它只决定「真出错时多久放弃」。
# 正常一次提问约 1.9～3.2 秒，给 20 秒是为了不让机器偶尔一忙就误报成服务故障。
REAL_UPSTREAM_TIMEOUT_SECONDS = 20


class TestWaitingStateRecoveryAndRetry(ChatTestCase):

    def page(self, client=None):
        client = client or tutor.app.test_client()
        return client.get("/").get_data(as_text=True)

    @staticmethod
    def body_tag(html):
        start = html.index("<body")
        return html[start:html.index(">", start)]

    def use_fake_upstream(self, mode, timeout=10):
        """把 app 的客户端换成真的 BoundedModelClient，指向本机假上游。"""
        upstream = FakeUpstream(mode)
        self.addCleanup(upstream.stop)
        tutor.client = model_client.BoundedModelClient(
            api_key="test-key-not-a-real-key", base_url=upstream.base_url,
            timeout=timeout)
        return upstream

    def ask_with_attempt(self, client, question, attempt):
        """像页面那样提问：问题 + 这次尝试的身份牌。"""
        return client.post("/", data={"question": question, "attempt": attempt})

    def run_in_node(self, script):
        """把一小段 JS 交给 node 跑（本机没有 node 就跳过这一条）。"""
        node = shutil.which("node")
        if node is None:
            self.skipTest("本机没有 node，跳过判定表")
        return subprocess.run([node, "-e", script], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)

    def recovery_logic(self):
        """把页面里那段【纯函数】原样抠出来 —— 测的就是它在浏览器里跑的那一份。"""
        html = self.page()
        found = re.search(r"// recovery-logic:start(.*?)// recovery-logic:end", html, re.S)
        self.assertIsNotNone(found, "找不到 recovery-logic 标记，判定表没法测")
        return found.group(1)

    # ---------- A. 页面把「服务端事实」交给客户端 ----------

    def test_the_page_carries_the_server_budget(self):
        """页面上那个等待上限必须来自服务端配置，不能在脚本里另写一个数字。"""
        html = self.page()
        body = self.body_tag(html)
        self.assertIn('data-model-budget-seconds="%d"'
                      % tutor.MODEL_REQUEST_TIMEOUT_SECONDS, body)
        self.assertIn("dataset.modelBudgetSeconds", html)
        # 没有错误时横幅必须真的藏着（客户端要往它里面写话，所以它得常驻在页面里）
        self.assertIn('id="error-banner" role="alert" hidden', html)

    def test_the_client_deadline_is_strictly_longer_than_the_server_budget(self):
        """看门狗必须比服务端的硬超时【晚】动手 —— 否则它会抢在正常回答前面报错。"""
        body = self.body_tag(self.page())
        budget = int(re.search(r'data-model-budget-seconds="(\d+)"', body).group(1))
        margin = int(re.search(r'data-stall-margin-seconds="(\d+)"', body).group(1))
        self.assertEqual(budget, tutor.MODEL_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(margin, tutor.MODEL_STALL_MARGIN_SECONDS)
        self.assertGreaterEqual(margin, 10, "余量太小，客户端可能先于服务端报错")
        self.assertIn("STALL_SECONDS", self.page())

    def test_the_user_message_count_is_exposed_for_the_restore_rule(self):
        """页面靠【条数】判断「刚才那一条记下来没有」，这个数只能由服务端数。

        用条数而不是「最后一条消息的文字」，是因为用户完全可能连着问两次一样的问题：
        那时文字一样、条数却会多。条数变了才说明库里有新东西。
        """
        client = tutor.app.test_client()
        self.assertIn('data-user-count="0"', self.body_tag(self.page(client)))
        self.ask(client, "已经问过的问题")
        self.assertIn('data-user-count="1"', self.body_tag(self.page(client)))
        self.ask(client, "再问一条")
        self.assertIn('data-user-count="2"', self.body_tag(self.page(client)))
        # 换一个浏览器看同一页：别人的问题不该算进来
        self.assertIn('data-user-count="0"',
                      self.body_tag(self.page(tutor.app.test_client())))

    def test_the_page_carries_a_data_epoch_that_ordinary_loads_do_not_change(self):
        """页面要拿到「这份数据（这个浏览器名下这一摊）的代号」。

        【为什么非得由服务端给】它是「那条暂存的问题属于哪一份数据」的凭据 ——
        要跟「删除」这件事对得上，而删除只在服务端发生，页面自己编一个没有意义。
        【它换不换，由删除那一组看着】这里只管两件事：页面确实拿到了；
        以及【平时刷新和提问都不许换】—— 换了就等于把用户手上那条还没答完的问题白白作废。
        """
        client = tutor.app.test_client()
        first = self.epoch_in(self.body_tag(self.page(client)))
        self.assertRegex(first, r"^[0-9a-f]{32}$", "代号该是 32 位十六进制")

        self.assertEqual(self.epoch_in(self.body_tag(self.page(client))), first,
                         "普通刷新不该换代号")
        self.ask(client, "随便问一句")
        self.assertEqual(self.epoch_in(self.body_tag(self.page(client))), first,
                         "提问也不该换代号")
        # 【不同的浏览器各是各的】它属于这个浏览器，不是全站共用的一个数
        self.assertNotEqual(self.epoch_in(self.body_tag(self.page(tutor.app.test_client()))),
                            first)

    @staticmethod
    def epoch_in(body_tag):
        """从 <body ...> 那段里把数据代号抠出来。"""
        found = re.search(r'data-data-epoch="([^"]*)"', body_tag)
        if found is None:
            raise AssertionError("首页没有下发数据代号：" + body_tag)
        return found.group(1)

    # ---------- B. 收回等待状态：只动界面，绝不重发 ----------

    def test_the_inline_script_parses(self):
        """【防打错】页面里那段内联脚本必须能通过语法检查。

        下面的断言都是按文本匹配的：万一我把某一行写坏了（少个括号），
        文本断言照样能过，但真实浏览器里整段脚本都不会执行 —— 那才是最难查的一种坏。
        这里用 node 只做语法解析（不执行），本机没有 node 就跳过。
        """
        node = shutil.which("node")
        if node is None:
            self.skipTest("本机没有 node，跳过语法检查")
        html = self.page()
        blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
        self.assertEqual(len(blocks), 1, "内联脚本块不止一个，这里只检查唯一的那段")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "inline.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(blocks[0])
            proc = subprocess.run([node, "--check", path], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=30)
        self.assertEqual(proc.returncode, 0, "内联脚本有语法错误：\n" + (proc.stderr or ""))

    def test_the_release_function_puts_the_composer_back(self):
        html = self.page()
        block = html[html.index("function releaseComposer"):]
        block = block[:block.index("\n    }")]
        for expected in ("input.readOnly = false", "classList.remove('waiting')",
                         "button.disabled = false", "button.innerHTML = buttonLabel",
                         "sending = false"):
            self.assertIn(expected, block, "收回等待状态时漏了：" + expected)

    def test_nothing_in_the_page_resends_or_reloads_on_its_own(self):
        """【红线】客户端超时只恢复界面：不重发、不刷新。

        客户端能不能「中止」那次 POST？不能 —— 表单导航一旦发出去，
        浏览器这边没有中止的口子。所以这里【不许】出现任何自动重发的写法。
        """
        html = self.page()
        for forbidden in ("fetch(", "XMLHttpRequest", "form.submit(",
                          "location.reload(", "location.href ="):
            self.assertNotIn(forbidden, html, "页面不该自己重发或重载：" + forbidden)

    def test_the_watchdog_only_lifts_the_waiting_state(self):
        html = self.page()
        block = html[html.index("watchdog = window.setTimeout"):]
        block = block[:block.index("}, STALL_SECONDS * 1000);")]
        self.assertIn("releaseComposer()", block)
        self.assertIn("notice(", block)
        self.assertNotIn("submit", block, "看门狗里不该有任何提交动作")

    def test_the_watchdog_notice_does_not_tell_the_user_to_just_resend(self):
        """【这一轮要改的正是这句话】看门狗只能把「结果未知」说清楚，
        并且给一个【手动】刷新入口；不能把「再发一次」当成建议递给用户 ——
        第一次可能早就答完了、只是响应丢了，重发就是第二次调用。"""
        html = self.page()
        block = html[html.index("watchdog = window.setTimeout"):]
        block = block[:block.index("}, STALL_SECONDS * 1000);")]
        self.assertIn("结果未知", block)
        self.assertIn("不会自动重发", block)
        for forbidden in ("再点「发送」", "重新发送", "可以直接重发"):
            self.assertNotIn(forbidden, block, "看门狗不该建议用户直接重发：" + forbidden)
        # 手动刷新入口：一个普通链接，点不点由用户决定
        self.assertIn('<a class="error-action" id="error-action" href="/"', html)
        self.assertIn("notice(", block)
        self.assertIn(", true)", block, "看门狗这条提示要把「刷新看看结果」露出来")

    def test_the_form_carries_an_attempt_token(self):
        """每次发送都要带一个身份牌，服务端才认得出「这是同一次尝试的重发」。"""
        html = self.page()
        self.assertIn('name="attempt" id="attempt-input"', html)
        self.assertIn("document.getElementById('attempt-input').value = attempt.token",
                      html)

    def test_a_second_submit_while_waiting_is_still_swallowed(self):
        """重复点击保护还在：等待期间再点，事件被拦下、不再发一次。"""
        html = self.page()
        block = html[html.index("form.addEventListener('submit'"):]
        block = block[:block.index("if (sending) {") + 200]
        self.assertIn("if (sending) {", block)
        self.assertIn("event.preventDefault();", block)

    # ---------- C. 两条判断规则（用 node 跑页面里同一段函数） ----------

    def test_the_restore_rule_table(self):
        """刷新后「放回输入框 / 认定答过了 / 丢掉」的判定表。

        【为什么用条数而不是文字】看第 4 条：连着问两次一模一样的问题时，
        文字一样、条数却多了一条 —— 只看文字会把「答完了」认成「没答」，或者反过来。
        【为什么清空数据是单独一列】看最后两条：那种情况下条数根本看不出来
        （基准是发出去时的 0，删完还是 0），只能靠服务端那个一次性信号判死。
        """
        script = self.recovery_logic() + """
const cases = [
  // 服务端回显, pending, 条数, 刚清空过, 现在还绑着吗, 期望 restore, 期望 clear, 期望 answered
  ['', null, 0, false, false, null, false, false],                 // 什么都没记着
  ['', {q:'还没答完的问题', before:0}, 0, false, false, '还没答完的问题', false, false], // 没被记下来 → 放回输入框
  ['', {q:'已答完的问题', before:0}, 2, false, false, null, true, true],  // 记下来了 → 认定答过了
  ['', {q:'同样的话', before:2}, 3, false, false, null, true, true],      // 连着问两次同样的话：条数 2→3
  ['', {q:'同样的话', before:2}, 2, false, false, '同样的话', false, false], // 条数没动 → 还是没答完
  ['服务端回显了', {q:'还没答完的问题', before:0}, 0, false, false, null, false, false], // 已回显 → 不重复写
  ['', {q:'   ', before:0}, 0, false, false, null, false, false],   // 空白不算一条问题
  ['', {q:'刚被清掉的问题', before:0}, 0, true, false, null, true, false],  // 刚清空过 → 丢掉，绝不放回
  ['服务端回显了', {q:'刚被清掉的问题', before:0}, 0, true, false, null, true, false], // 同上，回显也不留
  ['', null, 0, true, false, null, true, false],                    // 刚清空过、本来就没记着 → 清干净
  // 没有那个一次性信号时：靠「历史变短了」认出来（别的标签页删的，信号被它用掉了）
  ['', {q:'别的标签页删掉的', before:3}, 0, false, false, null, true, false],
  ['', {q:'别的标签页删掉的', before:3}, 2, false, false, null, true, false],
  ['', {q:'别的标签页删掉的', before:3}, 3, false, false, '别的标签页删掉的', false, false],
  // 条数看不出来的那一种（暂存时基准就是 0）：靠「原先绑着、现在不绑了」认出来
  ['', {q:'解绑前问出去的', before:0, bound:true}, 0, false, false, null, true, false],
  ['', {q:'解绑前问出去的', before:0, bound:true}, 0, false, true, '解绑前问出去的', false, false],
  ['', {q:'匿名时问的', before:0, bound:false}, 0, false, false, '匿名时问的', false, false], // 本来就没绑过
];
let bad = 0;
for (const [echo, pending, count, wiped, bound, wantRestore, wantClear, wantAnswered] of cases) {
  const got = planPendingRestore(echo, pending, count, wiped, bound);
  if (got.restore !== wantRestore || got.clear !== wantClear || got.answered !== wantAnswered) {
    bad++; console.log('FAIL', JSON.stringify([echo, pending, count, wiped, bound]), JSON.stringify(got));
  }
}
console.log(bad === 0 ? 'TABLE_OK' : 'TABLE_BAD');
"""
        proc = self.run_in_node(script)
        self.assertIn("TABLE_OK", proc.stdout, proc.stdout + proc.stderr)

    def test_the_attempt_rule_table(self):
        """「这次发送是不是上一次的重发」的判定表 —— 重发必须沿用同一个身份牌。"""
        script = self.recovery_logic() + """
const cases = [
  // pending, 这次打的字, 新 id, 条数, 现在绑着吗, 期望 token, 期望 before, 期望 reuse, 期望 bound
  [null, 'A', 'new1', 0, false, 'new1', 0, false, false],          // 第一次发：新牌子，记下当前条数
  [{q:'A', token:'t1', before:0, bound:true}, 'A', 'new2', 1, false, 't1', 0, true, true],
  [{q:'A', token:'t1', before:0}, '  A  ', 'new2', 1, false, 't1', 0, true, false], // 两端空白不算差别
  [{q:'A', token:'t1', before:0}, 'B', 'new2', 1, true, 'new2', 1, false, true],    // 改字 → 全新一次
  [{q:'   ', token:'t1', before:0}, 'A', 'new3', 2, true, 'new3', 2, false, true],  // 没记着有效的一条
];
let bad = 0;
for (const [pending, typed, fresh, count, bound, wantToken, wantBefore, wantReuse, wantBound] of cases) {
  const got = planAttempt(pending, typed, fresh, count, bound);
  if (got.token !== wantToken || got.before !== wantBefore || got.reuse !== wantReuse
      || got.bound !== wantBound) {
    bad++; console.log('FAIL', JSON.stringify([pending, typed, fresh, count, bound]), JSON.stringify(got));
  }
}
console.log(bad === 0 ? 'TABLE_OK' : 'TABLE_BAD');
"""
        proc = self.run_in_node(script)
        self.assertIn("TABLE_OK", proc.stdout, proc.stdout + proc.stderr)

    def test_the_epoch_stamp_rule_table(self):
        """发送时该往那条记录里盖哪个「数据代号」。

        【为什么要盖】盖的是【发出去那一刻】这份数据的代号。以后刷新时一比，
        对不上就说明这条记录指的是已经被删掉的那份数据（见下面那张表）。
        【为什么重发不重盖】重发还是同一次尝试，它指的还是原来那份数据 ——
        顺手盖成这一页的代号，就等于用一次重发把「中间被清空过」这件事抹掉了。
        """
        script = self.recovery_logic() + """
const cases = [
  // pending, 这次打的字, 新 id, 条数, 现在绑着吗, 这一页的代号, 期望 epoch, 期望 reuse
  [null, 'A', 'new1', 0, false, 'E1', 'E1', false],                             // 第一次发：盖这一页的代号
  [null, 'A', 'new1', 0, false, undefined, '', false],                          // 拿不到代号（老服务端）→ 空串
  [{q:'A', token:'t1', before:0, epoch:'E1'}, 'A', 'new2', 1, false, 'E1', 'E1', true],
  [{q:'A', token:'t1', before:0, epoch:'E1'}, 'A', 'new2', 1, false, 'E9', 'E1', true],  // 重发不重盖
  [{q:'A', token:'t1', before:0}, 'A', 'new2', 1, false, 'E9', 'E9', true],          // 老记录没盖过 → 补上这一页的
  [{q:'A', token:'t1', before:0, epoch:'E1'}, 'B', 'new3', 1, false, 'E9', 'E9', false], // 改了字 → 全新一次，盖新的
];
let bad = 0;
for (const [pending, typed, fresh, count, bound, page, wantEpoch, wantReuse] of cases) {
  const got = planAttempt(pending, typed, fresh, count, bound, page);
  if (got.epoch !== wantEpoch || got.reuse !== wantReuse) {
    bad++; console.log('FAIL', JSON.stringify([pending, typed, page]), JSON.stringify(got));
  }
}
console.log(bad === 0 ? 'TABLE_OK' : 'TABLE_BAD');
"""
        proc = self.run_in_node(script)
        self.assertIn("TABLE_OK", proc.stdout, proc.stdout + proc.stderr)

    def test_the_data_epoch_rule_table(self):
        """「数据代号对不上」这条规则单独过一遍判定表。

        【它要挡住的是哪条路】用户在这个标签页问了一句（那条记录盖着代号 E1），
        然后在【别的标签页】里把数据全删了 ——
          · 删除会换掉代号（E1 → E2），而代号在 cookie 里，两个标签页共用一份；
          · 那个一次性的「刚清空过」信号被删数据那个标签页用掉了；
          · 用户随后又拿一张新码【重新绑回来】，于是「绑定没了」这条线索也补了回去；
          · 这时候旧标签页才刷新 —— 条数是 0、当初的基准也是 0，看不出任何变化。
        上面每一条都绕开了老线索，只有代号对不上还认得出：这条问题指的是已经被删掉的那份数据。
        """
        script = self.recovery_logic() + """
const cases = [
  // 服务端回显, pending, 条数, 刚清空过, 还绑着吗, 服务端当前的代号, 期望 restore, clear, answered
  // —— 代号一样：照旧（这是绝大多数刷新）
  ['', {q:'问出去的问题', before:0, epoch:'E1'}, 0, false, true, 'E1', '问出去的问题', false, false],
  // —— 对不上：丢掉。条数一样、绑定还在，老线索全都说不出话
  ['', {q:'删掉之前问出去的', before:0, epoch:'E1'}, 0, false, true, 'E2', null, true, false],
  // —— 对不上，而且条数还「变多了」（删完又问了新的）：也不能说「已经答完了」，
  //     那份回答根本不在，说了就是撒谎
  ['', {q:'删掉之前问出去的', before:0, epoch:'E1'}, 3, false, true, 'E2', null, true, false],
  // —— 对不上，而且条数会「变短」（老线索本来也认得出）：还是丢掉，且不解释
  ['', {q:'删掉之前问出去的', before:3, epoch:'E1'}, 0, false, true, 'E2', null, true, false],
  // —— 对不上，人还是匿名的（没绑过）：照样丢掉
  ['', {q:'删掉之前问出去的', before:0, epoch:'E1'}, 0, false, false, 'E2', null, true, false],
  // —— 两个信号同时到：丢掉（和只给一次性信号时同一个结果）
  ['', {q:'删掉之前问出去的', before:0, epoch:'E1'}, 0, true, true, 'E2', null, true, false],
  // —— 服务端这次回显了这条问题（出错分支），但代号对不上：照样丢掉。
  //     丢掉的只是那个身份牌照；它的回答已经没了，服务端那边也不会再把它当成「答过了」
  ['服务端回显了', {q:'删掉之前问出去的', before:0, epoch:'E1'}, 0, false, true, 'E2', null, true, false],
  // —— 两头都得有值才比：老记录没盖过代号 → 退回老线索
  ['', {q:'问出去的问题', before:0}, 0, false, true, 'E2', '问出去的问题', false, false],
  // —— 这一页给不出代号（老服务端 / 会话确认不了的那一页）→ 退回老线索，不拿「空 vs 有」当证据
  ['', {q:'问出去的问题', before:0, epoch:'E1'}, 0, false, true, '', '问出去的问题', false, false],
];
let bad = 0;
for (const [echo, pending, count, wiped, bound, live, wantRestore, wantClear, wantAnswered] of cases) {
  const got = planPendingRestore(echo, pending, count, wiped, bound, live);
  if (got.restore !== wantRestore || got.clear !== wantClear || got.answered !== wantAnswered) {
    bad++; console.log('FAIL', JSON.stringify([echo, pending, count, wiped, bound, live]), JSON.stringify(got));
  }
}
console.log(bad === 0 ? 'TABLE_OK' : 'TABLE_BAD');
"""
        proc = self.run_in_node(script)
        self.assertIn("TABLE_OK", proc.stdout, proc.stdout + proc.stderr)

    # ---------- D. 真实链路：本机假上游（不联网、不用真密钥） ----------

    def test_a_fake_upstream_answer_still_goes_through(self):
        """先证明这套假上游能跑通正常路径，否则下面的失败场景可能只是假上游坏了。"""
        self.use_fake_upstream("ok", timeout=REAL_UPSTREAM_TIMEOUT_SECONDS)
        resp = self.ask(tutor.app.test_client(), "假上游正常问题")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(len(self.all_rows()), 2)

    def test_a_slow_but_alive_upstream_still_answers(self):
        self.use_fake_upstream("slow", timeout=REAL_UPSTREAM_TIMEOUT_SECONDS)
        self.assertEqual(self.ask(tutor.app.test_client(), "假上游慢问题").status_code, 302)

    def test_a_hung_upstream_becomes_a_retryable_503_page(self):
        """上游一直不吭声 → 客户端的 deadline 到点 → 一个可以重试的 503 页面。

        【这个超时值为什么不能小】见 HUNG_UPSTREAM_TIMEOUT_SECONDS 上面那段：
        子进程冷启动约 2 秒。给 2 秒的话，子进程常常还没连上就被杀掉，
        上游一次请求都收不到 —— 那时断言会红在「0 != 1」上，而真正的原因在别处。
        """
        upstream = self.use_fake_upstream("hang", timeout=HUNG_UPSTREAM_TIMEOUT_SECONDS)
        before = self.all_rows()
        resp = self.ask(tutor.app.test_client(), "会超时的问题")
        html = resp.get_data(as_text=True)

        # 【先确认这一次真的测到了「超时」】上游收到过请求，才谈得上「超时把它放弃」
        self.assertTrue(
            upstream.arrived.is_set(),
            "假上游一次请求都没收到：模型子进程还没连上就被 %d 秒的 deadline 杀掉了。"
            "这台机器上 worker 冷启动（Python + import openai）实测约 2 秒，"
            "deadline 必须明显大于它，否则测的是启动速度，不是超时处理。"
            % HUNG_UPSTREAM_TIMEOUT_SECONDS)
        self.assertEqual(upstream.requests, 1, "只该打上游一次")

        self.assertEqual(resp.status_code, 503)
        self.assertIn(tutor.MODEL_UNAVAILABLE_MESSAGE, html)      # 说清了「约半分钟后再试」
        self.assertIn('value="会超时的问题"', html)                # 文字没丢
        self.assertIn('data-model-budget-seconds="%d"'
                      % tutor.MODEL_REQUEST_TIMEOUT_SECONDS, html)
        self.assertEqual(self.all_rows(), before)                  # 历史没被写脏
        self.assertFalse(tutor._lock.locked())                     # 锁回来了
        self.assertGreater(tutor._model_retry_after, time.monotonic())   # 冷却开了
        # 出错页的横幅必须露出来（没有 hidden），文案就是服务端给的那一句
        self.assertIn('id="error-banner" role="alert">', html)

    def test_an_upstream_503_becomes_the_same_retryable_page(self):
        upstream = self.use_fake_upstream("status503", timeout=REAL_UPSTREAM_TIMEOUT_SECONDS)
        resp = self.ask(tutor.app.test_client(), "上游 503 的问题")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 503)
        self.assertIn(tutor.MODEL_UNAVAILABLE_MESSAGE, html)
        self.assertIn('value="上游 503 的问题"', html)
        self.assertEqual(upstream.requests, 1)

    def test_the_cooldown_blocks_a_second_question_without_touching_the_upstream(self):
        upstream = self.use_fake_upstream("status503", timeout=REAL_UPSTREAM_TIMEOUT_SECONDS)
        self.ask(tutor.app.test_client(), "第一次")
        before = upstream.requests
        second = self.ask(tutor.app.test_client(), "冷却期内的问题")
        self.assertEqual(second.status_code, 503)
        self.assertIn(tutor.MODEL_UNAVAILABLE_MESSAGE, second.get_data(as_text=True))
        self.assertEqual(upstream.requests, before, "冷却期内不该再打上游")
        self.assertEqual(self.used_today(), 1, "冷却期内不该再扣额度")

    def used_today(self):
        """今天已经扣过几次额度 —— 就是「到底问过模型几次」的权威口径。"""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT used FROM api_usage WHERE day = ?",
                               (tutor._utc_day(),)).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def test_a_double_submit_is_blocked_and_keeps_the_text(self):
        """第一台还卡在半路时，第二台再点发送：被锁挡下，而且【没有多花一次模型调用】。

        【为什么这里用 gate 而不是 hang】这一条本来靠「假上游装死 + 客户端超时」来
        制造并发，但那要拿秒数当节拍器：子进程冷启动约 2 秒，超时也是 2 秒，
        谁先谁后全看机器忙不忙 —— 测试会红，红得还不是被测的东西。
        改成 gate 之后：上游收下请求后【等测试放行】，第一个请求确定性地卡住，
        第二个请求一定撞上锁。不依赖任何秒数。
        """
        upstream = self.use_fake_upstream("gate", timeout=30)
        results = []
        first = tutor.app.test_client()
        thread = threading.Thread(target=lambda: results.append(
            self.ask(first, "第一台设备的问题")))
        thread.start()
        try:
            self.assertTrue(upstream.arrived.wait(15), "第一个请求没有到达上游")
            second = self.ask(tutor.app.test_client(), "第二台设备的问题")
            html = second.get_data(as_text=True)
            self.assertIn(tutor.LOCKED_MESSAGE, html)
            self.assertIn('value="第二台设备的问题"', html)       # 第二台也没丢字
        finally:
            upstream.release.set()                                # 放第一个请求走完
            thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].status_code, 302)             # 第一个正常答完了
        # 【用额度口径证明「只问过一次」】第二个请求在拿到锁之前就被挡下了，
        # 所以它连额度都不该占 —— 这才是「同一个问题没被问两遍」的权威证据
        # （比数假上游收到的请求数稳：worker 可能还没连上就被超时杀掉）。
        self.assertEqual(self.used_today(), 1, "重复点击不该多花一次模型调用")
        self.assertEqual(upstream.requests, 1)
        self.assertEqual(len(self.all_rows()), 2)                 # 历史里只有一问一答

    # ---------- E. 「结果未知」：响应丢了、用户又发一次，不能变成两次调用 ----------
    #
    # 【这一组要解决的场景】第一次已经答完、答案也写进库了，只是响应在回程丢了。
    # 这时服务端的锁【早就释放】了，第二次发送畅通无阻 —— 原来的设计挡不住它：
    # 多花一次额度，历史里多出一对一模一样的问答，用户自己还看不出来。
    # 现在靠 attempt 令牌挡住：同一个身份牌的第二次，不再调模型，直接把他送回有答案的那一页。
    #
    # 【这一组为什么用打桩的模型客户端，而不是本机假上游】
    # 要证明的是「模型被调用了几次」，而 FakeClient 正好把每一次 create() 都记在 calls 里 ——
    # 又准又稳。真子进程那一层由上面 D 组和下面浏览器那一组覆盖：
    # 那几条要证明的是「超时/上游报错长什么样」，本来就得走真链路。
    # （之前这里用假上游，会跟着「子进程冷启动快不快」一起抖，那不是被测的东西。）

    def test_a_resend_of_the_same_attempt_does_not_call_the_model_twice(self):
        client = tutor.app.test_client()
        first = self.ask_with_attempt(client, "会重复的问题", "attempt-aaaaaaaa")
        self.assertEqual(first.status_code, 302)
        self.assertEqual(self.used_today(), 1)

        again = self.ask_with_attempt(client, "会重复的问题", "attempt-aaaaaaaa")

        self.assertEqual(again.status_code, 302, "照常把他送回有答案的那一页")
        self.assertEqual(len(self.fake.calls), 1, "同一次尝试不该再问一次模型")
        self.assertEqual(self.used_today(), 1, "同一次尝试不该再扣一次额度")
        self.assertEqual(len(self.all_rows()), 2, "历史里不该多出一对一模一样的问答")

    def test_asking_the_same_text_twice_on_purpose_is_still_two_questions(self):
        """【连着两次问一模一样的话】是正常的用法，不能被令牌误伤：
        那是两次不同的尝试，页面会给新的身份牌。"""
        client = tutor.app.test_client()
        self.ask_with_attempt(client, "再讲一遍", "attempt-11111111")
        second = self.ask_with_attempt(client, "再讲一遍", "attempt-22222222")
        self.assertEqual(second.status_code, 302)
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual(self.used_today(), 2)
        self.assertEqual(len(self.all_rows()), 4)

    def test_an_attempt_that_failed_is_not_remembered(self):
        """失败的那次不能被记住 —— 否则用户就再也问不出这条问题了。

        只有「真的答完并写进库」才值得记。超时/上游报错都没产生回答，
        用户带着同一个身份牌再试一次，必须照常给他答案。
        """
        client = tutor.app.test_client()
        real_create = self.fake.chat.completions.create

        def failed(*args, **kwargs):
            raise RuntimeError("假的上游炸了")

        self.fake.chat.completions.create = failed
        first_try = self.ask_with_attempt(client, "第一次失败的问题", "attempt-bbbbbbbb")
        self.assertEqual(first_try.status_code, 503)
        self.assertEqual(self.all_rows(), [])
        self.assertEqual(self.fake.calls, [], "这一路根本没答出来")

        tutor._model_retry_after = 0.0        # 模拟冷却时间已经过去，无需真的等待
        self.fake.chat.completions.create = real_create
        retried = self.ask_with_attempt(client, "第一次失败的问题", "attempt-bbbbbbbb")

        self.assertEqual(retried.status_code, 302)
        self.assertEqual(len(self.all_rows()), 2)
        self.assertEqual(len(self.fake.calls), 1, "同一个身份牌重试，必须照常问一次模型")

    def test_a_token_from_another_session_does_not_silence_that_user(self):
        """身份牌只挡它自己那一次：另一个浏览器拿着同一个 id 来问，照常回答。"""
        self.ask_with_attempt(tutor.app.test_client(), "甲的问题", "attempt-shared00")
        other = self.ask_with_attempt(tutor.app.test_client(), "乙的问题", "attempt-shared00")
        self.assertEqual(other.status_code, 302)
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual(len(self.all_rows()), 4)

    def test_a_bogus_attempt_id_is_ignored_instead_of_breaking_the_page(self):
        """页面没跑脚本、或者有人乱塞一个超长字符串时：当作「没带」，一切照旧。"""
        for bogus in ("x" * 500, "带有中文和空格 的", "<script>", "  ", "", None):
            self.assertEqual(tutor._read_attempt_id(bogus), "")
        self.assertEqual(tutor._read_attempt_id("abc-123_XYZ"), "abc-123_XYZ")

        before = dict(tutor._completed_attempts)
        resp = self.ask_with_attempt(tutor.app.test_client(), "带着怪牌的问题", "x" * 500)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(len(self.fake.calls), 1, "照常回答，不受影响")
        self.assertEqual(tutor._completed_attempts, before, "怪值不该被塞进内存")

    def test_the_memory_of_finished_attempts_is_bounded(self):
        """这份记忆有上限、也会过期 —— 不能变成一个只涨不落的口袋。"""
        tutor._completed_attempts.clear()
        limit = tutor.COMPLETED_ATTEMPT_MAX
        for i in range(limit + 20):
            question = "问题%05d" % i
            message_id = tutor.save_exchange("s", question, "回答")
            tutor._remember_answered_attempt("s", "t%05d" % i, message_id, question)
        self.assertEqual(len(tutor._completed_attempts), limit, "超上限要先丢最旧的")
        self.assertFalse(tutor._attempt_already_answered("s", "t00000", "问题00000"))
        last = limit + 19
        self.assertTrue(tutor._attempt_already_answered(
            "s", "t%05d" % last, "问题%05d" % last))

        tutor._completed_attempts.clear()
        question = "过期测试的问题"
        message_id = tutor.save_exchange("s", question, "回答")
        tutor._remember_answered_attempt("s", "old", message_id, question)
        recorded = tutor._completed_attempts[("s", "old")]
        tutor._completed_attempts[("s", "old")] = (
            time.monotonic() - tutor.COMPLETED_ATTEMPT_TTL_SECONDS - 1,
            recorded[1], recorded[2])
        self.assertFalse(tutor._attempt_already_answered("s", "old", question), "过期的不算数")
        self.assertEqual(tutor._completed_attempts, {}, "过期的那条该被顺手清掉")


# =====================================================================
# 清空个人数据 × 尝试身份牌
#
# 【为什么单开一组】身份牌是内存里的一句断言：「(这个会话, 这个 id) 问过的问题已经答完了」。
# 这句话有两个前提，任何一个破掉都会【静默吃掉用户的问题】——不是报错，是假装答过了：
#
#   ① 那份回答【还在库里】。清空个人数据会把消息整行删掉，而删除流程碰不到内存里的记录。
#      删完之后用户再发一次同一条问题，就会被判成「答过了」，直接送回首页 ——
#      首页上什么都没有，问题就这么没了。（这个会话不会被轮换，所以键还对得上，
#      见 _ensure_session_id 与 profile_store.clear_all_data 的顺序。）
#   ② 是【同一条问题】。同一个身份牌配上另一条文字（页面出 bug、或有人手搓请求），
#      那是新问题，绝不能吞掉。
#
# 这两条现在都靠「记下用户消息的主键 + 问题摘要」来保证，见 app.py 的
# _attempt_already_answered。下面每一条测试都对着其中一个前提。
# =====================================================================

class TestDeletedDataAndAttemptToken(ChatTestCase):

    def bind(self, code="erase-code-1"):
        self.make_invite(code)
        client = tutor.app.test_client()
        self.enter_invite(client, code)
        return client

    def delete(self, client, confirm="yes", csrf=None):
        """提交「清空全部个人数据」表单（和 TestDeleteAllData 里那个同款）。"""
        if csrf is None:
            csrf = self.csrf_token(client, "/profile")
        data = {"csrf_token": csrf}
        if confirm is not None:
            data["confirm"] = confirm
        return client.post("/profile/delete", data=data)

    def ask_with_attempt(self, client, question, attempt):
        return client.post("/", data={"question": question, "attempt": attempt})

    @staticmethod
    def body_of(client):
        html = client.get("/").get_data(as_text=True)
        start = html.index("<body")
        return html[start:html.index(">", start)]

    # ---------- 前提①：回答已经被删掉，身份牌就该作废 ----------

    def test_a_token_from_before_the_deletion_cannot_swallow_a_new_question(self):
        """【核心】删完数据再用同一个身份牌问同一条问题：必须照常得到回答。

        这里刻意走完整条真实路径（真的提问、真的删除、再真的提问），
        因为要确认的正是「删除之后那个身份牌在内存里还对不对得上」。
        """
        client = self.bind()
        sid = self.session_id_of(client)
        self.assertEqual(
            self.ask_with_attempt(client, "删之前问过的问题", "attempt-del000001").status_code, 302)
        self.assertEqual(len(self.all_rows()), 2)
        self.assertTrue(tutor._attempt_already_answered(
            sid, "attempt-del000001", "删之前问过的问题"), "答完了就该认得出来")

        self.assertEqual(self.delete(client).status_code, 302)
        self.assertEqual(self.all_rows(), [], "数据该被清空了")
        self.assertIsNone(self.learner_id_of(client), "绑定该没了")

        # 【这一步是整条测试的要害】删除删的是绑定行，不是「作废标记」，
        # 所以这个浏览器的会话【不会】被轮换 —— 内存里那个 (会话, 身份牌) 键依然对得上。
        # 也就是说：能救场的只有「那条回答还在不在库里」这一查。
        self.assertEqual(self.session_id_of(client), sid, "删除不该换掉这个浏览器的会话")
        self.assertIn((sid, "attempt-del000001"), tutor._completed_attempts,
                      "身份牌还在内存里（删除流程不该去动它）")

        again = self.ask_with_attempt(client, "删之前问过的问题", "attempt-del000001")

        self.assertEqual(again.status_code, 302)
        self.assertEqual(len(self.fake.calls), 2, "回答已经不在了，这次必须真的再问一次模型")
        self.assertEqual(len(self.all_rows()), 2, "新的一问一答该写进去")
        self.assertIn("删之前问过的问题", [row[2] for row in self.all_rows()])

    def test_the_token_stops_counting_when_the_answer_row_is_gone(self):
        """直接盯住机制本身：那一行消息没了，身份牌立刻不算数。"""
        client = tutor.app.test_client()
        client.get("/")                      # 先走一次首页，这个浏览器才会有会话 ID
        sid = self.session_id_of(client)
        question_id = tutor.save_exchange(sid, "会被删掉的问题", "会被删掉的回答")
        tutor._remember_answered_attempt(sid, "attempt-gone00001", question_id,
                                         "会被删掉的问题")
        self.assertTrue(tutor._attempt_already_answered(
            sid, "attempt-gone00001", "会被删掉的问题"))

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        finally:
            conn.close()

        self.assertFalse(tutor._attempt_already_answered(
            sid, "attempt-gone00001", "会被删掉的问题"), "回答没了就不能再当成答过了")
        self.assertNotIn((sid, "attempt-gone00001"), tutor._completed_attempts,
                         "作废的那条该顺手清掉")

    def test_a_broken_check_never_swallows_the_question(self):
        """查库失败时按「没答过」处理 —— 宁可多问一次模型，也不能吞掉用户的问题。"""
        client = tutor.app.test_client()
        client.get("/")                      # 先走一次首页，这个浏览器才会有会话 ID
        sid = self.session_id_of(client)
        question_id = tutor.save_exchange(sid, "已有的问题", "已有的回答")
        tutor._remember_answered_attempt(sid, "attempt-boom00001", question_id, "已有的问题")

        real_connect = tutor._connect

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("内部细节-查不动了-7a1c")

        tutor._connect = boom
        try:
            self.assertFalse(tutor._attempt_already_answered(
                sid, "attempt-boom00001", "已有的问题"))
        finally:
            tutor._connect = real_connect

    # ---------- 前提②：同一个身份牌配上另一条问题 ----------

    def test_the_same_token_with_a_different_question_is_answered_not_swallowed(self):
        """【核心】换个问题就是新问题 —— 身份牌一样也不行。"""
        client = tutor.app.test_client()
        self.assertEqual(
            self.ask_with_attempt(client, "第一句", "attempt-mix000001").status_code, 302)

        second = self.ask_with_attempt(client, "第二句（完全不同的内容）", "attempt-mix000001")

        self.assertEqual(second.status_code, 302)
        self.assertEqual(len(self.fake.calls), 2, "另一条问题必须照常回答")
        self.assertEqual(len(self.all_rows()), 4)
        self.assertIn("第二句（完全不同的内容）", [row[2] for row in self.all_rows()])

    def test_the_same_question_with_extra_whitespace_is_still_the_same_attempt(self):
        """只有两端空白不同时，仍然是同一条问题（页面和服务端都是先 strip 再比）。"""
        client = tutor.app.test_client()
        self.assertEqual(
            self.ask_with_attempt(client, "  前后有空格的问题  ", "attempt-space0001").status_code,
            302)

        again = self.ask_with_attempt(client, "前后有空格的问题", "attempt-space0001")

        self.assertEqual(again.status_code, 302)
        self.assertEqual(len(self.fake.calls), 1, "两端空白不算另一条问题")
        self.assertEqual(len(self.all_rows()), 2)

    # ---------- 删除给页面的一次性信号 ----------

    def test_the_deletion_signal_is_sent_once_and_only_once(self):
        """删除成功后要告诉下一个页面「刚清空过」；而且只生效一次。"""
        client = self.bind()
        self.assertIn('data-just-deleted="false"', self.body_of(client), "平时不该有这个信号")

        self.assertEqual(self.delete(client).status_code, 302)

        self.assertIn('data-just-deleted="true"', self.body_of(client), "删除后那一次要给")
        self.assertIn('data-just-deleted="false"', self.body_of(client), "再刷一次就没了")

    def test_a_rejected_deletion_does_not_clear_the_pending_question(self):
        """删除没成功（没勾确认框）→ 不能发那个信号。

        发错了的后果是：用户那条「还没答完」的问题被白白丢掉，
        而他的数据根本还在。
        """
        client = self.bind()
        self.assertEqual(self.delete(client, confirm=None).status_code, 200)
        self.assertIn('data-just-deleted="false"', self.body_of(client))
        self.assertIsNotNone(self.learner_id_of(client), "数据不该被动过")

    def test_the_data_epoch_changes_only_when_the_data_is_really_erased(self):
        """「这份数据的代号」只在数据真的被删掉时换一个，换过之后就不许再变。

        【为什么这一位这么要紧】它是「暂存的那条问题属于哪一份数据」的凭据（见 index.html）：
          · 删除没成功却换了 → 用户那条还没答完的问题被白白作废；
          · 删成功了却不换，或者**重新绑一张码就换回去** → 那句已经被删掉的问题
            过一会儿又冒回输入框里，而它所指向的回答早就不存在了。
        上面那个「重新绑回去」不能只看一次性信号和绑定状态 —— 它俩恰恰在
        「别的标签页删了、重新绑上、旧标签页才刷新」这条路上都被绕开了。
        """
        client = self.bind()
        first = self.epoch_of(client)
        self.assertRegex(first, r"^[0-9a-f]{32}$", "代号该是 32 位十六进制")

        # ① 删除被挡下的那一次：一个字节都不许动
        self.assertEqual(self.delete(client, confirm=None).status_code, 200)
        self.assertIn('data-just-deleted="false"', self.body_of(client))
        self.assertEqual(self.epoch_of(client), first, "删除没成功，不该换代号")

        # ② 真的删掉了：必须换一个新的
        self.assertEqual(self.delete(client).status_code, 302)
        second = self.epoch_of(client)
        self.assertNotEqual(second, first, "数据被清空了，代号必须换")
        self.assertEqual(self.epoch_of(client), second, "换过之后要稳定下来，不能每次都换")

        # ③ 【关键的一条】重新拿一张新码绑回来：代号不许被换回去。
        #    换回去了，「这份数据被清空过」在旧标签页眼里就消失了。
        self.make_invite("rebind-code")
        self.enter_invite(client, "rebind-code")
        self.assertIsNotNone(self.learner_id_of(client), "该重新绑上了")
        self.assertEqual(self.epoch_of(client), second, "重新绑定不许把代号换回去")

    def epoch_of(self, client):
        """首页上那个「这份数据的代号」。"""
        found = re.search(r'data-data-epoch="([^"]*)"', self.body_of(client))
        self.assertIsNotNone(found, "首页没有下发数据代号")
        return found.group(1)

    # ---------- 响应乱序：迟到的 Set-Cookie 不许把代号盖回去 ----------

    def test_the_epoch_is_not_carried_in_the_session_cookie(self):
        """【结构性的一条】代号不在会话 cookie 里 —— 这是它不能被盖回去的原因。

        放进去就等于把「代号是不是旧的」交给浏览器手里那份 cookie 说了算，
        而 cookie 会被**迟到的响应**整个覆盖（见下一条）。所以这里直接盯住结构：
        签名过的会话里不许有它。
        """
        client = self.bind()
        self.assertRegex(self.epoch_of(client), r"^[0-9a-f]{32}$")

        with client.session_transaction() as sess:
            self.assertNotIn("data_epoch", sess,
                             "代号不该放在会话 cookie 里（会被迟到的响应盖回旧值）")

    def test_a_late_response_from_before_the_deletion_cannot_roll_the_epoch_back(self):
        """【响应乱序，用户点名的那条】删除之前就发出去、删除之后才回来的那个响应，
        不能把代号盖回旧值。

        【旧响应能改的只有它自己那份 cookie】所以这里就拿【删除之前的那份 cookie】
        当迟到的 Set-Cookie 用 —— 那正是它落地时会发生的事（这条测试里就是把它塞回浏览器）。

        路上每一步都是用户那条路：数据被清空（代号换掉）→ 拿新码重新绑上
        （「绑定没了」这条线索被补回来）→ 然后旧 cookie 才盖回来。
        这时候绑定是新的、历史是 0 条、一次性信号也用掉了 —— 要是代号被盖回去，
        四条线索就全都不作声，用户那句已经删掉的问题又会被当成「还没答完」放回输入框。
        """
        old = self.bind()
        self.ask(old, "删除之前问过的话")
        before = self.epoch_of(old)
        stale = old.get_cookie("session")                   # 删除之前的那份 cookie（旧响应手里那份）
        self.assertIsNotNone(stale)
        sid = self.session_id_of(old)

        self.assertEqual(self.delete(old).status_code, 302)  # 数据被清空
        rotated = self.epoch_of(old)
        self.assertNotEqual(rotated, before, "数据被清空了，代号必须换")

        self.make_invite("rebind-after-wipe")                # 用户拿新码重新绑上
        self.enter_invite(old, "rebind-after-wipe")
        self.assertIsNotNone(self.learner_id_of(old))
        self.assertEqual(self.epoch_of(old), rotated, "重新绑定不许把代号换回去")

        # ---- 迟到的那个响应到了：它带来的那份 cookie 被浏览器照单收下 ----
        late = tutor.app.test_client()
        late.set_cookie("session", stale.value)

        self.assertEqual(self.session_id_of(late), sid, "迟到的那份 cookie 还是同一个会话")
        self.assertNotIn("删除之前问过的话", self.body_of(late), "数据确实已经删掉了")
        self.assertEqual(self.epoch_of(late), rotated,
                         "迟到的响应把代号盖回旧值了 —— 那句已经删掉的问题会因此复活")

        # 【最后一句是给「下一次刷新」的】cookie 被盖过之后，再打开一次仍然必须是新代号：
        # 代号只由服务端按会话读出来，浏览器手里那份是什么都改不动它。
        self.assertEqual(self.epoch_of(late), rotated, "再刷新一次又回退了")


# =====================================================================
# 在【真实浏览器】里跑这个页面
#
# 【为什么要单开一组】上面那些断言都是「读页面文本」：文本对不代表脚本在浏览器里
# 真的跑得起来，更不代表定时器到点后界面真的能被收回来。这一组补的就是这一段。
#
# 【怎么做到不装任何东西】只借本机 Chrome 自己的两个开关：
#   · --dump-dom            跑完把最终 DOM 打到标准输出
#   · --virtual-time-budget 把计时器快进（本来要等四十几秒的看门狗，这里一秒跑完）
# 中间放一个转发代理：请求转给真的 Flask（真的会话 cookie、真的临时库），
# 顺手把页面里那两个 CDN 的 <script> 去掉 —— 测试不出网，也不受 CDN 抖动牵连。
# 需要时再往页面末尾追加一小段【测试自己的】脚本：替用户点一次发送、把输入框清空，
# 或者把当前 DOM 的状态塞进 <title>（DOM 里读的是属性值，改过 property 的要这样才读得到）。
# =====================================================================

BROWSER_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_browser():
    """找本机的 Chrome/Edge；找不到就返回 None（那一组测试会跳过）。"""
    for path in BROWSER_CANDIDATES:
        if os.path.exists(path):
            return path
    for name in ("chrome", "chromium", "google-chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


# ---------- 测试注入的几段脚本（属于测试，不属于产品） ----------

# 【几个探针共用的一小段】把页面当前状态读出来。
# DOM 转出来的 HTML 里看不到 property 的改动（比如脚本写进去的 input.value），
# 所以只能这样读，再塞进 <title> 交回给测试。
# 抽出来放一处，是因为会话存储那个键名在几个探针里都要写 —— 两边写岔了不会报错，
# 只会让某一条测试悄悄不再检查任何东西。
BROWSER_SNAPSHOT_FN = r"""
function snapshot() {
  var q = document.getElementById('question-input');
  var b = document.getElementById('error-banner');
  var a = document.getElementById('error-action');
  var t = document.getElementById('error-text');
  var btn = document.getElementById('send-button');
  var pending = null;
  try { pending = window.sessionStorage.getItem('ai-tutor-pending-question-v2'); }
  catch (e) { pending = 'ERROR'; }
  return {
    value: q ? q.value : null, readOnly: q ? q.readOnly : null,
    button: btn ? btn.textContent : null,
    bannerHidden: b ? b.hidden : null, banner: t ? t.textContent : '',
    actionHidden: a ? a.hidden : null, pending: pending
  };
}
"""

BROWSER_PROBE_JS = r"""
(function () {
  __SNAPSHOT__
  var token = '';
  try { token = newAttemptId(); } catch (e) { token = 'ERR'; }
  function stage() { try { return STAGE_EXPR; } catch (e) { return 'probe'; } }
  function full() {
    var out = snapshot();
    out.token = token;
    out.stage = stage();
    // 页面那段脚本到底有没有跑起来：这几个函数是它声明的，读得到就说明整段都活着
    out.logic = (typeof planPendingRestore === 'function') && (typeof planAttempt === 'function')
                && (typeof newAttemptId === 'function') && (typeof releaseComposer === 'function');
    return out;
  }
  function report() {
    // 两条路都发：一条塞进 <title>（按虚拟时间跑、最后 dump DOM 那种用得上），
    // 一条用图片探针回传给测试代理（按真实时间跑、要等某个阶段完成那种用得上）。
    document.title = 'PROBE' + JSON.stringify(full());
    try {
      var img = new Image();
      img.src = '/__report?' + encodeURIComponent(JSON.stringify(full()));
    } catch (e) { /* 探针发不出去不该影响页面 */ }
  }
  report();
  setTimeout(report, REPORT_AFTER_MS);
})();
"""

# 替用户点一次「发送」。只点一次：点完在 sessionStorage 里留个记号，
# 免得重定向回来之后又被点一次（那样会变成没完没了地提问）。
BROWSER_SUBMIT_ONCE_JS = r"""
(function () {
  var KEY = '__ai_tutor_test_submitted';
  try {
    if (window.sessionStorage.getItem(KEY)) return;
    window.sessionStorage.setItem(KEY, '1');
  } catch (e) { return; }
  setTimeout(function () { document.getElementById('ask-form').requestSubmit(); }, 0);
})();
"""

# 【把导航按住】只让页面自己的 submit 处理器跑完，请求一个字节都不发出去。
# 对页面来说，这和「请求发出去了、但永远回不来」是同一件事：界面停在等待状态，
# 唯一能结束它的是看门狗。
BROWSER_BLOCK_SUBMIT_JS = r"""
document.getElementById('ask-form').addEventListener('submit', function (event) {
  event.preventDefault();
});
document.getElementById('question-input').value = '被卡住的问题';
setTimeout(function () { document.getElementById('ask-form').requestSubmit(); }, 0);
"""

# 替用户把输入框清干净（连同浏览器会发的那一个 input 事件），然后跳到测试自己的
# /__probe 页 —— 那一页会把 sessionStorage 里的东西印在标题上。
BROWSER_CLEAR_THEN_CHECK_JS = r"""
(function () {
  var q = document.getElementById('question-input');
  q.value = '';
  q.dispatchEvent(new Event('input', { bubbles: true }));
  window.location.replace('/__probe');
})();
"""

# 用户改了草稿（但没发出去），然后刷新页面。只做一次，免得刷新回来又刷新。
BROWSER_EDIT_DRAFT_THEN_RELOAD_JS = r"""
(function () {
  var KEY = '__ai_tutor_test_edited';
  try {
    if (window.sessionStorage.getItem(KEY)) return;
    window.sessionStorage.setItem(KEY, '1');
  } catch (e) { return; }
  var q = document.getElementById('question-input');
  q.value = '我改了一半的新问题';
  q.dispatchEvent(new Event('input', { bubbles: true }));
  window.location.reload();
})();
"""

# 【真刀真枪那一版】真发一次请求（它会卡在服务端），然后盯着横幅：
# 一看门狗到点把横幅露出来，就立刻把这个状态回传给测试。
# 用 <img> 而不是 fetch：这是测试自己的探针，页面那边一个字的网络代码都没有。
BROWSER_LIVE_PROBE_JS = r"""
(function () {
  var KEY = '__ai_tutor_test_submitted';
  var sent = false;
  __SNAPSHOT__
  function send() {
    if (sent) return;
    sent = true;
    var img = new Image();
    img.src = '/__report?' + encodeURIComponent(JSON.stringify(snapshot()));
  }
  try {
    if (!window.sessionStorage.getItem(KEY)) {
      window.sessionStorage.setItem(KEY, '1');
      document.getElementById('question-input').value = '卡在路上的问题';
      setTimeout(function () { document.getElementById('ask-form').requestSubmit(); }, 0);
    }
  } catch (e) { /* 无痕模式之类，忽略 */ }
  setInterval(function () {
    var banner = document.getElementById('error-banner');
    if (banner && !banner.hidden) send();
  }, 200);
})();
"""

# 【用户原话里那个场景，端到端跑一遍】
#   第一次发送 → 服务端答完了、写进库了，但响应被代理吃掉（对他就是「响应丢了」）
#   → 看门狗到点 → 他再按一次发送 → 页面沿用同一个身份牌 → 服务端认出是同一次尝试。
# 注入的脚本负责这两次「按发送」和把结果回传（页面那边一个字的网络代码都没有）。
BROWSER_LIVE_RESEND_JS = r"""
(function () {
  var KEY = '__ai_tutor_test_resend';
  var resending = false;
  __SNAPSHOT__
  function report(stage) {
    var img = new Image();
    img.src = '/__report?' + encodeURIComponent(JSON.stringify(
      Object.assign(snapshot(), { stage: stage })));
  }
  try {
    if (!window.sessionStorage.getItem(KEY)) {
      window.sessionStorage.setItem(KEY, '1');
      document.getElementById('question-input').value = '会重复的问题';
      setTimeout(function () { document.getElementById('ask-form').requestSubmit(); }, 0);
    }
  } catch (e) { /* 忽略 */ }
  var timer = setInterval(function () {
    var banner = document.getElementById('error-banner');
    if (!banner || banner.hidden || resending) return;
    resending = true;
    clearInterval(timer);
    report('stalled');                       // 看门狗到点了：结果未知
    setTimeout(function () {
      document.getElementById('ask-form').requestSubmit();   // 用户又按了一次发送
      setTimeout(function () { report('resent'); }, 2500);
    }, 200);
  }, 200);
})();
"""

# 【真的删一遍】注入的脚本按顺序做四件事，每一步都是真表单、真跳转（不是 fetch）：
#   ① 首页 → 邀请码页；② 填码并提交（绑定）；③ 回到首页 → 档案页；
#   ④ 在档案页勾上复选框、提交「清空全部个人数据」。
# （③ 之所以要绕一下：绑定成功后应用按 PRG 把浏览器送回首页，不是档案页。）
# 删除成功后应用会重定向回首页，而首页这一次带着「刚清空过」的一次性信号 ——
# 那条待确认的问题必须就此消失，不能再冒回输入框里。
BROWSER_DELETE_FLOW_JS = r"""
(function () {
  var STEP_KEY = '__ai_tutor_test_step';
  var CODE = '__INVITE_CODE__';
  function step() {
    try { return window.sessionStorage.getItem(STEP_KEY) || 'start'; }
    catch (e) { return 'start'; }
  }
  function setStep(value) {
    try { window.sessionStorage.setItem(STEP_KEY, value); } catch (e) {}
  }
  function wipeForm() {
    return document.querySelector('form[action="/profile/delete"]');
  }
  function submitWipe() {
    var wipe = wipeForm();
    var confirmBox = wipe && wipe.querySelector('input[name="confirm"]');
    if (!wipe || !confirmBox) return false;
    confirmBox.checked = true;            // 没勾上的复选框根本不会被提交
    setStep('done');
    wipe.submit();
    return true;
  }
  var current = step();
  if (current === 'start') {
    setStep('invite');
    window.location.replace('/invite');
  } else if (current === 'invite') {
    var invite = document.querySelector('form[action="/invite"]');
    var field = invite && invite.querySelector('input[name="invite_code"]');
    if (!invite || !field) return;
    field.value = CODE;
    setStep('goto-profile');
    invite.submit();
  } else if (current === 'goto-profile') {
    // 绑好之后落在首页：这里没得删，去档案页
    if (!submitWipe()) {
      setStep('profile');
      window.location.replace('/profile');
    }
  } else if (current === 'profile') {
    submitWipe();
  }
})();
"""

# 【用户点名的那条路，端到端跑一遍】顺序一步不少：
#   ① 旧标签页真的发一条问题出去（响应被代理吃掉 —— 对他就是「结果未知」）。
#      这一发，页面自己就把那条「待确认的问题」写进 sessionStorage，带着它当时看到的数据代号；
#   ② 另一个标签页去：输码绑定 → 档案页 → 清空全部个人数据 → 再拿【第二张】码重新绑回来。
#      删数据那一步会把数据代号换掉，同时把「刚清空过」那个一次性信号用掉（跳回首页的那一页读走了它）；
#   ③ 旧标签页这时才刷新。
#
# 【为什么非要开第二个标签页】cookie 是两个标签页共用一份（数据代号就在 cookie 里），
# sessionStorage 却各是各的（那条待确认的问题在 sessionStorage 里）。只有真的开两个，
# 「一个标签页删、另一个标签页过一会儿才刷新」才成立 —— 那也正是这条路走不通的地方。
# 【两个标签页怎么对齐】localStorage 同源共用：另一个标签页走完写一个记号，旧标签页等它。
# 这是【测试自己的约定】，产品代码里一个字节都没有。
BROWSER_CROSS_TAB_DELETE_JS = r"""
(function () {
  var ROLE = '__ai_tutor_test_role';     // 这个【标签页】扮谁（sessionStorage：各是各的）
  var HANDOFF = '__ai_tutor_test_handoff';       // 「下一个新开的标签页去删数据」（localStorage：共用）
  var PHASE = '__ai_tutor_test_phase';   // 旧标签页走到哪一步了（它自己的标签页里）
  var FLOW = '__ai_tutor_test_flow';     // 另一个标签页走到哪一步了（它自己的标签页里）
  var DONE = '__ai_tutor_test_done';     // 另一个标签页：整个流程走完了（这个要共用）
  var FIRST = '__INVITE_FIRST__';
  var SECOND = '__INVITE_SECOND__';
  __SNAPSHOT__

  // 【角色必须按标签页分】可是 window.open 开出来的标签页并不干净：
  // 浏览器会把【开它的那个标签页】的 sessionStorage 复制一份过去（同源下如此），
  // 所以「sessionStorage 里没有角色 = 我是新来的」这个判断不成立 —— 复制过来的那份里就有。
  // 于是改成交接：旧标签页开新标签页之前，先往共享的 localStorage 里放一个记号；
  // 谁看到这个记号谁就是「去删数据的那一个」，看到就用掉（只够一个标签页用）。
  function role() {
    try {
      if (window.localStorage.getItem(HANDOFF) === '1') {
        window.localStorage.removeItem(HANDOFF);
        window.sessionStorage.setItem(ROLE, 'flow-tab');
        return 'flow-tab';
      }
      var mine = window.sessionStorage.getItem(ROLE);
      if (!mine) { mine = 'old-tab'; window.sessionStorage.setItem(ROLE, mine); }
      return mine;
    } catch (e) { return 'old-tab'; }
  }
  function store(key, value) {
    try {
      if (value === undefined) return window.localStorage.getItem(key);
      window.localStorage.setItem(key, value);
      return value;
    } catch (e) { return null; }
  }
  function mine(key, value) {
    try {
      if (value === undefined) return window.sessionStorage.getItem(key);
      window.sessionStorage.setItem(key, value);
      return value;
    } catch (e) { return null; }
  }
  function facts(stage) {
    var out = snapshot();
    out.stage = stage;
    out.epoch = document.body.dataset ? (document.body.dataset.dataEpoch || null) : null;
    out.bound = document.body.dataset ? (document.body.dataset.learnerBound || null) : null;
    out.count = document.body.dataset ? (document.body.dataset.userCount || null) : null;
    return out;
  }
  function report(stage) {
    document.title = 'PROBE' + JSON.stringify(facts(stage));
    try {
      var img = new Image();
      img.src = '/__report?' + encodeURIComponent(JSON.stringify(facts(stage)));
    } catch (e) { /* 探针发不出去不该影响这一页 */ }
  }
  function formFor(action) { return document.querySelector('form[action="' + action + '"]'); }

  // 【每一步都别悄悄卡住】找不到该有的东西就回传一个 'stuck'（带上停在哪一步），
  // 测试那边一眼看得出是流程断了，而不是「断言莫名失败」。
  function stuck(where) { report('stuck:' + where); }

  if (role() === 'old-tab') {
    // ---------- 旧标签页 ----------
    var phase = mine(PHASE);
    if (phase === 'reloading') {
      // 刷新之后：这里就是那条路的终点 —— 输入框里该是什么都没有
      report('after');
      return;
    }
    if (phase === null) {
      mine(PHASE, 'sent');
      document.getElementById('question-input').value = '删掉之前问出去的问题';
      setTimeout(function () {
        document.getElementById('ask-form').requestSubmit();   // 页面在这里记下那条待确认的问题
        report('sent');
        // 等那条记录落进 sessionStorage，再开另一个标签页去删数据。
        // 开之前先把「下一个新标签页扮什么」放进共享的 localStorage（见上面 role 的说明）。
        setTimeout(function () {
          store(HANDOFF, '1');
          window.open('/invite', '_blank');
        }, 1500);
      }, 0);
    }
    // 刚发出去、或者已经在等：都盯着另一个标签页，看它有没有把流程走完。
    // 【就在这一页里等，不刷新】—— 要的就是「旧标签页一直停在原地」这个状态。
    var waited = 0;
    var timer = setInterval(function () {
      waited += 200;
      if (store(DONE) === '1') {
        clearInterval(timer);
        mine(PHASE, 'reloading');
        window.location.replace('/');          // ← 这就是「旧标签页刷新」
      } else if (waited >= 45000) {
        clearInterval(timer);
        report('timeout');
      }
    }, 200);
    report('waiting');
    return;
  }

  // ---------- 另一个标签页：绑定 → 删数据 → 重新绑定 ----------
  var flow = mine(FLOW);
  if (flow === null) { flow = 'invite-first'; mine(FLOW, flow); }
  if (flow === 'invite-first') {
    var first = formFor('/invite');
    var field = first && first.querySelector('input[name="invite_code"]');
    if (!field) { stuck('invite-first'); return; }
    field.value = FIRST;
    mine(FLOW, 'to-profile');
    first.submit();
  } else if (flow === 'to-profile') {
    mine(FLOW, 'wipe');
    window.location.replace('/profile');
  } else if (flow === 'wipe') {
    var wipe = formFor('/profile/delete');
    var box = wipe && wipe.querySelector('input[name="confirm"]');
    if (!wipe || !box) { stuck('wipe'); return; }
    box.checked = true;                    // 没勾上的复选框根本不会被提交
    mine(FLOW, 'to-rebind');
    wipe.submit();                         // 删完跳回首页：那个一次性信号在【这一页】被用掉
  } else if (flow === 'to-rebind') {
    report('wiped');                       // 数据没了，代号也换了（这一页带着「刚清空过」的信号）
    mine(FLOW, 'invite-second');
    window.location.replace('/invite');
  } else if (flow === 'invite-second') {
    var second = formFor('/invite');
    var field2 = second && second.querySelector('input[name="invite_code"]');
    if (!field2) { stuck('invite-second'); return; }
    field2.value = SECOND;
    mine(FLOW, 'finish');
    second.submit();
  } else if (flow === 'finish') {
    report('rebound');                     // 重新绑上了：绑定这条线索又被补了回来
    store(DONE, '1');                      // 告诉旧标签页：可以刷新了
  }
})();
"""

# 预置 sessionStorage 之后立刻跳回首页：用来模拟「刷新时手上还留着一条没确认的问题」。
BROWSER_SEED_PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>SEED</title></head><body><script>
try { window.sessionStorage.setItem('ai-tutor-pending-question-v2', __PAYLOAD__); }
catch (e) { document.title = 'SEED-ERROR'; }
window.location.replace('/');
</script></body></html>"""

# 把 sessionStorage 里的东西印在标题上，供测试直接读。
BROWSER_STORAGE_PROBE_PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>STORAGE</title></head><body><script>
var out;
try { out = window.sessionStorage.getItem('ai-tutor-pending-question-v2'); }
catch (e) { out = 'ERROR'; }
document.title = 'STORAGE:' + (out === null ? 'NONE' : out);
</script></body></html>"""

_ONE_PIXEL_GIF = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!"
                  b"\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00"
                  b"\x00\x02\x02D\x01\x00;")


class BrowserHarness:
    """「用真的浏览器打开这一页」这件事，收在这一个小工具里。

    它起两个本机服务：真的 Flask（werkzeug 的多线程服务器，真的 socket）
    和前面那个转发代理。代理同时负责计数：浏览器到底发过几个 POST，一目了然。
    """

    def __init__(self, testcase, inject="", seed=None, swallow_posts=False, stage="'probe'"):
        self.browser = find_browser()
        # 几个探针都留了一个 __SNAPSHOT__ 的位置，在这儿统一填上共用那一段，
        # 免得每个调用点都要记着替换。stage 是探针回传时带上的「现在到哪一步了」。
        self.inject = (inject.replace("__SNAPSHOT__", BROWSER_SNAPSHOT_FN)
                             .replace("STAGE_EXPR", stage))
        self.seed = seed
        # 【把 POST 的响应吃掉】服务端照常处理、照常写库，只是那份响应不回给浏览器。
        # 对用户来说，这就是「我发出去了，但再也等不到回应」。
        #   True  = 每一个 POST 都吃掉；
        #   整数 N = 只吃前 N 个 —— 一个页面里要有「一次丢掉响应的提问」+「几次正常的表单提交」
        #            时用得上（比如 cross-tab 那条：旧标签页那一次提问要丢，
        #            另一个标签页的输码 / 删除不能丢）。判定看的是【顺序】，不是秒数。
        self.swallow_posts = swallow_posts
        self.swallowed = 0               # 已经吃掉几个了（只在 _lock 里读写）
        self.posts = []                  # 浏览器真的发出去的 POST 路径
        self.post_bodies = []            # POST 的表单内容（用来核对身份牌是不是同一个）
        self.hits = []                   # 所有请求
        self.reports = []                # 注入的探针回传的结果
        self.proc = None
        self.profiles = []
        self._lock = threading.Lock()

        from werkzeug.serving import make_server      # Flask 自带的服务器，不用额外装
        self.app_server = make_server("127.0.0.1", 0, tutor.app, threaded=True)
        # 收工时会 kill 浏览器，正在处理的请求会撞上一个「连接被重置」。
        # 那是收尾的正常现象，不是测试失败 —— 别把栈打到测试输出里去。
        self.app_server.handle_error = lambda *args, **kwargs: None
        self.app_server.log = lambda *args, **kwargs: None
        threading.Thread(target=self.app_server.serve_forever, daemon=True).start()
        self.app_port = self.app_server.server_address[1]

        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                outer._handle(self, "GET")

            def do_POST(self):
                outer._handle(self, "POST")

            def log_message(self, *args):
                pass                     # 别把代理的请求日志刷到测试输出里

        class QuietServer(http.server.ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                pass                     # 同上：收尾时被重置的连接不必打印

        self.proxy = QuietServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()
        self.port = self.proxy.server_address[1]
        testcase.addCleanup(self.stop)

    @property
    def origin(self):
        return "http://127.0.0.1:%d" % self.port

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:
                pass
            self.proc = None
        for server in (self.proxy, self.app_server):
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        for profile in self.profiles:
            shutil.rmtree(profile, ignore_errors=True)

    # ---------- 代理 ----------

    def _send(self, handler, status, content_type, body):
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _handle(self, handler, method):
        path = handler.path
        with self._lock:
            self.hits.append((method, path))
            if method == "POST":
                self.posts.append(urllib.parse.urlsplit(path).path)

        if path.startswith("/__report"):
            self._record_report(path)
            self._send(handler, 200, "image/gif", _ONE_PIXEL_GIF)
            return
        if path.startswith("/__probe"):
            self._send(handler, 200, "text/html; charset=utf-8",
                       BROWSER_STORAGE_PROBE_PAGE.encode("utf-8"))
            return
        if path.startswith("/__seed"):
            payload = json.dumps(json.dumps(self.seed or {}, ensure_ascii=False),
                                 ensure_ascii=False)
            page = BROWSER_SEED_PAGE.replace("__PAYLOAD__", payload)
            self._send(handler, 200, "text/html; charset=utf-8", page.encode("utf-8"))
            return

        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else None
        if method == "POST":
            with self._lock:
                self.post_bodies.append((body or b"").decode("utf-8", "replace"))
        status, headers, payload = self._forward(handler, method, path, body)
        if method == "POST" and self._swallow_this_one():
            return                       # 响应留在代理这里：浏览器那边就是「永远等不到」
        ctype = ""
        keep = []
        for name, value in headers:
            low = name.lower()
            if low == "content-type":
                ctype = value
            if low in ("content-length", "connection", "transfer-encoding",
                       "content-encoding", "date", "server"):
                continue
            keep.append((name, value))
        if "text/html" in ctype:
            payload = self._rewrite(payload.decode("utf-8")).encode("utf-8")
            ctype = "text/html; charset=utf-8"
        handler.send_response(status)
        handler.send_header("Content-Type", ctype or "application/octet-stream")
        for name, value in keep:
            if name.lower() != "content-type":
                handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _swallow_this_one(self):
        """这一个 POST 的响应要不要吃掉。True 全吃；N 只吃前 N 个。"""
        if self.swallow_posts is True:
            return True
        if not isinstance(self.swallow_posts, int) or isinstance(self.swallow_posts, bool):
            return False
        with self._lock:
            if self.swallowed >= self.swallow_posts:
                return False
            self.swallowed += 1
            return True

    def _forward(self, handler, method, path, body):
        """把请求原样转给真的 Flask —— cookie 要带上，不然会话就不是同一个了。"""
        headers = {}
        for name, value in handler.headers.items():
            if name.lower() in ("host", "connection", "content-length", "accept-encoding"):
                continue
            headers[name] = value
        conn = http.client.HTTPConnection("127.0.0.1", self.app_port, timeout=120)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            conn.close()

    def _rewrite(self, html):
        # 去掉两个 CDN 的脚本：测试不出网。页面本来就写了「取不到就跳过」。
        html = re.sub(r'<script src="https://cdn\.jsdelivr\.net/[^"]*"></script>\s*', "", html)
        if self.inject:
            html = html.replace("</body>", "<script>" + self.inject + "</script></body>")
        return html

    def _record_report(self, path):
        query = urllib.parse.urlsplit(path).query
        try:
            data = json.loads(urllib.parse.unquote(query))
        except Exception:
            data = {"unparsed": urllib.parse.unquote(query)}
        with self._lock:
            self.reports.append(data)

    # ---------- 开浏览器 ----------

    def _chrome_args(self, url, extra=()):
        profile = tempfile.mkdtemp(prefix="ai_tutor_chrome_")
        self.profiles.append(profile)
        return ([self.browser, "--headless=new", "--disable-gpu", "--no-first-run",
                 "--no-default-browser-check", "--disable-extensions",
                 # cross-tab 那条要用 window.open 开第二个标签页：脚本不是用户点的，
                 # 拦不拦是浏览器说了算，所以这里明确关掉。
                 "--disable-popup-blocking",
                 "--user-data-dir=" + profile] + list(extra) + [url])

    def open_virtual(self, path="/", budget_ms=8000, timeout=90, report_after_ms=0):
        """打开一个页面，让计时器按【虚拟时间】快进，跑完把最终 DOM 交回来。"""
        self.inject = self.inject.replace("REPORT_AFTER_MS", str(report_after_ms))
        args = self._chrome_args(self.origin + path,
                                 ["--virtual-time-budget=%d" % budget_ms, "--dump-dom"])
        proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
        return proc.stdout or ""

    def open_live(self, path="/", wait_seconds=30, until_stage=None):
        """打开页面，用【真实时间】跑：等注入的探针回传结果再回来。

        【为什么不用虚拟时间】实测 headless Chrome 在「有请求挂着没回来」时会把
        虚拟时钟按住（一直挂到真实超时），所以「请求真的卡住」这一种只能用真实时间。

        until_stage：等哪一步的回传才算数（探针会分几步回传）。
        """
        self.inject = self.inject.replace("REPORT_AFTER_MS", "0")
        self.proc = subprocess.Popen(self._chrome_args(self.origin + path),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            with self._lock:
                if self.reports and (until_stage is None or any(
                        report.get("stage") == until_stage for report in self.reports)):
                    break
            time.sleep(0.1)
        with self._lock:
            return list(self.reports), list(self.posts)

    def attempt_tokens(self):
        """从浏览器真的发出去的 POST 里，把身份牌抠出来。"""
        tokens = []
        for body in list(self.post_bodies):
            fields = urllib.parse.parse_qs(body)
            tokens.append((fields.get("attempt") or [""])[0])
        return tokens


class TestWaitingStateInRealBrowser(ChatTestCase):
    """真实浏览器里的等待 / 重发行为（本机没有 Chrome/Edge 就整组跳过）。

    【这一组能证明什么】
      · 页面那段内联脚本在真浏览器里真的跑起来了（不只是「文本看着对」）；
      · 刷新/重新打开时，那条还没确认的问题会不会回到输入框 —— 真的 sessionStorage、
        真的页面跳转、真的服务端历史；
      · 用户把输入框清空 / 改了草稿之后刷新，分别会发生什么；
      · 真的把个人数据删掉一遍（真邀请码表单 + 真档案页 + 真删除表单）之后，
        那条待确认的问题不会又冒回输入框（test_a_pending_question_...）；
      · 在别的标签页里删掉时也认得出 —— 一次性信号被那个标签页用掉了，
        还有「历史变短了」和「绑定没了」两条耐久的线索（..._does_not_survive_... 两条）；
      · 把那条路整条走完（..._does_not_revive_across_a_delete_and_a_rebind）：真的开两个标签页 ——
        一个发问、另一个删数据（信号被它用掉）**再重新绑一张码**，然后第一个标签页才刷新。
        老线索在这条路上被逐个绕开，认得出的是「数据代号对不上」；
      · 看门狗到点后：界面被收回、说清「结果未知」、给出手动刷新入口，
        而且浏览器【一个多余的请求都没发】；
      · 用户原话里那个场景端到端跑一遍（test_a_lost_response_resend_...）：
        真浏览器、真表单、真服务端、真模型客户端。第一次的响应被代理吃掉之后，
        用户再按一次发送 —— 模型只被调用了一次，两次 POST 带的是同一个身份牌。

    【这一组证明不了什么（验证边界）】
      · 「请求真的发出去、卡在网络上」这一种在 --virtual-time-budget 下跑不起来：
        只要有请求挂着没回来，Chrome 就会暂停虚拟时钟（实测会一直挂到真实超时）。
        所以那条按虚拟时间跑的看门狗用例（test_the_watchdog_releases_...）是用
        「把表单导航按住、一个字节都不发」来模拟的 —— 对页面来说同样是永远等不到回应，
        但不等于「服务端正在处理」。真正在跑的那两条（..._is_really_still_out_there /
        ..._a_lost_response_resend_...）用的是真实时间、真请求，不存在这个折扣。
      · 删除那一条走的是页面自己那两个表单，但「点」是注入脚本代点的：不是真鼠标。
      · 多标签页：只有 ..._does_not_revive_across_a_delete_and_a_rebind 那一条真的开了两个标签页，
        走的也是「一个标签页在删、另一个在等」这种【先后关系】；两个标签页同时提问、
        真的互相抢那把锁，这里没有覆盖（两个进程共用一把锁那部分在 Python 那一组）。
        同一个会话里「另一个标签页先答完了」会怎样，只由判定表里「条数变多 → 认定答过了」那一条覆盖。
      · 那条两标签页的用例里，「第二个标签页」是注入脚本用 window.open 开的（不是真鼠标点了链接），
        两个标签页之间靠 localStorage 对齐 —— 那是测试自己的约定，产品代码里没有这种代码。
      · 真机上人手点击、切后台、断网重连这些浏览器特性没有覆盖。
      · 浏览器用的是本机的 Chrome/Edge；没装（或装在别处）就整组跳过。
    """

    # 探针回传时用的「阶段」表达式（见各条用例）。
    # 【一律读服务端渲染出来的事实，不用客户端自己设的标记】踩过一次：
    # 注入脚本是在【同一个页面】上先把标记改成「完成了」再去提交的，
    # 于是探针立刻回传「完成」，测试在流程真的跑完之前就断言了。
    STAGE_ANSWERED = "document.body.dataset.userCount === '1' ? 'after' : 'before'"
    STAGE_WIPED = "document.body.dataset.justDeleted === 'true' ? 'done' : 'before'"

    def browser(self, inject="", seed=None, swallow_posts=False, stage="'probe'"):
        if find_browser() is None:
            self.skipTest("本机没有 Chrome/Edge，跳过真实浏览器测试")
        return BrowserHarness(self, inject=inject, seed=seed,
                              swallow_posts=swallow_posts, stage=stage)

    @staticmethod
    def probe(dom):
        """把注入的探针写进 <title> 的那段 JSON 读回来。"""
        found = re.search(r"<title>PROBE(\{.*?\})</title>", dom, re.S)
        if found is None:
            raise AssertionError("页面里没有探针结果，DOM 开头是：\n" + dom[:400])
        return json.loads(html_module.unescape(found.group(1)))

    # ---------- A. 页面脚本在真浏览器里是活的 ----------

    def test_the_page_script_runs_in_a_real_browser(self):
        harness = self.browser(inject=BROWSER_PROBE_JS)
        state = self.probe(harness.open_virtual("/", budget_ms=4000))

        self.assertTrue(state["logic"], "页面里的判断函数没找着：那段脚本没跑起来")
        self.assertRegex(state["token"], r"^[0-9a-f]{32}$", "身份牌该是 32 位十六进制")
        self.assertEqual(state["value"], "")
        self.assertFalse(state["readOnly"])
        self.assertIn("发送", state["button"])
        self.assertTrue(state["bannerHidden"], "没出错时横幅该是藏着的")
        self.assertIsNone(state["pending"], "什么都没发过，不该有待确认的记录")
        self.assertEqual(harness.posts, [])
        self.assertTrue(all(method == "GET" for method, _ in harness.hits),
                        "只是打开一次页面，不该发出别的请求：%r" % (harness.hits,))

    # ---------- B. 刷新之后问题文字还在不在（真的 sessionStorage、真的跳转） ----------

    def test_a_staged_question_comes_back_after_a_real_reload(self):
        """还没被服务端记下来的那条：刷新后回到输入框，并说清「结果未知」。"""
        harness = self.browser(inject=BROWSER_PROBE_JS,
                               seed={"q": "还没答完的问题", "token": "seedtoken0000001",
                                     "before": 0, "stalled": True})
        state = self.probe(harness.open_virtual("/__seed", budget_ms=6000))

        self.assertEqual(state["value"], "还没答完的问题")
        self.assertFalse(state["bannerHidden"])
        self.assertIn("没有】记录", state["banner"])
        self.assertIn("不会自动重发", state["banner"])
        # 这一页就是刚刷新出来的，服务端的话已经是最新的 —— 不必再劝他刷新一次
        self.assertTrue(state["actionHidden"])
        # 【身份牌必须留着】这条重发还是同一次尝试，服务端才挡得住第二次调用
        self.assertIn("seedtoken0000001", state["pending"])

    def test_a_question_that_was_really_answered_is_not_pushed_back(self):
        """真的提交一次、真的让假上游回答、真的写进临时库，再跟着 302 回到首页。

        对页面来说：上次那条在服务端【有】记录了 —— 就不该再塞回输入框，
        也不该劝用户再发一次（那才是重复调用）。

        【为什么这一条按真实时间跑】它中间有一次真的请求在飞。
        headless Chrome 按虚拟时间跑的时候，对「有请求挂着没回来」的处理很别扭
        （见类注释里的边界说明），所以这里改成：让它真跑，等页面自己回传「答完回来了」。
        """
        upstream = self.use_fake_upstream("ok", timeout=REAL_UPSTREAM_TIMEOUT_SECONDS)
        harness = self.browser(inject=BROWSER_SUBMIT_ONCE_JS + BROWSER_PROBE_JS,
                               seed={"q": "真的问过的问题", "token": "seedtoken0000002",
                                     "before": 0, "stalled": True},
                               stage=self.STAGE_ANSWERED)

        reports, _ = harness.open_live("/__seed", wait_seconds=40, until_stage="after")

        self.assertIn("after", [r.get("stage") for r in reports],
                      "页面没有回传「答完回来了」：%r" % (reports,))
        state = [r for r in reports if r.get("stage") == "after"][-1]
        self.assertEqual(upstream.requests, 1, "应该真的问过一次模型")
        self.assertEqual(state["value"], "", "已经答过了，不该再塞回输入框")
        self.assertIsNone(state["pending"], "待确认的记录该被清掉")
        self.assertIn("已经答完了", state["banner"])
        self.assertEqual(len(self.all_rows()), 2)

    def test_editing_the_draft_after_a_stall_does_not_hide_the_unsent_question(self):
        """用户改了草稿（没发出去）然后刷新，两件事必须分清楚：

        · 没发出去的改动会丢 —— 浏览器不会替保存草稿，这一点我们改变不了；
        · 但【真正要紧的那条】（已经发出去、结果还未知）必须回来，
          而且带着原来的身份牌回来 —— 这样他要再发一次也不会变成第二次调用。
        """
        harness = self.browser(inject=BROWSER_EDIT_DRAFT_THEN_RELOAD_JS + BROWSER_PROBE_JS,
                               seed={"q": "已经发出去的问题", "token": "seedtoken0000004",
                                     "before": 0, "stalled": True})
        state = self.probe(harness.open_virtual("/__seed", budget_ms=10000))

        self.assertEqual(state["value"], "已经发出去的问题", "回来的该是发出去的那条")
        self.assertIn("seedtoken0000004", state["pending"], "身份牌要留着")
        self.assertFalse(state["bannerHidden"], "该跟他说清这条结果未知")

    def test_clearing_the_box_drops_the_staged_question(self):
        """用户把输入框清干净 = 这条问题不要了：刷新时不该再塞回来。"""
        harness = self.browser(inject=BROWSER_CLEAR_THEN_CHECK_JS,
                               seed={"q": "被用户删掉的问题", "token": "seedtoken0000003",
                                     "before": 0, "stalled": True})
        dom = harness.open_virtual("/__seed", budget_ms=8000)
        self.assertIn("STORAGE:NONE", dom, "清空输入框之后，暂存的那条该被丢掉")

    def test_a_staged_question_does_not_survive_a_deletion_in_another_tab(self):
        """【跨标签页那一种】在别的标签页里把数据删了，这个标签页过一会儿才刷新。

        那时候首页那次「刚清空过」的一次性信号早被别的标签页用掉了，所以这一条靠的是
        另一条更耐久的规则：历史【变短了】—— 消息只会被删、不会凭空少，
        条数变小只有一种解释：那些记录被清空了。暂存记录里记着
        「我发出去的时候有 3 条用户消息」，现在一条都没有，这条问题就不该再放回输入框。
        """
        harness = self.browser(inject=BROWSER_PROBE_JS,
                               seed={"q": "别的标签页删掉之前的问题", "token": "seedtoken0000006",
                                     "before": 3, "stalled": True})
        state = self.probe(harness.open_virtual("/__seed", budget_ms=6000))

        self.assertEqual(state["value"], "", "历史变短了 = 那些记录被清空了，不该放回输入框")
        self.assertIsNone(state["pending"], "那条记录该被丢掉")
        self.assertTrue(state["bannerHidden"], "这件事不必再向他解释一句")

    def test_a_staged_question_does_not_survive_losing_the_binding(self):
        """【跨标签页的另一种情形】暂存的时候历史里本来就什么都没有（基准是 0），
        所以「历史变短了」这条看不出来 —— 但删除必然解绑，靠这一位仍然认得出。

        这一页既没有那个一次性信号，历史条数也没变（都是 0），
        它认得出来只可能是因为「原先绑着学习者、现在不绑了」。
        """
        harness = self.browser(inject=BROWSER_PROBE_JS,
                               seed={"q": "解绑之前问出去的问题", "token": "seedtoken0000007",
                                     "before": 0, "bound": True, "stalled": True})
        state = self.probe(harness.open_virtual("/__seed", budget_ms=6000))

        self.assertEqual(state["value"], "", "绑定没了 = 他的数据被清空过，不该放回输入框")
        self.assertIsNone(state["pending"], "那条记录该被丢掉")

    def test_a_pending_question_does_not_come_back_after_deleting_all_data(self):
        """【核心】真的把个人数据删掉之后，浏览器里那条「待确认的问题」不许再冒出来。

        这一条里全是真操作：真浏览器、真邀请码表单、真档案页、真删除表单，
        以及服务端那次「刚清空过」的一次性信号。跑完之后输入框必须是空的、
        浏览器里那条记录必须没了 —— 用户刚说过「全部删掉」。
        """
        self.make_invite("browser-erase")
        harness = self.browser(
            inject=BROWSER_DELETE_FLOW_JS.replace("__INVITE_CODE__", "browser-erase")
                   + BROWSER_PROBE_JS,
            seed={"q": "删除前留下的问题", "token": "seedtoken0000005",
                  "before": 0, "stalled": True},
            stage=self.STAGE_WIPED)

        reports, posts = harness.open_live("/__seed", wait_seconds=40, until_stage="done")

        # 【先确认这条测试真的删掉了东西】不然下面几条断言可能只是「什么都没发生」
        self.assertIn("done", [r.get("stage") for r in reports],
                      "删除流程没有走完：%r" % ([r.get("stage") for r in reports],))
        self.assertIn("/profile/delete", posts, "删除表单没有真的提交")
        self.assertEqual(self.count_rows("learner_sessions"), 0, "会话绑定该被删掉")
        self.assertEqual(self.count_rows("learners"), 0, "学习者身份该被删掉")

        state = [r for r in reports if r.get("stage") == "done"][-1]
        self.assertEqual(state["value"], "", "刚删过数据，那条问题不该再放回输入框")
        self.assertIsNone(state["pending"], "待确认的记录该被丢掉")
        self.assertTrue(state["bannerHidden"], "这件事不必再向他解释一句")

    def test_a_staged_question_does_not_revive_across_a_delete_and_a_rebind(self):
        """【用户点名的那条路，一步都不少】删除数据 → 另一个标签页用掉一次性信号 →
        重新绑定 → 旧标签页刷新。

        【为什么这条非单独测不可】前面两条跨标签页的用例各只借了一条老线索，
        而这条路会把它们**逐个绕开**：
          · 那个一次性的「刚清空过」信号 —— 被删数据那个标签页回首页时用掉了；
          · 「历史变短了」—— 暂存的时候基准就是 0（那是这个浏览器的第一句、回答还没落地），
            删完还是 0，两头一模一样，看不出任何变化；
          · 「绑定没了」—— 用户随后拿第二张码**重新绑了回来**，这条线索也被补了回去。
        只剩「数据代号对不上」还认得出：这条记录指的是已经被删掉的那份数据。

        全部是真操作：两个真标签页、真 cookie（代号在里面，两个标签页共用）、真 sessionStorage、
        真邀请码表单、真档案页、真删除表单、真模型客户端（打本机假上游，不联网、不用真密钥）。
        """
        self.make_invite("browser-first")
        self.make_invite("browser-second")
        harness = self.browser(
            inject=BROWSER_CROSS_TAB_DELETE_JS.replace("__INVITE_FIRST__", "browser-first")
                                           .replace("__INVITE_SECOND__", "browser-second"),
            # 只吃掉【第一个】POST 的响应：就是旧标签页那一次提问（他要的就是「结果未知」）。
            # 另一个标签页的输码 / 删除必须拿到响应，否则它根本走不下去。
            swallow_posts=1)

        reports, posts = harness.open_live("/", wait_seconds=120, until_stage="after")

        stages = [r.get("stage") for r in reports]
        self.assertIn("rebound", stages, "另一个标签页没走完「删数据 + 重新绑定」：%r" % (reports,))
        self.assertIn("after", stages, "旧标签页刷新那一步没有回传：%r" % (reports,))
        sent = [r for r in reports if r.get("stage") == "sent"][-1]
        final = [r for r in reports if r.get("stage") == "after"][-1]

        # 【先确认这条路真的走过了】不然下面那几条断言可能只是「什么都没发生」
        self.assertIn("/profile/delete", posts, "删除表单没有真的提交")
        self.assertEqual(self.count_rows("messages"), 0, "聊天记录该被删干净了")
        self.assertEqual(final["count"], "0", "刷新的时候历史确实是空的")
        self.assertEqual(final["bound"], "true", "用户重新绑上了 —— 「绑定没了」这条线索被补了回来")
        self.assertNotEqual(final["epoch"], sent["epoch"],
                            "数据被清空了，代号必须换一个（下面那条断言认的就是它）")

        # 【要测的那件事】那句已经被删掉的问题，绝不能又冒回输入框
        self.assertEqual(final["value"], "", "刚被删掉的那句问题又冒回输入框了")
        self.assertIsNone(final["pending"], "那条待确认的记录该被丢掉")
        self.assertTrue(final["bannerHidden"], "这件事不必再向他解释一句")

    def count_rows(self, table):
        """临时库里某张表现在有几行（表名由调用点写死，不来自外部输入）。"""
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
        finally:
            conn.close()

    # ---------- C. 看门狗到点：收回界面、不重发 ----------

    def test_the_watchdog_releases_the_page_without_sending_anything(self):
        """界面收得回来、话说明白、手动刷新入口露出来 —— 而且一个请求都没多发。"""
        tutor.MODEL_REQUEST_TIMEOUT_SECONDS = 5          # 页面上显示成「预算 5 秒」
        tutor.MODEL_STALL_MARGIN_SECONDS = 2             # 看门狗 7 秒后到点
        harness = self.browser(inject=BROWSER_BLOCK_SUBMIT_JS + BROWSER_PROBE_JS)
        state = self.probe(harness.open_virtual("/", budget_ms=12000, report_after_ms=9000))

        self.assertFalse(state["bannerHidden"], "看门狗到点后横幅该露出来")
        self.assertIn("结果未知", state["banner"])
        self.assertFalse(state["actionHidden"], "「刷新看看结果」该露出来")
        self.assertFalse(state["readOnly"], "输入框该能再编辑了")
        self.assertIn("发送", state["button"])
        self.assertEqual(state["value"], "被卡住的问题", "用户打的字不该丢")
        self.assertIn('"stalled":true', state["pending"].replace(" ", ""),
                      "该把这条记成「结果未知」，刷新时才解释得清")
        # 【红线】整个过程浏览器一个 POST 都没发出去 —— 看门狗不会自己重发
        self.assertEqual(harness.posts, [])

    def test_the_watchdog_fires_while_the_request_is_really_still_out_there(self):
        """真刀真枪那一版：请求真的发出去、真的卡在服务端，看门狗照样只收界面。

        【和上一条的区别】这一条不按虚拟时间跑：请求是真的、卡住是真的、
        七秒也是真的。页面在看门狗到点后主动把状态回传给测试（用测试注入的探针）。
        """
        upstream = self.use_fake_upstream("gate", timeout=30)   # 卡在服务端，一直不答
        tutor.MODEL_REQUEST_TIMEOUT_SECONDS = 5                 # 页面上显示成「预算 5 秒」
        tutor.MODEL_STALL_MARGIN_SECONDS = 2                    # 看门狗 7 秒后到点
        harness = self.browser(inject=BROWSER_LIVE_PROBE_JS)

        reports, posts = harness.open_live("/", wait_seconds=40)

        self.assertTrue(reports, "探针没有回传：看门狗可能压根没到点，或者脚本没跑起来")
        state = reports[0]
        self.assertTrue(upstream.arrived.wait(5), "请求应该真的发到了上游并被卡住")
        self.assertIn("结果未知", state["banner"])
        self.assertFalse(state["actionHidden"])
        self.assertFalse(state["readOnly"], "界面该被收回来了")
        self.assertIn("发送", state["button"])
        self.assertEqual(state["value"], "卡在路上的问题")
        self.assertEqual(posts, ["/"], "看门狗不该自己再发一次（只该有用户那一次）")
        upstream.release.set()          # 放它走完，别留一个半截的请求在后台
        time.sleep(0.5)

    def test_a_lost_response_resend_in_a_real_browser_still_calls_the_model_once(self):
        """【这一轮要解决的场景，端到端跑一遍】

        流程全部是真的：真浏览器、真表单、真服务端、真模型客户端（打本机假上游）。
          · 第一次发送：服务端答完了、写进库了，但响应被代理吃掉 —— 对他就是「丢了」；
          · 看门狗到点：说清「结果未知」，不自动重发；
          · 他再按一次发送：页面沿用【同一个身份牌】；
          · 服务端认出这是同一次尝试：不再调模型、不再扣额度，把已有的回答还给他。

        【证明的落点】模型只被调用了一次（upstream.requests == 1）、库里只有一对问答，
        而浏览器确实发过两次 POST，且两次带的身份牌一模一样。
        """
        upstream = self.use_fake_upstream("gate", timeout=30)
        tutor.MODEL_REQUEST_TIMEOUT_SECONDS = 5
        tutor.MODEL_STALL_MARGIN_SECONDS = 2                    # 看门狗 7 秒后到点
        harness = self.browser(inject=BROWSER_LIVE_RESEND_JS, swallow_posts=True)

        # 第一次请求一发到上游，就放行 —— 让「服务端答完但响应丢了」真的发生
        threading.Thread(target=lambda: (upstream.arrived.wait(20), upstream.release.set()),
                         daemon=True).start()
        reports, posts = harness.open_live("/", wait_seconds=60, until_stage="resent")

        stages = [report.get("stage") for report in reports]
        self.assertIn("resent", stages, "重发那一步没有回传：%r" % (reports,))
        state = [r for r in reports if r.get("stage") == "resent"][0]

        self.assertTrue(upstream.arrived.is_set(), "第一次请求应该真的发出去了")
        self.assertEqual(upstream.requests, 1, "同一次尝试不该再问一次模型")
        self.assertEqual(len(self.all_rows()), 2, "库里只该有一对问答")
        self.assertEqual(posts, ["/", "/"], "浏览器应该发过两次：用户第一次 + 他后来的重发")
        tokens = harness.attempt_tokens()
        self.assertEqual(len(tokens), 2)
        self.assertRegex(tokens[0], r"^[0-9a-f]{32}$")
        self.assertEqual(tokens[0], tokens[1],
                         "重发必须沿用同一个身份牌，服务端才认得出是同一次尝试")
        self.assertIn(tokens[1], state["pending"], "身份牌该一直留在页面上")
        self.assertIn("结果未知", state["banner"])


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_app.py 也能跑
