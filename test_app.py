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
import sys                                 # 把当前文件夹加进模块搜索路径
import json                                # 假模型要返回 JSON，这里负责拼
import atexit                              # 程序结束时顺手清理临时文件
import shutil                              # 删除临时文件夹用
import sqlite3                             # 测试里要直接查数据库，验证数据真的写进去了
import tempfile                            # 建临时文件夹，让测试用独立的数据库文件
import unittest                            # Python 自带的测试框架，不用额外安装
import importlib                           # 用来「重新加载模块」，模拟重启 Flask
import threading                           # 并发测每日额度时要起多个线程一起抢

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
        return FakeResponse(self.owner.reply)

class FakeChat:
    def __init__(self, owner):
        self.completions = FakeCompletions(owner)

class FakeClient:
    def __init__(self):
        self.calls = []                     # 每次被调用都记在这里，用来证明「调了几次」「传了什么」
        self.reply = rag_reply()            # 默认返回一个合法的 RAG 结果
        self.chat = FakeChat(self)


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
        """AI 报错之后锁必须还回来，否则页面就再也用不了了。

        【注意行为变了】以前模型报错会让整个请求失败、显示一条错误。
        现在 rag.py 会把模型异常安全降级成「证据不足」，所以请求本身是成功的
        （302 跳转），只是回答变成了一句安全的拒答。锁照样必须释放。
        """
        c = tutor.app.test_client()
        c.get("/")

        def boom(model, messages, **kwargs):                # 让假的 AI 故意炸一次
            raise RuntimeError("模拟网络故障")
        self.fake.chat.completions.create = boom

        r = self.ask(c, "这一问会失败")

        self.assertEqual(r.status_code, 302)                # 安全降级后照常走 PRG
        self.assertFalse(tutor._lock.locked(), "出错后锁必须被释放")

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

    def test_api_exception_does_not_leak_and_is_saved_as_safe_refusal(self):
        """【核心】模型异常时：不泄露异常原文，同时记下一条安全拒答。"""
        marker = "内部细节-接口炸了-7c1e"

        def boom(model, messages, **kwargs):
            raise RuntimeError(marker)
        self.fake.chat.completions.create = boom

        c = tutor.app.test_client()
        self.ask(c, "随便问一句")

        rows = self.all_rows()
        self.assertEqual(len(rows), 2, "安全降级后仍应记下这一问一答")
        dump = " ".join(str(r) for r in rows)
        self.assertNotIn(marker, dump, "异常原文泄露进数据库了")

        html = c.get("/").get_data(as_text=True)
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
        """【诚实性】没做过的事就要写没做过。"""
        doc = self.read("DEPLOYMENT.md")
        self.assertIn("没有实际创建任何备份", doc)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_app.py 也能跑
