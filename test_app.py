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

    def test_no_chunks_skips_the_model_and_saves_a_refusal(self):
        """【核心】没检索到资料时，绝不调用模型，但要把这条安全拒答记下来。"""
        self.use_empty_retrieval()
        c = tutor.app.test_client()
        self.ask(c, "课程多少钱？")

        self.assertEqual(len(self.fake.calls), 0, "没有资料时不该调用模型")
        self.assertEqual(len(RETRIEVAL_CALLS), 1, "检索还是要跑一次的")

        rows = self.all_rows()
        self.assertEqual(len(rows), 2, "拒答也要如实记进历史")
        self.assertTrue(rows[1][2].strip(), "拒答内容不能是空的")

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

    def test_model_receives_only_the_current_question(self):
        """【本轮的边界】模型只收到当前问题，不再携带历史。

        以前是把整段历史一起发给模型，让它「有上下文」。
        现在走 RAG：只按当前问题检索资料再回答。
        于是依赖上一轮指代的问题（比如「那它呢？」）这一版不保证正确——
        这是刻意的取舍，README 里已如实写明，不让用户误以为它能听懂上下文。
        """
        c = tutor.app.test_client()
        self.ask(c, "第一轮问的话")
        self.ask(c, "第二轮问的话")

        sent = self.fake.calls[-1]                          # 最后一次调用
        contents = " ".join(m["content"] for m in sent["messages"])

        self.assertIn("第二轮问的话", contents)
        self.assertNotIn("第一轮问的话", contents, "历史不该被发给模型")

    def test_history_is_still_stored_even_though_it_is_not_sent(self):
        """历史不再发给模型，但仍要照常保存和展示——两件事不能混为一谈。"""
        c = tutor.app.test_client()
        self.ask(c, "第一轮问的话")
        self.ask(c, "第二轮问的话")

        self.assertEqual(len(self.all_rows()), 4)           # 两问两答，一条不少
        html = c.get("/").get_data(as_text=True)
        self.assertIn("第一轮问的话", html)                   # 第一轮在页面上仍然看得到
        self.assertIn("第二轮问的话", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_app.py 也能跑
