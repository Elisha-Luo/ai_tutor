# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# ai_tutor 的自动化测试
#
# 【最重要的一条原则】这些测试绝不调用真实的 DeepSeek API。
# 做法是把 app 模块里的 client 整个换成一个假的，所有请求都走假货。
# 这样测试跑起来又快又不花钱，而且断网也能跑。
#
# 运行方式（在 ai_tutor 文件夹里）：
#     python -m unittest test_app -v
# =====================================================================

import os                                  # 设环境变量、拼路径
import sys                                 # 把当前文件夹加进模块搜索路径
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
        self.reply = "这是测试用的假回答。"
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

        self.fake = FakeClient()
        tutor.client = self.fake                           # 【打桩】把真客户端换掉，从此不可能调真 API
        tutor.app.config["TESTING"] = True

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


# ===================== 1. 会话隔离 =====================

class TestSessionIsolation(ChatTestCase):

    def test_two_sessions_do_not_leak(self):
        """两个独立会话，各自只能看到自己的对话，绝不串话。"""
        c1 = tutor.app.test_client()       # 第一个「浏览器」
        c2 = tutor.app.test_client()       # 第二个「浏览器」，cookie 是分开的

        c1.post("/", data={"question": "会话一的秘密问题"})
        c2.post("/", data={"question": "会话二的问题"})

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
        c1.post("/", data={"question": "甲的问题"})
        c2.post("/", data={"question": "乙的问题"})

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
        c.post("/", data={"question": "重启之前问的问题"})

        cookie = c.get_cookie("session")                    # 浏览器手里那个签名过的 cookie
        self.assertIsNotNone(cookie, "提交之后应该拿到会话 cookie")

        # ---- 模拟「重启服务器」----
        # 重新加载模块 = 进程里的所有状态清空：锁是新的、Flask 应用对象是新的。
        # 唯一活下来的是磁盘上的数据库文件，以及浏览器手里的 cookie。
        tutor = importlib.reload(tutor)
        tutor.client = FakeClient()                         # 新模块也要重新打桩

        c2 = tutor.app.test_client()                        # 全新的应用实例
        c2.set_cookie("session", cookie.value)              # 但用的是同一个浏览器 cookie

        html = c2.get("/").get_data(as_text=True)
        self.assertIn("重启之前问的问题", html)               # 历史还在

    def test_history_read_from_database_not_memory(self):
        """确认历史真的是从数据库读的，而不是留在某个内存变量里。"""
        c = tutor.app.test_client()
        c.post("/", data={"question": "这句话应该落进数据库"})

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

        c.post("/", data={"question": ""})
        c.post("/", data={"question": "   "})
        c.post("/", data={"question": "\n\t  "})

        self.assertEqual(self.all_rows(), [], "空问题不该在数据库里留下任何记录")
        self.assertEqual(len(self.fake.calls), 0, "空问题不该调用 AI")

    def test_empty_question_does_not_block_later_questions(self):
        """空问题不能把锁占住——之后正常提问还得能用。"""
        c = tutor.app.test_client()
        c.post("/", data={"question": "   "})               # 先来一个空的
        r = c.post("/", data={"question": "之后正常的问题"})  # 再来一个正常的

        self.assertEqual(r.status_code, 302)                # 正常走了 POST-Redirect-GET
        self.assertEqual(len(self.all_rows()), 2)           # 一问一答正好两条
        self.assertEqual(self.fake.reply, self.all_rows()[1][2])


# ===================== 4. 原有行为没被破坏 =====================

class TestExistingBehaviour(ChatTestCase):

    def test_post_redirect_get_still_works(self):
        """成功后仍然是 302 跳转，不是直接渲染——F5 才不会重复提问。"""
        c = tutor.app.test_client()
        r = c.post("/", data={"question": "随便问一句"})
        self.assertEqual(r.status_code, 302)

    def test_duplicate_request_is_blocked(self):
        """上一个请求还在处理时，第二次提问被挡住：不调 AI，也不写数据库。"""
        c = tutor.app.test_client()
        c.get("/")

        tutor._lock.acquire()                               # 假装「上一个问题还在问 AI」
        try:
            c.post("/", data={"question": "这是重复按回车发出的"})
        finally:
            tutor._lock.release()                           # 记得还回去，否则后面的测试都会卡住

        self.assertEqual(len(self.fake.calls), 0, "重复请求不该调用 AI")
        self.assertEqual(self.all_rows(), [], "重复请求不该写数据库")

    def test_lock_is_released_after_ai_error(self):
        """AI 报错之后锁必须还回来，否则页面就再也用不了了。"""
        c = tutor.app.test_client()
        c.get("/")

        def boom(model, messages, **kwargs):                # 让假的 AI 故意炸一次
            raise RuntimeError("模拟网络故障")
        self.fake.chat.completions.create = boom

        r = c.post("/", data={"question": "这一问会失败"})
        html = r.get_data(as_text=True)

        self.assertEqual(r.status_code, 200)                # 不白屏
        self.assertIn("出错了", html)
        self.assertFalse(tutor._lock.locked(), "出错后锁必须被释放")
        self.assertEqual(self.all_rows(), [], "失败的对话不该写进数据库")

    def test_no_api_key_in_database(self):
        """数据库里绝不能出现 API 密钥。"""
        c = tutor.app.test_client()
        c.post("/", data={"question": "随便问一句"})

        conn = sqlite3.connect(self.db_path)
        try:
            dump = " ".join(str(r) for r in conn.execute("SELECT * FROM messages").fetchall())
        finally:
            conn.close()
        self.assertNotIn(tutor.API_KEY, dump)


# ===================== 5. 测试自身的安全保证 =====================

class TestHarnessSafety(ChatTestCase):

    def test_uses_fake_key_not_the_real_one(self):
        """确认测试用的是假密钥。万一哪天环境变量被子进程污染，这条会立刻失败。"""
        self.assertEqual(tutor.API_KEY, "test-key-not-a-real-key")

    def test_client_is_stubbed(self):
        """确认真的客户端已经被换成假的——这是「绝不调用真实 API」的结构性保证。"""
        self.assertIsInstance(tutor.client, FakeClient)

    def test_fake_client_actually_receives_calls(self):
        """反面验证：正常提问时假客户端确实被调用了，说明上面那些断言不是空转。"""
        c = tutor.app.test_client()
        c.post("/", data={"question": "验证假客户端有被调用"})
        self.assertEqual(len(self.fake.calls), 1)

        # 顺便确认发出去的对话结构是对的：系统提示 + 历史 + 本次提问
        sent = self.fake.calls[0]["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[-1]["content"], "验证假客户端有被调用")

    def test_multiturn_context_comes_from_this_session_only(self):
        """多轮上下文：第二轮请求里应带上第一轮的内容，且只带自己会话的。"""
        c1 = tutor.app.test_client()
        c2 = tutor.app.test_client()

        c1.post("/", data={"question": "第一轮：甲说的话"})
        c2.post("/", data={"question": "乙说的话"})
        c1.post("/", data={"question": "第二轮：甲又说话了"})

        sent = self.fake.calls[-1]                          # 最后一次调用（甲的第二轮）
        contents = [m["content"] for m in sent["messages"]]

        self.assertIn("第一轮：甲说的话", contents)           # 带着自己的上一轮
        self.assertNotIn("乙说的话", contents)               # 但绝不带别人的


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_app.py 也能跑
