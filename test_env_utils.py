# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# env_utils.py（共享的 .env 读取工具）的自动化测试
#
# 【原则】完全离线：不联网、不调用任何模型、不碰项目里真实的 .env。
# 所有测试都在临时目录里造自己的 .env 文件，跑完就删。
#
# 【为什么环境变量要「先备份、再还原」】
# 这些测试会往 os.environ 里写东西。如果不还原，就会污染同一个进程里
# 后面跑的测试——那种失败特别难查，因为出错的地方离肇事的地方很远。
#
# 运行方式（在 ai_tutor 文件夹里）：
#     python -m unittest test_env_utils -v
# =====================================================================

import os           # 环境变量、拼路径
import io           # 接住打印输出，用来证明「什么都没打印」
import json         # 把返回对象序列化，检查里面有没有密钥
import subprocess   # 在子进程里真的跑一次 app.py，验证它会停下来
import sys          # 模块搜索路径
import tempfile     # 造临时 .env 文件，不碰项目里真的那个
import contextlib   # 重定向标准输出/报错
import unittest     # Python 自带的测试框架

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import env_utils     # 被测对象


# 测试用的假密钥。故意做得很好认，万一它出现在不该出现的地方，
# 一眼就能看出来。注意：这不是真密钥，只是一串固定的测试字符串。
FAKE_KEY = "AITUTOR_TEST_SECRET"
FAKE_SECRET = "sk-fake-SECRET-VALUE-do-not-leak-1234567890"


class EnvUtilsTestCase(unittest.TestCase):
    """共用的准备工作：临时目录 + 环境变量的备份与还原。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._saved = {}
        self._restore_registered = False

    def protect(self, *keys):
        """备份这些环境变量、先把它们清掉，测试结束后自动还原。"""
        for key in keys:
            if key not in self._saved:
                self._saved[key] = os.environ.get(key)
            os.environ.pop(key, None)

        if not self._restore_registered:
            self.addCleanup(self._restore)
            self._restore_registered = True

    def _restore(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def write_env(self, content, name=".env"):
        """在临时目录里写一个 .env 文件，返回它的路径。"""
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def write_env_bytes(self, data, name=".env"):
        """写一个内容为原始字节的 .env（用来造「不是 UTF-8」的坏文件）。"""
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path


# ===================== 1. 基本解析 =====================

class TestParsing(EnvUtilsTestCase):

    def test_only_env_file_no_process_variable(self):
        """【核心】只有 .env、没有进程环境变量时，也要能读出来。"""
        self.protect(FAKE_KEY)
        path = self.write_env(FAKE_KEY + "=" + FAKE_SECRET + "\n")

        result = env_utils.load_dotenv(path)

        self.assertTrue(result["loaded"])
        self.assertIn(FAKE_KEY, result["keys"])
        self.assertEqual(os.environ.get(FAKE_KEY), FAKE_SECRET)

    def test_blank_lines_comments_and_junk_are_skipped(self):
        """空行、注释行、没有等号的行，一律跳过。"""
        self.protect(FAKE_KEY)
        path = self.write_env(
            "# 这是注释\n"
            "\n"
            "    \n"
            "这一行没有等号\n"
            "# 另一条注释\n"
            + FAKE_KEY + "=value\n"
            "\n"
        )

        result = env_utils.load_dotenv(path)
        self.assertEqual(result["keys"], [FAKE_KEY])
        self.assertGreaterEqual(result["skipped"], 5)

    def test_quotes_are_stripped(self):
        """值两边的引号要去掉（单引号双引号都处理）。"""
        self.protect(FAKE_KEY)
        for raw, expected in [('"double"', "double"), ("'single'", "single")]:
            os.environ.pop(FAKE_KEY, None)
            path = self.write_env(FAKE_KEY + "=" + raw + "\n")
            env_utils.load_dotenv(path)
            self.assertEqual(os.environ.get(FAKE_KEY), expected)

    def test_value_may_contain_equals_sign(self):
        """值里再有等号不能被切坏——这就是用 partition 而不是 split 的原因。"""
        self.protect(FAKE_KEY)
        path = self.write_env(FAKE_KEY + "=a=b=c\n")
        env_utils.load_dotenv(path)
        self.assertEqual(os.environ.get(FAKE_KEY), "a=b=c")

    def test_whitespace_around_key_and_value_is_trimmed(self):
        self.protect(FAKE_KEY)
        path = self.write_env("   " + FAKE_KEY + "   =   spaced value   \n")
        env_utils.load_dotenv(path)
        self.assertEqual(os.environ.get(FAKE_KEY), "spaced value")

    def test_line_without_a_key_is_skipped(self):
        """形如 "=abc" 的行没有键名，要跳过而不是把空字符串塞进环境变量。"""
        self.protect(FAKE_KEY)
        path = self.write_env("=只有值没有键\n" + FAKE_KEY + "=ok\n")
        result = env_utils.load_dotenv(path)
        self.assertEqual(result["keys"], [FAKE_KEY])
        self.assertNotIn("", os.environ)

    def test_missing_file_is_silent_and_reports_not_loaded(self):
        """文件不存在属于正常情况：安静跳过，不报错。"""
        result = env_utils.load_dotenv(os.path.join(self.tmp.name, "根本没有这个文件.env"))
        self.assertFalse(result["loaded"])
        self.assertEqual(result["keys"], [])
        self.assertFalse(result["read_error"])


# ===================== 2. 环境变量优先级 =====================

class TestPrecedence(EnvUtilsTestCase):

    def test_process_variable_wins_over_env_file(self):
        """【核心】进程环境变量与 .env 同时存在时，进程环境变量优先。"""
        self.protect(FAKE_KEY)
        os.environ[FAKE_KEY] = "来自真实环境变量"

        path = self.write_env(FAKE_KEY + "=来自dotenv文件\n")
        env_utils.load_dotenv(path)

        self.assertEqual(os.environ.get(FAKE_KEY), "来自真实环境变量")

    def test_env_file_fills_in_what_is_missing(self):
        """环境变量里没有的键，.env 负责补上（这正是「兜底」的意思）。"""
        self.protect(FAKE_KEY, "AITUTOR_TEST_OTHER")
        os.environ[FAKE_KEY] = "已有"
        path = self.write_env(FAKE_KEY + "=会被忽略\nAITUTOR_TEST_OTHER=被补上\n")

        result = env_utils.load_dotenv(path)

        self.assertEqual(os.environ.get(FAKE_KEY), "已有")
        self.assertEqual(os.environ.get("AITUTOR_TEST_OTHER"), "被补上")
        # 两个键都会被记录：setdefault 对已有的键是「尝试过」，对没有的键是「写入了」
        self.assertCountEqual(result["keys"], [FAKE_KEY, "AITUTOR_TEST_OTHER"])


# ===================== 3. 绝不泄露密钥 =====================

class TestNoSecretLeakage(EnvUtilsTestCase):

    def test_returned_object_contains_no_secret(self):
        """【核心】返回的统计信息里绝不能出现密钥的值。"""
        self.protect(FAKE_KEY)
        path = self.write_env(FAKE_KEY + "=" + FAKE_SECRET + "\n")

        result = env_utils.load_dotenv(path)

        blob = json.dumps(result, ensure_ascii=False)
        self.assertNotIn(FAKE_SECRET, blob)
        # 但键名是允许出现的——它不是秘密，.env.example 里就公开写着
        self.assertIn(FAKE_KEY, blob)

    def test_nothing_is_printed(self):
        """【核心】读取过程一个字符都不打印，更不会把密钥打出来。"""
        self.protect(FAKE_KEY)
        path = self.write_env(FAKE_KEY + "=" + FAKE_SECRET + "\n")

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            env_utils.load_dotenv(path)

        self.assertEqual(out.getvalue(), "", "读取 .env 不该打印任何东西")
        self.assertEqual(err.getvalue(), "", "读取 .env 不该往报错流写任何东西")

    def test_read_error_does_not_leak_file_content(self):
        """【核心】读取失败时，返回信息里也不能带出文件内容。

        造一个「不是 UTF-8」的坏文件：坏字节后面紧跟着「密钥」。
        如果实现把 UnicodeDecodeError 原样往外抛或原样记下来，
        那段密钥所在的内容就会被带出来——这里就是专门防这个。
        """
        self.protect(FAKE_KEY)

        data = (FAKE_KEY + "=").encode("utf-8") + b"\xff\xfe" + FAKE_SECRET.encode("utf-8")
        path = self.write_env_bytes(data)

        result = env_utils.load_dotenv(path)      # 不该抛异常

        self.assertTrue(result["read_error"])
        blob = json.dumps(result, ensure_ascii=False)
        self.assertNotIn(FAKE_SECRET, blob, "读取失败的返回信息里泄露了文件内容")
        self.assertNotIn("�", blob)

    def test_bad_file_does_not_crash_and_leaves_no_junk(self):
        """坏文件不能让程序崩，也不该在环境变量里留下半截垃圾。"""
        self.protect(FAKE_KEY)
        path = self.write_env_bytes((FAKE_KEY + "=").encode("utf-8") + b"\xff\xfe")

        result = env_utils.load_dotenv(path)
        self.assertTrue(result["read_error"])
        self.assertNotEqual(os.environ.get(FAKE_KEY), "")


# ===================== 4. 与项目实际约定一致 =====================

class TestProjectConventions(EnvUtilsTestCase):

    def test_default_path_points_at_project_root_env(self):
        """默认路径必须是项目根目录下的 .env —— 和 app.py 读的是同一个文件。"""
        expected = os.path.join(env_utils.BASE_DIR, ".env")
        self.assertEqual(env_utils.DEFAULT_ENV_PATH, expected)

    def test_env_utils_module_never_prints_or_logs(self):
        """【结构性保证】这个模块里不该出现 print / logging / warnings。"""
        path = os.path.join(BASE, "env_utils.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()

        for banned in ["print(", "logging.", "warnings.warn"]:
            # 只检查真正的代码行，跳过注释和文档字符串
            for line in source.splitlines():
                code = line.split("#")[0]
                self.assertNotIn(banned, code,
                                 "env_utils.py 的代码里出现了 " + banned + "：" + line.strip())

    def test_app_and_eval_runner_share_the_same_loader(self):
        """【结构性保证】两个入口都必须 import 同一个 load_dotenv，不许各写一份。"""
        for filename in ["app.py", os.path.join("evals", "run_rag_eval.py")]:
            path = os.path.join(BASE, filename)
            with open(path, encoding="utf-8") as f:
                source = f.read()
            self.assertIn("from env_utils import load_dotenv", source,
                          filename + " 没有使用共享的 .env 读取工具")
            self.assertNotIn("def load_dotenv", source,
                             filename + " 里还留着自己那份 load_dotenv 实现")


# ===================== 示例密钥必须被提前拦住 =====================
#
# 【真实踩到的坑】
# 把 `.env.example` 复制成 `.env` 之后忘了改示例值。程序【照常启动、照常发请求】，
# 每一次都失败 —— 失败又被 RAG 的安全兜底挡住，最后看起来「跑完了」，
# 实际一次都没跑通，白白浪费一整轮排查。
#
# 【修正】在【调用网络之前】就停下来，并且只说一句固定的话，
# 不打印密钥本身、前缀、后缀或长度。

class ExampleApiKeyConstantTest(unittest.TestCase):
    """常量本身要守住的几件事。"""

    def test_example_key_constant_is_not_empty(self):
        """示例值不能是空串 —— 否则「空密钥」也会被误判成示例值。"""
        self.assertTrue(env_utils.EXAMPLE_API_KEY.strip())

    def test_message_does_not_contain_the_example_key(self):
        """给用户看的提示里不能夹带示例值本身。"""
        self.assertNotIn(env_utils.EXAMPLE_API_KEY, env_utils.EXAMPLE_KEY_MESSAGE)

    def test_message_points_at_the_variable_name(self):
        """提示要够具体，用户才知道该改哪儿。"""
        self.assertIn("DEEPSEEK_API_KEY", env_utils.EXAMPLE_KEY_MESSAGE)

    def test_constant_matches_the_public_template_file(self):
        """【核心】常量必须和 .env.example 里那一行完全一致。

        只读 `.env.example`（公开模板，提交进仓库的），
        【不读 `.env`】—— 那个文件里可能有真实密钥。
        这条测试保证：以后有人改了模板却忘了改常量，这里立刻会红。
        """
        path = os.path.join(BASE, ".env.example")
        if not os.path.exists(path):
            self.skipTest("没有 .env.example，跳过")

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("DEEPSEEK_API_KEY"):
                    _, _, value = line.partition("=")
                    self.assertEqual(value.strip().strip('"').strip("'"),
                                     env_utils.EXAMPLE_API_KEY,
                                     ".env.example 的示例值和 env_utils.EXAMPLE_API_KEY 对不上了")
                    return
        self.fail(".env.example 里没有 DEEPSEEK_API_KEY 这一行")


class IsExampleApiKeyTest(unittest.TestCase):
    """判断函数本身。"""

    def test_exact_example_value_is_detected(self):
        self.assertTrue(env_utils.is_example_api_key(env_utils.EXAMPLE_API_KEY))

    def test_surrounding_whitespace_still_detected(self):
        """值两边带空格或引号，也要认出来。"""
        self.assertTrue(env_utils.is_example_api_key("  " + env_utils.EXAMPLE_API_KEY + " "))

    def test_normal_test_value_is_not_flagged(self):
        """【核心】普通的测试字符串不能被误伤 —— 否则会打断离线测试。"""
        for value in ("test-key-not-a-real-key", FAKE_KEY, FAKE_SECRET, "sk-abcdef123456"):
            self.assertFalse(env_utils.is_example_api_key(value), value)

    def test_empty_and_none_are_not_flagged(self):
        """空值不是「示例值」，是「没配」——那是另一条分支管的。"""
        for value in ("", "   ", None):
            self.assertFalse(env_utils.is_example_api_key(value))

    def test_no_length_or_format_rule(self):
        """【核心】只认「长得和模板一样」，不认长度。

        服务商的密钥格式以后可能变（33 位、35 位、40 位……），
        硬编码长度规则迟早会误伤真实密钥。
        """
        for value in ("a" * 20, "b" * 35, "c" * 64, "x"):
            self.assertFalse(env_utils.is_example_api_key(value), value)

    def test_detection_prints_nothing(self):
        """【安全】判断过程一个字都不能打印。"""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            env_utils.is_example_api_key(env_utils.EXAMPLE_API_KEY)
            env_utils.is_example_api_key(FAKE_SECRET)
            env_utils.is_example_api_key(None)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "")

    def test_detection_result_carries_no_key_material(self):
        """【安全】返回值只能是个布尔值，不能顺带把密钥捎出去。"""
        result = env_utils.is_example_api_key(FAKE_SECRET)
        self.assertIsInstance(result, bool)
        self.assertNotIn(FAKE_SECRET, repr(result))


class MakeClientRejectsBadKeysTest(unittest.TestCase):
    """两个入口（app.py / 评测器）在建客户端之前就要停下来。"""

    def setUp(self):
        self.saved = os.environ.get("DEEPSEEK_API_KEY")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = self.saved

    @staticmethod
    def _make_client():
        from evals import run_rag_eval as runner
        return runner.make_client

    def test_missing_key_is_rejected(self):
        """① 压根没配密钥 → 退出，并指路 README。"""
        os.environ.pop("DEEPSEEK_API_KEY", None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as cm:
                self._make_client()()
        self.assertNotEqual(cm.exception.code, 0)
        self.assertIn("DEEPSEEK_API_KEY", out.getvalue())

    def test_exact_example_value_is_rejected(self):
        """【核心】密钥正是 .env.example 里那个公开示例值 → 退出。"""
        os.environ["DEEPSEEK_API_KEY"] = env_utils.EXAMPLE_API_KEY
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as cm:
                self._make_client()()
        self.assertNotEqual(cm.exception.code, 0)
        self.assertIn(env_utils.EXAMPLE_KEY_MESSAGE, out.getvalue())

    def test_example_rejection_does_not_leak_the_value(self):
        """【安全】拒绝时不能打印密钥本身、前缀、后缀或长度。"""
        os.environ["DEEPSEEK_API_KEY"] = env_utils.EXAMPLE_API_KEY
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit):
                self._make_client()()
        printed = out.getvalue() + err.getvalue()
        self.assertNotIn(env_utils.EXAMPLE_API_KEY, printed)
        # 除了那句固定提示里提到的 "sk-" 之外，不能再出现别的前缀片段
        self.assertNotIn("sk-", printed.replace(env_utils.EXAMPLE_KEY_MESSAGE, ""))

    def test_normal_test_value_can_still_build_a_client(self):
        """【核心】普通测试值仍然可以建客户端 —— 不能误伤离线测试。

        OpenAI(...) 只是构造对象，【不发网络请求】，所以这条测试是离线的。
        """
        os.environ["DEEPSEEK_API_KEY"] = "test-key-not-a-real-key"
        client = self._make_client()()
        self.assertIsNotNone(client)
        self.assertEqual(client.base_url.host, "api.deepseek.com")


class AppRefusesExampleKeyTest(unittest.TestCase):
    """【端到端】网页厨房（app.py）也必须拦住示例值。

    用子进程跑 `import app`，这样是真的执行 app.py 的模块级检查，
    而不是在源码里 grep 一个字符串 —— grep 过了不等于程序真的会停。
    子进程完全离线：检查发生在创建客户端之前。
    """

    def _import_app(self, key_value):
        """在子进程里 import app，返回 (退出码, 输出)。

        key_value 传 None 表示【不设】这个环境变量。
        """
        code = "import app"
        env = dict(os.environ)
        env.pop("DEEPSEEK_API_KEY", None)
        if key_value is not None:
            env["DEEPSEEK_API_KEY"] = key_value
        # PYTHONIOENCODING 保证中文提示在子进程里不会因为编码问题炸掉
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=BASE, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def test_app_refuses_the_example_value(self):
        """【核心】用示例值启动 app.py → 直接失败，并给出固定提示。"""
        code, output = self._import_app(env_utils.EXAMPLE_API_KEY)
        self.assertNotEqual(code, 0, "app.py 居然带着示例密钥正常启动了")
        self.assertIn(env_utils.EXAMPLE_KEY_MESSAGE, output)

    def test_app_refusal_does_not_echo_the_value(self):
        """【安全】失败输出里不能出现密钥本身。"""
        _, output = self._import_app(env_utils.EXAMPLE_API_KEY)
        self.assertNotIn(env_utils.EXAMPLE_API_KEY, output)

    def test_app_refuses_missing_key(self):
        """没配密钥同样要停 —— 这是原本就有的行为，别被这次修改弄丢。"""
        code, output = self._import_app(None)
        self.assertNotEqual(code, 0)
        self.assertIn("DEEPSEEK_API_KEY", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_env_utils.py 也能跑
