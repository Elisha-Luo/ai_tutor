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
        """【结构性保证】两个入口都必须 import 同一个 load_dotenv，不许各写一份。

        【为什么用正则而不是查一整句字面量】
        这行 import 可能写成单行，也可能因为名字变多而写成括号换行的形式 ——
        两者都合法。原来查的是 `"from env_utils import load_dotenv"` 这句字面量，
        一换成括号写法就会误报（本轮就撞上了）。
        正则只要求「从 env_utils 导入的东西里有 load_dotenv」，不受排版影响，
        该守的规矩一点没松：下面那条仍然禁止自己再实现一份。
        """
        for filename in ["app.py", os.path.join("evals", "run_rag_eval.py")]:
            path = os.path.join(BASE, filename)
            with open(path, encoding="utf-8") as f:
                source = f.read()
            self.assertRegex(source, r"from\s+env_utils\s+import\s*\(?[^\n]*\bload_dotenv\b",
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


# ===================== 另外两个密钥的示例值 =====================
#
# 【为什么单开一组】DEEPSEEK_API_KEY 一直有「拦住示例值」的检查，
# 但 FLASK_SECRET_KEY 和 INVITE_CODE_PEPPER 只查了「非空」——
# 而 .env.example 里那句 change-me-... 恰好也是非空的。
# 于是复制模板忘了改的人，应用照常启动，只是：
#   · cookie 签名密钥是公开的 → 谁都能伪造会话
#   · 邀请码 pepper 是公开的   → 摘要可以被离线爆破
#
# 这一组把这三条路都钉住，并且用【子进程真的 import app】来证明程序确实会停，
# 而不是只在源码里 grep 一个字符串。

class ExampleSecretConstantsTest(unittest.TestCase):
    """常量本身：对不对、会不会泄露、和模板是否一致。"""

    def test_constants_are_not_empty(self):
        self.assertTrue(env_utils.EXAMPLE_FLASK_KEY.strip())
        self.assertTrue(env_utils.EXAMPLE_PEPPER.strip())

    def test_the_two_secrets_use_different_placeholders_than_the_api_key(self):
        """三个占位符不能互相混淆 —— 否则一个检查会误伤另一个变量。"""
        self.assertNotEqual(env_utils.EXAMPLE_FLASK_KEY, env_utils.EXAMPLE_API_KEY)
        self.assertNotEqual(env_utils.EXAMPLE_PEPPER, env_utils.EXAMPLE_API_KEY)

    def test_the_message_does_not_contain_any_value(self):
        """【安全】提示里只有变量名，没有值。"""
        message = env_utils.example_secret_message("FLASK_SECRET_KEY")
        self.assertIn("FLASK_SECRET_KEY", message)          # 变量名不是秘密
        self.assertNotIn(env_utils.EXAMPLE_FLASK_KEY, message)
        self.assertNotIn(env_utils.EXAMPLE_PEPPER, message)
        self.assertNotIn(env_utils.EXAMPLE_API_KEY, message)

    def test_the_constants_match_the_shipped_template(self):
        """【核心】常量必须和 `.env.example` 里那几行完全一致 —— 防模板与代码漂移。

        只读 `.env.example`（公开模板，提交进仓库的），【绝不读 .env】。
        """
        path = os.path.join(BASE, ".env.example")
        if not os.path.exists(path):
            self.skipTest("没有 .env.example，跳过")

        with open(path, encoding="utf-8") as f:
            values = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")

        for name, constant in (("DEEPSEEK_API_KEY", env_utils.EXAMPLE_API_KEY),
                               ("FLASK_SECRET_KEY", env_utils.EXAMPLE_FLASK_KEY),
                               ("INVITE_CODE_PEPPER", env_utils.EXAMPLE_PEPPER),
                               ("ADMIN_MINT_TOKEN", env_utils.EXAMPLE_ADMIN_TOKEN)):
            with self.subTest(variable=name):
                self.assertIn(name, values, ".env.example 里没有 " + name + " 这一行")
                self.assertEqual(values[name], constant,
                                 ".env.example 的示例值和 env_utils 里的常量对不上了")


class ExampleSecretDetectionTest(unittest.TestCase):
    """is_example_secret 的判断规则：精确相等，不做泛化限制。"""

    def test_the_exact_example_value_is_detected(self):
        self.assertTrue(env_utils.is_example_secret(
            env_utils.EXAMPLE_FLASK_KEY, env_utils.EXAMPLE_FLASK_KEY))
        self.assertTrue(env_utils.is_example_secret(
            env_utils.EXAMPLE_PEPPER, env_utils.EXAMPLE_PEPPER))

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertTrue(env_utils.is_example_secret(
            "  " + env_utils.EXAMPLE_PEPPER + "  ", env_utils.EXAMPLE_PEPPER))

    def test_real_looking_values_are_not_flagged(self):
        """【核心】真实值绝不能被误伤 —— 尤其不能按长度/格式去猜。"""
        for value in ("test-secret-not-a-real-key",          # 测试用假值
                      "test-pepper-not-a-real-pepper",
                      "a" * 64,                              # 很长的随机串
                      "change-me",                           # 只是像，但不相等
                      "change-me-run-the-command-above",     # 少了后半截
                      env_utils.EXAMPLE_PEPPER + "x",        # 多了一个字符
                      ""):
            with self.subTest(value=value[:20]):
                self.assertFalse(env_utils.is_example_secret(value, env_utils.EXAMPLE_PEPPER))

    def test_none_is_not_an_example_value(self):
        """没配（None）不是「示例值」—— 那是另一条错误提示的事，别混在一起。"""
        self.assertFalse(env_utils.is_example_secret(None, env_utils.EXAMPLE_PEPPER))

    def test_the_other_variables_placeholder_does_not_match(self):
        """传错常量不该命中 —— 每个变量只认自己那一句占位符。"""
        self.assertFalse(env_utils.is_example_secret(
            env_utils.EXAMPLE_PEPPER, env_utils.EXAMPLE_API_KEY))


class AppRefusesExampleSecretsTest(unittest.TestCase):
    """【端到端】子进程里真的 import app，证明它会停下来。"""

    def _import_app(self, **overrides):
        """在子进程里 import app。返回 (退出码, 输出)。

        基线给一套【能通过检查】的假值，再用 overrides 覆盖其中某一项。
        值传 None 表示把这个变量删掉。
        """
        tmpdir = tempfile.mkdtemp(prefix="ai_tutor_envcheck_")
        self.addCleanup(__import__("shutil").rmtree, tmpdir, ignore_errors=True)

        env = dict(os.environ)
        env["DEEPSEEK_API_KEY"] = "test-key-not-a-real-key"
        env["FLASK_SECRET_KEY"] = "test-secret-not-a-real-key"
        env["INVITE_CODE_PEPPER"] = "test-pepper-not-a-real-pepper"
        env["CHAT_DB_PATH"] = os.path.join(tmpdir, "check.db")
        env["PYTHONIOENCODING"] = "utf-8"
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value

        proc = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=BASE, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def test_the_baseline_fake_values_still_work(self):
        """【核心】测试专用的假值必须照常可用 —— 别把检查做得太狠。

        （整条测试套件都靠这两个假值跑起来，这条是它们的地基。）
        """
        code, output = self._import_app()
        self.assertEqual(code, 0, "测试用的假值居然被拦住了：\n" + output[-500:])

    def test_the_example_flask_key_is_refused(self):
        """【核心】cookie 签名密钥是示例值 → 启动失败 + 固定提示。"""
        code, output = self._import_app(FLASK_SECRET_KEY=env_utils.EXAMPLE_FLASK_KEY)

        self.assertNotEqual(code, 0, "app.py 带着公开的示例签名密钥启动了")
        self.assertIn("FLASK_SECRET_KEY", output)
        self.assertIn("示例值", output)

    def test_the_example_pepper_is_refused(self):
        """【核心】邀请码 pepper 是示例值 → 启动失败 + 固定提示。"""
        code, output = self._import_app(INVITE_CODE_PEPPER=env_utils.EXAMPLE_PEPPER)

        self.assertNotEqual(code, 0, "app.py 带着公开的示例 pepper 启动了")
        self.assertIn("INVITE_CODE_PEPPER", output)
        self.assertIn("示例值", output)

    def test_the_refusal_does_not_echo_the_value(self):
        """【安全】失败输出里不能出现那个值本身。"""
        for name, value in (("FLASK_SECRET_KEY", env_utils.EXAMPLE_FLASK_KEY),
                            ("INVITE_CODE_PEPPER", env_utils.EXAMPLE_PEPPER)):
            with self.subTest(variable=name):
                _, output = self._import_app(**{name: value})
                self.assertNotIn(value, output)


class AdminTokenSwitchTest(unittest.TestCase):
    """管理令牌的开关：配成示例值 / 不配 → 入口关着；配了真值 → 打开。

    【和上面两个密钥的关键区别】管理令牌【不是必填】：
    配成示例值或不配时，应用要**照常启动**，只是管理入口关掉 —— 绝不能启动失败。
    这里用子进程真的 import app，读出它算出来的 ADMIN_ENABLED，
    而不是在测试里手写一个 False（那样测的是测试自己）。
    """

    def _admin_enabled(self, token_value):
        """在子进程里 import app，返回 (退出码, ADMIN_ENABLED 的字面值, 输出)。"""
        tmpdir = tempfile.mkdtemp(prefix="ai_tutor_admincheck_")
        self.addCleanup(__import__("shutil").rmtree, tmpdir, ignore_errors=True)

        env = dict(os.environ)
        env["DEEPSEEK_API_KEY"] = "test-key-not-a-real-key"
        env["FLASK_SECRET_KEY"] = "test-secret-not-a-real-key"
        env["INVITE_CODE_PEPPER"] = "test-pepper-not-a-real-pepper"
        env["CHAT_DB_PATH"] = os.path.join(tmpdir, "check.db")
        env["PYTHONIOENCODING"] = "utf-8"
        if token_value is None:
            env.pop("ADMIN_MINT_TOKEN", None)
        else:
            env["ADMIN_MINT_TOKEN"] = token_value

        proc = subprocess.run(
            [sys.executable, "-c",
             "import app; print('ADMIN_ENABLED=' + str(app.ADMIN_ENABLED))"],
            cwd=BASE, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output

    def test_no_token_still_starts_but_the_entry_is_closed(self):
        """【核心】不配令牌 → 应用照常启动，管理入口关着。"""
        code, output = self._admin_enabled(None)

        self.assertEqual(code, 0, "管理令牌不是必填，不该启动失败：\n" + output[-400:])
        self.assertIn("ADMIN_ENABLED=False", output)

    def test_the_example_token_still_starts_but_the_entry_is_closed(self):
        """【核心】配成公开的示例值 → 照常启动，但入口必须关着。

        否则任何人拿模板里那句公开的话就能给自己发邀请码。
        """
        code, output = self._admin_enabled(env_utils.EXAMPLE_ADMIN_TOKEN)

        self.assertEqual(code, 0, "不该因为示例值就启动失败（它不是必填项）")
        self.assertIn("ADMIN_ENABLED=False", output)

    def test_a_real_looking_token_opens_the_entry(self):
        code, output = self._admin_enabled("some-real-admin-token-9f3a" + "0" * 30)

        self.assertEqual(code, 0)
        self.assertIn("ADMIN_ENABLED=True", output)

    def test_whitespace_only_and_short_tokens_keep_the_entry_closed(self):
        """【核心】fail closed：只有空白、或者明显太短的值，都不算「配好了」。

        · 只有空白："   " 在 Python 里是【真值】—— 只写 bool() 就会误开门
        · 明显太短：多半是打错、截断，或者随手填了个占位符
        """
        for token in ("   ", "\t", "short", "x" * 31, "changeme"):
            with self.subTest(token=repr(token)[:12]):
                code, output = self._admin_enabled(token)
                self.assertEqual(code, 0)                       # 仍然不该启动失败
                self.assertIn("ADMIN_ENABLED=False", output,
                              "这种值居然把管理入口打开了：" + repr(token))

    def test_the_boundary_length_is_accepted(self):
        """正好等于最小长度的值算配好了（边界是「短于」才算过短）。"""
        code, output = self._admin_enabled("y" * env_utils.ADMIN_MIN_TOKEN_CHARS)

        self.assertEqual(code, 0)
        self.assertIn("ADMIN_ENABLED=True", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)      # 直接 python test_env_utils.py 也能跑
