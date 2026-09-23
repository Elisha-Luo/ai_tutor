# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

import os                                           # 读环境变量、拼路径
import uuid                                         # 生成随机的「会话 ID」
import time                                         # 给每个请求计时（日志里要记耗时）
import sqlite3                                      # Python 自带的轻量数据库，不用额外安装任何东西
import logging                                      # 打结构化日志
import threading                                    # 用它的「锁」防止同一时间处理两个请求
from datetime import datetime, timedelta, timezone  # datetime 记录消息时间；timedelta 设置 cookie 有效期；timezone 算 UTC 日期
from flask import Flask, render_template, request, redirect, url_for, session, jsonify   # session 给每个浏览器发身份标记；jsonify 拼 JSON 响应
from openai import OpenAI                           # 从 openai 库里导入 OpenAI 类（DeepSeek 兼容它的接口）
from env_utils import load_dotenv, is_example_api_key, EXAMPLE_KEY_MESSAGE   # 共用 .env 读取；示例密钥检测也在那边
import retriever                                     # 本地检索层：把问题变成「最相关的几段资料」
import rag                                           # 生成与引用层：让模型照着资料回答，并校验它引用的来源


# ===================== 读取 .env（如果存在）=====================
# 密钥不能写进代码，只能放环境变量。但每次开新终端都要重新设一遍太麻烦，
# 所以约定把密钥写在一个叫 .env 的文件里，程序启动时自动读进来。
# 这个文件被 .gitignore 忽略，永远不会被推到 GitHub。
#
# 【为什么读取逻辑放在 env_utils.py 而不是写在这里】
# 这个项目有两个入口都要用 .env：网页（本文件）和评测器（evals/run_rag_eval.py）。
# 以前两边各写各的，结果评测器压根没读 .env —— 密钥只写在 .env 里的人，
# 网页能用、评测器却报「密钥未设置」。同一份配置两种结论，非常难查。
# 现在全项目只有 env_utils.load_dotenv 一份实现，从根上避免再次漂移。

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 本文件（app.py）所在的文件夹，后面拼路径都以它为基准

load_dotenv(os.path.join(BASE_DIR, ".env"))         # 启动时先读 .env，下面再从环境变量取值
                                                    # 注意：.env 只作兜底，真实环境变量优先


# ===================== 配置区 =====================

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")    # 从环境变量 DEEPSEEK_API_KEY 读取密钥；没设置的话得到空字符串
if not API_KEY:                                     # 如果没读到密钥
    raise RuntimeError(                             # 直接中止程序并报错——没密钥根本用不了，不如启动时就明确报出来
        "没有找到 DeepSeek 密钥。请先设置环境变量 DEEPSEEK_API_KEY，"
        "方法见 README.md 的「配置密钥」一节。"
    )

# 【示例值要拦住】.env.example 是公开的模板文件，里面的示例值不可能管用。
# 常见事故：复制成 .env 之后忘了改 —— 应用照常启动、照常发请求，只是每次都失败。
# 与其跑完一整轮才发现，不如启动就明确报错。
# 【只提示固定文字，绝不打印密钥的任何部分】
if is_example_api_key(API_KEY):
    raise RuntimeError(EXAMPLE_KEY_MESSAGE)

# Flask 用这个密钥给浏览器 cookie 做签名，防止别人伪造 cookie 冒充成别的会话。
# 它必须是一串够长、够随机的字符，而且每个人都不一样——所以绝不能在代码里写死一个默认值。
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "")
if not SECRET_KEY:
    raise RuntimeError(
        "没有找到 FLASK_SECRET_KEY。请先设置环境变量 FLASK_SECRET_KEY，"
        "方法见 README.md 的「配置密钥」一节。"
    )

BASE_URL = "https://api.deepseek.com"                 # DeepSeek 的接口地址
MODEL = "deepseek-chat"                               # 要调用的模型名字

RETRIEVE_TOP_K = 3                                    # 每个问题检索几段资料。和评测器用同一个默认值

# 【出错时给用户看的话，永远只有这一句】
# 为什么不写 str(e)：异常原文里可能有内部路径、请求内容、甚至密钥片段。
# 出错的细节属于服务端的事，不该端到用户面前。
SAFE_ERROR_MESSAGE = "抱歉，这次没能处理你的问题，请稍后再试一次。"

# 【注意：这里不再有 SYSTEM_PROMPT 了】
# 以前是网页把 SYSTEM_PROMPT + 历史记录 + 问题一起发给模型，让它自由发挥。
# 现在 AI 的回答统一由 rag.py 负责：它有自己的提示词（要求模型只依据资料回答、
# 并按固定 JSON 格式输出），而且会逐条校验模型引用的来源是否真的存在。
# 提示词只保留一份——两边各写一份，迟早会悄悄漂移成两个不同的产品。

# ===================== 生产化配置 =====================
#
# 【总原则】凡是「换台机器就可能要改」的东西，都从环境变量读，代码里只留默认值。
# 而且默认值必须是【安全的那个】——忘了设不会出事，而不是忘了设才出事。

# 【为什么把调试模式默认关掉】
# 开发服务器自带调试器和交互式报错页。本地开发时很方便，
# 但一旦开到公网上，任何人都能通过报错页在服务器上执行代码——这是灾难级的。
# 所以默认关，只在本地显式设 FLASK_DEBUG=1 才打开。
DEBUG = os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _read_positive_int(name, default, minimum=1, maximum=None):
    """从环境变量读一个正整数。没设置就用默认值；设了但不合法就【明确报错】。

    【为什么非法时要报错，而不是悄悄退回默认值】
    「配置写错了但程序照常启动」是最难查的一类故障：
    你以为限流开着，其实没开；你以为上限是 500，其实是 5000。
    在【启动那一刻】就把问题喊出来，远好过让它带着错误配置悄悄跑一整天。

    错误信息里带上变量名和那个不合法的值。它们是配置项、不是密钥，
    写出来是安全的；不写反而没法排查。
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default                              # 没设置：最常见的情况，安静地用默认值

    text = str(raw).strip()
    try:
        value = int(text)
    except ValueError:
        raise RuntimeError(
            "环境变量 " + name + " 配置不合法：必须是整数，当前值是 " + repr(text)
            + "。请检查部署环境里的这个变量，或者删掉它改用默认值 " + str(default) + "。"
        )

    if value < minimum:
        raise RuntimeError(
            "环境变量 " + name + " 配置不合法：必须 >= " + str(minimum)
            + "，当前值是 " + str(value) + "。"
        )
    if maximum is not None and value > maximum:
        raise RuntimeError(
            "环境变量 " + name + " 配置不合法：必须 <= " + str(maximum)
            + "，当前值是 " + str(value) + "。"
        )
    return value


# 一个问题最多允许多少个字符。超长的问题不会被检索、不会送进模型、也不会写进数据库。
MAX_QUESTION_LENGTH = _read_positive_int("MAX_QUESTION_LENGTH", 500, minimum=1)

# 全站每天最多调用多少次模型（按 UTC 日期算），用来保护演示费用。
DAILY_API_LIMIT = _read_positive_int("DAILY_API_LIMIT", 50, minimum=1)

# 请求体大小上限（64 KB）。
# 【它和 MAX_QUESTION_LENGTH 管的不是一回事】
# MAX_QUESTION_LENGTH 管「问题有多少个字」，但要先把请求体读进来才知道有多少字。
# 别人完全可以不经过网页，直接往这个地址 POST 一个几百 MB 的请求体——
# 那在解析成表单之前就把内存吃光了。这一条是在【读请求体】那一步就拦下来。
MAX_CONTENT_LENGTH = 64 * 1024

# 给用户看的固定提示。都写死在这儿，不拼任何异常内容或内部信息。
TOO_LONG_MESSAGE = ("你问的问题太长了（最多 " + str(MAX_QUESTION_LENGTH)
                    + " 个字符），请精简一下再发。")
TOO_LARGE_MESSAGE = "你发送的内容太大了，请把问题缩短一些再试。"
QUOTA_MESSAGE = "今天的使用额度已经用完了，请明天再来。"
LOCKED_MESSAGE = "上一个问题还在处理中，等它回答完再问下一个吧。"


# ===================== 数据库位置 =====================
#
# 用「本文件所在位置」拼绝对路径，这样不管从哪个目录运行，记录都落在同一个文件里。
# 允许用环境变量 CHAT_DB_PATH 覆盖：自动化测试靠它把数据库指向临时文件，
# 从而绝不会碰到本地真实的聊天记录；Railway 上则把它指向持久卷（/data/chat.db）。
DB_PATH = os.environ.get("CHAT_DB_PATH") or os.path.join(BASE_DIR, "chat.db")


def _check_db_path(path):
    """启动时检查数据库所在目录能不能用。有问题就【立刻启动失败】，绝不将就。

    【为什么不自动建目录、也不退回项目目录】
    在 Railway 上，如果 CHAT_DB_PATH 设成了 /data/chat.db 却忘了挂持久卷，
    /data 这个目录根本不存在。这时候如果程序「贴心」地退回项目目录建一个库，
    应用会一切正常地跑起来——但数据其实写在容器的临时磁盘上，一重启就全没了。

    这种「看起来正常、实际没有持久化」的假成功，比直接启动失败危险得多：
    你会在真正丢了数据之后才发现。所以宁可开不起来。
    """
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise RuntimeError(
            "数据库所在目录不存在：" + directory + "\n"
            "（当前的 CHAT_DB_PATH = " + path + "）\n"
            "常见原因：部署时忘了把持久卷挂到这个目录，或者 CHAT_DB_PATH 写错了。\n"
            "本地开发一般不用设 CHAT_DB_PATH——不设就会自动用项目目录下的 chat.db。"
        )


_check_db_path(DB_PATH)

# ===================== 日志 =====================
#
# 【要记什么】应用启动、收到问题、RAG 决策、耗时、引用数、额度拒绝、内部错误。
# 【绝对不记什么】用户问题正文、AI 回答正文、任何密钥、原始 session_id、完整提示词、异常原文。
#
# 【为什么这条界线这么严】
# 日志会被复制、被上传、被同事随手贴进聊天窗口。用户问了什么、AI 答了什么，
# 属于用户的内容；密钥泄漏则是直接的安全事故。两者都不该出现在日志里。

logger = logging.getLogger("ai_tutor")
if not logger.handlers:                               # 避免被重复添加（模块被 reload 时会发生）
    _handler = logging.StreamHandler()                # 打到标准输出，Railway 会自己收集
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False                          # 不往根 logger 传，免得同一条打两遍

# 【日志字段白名单】
# 为什么用白名单而不是「记得别传正文」：靠人自觉的规矩迟早会破。
# 白名单反过来管——没登记过的字段名一律丢掉。
# 这样哪怕哪天有人手滑写了 log_event("x", question=用户原话)，
# 那行也只会打出一个光秃秃的事件名，正文根本进不去。
_SAFE_LOG_FIELDS = frozenset({
    "port", "workers", "db_file", "debug",            # 启动信息
    "question_length", "decision", "citation_count", "elapsed_ms",   # 每个问题的统计
    "diagnostic_code",                                # 固定诊断枚举：区分「模型主动判的」和「校验没过降级的」
    "context_messages", "context_chars",              # 上下文【只记条数和字符数】，绝不记正文
    "limit", "used",                                  # 额度
    "error_type", "stage",                            # 内部错误（只记类型，不记内容）
})


def _log_value(value):
    """把字段值收拾成一行安全的短文本。"""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > 40:
        text = text[:40] + "…"                        # 截断，防止有人塞一大段进来把日志撑爆
    return text


def log_event(event, **fields):
    """打一条结构化日志：事件名 + 若干 key=value。

    【为什么不直接 logger.info("用户问了 " + question)】
    那样写的话，每个调用点都要自己记得「什么能记、什么不能记」。
    用这个函数，规矩就写死在代码里：字段名不在白名单里自动丢掉，值一律截断成一行。
    """
    parts = []
    for key, value in fields.items():
        if key not in _SAFE_LOG_FIELDS:
            continue                                  # 没登记的字段名：不记。宁可少记，不可错记
        parts.append(key + "=" + _log_value(value))
    logger.info(event + ((" " + " ".join(parts)) if parts else ""))


app = Flask(__name__)                                 # 创建一个 Flask 应用对象
app.secret_key = SECRET_KEY                           # 交给 Flask，用来签名 cookie
app.permanent_session_lifetime = timedelta(days=30)   # cookie 有效期 30 天（浏览器关掉再打开，历史还在）
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH # 请求体超过 64KB 直接拒掉，不会先整个读进内存

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)   # 创建 OpenAI 客户端，整个程序共用一个

# 【防止重复提问的第一道防线（服务端）】
# 「锁」可以理解成一支只有一把的钥匙：谁先拿到，谁才能干活；其他人拿不到就得等着。
# 我们故意设成「拿不到就不等，直接放弃」——因为拿不到就说明已经有一个问题在处理了。
_lock = threading.Lock()


# ===================== 数据库相关的三个函数 =====================

def init_db():
    """建好数据库和表。可以重复调用——IF NOT EXISTS 保证「已经存在就什么都不做」。"""
    conn = sqlite3.connect(DB_PATH)                   # 打开（或创建）数据库文件
    try:
        with conn:                                    # with conn 是「事务」：中间没出错就自动提交，出错就自动撤销
            conn.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                       id         INTEGER PRIMARY KEY AUTOINCREMENT,
                       session_id TEXT    NOT NULL,
                       role       TEXT    NOT NULL,
                       content    TEXT    NOT NULL,
                       created_at TEXT    NOT NULL
                   )"""
            )
            # 建索引：让「按会话查消息」变快。数据少时看不出差别，但这是好习惯。
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
            )
            # 【每日额度表】一天一行，记这一天已经用掉多少次模型调用。
            #
            # 【为什么存在数据库里，而不是放在内存的一个变量里】
            # 内存里的计数器一重启就归零 —— 那等于「重启一下就刷新额度」，
            # 限制形同虚设。存进 SQLite 才能跨重启保留；配上持久卷，
            # 连重新部署都不会把额度冲掉。
            conn.execute(
                """CREATE TABLE IF NOT EXISTS api_usage (
                       day  TEXT    PRIMARY KEY,
                       used INTEGER NOT NULL DEFAULT 0
                   )"""
            )
    finally:
        conn.close()                                  # 【必须显式关闭】with conn 只管事务，不管关连接


def load_history(session_id):
    """读出某个会话的全部聊天记录，按时间先后返回。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        # 【安全】用 ? 占位符，而不是把变量拼进 SQL 字符串里。
        # 拼接的话，别人只要在会话 ID 里塞一段 SQL 就能把你的数据库读光——这叫 SQL 注入。
        # sqlite3 会自动帮我们把参数转义好，这是唯一正确的写法。
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()                                  # fetchall 把所有结果行一次性取出来
    finally:
        conn.close()
    # 把数据库的「行」转成模板需要的「字典列表」，模板那边一行都不用改
    return [{"role": r[0], "content": r[1]} for r in rows]


def load_recent_messages(session_id, limit=None):
    """读出某个会话【最近的几条】消息，按时间顺序返回。只给模型当短期上下文用。

    【它和 load_history 的区别，一定要分清楚】
      load_history()         —— 读【全部】记录，给页面展示用（用户要能往回翻）
      load_recent_messages() —— 读【最近几条】，只喂给模型当对话上下文

    合成一个函数是错的：展示要完整，上下文要克制，两者目的相反。
    合并之后迟早会有人为了「省一次查询」把整段历史塞进提示词。

    【会话隔离】WHERE session_id = ? —— 只可能读到当前这个会话的记录。
    绝不按 learner 读，也绝不读全库：那会把别人的对话拼进你的上下文里。
    （这个项目现在只有 session_id 这一层身份，还没有 learner 的概念，
     但话要说死 —— 免得以后加了 learner 就从这里开始漏。）

    【为什么按 id DESC 取完再翻回来】
    先取【最新】的几条（DESC + LIMIT），再把顺序翻正（老 → 新）。
    写成 ORDER BY id ASC LIMIT 6 取到的是【最早】的 6 条 —— 正好是错的那一头。

    【条数上限为什么从 rag 那边取】
    「上下文带几条」属于提示词预算，归 rag.py 管。这里只按它说的条数去读，
    两边不会各写一个 6、然后慢慢漂移成两个不同的数。
    """
    if limit is None:
        limit = rag.RECENT_MESSAGE_LIMIT                 # 单一事实来源：rag 里的常量

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def save_exchange(session_id, question, answer):
    """把「一问一答」两条一起写进数据库。"""
    now = datetime.now().isoformat(timespec="seconds")   # 比如 2026-09-15T14:04:02
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:                                    # 用事务包住两条插入：要么都成功，要么都不写。
                                                      # 避免出现「只有问题没有回答」这种半截记录
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "user", question, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "assistant", answer, now),
            )
    finally:
        conn.close()


def _utc_day(now=None):
    """当前 UTC 日期，形如 2026-09-22。额度按这个分组。"""
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y-%m-%d")


def reserve_api_call(limit, day=None):
    """尝试占用一次模型调用额度，返回 (是否拿到, 当天已用次数)。

    【为什么必须用事务，而且要用 BEGIN IMMEDIATE】
    想象额度只剩最后一次，两个请求同时到达。
    如果写成「先 SELECT 看剩多少，再 UPDATE 加一」，两个请求都会读到「还有 1 次」，
    然后各加一次 —— 一共调了 2 次，超了。

    这里用 BEGIN IMMEDIATE 开事务：它会【立刻拿写锁】，SQLite 会把两个事务彻底串行化，
    后到的那个必须等前一个提交完才能往下走。于是「读 → 判断 → 写」这三步变成一个
    不可分割的整体，最后一个名额绝不会被两个人同时抢到。

    【为什么超额不抛异常，而是返回 False】
    额度用完是【正常业务情况】，不是程序故障。用返回值表达，调用方好处理，
    也不会被外面那个大而全的 except 误判成「内部错误」。
    """
    if day is None:
        day = _utc_day()

    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)  # 自己管事务，关掉自动提交
    try:
        conn.execute("BEGIN IMMEDIATE")               # 立刻拿写锁，把并发请求串起来
        row = conn.execute("SELECT used FROM api_usage WHERE day = ?", (day,)).fetchone()
        used = row[0] if row else 0

        if used >= limit:                             # 名额已经用完
            conn.execute("ROLLBACK")
            return False, used

        if row:
            conn.execute("UPDATE api_usage SET used = used + 1 WHERE day = ?", (day,))
        else:
            conn.execute("INSERT INTO api_usage (day, used) VALUES (?, 1)", (day,))
        conn.execute("COMMIT")
        return True, used + 1
    except Exception:
        try:
            conn.execute("ROLLBACK")                  # 出错了要收尾，否则写锁会一直挂着
        except Exception:
            pass
        raise
    finally:
        conn.close()


# ===================== 把 RAG 结果变成「能显示的一段文字」 =====================

# 【general_answer 的标识文字】
# 用户必须一眼看出「这段回答不是从项目资料里来的」。
# 放在正文【前面】而不是后面 —— 免得用户读完了才发现它没有依据。
GENERAL_ANSWER_MARKER = "AI 通用知识回答"


def format_answer_with_sources(result):
    """把 rag.generate_answer() 的返回值，格式化成最终要显示、并存入数据库的文字。

    【为什么单独抽成一个纯函数】
    它没有副作用、只看传进来的字典，所以能单独测：喂一个结果字典进去，
    断言输出的文字对不对——不用起 Flask、不用连数据库、不用碰模型。
    这段格式化的逻辑是网页和"引用长什么样"之间的唯一约定，值得单独钉住。

    【三种情况的显示规则】

    | decision | 显示什么 |
    | --- | --- |
    | `answer` | 正文 + 「资料来源」 |
    | `general_answer` | **「AI 通用知识回答」标识 + 正文**（【绝不】加资料来源） |
    | `insufficient_evidence` / `refuse` | 只显示固定话术，**没有任何来源区** |

    【为什么 general_answer 必须单独标出来】
    它的依据是模型的通用知识，不是项目资料。
    不标的话，用户会以为它和 answer 一样有资料支撑 —— 那是误导。
    但反过来也【绝不能】给它加「资料来源」——那等于伪造依据，比不标更糟。

    【为什么用纯文本而不是 HTML】
    存进数据库的是文字，不是标签。模板那边会统一把 AI 的回复按 Markdown 渲染，
    这里只要给出清楚的分行结构就够了，不必也不该自己拼 HTML。
    """
    answer = (result.get("answer") or "").strip()      # 回答正文；缺了就当成空字符串，不让它变成 None
    decision = result.get("decision")
    citations = result.get("citations") or []

    # 【情况一】general_answer：有回答，但没有资料依据 —— 必须标出来
    if decision == "general_answer":
        return "【" + GENERAL_ANSWER_MARKER + "】\n\n" + answer

    # 【情况二】不是真的回答了，就没有「资料来源」这回事
    if decision != "answer" or not citations:
        return answer

    entries = []                                       # 一行一条来源
    for c in citations:
        source = str(c.get("source", "")).strip()
        heading = str(c.get("heading", "")).strip()
        if source and heading:
            entries.append("- " + source + " · " + heading)   # 文件名 · 二级标题
        elif source:
            entries.append("- " + source)                     # 只有文件名也照常显示

    # 【第二道判断】万一 citations 里的每一项都缺少文件名，就干脆不加这一节
    # ——宁可不显示，也不要留一个空的「资料来源：」标题在那儿
    if not entries:
        return answer

    return answer + "\n\n资料来源：\n" + "\n".join(entries)


init_db()                                             # 程序启动时先把表建好（已存在就什么都不做）
# 启动日志。【注意这里放的是模块级，不是 __main__ 里】
# 线上是 Gunicorn 加载 app:app，根本不会执行 __main__ 那段——
# 日志要是写在 __main__ 里，线上就一条启动记录都没有。
log_event("app_start", debug=DEBUG, db_file=DB_PATH)


# ===================== 页面 =====================

def _ensure_session_id():
    """拿到（必要时创建）本次请求所属的会话 ID。

    【会话隔离的关键】第一次访问时，给这个浏览器发一个随机 ID，存在「签名过的 cookie」里。
    之后这个浏览器的每次请求都会带着它，我们就知道「这段对话是谁的」。
    为什么用 uuid4 随机数：别人猜不到，也就没法看别人的对话。
    """
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex             # 32 位十六进制随机串，重复概率可以忽略
        session.permanent = True                       # 让 cookie 活 30 天，关掉浏览器再打开历史还在
    return session["sid"]


def _render(sid, question="", error="", status=200):
    """渲染首页。把「读历史 + 传参数」收在一处，几个分支共用。"""
    return render_template("index.html", history=load_history(sid),
                           question=question, error=error), status


# ===================== 健康检查 =====================

@app.route("/health")
def health():
    """给 Railway（以及任何运维工具）用的存活探针。

    【探针要的东西和网页完全不一样】它不需要好看的页面，只要一个能机器判断的答案：
    「这个实例现在到底能不能干活？」

    【为什么必须真的碰一下数据库】
    进程活着不等于服务可用。要是 /data 挂了、磁盘满了、数据库文件坏了，
    进程照样在跑，但每个请求都会失败。只看「进程还在」的探针会一直报健康，
    于是你要等到用户投诉才会发现。所以这里真的执行一次 SELECT 1。

    【这一条路由绝对不能碰的东西】
      · 不调用检索器 —— 探针会被频繁调用，不该有 CPU 开销
      · 不调用 DeepSeek —— 探针不该花钱，也不该依赖外部网络
      · 不读写聊天记录 —— 探针不是业务请求
      · 不创建用户会话 —— 探针没有浏览器，建会话只会白白攒下一堆垃圾数据
      · 不返回数据库路径、异常原文等内部信息 —— 这个地址是公开的
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        try:
            conn.execute("SELECT 1").fetchone()        # 真的查一下，证明数据库可访问
        finally:
            conn.close()
    except Exception as exc:
        # 【只记异常类型，不记原文】原文里可能有文件路径等内部信息。
        log_event("health_check_failed", error_type=type(exc).__name__)
        return jsonify({"status": "unhealthy"}), 503   # 固定的、安全的响应体

    return jsonify({"status": "ok"}), 200


# ===================== 请求体过大 =====================

@app.errorhandler(413)
def request_too_large(_error):
    """请求体超过 MAX_CONTENT_LENGTH 时走这里。

    这条挡的是「有人不经过网页，直接往这个地址 POST 一个几百 MB 的包」——
    那种请求在解析成表单之前就会把内存吃光。到这个位置只是拒绝，
    不需要也不该跟用户解释技术细节。
    """
    sid = _ensure_session_id()
    log_event("request_too_large")
    return _render(sid, error=TOO_LARGE_MESSAGE, status=413)


# ===================== 主页面 =====================

@app.route("/", methods=["GET", "POST"])              # 绑定到网站根地址 "/"，允许 GET 和 POST
def index():                                          # 用户每次打开页面或点发送都会执行它
    sid = _ensure_session_id()

    question = ""                                     # 学生这次问的话
    error = ""                                        # 要显示给用户的错误提示

    if request.method == "POST":
        # 【超大的请求已经在更早的地方被拦掉了】
        # 请求体超过 MAX_CONTENT_LENGTH 的会被上面的 413 处理函数接走，根本走不到这里。
        # 所以到这一步，question 一定是「装得进内存」的。
        question = request.form.get("question", "").strip()

        # 【第一道关：空问题】空字符串在 Python 里算 False。
        # 空输入不检索、不调模型、不写库、【也不消耗额度】。
        if question and len(question) > MAX_QUESTION_LENGTH:
            # 【第二道关：问题太长】同样什么都不做，只给一句友好的固定提示。
            # 注意：日志只记长度，不记问题原文。
            log_event("question_too_long",
                      question_length=len(question), limit=MAX_QUESTION_LENGTH)
            return _render(sid, question="", error=TOO_LONG_MESSAGE)

        if question:
            started = time.perf_counter()
            log_event("question_received", question_length=len(question))

            # 试着去拿那把锁。拿不到 = 已经有一个请求正在问 AI（多半是用户又按了一次回车）。
            if not _lock.acquire(blocking=False):
                return _render(sid, question=question, error=LOCKED_MESSAGE)

            # try ... finally：不管中间成功还是失败，最后都必须把锁还回去。
            # 失败时忘了还锁的话，这把锁就永远拿不回来了，之后所有提问都会被拒绝。
            try:
                # ---------- 第一步：找出相关的那几段资料 ----------
                # 【为什么检索要单独包一层 try】
                # 检索这一步自己炸了，属于程序缺陷，和「资料里确实没有」完全是两回事：
                #   · 检索出错 → 我们并不知道资料里到底有没有答案，写一句「资料里没有」就是撒谎，
                #                所以这里【不写数据库】，只给一句固定的抱歉提示；
                #   · 检索正常但没找到 → 那是正常结果，交给 rag 判断
                #                （可能是 general_answer，也可能是 refuse），并如实记进历史。
                try:
                    chunks = retriever.retrieve(question, top_k=RETRIEVE_TOP_K)
                except Exception as exc:
                    log_event("retrieval_failed", error_type=type(exc).__name__)
                    return _render(sid, question=question, error=SAFE_ERROR_MESSAGE)

                # ---------- 第二步：取【本会话】最近的几条消息，当短期上下文 ----------
                # 用户问完「帮我改一下这句」，接着问「再给一个例子」「为什么这样改」，
                # 这两句离开上一轮就无从作答。所以把最近的几条历史一起带给模型。
                #
                # 【为什么必须在这里取，不能更晚】
                # 当前这个问题【还没有存库】（存库在第四步）。所以此刻读到的历史，
                # 一定是「当前问题之前」的 —— 不会把当前问题重复当成上下文喂回去。
                #
                # 【只取本会话】load_recent_messages 里写死了 WHERE session_id = ?。
                # 别人的会话、全库，都不在读取范围内。
                #
                # 【取不到怎么办：降级成「没有上下文」，而不是让整个请求失败】
                # 上下文是锦上添花的东西。读不到它，答案会差一点，但仍然能用；
                # 为了它把用户刚问的问题整个丢掉，代价大于收益。
                # 异常只记类型，正文和原文都不进日志。
                try:
                    recent_messages = load_recent_messages(sid)
                except Exception as exc:
                    log_event("history_load_failed", error_type=type(exc).__name__)
                    recent_messages = []

                # ---------- 第三步：占一次额度 ----------
                # 【为什么现在是无条件占】
                # 以前是「没检索到资料就不调模型、不占额度」。
                # 现在不行了 —— 检索为空也可能要调模型（去判断这是不是一个正常的英语问题），
                # 所以【只要走到这一步，就一定会调用模型】，必须先占额度。
                #
                # 【哪些情况仍然不占】空输入、超长问题、检索异常 ——
                # 它们在更早的地方就返回了，根本走不到这里，自然不占。
                allowed, used = reserve_api_call(DAILY_API_LIMIT)
                if not allowed:
                    # 【额度用完】是正常业务情况，不是程序故障。
                    # 不调用模型、不写库，只给一句固定提示，也不记问题正文。
                    log_event("quota_rejected", limit=DAILY_API_LIMIT, used=used)
                    return _render(sid, question=question, error=QUOTA_MESSAGE)

                # ---------- 第四步：交给 RAG 生成，并校验它引用的来源 ----------
                # rag.generate_answer_with_context 内部已经兜住了模型的所有异常和不合规输出：
                # 坏 JSON、编造来源、空引用、接口报错……它一律安全降级成「不回答」。
                # 它还能处理「资料没支持、但属于正常英语问题」的情况（general_answer）。
                #
                # 【为什么用带上下文的那个入口】历史只在这里拼进那一次请求，
                # 而且是【唯一一次】模型调用 —— 不会为了「先理解指代再回答」调两次。
                #
                # 【额度什么时候算掉】只要走进了这一步，就已经占过一次了。
                # 哪怕模型调用失败、哪怕 rag 内部降级，这一次【照样计入】——
                # 因为它已经真实地发出去过一次外部请求，风险已经产生了。
                # 【为什么用带诊断的那个入口】
                # 日志里原来只有最终的 decision，看不出这个结论是模型主动给的，
                # 还是某一关校验没过、被安全降级下来的 —— 两者排查方向完全相反。
                # diagnostic_code 是一个【不含任何内容的固定短枚举】，
                # 只写日志，【绝不】进页面、也不进数据库（result 仍然只有三个键）。
                result, diagnostic_code = rag.generate_answer_with_context_and_diagnostics(
                    question, chunks, recent_messages, client, MODEL)

                # ---------- 第五步：格式化 + 存库 ----------
                # 只有真的回答了，才会在后面附上「资料来源」；
                # 拒答和证据不足不会凭空多出一个来源列表。
                answer = format_answer_with_sources(result)
                save_exchange(sid, question, answer)

                # 【日志只记统计量】决策、引用数、耗时、上下文的条数和字符数。
                # 上下文【只记这两个数字】，历史正文一个字都不许进日志 ——
                # 那是用户的内容，和 problem/answer 正文同一条红线。
                #
                # 这里的 context_for_log 是拿同一个纯函数算出来的，输入相同结果就一定相同，
                # 所以它反映的就是真正拼进提示词的那一份（不是「读到的条数」）。
                #
                # 【diagnostic_code 就是这把尺子】
                #   · ok                       → 这个 decision 是模型主动给的
                #   · invalid_json             → 模型没按 JSON 格式回，被降级
                #   · invalid_citations        → 引了本次片段之外的来源，被降级
                #   · citations_on_general_answer → 通用知识回答却带了引用，被降级
                #   · api_or_response_error    → 接口层就没成功，压根没拿到模型输出
                # 全部是写死的短枚举，不含模型原文、异常内容、问题、回答或上下文。
                context_for_log = rag.build_recent_context(recent_messages)
                log_event("rag_decision",
                          decision=result.get("decision"),
                          diagnostic_code=diagnostic_code,
                          citation_count=len(result.get("citations") or []),
                          context_messages=len(context_for_log),
                          context_chars=sum(len(e["text"]) for e in context_for_log),
                          elapsed_ms=int(round((time.perf_counter() - started) * 1000)))

                # 【为什么成功后要「跳转」】这叫 POST-Redirect-GET 模式。
                # 不跳转的话地址栏里留的是刚才那次 POST，用户一按 F5 浏览器就会问
                # 「要重新提交吗」，点「是」就又问了 AI 一遍。跳转后地址变回普通 GET。
                return redirect(url_for("index"))

            except Exception as exc:
                # 【绝不显示 str(e)，日志里也只记异常类型】
                # 异常原文里可能有内部路径、请求内容、甚至密钥片段。
                # 用户只需要知道「这次没成」。这里也【不写数据库】——
                # 存一条半截的对话，比不存更糟。
                log_event("internal_error", error_type=type(exc).__name__, stage="index")
                error = SAFE_ERROR_MESSAGE

            finally:
                _lock.release()                                              # 无论成功还是出错，都把锁还回去

    # 渲染页面。历史每次都从数据库读，所以重启程序也不会丢。
    return _render(sid, question=question, error=error)


if __name__ == "__main__":
    # 【只有本地开发才会走到这里】
    # 线上是 Gunicorn 直接加载 app:app，压根不会执行这段。
    # 也就是说：本地用的开发服务器和线上用的生产服务器是【两个不同】的东西，
    # 「本地能跑」不等于「线上能跑」——这一点部署文档里写清楚了。

    port = _read_positive_int("PORT", 5000, minimum=1, maximum=65535)
    log_event("dev_server_start", port=port, debug=DEBUG)

    # 【debug 默认关】调试模式自带交互式报错页，开到公网上等于让人在你服务器上执行代码。
    # 本地要开就设环境变量 FLASK_DEBUG=1。
    app.run(host="127.0.0.1", port=port, debug=DEBUG)
