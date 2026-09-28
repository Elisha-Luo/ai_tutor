# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

import os                                           # 读环境变量、拼路径
import uuid                                         # 生成随机的「会话 ID」
import time                                         # 给每个请求计时（日志里要记耗时）
import hmac                                         # CSRF token 用常数时间比较，别用 == 比
import secrets                                      # 生成 CSRF token 用的密码学安全随机源
import sqlite3                                      # Python 自带的轻量数据库，不用额外安装任何东西
import logging                                      # 打结构化日志
import threading                                    # 用它的「锁」防止同一时间处理两个请求
from datetime import datetime, timedelta, timezone  # datetime 记录消息时间；timedelta 设置 cookie 有效期；timezone 算 UTC 日期
from flask import Flask, render_template, request, redirect, url_for, session, jsonify   # session 给每个浏览器发身份标记；jsonify 拼 JSON 响应
from openai import OpenAI                           # 从 openai 库里导入 OpenAI 类（DeepSeek 兼容它的接口）
from env_utils import (load_dotenv, is_example_api_key, EXAMPLE_KEY_MESSAGE,   # 共用 .env 读取；示例值检测也在那边
                       is_example_secret, example_secret_message,
                       EXAMPLE_FLASK_KEY, EXAMPLE_PEPPER)
import retriever                                     # 本地检索层：把问题变成「最相关的几段资料」
import rag                                           # 生成与引用层：让模型照着资料回答，并校验它引用的来源
import profile_store                                 # 学习档案存储层：邀请码、学习者、会话绑定、偏好


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

# 【示例值同样要拦住】只查「非空」是不够的：.env.example 里那句 change-me-... 也是非空。
# 而它是一个【公开】的值 —— 谁都能用它伪造签名 cookie、冒充别的会话。
# 和上面那条一样：只看「是不是和公开模板长得一模一样」，不猜格式、不猜长度。
if is_example_secret(SECRET_KEY, EXAMPLE_FLASK_KEY):
    raise RuntimeError(example_secret_message("FLASK_SECRET_KEY"))

# 【邀请码的 pepper】把邀请码算成摘要时用的服务端密钥。
# 它决定了「数据库里那串摘要」能不能被反推出邀请码 —— 所以它和 API 密钥是同一级别的凭证：
# 只从环境变量读，绝不进代码、绝不进数据库、绝不进日志。
#
# 【为什么没设置就直接启动失败，而不是「先跑起来再说」】
# 如果真的允许空 pepper，那摘要就等于「没有密钥的 HMAC」—— 拿到数据库的人
# 可以拿一个常见词表挨个试，摘要一比就出来了。那等于邀请码明文存储。
# 宁可开不起来，也不要开着一个「看起来有保护、其实没有」的服务。
INVITE_CODE_PEPPER = os.environ.get("INVITE_CODE_PEPPER", "")
if not INVITE_CODE_PEPPER:
    raise RuntimeError(
        "没有找到 INVITE_CODE_PEPPER。请先设置环境变量 INVITE_CODE_PEPPER，"
        "方法见 README.md 的「配置密钥」一节。"
    )

# 【示例值同样要拦住】公开的 pepper 等于没有 pepper：
# 拿到数据库的人可以直接对摘要做离线爆破，把邀请码还原出来。
if is_example_secret(INVITE_CODE_PEPPER, EXAMPLE_PEPPER):
    raise RuntimeError(example_secret_message("INVITE_CODE_PEPPER"))

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


def _connect(timeout=5, isolation_level=""):
    """开一个数据库连接，并【每一次】都把外键检查打开。

    【为什么必须每次开】SQLite 出于历史原因，默认【不检查】外键 ——
    也就是说 profile_store 里写的那些 ON DELETE CASCADE 和 REFERENCES，
    不打开这个开关就只是一句注释：删掉一个学习者，他的偏好和会话绑定会留在库里，
    变成谁也访问不到的孤儿数据。
    PRAGMA 是【连接级】的，所以每个新连接都要设一次，不能只设一次了事。
    """
    conn = sqlite3.connect(DB_PATH, timeout=timeout, isolation_level=isolation_level)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

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
    "deleted_sessions", "deleted_messages",           # 清空全部数据时【只记删了多少条】
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
    conn = _connect()                                 # 打开（或创建）数据库文件，并打开外键检查
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

            # 【档案相关的四张表】交给 profile_store 建 —— 那些表的规则（邀请码摘要、
            # 白名单、会话唯一绑定）都归它管，schema 和读写代码放在一起才不会漂。
            # 这里只是「启动时确保它们存在」。
            profile_store.ensure_schema(conn)
    finally:
        conn.close()                                  # 【必须显式关闭】with conn 只管事务，不管关连接


def load_history(session_id):
    """读出某个会话的全部聊天记录，按时间先后返回。"""
    conn = _connect()
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

    conn = _connect()
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
    conn = _connect()
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

    conn = _connect(timeout=10, isolation_level=None)  # 自己管事务，关掉自动提交
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

class SessionUnverifiable(Exception):
    """【会话状态确认不了】读数据库失败时抛出，用来中断整个请求。

    【为什么必须是「中断」而不是「照旧用」】
    确认不了的意思就是：我们**不知道**手里这个 session_id 是不是已经失效了。
    如果照旧用它，而它其实已经被作废，那么这次提问的消息会写进【别人的旧会话】下 ——
    正是之前修掉的那个问题（以后清空那位学习者的数据时会误删这些无关消息）。
    所以这种情况只能【宁可停下，也不猜】：不提问、不兑换邀请码、不写任何东西。

    【为什么用异常】依赖会话身份的路由有六个（提问 / 邀请码 / 档案 / 清空 / 删除 / 413）。
    在一个地方抛、在一个地方收（见下面的 errorhandler），比每处都写一遍判断更不容易漏。
    """


SESSION_RETRY_MESSAGE = "暂时无法确认这个浏览器的状态，请稍后再试一次。"


@app.errorhandler(SessionUnverifiable)
def _handle_session_unverifiable(_error):
    """会话状态确认不了时的统一收尾。

    【三条纪律】
      · 只说一句固定的话，不含任何异常细节
      · 【不碰数据库】—— 数据库这会儿正不正常还不知道，别再往上压请求
        （所以这里直接把 history 传空、learner_id 传 None，而不是走 _render）
      · 返回 503：这不是「正常响应」，而是「此刻没法服务」，让调用方知道可以重试
    """
    log_event("session_unverifiable")
    return render_template("index.html", history=[], question="",
                           error=SESSION_RETRY_MESSAGE, learner_id=None), 503


def _ensure_session_id():
    """拿到本次请求应当使用的会话 ID（必要时创建，或者【轮换】）。

    【会话隔离的关键】第一次访问时，给这个浏览器发一个随机 ID，存在「签名过的 cookie」里。
    之后这个浏览器的每次请求都会带着它，我们就知道「这段对话是谁的」。
    为什么用 uuid4 随机数：别人猜不到，也就没法看别人的对话。

    【会话轮换：手里的会话已经失效了，就换一个新的】
    什么算「失效」？这个 session_id 曾经绑过某个学习者，后来那个码被作废了
    （learner_sessions 那一行打了 revoked_at 标记）。此时必须换一个新的 session_id：

      1. **不换的话，匿名提问会继续挂在这个旧 id 下。**
         而旧 id 仍然关联着原来那个学习者 —— 将来他要求「清空全部个人数据」时，
         我们会按 learner_sessions 找会话、连带删消息，于是【这些无关的匿名消息也会被删掉】。
      2. **不换的话，输入一张新邀请码会撞上那条「已失效但仍属于别人」的绑定**，
         绑定被拒绝，用户白白消耗掉一张码（这个坑在 profile_store.redeem_and_bind 里也堵了一道）。

    【为什么不干脆把旧行删掉】那正是之前的 bug：删了行，「这个会话属于谁」也就丢了，
    以后清空数据时找不到它名下的旧聊天记录。所以旧行【留着】，
    只是这个浏览器以后不再用它 —— 新消息落在新 id 下，两者清清楚楚分开。

    【查不出来时抛异常，不返回旧 sid】
    会话状态确认不了，就等于「不知道这个 id 还能不能用」。
    照旧用它有可能把消息写进别人已被作废的旧会话 —— 所以整个请求停在这里，
    由 SessionUnverifiable 的 errorhandler 给一句重试提示。详见那个异常类的说明。
    """
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex             # 32 位十六进制随机串，重复概率可以忽略
        session.permanent = True                       # 让 cookie 活 30 天，关掉浏览器再打开历史还在

    sid = session["sid"]

    # 查一次：这个会话的绑定是不是已经失效了
    try:
        conn = _connect()
        try:
            revoked = profile_store.session_is_revoked(conn, sid)
        finally:
            conn.close()
    except Exception as exc:
        # 【查不出来就停下，绝不照旧用这个 sid】
        # 早期版本这里是「当成没失效，返回旧 sid」—— 那是个真窟窿：
        # 如果这个 sid 其实已经被作废，提问会照常成功，新消息就写进了别人的旧会话，
        # 以后清空那位学习者的数据时会被一起删掉。
        # 确认不了就什么都不能做：抛出去，由 errorhandler 统一给一句重试提示。
        log_event("session_check_failed", error_type=type(exc).__name__)
        raise SessionUnverifiable() from None

    if not revoked:
        return sid

    # 失效了 → 换一个新的会话 ID（旧的留在库里，关联不动）
    session["sid"] = uuid.uuid4().hex
    session.permanent = True
    log_event("session_rotated")
    return session["sid"]


def _render(sid, question="", error="", status=200):
    """渲染首页。把「读历史 + 传参数」收在一处，几个分支共用。"""
    return render_template("index.html", history=load_history(sid),
                           question=question, error=error,
                           learner_id=_learner_id_for(sid)), status


# ===================== CSRF 保护 =====================
#
# 【要防的是什么】假设用户登录着我们的站点，同时又打开了一个恶意页面。
# 那个页面可以偷偷向我们的地址提交一个表单 —— 浏览器会自动带上用户的 cookie，
# 于是「这个请求是用户自己发的」看起来就成立了。
# 这叫 CSRF（跨站请求伪造）。防御办法：让每个表单都带一个【恶意页面猜不到】的随机值。
#
# 【为什么 token 放在 Flask 的 session 里】
# 这个 session 是「签名过的 cookie」—— 用户能看到它，但改不了它。
# 恶意页面既读不到这个值（浏览器不允许跨站读 cookie），也伪造不出签名。
# 所以它拿不出一个合法的 token，请求就被挡下了。
#
# 【为什么比较要用 hmac.compare_digest 而不是 ==】
# 用 == 比较字符串时，Python 会在第一个不同的字符处提前返回 ——
# 耗时随「猜对了几个前缀」而变化，理论上能被逐字节试出来。
# compare_digest 无论内容如何都花同样的时间。

def _csrf_token():
    """拿到本次会话的 CSRF token；第一次用就生成一个。"""
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def _csrf_ok(submitted):
    """校验表单提交上来的 token。不通过一律拒绝。

    【为什么两边都要 .encode()】这是个真实的坑：
    hmac.compare_digest 对【字符串】只接受纯 ASCII ——
    传进来的 token 里只要有一个中文字符，它会直接抛 TypeError，
    于是「伪造的 token」变成了一次 500 错误（还顺带把堆栈暴露在日志里），
    而不是一次干净的拒绝。转成 bytes 之后任何输入都能安全比较。
    """
    expected = session.get("csrf")
    if not expected or not submitted:
        return False
    return hmac.compare_digest(str(expected).encode("utf-8"),
                               str(submitted).encode("utf-8"))


CSRF_MESSAGE = "页面可能已过期，请刷新后重试。"


# ===================== 邀请码尝试限速 =====================
#
# 【为什么要限速】邀请码虽然很长（约 32 个字符），但如果可以无限次试，
# 攻击者就能一直猜。限速让「猜」这件事在时间上变得不划算。
#
# 【为什么只数失败的尝试】成功的尝试不该把人锁在门外 ——
# 正常用户输对一次就进去了，之后不该再受限。
#
# 【存在内存里意味着什么，要说清楚】
#   · 重启进程就清零（和那把防重复的锁一样）
#   · 多 worker 部署时各算各的
# 所以它是「抬高尝试成本」的保险丝，不是生产级风控。
# 真要对外提供服务，应该像每日额度那样落库（这里没做，因为本轮范围不包含它）。
INVITE_MAX_FAILURES = 10           # 窗口内最多允许失败几次
INVITE_WINDOW_SECONDS = 600        # 窗口长度：10 分钟
_invite_failures = {}              # {ip: [失败时间戳, ...]}


def _prune_invite_failures(ip, now=None):
    """把窗口外的失败记录清掉，返回还留在窗口内的那些。"""
    moment = now if now is not None else time.time()
    kept = [t for t in _invite_failures.get(ip, []) if moment - t < INVITE_WINDOW_SECONDS]
    if kept:
        _invite_failures[ip] = kept
    else:
        _invite_failures.pop(ip, None)             # 空列表不留在字典里，免得越攒越多
    return kept


def _invite_rate_limited(ip):
    """这个来源现在还能继续试吗？"""
    return len(_prune_invite_failures(ip)) >= INVITE_MAX_FAILURES


def _record_invite_failure(ip):
    """记一次失败尝试。

    【先剪枝再加】不先清掉过期的记录，窗口就变成了「从第一次失败起累计」，
    失败十次之后即使过了一天也还是被锁着。
    """
    kept = _prune_invite_failures(ip)              # 先把这个来源过期的失败记录清掉
    kept.append(time.time())                       # 再把这一次记上
    _invite_failures[ip] = kept


INVITE_INVALID_MESSAGE = "邀请码无效。"
INVITE_REVOKED_MESSAGE = "这个邀请码已失效。"
INVITE_EMPTY_MESSAGE = "请输入邀请码。"
INVITE_RATE_MESSAGE = "尝试次数太多了，请过一会儿再试。"
# 【为什么要有这句话】这个浏览器已经绑着一个学习档案了，不能再改用另一个人的码。
# 说清楚「码本身没问题，是这个浏览器已经有身份了」，并且给出【真的做得到】的下一步。
#
# 【这段文案改过两轮，两个坑都记在这里】
#   ❌ 第一版：「先在档案页清空当前档案」—— 那只清偏好，【不会解绑】，
#      照着做一遍绑定还在，再输码还是被拒。
#   ❌ 第二版：「或者做一次清空全部个人数据」—— 事实没错，但把它摆成「两条路之一」，
#      等于【诱导用户为了换号去点一个不可撤销的删除】。换身份不需要摧毁自己的数据。
#
# 【现在的口径】
#   · 先说实话：这个浏览器【不能无损换绑】—— 它已经属于某个学习者了
#   · 想用另一张码：换一个**真的不共享 cookie** 的环境（无痕窗口 / 另一个浏览器）
#   · 顺手堵掉「清空学习档案」这个错误猜测（它只清偏好，不解绑）
#   · 全清只作为【另一件事】被提及，并如实写全它的代价 + 明确说不要为换号去点
INVITE_BOUND_ELSEWHERE_MESSAGE = (
    "这个浏览器已经绑定了一个学习档案，不能再绑另一张邀请码 —— "
    "它也没法无损换成另一个身份。"
    "如果你想用的是另一张邀请码，请换一个不共享 cookie 的浏览器环境"
    "（比如无痕窗口，或者另一个浏览器）再输码。"
    "（在档案页「清空学习档案」只清偏好，不能解除绑定。）"
    "「清空全部个人数据」是另一回事：那是一次不可撤销的删除，"
    "会连带删掉全部聊天记录、学习者身份和会话绑定，并作废已有邀请码 —— "
    "不要为了换一个身份去点它。"
)


# ===================== 档案相关的读操作 =====================

def _learner_id_for(sid):
    """当前会话绑定的学习者编号；没有绑定返回 None（= 匿名访客）。"""
    conn = _connect()
    try:
        return profile_store.learner_id_for_session(conn, sid)
    finally:
        conn.close()


def _preferences_for_prompt(learner_id):
    """取「要拼进提示词的那份偏好」。返回 None 表示【这次不带档案这段】。

    【三种情况，行为故意不一样】
      · 匿名访客（learner_id 是 None）→ 返回 None
        提示词里【完全不会】多出「学习者档案」那一段 —— 对他而言，
        行为和加这个功能之前一模一样，一个字符都没变。
      · 绑定了但还没填档案 → 给一套默认值
        他已经在用档案功能了，总得有个「按什么方式回答」的起点。
      · 填过 → 用他填的

    【和档案页显示用的那份不是一回事】
    档案页要能说「你还没设置过」，所以那边用的是 profile_store.get_preferences()
    的原样返回（没填过是 None）。这里要的是「给模型的输入」，
    所以匿名 → None（不带这一段），已绑定未填 → 默认值。
    """
    if learner_id is None:
        return None                                   # 匿名访客：不带档案

    conn = _connect()
    try:
        stored = profile_store.get_preferences(conn, learner_id)
    finally:
        conn.close()
    return dict(stored) if stored else dict(profile_store.DEFAULT_PREFERENCES)


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
        conn = _connect(timeout=5)
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
    learner_id = _learner_id_for(sid)                 # 这个浏览器有没有绑定学习档案（None = 匿名）

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

                # ---------- 取这位学习者的【学习档案】（没档案就用默认的一套） ----------
                # 和上下文一样，读失败不能让整个提问失败：档案只是「怎么说」的偏好，
                # 读不到就按默认方式回答，用户照样得到答案。异常只记类型。
                try:
                    preferences = _preferences_for_prompt(learner_id)
                except Exception as exc:
                    log_event("preferences_load_failed", error_type=type(exc).__name__)
                    preferences = None            # 读不到就当没有档案，绝不猜用户填了什么

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
                # 【为什么用带档案 + 带诊断的那个入口】
                # 档案：让模型按这位学习者的水平、长度、语言偏好来回答。
                #      没有档案时会传入一套默认值（见 _preferences_for_prompt），
                #      所以匿名访客照常能用，行为和不带档案时完全一致。
                # 诊断：日志里原来只有最终的 decision，看不出这个结论是模型主动给的，
                #      还是某一关校验没过、被安全降级下来的 —— 两者排查方向完全相反。
                #      diagnostic_code 是【不含任何内容的固定短枚举】，只写日志，
                #      【绝不】进页面、也不进数据库（result 仍然只有三个键）。
                result, diagnostic_code = rag.generate_answer_with_context_and_profile(
                    question, chunks, recent_messages, preferences, client, MODEL)

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


# ===================== 邀请码入口 =====================
#
# 【整个流程】用户拿到一个邀请码 → 在这里输进来 → 换到一个 learner_id →
# 这个浏览器的 session 就绑在他身上了 → 之后他填的档案、问的话都记在这个 id 下。
#
# 【为什么码走 POST 而不是放在 URL 里】
# URL 会被浏览器历史、代理日志、服务器访问日志记下来。
# 邀请码等于访问某个用户全部学习数据的凭证，绝不能出现在 URL 里。

@app.route("/invite", methods=["GET", "POST"])
def invite():
    sid = _ensure_session_id()

    if request.method == "GET":
        return _render_invite(sid)

    # ---------- 第一道关：CSRF ----------
    if not _csrf_ok(request.form.get("csrf_token")):
        log_event("csrf_rejected", stage="invite")
        return _render_invite(sid, error=CSRF_MESSAGE)

    # ---------- 第二道关：限速（防暴力枚举） ----------
    ip = request.remote_addr or "unknown"
    if _invite_rate_limited(ip):
        log_event("invite_rate_limited", limit=INVITE_MAX_FAILURES)
        return _render_invite(sid, error=INVITE_RATE_MESSAGE)

    code = (request.form.get("invite_code") or "").strip()
    if not code:
        return _render_invite(sid, error=INVITE_EMPTY_MESSAGE)

    # ---------- 第三道关：兑换 + 绑定（同一事务，要么都成要么都不做） ----------
    # 【为什么不再分两步调用】旧写法是 redeem_invite() 先提交、再 link_session()。
    # 第二步可能失败（比如这个浏览器已经绑着别人了），而第一步已经落库 ——
    # 结果是：码被白白消耗、建出一个谁都进不去的孤儿学习者、网页还报「成功」。
    # 现在交给 redeem_and_bind：绑定失败就整体回滚，码仍然可用。
    conn = _connect()
    try:
        outcome, learner_id = profile_store.redeem_and_bind(
            conn, code, INVITE_CODE_PEPPER, sid)
    finally:
        conn.close()

    if outcome in (profile_store.REDEEM_INVALID, profile_store.REDEEM_REVOKED,
                   profile_store.REDEEM_BIND_REFUSED):
        _record_invite_failure(ip)
        # 【日志里绝不能出现邀请码，摘要也不行】只记「失败」和是哪一类失败。
        log_event("invite_rejected", stage=outcome)
        if outcome == profile_store.REDEEM_INVALID:
            return _render_invite(sid, error=INVITE_INVALID_MESSAGE)
        if outcome == profile_store.REDEEM_REVOKED:
            return _render_invite(sid, error=INVITE_REVOKED_MESSAGE)
        return _render_invite(sid, error=INVITE_BOUND_ELSEWHERE_MESSAGE)

    # 【只有真的绑上了才记「接受」】上面那条分支里的兑换已经被回滚，码没有消耗。
    log_event("invite_accepted", stage=outcome)
    return redirect(url_for("index"))                 # PRG：避免刷新时重复提交邀请码


def _render_invite(sid, error="", status=200):
    """渲染邀请码输入页。"""
    return render_template("invite.html", error=error,
                           csrf_token=_csrf_token()), status


# ===================== 学习档案页 =====================

def _form_choice(form, name):
    """从表单里取一个可选项。空字符串 → None（= 用户没选）。

    【为什么不在这里校验白名单】非法值要原样交给存储层，
    让它拒绝并返回 False —— 那样用户会看到一句「选项不合法」，
    而不是被悄悄改成默认值还以为设置成功了。
    """
    value = (form.get(name) or "").strip()
    return value or None


@app.route("/profile", methods=["GET", "POST"])
def profile():
    sid = _ensure_session_id()
    learner_id = _learner_id_for(sid)

    # 【第一道门：没绑定学习档案的人，档案页不对他开放】
    # 档案属于某个学习者，匿名访客没有可看的东西 —— 直接引导去输邀请码。
    if learner_id is None:
        log_event("profile_denied", stage="anonymous")
        return redirect(url_for("invite"))

    if request.method == "GET":
        return _render_profile(sid, learner_id)

    # 【第二道门：CSRF】改档案是「会修改数据的操作」，必须带 token
    if not _csrf_ok(request.form.get("csrf_token")):
        log_event("csrf_rejected", stage="profile")
        return _render_profile(sid, learner_id, error=CSRF_MESSAGE)

    conn = _connect()
    try:
        saved = profile_store.save_preferences(
            conn, learner_id,
            level_code=_form_choice(request.form, "level_code"),
            # 「不确定」是个单独的勾选项。用户没确认过水平时，模型那边会被提醒别太当真
            level_uncertain=(request.form.get("level_uncertain") == "1"),
            language_mode=_form_choice(request.form, "language_mode"),
            length_mode=_form_choice(request.form, "length_mode"),
            goal_code=_form_choice(request.form, "goal_code"),
            focus_code=_form_choice(request.form, "focus_code"),
        )
    finally:
        conn.close()

    if not saved:
        log_event("profile_rejected", stage="invalid")
        return _render_profile(sid, learner_id, error=PROFILE_INVALID_MESSAGE)

    log_event("profile_saved")                        # 只记「存了」，不记存了什么
    return redirect(url_for("profile"))               # PRG：F5 不会重复提交


PROFILE_INVALID_MESSAGE = "选项不合法，档案没有保存，请重新选择。"
CLEAR_CONFIRM_MESSAGE = "请先勾选上面的确认框，再点清空。"


@app.route("/profile/clear", methods=["POST"])
def profile_clear():
    """清空【这位学习者自己】的偏好。

    【它清什么、不清什么，见 profile_store.clear_preferences 的说明】
    简单说：只清偏好那五个选项；不动学习者身份、不动邀请码、不动聊天记录。
    所以清完之后他还能用同一张码继续用，只是从「未设置」重新开始。
    """
    sid = _ensure_session_id()
    learner_id = _learner_id_for(sid)

    # 【第一道门：没有绑定的人不能清】和档案页一样
    if learner_id is None:
        log_event("profile_denied", stage="anonymous")
        return redirect(url_for("invite"))

    # 【第二道门：CSRF】清空是「会修改数据的操作」，而且不可撤销，必须有 token
    if not _csrf_ok(request.form.get("csrf_token")):
        log_event("csrf_rejected", stage="profile_clear")
        return _render_profile(sid, learner_id, error=CSRF_MESSAGE)

    # 【第三道门：二次确认】必须显式勾选。
    # 前端还有一个 confirm() 弹窗，但那只是体验；真正说了算的是这一行 ——
    # 前端可以被绕过（禁用 JS、直接发请求），服务端这一关不能。
    if request.form.get("confirm") != "yes":
        log_event("profile_clear_rejected", stage="no-confirm")
        return _render_profile(sid, learner_id, error=CLEAR_CONFIRM_MESSAGE)

    conn = _connect()
    try:
        profile_store.clear_preferences(conn, learner_id)
    finally:
        conn.close()

    # 【日志只记「清过了」】不记清掉了哪几项、更不记清之前的值
    log_event("profile_cleared")
    return redirect(url_for("profile"))               # PRG：F5 不会重复提交


DELETE_CONFIRM_MESSAGE = "请先勾选上面的确认框，再点删除。"
DELETE_BUSY_MESSAGE = "有提问正在处理中，请等它结束之后再删除。"

# 等那把锁最多等多久（秒）。
# 【为什么要等，而不是直接拒绝】删除是不可逆的，用户是认真按下的；
# 「正在回答另一个问题，稍后再试」会让人以为没成功、反复按。
# 等它做完再删更符合直觉：反正那次提问写进去的消息，紧接着也会被删掉。
# 上限是为了不出「一直转圈」：模型调用本身有超时，正常几秒内就会释放。
DELETE_LOCK_TIMEOUT = 10


@app.route("/profile/delete", methods=["POST"])
def profile_delete():
    """清空这位学习者在本系统里的全部个人数据。**不可撤销。**

    【四道关，一道都不能少】
      ① 必须是已绑定的学习者（匿名访客没有东西可删，也不该能触发删除）
      ② CSRF token 必须对（删除请求不能是别的网站伪造出来的）
      ③ 必须收到用户【真的勾选】提交上来的 confirm=yes
      ④ 必须拿到那把处理提问的锁 —— 见下面那段
    """
    sid = _ensure_session_id()
    learner_id = _learner_id_for(sid)

    if learner_id is None:
        log_event("profile_denied", stage="anonymous")
        return redirect(url_for("invite"))

    if not _csrf_ok(request.form.get("csrf_token")):
        log_event("csrf_rejected", stage="profile_delete")
        return _render_profile(sid, learner_id, error=CSRF_MESSAGE)

    # 【为什么必须是表单里真的带上来的值】页面上这个 confirm 由复选框自己提交：
    # 没勾选时浏览器根本不会带上这个字段。所以这里收到的 yes，就是用户真的勾了。
    if request.form.get("confirm") != "yes":
        log_event("profile_delete_rejected", stage="no-confirm")
        return _render_profile(sid, learner_id, error=DELETE_CONFIRM_MESSAGE)

    # 【必须和提问共用同一把锁 —— 这是防「删完又冒出来」的关键】
    # 一个提问的处理过程是：检索 → 调模型 → 格式化 → 写库。
    # 如果不加这道锁，就可能出现：这边正在调模型，那边把数据全删了，
    # 然后模型返回、那个请求把消息【又写回数据库】—— 用户以为删干净了，其实没有。
    # 提问路径从头到尾都持有这把锁（写在 finally 里释放），所以：
    #   · 删除会等正在进行的提问彻底结束（连同它的写库）才开始
    #   · 删除期间的新提问拿不到锁，会被挡成「上一个问题还在处理中」
    # 等到删除提交时，不可能再有旧请求往回写。
    if not _lock.acquire(timeout=DELETE_LOCK_TIMEOUT):
        log_event("profile_delete_blocked", stage="busy")
        return _render_profile(sid, learner_id, error=DELETE_BUSY_MESSAGE)

    try:
        conn = _connect()
        try:
            stats = profile_store.clear_all_data(conn, learner_id)
        finally:
            conn.close()
    except Exception as exc:
        # 事务里任何一步失败都会整体回滚，所以这里什么都不用修补。
        # 只记异常类型 —— 原文可能带路径等信息。
        log_event("profile_delete_failed", error_type=type(exc).__name__)
        return _render_profile(sid, learner_id, error=SAFE_ERROR_MESSAGE)
    finally:
        _lock.release()

    # 【只记数量，不记内容】删掉了多少条记录属于统计量；聊了什么都不记。
    log_event("profile_deleted",
              deleted_sessions=stats["sessions"],
              deleted_messages=stats["messages"])

    # 这个浏览器现在已经不是那位学习者了（绑定被删），回首页就是匿名状态。
    return redirect(url_for("index"))


def _render_profile(sid, learner_id, error="", status=200):
    """渲染档案页。

    【传两份东西给模板】
      · stored —— 数据库里【真正存着】的偏好；None 表示用户还没设置过
      · current —— 表单要预选的那份；没设置过就先用默认值填上
    这样页面既能如实说「你还没设置过」，又能让表单有个合理的起点。
    """
    conn = _connect()
    try:
        stored = profile_store.get_preferences(conn, learner_id)
    finally:
        conn.close()

    return render_template(
        "profile.html",
        preferences=stored,                            # None = 还没设置过
        current=stored or dict(profile_store.DEFAULT_PREFERENCES),
        is_set=stored is not None,
        learner_id=learner_id,
        error=error,
        csrf_token=_csrf_token(),
        level_options=profile_store.LEVEL_OPTIONS,
        language_options=profile_store.LANGUAGE_OPTIONS,
        length_options=profile_store.LENGTH_OPTIONS,
        goal_options=profile_store.GOAL_OPTIONS,
        focus_options=profile_store.FOCUS_OPTIONS,
    ), status


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
