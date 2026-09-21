# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

import os                                           # 读环境变量、拼路径
import uuid                                         # 生成随机的「会话 ID」
import sqlite3                                      # Python 自带的轻量数据库，不用额外安装任何东西
import threading                                    # 用它的「锁」防止同一时间处理两个请求
from datetime import datetime, timedelta            # datetime 记录消息时间；timedelta 设置 cookie 有效期
from flask import Flask, render_template, request, redirect, url_for, session   # session 用来给每个浏览器发一个签名过的身份标记
from openai import OpenAI                           # 从 openai 库里导入 OpenAI 类（DeepSeek 兼容它的接口）
from env_utils import load_dotenv                   # 共用同一份 .env 读取逻辑（实现和「为什么」都在 env_utils.py）
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

# 聊天记录数据库文件。用「本文件所在位置」拼绝对路径，这样不管从哪个目录运行，记录都落在同一个文件里。
# 允许用环境变量 CHAT_DB_PATH 覆盖——自动化测试就是靠它把数据库指向临时文件，
# 从而绝不会碰到你本地的真实聊天记录。
DB_PATH = os.environ.get("CHAT_DB_PATH") or os.path.join(BASE_DIR, "chat.db")

app = Flask(__name__)                                 # 创建一个 Flask 应用对象
app.secret_key = SECRET_KEY                           # 交给 Flask，用来签名 cookie
app.permanent_session_lifetime = timedelta(days=30)   # cookie 有效期 30 天（浏览器关掉再打开，历史还在）

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


# ===================== 把 RAG 结果变成「能显示的一段文字」 =====================

def format_answer_with_sources(result):
    """把 rag.generate_answer() 的返回值，格式化成最终要显示、并存入数据库的文字。

    【为什么单独抽成一个纯函数】
    它没有副作用、只看传进来的字典，所以能单独测：喂一个结果字典进去，
    断言输出的文字对不对——不用起 Flask、不用连数据库、不用碰模型。
    这段格式化的逻辑是网页和"引用长什么样"之间的唯一约定，值得单独钉住。

    【为什么只有 answer 才加「资料来源」】
    refuse 和 insufficient_evidence 本来就没有给出回答，更没有出处可给。
    给它们硬加一个来源列表，等于伪造依据——那正是整个 RAG 要防的事。
    所以这里先看 decision，不是 answer 就直接返回原话。

    【为什么用纯文本而不是 HTML】
    存进数据库的是文字，不是标签。模板那边会统一把 AI 的回复按 Markdown 渲染，
    这里只要给出清楚的分行结构就够了，不必也不该自己拼 HTML。
    """
    answer = (result.get("answer") or "").strip()      # 回答正文；缺了就当成空字符串，不让它变成 None
    citations = result.get("citations") or []

    # 【第一道判断】不是真的回答了，就没有「资料来源」这回事
    if result.get("decision") != "answer" or not citations:
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


# ===================== 页面 =====================

@app.route("/", methods=["GET", "POST"])              # 装饰器：把下面这个函数绑定到网站根地址 "/"，并允许 GET 和 POST
def index():                                          # 首页处理函数：用户每次打开页面或点发送都会执行它
    # 【会话隔离的关键】第一次访问时，给这个浏览器发一个随机 ID，存在「签名过的 cookie」里。
    # 之后这个浏览器的每次请求都会带着它，我们就知道「这段对话是谁的」。
    # 为什么用 uuid4 随机数：别人猜不到，也就没法看别人的对话。
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex             # 32 位十六进制随机串，重复概率可以忽略
        session.permanent = True                       # 让 cookie 活 30 天，关掉浏览器再打开历史还在
    sid = session["sid"]                              # 取出本次请求所属的会话 ID

    question = ""                                     # 准备一个变量存学生这次问的话，先设为空字符串
    error = ""                                        # 准备一个变量存错误信息，先设为空字符串

    if request.method == "POST":                      # 如果这次访问是「提交表单」（POST），而不是单纯打开网页（GET）
        question = request.form.get("question", "").strip()   # 从表单里取问题，去掉前后空格

        # 【空问题】空字符串在 Python 里算 False，所以这一步顺带挡掉了空输入：
        # 既不会问 AI，也不会往数据库里写任何东西。
        if question:

            # 试着去拿那把锁。拿不到 = 已经有一个请求正在问 AI 了（多半是用户又按了一次回车）。
            if not _lock.acquire(blocking=False):
                return render_template(
                    "index.html", history=load_history(sid), question=question,
                    error="上一个问题还在处理中，等它回答完再问下一个吧。",
                )

            # try ... finally：不管中间成功还是失败，最后都必须把锁还回去。
            # 如果失败时忘了还锁，这把锁就永远拿不回来了，之后所有提问都会被拒绝，页面彻底用不了。
            try:
                # ---------- 第一步：找出相关的那几段资料 ----------
                # 【为什么检索要单独包一层 try】
                # 检索这一步自己炸了，属于程序缺陷，和「资料里确实没有」完全是两回事。
                # 两者必须分开处理：
                #   · 检索出错 → 我们并不知道资料里到底有没有答案，写一句「资料里没有」就是撒谎，
                #                所以这里【不写数据库】，只给一句固定的抱歉提示；
                #   · 检索正常但没找到 → 那是正常结果，交给 rag 拒答，并如实记进历史。
                try:
                    chunks = retriever.retrieve(question, top_k=RETRIEVE_TOP_K)
                except Exception:
                    return render_template(
                        "index.html", history=load_history(sid), question=question,
                        error=SAFE_ERROR_MESSAGE,
                    )

                # ---------- 第二步：交给 RAG 生成，并校验它引用的来源 ----------
                # rag.generate_answer 内部已经兜住了模型的所有异常和不合规输出：
                # 坏 JSON、编造来源、空引用、接口报错……它一律安全降级成「不回答」。
                # 所以拿回来的永远是一个安全的、格式固定的结果，这里不需要再判一次。
                #
                # 【为什么没有把历史记录发给模型】
                # rag 只按「当前这个问题」检索和回答。多轮指代（比如「那它呢？」）
                # 这一版还不支持，README 里已如实写明——宁可把边界说清楚，
                # 也不要让用户误以为它能听懂上下文。
                result = rag.generate_answer(question, chunks, client, MODEL)

                # ---------- 第三步：格式化成能显示的文字 ----------
                # 只有真的回答了，才会在后面附上「资料来源」；
                # 拒答和证据不足不会凭空多出一个来源列表。
                answer = format_answer_with_sources(result)

                # ---------- 第四步：存库 ----------
                # 存的是上面那段【格式化之后】的文字，所以历史记录里也带着来源，
                # 刷新页面、重启程序之后看到的都和当时一样。
                save_exchange(sid, question, answer)

                # 【为什么成功后要「跳转」】这叫 POST-Redirect-GET 模式。
                # 不跳转的话，地址栏里留的是「刚才那次 POST」的结果，用户一按 F5 浏览器就会问
                # 「要重新提交吗」，点「是」就又问了 AI 一遍。跳转之后地址变回普通 GET，F5 不会重复提问。
                return redirect(url_for("index"))

            except Exception:
                # 【绝不显示 str(e)】
                # 异常原文里可能有内部路径、请求内容、甚至密钥片段。
                # 用户只需要知道「这次没成」，具体原因留在服务端就好。
                # 注意这里也【不写数据库】——存一条半截的对话，比不存更糟。
                error = SAFE_ERROR_MESSAGE

            finally:
                _lock.release()                                              # 无论成功还是出错，都把锁还回去

    # 渲染页面。历史每次都从数据库读，所以重启程序也不会丢。
    return render_template("index.html", history=load_history(sid), question=question, error=error)


if __name__ == "__main__":        # 判断：只有直接运行 python app.py 时才执行下面这句（被别人 import 时不执行）
    app.run(debug=True)           # 启动网站服务器。debug=True 表示改了代码会自动重启，方便调试；正式上线要关掉
