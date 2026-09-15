# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

import os                                           # 读环境变量、拼路径
import uuid                                         # 生成随机的「会话 ID」
import sqlite3                                      # Python 自带的轻量数据库，不用额外安装任何东西
import threading                                    # 用它的「锁」防止同一时间处理两个请求
from datetime import datetime, timedelta            # datetime 记录消息时间；timedelta 设置 cookie 有效期
from flask import Flask, render_template, request, redirect, url_for, session   # session 用来给每个浏览器发一个签名过的身份标记
from openai import OpenAI                           # 从 openai 库里导入 OpenAI 类（DeepSeek 兼容它的接口）


# ===================== 读取 .env（如果存在）=====================
# 为什么要有这个：密钥不能写进代码，只能放环境变量。但每次开新终端都要重新设一遍太麻烦，
# 所以约定把密钥写在一个叫 .env 的文件里，程序启动时自动读进来。
# 这个文件被 .gitignore 忽略，永远不会被推到 GitHub。
# 这里只用 Python 标准库自己解析，不额外装 python-dotenv。

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 本文件（app.py）所在的文件夹，后面拼路径都以它为基准


def load_dotenv(path):
    """把 .env 文件里的 KEY=VALUE 逐行读进环境变量。文件不存在就安静地跳过。"""
    if not os.path.exists(path):                    # 没有 .env 文件属于正常情况（比如在服务器上直接设了环境变量）
        return
    with open(path, encoding="utf-8") as f:         # 打开文件，用 utf-8 读，中文注释才不会乱码
        for raw in f:                               # 一行一行读
            line = raw.strip()                      # 去掉首尾空白
            if not line or line.startswith("#") or "=" not in line:   # 空行、注释行、没有等号的行，都跳过
                continue
            key, _, value = line.partition("=")     # 用第一个等号切成「键」和「值」。partition 只会切一刀，值里再有等号也不受影响
            key = key.strip()                       # 去掉键两边的空白
            value = value.strip().strip('"').strip("'")   # 去掉值两边的空白，以及可能存在的引号
            os.environ.setdefault(key, value)       # setdefault：只有环境变量里还没有这个键时才写入。
                                                    # 意思是「真实环境变量优先，.env 只作兜底」——方便临时覆盖


load_dotenv(os.path.join(BASE_DIR, ".env"))         # 启动时先读 .env，下面再读环境变量


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
SYSTEM_PROMPT = "你是一个耐心的 AI 学习助手。请用简洁、通俗的中文回答问题；如果学生的描述不够清楚，就先追问一句。"

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
                # 要发给 AI 的内容 = 系统提示 + 【这个会话自己的历史】 + 学生这次问的。
                # 注意是从数据库按 sid 读的，所以拿到的永远是自己的对话，不会是别人的。
                messages_to_send = (
                    [{"role": "system", "content": SYSTEM_PROMPT}]
                    + load_history(sid)
                    + [{"role": "user", "content": question}]
                )

                response = client.chat.completions.create(                    # 调用接口，向 DeepSeek 发请求
                    model=MODEL,
                    messages=messages_to_send,
                )
                answer = response.choices[0].message.content                 # 从返回结果里一层层取出 AI 回答的文字

                save_exchange(sid, question, answer)                          # 成功之后，才把这一问一答写进数据库

                # 【为什么成功后要「跳转」】这叫 POST-Redirect-GET 模式。
                # 不跳转的话，地址栏里留的是「刚才那次 POST」的结果，用户一按 F5 浏览器就会问
                # 「要重新提交吗」，点「是」就又问了 AI 一遍。跳转之后地址变回普通 GET，F5 不会重复提问。
                return redirect(url_for("index"))

            except Exception as e:
                error = "出错了：" + str(e)                                  # 记下错误信息，稍后显示在网页上

            finally:
                _lock.release()                                              # 无论成功还是出错，都把锁还回去

    # 渲染页面。历史每次都从数据库读，所以重启程序也不会丢。
    return render_template("index.html", history=load_history(sid), question=question, error=error)


if __name__ == "__main__":        # 判断：只有直接运行 python app.py 时才执行下面这句（被别人 import 时不执行）
    app.run(debug=True)           # 启动网站服务器。debug=True 表示改了代码会自动重启，方便调试；正式上线要关掉
