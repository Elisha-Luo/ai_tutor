# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

import os                                           # 导入 os 模块，用来读取操作系统里的「环境变量」
import threading                                     # 导入 threading 模块，用它的「锁」来防止同一时间处理两个请求
from flask import Flask, render_template, request, redirect, url_for   # 从 flask 导入：Flask 建网站、render_template 渲染网页、request 读取用户提交的数据、redirect/url_for 用来跳转页面
from openai import OpenAI                            # 从 openai 库里导入 OpenAI 类（DeepSeek 兼容它的接口）

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")     # 从环境变量 DEEPSEEK_API_KEY 读取密钥；没设置的话得到空字符串
if not API_KEY:                                      # 如果没读到密钥
    raise RuntimeError(                              # 直接中止程序并报错——网站没密钥根本用不了，不如启动时就明确报出来
        "没有找到 DeepSeek 密钥。请先设置环境变量 DEEPSEEK_API_KEY，方法见 README.md 的「配置 API 密钥」一节。"
    )

BASE_URL = "https://api.deepseek.com"                 # DeepSeek 的接口地址
MODEL = "deepseek-chat"                               # 要调用的模型名字
SYSTEM_PROMPT = "你是一个耐心的 AI 学习助手。请用简洁、通俗的中文回答问题；如果学生的描述不够清楚，就先追问一句。"   # 给 AI 定的人设，每次请求都会带上

app = Flask(__name__)                                 # 创建一个 Flask 应用对象，__name__ 是当前文件的名字，Flask 靠它去找 templates 文件夹

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)   # 创建 OpenAI 客户端，整个程序共用一个

history = []                                          # 用一个列表存整场对话。每次请求都把整个列表发给 AI，它才「记得」前面聊过什么

# 【防止重复提问的第一道防线（服务端）】
# 「锁」可以理解成一支只有一把的钥匙：谁先拿到，谁才能干活；其他人拿不到就得等着。
# 我们故意设成「拿不到就不等，直接放弃」——因为拿不到就说明已经有一个问题在处理了。
_lock = threading.Lock()


@app.route("/", methods=["GET", "POST"])              # 装饰器：把下面这个函数绑定到网站根地址 "/"，并允许 GET 和 POST 两种访问方式
def index():                                          # 这是首页的处理函数，用户每次打开页面或点发送都会执行它
    question = ""                                     # 准备一个变量存学生这次问的话，先设为空字符串
    error = ""                                        # 准备一个变量存错误信息，先设为空字符串

    if request.method == "POST":                      # 如果这次访问是「提交表单」（POST），而不是单纯打开网页（GET）
        question = request.form.get("question", "").strip()   # 从表单里取出名字叫 question 的输入框内容，去掉前后空格

        if question:                                  # 如果学生确实输入了内容（空字符串在 Python 里算 False，所以这一步顺带挡掉了空输入）

            # 试着去拿那把锁。acquire(blocking=False) 的意思是「拿不到立刻返回 False，不要在这里干等」。
            # 拿不到 = 已经有一个请求正在问 AI 了。这种情况几乎只有一个原因：
            # 用户等不及，又按了一次回车。那就直接别问了——不然会白白多花一次钱，
            # 而且对话里会凭空多出一遍一模一样的问题。
            if not _lock.acquire(blocking=False):
                return render_template(
                    "index.html", history=history, question=question,
                    error="上一个问题还在处理中，等它回答完再问下一个吧。",
                )

            # try ... finally：不管中间成功还是失败，最后都必须把锁还回去。
            # 这一点极其重要——如果失败时忘了还锁，这把锁就永远拿不回来了，
            # 之后所有提问都会被当成「有请求在处理」而拒绝，页面就彻底用不了了。
            try:
                # 要发给 AI 的内容 = 系统提示 + 之前的所有对话 + 学生这次问的
                messages_to_send = (
                    [{"role": "system", "content": SYSTEM_PROMPT}]   # 第一条：给 AI 的人设
                    + history                                        # 中间：之前所有的对话，AI 靠它记住上下文
                    + [{"role": "user", "content": question}]        # 最后一条：学生这次问的话
                )

                response = client.chat.completions.create(                    # 调用接口，向 DeepSeek 发请求
                    model=MODEL,                                              # 指定用哪个模型
                    messages=messages_to_send,                                # 上面拼好的完整对话
                )
                answer = response.choices[0].message.content                 # 从返回结果里一层层取出 AI 回答的文字

                history.append({"role": "user", "content": question})        # 成功之后，才把学生这句记进对话历史
                history.append({"role": "assistant", "content": answer})     # 再把 AI 的回答也记进去，下一轮它才知道自己刚说过什么

                # 【为什么成功后要「跳转」而不是直接显示】这叫做 POST-Redirect-GET 模式，是网页开发的标准做法。
                # 如果这里直接渲染页面，浏览器地址栏里留的就是「刚才那次 POST」的结果；
                # 用户一按 F5，浏览器会弹窗问「要重新提交吗」，点「是」就又问了 AI 一遍 —— 又是一次重复提问。
                # 改成跳转之后，地址变回普通的 GET，F5 只是重新打开页面，不会重复提问。
                return redirect(url_for("index"))

            except Exception as e:                                           # 如果上面出错，把错误对象存进变量 e
                error = "出错了：" + str(e)                                  # 记下错误信息，稍后显示在网页上（这条不进历史，免得污染上下文）

            finally:                                                         # 无论上面成功还是出错，都会走到这里
                _lock.release()                                              # 把锁还回去，让下一次提问可以进来

    return render_template("index.html", history=history, question=question, error=error)   # 渲染 templates/index.html，把这三样东西传进网页里用


if __name__ == "__main__":        # 判断：只有直接运行 python app.py 时才执行下面这句（被别人 import 时不执行）
    app.run(debug=True)           # 启动网站服务器。debug=True 表示改了代码会自动重启，方便调试；正式上线要关掉
