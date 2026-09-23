# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# 共享的 .env 读取工具
#
# 【为什么要有这个文件】
# 这个项目有两个入口都要用 .env：
#     app.py                  网页厨房
#     evals/run_rag_eval.py   评测厨房
#
# 一开始只有 app.py 会读 .env，评测器没读——结果就是：
# 密钥只写在 .env 里的人，网页能用，评测器却报告「密钥未设置」。
# 同一份配置，两个入口给出两种结论，这种 bug 特别难查，因为两边单独看都「没错」。
#
# 所以约定：读 .env 这件事，全项目只有这一份实现，两边都 import 它。
# 以后要改读取规则（比如支持多行值），改一处就够，不会再各自漂移。
# =====================================================================
# 【安全约定，改动时务必守住】
#   · 本模块【不打印】任何东西
#   · 本模块【不记录】任何东西
#   · 本模块的返回值里【绝不包含密钥的值】，只包含「读了几个键」和键名
#   · 出错时抛出的/返回的信息里【绝不包含文件内容】
#
# 最后一条容易被忽略：像 UnicodeDecodeError 这种异常，它的消息里
# 会带上出错位置附近的一小段原始内容——而那段内容可能正好就是密钥所在的行。
# 所以读取失败时，这里只回一个笼统的标记，绝不把异常原样往外传。
# =====================================================================

import os      # 读文件、改环境变量、拼路径


# 本文件所在目录 = 项目根目录（app.py、evals/ 都在这一层）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认去读项目根目录下的 .env
DEFAULT_ENV_PATH = os.path.join(BASE_DIR, ".env")


def load_dotenv(path=None):
    """把 .env 文件里的 KEY=VALUE 逐行读进环境变量。

    参数：
        path —— .env 文件的路径。不传就用项目根目录下的 .env。

    返回（【只有统计信息，没有任何密钥内容】）：
        {
          "path":       读了哪个文件,
          "loaded":     文件存在且读完了吗（True / False）,
          "keys":       读到并处理过的【键名】列表，比如 ["DEEPSEEK_API_KEY"]。
                        注意是键名不是值。另外：环境变量里已经有的键也会出现在这里
                        （它被「尝试写入」过，只是 setdefault 没有真的覆盖），
                        所以这个列表表示「文件里有哪些配置项」，不表示「哪些被改动了」。
          "skipped":    跳过了几行（空行、注释行、格式不对的行）,
          "read_error": 读取过程出错了吗（True / False）,
        }

    【关于 keys 只放键名】
    "DEEPSEEK_API_KEY" 这个字符串本身不是秘密——它就写在 .env.example 里，
    公开可见。真正要保护的是等号后面的值。返回键名是为了让调用方能报告
    「读到了哪几项配置」，同时又不用碰值。
    """
    if path is None:
        path = DEFAULT_ENV_PATH

    result = {
        "path": path,
        "loaded": False,
        "keys": [],
        "skipped": 0,
        "read_error": False,
    }

    # 没有 .env 文件是完全正常的情况（比如在服务器上直接设了环境变量）。
    # 安静跳过，不报错、不打印。
    if not os.path.exists(path):
        return result

    loaded_keys = []
    skipped = 0

    try:
        with open(path, encoding="utf-8") as f:      # 用 UTF-8 读，中文注释才不会乱码
            for raw in f:
                line = raw.strip()                   # 去掉首尾空白

                # 空行、以 # 开头的注释行、没有等号的行 —— 一律跳过
                if not line or line.startswith("#") or "=" not in line:
                    skipped += 1
                    continue

                # 【用 partition 而不是 split】只切第一个等号。
                # 这样值里再出现等号（比如密钥或 URL 里带 =）也不会被切坏。
                key, _, value = line.partition("=")

                key = key.strip()
                # 值去掉两边空白，再去掉可能包着的引号（单引号和双引号都处理）
                value = value.strip().strip('"').strip("'")

                if not key:                          # 形如 "=abc" 的行，没有键名，跳过
                    skipped += 1
                    continue

                # 【最关键的一行】setdefault 只在环境变量里【还没有】这个键时才写入。
                # 意思是「真实环境变量优先，.env 只作兜底」——
                # 这样临时想覆盖某个值时，在命令行设一下就行，不用去改 .env。
                os.environ.setdefault(key, value)

                loaded_keys.append(key)

    except Exception:
        # 【绝不把异常原样往外传】UnicodeDecodeError 之类的消息里会带上
        # 出错位置附近的一小段原始内容，而那可能正是密钥所在的那一行。
        # 所以这里吞掉它，只回一个笼统的 read_error 标记。
        result["read_error"] = True
        result["skipped"] = skipped
        return result

    result["loaded"] = True
    result["keys"] = loaded_keys
    result["skipped"] = skipped
    return result


# ===================== 示例密钥检测 =====================
#
# 【为什么要有这个】
# `.env.example` 是提交到仓库里的**模板文件**，里面的示例值【公开可见】。
# 所以它绝不可能是一个能用的密钥 —— 可偏偏有人复制完忘了改。
#
# 真实踩到的坑：把 .env.example 复制成 .env 之后没改示例值，
# 应用【照常启动、照常发请求】，只是每一次都失败。
# 而失败又被 RAG 的安全兜底挡住 —— 最后看起来「跑完了」，
# 实际一次都没跑通，还浪费了一轮排查。
#
# 【所以：在调用网络之前就拦住】而不是让它带着假密钥跑完全程。

# 【必须和 .env.example 里的示例值完全一致】
# 改模板的时候，这里也要跟着改。
EXAMPLE_API_KEY = "sk-在这里填你自己的密钥"

# 给用户看的固定提示。不含密钥的任何部分。
EXAMPLE_KEY_MESSAGE = "DEEPSEEK_API_KEY 仍是示例值，请配置真实密钥。"


def is_example_api_key(value):
    """这个值是不是 `.env.example` 里那个公开的示例占位符？

    【只用「精确相等」判断，不做长度 / 格式校验】
    因为服务商的密钥格式以后可能改（长度、前缀都可能变），
    硬编码「必须 35 位」这类规则迟早会误伤真实密钥。
    这里只认一个事实：**它和仓库里那份公开模板长得一模一样**。

    【安全】这个函数不打印、不返回密钥内容 —— 只回一个布尔值。
    """
    return (value or "").strip() == EXAMPLE_API_KEY
