# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# 本地管理脚本：发邀请码 / 作废邀请码
#
# 【它在自己这台机器上跑，不碰数据库、不碰容器】
# 它做的事情只有一件：向线上那台服务发一个 HTTPS 请求（带管理令牌）。
# 邀请码是在【这里】用密码学安全随机源生成的 —— 服务端只负责登记它的摘要。
#
# 【为什么码要在本地生成】
# 如果码由服务端生成、再通过响应传回来，那么"响应丢了"就等于"码丢了"：
# 你不知道线上到底有没有多出一张谁也认不得的码。
# 本地生成之后，明文【从始至终在你手里】—— 响应丢了就拿同一张码重试，
# 而且重试是幂等的（服务端对同一个码只会有一行）。
#
# 【四条安全规矩，改这个文件时别破】
#   ① 令牌只从【隐藏输入】读：不放命令行参数（会进 shell 历史 / 进程列表）、
#      不放 URL、不写进文件、不打印
#   ② 码明文只打印【一次】，而且是在【发送之前】——
#      提示你"先安全复制；确认登记成功之前不要转发"
#   ③ 状态未知时【绝不声称成功】，也【绝不自动生成第二张码】
#   ④ 这个脚本不写任何日志文件；令牌和码都不落盘
#
# 用法：
#     python admin_invite.py new    --base-url https://<你的域名>
#     python admin_invite.py revoke --base-url https://<你的域名>
# =====================================================================

import argparse
import getpass      # 隐藏输入（不回显），读令牌和邀请码
import json
import re
import secrets      # 本地生成邀请码用的密码学安全随机源
import sys
import urllib.error
import urllib.parse
import urllib.request


# ===================== 服务端会回的结果 =====================
#
# 【为什么要逐个区分】这些状态对应【完全不同的下一步动作】：
# 有的可以照常把码发出去，有的必须换一张，有的只能重试。
# 把它们混成一句"失败"，操作者就只能瞎猜。

OUT_CREATED = "created"                        # 新登记成功
OUT_ALREADY_REGISTERED = "already_registered"  # 同一张码之前已登记，仍有效未绑定
OUT_ALREADY_BOUND = "already_bound"            # 这张码已经被用掉了
OUT_REVOKED = "revoked"                        # 这张码已被作废
OUT_REJECTED = "rejected"                      # 服务端拒绝（内部错误）

OUT_UNAUTHORIZED = "unauthorized"              # 401/404：多半是令牌或入口配置的问题
OUT_BAD_REQUEST = "bad_request"                # 400：多半是请求格式的问题
OUT_REDIRECT_BLOCKED = "redirect_blocked"      # 收到 3xx：不跟随、也不转发令牌
OUT_UNKNOWN = "unknown"                        # 其它一切 —— 都按「状态未知」处理
OUT_NETWORK = "network"                        # 连不上 / 超时（也没收到任何响应）

# 【判定规则只有一条：只有能识别的 HTTP 200 业务结果，才下结论】
#
#   ✅ 200 + {"result": "created" / "already_registered" / "already_bound" / "revoked"}
#      → 这是服务端明确告诉我们的业务结果，可以照实说
#   ❌ 其它任何响应（包括 500，包括带我们 JSON 标记的错误）
#      → 【只能说「状态未知」】
#
# 【为什么连「我们自己回的 500」都不能算"没登记"】
# 早先的版本试图区分"已确认"和"未知"：如果错误响应带着我们约定的 JSON 标记
# （{"error": "internal_error"}），就认为"这是我们服务端回的，所以事务没提交"。
# **这个推理是错的**：标记只证明"响应来自我们的进程"，完全不证明"数据库没写进去"——
# 完全可能是事务已经 COMMIT，之后才在别的地方出错并返回 500。
# 一旦在这种情况说"没有被登记"，操作者就会放心地用同一张码重试或直接丢弃它，
# 而线上其实已经多了一张码。
#
# 所以规则简化成：**请求一旦发出，非 200 一律"状态未知 + 核对配置 + 用同一张码重试"。**
# 重试是幂等的（同一个码只会有一行），所以这个保守说法不会让操作者多做无用工。

# 撤销的结果
OUT_REVOKED_OK = "revoked_ok"
OUT_REVOKE_NOT_FOUND = "revoke_not_found"


# ===================== 生成邀请码（本地） =====================

def generate_code():
    """本地生成一个邀请码。

    【为什么和服务端同一个算法】24 字节的 token_urlsafe 约 32 个字符，
    高熵、不含可识别信息。这里不需要和服务端"算得一样"，
    只要熵够、且服务端把它当普通字符串登记即可。
    """
    return secrets.token_urlsafe(24)


# ===================== 把 HTTP 响应翻译成人话 =====================

def _is_our_error(payload, name):
    """这个响应体是不是【我们自己的服务端】回的那个固定错误？

    【它现在的用途只有一个：给操作者一个更准的排查方向。】
    我们自己的错误一律是 {"error": "unauthorized" / "not_found" / ...} 这种固定 JSON；
    代理、网关、NGINX 的 404/401 页面不会长这样。所以看到这个标记，
    就能提示"多半是令牌/配置的问题"。

    【它【不再】用来下任何结论】
    早先的版本拿它当"已确认没登记"的依据 —— 那是错的：
    标记只证明"这个响应来自我们的进程"，完全不证明"数据库没被写过"
    （事务可能已经 COMMIT，之后才出错返回 500）。所以现在它只影响提示措辞，
    不影响判定：非 200 一律是「状态未知」。
    """
    return isinstance(payload, dict) and payload.get("error") == name


def classify_register(status_code, payload):
    """把一次登记请求的结果翻译成一个状态。

    【只有 200 + 认得出的业务结果才算「知道结果」】其它一律 unknown 或「多半是什么问题」。
    """
    if status_code == 200 and isinstance(payload, dict):
        result = payload.get("result")
        if result in (OUT_CREATED, OUT_ALREADY_REGISTERED,
                      OUT_ALREADY_BOUND, OUT_REVOKED):
            return result
        return OUT_UNKNOWN                           # 200 但看不懂 → 未知

    # 【3xx：我们根本不跟随，也绝不算成功】见 NoRedirectHandler 的说明
    if status_code is not None and 300 <= status_code < 400:
        return OUT_REDIRECT_BLOCKED

    # 下面两种只是【提示可能的原因】，不是"确认了什么"：
    # 带我们标记的错误响应说明「多半是我们的服务端在说：你的令牌/请求有问题」，
    # 但**绝不代表数据库没被写过**（见文件顶部那段说明）。
    if status_code in (401, 404) and (_is_our_error(payload, "unauthorized")
                                      or _is_our_error(payload, "not_found")):
        return OUT_UNAUTHORIZED
    if status_code == 400 and _is_our_error(payload, "bad_request"):
        return OUT_BAD_REQUEST

    return OUT_UNKNOWN                               # 其它一切（含 500、无标记的 4xx/5xx）


def classify_revoke(status_code, payload):
    """撤销走同一套原则：只有 200 的明确业务结果才下结论。"""
    if status_code == 200 and isinstance(payload, dict):
        result = payload.get("result")
        if result == "revoked":
            return OUT_REVOKED_OK
        if result == "not_found":
            return OUT_REVOKE_NOT_FOUND
        return OUT_UNKNOWN

    if status_code is not None and 300 <= status_code < 400:
        return OUT_REDIRECT_BLOCKED

    if status_code in (401, 404) and (_is_our_error(payload, "unauthorized")
                                      or _is_our_error(payload, "not_found")):
        return OUT_UNAUTHORIZED
    if status_code == 400 and _is_our_error(payload, "bad_request"):
        return OUT_BAD_REQUEST

    return OUT_UNKNOWN


# 【统一口径：请求发出之后的任何非 200，都不许说"登记了"或"没登记"】
#   · 只有 200 + 认得出的业务结果，才照实说
#   · 其余一律：状态未知 → 核对配置 → 用【同一张码】重试（重试是幂等的）
#   · 401/404/400 可以**提示可能原因**，但那只是排查方向，不是结论
#
# 【退出码】0 = 有明确结果（或本地取消、什么都没发）
#          2 = 200 明确告诉我们「这不是一张可用的新码」
#          3 = 请求发出去了，但我们【无法确认】结果 —— 一律重试同一张码
#          4 = 请求【根本没有发出去】（本地校验没过 / 没有输入）
UNKNOWN_EXIT = 3
NOT_SENT_EXIT = 4


def _unknown_register_message(hint, status=None):
    """「状态未知」的统一说法。hint 是可能原因，可省略。"""
    lines = ["⚠️ 状态未知" + ("（HTTP " + str(status) + "）" if status else "") + "。"]
    if hint:
        lines.append("   可能的原因：" + hint + "（但这只是排查方向，不是结论）")
    lines.append("   【无法确认】这次登记到底有没有发生 —— 服务端也可能已经写进去了。")
    lines.append("   【不要】重新生成第二张码：请核对上面的配置，")
    lines.append("   然后用【同一张码】再跑一次本脚本（选「重试」）。重试是幂等的。")
    return ("\n".join(lines), UNKNOWN_EXIT)


def register_message(outcome, code, status=None):
    """返回 (给操作者看的话, 退出码)。话里【必须】写清下一步该做什么。"""
    # ---- 只有这些是「知道结果」的 ----
    if outcome == OUT_CREATED:
        return ("✅ 登记成功（新码）。现在可以把这张码发给用户了。", 0)
    if outcome == OUT_ALREADY_REGISTERED:
        return ("✅ 已经登记过了（就是同一张码，没有重复发码）。可以发给用户。", 0)
    if outcome == OUT_ALREADY_BOUND:
        return ("⛔ 这张码【已经被某位学习者用掉了】，它不是一张新码。\n"
                "   请重新运行本脚本生成一张【新的】码。", 2)
    if outcome == OUT_REVOKED:
        return ("⛔ 这张码【已经被作废】，不能再用。\n"
                "   请重新运行本脚本生成一张【新的】码。", 2)

    # ---- 下面这些都没法下结论 ----
    if outcome == OUT_UNAUTHORIZED:
        return _unknown_register_message("令牌不对，或者线上没有配置管理令牌"
                                         "（去 Railway Variables 核对 ADMIN_MINT_TOKEN，"
                                         "注意别配成模板里的示例值）", status)
    if outcome == OUT_BAD_REQUEST:
        return _unknown_register_message("请求格式被拒（脚本版本不匹配？中间有代理改写了请求？）",
                                         status)
    if outcome == OUT_REDIRECT_BLOCKED:
        return ("⚠️ 状态未知：服务端回了「跳转」（3xx）。这通常是 --base-url 写错，或者中间有代理。\n"
                "   【令牌没有被转发到新地址】——本脚本不跟随重定向。\n"
                "   【无法确认】这次登记到底有没有发生：请核对域名，并用【同一张码】重试。",
                UNKNOWN_EXIT)
    if outcome == OUT_NETWORK:
        return ("⚠️ 状态未知：网络没通（连不上或超时）。服务端也有可能已经收到了请求。\n"
                "   【不要】重新生成第二张码 —— 请用【同一张码】再跑一次本脚本（选「重试」）。",
                UNKNOWN_EXIT)
    return _unknown_register_message("服务端返回了无法解读的响应", status)


def revoke_message(outcome, status=None):
    """撤销也是同一套口径：只有 200 的明确结果才算知道结果。"""
    if outcome == OUT_REVOKED_OK:
        return ("✅ 已作废。这张码不能再用来进入，"
                "该学习者名下【已绑定的浏览器会话也已断开】。", 0)
    if outcome == OUT_REVOKE_NOT_FOUND:
        return ("⚠️ 没有找到这张码（可能本来就输错了）。服务端明确回了这个结果，"
                "所以可以确认：什么都没有改动。", 2)

    if outcome == OUT_UNAUTHORIZED:
        return _unknown_revoke_message("令牌不对，或者线上没有配置管理令牌", status)
    if outcome == OUT_BAD_REQUEST:
        return _unknown_revoke_message("请求格式被拒", status)
    if outcome == OUT_REDIRECT_BLOCKED:
        return ("⚠️ 状态未知：服务端回了「跳转」（3xx）。这通常是 --base-url 写错，或者中间有代理。\n"
                "   【令牌没有被转发到新地址】。\n"
                "   【无法确认】这次作废到底有没有执行：请核对域名，再用【同一个码】重试。",
                UNKNOWN_EXIT)
    if outcome == OUT_NETWORK:
        return ("⚠️ 状态未知：网络没通。服务端也有可能已经收到了请求。\n"
                "   【不要】当成作废成功了 —— 请用【同一个码】再跑一次本脚本。", UNKNOWN_EXIT)
    return _unknown_revoke_message("服务端返回了无法解读的响应", status)


def _unknown_revoke_message(hint, status=None):
    lines = ["⚠️ 状态未知" + ("（HTTP " + str(status) + "）" if status else "") + "。"]
    if hint:
        lines.append("   可能的原因：" + hint + "（但这只是排查方向，不是结论）")
    lines.append("   【无法确认】这次作废到底有没有执行 —— 服务端也可能已经作废了。")
    lines.append("   请核对上面的配置，然后用【同一个码】再跑一次本脚本。")
    return ("\n".join(lines), UNKNOWN_EXIT)


# ===================== 目标地址的校验 =====================
#
# 【为什么必须在【要令牌之前】就校验】
# 令牌是这个脚本里最敏感的东西，它会被放进请求头发出去。
# 如果 --base-url 写成了 http:// 或者被人塞了诱导性的形式，
# 令牌就等于直接送给别人了 —— 而操作者往往是在事后才发现。
# 所以：**先把目标地址验干净、并把最终域名明确打给操作者看，然后再问令牌。**
#
# 【拒绝哪些形式】
#   · 不是 https（明文传输，令牌在链路上裸奔）
#   · 带用户名/密码（http://user:pass@host —— 既是明文凭据，又会让人看错主机）
#   · 带查询串 / 片段（?x=1、#y —— 不影响实际目标却容易误导）
#   · 带路径（我们还要往后拼 /admin/invites，带路径会拼出看不懂的地址）
#   · 主机名不像话（空、带空格、没有点的单标签）

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.\-]*[A-Za-z0-9])?$")


def validate_base_url(raw):
    """校验目标地址。返回 (规范化后的地址 或 None, 给操作者看的原因)。

    【只做「形式」上的校验】它不能证明对面真的是你的服务 ——
    那是 TLS 证书和域名本身的事。这里做的是：挡掉那些"一看就不对"的形式，
    并强制 HTTPS + 把最终域名原样打出来，让操作者自己再确认一眼。
    """
    if not raw or not str(raw).strip():
        return None, "没有提供 --base-url。"

    text = str(raw).strip()
    if any(ch.isspace() for ch in text):
        return None, "地址里有空格或换行，请检查。"

    parts = urllib.parse.urlsplit(text)

    if parts.scheme.lower() != "https":
        return None, ("只允许 https:// （当前是 " + (parts.scheme or "（空）") + "://）。"
                      "管理令牌会放在请求头里，明文 http 等于把它送到链路上。")

    if "@" in parts.netloc:
        return None, "地址里不能带用户名或密码（user:pass@host 这种形式）。"

    if parts.query:
        return None, "地址里不能带查询串（? 后面的部分）。"

    if parts.fragment:
        return None, "地址里不能带片段（# 后面的部分）。"

    if parts.path not in ("", "/"):
        return None, ("地址里不能带路径，只填域名（例如 https://xxx.up.railway.app）。"
                      "本脚本会自己往后拼 /admin/invites。")

    host = parts.hostname or ""
    if not host:
        return None, "地址里没有主机名。"
    if not _HOSTNAME_RE.match(host):
        return None, "主机名看起来不合法：" + host
    if "." not in host:
        return None, ("主机名必须是一个完整的域名（要带点），当前是：" + host
                      + "。这样能挡掉 localhost 之类写错的目标。")

    try:
        port = parts.port                              # 非法端口会在这里抛错
    except ValueError:
        return None, "端口不合法。"

    normalized = "https://" + host + ((":" + str(port)) if port else "")
    return normalized, ""


# ===================== HTTP =====================

class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """【绝不跟随重定向】—— 这一条是安全要求，不是洁癖。

    Python 默认会跟随 3xx，而且它的重定向处理会把请求头**复制**到新地址 ——
    包括我们加的 `X-Admin-Token`。也就是说：
    如果服务端（或中间的代理）回一个 302 到别的域名，
    我们会**把管理令牌原样发到那个新地址去**。
    所以这里直接拒绝跟随：返回 None 表示"这个 3xx 我不处理"，
    于是 urlopen 会把它当 HTTPError 抛出来，交给 classify 判成「状态未知」。

    返回 None 之后，只有一个结果：**令牌只可能发给我们最初校验过的那个域名。**
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def build_opener():
    """构造发管理请求用的 opener（不跟随重定向）。单独写出来是为了能在测试里检查。"""
    return urllib.request.build_opener(NoRedirectHandler)


_OPENER = build_opener()


def http_post(base_url, path, fields, token, timeout=30, opener=None):
    """发一个表单 POST。令牌放【请求头】，码放【请求体】——都不进 URL。

    返回 (状态码 或 None, 解析出来的 JSON 或 None)。连不上时返回 (None, None)。
    opener 可以注入（测试用），默认是那个不跟随重定向的。
    """
    opener = opener or _OPENER
    url = base_url.rstrip("/") + path
    body = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    request.add_header("X-Admin-Token", token)
    request.add_header("Cache-Control", "no-store")

    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, _parse_json(response.read())
    except urllib.error.HTTPError as exc:
        # 4xx/5xx/3xx 也带着响应体，读出来交给 classify 判断
        try:
            return exc.code, _parse_json(exc.read())
        except Exception:
            return exc.code, None
    except Exception:
        # 【连不上 / 超时 / DNS 挂了】—— 都归为「状态未知」，绝不猜
        return None, None


def _parse_json(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


# ===================== 主流程 =====================

def main(argv=None, *, ask_token=None, ask_secret=None, ask_confirm=None,
         say=None, post=None):
    """命令行入口。

    【为什么这些 I/O 都能注入】这样离线测试可以完整地驱动一遍流程
    （包括"生成 → 展示 → 确认 → 发送 → 解读状态"），而不需要真的联网。
    """
    ask_token = ask_token or (lambda label: getpass.getpass(label))
    ask_secret = ask_secret or (lambda label: getpass.getpass(label))
    ask_confirm = ask_confirm or (lambda label: input(label).strip().lower() in ("y", "yes"))
    say = say or print
    post = post or http_post

    parser = argparse.ArgumentParser(
        description="发邀请码 / 作废邀请码（本地生成，服务端只登记摘要）")
    parser.add_argument("action", choices=["new", "revoke"])
    parser.add_argument("--base-url", required=True,
                        help="线上地址，例如 https://xxx.up.railway.app（不是秘密，可以直接写在命令行）")
    args = parser.parse_args(argv)

    # ---- 【第一件事：把目标地址验干净】在问令牌、在生成码之前 ----
    base_url, problem = validate_base_url(args.base_url)
    if base_url is None:
        say("")
        say("⛔ --base-url 不合法：" + problem)
        say("   地址没问题之前，本脚本【不会】索取令牌，也不会生成任何邀请码。")
        say("   （什么都没有发送。）")
        return NOT_SENT_EXIT

    # 【把最终目标明确打出来】操作者必须看得见令牌将要发往哪个域名
    target = urllib.parse.urlsplit(base_url).hostname
    say("")
    say("目标服务：" + base_url + "   （域名：" + target + "）")
    say("令牌只会发往这个域名，且本脚本不跟随任何跳转。")

    # ---- 令牌：隐藏输入，不打印、不落盘 ----
    say("")
    say("请输入管理令牌 ADMIN_MINT_TOKEN（输入不会显示，也不会被记录）：")
    token = ask_token("令牌（发往 " + target + "）: ").strip()
    if not token:
        say("⛔ 没有输入令牌，什么都没做。")
        return 4

    if args.action == "new":
        return _do_register(base_url, token, ask_secret, ask_confirm, say, post)
    return _do_revoke(base_url, token, ask_secret, ask_confirm, say, post)


def _do_register(base_url, token, ask_secret, ask_confirm, say, post):
    # ---- 是重试还是新生成？ ----
    is_retry = ask_confirm("这是重试【之前那张】待登记的码吗？(y/N) ")
    if is_retry:
        say("请粘贴之前那张待登记的邀请码（输入不会显示）：")
        code = ask_secret("邀请码: ").strip()
        if not code:
            say("⛔ 没有输入邀请码，什么都没做。")
            return 4
        say("")
        say("即将用【同一张码】重试登记。这是幂等的：服务端只会有一行。")
    else:
        code = generate_code()
        say("")
        say("=" * 68)
        say("  待登记邀请码（只显示这一次）：")
        say("")
        say("      " + code)
        say("")
        say("  ⚠️ 请先【安全复制】到你的密码管理器或临时安全的地方。")
        say("  ⚠️ 在下面这一步确认登记成功之前，【不要】把它转发给用户 ——")
        say("     万一登记失败或被拒绝，你还需要用它重试。")
        say("=" * 68)
        say("")

    # ---- 发送【之前】的最后一道确认 ----
    say("提示：登记成功后，这张码就可以发给用户了。")
    say("     现在【不会】有任何码被打印第二次，请确认你已经复制好了。")
    if not ask_confirm("发送登记请求？(y/N) "):
        say("")
        say("已取消，什么都没有发送。")
        say("这张码还在你手上 —— 想登记时重新运行本脚本，选「重试」并粘贴同一张码。")
        return 0

    status, payload = post(base_url, "/admin/invites", {"code": code}, token)
    outcome = OUT_NETWORK if status is None else classify_register(status, payload)

    text, code_exit = register_message(outcome, code, status)
    say("")
    say(text)
    return code_exit


def _do_revoke(base_url, token, ask_secret, ask_confirm, say, post):
    say("请粘贴要作废的邀请码（输入不会显示）：")
    code = ask_secret("邀请码: ").strip()
    if not code:
        say("⛔ 没有输入邀请码，什么都没做。")
        return 4

    say("")
    say("将要执行：")
    say("  · 这张码之后不能再用来进入")
    say("  · 该学习者名下【已绑定的浏览器会话会被断开】（数据不会被删除）")
    say("  · 这个动作【不可撤销】")
    say("")
    if not ask_confirm("确认作废？(y/N) "):
        say("已取消，什么都没有改动。")
        return 0

    status, payload = post(base_url, "/admin/invites/revoke",
                           {"code": code, "confirm": "yes"}, token)
    outcome = OUT_NETWORK if status is None else classify_revoke(status, payload)

    text, code_exit = revoke_message(outcome, status)
    say("")
    say(text)
    return code_exit


if __name__ == "__main__":
    raise SystemExit(main())
