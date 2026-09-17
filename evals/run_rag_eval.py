# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# 完整 RAG 评测器 —— 整桌试吃
#
# 【厨房类比，这个类比能解释清楚每一块在干嘛】
#   knowledge_base/          是食材
#   retriever.py             是找食材的助手
#   rag.py                   是厨师
#   evals/rag_cases.json     是 33 位客人的点菜单
#   本文件                   是试吃员
#
# 试吃员要检查厨师有没有：
#   · 乱用食材（引用了没给它、或者根本不存在的资料）
#   · 乱报食材来源（引用对不上点菜单上写的）
#   · 没有食材还硬上菜（资料明明不支持，却硬给一个 answer）
#   · 该上菜却不上（资料里有，却拒答了）
# =====================================================================
# 【两种模式，边界必须清楚】
#
#   --dry-run（默认）  不调用 DeepSeek。只检查：
#                        题库格式 / 检索结果 / 结果记录格式 / 运行环境
#                      用来在花钱之前，先把所有不要钱的部分验一遍。
#
#   --live             明确执行真实 DeepSeek 调用，每一题一次，33 次。
#                      只有带上这个参数，才会真的联网、真的花钱。
#
# 【为什么默认是 dry-run 而不是 live】
# 让「什么都不加」是最安全的那个。想花钱必须显式说出口——
# 这样就不会出现「本想试跑一下，结果一回车花了 33 次调用」这种事。
# =====================================================================
# 【两道安全线，本轮必须守住】
#   1. API 密钥只从环境变量读，绝不打印、绝不写进结果文件
#   2. 模型的原始异常信息绝不写进结果、绝不打印细节
# =====================================================================

import os          # 拼路径、读环境变量（只读有无，不打印值）
import sys         # 模块搜索路径、退出码
import json        # 读写题库和结果
import time        # 给每一题计时
import argparse    # 解析命令行参数
import datetime    # 给结果文件起时间戳名字

# 本文件在 evals/ 里，它的上一级才是项目根目录（retriever.py / rag.py 在那）
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_HERE)

# 把项目根目录加进模块搜索路径，这样下面两行 import 才能找到同级的 retriever / rag。
# 放在 import 之前做，是因为 Python 找模块的路径必须在 import 那一刻就已经就位。
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import retriever as R     # 找资料
import rag as G           # 用资料（含引用校验）
from env_utils import load_dotenv   # 【和 app.py 共用同一份】.env 读取逻辑


# ===================== 配置 =====================

CASES_PATH = os.path.join(BASE_DIR, "evals", "rag_cases.json")     # 点菜单
RESULTS_DIR = os.path.join(BASE_DIR, "evals", "results")           # 试吃记录存这儿

# 【和网页读的是同一个文件】app.py 读的也是项目根目录下的 .env。
# 两个入口必须用同一份配置，否则会出现「网页能用、评测器说没密钥」这种怪事。
ENV_PATH = os.path.join(BASE_DIR, ".env")

BASE_URL = "https://api.deepseek.com"     # 只在 --live 时才会用到
MODEL = "deepseek-chat"                   # 只在 --live 时才会用到
DEFAULT_TOP_K = 3                         # 每题检索几段资料

# 题库里每一道题必须具备的字段。缺了就没法判分，属于题库本身的错误。
REQUIRED_CASE_KEYS = ("id", "question", "expected_behavior", "expected_sources", "category")

# 结果文件里每一条记录必须具备的字段。测试会拿这个清单去核对。
REQUIRED_RECORD_KEYS = (
    "id", "question", "category",
    "expected_behavior", "expected_sources",
    "retrieved_sources",
    "decision", "answer", "citations",
    "elapsed_ms",
    "strict_pass", "safe_pass",
    "needs_human_review", "failure_reason",
)


# ===================== 读题库 =====================

def load_cases(path=CASES_PATH):
    """把 33 道题读出来。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_cases(cases):
    """检查题库本身有没有毛病。返回一串问题描述，空列表表示没问题。

    【为什么要单独检查题库】
    题库是判分的唯一依据。它自己写错了（比如 id 重复、拒答题却写了来源），
    后面的判分再准也是错的——而且这种错很难发现，因为它看起来「跑通了」。
    """
    problems = []

    if not isinstance(cases, list):
        return ["题库的根节点不是数组，格式不对"]

    if not cases:
        return ["题库是空的"]

    seen_ids = set()

    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            problems.append("第 " + str(i + 1) + " 项不是对象")
            continue

        for key in REQUIRED_CASE_KEYS:
            if key not in c:
                problems.append("第 " + str(i + 1) + " 项缺少字段：" + key)

        cid = c.get("id")
        if cid in seen_ids:
            problems.append("id 重复：" + str(cid))
        seen_ids.add(cid)

        behavior = c.get("expected_behavior")
        if behavior not in ("answer", "refuse"):
            problems.append(str(cid) + "：expected_behavior 只能是 answer 或 refuse，实际是 " + str(behavior))

        sources = c.get("expected_sources")
        if not isinstance(sources, list):
            problems.append(str(cid) + "：expected_sources 必须是数组")
        elif behavior == "refuse" and sources:
            # 拒答题却写了来源，自相矛盾——判分时会永远判不过
            problems.append(str(cid) + "：拒答题的 expected_sources 必须是空数组")
        elif behavior == "answer" and not sources:
            # 该答题却没写来源，就没法判「引用对不对」了
            problems.append(str(cid) + "：该答题必须至少写一个 expected_sources")

    return problems


# ===================== 判分 =====================

def _fmt_sources(sources):
    """把来源集合排好序、拼成一串，方便打印和写进失败原因。"""
    return "[" + "、".join(sorted(sources)) + "]" if sources else "[]"


def score_case(case, result):
    """按点菜单给一道题判分。返回一个字典。

    【两个概念，必须分清楚】
      strict_pass —— 完全符合期望（该答的答了且引用完全对；该拒的明确拒了）
      safe_pass   —— 没有产生危险输出（没有编、没有硬答、引用是空的）

    这两个分开的价值：一道题可能「不算过，但也不危险」。
    比如该拒答的题，模型返回了 insufficient_evidence——它没有编造，
    所以 safe_pass 是 True；但它没有明确说「资料里没有」，
    所以 strict_pass 是 False。这种题不算灾难，但值得知道有多少。
    """
    behavior = case.get("expected_behavior")
    expected = set(case.get("expected_sources") or [])
    decision = result.get("decision") if result else None
    citations = (result.get("citations") if result else None) or []
    cited = {c.get("source") for c in citations if isinstance(c, dict)}

    # ---------- 该答题 ----------
    if behavior == "answer":
        if decision != "answer":
            # 没有硬答，所以是安全的；但该上菜没上，不算过
            return {
                "strict_pass": False,
                "safe_pass": True,
                "failure_reason": "期望回答，实际没有回答（decision=" + str(decision) + "）",
                "needs_human_review": None,
            }

        if cited != expected:
            # 【最需要盯的一类】它答了，但引用对不上——
            # 要么引错了资料，要么引了根本没给它的资料（编的）。
            return {
                "strict_pass": False,
                "safe_pass": False,
                "failure_reason": (
                    "回答了，但引用来源与期望不一致——期望 " + _fmt_sources(expected)
                    + "，实际 " + _fmt_sources(cited)
                ),
                "needs_human_review": "回答内容需人工复核（引用对不上，回答本身也可能是错的）",
            }

        # 引用完全正确 —— 但也只是「引用」对。
        return {
            "strict_pass": True,
            "safe_pass": True,
            "failure_reason": None,
            "needs_human_review": (
                "回答内容需人工复核（引用来源与期望一致，但**引用正确不等于内容正确**——"
                "校验层只能证明出处存在，证明不了这段话真的被那段资料支持）"
            ),
        }

    # ---------- 该拒答题（refuse / trap_refuse）----------
    strict = (decision == "refuse")                       # 明确拒答才算严格通过
    safe = (decision != "answer") and (len(citations) == 0)   # 没硬答、且没带引用才算安全

    reason = None
    if decision == "answer":
        reason = "资料不支持回答，模型却给出了 answer —— 属于强行作答（编造风险）"
    elif decision != "refuse":
        reason = "期望明确拒答（refuse），实际是 " + str(decision)

    review = None
    if decision == "answer":
        review = "回答内容需人工复核"
    if not safe:
        review = (review + "；" if review else "") + "失败需复核：" + str(reason)

    return {
        "strict_pass": strict,
        "safe_pass": safe,
        "failure_reason": reason,
        "needs_human_review": review,
    }


# ===================== 组装一条记录 =====================

def build_record(case, chunks, result, elapsed_ms):
    """把「一道题 + 检索结果 + 模型输出 + 耗时」打包成一条记录。

    result 传 None 时（dry-run），decision/answer 那几项留空，
    判分字段也留 None —— 因为这一轮压根没问模型，判分是没有意义的。
    """
    chunks = chunks or []
    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "category": case.get("category"),

        "expected_behavior": case.get("expected_behavior"),
        "expected_sources": list(case.get("expected_sources") or []),

        # 检索层实际捞到了什么 —— 判分出错时，第一个要看的就这一行
        "retrieved_sources": [c.get("source") for c in chunks],
        "retrieved_headings": [c.get("heading") for c in chunks],

        # 模型层实际产出了什么
        "decision": result.get("decision") if result else None,
        "answer": result.get("answer") if result else None,
        "citations": list(result.get("citations") or []) if result else [],

        "elapsed_ms": elapsed_ms,

        # 判分（dry-run 时全部为 None）
        "strict_pass": None,
        "safe_pass": None,
        "needs_human_review": None,
        "failure_reason": None,
    }


# ===================== 汇总 =====================

def summarize(records, mode, model=None, top_k=DEFAULT_TOP_K):
    """把逐条记录汇总成整份报告。"""
    counts = {"strict_pass": 0, "safe_pass": 0, "failed": 0}
    by_category = {}
    failures = []              # 不安全的输出（必须处理）
    safe_misses = []           # 安全、但没达到期望（要改进，但不危险）

    for r in records:
        cat = r.get("category") or "未分类"
        bucket = by_category.setdefault(cat, {"total": 0, "strict_pass": 0, "safe_pass": 0})
        bucket["total"] += 1

        if r.get("strict_pass"):
            counts["strict_pass"] += 1
            bucket["strict_pass"] += 1

        if r.get("safe_pass"):
            counts["safe_pass"] += 1
            bucket["safe_pass"] += 1
            if not r.get("strict_pass"):
                # 没有编造、也没有硬答，但没达到点菜单上的期望。
                # 【必须和 failures 分开】这类不是事故，是能力不足——
                # 混在一起会让「危险」这件事失去焦点。
                safe_misses.append({
                    "id": r.get("id"),
                    "category": r.get("category"),
                    "question": r.get("question"),
                    "reason": r.get("failure_reason"),
                })
        else:
            counts["failed"] += 1
            failures.append({
                "id": r.get("id"),
                "category": r.get("category"),
                "question": r.get("question"),
                "reason": r.get("failure_reason"),
            })

    return {
        "mode": mode,
        "time": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model": model,
        "top_k": top_k,
        "total": len(records),
        "counts": counts,
        "by_category": by_category,
        "failures": failures,
        "safe_misses": safe_misses,
        "cases": records,
    }


# ===================== dry-run：不花钱的那部分 =====================

def run_dry(cases, retrieve_fn=None, top_k=DEFAULT_TOP_K):
    """走一遍完整流程，但【绝不调用模型】。

    【注意这个函数没有 client 参数 —— 这不是疏忽，是刻意的】
    没有客户端，它在结构上就不可能发起模型调用。测试里有一条专门
    检查这一点：哪怕有人日后手滑想在这里加调用，签名先就对不上。
    """
    if retrieve_fn is None:
        retrieve_fn = R.retrieve

    report = {
        "mode": "dry-run",
        "time": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "total": len(cases),
        "case_problems": validate_cases(cases),
        "index": {},
        "retrieval": {"empty": [], "per_case": []},
        "record_shape_ok": True,
        "record_shape_problems": [],
        "environment": {},
    }

    # ---------- 索引情况 ----------
    try:
        r = R.Retriever()
        report["index"] = {
            "chunks": len(r.chunks),
            "sources": r.sources,
            "vocab_size": r.vocab_size,
            "readme_indexed": any("readme" in s.lower() for s in r.sources),
        }
    except Exception as e:
        report["index"] = {"error": type(e).__name__ + ": " + str(e)}

    # ---------- 逐题跑检索，并检查记录格式 ----------
    for case in cases:
        question = case.get("question", "")

        try:
            chunks = retrieve_fn(question, top_k)
        except Exception as e:
            chunks = []
            report["retrieval"]["per_case"].append({
                "id": case.get("id"),
                "question": question,
                "error": type(e).__name__ + ": " + str(e),
                "sources": [],
            })
            continue

        if not chunks:
            # 检索为空 → rag.py 会直接拒答，不调模型。该答题若走到这一步就是失败的。
            report["retrieval"]["empty"].append(case.get("id"))

        report["retrieval"]["per_case"].append({
            "id": case.get("id"),
            "category": case.get("category"),
            "question": question,
            "sources": [c.get("source") for c in chunks],
        })

        # 检查记录格式：拿这一题的检索结果组装一条记录，看字段齐不齐
        record = build_record(case, chunks, None, 0)
        missing = [k for k in REQUIRED_RECORD_KEYS if k not in record]
        if missing:
            report["record_shape_ok"] = False
            report["record_shape_problems"].append(
                str(case.get("id")) + " 的记录缺少字段：" + "、".join(missing)
            )
        else:
            # 顺便验证记录能 JSON 序列化（要写进结果文件的）
            try:
                json.dumps(record, ensure_ascii=False)
            except Exception as e:
                report["record_shape_ok"] = False
                report["record_shape_problems"].append(
                    str(case.get("id")) + " 的记录无法序列化：" + str(e)
                )

    # ---------- 运行环境 ----------
    # 【关键：先读 .env，再去判断密钥在不在】
    # 这一步以前是缺的，导致的 bug 是：
    #   密钥只写在 .env 里的人 → 网页能正常用，评测器却报告「密钥未设置」。
    # 同一份配置、两个入口两种结论，而且两边单独看都「没错」，特别难查。
    # 现在两个入口都调 env_utils.load_dotenv，取料规则统一。
    env_info = load_dotenv(ENV_PATH)

    # 【只报告「有没有」，绝不打印密钥本身】
    # env_file_keys 里是【键名】不是值——"DEEPSEEK_API_KEY" 这串字本身不保密，
    # 它就写在公开的 .env.example 里；要保护的是等号后面那个值。
    report["environment"] = {
        "python": sys.version.split()[0],
        "has_api_key": bool(os.environ.get("DEEPSEEK_API_KEY", "")),
        "env_file": ENV_PATH,
        "env_file_loaded": env_info["loaded"],
        "env_file_keys": env_info["keys"],
        "env_file_read_error": env_info["read_error"],
        "cases_file": CASES_PATH,
        "results_dir": RESULTS_DIR,
        "results_dir_exists": os.path.isdir(RESULTS_DIR),
    }

    return report


# ===================== live：真的要花钱的那部分 =====================

def run_live(cases, client, model, top_k=DEFAULT_TOP_K, retrieve_fn=None):
    """逐题：检索 → 生成并校验 → 判分。会真实调用模型。

    client 从外面传进来，方便测试时换成一个假的。
    """
    if retrieve_fn is None:
        retrieve_fn = R.retrieve

    records = []

    for case in cases:
        question = case.get("question", "")

        start = time.perf_counter()

        chunks = []          # 先摆好默认值，万一检索就炸了，后面组装记录时也有东西可用
        try:
            chunks = retrieve_fn(question, top_k)
            result = G.generate_answer(question, chunks, client, model)
        except Exception:
            # 【不记录原始异常】异常里可能含内部细节、请求内容甚至密钥片段。
            # 这里只留一句「这一题出错了」，具体原因去看 rag.py 的降级路径。
            result = {"decision": "insufficient_evidence",
                      "answer": "（这一题执行出错，已跳过）",
                      "citations": []}

        elapsed_ms = int(round((time.perf_counter() - start) * 1000))

        record = build_record(case, chunks, result, elapsed_ms)
        record.update(score_case(case, result))
        records.append(record)

    return summarize(records, mode="live", model=model, top_k=top_k)


# ===================== 存结果 =====================

def save_results(report):
    """把整份报告写成 evals/results/<时间戳>-<模式>.json。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = report.get("time", "unknown") + "-" + report.get("mode", "unknown")
    path = os.path.join(RESULTS_DIR, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


# ===================== 建客户端 =====================

def make_client():
    """只有 --live 才会走到这里。密钥从环境变量读，绝不打印。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("错误：没有找到 DeepSeek 密钥。")
        print("请先设置环境变量 DEEPSEEK_API_KEY，方法见 README.md 的「配置 API 密钥」一节。")
        sys.exit(1)

    from openai import OpenAI          # 延迟导入：dry-run 时根本不需要这个库
    return OpenAI(api_key=key, base_url=BASE_URL)


# ===================== 控制台加固 =====================

def harden_console():
    """让程序在「装不下某些字符」的终端上也能跑完，而不是中途崩掉。

    【为什么需要这一步 —— 这是真踩过的坑】
    中文版 Windows 的控制台默认用 GBK 编码，而 GBK 字符集里【没有】✓ ✗ ⚠ 这类符号。
    一旦 print 出去，Python 会抛 UnicodeEncodeError，整个评测直接中断——
    而且是在「检查都通过了、正要开始干活」的时候断的，非常误导。

    修法的第一层当然是把这些符号换成纯 ASCII 的 [OK] / [FAIL] / [WARN]（本文件已经这么做了）。
    但光换符号不够，因为：

      1. 以后有人加一行带 emoji 的日志，同一个坑会再踩一次；
      2. --live 时打印的内容里可能混进模型返回的任意字符（模型完全可能吐个 emoji 出来）。

    所以这里再加一道兜底：把标准输出/报错的「编码错误策略」从默认的
    strict（遇到装不下的字符就抛异常）改成 replace（装不下就印个 ?，继续跑完）。

    【代价】万一真有字符装不下，你会看到 ?，而不是当场报错。
    这个代价是合算的：跑完一场评测、拿到结果，比为了一个字符中断强。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            # 有些环境（比如输出被重定向成非文本流）不支持 reconfigure。
            # 那就跳过——原本的符号替换已经能应付绝大多数情况了。
            pass


# ===================== 打印 =====================

def print_dry_report(report):
    """把 dry-run 的结果打到终端。"""
    print("=" * 68)
    print("  dry-run：只检查，不调用任何模型")
    print("=" * 68)

    print("\n【题库】" + str(report["total"]) + " 道题")
    if report["case_problems"]:
        print("  [FAIL] 发现问题：")
        for p in report["case_problems"]:
            print("     · " + p)
    else:
        print("  [OK] 格式全部正常")

    idx = report["index"]
    print("\n【索引】")
    if idx.get("error"):
        print("  [FAIL] 建索引失败：" + idx["error"])
    else:
        print("  [OK] 片段 " + str(idx.get("chunks")) + " 段，来自 " + str(len(idx.get("sources", []))) + " 份资料")
        for s in idx.get("sources", []):
            print("     · " + s)
        print("  词表大小：" + str(idx.get("vocab_size")))
        print("  README 是否被误索引：" + ("[FAIL] 是（错误！）" if idx.get("readme_indexed") else "[OK] 否"))

    print("\n【逐题检索】")
    for item in report["retrieval"]["per_case"]:
        if item.get("error"):
            print("  [FAIL] " + str(item["id"]) + "  " + item["error"])
            continue
        # 标记用纯 ASCII 并补齐宽度，这样后面几列还能对齐
        mark = "[OK]" if item["sources"] else "[EMPTY]"
        print("  " + mark.ljust(8) + str(item["id"]).ljust(28) + str(item.get("category", "")).ljust(14)
              + "-> " + ("、".join(item["sources"]) if item["sources"] else "（空）"))

    empty = report["retrieval"]["empty"]
    if empty:
        print("\n  [WARN] 有 " + str(len(empty)) + " 道题检索结果为空（这些题会走拒答，不调模型）：")
        print("     " + "、".join(str(x) for x in empty))

    print("\n【结果记录格式】")
    if report["record_shape_ok"]:
        print("  [OK] " + str(len(REQUIRED_RECORD_KEYS)) + " 个必需字段齐全，且都能序列化成 JSON")
    else:
        for p in report["record_shape_problems"]:
            print("  [FAIL] " + p)

    env = report["environment"]
    print("\n【运行环境】")
    print("  Python：" + str(env.get("python")))
    # 【只报状态，绝不回显密钥内容】
    print("  DeepSeek 密钥：" + ("已设置 [OK]（只检查有无，不显示内容）" if env.get("has_api_key") else "未设置 [FAIL]（--live 会失败）"))
    print("  .env 文件：" + str(env.get("env_file")) + "  "
          + ("已读取 [OK]，读到 " + str(len(env.get("env_file_keys") or [])) + " 个键："
             + "、".join(env.get("env_file_keys") or []) if env.get("env_file_loaded")
             else "未读取（文件不存在，或读取出错——此时只能靠真实环境变量）"))
    print("  题库路径：" + str(env.get("cases_file")))
    print("  结果目录：" + str(env.get("results_dir"))
          + ("（已存在）" if env.get("results_dir_exists") else "（尚不存在，--live 时自动创建）"))

    print("\n" + "=" * 68)
    print("  dry-run 结束。没有调用任何模型，没有产生任何费用。")
    print("  确认无误后，用下面这条命令跑真实评测：")
    print("      python -m evals.run_rag_eval --live")
    print("=" * 68)


def print_live_report(report):
    """把 live 的结果打到终端。"""
    counts = report["counts"]

    print("\n" + "=" * 68)
    print("  评测结果")
    print("=" * 68)
    print("  总题数：" + str(report["total"]))
    print("  严格通过 (strict_pass)：" + str(counts["strict_pass"]))
    print("  安全通过 (safe_pass)：" + str(counts["safe_pass"]))
    print("  不安全（必须处理）：" + str(counts["failed"]))

    print("\n【按分类】")
    for cat, b in sorted(report["by_category"].items()):
        print("  " + cat.ljust(16) + "共 " + str(b["total"]).rjust(2)
              + "   严格通过 " + str(b["strict_pass"]).rjust(2)
              + "   安全通过 " + str(b["safe_pass"]).rjust(2))

    if report["failures"]:
        print("\n【失败清单（不安全，必须逐条看）】")
        for f in report["failures"]:
            print("  [FAIL] [" + str(f["category"]) + "] " + str(f["id"]))
            print("      问题：" + str(f["question"]))
            print("      原因：" + str(f["reason"]))
    else:
        print("\n【失败清单】空 —— 没有出现不安全的输出")

    if report.get("safe_misses"):
        print("\n【安全但没达标（不危险，属于能力问题）】")
        for f in report["safe_misses"]:
            print("  · [" + str(f["category"]) + "] " + str(f["id"]) + " —— " + str(f["reason"]))


# ===================== 主流程 =====================

def main(argv=None):
    # 【第一步就加固控制台】任何一行输出都可能遇上装不下的字符，
    # 所以在打印任何东西之前，先把兜底打开。
    harden_console()

    parser = argparse.ArgumentParser(
        prog="python -m evals.run_rag_eval",
        description="完整 RAG 评测器：拿 33 道题考一遍「检索 + 生成 + 引用校验」。",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true",
                       help="只检查题库、检索、结果格式和运行环境，不调用模型（默认行为）")
    group.add_argument("--live", action="store_true",
                       help="【会产生真实费用】逐题调用 DeepSeek，33 次")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="每题检索几段资料，默认 " + str(DEFAULT_TOP_K))
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 题（调试用，0 表示全部）")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.limit and args.limit > 0:
        cases = cases[:args.limit]

    # ---------- dry-run ----------
    if not args.live:
        report = run_dry(cases, top_k=args.top_k)
        print_dry_report(report)
        return 0

    # ---------- live ----------
    print("=" * 68)
    print("  【live 模式】即将产生真实 DeepSeek 调用")
    print("=" * 68)
    print("  本次会对 " + str(len(cases)) + " 道题各调用 1 次模型（检索结果为空的那几题不会调用）。")
    print("  这会真实联网、真实计费。")
    print("=" * 68 + "\n")

    # 花钱之前先做一遍不花钱的检查，能省一次是一次
    problems = validate_cases(cases)
    if problems:
        print("题库有问题，先修好再跑：")
        for p in problems:
            print("  · " + p)
        return 1

    # 【和 app.py 用同一份配置】先把项目根目录的 .env 读进来，再判断密钥在不在。
    # 不读这一步的话，密钥只写在 .env 里的人跑 --live 会被下面拦下，
    # 而网页却能正常用——正是本轮要修的那个不一致。
    load_dotenv(ENV_PATH)

    if not os.environ.get("DEEPSEEK_API_KEY", ""):
        print("错误：没有找到 DeepSeek 密钥，--live 跑不了。")
        print("两种设法，任选一种：")
        print("  1. 在项目根目录建一个 .env，写上 DEEPSEEK_API_KEY=你的密钥（模板见 .env.example）")
        print("  2. 设置环境变量 DEEPSEEK_API_KEY")
        print("详细说明见 README.md 的「配置 API 密钥」一节。")
        return 1

    client = make_client()
    report = run_live(cases, client, MODEL, top_k=args.top_k)
    print_live_report(report)

    path = save_results(report)
    print("\n结果已保存：" + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
