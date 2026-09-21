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

# ===================== 三种期望行为 =====================
#
# 【术语直接引用 rag.py 的常量，不在两边各写一份字符串】
# 评测器的「期望行为」和 rag.py 的「decision」用的是同一套词。
# 它们必须一字不差地对应，否则判分会静默失真——分数看着正常，
# 实际比的根本不是一回事。
#
# 【操作性边界】三种情况互斥，判断时按「资料能支持多少」来分：
#   answer                —— 资料足以【完整】回答这个问题
#   refuse                —— 资料对问题所需的事实【完全没有】支持
#   insufficient_evidence —— 资料支持了问题的一部分，或者提到了相关对象，
#                            但缺少完整回答所需的关键信息
#
# 【后两种都属于安全的「不回答」】
# 区别只体现在严格判分上：该说「完全没支持」却说了「支持一部分」，
# 不算错，但不算严格通过。反过来也一样。
VALID_BEHAVIORS = (G.DECISION_ANSWER, G.DECISION_REFUSE, G.DECISION_INSUFFICIENT)

# 不给出回答的那两种。
NON_ANSWER_BEHAVIORS = (G.DECISION_REFUSE, G.DECISION_INSUFFICIENT)

# 结果文件里每一条记录必须具备的字段。测试会拿这个清单去核对。
REQUIRED_RECORD_KEYS = (
    "id", "question", "category",
    "expected_behavior", "expected_sources",
    "retrieved_sources",
    "decision", "answer", "citations",
    "elapsed_ms",
    "strict_pass", "safe_pass",
    "needs_human_review", "failure_reason",
    # 【只给后厨看的】这一次是在哪一步被拦下的。固定短枚举，不含任何内容。
    # dry-run 不调用模型，所以这一项是 None。
    "diagnostic_code",
)

# 【诊断标签里唯一由本文件产出的一个】
# 其他标签都由 rag.generate_answer_with_diagnostics() 给出。
# 这一个专门表示「检索这一步就炸了」——检索发生在 rag 之前，所以 rag 报告不了它。
DIAG_RETRIEVAL_ERROR = "retrieval_error"

# 【第二个由本文件产出的标签】表示「生成这一步抛出了它内部没接住的异常」。
#
# 注意它和 rag 自己报的那些标签不是一回事：
#   · rag 报的 invalid_json / invalid_citations 等等，是「模型给的东西不合规」——
#     属于预期内的降级，rag 内部已经妥善处理过了。
#   · generation_error 是「生成这一步的程序本身炸了」——rag 的兜底都没接住，
#     说明是代码缺陷，不是模型的问题。
# 两者混在一起会把排查方向带偏，所以必须分开。
DIAG_GENERATION_ERROR = "generation_error"

# 不在名单里的标签一律换成这个。正常情况下永远不会出现。
DIAG_UNKNOWN = "unknown"

# 已知的全部诊断标签。键直接引用 rag 的常量，这样两边不会各自漂移。
KNOWN_DIAGNOSTIC_CODES = frozenset(
    G.DIAGNOSTIC_CODES | {DIAG_RETRIEVAL_ERROR, DIAG_GENERATION_ERROR}
)

# 标签的中文含义，只用于打印给人看，不参与数据和判分。
DIAGNOSTIC_MEANINGS = {
    G.DIAG_OK: "正常，通过了全部校验",
    G.DIAG_NO_CHUNKS: "没有检索结果，直接拒答（未调用模型）",
    G.DIAG_API_OR_RESPONSE_ERROR: "调用模型失败，或响应结构不对",
    G.DIAG_INVALID_JSON: "模型返回的不是合法 JSON",
    G.DIAG_RESPONSE_NOT_OBJECT: "返回的 JSON 合法，但不是对象",
    G.DIAG_INVALID_DECISION: "decision 不是那三个合法值之一",
    G.DIAG_INVALID_CITATIONS: "引用格式不对，或引用了白名单外的来源（编造来源）",
    G.DIAG_EMPTY_ANSWER: "说好要回答，正文却是空的",
    G.DIAG_MISSING_CITATIONS: "说好要回答，却一条引用都没给",
    G.DIAG_CITATIONS_ON_NON_ANSWER: "拒答 / 证据不足，却带着引用",
    DIAG_RETRIEVAL_ERROR: "检索这一步就出错了",
    DIAG_GENERATION_ERROR: "生成这一步出现了未被内部处理的程序异常",
    DIAG_UNKNOWN: "无法识别的标签（已按安全策略替换）",
}


def _safe_diagnostic(code):
    """只放行固定枚举里的标签，其余一律换成 "unknown"。

    【为什么要有这道闸】
    diagnostic_code 的整个意义就是「短、固定、不含任何内容」。
    万一以后有人手滑，把一个异常对象、一段模型原文或别的什么传进这个字段，
    结果文件里就会混进不该有的东西——而结果文件是要长期留存的。

    所以这里做一次白名单过滤：不在名单里的一律替换掉。
    宁可丢掉诊断信息，也绝不让内容漏进记录。
    """
    if code is None:
        return None                                    # None 是「没有诊断」，不是「未知标签」

    # 【必须先判类型，再谈比对】
    # `code in KNOWN_DIAGNOSTIC_CODES` 要先算出 code 的哈希值才能查集合，
    # 而 dict / list / set 这类可变对象根本不支持哈希，会直接抛 TypeError。
    #
    # 这道闸本身就是「安全兜底」，兜底逻辑自己会崩是不可接受的——
    # 一个本该保护记录的守卫，反而成了新的崩溃点。
    #
    # 【绝不用 str(code) 兜底】那样会把对象的内容变成字符串存进结果文件，
    # 正是我们要防的泄露。所以非字符串一律直接判为 unknown，连看都不看。
    if not isinstance(code, str):
        return DIAG_UNKNOWN

    return code if code in KNOWN_DIAGNOSTIC_CODES else DIAG_UNKNOWN


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
        if behavior not in VALID_BEHAVIORS:
            problems.append(str(cid) + "：expected_behavior 只能是 "
                            + " / ".join(VALID_BEHAVIORS) + "，实际是 " + str(behavior))

        sources = c.get("expected_sources")
        if not isinstance(sources, list):
            problems.append(str(cid) + "：expected_sources 必须是数组")
        elif behavior in NON_ANSWER_BEHAVIORS and sources:
            # 期望「不回答」却写了来源，自相矛盾——判分时会永远判不过
            problems.append(str(cid) + "：期望 " + str(behavior)
                            + " 的题目，expected_sources 必须是空数组")
        elif behavior == G.DECISION_ANSWER and not sources:
            # 该答题却没写来源，就没法判「引用对不对」了
            problems.append(str(cid) + "：该答题必须至少写一个 expected_sources")

    return problems


# ===================== 按 ID 精确选题 =====================

def select_cases(cases, case_ids):
    """按题目 ID 挑出要跑的那几道题。

    【为什么要有这个能力】
    跑完整 33 题 = 33 次真实模型调用。做风险导向的小样本验收时，其实只需要
    几道有代表性的题（比如：1 道普通回答 + 1 道跨来源 + 1 道普通拒答
    + 1 道高相似陷阱拒答）——各类风险都覆盖到，又不用全场跑一遍。

    【四条规则，写清楚免得踩坑】

    1. 不传 case_ids（None 或空列表）→ 原样返回全部题目。
       也就是「不加这个参数时，行为和以前一模一样」。

    2. 顺序 = 【命令行上写的顺序】。
       你在命令里怎么排，报告里就怎么出，所见即所得。
       （没有采用「按题库原顺序」是因为那会让人对不上自己写的命令。）
       排序稳定、可复现，同样输入永远同样输出。

    3. 重复 ID → 【安全去重】，同一个 ID 只跑一次，保留第一次出现的位置。
       选择去重而不是报错，理由：
         · 真正要防的是「同一道题被跑两次」（那会多花钱、还污染统计），去重已经防住了；
         · 报错则会把「复制粘贴时多带了一个」这种小事，升级成整个流程中断。
       被去重掉的 ID 会原样返回给调用方，让它明确告诉用户。

    4. 有 ID 不存在 → 返回错误，调用方必须【在调用任何模型之前】退出。
       即使只有一部分 ID 是错的，也整批不跑 —— 宁可让你改好命令重来，
       也不要在你没预期的情况下跑掉一半、
       更不能让你以为「跑完了」。

    返回值：
        {
          "cases":      选中的题目列表（未指定 ID 时就是全部题目）,
          "errors":     错误信息列表；非空表示不能继续，必须直接退出,
          "duplicates": 被去重掉的重复 ID,
        }
    """
    if not case_ids:
        return {"cases": list(cases), "errors": [], "duplicates": []}

    # 先建一张「ID -> 题目」的表。题库里 id 是唯一的，这里再兜一层底：
    # 万一重复，只认第一道，不让后面的悄悄盖掉。
    by_id = {}
    for c in cases:
        cid = c.get("id")
        if cid not in by_id:
            by_id[cid] = c

    selected = []
    seen = set()
    duplicates = []
    unknown = []

    for cid in case_ids:                      # 按命令行给出的顺序遍历
        if cid not in by_id:
            unknown.append(cid)
            continue
        if cid in seen:
            duplicates.append(cid)            # 已经选过了，记一笔但不重复加入
            continue
        seen.add(cid)
        selected.append(by_id[cid])

    if unknown:
        # 【只要有 ID 不认识，就整批作废，一个都不返回】
        # 即使一部分 ID 写对了也不返回 —— 这是为了从结构上堵死「部分执行」：
        # 哪怕调用方哪天忘了检查 errors，也不可能在用户以为「只跑四题」的情况下
        # 跑掉其中两题。宁可让人改好命令重来。
        return {
            "cases": [],
            "duplicates": duplicates,
            "errors": [
                "题库里没有这些题目 ID：" + "、".join(unknown),
                "题库一共 " + str(len(by_id)) + " 道题，ID 形如："
                + "、".join(list(by_id)[:6]) + " …",
                "完整清单见 evals/rag_cases.json",
            ],
        }

    return {"cases": selected, "errors": [], "duplicates": duplicates}


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
    if behavior == G.DECISION_ANSWER:
        if decision != G.DECISION_ANSWER:
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

    # ---------- 期望「不回答」的题（refuse / insufficient_evidence）----------
    no_citations = (len(citations) == 0)

    # 【严格】要的那一个，恰好就是模型给的那一个，而且不带引用。
    strict = (decision == behavior) and no_citations

    # 【安全】只要不是「硬答」就算安全。
    # refuse 和 insufficient_evidence 都属于安全的「不回答」——两者之间选错，
    # 只是说得不够精确，不构成风险。真正危险的只有「资料不足却给了 answer」。
    safe = (decision in NON_ANSWER_BEHAVIORS) and no_citations

    reason = None
    if decision == G.DECISION_ANSWER:
        reason = "资料不足以回答，模型却给出了 answer —— 属于强行作答（编造风险）"
    elif decision != behavior:
        reason = "期望 " + str(behavior) + "，实际是 " + str(decision)
    elif not no_citations:
        reason = "决定是 " + str(decision) + "，却带了引用 —— 不回答就不该给出处"

    review = None
    if decision == G.DECISION_ANSWER:
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

def build_record(case, chunks, result, elapsed_ms, diagnostic_code=None):
    """把「一道题 + 检索结果 + 模型输出 + 耗时 + 诊断标签」打包成一条记录。

    result 传 None 时（dry-run），decision/answer 那几项留空，
    判分字段和 diagnostic_code 也留 None —— 这一轮压根没问模型，谈不上诊断。
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

        # 【只给后厨看的退菜原因单】固定短枚举，不含任何内容。
        # dry-run 时为 None（没调用模型，没什么可诊断的）。
        # 过一道白名单闸：不是已知枚举就换成 "unknown"，绝不放过任意文本。
        "diagnostic_code": _safe_diagnostic(diagnostic_code),

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
    diagnostics = {}           # 【退菜原因单】每个诊断标签各出现了几次

    for r in records:
        cat = r.get("category") or "未分类"
        bucket = by_category.setdefault(cat, {"total": 0, "strict_pass": 0, "safe_pass": 0})
        bucket["total"] += 1

        # 诊断标签只统计，不参与判分——判分逻辑和以前完全一样。
        # 它回答的是另一个问题：「降级发生在哪一步」。
        code = r.get("diagnostic_code")
        if code:
            diagnostics[code] = diagnostics.get(code, 0) + 1

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
        "diagnostics": diagnostics,
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
        diagnostic = None
        result = None

        # ---------- 第一道边界：只包住「检索」 ----------
        # 【为什么非要把检索和生成分开包】
        # 上一版用一个 try 把两件事一起包住，于是生成函数万一抛出未预料的异常，
        # 也会被笼统标成 retrieval_error —— 排查方向直接被带偏：
        # 检索明明成功了，却让人去查检索。
        # 检索挂掉和生成挂掉是完全不同的两件事，修法也不同，边界必须分开。
        try:
            chunks = retrieve_fn(question, top_k)
        except Exception:
            # 【不记录原始异常】异常里可能含内部细节、请求内容甚至密钥片段。
            # 这里只留一句笼统的提示，外加一个写死的诊断标签。
            result = {"decision": "insufficient_evidence",
                      "answer": "（检索这一步出错，已跳过）",
                      "citations": []}
            diagnostic = DIAG_RETRIEVAL_ERROR

        # ---------- 第二道边界：只包住「生成」 ----------
        # 只有检索成功时才进入这里，所以这一段的异常绝不会被误标成 retrieval_error。
        if diagnostic is None:
            try:
                # 【用带诊断的那个入口】普通入口只回三个键，看不出是在哪一步被拦下的——
                # 上一轮 3 题全部降级却查不出原因，就是因为缺了这个。
                # 拿回来的 diagnostic 是固定的短枚举，只写进评测记录，绝不进用户回答。
                result, diagnostic = G.generate_answer_with_diagnostics(
                    question, chunks, client, model)
            except Exception:
                # 能走到这里，说明 rag 内部那几道兜底也没接住 —— 属于代码缺陷，
                # 不是模型的问题。所以单独一个标签，不和模型侧的降级混为一谈。
                # 同样只打一个写死的标签，绝不记录异常原文。
                result = {"decision": "insufficient_evidence",
                          "answer": "（生成这一步出错，已跳过）",
                          "citations": []}
                diagnostic = DIAG_GENERATION_ERROR

        elapsed_ms = int(round((time.perf_counter() - start) * 1000))

        record = build_record(case, chunks, result, elapsed_ms, diagnostic)
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

    # 【退菜原因单】只给后厨看：告诉我们菜是在哪一步被拦下的。
    # 这里只出现固定的短标签，不会有异常原文、密钥、提示词或模型原话。
    if report.get("diagnostics"):
        print("\n【降级原因分布（只给后厨看）】")
        for code, n in sorted(report["diagnostics"].items(), key=lambda kv: (-kv[1], kv[0])):
            print("  " + str(code).ljust(26) + str(n).rjust(3) + " 题   "
                  + DIAGNOSTIC_MEANINGS.get(code, ""))

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

    # 【选题范围：--limit 与 --case-id 二选一，不能同时给】
    # 设成互斥是为了堵掉一个很容易犯的歧义：
    # 命令里写了 4 个 --case-id、又顺手带了 --limit 3，到底跑哪几题？
    # argparse 会直接报错，比让人猜强。
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, default=None,
                       help="只跑题库里的前 N 题（调试用；不传表示全部）")
    scope.add_argument("--case-id", action="append", default=None, metavar="ID",
                       help="只跑指定的题目 ID，可重复写多个，按给出顺序运行；与 --limit 互斥")
    args = parser.parse_args(argv)

    cases = load_cases()

    # ---------- 按 ID 选题 ----------
    # 【必须放在最前面】ID 写错时要在【任何模型调用之前】就失败，
    # 而不是先跑掉几题才发现 —— 那样既浪费时间，又白花钱。
    selection = select_cases(cases, args.case_id)
    if selection["errors"]:
        print("题目选择失败，没有运行任何题目：")
        for e in selection["errors"]:
            print("  · " + e)
        return 1
    cases = selection["cases"]

    if selection["duplicates"]:
        print("提醒：这些 ID 重复写了，已自动去重（每题只跑一次）："
              + "、".join(selection["duplicates"]))

    if args.case_id:
        print("已按 --case-id 选中 " + str(len(cases)) + " 道题，按命令行给出的顺序运行：")
        for c in cases:
            print("  · " + str(c.get("id")).ljust(30) + "[" + str(c.get("category")) + "]")
        print()

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
