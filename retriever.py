# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# 本地检索层 —— RAG 的第一步
#
# 【这个文件干什么】
# 把 knowledge_base/ 里的教学资料切成小段，然后针对用户的问题，
# 找出最相关的几段，并且告诉你每一段出自哪个文件、哪个标题。
#
# 「先检索、再让模型照着检索结果回答」——这就是 RAG 的核心思路。
# 本轮只做「检索」这一半，不接模型。
#
# =====================================================================
# 【诚实说明：这是什么检索，不是什么检索】
#
# 这是第一版的「词法检索」（lexical retrieval），也就是**关键词匹配**。
# 它**不是向量检索**，也不是语义检索。
#
# 它做的事很朴素：把问题和每一段资料都切成词，看哪些词重合得多，
# 重合得多、而且那些词越罕见，分数就越高。
#
# 优点：
#   · 不需要任何额外服务（不用向量数据库、不用 embedding 接口）
#   · 不花一分钱、完全离线，断网也能跑、也能测
#   · 结果可解释——你能明确说出「因为这两个词命中了，所以这段排第一」
#   · 极快，几十段资料是毫秒级
#
# 局限（很重要，别指望它做不到的事）：
#   · 不理解同义词。用户问「英文时态」，资料里写的是「现在完成时」，
#     两者一个词都不重合，它就找不到——哪怕意思上明显相关
#   · 不理解复杂语义和指代。问「它和另一个有什么区别」，「它」指什么，它不知道
#   · 换个说法就可能检索不到，对用词很敏感
#   · 中文是按「相邻两个字」切的，偶尔会切出没有意义的组合
#
# 后续会升级成向量检索（把文字变成向量，比「意思」而不是比「字面」）。
# 但词法检索不会白做：它简单、可解释，是很好的对照基线，
# 而且升级之后通常会「词法 + 向量」混着用，各补各的短板。
# =====================================================================

import os      # 列目录、拼路径
import re      # 正则：切标题、切词
import math    # 算 BM25 里的对数（log）

# 本文件所在目录。用它拼路径，不管从哪个目录运行都找得到知识库
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 知识库目录
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")

# 【必须排除的文件】knowledge_base/README.md 是语料库自己的说明文档，
# 不是教学内容。如果把它也索引进去，用户问「RAG 是什么」这类问题时，
# 可能检索到说明文档里的片段并当作「依据」引用——那是错的。
EXCLUDED_FILES = {"readme.md"}      # 统一转小写比较，避免大小写写错漏掉

# 二级标题：行首两个 # 加空格。注意这不会匹配到三个 # 的 ### 小标题
H2_RE = re.compile(r"^##\s+(.*)$")

# 文档一级标题：行首一个 # 加空格
H1_RE = re.compile(r"^#\s+(.*)$")

# 默认返回几段
DEFAULT_TOP_K = 3

# ---- BM25 的两个可调参数（用的是文献里的常见取值）----
# K1 控制「同一个词出现很多次，分数还能涨多少」——涨到一定程度就饱和，防止刷词
# B  控制「长段落要不要惩罚」——0 是不惩罚，1 是完全按长度归一化
K1 = 1.5
B = 0.75

# ---- 字段权重 ----
# 标题里的词是强信号，重复算几次，相当于给标题加权。
# 用「重复计数」而不是另写一套加权公式，是为了保持代码简单、可解释。
HEADING_WEIGHT = 3      # 二级标题里的词，算 3 遍
DOC_TITLE_WEIGHT = 2    # 整份文档的一级标题里的词，算 2 遍


# ===================== 分词 =====================

# 英文单词 / 数字：连续的字母数字算一个词
ASCII_WORD_RE = re.compile(r"[a-z0-9]+")

# 中文片段：连续的中文字符。
# 用 一-鿿 这种转义写法，比直接写汉字更清楚，也不怕文件编码出问题。
CJK_RUN_RE = re.compile("[一-鿿]+")


def tokenize(text):
    """把一段文字切成「可以互相比较的词」。

    中文和英文的切法不一样：

    · 英文有空格，直接按空格（准确地说是按字母数字）切就行：
        "Have you ever been"  ->  ["have", "you", "ever", "been"]

    · 中文没有空格，而且要装分词库就得引入第三方依赖（本轮不允许）。
      所以这里用最经典的无依赖办法：**相邻两个字切成一片**（bigram）。
        "现在完成时"  ->  ["现在", "在完", "完成", "成时"]

      为什么用两个字而不是一个字？单个汉字太常见了（"的"、"是"、"了"），
      几乎每段都有，没法区分。两个字的组合已经有相当的区分度。
      比如「完成」比「成」有用得多。
    """
    text = text.lower()                                     # 统一转小写，这样 Have 和 have 算同一个词
    tokens = []

    for word in ASCII_WORD_RE.findall(text):                # 先把英文单词和数字挑出来
        tokens.append(word)

    for run in CJK_RUN_RE.findall(text):                    # 再处理中文片段
        if len(run) == 1:                                   # 只有一个字，没法组 bigram，就单独算一个词
            tokens.append(run)
        else:
            for i in range(len(run) - 1):                   # 挨着的两个字组成一个词
                tokens.append(run[i:i + 2])

    return tokens


# ===================== 读取与切分 =====================

def load_chunks(kb_dir=KB_DIR):
    """读取知识库里的 .md 文件，按二级标题（##）切分成片段。

    返回一个列表，每一项是一个片段字典：
        {
            "source":    文件名，比如 "grammar_present_perfect.md"
            "heading":   二级标题，比如 "基本结构"
            "text":      这一段正文
            "doc_title": 整份文档的一级标题，比如 "现在完成时（Present Perfect）"
        }
    """
    chunks = []

    for filename in sorted(os.listdir(kb_dir)):             # sorted 保证每次顺序一样，结果才稳定
        if not filename.lower().endswith(".md"):            # 只要 Markdown 文件
            continue
        if filename.lower() in EXCLUDED_FILES:              # 【关键】说明文档绝不索引
            continue

        path = os.path.join(kb_dir, filename)

        # 用 utf-8-sig 读：万一文件头部混进了看不见的 BOM 字符，它会自动去掉。
        # （这个项目早前在 BOM 上踩过坑，所以这里主动防一手）
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()

        # 整份文档的标题，后面用它给检索加权
        title_match = H1_RE.search(text)
        doc_title = title_match.group(1).strip() if title_match else filename

        current = None                                      # 当前正在收集的片段

        for line in text.splitlines():
            m = H2_RE.match(line)                           # 碰到二级标题，就开一个新片段
            if m:
                current = {
                    "source": filename,
                    "heading": m.group(1).strip(),
                    "text": [],
                    "doc_title": doc_title,
                }
                chunks.append(current)
                continue

            # 二级标题之前的内容（文档大标题、"来源/用途"那段声明）不属于任何片段，
            # 直接丢掉。理由：那段声明三份文件几乎一模一样，
            # 留着只会给每个片段加上一堆相同的词，纯属干扰。
            if current is not None:
                current["text"].append(line)

        # 把收集到的行拼成整段文字，去掉首尾空白
        for c in chunks:
            if isinstance(c["text"], list):
                c["text"] = "\n".join(c["text"]).strip()

    return chunks


# ===================== 检索器 =====================

class Retriever:
    """把切好的片段建成索引，然后回答问题。

    用法：
        r = Retriever()
        results = r.retrieve("since 和 for 有什么区别？", top_k=3)
    """

    def __init__(self, kb_dir=KB_DIR):
        self.kb_dir = kb_dir
        self.chunks = load_chunks(kb_dir)                   # 切好的所有片段
        self._build()                                       # 建索引

    # ---------- 建索引 ----------

    def _build(self):
        """预先算好每个片段的词频、长度，以及每个词有多罕见。"""
        self._tf = []                                       # 每个片段的词频表：{词: 出现次数}
        self._lengths = []                                  # 每个片段的总词数

        for c in self.chunks:
            # 把标题和文档标题的词重复算几遍 —— 这就是前面说的字段加权
            tokens = (
                tokenize(c["heading"]) * HEADING_WEIGHT
                + tokenize(c["doc_title"]) * DOC_TITLE_WEIGHT
                + tokenize(c["text"])
            )

            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self._tf.append(freq)
            self._lengths.append(len(tokens))

        n = len(self.chunks)
        self._avg_len = (sum(self._lengths) / n) if n else 0.0   # 平均长度，BM25 里要用

        # 每个词出现在多少个片段里（文档频率 df）
        df = {}
        for freq in self._tf:
            for t in freq:
                df[t] = df.get(t, 0) + 1

        # IDF：一个词越罕见，说明它越有区分度，权重越大。
        # 用 BM25 常用的那个公式（加 1 和 0.5 是为了防止除以零和出现负数）。
        self._idf = {}
        for t, d in df.items():
            self._idf[t] = math.log(1 + (n - d + 0.5) / (d + 0.5))

    @property
    def vocab_size(self):
        """索引里一共有多少个不重复的词。调试时看一眼心里有数。"""
        return len(self._idf)

    @property
    def sources(self):
        """索引里出现过哪些文件。测试用它来确认有没有混进不该索引的文件。"""
        return sorted({c["source"] for c in self.chunks})

    # ---------- 检索 ----------

    def retrieve(self, query, top_k=DEFAULT_TOP_K):
        """找出和 query 最相关的 top_k 个片段。

        返回一个列表，每项是片段内容，外加一个 score 字段（分数越高越相关）。
        找不到任何相关片段时返回空列表 —— 【不会】硬凑一个出来。
        """
        if not self.chunks or not query:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            # 问题是空的，或者只有标点符号之类切不出词的东西。
            # 这时候不能瞎返回 —— 返回空列表才是诚实的。
            return []

        # 同一个词在问题里出现两次，不该让它的权重翻倍，所以先去重
        q_terms = set(q_tokens)

        scored = []
        for i, freq in enumerate(self._tf):
            score = 0.0

            for t in q_terms:
                f = freq.get(t, 0)
                if f == 0:
                    continue                                    # 这段里没有这个词，跳过

                idf = self._idf.get(t, 0.0)                     # 词越罕见，idf 越大

                # BM25 的核心公式。
                # 分母那坨的意思是：同一个词重复出现，分数会涨，但涨到一定程度就饱和；
                # 同时长段落会被轻微惩罚，免得「篇幅长所以命中多」占便宜。
                denom = f + K1 * (1 - B + B * self._lengths[i] / self._avg_len)
                score += idf * (f * (K1 + 1)) / denom

            if score > 0:
                scored.append((score, i))

        # 分数从高到低排。分数一样时按片段原本的顺序排（index 小的在前），
        # 这样同样的输入永远得到同样的输出，结果稳定、可复现。
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        # ------------------------------------------------------------------
        # 【来源多样性：限制单个文件最多贡献几段】
        #
        # 为什么需要这一步——这是实测逼出来的，不是想当然加的。
        #
        # 有些问题的答案分散在两个文件里。但纯按分数排序时，
        # 一个文件里语义相近的几段可能把前几名全占了，另一个文件被挤到第 4、第 5 位，
        # 结果就是「该引两份，只引到一份」。
        #
        # 实测例子：问「学过的单词老是忘，是不是我的学习方法有问题？」
        #   排序是 course_faq(9.28) → course_faq(6.98) → vocabulary(6.02)
        #   取前两名的话，两份都来自 course_faq，而真正讲方法的 vocabulary 被漏掉了。
        #
        # 所以这里加一条限制：每个文件最多贡献 ceil(top_k / 2) 段。
        #   top_k=1 → 每份最多 1 段（不影响，本来只要一段）
        #   top_k=2 → 每份最多 1 段（强制覆盖两个来源）
        #   top_k=3 → 每份最多 2 段（既保证覆盖，又允许同一份多给一段）
        #
        # 【代价】如果某个问题的答案真的全在一个文件里，这样会掺进一段别处的、
        # 相关性较低的内容。这是「覆盖度」和「精度」之间的取舍——
        # 对问答场景来说，宁可多给一段略偏的内容，也不要漏掉真正相关的那个文件。
        # ------------------------------------------------------------------
        cap = max(1, math.ceil(top_k / 2))                      # 每个来源最多贡献几段
        per_source = {}                                         # 记录每个来源已经取了几段

        results = []
        for score, i in scored:                                 # 注意遍历的是全部候选，不只是前 top_k 个
            src = self.chunks[i]["source"]
            if per_source.get(src, 0) >= cap:
                continue                                        # 这个来源已经取够了，跳过，继续往后找别的来源

            per_source[src] = per_source.get(src, 0) + 1

            item = dict(self.chunks[i])                         # 复制一份，避免外面改到索引里的原始数据
            item["score"] = round(score, 4)
            results.append(item)

            if len(results) >= top_k:                           # 够了就停
                break

        return results


# ===================== 方便直接调用的函数 =====================

def build_index(kb_dir=KB_DIR):
    """构建知识库索引。返回一个可以直接用的 Retriever 对象。"""
    return Retriever(kb_dir)


_default_retriever = None       # 缓存一份默认索引，避免每次调用都重新读文件


def retrieve(query, top_k=DEFAULT_TOP_K):
    """用默认索引检索。第一次调用时才真正建索引。"""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever.retrieve(query, top_k)


if __name__ == "__main__":
    # 直接运行 python retriever.py 时，进入一个简单的交互模式，方便手动试检索效果
    r = Retriever()
    print("=" * 64)
    print("  本地检索层（词法检索 · 非向量检索）")
    print("=" * 64)
    print("已索引 " + str(len(r.chunks)) + " 个片段，来自 " + str(len(r.sources)) + " 份资料：")
    for s in r.sources:
        print("  · " + s)
    print("词表大小：" + str(r.vocab_size))
    print("")
    print("输入问题试检索，直接回车退出。")

    while True:
        q = input("\n> ").strip()
        if not q:
            break
        hits = r.retrieve(q, top_k=3)
        if not hits:
            print("  （没有找到相关片段）")
            continue
        for h in hits:
            print("  [" + str(h["score"]) + "] " + h["source"] + "  →  " + h["heading"])
