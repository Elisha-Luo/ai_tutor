# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# 本地检索层的自动化测试
#
# 【原则】这些测试完全离线运行：不联网、不调用任何模型 API、不需要密钥。
# 检索层本身就是纯本地的文字匹配，所以测试跑起来是毫秒级。
#
# 运行方式（在 ai_tutor 文件夹里）：
#     python -m unittest test_retriever -v
# =====================================================================

import os
import json
import unittest

import retriever as R


# 三份内容资料。knowledge_base/README.md 是说明文档，【不在】这个名单里，
# 而且必须永远不出现在索引中。
CONTENT_SOURCES = {
    "grammar_present_perfect.md",
    "vocabulary_study_method.md",
    "course_faq.md",
}

# 验收题库的位置
CASES_PATH = os.path.join(R.BASE_DIR, "evals", "rag_cases.json")


def load_cases():
    """读出 33 道验收题。"""
    with open(CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ===================== 1. 分词 =====================

class TestTokenize(unittest.TestCase):

    def test_chinese_split_into_bigrams(self):
        """中文按「相邻两个字」切。"""
        self.assertEqual(R.tokenize("现在完成时"), ["现在", "在完", "完成", "成时"])

    def test_english_lowercased(self):
        """英文统一转小写，这样 Have 和 have 算同一个词。

        【样本在 2026-10-08 改过一次】原来用的是 "Have YOU Ever"，
        但 you 现在属于英文功能词（见第 8 节），会被过滤掉 ——
        拿一个会被过滤的词来测「转小写」会得出误导性的结论。
        换成同样的句型、但全是实义词，测的仍然是同一件事。
        """
        self.assertEqual(R.tokenize("Have EVER Seen"), ["have", "ever", "seen"])

    def test_mixed_chinese_english(self):
        """中英混排时两边的词都要切出来。"""
        tokens = R.tokenize("since 和 for 的区别")
        self.assertIn("since", tokens)
        self.assertIn("for", tokens)
        self.assertIn("区别", tokens)

    def test_empty_and_punctuation(self):
        """空字符串、纯标点都切不出词，应该返回空列表而不是报错。"""
        self.assertEqual(R.tokenize(""), [])
        self.assertEqual(R.tokenize("   "), [])
        self.assertEqual(R.tokenize("？？！——"), [])


# ===================== 2. 切分（chunking）=====================

class TestChunking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = R.Retriever()          # 整个类共用一份索引，省时间

    def test_index_contains_only_the_three_content_files(self):
        """索引里只能有那三份内容资料。"""
        self.assertEqual(set(self.r.sources), CONTENT_SOURCES)

    def test_readme_is_never_indexed(self):
        """【核心】说明文档 README.md 绝不能出现在索引里。"""
        sources = [c["source"] for c in self.r.chunks]
        self.assertNotIn("README.md", sources)
        self.assertFalse(
            any("readme" in s.lower() for s in sources),
            "索引里混进了 README 相关文件：" + str(sorted(set(sources))),
        )

    def test_every_chunk_keeps_required_fields(self):
        """每个片段都必须保留 source、heading、text 三个字段。"""
        self.assertTrue(self.r.chunks, "索引是空的，说明根本没读到资料")

        for c in self.r.chunks:
            for field in ("source", "heading", "text"):
                self.assertIn(field, c, "片段缺少字段 " + field + "：" + str(list(c.keys())))

            self.assertIsInstance(c["source"], str)
            self.assertIsInstance(c["heading"], str)
            self.assertIsInstance(c["text"], str)

            self.assertTrue(c["source"].strip(), "source 是空的")
            self.assertTrue(c["heading"].strip(), "heading 是空的")
            self.assertTrue(c["text"].strip(), "text 是空的：" + c["source"] + " → " + c["heading"])

    def test_source_is_a_real_file_name(self):
        """source 必须是 knowledge_base 里真实存在的文件名。"""
        real = set(os.listdir(R.KB_DIR))
        for c in self.r.chunks:
            self.assertIn(c["source"], real, "来源文件不存在：" + c["source"])

    def test_chunks_are_split_by_h2(self):
        """切分依据是二级标题，所以语法资料的 7 个 ## 应该切成 7 段。"""
        grammar = [c for c in self.r.chunks if c["source"] == "grammar_present_perfect.md"]
        self.assertEqual(len(grammar), 7, [c["heading"] for c in grammar])

    def test_three_level_headings_stay_inside_chunk(self):
        """### 三级标题不该被当成切分点，应该留在所属片段里。"""
        methods = [c for c in self.r.chunks if c["heading"] == "四个被验证有效的方法"]
        self.assertEqual(len(methods), 1)
        # 四个 ### 小标题的内容都应该在这同一段里
        for keyword in ["间隔重复", "主动回忆", "在语境里记", "词根词缀"]:
            self.assertIn(keyword, methods[0]["text"])


# ===================== 3. 检索质量（对照验收题库）=====================

class TestRetrievalQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = R.Retriever()
        cls.cases = load_cases()

    def test_there_are_cases_to_test(self):
        """先确认题库读出来了，否则后面的循环会「空转通过」。"""
        self.assertEqual(len(self.cases), 37, len(self.cases))

    def test_all_single_source_questions_hit_top1(self):
        """所有单来源题：top_1 的结果必须来自期望的那份资料。"""
        checked = 0
        failures = []

        for c in self.cases:
            if c["expected_behavior"] != "answer" or len(c["expected_sources"]) != 1:
                continue
            checked += 1

            want = c["expected_sources"][0]
            hits = self.r.retrieve(c["question"], top_k=1)
            got = hits[0]["source"] if hits else "(没有返回任何结果)"

            if got != want:
                failures.append(c["id"] + "：期望 " + want + "，实际 " + got
                                + "\n        问题：" + c["question"])

        self.assertGreaterEqual(checked, 9, "单来源题少于 9 道，覆盖不足")
        self.assertEqual(failures, [], "top_1 检索错误：\n      " + "\n      ".join(failures))

    def test_all_cross_source_questions_covered_by_top2(self):
        """所有跨来源题：top_2 合起来必须覆盖两份期望资料。"""
        checked = 0
        failures = []

        for c in self.cases:
            if c.get("category") != "cross_source":
                continue
            checked += 1

            want = set(c["expected_sources"])
            hits = self.r.retrieve(c["question"], top_k=2)
            got = {h["source"] for h in hits}

            if not want.issubset(got):
                failures.append(c["id"] + "：期望覆盖 " + str(sorted(want))
                                + "，实际只覆盖 " + str(sorted(got))
                                + "\n        问题：" + c["question"])

        self.assertGreaterEqual(checked, 2, "跨来源题少于 2 道，覆盖不足")
        self.assertEqual(failures, [], "top_2 覆盖不足：\n      " + "\n      ".join(failures))

    def test_results_carry_source_and_heading(self):
        """检索结果必须能回答「这段出自哪个文件、哪个标题」。"""
        hits = self.r.retrieve("since 和 for 有什么区别？", top_k=3)
        self.assertTrue(hits)
        for h in hits:
            self.assertIn("source", h)
            self.assertIn("heading", h)
            self.assertIn("text", h)
            self.assertIn("score", h)
            self.assertIn(h["source"], CONTENT_SOURCES)


# ===================== 4. 安全边界 =====================

class TestSafety(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = R.Retriever()

    def test_readme_related_query_never_returns_readme(self):
        """问「RAG 是什么」这类和说明文档有关的问题，结果也绝不能来自 README.md。"""
        for q in ["RAG 是什么", "这个知识库怎么用", "knowledge_base 里的资料能公开吗"]:
            hits = self.r.retrieve(q, top_k=5)
            for h in hits:
                self.assertNotEqual(h["source"], "README.md",
                                    "问题「" + q + "」检索到了说明文档")

    def test_empty_query_returns_nothing(self):
        """空查询不崩溃，而且不返回任何东西。"""
        for q in ["", "   ", "\n", "\t\t"]:
            self.assertEqual(self.r.retrieve(q), [], "空查询 " + repr(q) + " 居然返回了结果")

    def test_punctuation_only_query_returns_nothing(self):
        """只有标点、切不出词的查询，也返回空。"""
        self.assertEqual(self.r.retrieve("？？！—— ……"), [])

    def test_unmatched_query_returns_nothing_not_a_guess(self):
        """完全匹配不上的查询要老老实实返回空，不能硬凑一段出来。"""
        self.assertEqual(self.r.retrieve("xyzzyqqq zzzz"), [])

    def test_returned_chunks_all_come_from_the_index(self):
        """所有返回的片段必须真的来自索引，不能是凭空造出来的。"""
        real = {(c["source"], c["heading"]) for c in self.r.chunks}
        for q in ["现在完成时怎么用", "背单词的方法", "课程要花多少时间"]:
            for h in self.r.retrieve(q, top_k=3):
                self.assertIn((h["source"], h["heading"]), real,
                              "返回了索引里不存在的片段：" + str((h["source"], h["heading"])))

    def test_retrieval_is_stable(self):
        """同一个问题问两遍，结果必须完全一样（可复现）。"""
        q = "since 和 for 有什么区别？"
        first = [(h["source"], h["heading"], h["score"]) for h in self.r.retrieve(q, top_k=3)]
        second = [(h["source"], h["heading"], h["score"]) for h in self.r.retrieve(q, top_k=3)]
        self.assertEqual(first, second)

    def test_top_k_is_respected(self):
        """要几段就给几段，不多给。"""
        for k in (1, 2, 3):
            self.assertLessEqual(len(self.r.retrieve("背单词为什么老忘", top_k=k)), k)

    def test_does_not_mutate_the_index(self):
        """调用检索不能改到索引本身（返回的是副本）。"""
        before = [dict(c) for c in self.r.chunks]
        self.r.retrieve("现在完成时", top_k=3)
        self.assertEqual(self.r.chunks, before)

    def test_no_network_or_model_libraries_are_imported(self):
        """【结构性保证】检索层不能引入网络或模型库——它是纯本地的。"""
        banned = ["requests", "urllib", "httpx", "aiohttp", "socket", "openai", "flask"]
        path = os.path.join(R.BASE_DIR, "retriever.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()

        imported = []
        for line in source.splitlines():
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                imported.append(line)

        for line in imported:
            for bad in banned:
                self.assertNotIn(bad, line,
                                 "retriever.py 引入了不该引入的库：" + line)

    def test_offline_note_is_documented(self):
        """文件里必须诚实写明这是词法检索、不是向量检索。"""
        path = os.path.join(R.BASE_DIR, "retriever.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()

        self.assertIn("不是向量检索", source, "代码里没写明这不是向量检索")
        self.assertIn("同义词", source, "代码里没写明不理解同义词这个局限")


# ===================== 5. 精确到 heading 的排序回归 =====================
#
# 【为什么必须测到 heading 这一层】
# 「来源文件对了」不等于「内容对了」。
# 同一个文件里有七八个小节。检索把「和一般过去时的区别」当成答案送过去时，
# source 字段照样是 grammar_present_perfect.md —— 光看来源，一切正常，
# 实际上是答非所问。
#
# 这就是为什么验收题库里那 17 道单来源题没能抓到这个问题：
# 它们只断言 top-1 的【来源】对不对，而错误答案恰好也在同一个文件里。
# 所以断言必须下沉到 heading。

class TestHeadingLevelRanking(unittest.TestCase):

    STRUCTURE_Q = "现在完成时的句子结构是怎样的？"

    @classmethod
    def setUpClass(cls):
        cls.r = R.Retriever()

    def test_structure_query_recalls_the_structure_section(self):
        """【核心回归】问「句子结构」，top-3 里必须出现「基本结构」那一节。"""
        hits = self.r.retrieve(self.STRUCTURE_Q, top_k=3)
        headings = [h["heading"] for h in hits]
        self.assertIn("基本结构", headings,
                      "top-3 里没有「基本结构」，实际返回的是：" + str(headings))

    def test_structure_section_is_ranked_first(self):
        """【理想目标】「基本结构」应该是 top-1，而不是勉强挤进前三。"""
        hits = self.r.retrieve(self.STRUCTURE_Q, top_k=1)
        self.assertTrue(hits, "这个查询居然什么都没检索到")
        self.assertEqual(hits[0]["heading"], "基本结构",
                         "top-1 是「" + hits[0]["heading"] + "」，不是「基本结构」")

    def test_structure_hit_has_both_right_source_and_heading(self):
        """命中的那一段，来源和标题都要对。"""
        hits = self.r.retrieve(self.STRUCTURE_Q, top_k=3)
        matched = [h for h in hits if h["heading"] == "基本结构"]
        self.assertTrue(matched)
        self.assertEqual(matched[0]["source"], "grammar_present_perfect.md")

    def test_recalled_chunk_actually_holds_the_answer(self):
        """召回的正文里确实写着答案，不是标题碰巧对上。"""
        hits = self.r.retrieve(self.STRUCTURE_Q, top_k=1)
        text = hits[0]["text"]
        self.assertIn("have", text)
        self.assertIn("过去分词", text)

    def test_it_does_not_recall_the_contrast_section(self):
        """旧版本会把「和一般过去时的区别」排第一，那是错的：问结构，不是问区别。"""
        hits = self.r.retrieve(self.STRUCTURE_Q, top_k=1)
        self.assertNotEqual(hits[0]["heading"], "和一般过去时的区别")


# ===================== 6. 通用机制：跨结构助词的 bigram =====================
#
# 上面那条回归能过，靠的是这条通用规则，所以规则本身也要单独钉住。
# 它针对的是【一类字】，不是某道题 —— 换任何一句中文都成立。

class TestBigramStopChars(unittest.TestCase):

    def test_bigrams_containing_de_are_dropped(self):
        """含「的」的两字组合要丢掉，因为它们横跨词边界、本身不是词。"""
        tokens = R.tokenize("现在完成时的句子结构")
        self.assertNotIn("时的", tokens)
        self.assertNotIn("的句", tokens)
        # 真正有意义的词不能被误伤
        self.assertIn("现在", tokens)
        self.assertIn("完成", tokens)
        self.assertIn("句子", tokens)
        self.assertIn("结构", tokens)

    def test_the_rule_is_general_not_word_specific(self):
        """任何含「的」的组合都丢，不挑词——证明这是通用规则而非硬编码。"""
        for text in ["我的书", "便宜的票", "最重要的区别", "的一句话", "吃的喝的"]:
            for token in R.tokenize(text):
                self.assertNotIn("的", token,
                                 "「" + text + "」里漏掉了含「的」的组合：" + token)

    def test_a_lone_de_is_dropped(self):
        """孤零零一个「的」也丢掉——它自己不带任何信息。"""
        self.assertEqual(R.tokenize("的"), [])

    def test_other_chinese_words_are_unaffected(self):
        """不涉及「的」的中文照常切分。"""
        self.assertEqual(R.tokenize("现在完成时"), ["现在", "在完", "完成", "成时"])

    def test_english_and_mixed_content_still_work(self):
        """中英混排不受影响。"""
        tokens = R.tokenize("since 和 for 的区别")
        self.assertIn("since", tokens)
        self.assertIn("for", tokens)
        self.assertIn("区别", tokens)
        self.assertNotIn("的区", tokens)

    def test_empty_and_punctuation_still_return_nothing(self):
        """空输入和纯标点的行为保持不变。"""
        for text in ["", "   ", "？？！——"]:
            self.assertEqual(R.tokenize(text), [])


# ===================== 7. 文档一级标题必须真的被读到 =====================
#
# 【这里曾经有个藏了很久的 bug】
# H1_RE 少写了 re.MULTILINE。不带它时，^ 只认整个字符串的开头、$ 只认结尾，
# 而标题虽然在文件第一行、文件却远不止一行，于是 search 永远匹配不上，
# doc_title 悄悄退回了文件名 —— 那行「文档标题加权」因此一直是死代码。
# 这组测试盯着它别再退回去。

class TestDocumentTitleIsReallyUsed(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = R.Retriever()

    def test_doc_title_is_not_a_filename(self):
        """【核心回归】doc_title 必须是真标题，不能退化成文件名。"""
        for c in self.r.chunks:
            self.assertFalse(
                c["doc_title"].lower().endswith(".md"),
                c["source"] + " 的 doc_title 退化成了文件名：" + repr(c["doc_title"]))

    def test_doc_title_matches_the_first_heading_line(self):
        """doc_title 要和文件第一行的 # 标题逐字一致。"""
        for source in self.r.sources:
            with open(os.path.join(R.KB_DIR, source), encoding="utf-8-sig") as f:
                first_line = f.readline().strip()

            self.assertTrue(first_line.startswith("# "), source + " 第一行不是一级标题")
            expected = first_line[2:].strip()

            titles = {c["doc_title"] for c in self.r.chunks if c["source"] == source}
            self.assertEqual(titles, {expected}, source + " 的 doc_title 不对")

    def test_doc_title_is_present_on_every_chunk(self):
        """每个片段都要带 doc_title 字段。"""
        for c in self.r.chunks:
            self.assertIn("doc_title", c)
            self.assertTrue(c["doc_title"].strip())


# ===================== 8. 英文功能词过滤 =====================
#
# 【要解决什么 —— 来自一次真实验收】
# 一句**英文**改写题（"Can you improve this sentence ..."）检索到一段讲
# **现在完成时**的资料，还拿了个 **13.14** 的高分。把它命中的词逐个数出来：
# 只有 a / it / the / they / to / you —— **全是功能词，一个实词都没有**。
# 那段资料因此被当成「可用资料」递给模型，后来真的被引用到了页面上。
#
# 【这组测试能证明什么、不能证明什么】
# 能证明：功能词过滤**真的在生效**；过滤之后 37 道验收题该找到的来源一个没少；
# 那句英文题的「高分假命中」没有了。
# **证明不了**：模型从此不会再引用无关资料。
# 过滤之后那句英文题**仍然**会因为 learn 这种实词命中一段无关资料（分数低，但非零）。
# **检索改善 ≠ 引用相关性已获保证** —— 引用是否真的支持回答，这一层至今没人管。
#
# 【这组测试允许针对那句题写死，因为它是「回归证据」】
# 被测试盯住的是**行为**；而修复本身（ENG-功能词表）是通用的，
# 里没有任何题目、题号或文件名。这个区别很重要，别搞反。

class TestEnglishFunctionWords(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = R.Retriever()

    # ---------- A. 规则本身 ----------

    def test_pure_function_words_are_dropped(self):
        """冠词、人称代词、不定式 to、常见介词、连词都不再作为词元。

        【注意 for 不在这个名单里】它是现在完成时的信号词，被**有意保留** ——
        所以它仍然会参与匹配（这也是那句英文题过滤后还剩 2.9 分的原因之一）。
        """
        tokens = R.tokenize("Can you improve this sentence for a classroom discussion?")
        for gone in ("you", "a", "to", "and"):
            self.assertNotIn(gone, tokens, gone + " 是纯语法胶水，应该被过滤掉")
        self.assertIn("for", tokens, "for 是信号词，必须保留")
        self.assertIn("this", tokens, "this 是常见考点，必须保留")
        # 实词照常保留
        self.assertIn("improve", tokens)
        self.assertIn("sentence", tokens)
        self.assertIn("classroom", tokens)
        self.assertIn("discussion", tokens)

    def test_articles_and_pronouns_are_gone_from_a_mixed_query(self):
        """中英混排时，英文功能词照样被丢掉，中文一个字都不受影响。"""
        tokens = R.tokenize("帮我看看 they 和 it 用对了没有")
        self.assertNotIn("they", tokens)
        self.assertNotIn("it", tokens)
        self.assertIn("看看", tokens)

    def test_knowledge_bearing_words_are_kept(self):
        """【最关键的一条】可能承载英语知识点的词，一个都不许过滤掉。

        这份名单是**照着本知识库核对**出来的，不是照搬通用停用词表：
        通用表会把 have / has / been / since / for 全删掉，
        而在这个项目里它们恰恰是知识点本身。
        """
        protected = [
            # 现在完成时的信号词（知识库专门有一节讲它们）
            "since", "for", "just", "already", "yet", "ever", "never",
            "far", "now", "recently",
            # be 动词家族
            "be", "being", "been", "am", "is", "are", "was", "were",
            # 助动词
            "do", "does", "did",
            # 指示代词（本身是常见考点）
            "this", "that", "these", "those",
            # 疑问词
            "what", "which", "who", "when", "where", "why", "how",
        ]
        for word in protected:
            self.assertNotIn(word, R.ENGLISH_FUNCTION_WORDS,
                             word + " 不该被过滤 —— 它在本知识库里是知识点")

    def test_filter_only_contains_plain_grammatical_glue(self):
        """过滤表里只准出现「纯语法胶水」，不许混进实词。"""
        for word in R.ENGLISH_FUNCTION_WORDS:
            self.assertRegex(word, r"^[a-z]+$")
        # 抽查几个明显是实义的词，绝不能在表里
        for content in ("learn", "student", "sentence", "grammar", "vocabulary"):
            self.assertNotIn(content, R.ENGLISH_FUNCTION_WORDS)

    def test_chinese_tokenization_is_untouched(self):
        """这次改的只是英文侧；中文 bigram 行为一个字符都不许变。"""
        self.assertEqual(R.tokenize("现在完成时"), ["现在", "在完", "完成", "成时"])
        self.assertEqual(R.tokenize("的"), [])

    # ---------- B. 全语料回归：该找到的还得找得到 ----------

    def test_no_expected_source_is_lost_across_the_eval_cases(self):
        """【核心回归】37 道验收题，期望来源一题都不许丢。

        过滤功能词会改变检索分数。这条测试保证「改动没有把真正有资料依据的回答搞坏」。
        """
        cases = _load_eval_cases()
        lost = []
        for case in cases:
            expected = set(case.get("expected_sources") or [])
            found = {h["source"] for h in self.r.retrieve(case["question"], top_k=3)}
            missing = expected - found
            if missing:
                lost.append(str(case.get("id")) + " 丢了 " + str(sorted(missing)))
        self.assertEqual(lost, [], "过滤之后有题找不回期望来源：" + "；".join(lost))

    def test_the_three_cross_source_cases_still_keep_both_files(self):
        """跨来源题最容易被误伤 —— 两份来源必须都还在。"""
        cases = [c for c in _load_eval_cases() if c.get("category") == "cross_source"]
        self.assertEqual(len(cases), 3, "跨来源题应该正好 3 道（题库变了就来看这里）")
        for case in cases:
            expected = set(case["expected_sources"])
            found = {h["source"] for h in self.r.retrieve(case["question"], top_k=3)}
            self.assertEqual(expected - found, set(),
                             str(case["id"]) + " 丢了一份来源：" + str(sorted(expected - found)))

    def test_the_one_on_one_tutoring_trap_still_gets_material(self):
        """一对一辅导那道题期望 insufficient_evidence，前提是【资料提到了相关机制】。

        它的检索结果一旦变空，模型就只能改口说 refuse —— 那不是我们想要的改变。
        """
        case = [c for c in _load_eval_cases() if c["id"] == "trap-one-on-one-tutoring"][0]
        hits = self.r.retrieve(case["question"], top_k=3)
        self.assertTrue(hits, "这道题的检索结果不该为空")
        self.assertIn("course_faq.md", {h["source"] for h in hits},
                      "该题的依据在 course_faq.md 里，被过滤掉了")

    # ---------- C. 那句真实英文题的回归 ----------

    REAL_WRITING_QUESTION = (
        "Can you improve this sentence for a classroom discussion and explain the main "
        "mistakes? 'AI make student learn more fast, but they depend to it too much.'"
    )

    def test_the_english_writing_question_no_longer_scores_high_on_function_words(self):
        """【本轮的核心回归】功能词过滤之前，这句题的最高分是 **13.14**。

        那个分数完全由 a / it / the / they / to / you 撑起来 —— 一个实词都没有。
        过滤之后最高分必须掉到 **3 以下**。
        3 这个数字不是拍脑袋：全题库里期望回答的题，最低分是 4.395
        （vs-context-memory），所以 3 已经在「真正有依据」的下方。
        """
        hits = self.r.retrieve(self.REAL_WRITING_QUESTION, top_k=3)
        top = hits[0]["score"] if hits else 0.0
        self.assertLess(top, 3.0,
                        "英文改写题又靠功能词拿到高分了：" + str(top))

    def test_the_unrelated_present_perfect_section_is_gone(self):
        """具体到那次事故：讲现在完成时的「基本结构」不该再被这句题捞出来。

        它当初是**第一名**（13.14）。这里针对题面写死是【回归证据】，不是修复方式。
        """
        hits = self.r.retrieve(self.REAL_WRITING_QUESTION, top_k=3)
        got = {(h["source"], h["heading"]) for h in hits}
        self.assertNotIn(("grammar_present_perfect.md", "基本结构"), got)

    def test_the_residual_match_is_weak_and_honestly_bounded(self):
        """【诚实记录残留】过滤之后还剩什么？

        剩下的命中来自 for / this 这类**被有意保留**的词，以及 learn 这个实词。
        它们都是弱命中 —— 这条测试把「弱」钉住，防止有人误以为已经彻底解决。
        """
        hits = self.r.retrieve(self.REAL_WRITING_QUESTION, top_k=3)
        for h in hits:
            self.assertLess(h["score"], 3.0, "残留命中不该是高分的")


def _load_eval_cases():
    """读验收题库。检索层的回归拿它当「不该被改坏」的清单。

    【为什么检索层要读评测题库】
    题库里的 expected_sources 是「这个问题该找到哪份资料」的权威答案。
    拿它当标尺，才能证明「过滤功能词没有把真正有依据的回答搞坏」。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "evals", "rag_cases.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
