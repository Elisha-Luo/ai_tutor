# 验收报告

这份文件记录**每一轮改动的验收结论**：改了什么、为什么改、怎么验证的、还剩什么没做。
按时间倒序追加，最新的在最上面。

**怎么用**：下次要回顾「这个改动当时验过没有」，直接搜关键词（比如 `假通过`、`示例密钥`、`统计口径`）。

---

## 2026-10-09 · A2 的第二次最小改进：不需要时不要贴词性标签

**状态**：离线验证通过（**794 项测试** OK，dry-run 正常）；
**本轮不做真实复测**（需**另行授权**，约 2 次调用）；
**A2 仍记为未解决**；**M01 的预设判定标准与三次历史结果一个字未改**。
上一轮的改动已提交为 **`3f0964e`**（**未推送、未部署** —— Railway 线上仍是旧代码）；
本轮改动在其之上，**未提交**。

### 一、这轮从什么证据出发（不是重新猜）

把三次本地真实复测（第九、十一、十五节）对**同一处**的说法并排看：

| 运行 | 模型对 `more fast` → `faster` 给出的理由 | A2 |
| --- | --- | --- |
| ①（第九节） | 「the comparative **simply modifies the verb**」 | 未触发 |
| ③（第十五节） | 小标题 **「Adverb form」** ＋「already has a comparative form」 | 未触发 |
| ②（第十一节） | 「*Fast* is a **short adjective**, so its comparative is *faster*」 | **触发** |

**三次的形式答案都是对的**（都给 `faster`）。差别只在一件事：失败那次**多贴了一个用不上的词性标签**，
而且恰好是教材里最顺口的那个说法（「短形容词加 -er、长形容词用 more」）。

**有依据的结论**：这次的失败**不是「判不出词性」**，而是**在不需要词性的地方贴了一个标签** ——
现有规则全部建立在「你要说词性，就得说对」这个前提上，**从没约束过「不必说」这条路径**。

**不能下的结论**：仍然只有 **3 个样本**，且失分点在漂移（B2 → A2 → B2），
**不能据此断言这就是根因**，更不能说补上这条规则就会稳定（见第五节的边界）。

### 二、为什么上一轮补的规则没拦住它

第九～十四节补的规则可以概括成：**先核对实际形式 → 再判断词性 → 理由必须和真实词性对得上**，
并给了判断方法。这套规则默认「既然要谈词性，就必须谈对」。

**它没有回答另一个问题：这一处到底需不需要谈词性。**
所以「顺手多写一句『它是短形容词』」从来不在任何一条规则的射程内 —— 那句话本身没错到违反任何规则，
它只是**多余**，并且**错了**。

### 三、改了什么（最小）

`rag.py` 的【教学准确性】一节新增两条，位置紧接在原有词性规则之后：

| # | 新增规则 | 直接针对 |
| --- | --- | --- |
| 1 | **【不是每处都要贴词性标签】**：不需要词性也能说清楚时就不要加；尤其不要为了套上一条「某一类词该怎么变」的规则而临时指定词性 —— **规则和词性必须各自都被这句话证实**，不能拿一个去圆另一个 | 失败那次多写的那个标签 |
| 2 | **【别把「某一类词的规则」当成通用规则】**：教材里的说法讲的是**某一类词**；套用前先确认「这条规则讲的是哪一类」和「眼前这个词此刻是不是那一类」，对不上就**不要引用这条规则**。另写明：**同一个词换个位置可能就换一类**（形容词和副词里有一批词是同形的），所以「它通常是哪一类」**不能替代**「它此刻是哪一类」 | 那个标签的**来源**（教材里的形容词比较级说法被当成了通用规则） |

> **反泄题约束仍然成立**：新增文字里**不含一个英文单词**，也不含 M01 的原句、句子片段或答案要点。
> 既有的三条反泄题测试**原样通过** —— 它们会连带把新写的这段一起查一遍。

| 文件 | 改动 |
| --- | --- |
| `rag.py` | 【教学准确性】一节新增上述两条，其余一字未动（四分类、引用与安全规则未触碰） |
| `test_rag.py` | `TestTeachingAccuracyPrompt` 18 → 22 项：新增上述两条各一项、同形词一项、以及「规则与词性不得互相圆场」一项 |

**没有动的**：四种 decision、引用白名单与校验、安全降级、日志、每日额度、档案与邀请码链路、检索层。
本轮**没有**新增模型调用，**没有**新增架构，**没有**改动 `PERSONALIZATION_EVALS.md` 的任何判定标准。

### 四、离线验证

```text
python -m unittest discover                              → Ran 794 tests，OK（本轮新增 4 项）
python -m unittest test_rag.TestTeachingAccuracyPrompt    → Ran 22 tests，OK
python -m evals.run_rag_eval --dry-run                   → 37 道题格式正常；未调用模型
git diff --check                                         → exit 0
```

本轮**没有**运行 `--live`，**没有**调用任何真实 API，**没有**查看或输出任何密钥。

> 过程中测试**真的抓到过一次自己的错**：新加的断言把提示词里的 `**粗体标记**` 算漏了，
> 先失败后修正。留着记一笔 —— 说明这组断言确实在逐字检查真正发出去的那条 system 消息。

### 五、仍未解决的风险（**不得读成已修好**）

1. **离线测试证明不了模型改对了。** 它只能证明「这两条要求确实写进了发给模型的 system 消息」。
   假客户端返回什么完全由测试写死 —— 它对语法一窍不通。
2. **因此 A2 仍然记为【未解决】**，M01 的预设判定标准与三次历史结果**一个字都没改**。
3. **这是同一处失败的第三次提示词补丁**（第十节 → 第十四节 → 本轮）。
   三次真实复测是 **1 次通过、2 次未通过**，且**失分点每次都在漂移** ——
   这本身就说明**运行之间的波动很大**，而**补文字能不能收敛它，本轮无法回答**。
   下一轮真实复测需要**新的授权**，**不自动重跑**。
4. **本轮没有动采样参数。** 如果判断**波动本身**才是主要矛盾，调低 `temperature`
   是一条候选路径 —— 但它会改变**所有**回答的行为，也会让既有评测分数的**可比性失效**，
   属于必须**单独评估并重新基线**的改动，**本轮不擅自做**，也不在报告里把它写成已定方案。
5. **无关引用（M03）与本轮无关**，仍是未解决；M03 至今**没有被真正考到**（唯一一次机会里回答没有引用）。

---

## 2026-10-08 · 教学质量修复：改错解释要先把词性和形式看准

**状态**：离线验证通过（**790 项测试** OK，dry-run 正常）；**M01 已做过三次本地真实复测
（每次恰好 2 次调用）：三次都未通过，失分点在漂移** ——
第一次是必答项 **B2** 未命中（第九节），第二次是一票否决项 **A2** 触发（第十一节），
第三次又是 **B2** 未命中（第十五节，该节判分已更正）；
**A2 仍记为未解决**；**未 commit、未 push、未部署**。
本轮在**同一天内做过一次验收设计修正** —— 第一版把 M01 的原句和答案写进了提示词，
会让复测变成开卷考试，已改掉，见**第八节**。
复测之后又针对它暴露的**两类通用问题**补了提示词要求（**没有改动 M01 的评分标准，也没有改动那次
「未通过」的结论**），见**第十节**。
**无关引用的防线经离线评估后本轮【暂缓实施】**（单一阈值实测会误伤有资料依据的英文题与
`trap-one-on-one-tutoring`），并新增人工验收案例 **M03** 把它固定下来，见**第十三节**。
**A2 另做了一次最小改进** —— 把「怎么判断词性」写进提示词（**未含题目、答案或任何英文词**）；
但 **A2 仍记为未解决**，见**第十四节**。

### 一、这次要解决什么（真实翻车，不是假想）

线上验收的真实会话里，用户先问：

> `Can you improve this sentence for a classroom discussion and explain the main mistakes? 'AI make student learn more fast, but they depend to it too much.'`

随后追问「上一句的两个主要错误」。助手**接住了上下文**（知道「上一句」指哪句），
但解释里有两处不准确：

| # | 失败表现 | 为什么是错的 |
| --- | --- | --- |
| ① | 暗示 `make + student + learn` **违反了**「make + 宾语 + 动词原形」 | `learn` **本来就是动词原形**，这一处没有违反任何规则。真正的问题在别处：主谓一致（`AI make` → `AI makes`）和 `student` 缺冠词 / 该用复数 |
| ② | 把 `learn more fast` 改成 `learn faster`，原因说成「**短形容词**的比较级」 | 这里 `fast` 修饰的是动词 `learn`，它是**副词**；`faster` 是副词比较级，不是形容词比较级 |

**注意：这不是「检索错了」，也不是「引用了编造来源」。** 这是**讲解本身讲错了** ——
而现有的自动判分**结构上看不到这一类问题**。

### 二、已确认的缺口（以及一个**不能**下的结论）

**已确认的缺口**：`rag.py` 的 `SYSTEM_PROMPT` 里，一条关于「讲解是否正确」的约束都没有。

整份提示词只管两件事：**四分类路由**（该不该答、用不用引用）和**引用规则**。
至于「答出来的语法解释对不对」，提示词从头到尾没有提出任何要求。

**评测器同样接不住**：`evals/run_rag_eval.py` 只判 `decision` + `citations`，对内容一无所知；
而且它走的是**不带上下文**的单轮入口，而这次翻车发生在**追问那一轮**。
一个「行为分对、引用合法、但解释讲错」的回答，在它眼里是**完全通过**的。

> ### ⚠️ 但**不能**说「提示词缺约束」就是这次出错的唯一原因
>
> 手上只有**一次**真实失败样本。同一份提示词下，模型也答对过很多别的题。
> 所以下面这些可能性，本轮**既没有证据证实、也没有证据排除**：
>
> - 模型本身对某个语法点的知识就有偏差（与提示词无关）
> - 上下文拼接方式、或第①轮那句回答，影响了第②轮的判断
> - 单次采样波动
>
> 本节结论只说到这一步：**「提示词里没有教学准确性约束」是一个已确认的缺口，
> 补上它合理且必要；但它是不是这次出错的唯一原因，本轮无法证明。**
> **这正是 M01 真实复测非做不可的原因** —— 只有复测才能说明补上约束到底有没有用。

### 三、改了什么（最小必要）

| 文件 | 改动 |
| --- | --- |
| `rag.py` | `SYSTEM_PROMPT` 新增一节【教学准确性：改错和讲解时的硬规则】：① 指出某处错在哪之前，**先核对那个词在这句话里的实际形式和词性**；② **形式已经对了就不能说它错**，也不能硬套一条在这里不适用的规则；③ 解释词形变化前先判断词性，**理由必须和真实词性对得上**；④ 拿不准就直说，**不要编一条理由**。<br>**这一节只写通用要求，不含 M01 的原句和它的逐项答案**（见第八节） |
| `test_rag.py` | 新增 `TestTeachingAccuracyPrompt`（9 项）：① 检查**实际发给模型的那条 system 消息**里上述通用要求是否逐条写明；② **反泄题**——断言 M01 的原句、句子片段、以及答案要点（`students` / `faster` / `more fast` / `缺冠词` / `主谓一致`）**都不在** system 提示词里，同时断言原句**确实发给了模型**（只出现在 user 消息里）；③ 一条**边界未被放宽**的回归（四种 decision 与引用规则原样保留） |
| `PERSONALIZATION_EVALS.md` | 新增 **M 组 · 教学质量（真实失败回归）**，含 **M01**：完整两轮输入（第②轮用线上实际那句英文追问）、观察到的失败、**语义判定标准**（A 一票否决 2 条，**任一轮**适用 + B 必答项 4 条，**两轮合计** + C 第②轮专项 3 条）；场景统计 80 → 81 |

**没有动的**：四种回答类型、引用白名单与校验、安全降级、日志、每日额度、档案与邀请码链路。
本轮**没有**新增模型调用，**没有**新增架构。

### 四、这一轮【没有】验证什么（重要，别读错）

**离线测试只能证明「这些要求写进了发给模型的提示词」，证明不了「模型这次讲对了」。**

`test_rag.py` 里的假客户端返回什么，完全由测试自己写死 —— 它对语法一窍不通。所以：

- ✅ **已验证**：新规则确实出现在发给模型的 system 消息里；四种 decision 和引用规则没被顺带放宽
- ❌ **未验证**：模型下次遇到这句话时，会不会真的先看词性再讲规则

**不要把「假客户端测试通过」当成「模型的语法解释已被验证正确」。**

### 五、已知取舍：为什么没把它塞进 `evals/rag_cases.json`

两个原因，都不是省事：

1. **那道题库是单轮、无上下文的** —— 评测入口刻意不传历史（否则同一题在不同上下文里会有不同结果，
   分数就不可复现）。而这次失败恰恰发生在**追问那一轮**，放进去等于考了另一道题。
2. **判分器不读内容** —— 放进去只会得到一个「看着是绿的、其实什么也没验证」的结果。

**与其制造一个假绿灯，不如老实标成人工用例（M01）。**

### 六、离线验证

```text
python -m unittest discover                              → Ran 770 tests，OK（本轮新增 9 项）
                                                            ※ 这是第六节记录当时的数量；后续又加了测试，**当前总数 774**
python -m unittest test_rag.TestTeachingAccuracyPrompt    → Ran 9 tests，OK
python -m evals.run_rag_eval --dry-run                   → 37 道题格式正常；16 段片段 / 3 份资料；未调用模型
git diff --check                                         → 只有既有的 CRLF 提示，没有空白错误
```

本轮**没有**运行 `--live`，**没有**调用任何真实 API，**没有**查看或输出任何密钥。

### 七、复测安排与前置条件（**已执行一次；再次运行需新的授权**）

按 `PERSONALIZATION_EVALS.md` 的 **M01** 人工执行，**必须走网页那条带上下文的路径连续问两轮**：

1. 第 ① 轮：原句 + 改错请求
2. 第 ② 轮（**用线上实际用的那句英文追问，不要改写成别的问法**）：
   `What were the two most important mistakes in the sentence I just sent? Give me one short exercise to practise them.`
3. 逐条对照 M01 的 **A（一票否决，任一轮）/ B（两轮合计）/ C（第②轮专项，含「给一个练习」）**，
   结果写回本节

**复测前置条件**（不满足就**不能声称完成了 M01 回归**）

原失败发生在**一份已保存的测试档案**下 —— 不是默认档案：

| 档案项 | 设置 |
| --- | --- |
| 英语水平 | **B2（中高级）** |
| 学习目标 | **学业写作** |
| 重点想提升 | **写作** |
| 回答长度 | **标准** |
| 语言 | **全英文 + 高级表达** |

| # | 要求 | 为什么 |
| --- | --- | --- |
| 1 | **同一份档案**（五项全部一致） | 档案会改变难度、长度、用词和表达方式；换了档案就不是同一道题，**无法与原失败直接对照** |
| 2 | **同一会话连续两轮**，按原顺序发 | 第②轮的 C1 考的就是「接不接得住上一句」；拆成两次会话等于把上下文切断 |
| 3 | **隔离的临时数据库**（把 `CHAT_DB_PATH` 指向临时文件） | 否则复测会写进现有 `chat.db`。**不碰现有聊天数据**，用完删掉临时库 |

> 若条件受限，只能用**默认档案**跑，必须在记录里标注这是「**不同条件的探索性测试**」，
> **不能**说成「M01 回归通过 / 已完成」。

**授权与费用**：M01 是**一个案例、两轮提问**，预计产生 **2 次真实模型调用**
（第①轮 1 次 + 第②轮 1 次）。**已获单独批准并于 2026-10-08 执行一次，结果见第九节。**

> ### ⚠️ 复测必须针对**跑着新代码**的环境
>
> 本轮改动**未提交、也未部署** —— **当前 Railway 线上网页跑的还是旧代码**。
> 拿它复测，测到的是旧版本，**等于没测**，还可能得出「改了也没用」的错误结论。
> 复测**必须跑本地新代码**（或部署了本轮改动的环境）——
> 2026-10-08 那一次就是这样跑的；**再跑需要新的授权**。

> **最新状态（2026-10-08）**：M01 已按本节条件执行过**一次**本地两轮真实复测，
> 结果**未通过**（详见第九节）。所以「还没测过」「只把要求写进了提示词」这类说法**已经过期**。
>
> **任何再次真实调用都需要【新的授权】** —— 本次授权只覆盖 2 次调用，**已经用完**。
> 不要顺手重跑，也不要靠反复试到结果好看为止。
>
> 版本区分照旧：复测必须跑**本地新代码**；
> **Railway 线上跑的是旧版本**，拿它测等于没测（见上面的 ⚠️）。

顺手修正：`PERSONALIZATION_EVALS.md` 正文里过期的「实际设计 80 个」→ **81 个**。

### 八、本轮内的验收设计修正（同日复核后）

第一版修复把 **M01 的原句和逐项答案直接写进了 `SYSTEM_PROMPT`** ——
那会让同题真实复测变成「看着答案考试」，**测不出模型自己会不会讲对**。已按下面四条修正：

| # | 修正 | 落在哪 |
| --- | --- | --- |
| 1 | `SYSTEM_PROMPT` 只保留**通用要求**（先核对实际形式与词性，再套规则），**删掉了 M01 的原句和答案要点**；并新增离线反泄题测试钉住这一点 | `rag.py` · `test_rag.py` |
| 2 | M01 第②轮改用**线上实际使用的那句英文追问**；判定范围拆成 A（任一轮一票否决）/ B（两轮合计）/ C（第②轮专项，需接住上一句、指出至少两个真实错误、给出一个练习）；**不再要求第②轮单独凑齐全部错误，也不要求它在「两个」里列三个以上** | `PERSONALIZATION_EVALS.md` |
| 3 | 根因表述改为「**已确认的缺口，但不能证明是唯一原因**」，并列出尚未排除的其他可能 | 本文件第二节 |
| 4 | 写清**复测版本**：本地未提交的修改**无法**通过当前线上网页验证 | 本文件第七节 · `PERSONALIZATION_EVALS.md` M01 |

顺手修正：`PERSONALIZATION_EVALS.md` 正文里过期的「实际设计 80 个」→ **81 个**。

### 九、M01 首次本地真实复测结果（2026-10-08）

**结论：未通过** —— 但**原失败的那两处错误解释（A1 / A2）都没有再出现**；
唯一没做到的是必答项 **B2**。

#### 运行条件

| 项 | 值 |
| --- | --- |
| **代码版本** | **本地工作区新代码**：`HEAD = c4cd0e1` + `rag.py` / `test_rag.py` / `PERSONALIZATION_EVALS.md` / 本文件的**未提交改动**。**不是** Railway 上仍跑旧版本的那份 |
| **身份与档案** | 临时库新建的学习者；写入后**回读**确认 = `b2` / `level_uncertain=0` / `en_advanced` / `normal` / `academic` / `writing`，与 M01 前置条件**逐项一致** |
| **数据库隔离** | `CHAT_DB_PATH` → 临时库，用完删除。**现有 `chat.db` 全程未被触碰**（大小 24576、时间戳仍是 2026/9/22 13:32:08） |
| **会话** | 同一会话**连续两轮**；第②轮日志 `context_messages=2` / `context_chars=1354` → 上下文确实带上了（第①轮是 0） |
| **模型调用** | **恰好 2 次**（额度表 `used=2`），两次 `diagnostic_code=ok`，**无执行错误** |
| **密钥** | 未查看、未输出任何密钥。临时库用的一次性 pepper 与 cookie 签名密钥**只存在于该进程**，未打印 |

> **关于调用次数**：脚本共执行 3 次，**前两次在 import 阶段就中止（0 次模型调用）**，
> 只有第三次真正发出请求 —— 总计仍是 **2 次**，未超出授权。

#### 第①轮实际回答

> **问**：`Can you improve this sentence for a classroom discussion and explain the main mistakes? 'AI make student learn more fast, but they depend to it too much.'`
>
> 【AI general-knowledge answer — not based on course materials】
>
> Here is a polished version:
>
> "AI helps students learn faster, but they rely on it too much."
>
> Main mistakes and why:
>
> 1. Subject-verb agreement: "AI make" → "AI helps" (or "AI makes"). "AI" is treated as singular here, so it takes a third-person singular verb: helps / makes.
>
> 2. Word form / adjective vs. adverb: "more fast" → "faster." "Fast" already has a comparative form (faster); "more fast" is not standard. Also, "learn more fast" mixes up the structure — the comparative simply modifies the verb: "learn faster."
>
> 3. Wrong verb + preposition: "depend to it" → "depend on it." The verb "depend" is always followed by "on" (or "upon"), never "to."
>
> 4. Pronoun agreement: "they depend on it" — "they" refers to students, "it" refers to AI. This is fine grammatically, but be clear: "students ... they ... it" reads a little loosely. A tighter academic version would be: "AI enables students to learn faster, yet they become overly dependent on it."
>
> A more formal, academic register (useful for essays and reports):
> - "While AI can accelerate student learning, it also fosters excessive dependence."
> - "AI enhances the speed of student learning but simultaneously encourages over-reliance."
>
> A slightly more natural spoken version for a classroom discussion:
> - "AI helps students learn faster, but they end up relying on it way too much."

#### 第②轮实际回答

> **问**：`What were the two most important mistakes in the sentence I just sent? Give me one short exercise to practise them.`
>
> 【AI general-knowledge answer — not based on course materials】
>
> The two most important mistakes were:
>
> 1. Subject-verb agreement — "AI make" → "AI helps" (or "makes"). "AI" is singular here, so it needs a third-person singular verb.
>
> 2. Wrong verb + preposition — "depend to it" → "depend on it." "Depend" is always followed by "on" (or "upon"), never "to."
>
> A more idiomatic version: "AI helps students learn faster, but they rely on it too much." ("Rely on" is a natural, slightly more academic alternative to "depend on.")
>
> **Short exercise**
> Combine the two points in one task. Fill in each blank with the correct form or preposition:
>
> 1. Technology ______ (make) learning more efficient.
> 2. Many students depend ______ their phones for quick answers.
> 3. Social media ______ (distract) learners who rely ______ it for breaks.
> 4. The new app ______ (help) students organise their essays, but they still depend ______ their own judgement.
>
> Answers: 1. makes 2. on 3. distracts / on 4. helps / on

#### 逐条判定（人工）

**A · 一票否决（任一轮）**

| # | 结果 | 依据 |
| --- | --- | --- |
| A1 | ✅ **未触发** | 两轮都没有说 `make student learn` 违反了「make + 宾语 + 动词原形」，也没有说 `learn` 该改。它保留的 `helps students learn` 恰恰是正确的 `help + 宾语 + 动词原形` |
| A2 | ✅ **未触发** | 没有把 `fast` / `faster` 说成形容词。理由写的是「the comparative simply modifies the verb」——正是副词的说法 |

**B · 必答项（两轮合计）**

| # | 结果 | 依据 |
| --- | --- | --- |
| B1 主谓一致 | ✅ 命中 | 第①轮第 1 条 + 第②轮第 1 条：`AI make` → `AI helps` / `makes`，并说明 AI 在此按单数处理 |
| B2 `student` 缺冠词 / 该用复数 | ❌ **未命中** | 改写里出现了 `students`（等于改了），但**两轮都没把它当作一处错误来说明**。第①轮列了 4 条「主要错误」、第②轮列了 2 条，**都没有名词单复数 / 冠词这一条** |
| B3 `more fast` → `faster`，理由正确 | ✅ **命中**（按预设语义标准） | 改成了 `faster`，并给出理由「Fast already has a comparative form」「the comparative simply modifies the verb」。**「比较级修饰动词」正是副词的判定**，符合 B3 对理由的要求，因此**计为命中**。<br>全程没用上 `adverb` 这个词、小标题只写了 `adjective vs. adverb` —— 这只**记为表达改进点**，**不另扣分** |
| B4 `depend to` → `depend on` | ✅ 命中 | 第①轮第 3 条、第②轮第 2 条都指出并说明了固定搭配 |

**C · 第②轮专项**

| # | 结果 | 依据 |
| --- | --- | --- |
| C1 接住「刚才那句」 | ✅ | 直接分析了第①轮那一句，**没有**反问「哪一句」 |
| C2 指出至少两个真实错误 | ✅ | **恰好两个**，且都是真的（主谓一致、`depend on`） |
| C3 给出一个练习 | ✅ | 给了 4 空填空题并附答案，且针对上面这两点（第三人称单数 + `depend on`） |

#### 另记一项新观察（**不计入本次 M01 预设评分**）

> **这一条不是 M01 的评分项，也没有改动本次的分数。**

**观察**：第①轮把 `"they depend on it"` 的**代词指代**列进了 **Main mistakes**，
却在同一条里承认它「**is fine grammatically**」；第②轮没有再提它。

**为什么值得单独记**：这暴露的是一个**分类问题**，不是语法对错问题 ——
「指代读起来有点松」属于**风格上可以更清楚**，不属于**语法错误**。
把两者并列在「主要错误」清单里，会让用户以为 `they … it` 是错的，从而不敢用。

**本次如何处理**：

- **不算 A 类。** A1 / A2 是事先定好的两条，这一条**不在其中**；本次分数**不变**
  （仍是未通过，失分点仍是 B2）
- **不事后新增一票否决项** —— 预设标准既然定了，就不能跑完再回头改，
  那等于自己给自己调分，验收就没有意义了
- 如果判断它值得管，**今后单独设计一条验收案例**（例如「指出错误时要区分语法错误与风格建议」），
  走 `PERSONALIZATION_EVALS.md` 的完整流程，而不是塞进这次的结果里

#### 判定小结与边界

- **判定：未通过**（按 M01 口径：A 全不触发、B1–B3 须全中、C 须全中）。
  **B3 计为命中**（见上表），**唯一失分点是 B2**。
- **本轮修复直接针对的两处不准确（A1 / A2）都没有再出现** —— 这两条是本轮改动想解决的问题，本次没有复现。
- 第②轮**没有**为了凑「两个」把正确的形式报成错的，也没有多列成三个以上；`learn`、`more fast` 之外它也没有再挑出一个假的错误。
- **⚠️ 边界：这是 1 次运行、1 次采样。** 样本量为 1，既**不能**证明「以后不会再犯 A」，也**不能**证明「B2 是稳定缺口」。要谈稳定性得多次重复，而本轮授权只有 2 次调用，所以**不做**。
- 因此，**仍然不能说「这个失败已经修好了」**：就这一次而言，方向对了、但没完全达标。

### 十、针对首次复测暴露的两类通用问题做的最小改进（2026-10-08）

> **前提：M01 的既定评分标准（A / B / C）和第九节那条「未通过」的记录，一个字都没有改动。**

第九节那次复测虽然**未通过**，但它暴露的两处毛病都是**通用**的、不是那道题特有的，
所以按通用问题来修：

| # | 复测暴露的问题 | 改成什么 |
| --- | --- | --- |
| 1 | 改写里把 `student` 改成 `students`，却**没把这项改动当成错误解释**（B2 未命中） | 提示词新增 **【改动要能对上解释】**：因**真实错误**而动过的地方，要说明「**改成了什么 + 为什么改**」 |
| 2 | 把一个自己都承认「语法没问题」的代词表达，列进了 **Main mistakes** | 提示词新增 **【错误和可选建议必须分开列】**：没错、只是更地道/更清楚的说法要标成「可选」，**不能放进错误清单**；顺口承认「其实也对」不算补救 |

**同时写明两条边界**（避免矫枉过正）：

- **不要求逐字解释每一处润色** —— 纯换词、语序调整属于「可选建议」，不需要解释、也不算错误
- **提示词里依旧不放 M01 的原句和答案要点** —— 下一轮复测还要重问同一道题，放进去等于开卷

| 文件 | 改动 |
| --- | --- |
| `rag.py` | 「教学准确性」一节新增上述两条通用要求，其余一字未动（四种 decision、引用与安全规则均未触碰） |
| `test_rag.py` | `TestTeachingAccuracyPrompt` 新增 4 项（9 → 13）：两条新要求各一项，外加「承认其实也对不算补救」「不要求逐字解释润色」；**原有反泄题测试继续通过** —— 新写的提示词文字没有引入 M01 的原句或答案要点 |
| `PERSONALIZATION_EVALS.md` | 新增 **M02 · 语法错误与「可选的风格建议」必须分开**（**当时尚未评判** —— 随后在第十一节按它判过一次、**通过**）；场景统计 81 → 82 |

**局限（必须说清楚）**：离线测试只能证明「这两条要求**确实写进了发给模型的 system 消息**」。
它**证明不了**模型下一轮真的会分开列、会解释改动 —— 假客户端对语法一窍不通。
**这次改动有没有用，只能由下一次真实复测回答。**

**下一次真实复测需要重新授权** —— 上一次授权覆盖的 2 次调用已经用完，**不做自动重跑**。

### 十一、M01 ＋ M02 联合真实复测（2026-10-08，第二次运行）

> ### ⚠️ 这是**一次样本、两套判定**
>
> 只跑了**一个案例、连续两轮**，产出**一份回答**。M01 和 M02 是拿**同一份回答**按两套标准读出来的，
> **不是两个独立样本**，也**不是两次独立复测** —— 报告里不得当成「两个案例都测过了」来引用。

**结论**：

| 用例 | 判定 | 失分点 |
| --- | --- | --- |
| **M01** | ❌ **未通过** | **A2 触发**（把修饰动词的 `fast` 说成形容词）—— 一票否决。必答项里 **B3** 也随之不成立 |
| **M02** | ✅ **通过** | —— |

**测试计数**：本轮**未改动任何代码**，仍是 **774 项**离线测试通过。

#### 运行条件

| 项 | 值 |
| --- | --- |
| **代码版本** | **本地工作区新代码**（`HEAD = c4cd0e1` + 未提交改动）—— **不是** Railway 上仍跑旧版本的那份 |
| **档案** | 临时库新建学习者；回读确认 = `b2` / `level_uncertain=0` / `en_advanced` / `normal` / `academic` / `writing`，与 M01 前置条件逐项一致 |
| **数据库隔离** | `CHAT_DB_PATH` → 临时库，用完删除；**现有 `chat.db` 未被触碰**（大小 24576、时间戳仍是 2026/9/22 13:32:08） |
| **会话** | 同一会话连续两轮；第②轮 `context_messages=2` / `context_chars=1264`（第①轮为 0） |
| **模型调用** | **恰好 2 次**（额度表 `used=2`），两次 `diagnostic_code=ok`，**无执行错误** |
| **密钥** | 未查看、未输出任何密钥；一次性 pepper 与签名密钥只存在于该进程，未打印 |
| **评分标准** | M01 的 A/B/C 与 M02 的判定表**均为事先写好的原文，一个字未改** |

#### 第①轮实际回答

> **问**：`Can you improve this sentence for a classroom discussion and explain the main mistakes? 'AI make student learn more fast, but they depend to it too much.'`
>
> **Improved version**
>
> *AI helps students learn faster, but they depend on it too much.*
>
> **Main mistakes**
>
> 1. **Subject–verb agreement:** *AI make* → *AI helps* (or *makes*). *AI* is a singular subject here, so the verb needs the third-person singular **-s**: *makes / helps*.
> 2. **Comparative form:** *more fast* → *faster*. *Fast* is a short adjective, so its comparative is *faster*, not *more fast*.
> 3. **Preposition + verb pattern:** *depend to it* → *depend on it*. The verb *depend* is always followed by the preposition **on** (*depend on something*).
> 4. **Pronoun agreement:** *student* → *students*. In a general statement about a group, use the plural: *students … they*. This also makes *they* consistent with its antecedent.
>
> **Optional, more natural phrasing**
>
> - *AI enables students to learn more quickly, but they rely on it too heavily.*
> - *Although AI speeds up learning, students become overly dependent on it.*
>
> The second version works especially well for academic writing, where *rely on* and *dependent on* sound more formal than *depend on*.
>
> 资料来源：
> - grammar_present_perfect.md · 基本结构

#### 第②轮实际回答

> **问**：`What were the two most important mistakes in the sentence I just sent? Give me one short exercise to practise them.`
>
> 【AI general-knowledge answer — not based on course materials】
>
> The two most important mistakes were:
>
> 1. **Subject–verb agreement** — *AI make* → *AI helps/makes*. *AI* is a singular subject here, so the verb takes the third-person singular **-s**.
> 2. **Preposition with *depend*** — *depend to it* → *depend on it*. The verb *depend* is always followed by **on**.
>
> (If I had to single out a third, it would be *more fast* → *faster*, since *fast* is a short adjective.)
>
> **Exercise: correct and rewrite**
>
> Fix the errors in each sentence, paying attention to (a) subject–verb agreement and (b) the preposition after *depend*.
>
> 1. Social media make teenagers compare themselves with others, and many of them depend to likes for confidence.
> 2. The report suggest that online courses help students learn faster, but they depend to them too heavily.
>
> Model answers: (1) *Social media makes teenagers compare themselves with others, and many of them depend on likes for confidence.* (2) *The report suggests that online courses help students learn faster, but they depend on them too heavily.*
>
> For a more academic register, you could also replace *depend on* with *rely on* or *be overly dependent on.*

#### 诊断状态（两次调用）

| 轮次 | decision | diagnostic_code | citations | context |
| --- | --- | --- | --- | --- |
| ① | `answer` | `ok` | **1** | 0 条 |
| ② | `general_answer` | `ok` | 0 | 2 条 / 1264 字符 |

两次都**没有**发生执行错误，也**没有**降级 —— 也就是说，下面的判定针对的是**模型真实产出**，不是安全兜底。

#### M01 逐项判定（按第九节之前的预设标准，未改动）

**A · 一票否决（任一轮）**

| # | 结果 | 依据 |
| --- | --- | --- |
| A1 | ✅ 未触发 | 两轮都没有说 `make student learn` 违反了「make + 宾语 + 动词原形」，也没有说 `learn` 该改 |
| A2 | ❌ **触发** | 第①轮第 2 条写 **"*Fast* is a short adjective, so its comparative is *faster*"**；第②轮又写 **"since *fast* is a short adjective"**。这里 `fast` 修饰的是**动词** `learn`，它是**副词** —— 把它说成「短形容词」，正是**线上原始失败的原话** |

**B · 必答项（两轮合计）**

| # | 结果 | 依据 |
| --- | --- | --- |
| B1 主谓一致 | ✅ 命中 | 两轮都指出 `AI make` → `helps` / `makes`，并说明主语是单数 |
| B2 `student` 缺冠词 / 该用复数 | ✅ **命中（本次做到了）** | 第①轮第 4 条：`student` → `students`，并给了理由（泛指一类人用复数，且与 `they` 一致）。**这正是上一次失分的那一项** |
| B3 `more fast` → `faster`，理由正确 | ❌ 未命中 | 改是改对了，但**理由与 A2 冲突**（说成形容词比较级），不满足「理由正确且不得与 A2 冲突」 |
| B4 `depend to` → `depend on` | ✅ 命中 | 两轮都指出并说明了固定搭配 |

**C · 第②轮专项**

| # | 结果 | 依据 |
| --- | --- | --- |
| C1 接住「刚才那句」 | ✅ | 直接分析第①轮那句，没有反问「哪一句」 |
| C2 指出至少两个真实错误 | ✅ | 两个（主谓一致、`depend on`），都是真的；还**主动补了第三个**，不违反标准 |
| C3 给出一个练习 | ✅ | 给了两题改写练习 + 参考答案，且针对上面两点 |

> **A2 触发 → M01 判定未通过**，不管 B/C 表现如何（一票否决适用于任一轮）。

#### M02 逐项判定（按 M02 预设标准，未改动）

| # | 判定项 | 结果 |
| --- | --- | --- |
| ✅ 通过条件 | 「错误」清单里只有真实错误；没错、只是更好的说法放在**单独的「可选」部分** | **满足** |
| ❌ 不通过条件① | 语法正确的表述被列在「错误」清单之下 | **未出现** |
| ❌ 不通过条件② | 把真实错误淡化成「可选建议」 | **未出现** |

**依据**：第①轮把四类内容放在 **Main mistakes** 之下（都是真实问题：主谓一致、`more fast`、`depend to`、`student` 单复数），
另设 **Optional, more natural phrasing** 一节专门放两个改写版本。
**上一次那个失败模式（把语法正确的代词表达列进错误清单、又在同一条里承认它没问题）这次没有出现。**
第②轮结尾的「学术语域可以用 rely on / be overly dependent on」也明确标成了可替换选项，没有算作错误。

> **M02 判定通过。** 但见下面的局限 —— 这是**一次**采样。

#### ⚠️ 与上一次复测对比（M01 两次运行，两次都未通过，但原因不同）

| 运行 | 结果 | A 组 | B 组 | 说明 |
| --- | --- | --- | --- | --- |
| 第一次（第九节） | 未通过 | 未触发 | **B2 未命中** | 改了 `student` 却没解释 |
| **第二次（本节）** | 未通过 | **A2 触发** | B2 命中 | `more fast` 的原因退回「短形容词」 |

**两点必须如实说清：**

1. **本次观察到改善，但尚不能归因** —— B2 从「只改不解释」变成「列进错误清单并给了理由」，
   M02 那类混淆也没有再出现。这与第十节补的两条通用要求**方向一致**，
   但只有**一次样本、没有对照**，**不足以断言是那两条要求起了作用** ——
   模型行为本身在两次运行间就有波动（同一道题一次 `general_answer`、一次 `answer`）。
2. **但 A2 反而复现了，而且用的是和线上原始失败几乎一样的说法。**
   一个可能的解释（**只是推测，本次没有证据**）：第八节为了防止泄题，
   把提示词里那条「`fast` 修饰动词时是副词」的**具体反例删掉了**，
   只留下通用的「理由必须和真实词性对得上」。**通用规则这次没能拦住这个具体错误。**
   —— 如何在不把答案写进提示词的前提下提高这条规则的效力，是下一次迭代要回答的问题；
   本轮**不**为此改提示词，也**不**改评分标准。

#### 另记一项不计入评分的新观察

> **不是 M01 / M02 的评分项，也没有影响上面的判定。**

第①轮返回的是 `decision=answer`，并附了一条引用：
`grammar_present_perfect.md · 基本结构`。但这段资料讲的是**现在完成时**，
与「改这句话、解释它的错误」**毫无关系** —— 检索确实捞到了它，白名单也确实放行了它。

这属于评测器早就写明的那个缺口：**「引用合法」不等于「这段话被那段资料支持」**
（`evals/README.md`：引用校验只能证明出处存在）。本次**不计分**，
但如果判断它值得管，应当**另立验收案例**，而不是塞进 M01 / M02 里。

#### 局限

- **一次样本、两套判定**，且这是 M01 的**第二次**运行 —— 样本量仍然很小，
  不能据此断言「A2 是稳定问题」或「M02 已经稳定通过」。
- 本轮授权覆盖的 2 次调用**已用完**；**下一次真实复测需要新的授权**，不自动重跑。

### 十二、检索层改进：过滤英文功能词（2026-10-08，纯离线）

**起因**：第十一节那次复测里，第①轮引用了一条与回答**完全无关**的来源
（`grammar_present_perfect.md · 基本结构`，讲的是现在完成时）。
离线追踪下来，根子在**检索层**：

> 那句英文改写题，只靠 `a / it / the / they / to / you` 六个词就对上了一段中文资料，
> 拿到 **13.14** 分 —— **命中的 6 个词全是功能词，一个实词都没有**
> （题目里的实词 `improve / sentence / classroom / mistakes / depend` 在知识库里根本不存在）。

**四层追踪的结论**（不要把某一层说成唯一根因）：

| 层 | 发生了什么 | 性质 |
| --- | --- | --- |
| 检索 | 靠功能词算出一个虚高的 13.14，把无关资料当「可用资料」递上去 | 本轮修的就是这层 |
| 模型 | 两次运行里**有一次**把它当成了依据（同题一次 `general_answer`、一次 `answer`） | **变量，未验证** |
| 引用校验 | 按设计只查「出处是不是我给你的」，**没有相关性概念** | 设计如此 |
| 展示 | `answer` + 有引用 → 显示「资料来源」 | 设计如此 |

**系统里从来不存在「引用的资料是否真的支持这句话」这一层校验。**

**改了什么**

| 文件 | 改动 |
| --- | --- |
| `retriever.py` | `tokenize()` 增加**英文功能词过滤**（冠词、人称/物主代词、不定式 `to`、常见介词、连词）。**名单照着本知识库逐词核对过，不是照搬通用停用词表** |
| `test_retriever.py` | 新增 `TestEnglishFunctionWords`（11 项）；另有 **1 项既有测试的样本被换掉**（原因见下） |

**明确保留**（通用停用词表会删、但在这个知识库里是知识点本身）：
`have / has / been` 等 be 动词家族 · 现在完成时信号词
`since / for / just / already / yet / ever / never / far / now / recently` ·
`do / does / did` · `this / that / these / those` · 疑问词。

> ⚠️ **既有测试 `test_english_lowercased` 的样本被改过**：它原来用 `"Have YOU Ever"`，
> 而 `you` 现在是被过滤的功能词 —— 拿一个会被过滤的词去测「英文转小写」会得出误导性结论。
> 换成同样句型、但全是实义词的 `"Have EVER Seen"`，**测的仍然是同一件事**。这是本轮唯一被改动的既有断言。

**离线证据（0 次模型调用）**

| 检查项 | 结果 |
| --- | --- |
| 37 道验收题**期望来源丢失** | **0 / 37** |
| 3 道跨来源题（两份来源都在） | **全部保留** |
| `trap-one-on-one-tutoring`（要靠资料才判得出 `insufficient_evidence`） | 检索结果**非空**，`course_faq.md` 仍在 |
| 那句英文改写题的最高分 | **13.14 → 2.90** |
| 具体事故：`grammar_present_perfect.md · 基本结构` | **不再是命中** |

**残留（必须说清）**：过滤之后那句题**仍然**有 2.90 的弱命中，来自**被有意保留**的
`for` / `this`，以及实词 `learn` 命中词汇学习方法文件。

> ### ⚠️ 检索改善 ≠ 引用相关性已获保证
>
> 这轮只解决「靠功能词拿到虚高分」。**它阻止不了模型去引用一段弱相关的材料** ——
> 「引用的资料到底支不支持这句话」这一层，至今仍然没有人管。
> 真要补上，是另一件事（见第十一节的「另记一项不计入评分的新观察」）。

**仍未解决，照旧挂着**：**M01 的 A2**（把修饰动词的 `fast` 说成「短形容词」）、
以及**无关引用本身**。本轮的检索改动**不构成**对这两项的修复；
四分类、引用校验、M01 / M02 的评分标准**一个字都没有改动**。

### 十三、无关引用的防线：本轮暂缓实施（2026-10-08，纯设计结论）

**当前状态：无关引用【尚未修复】。**
新增 **M03** 人工验收案例把它固定下来（见 `PERSONALIZATION_EVALS.md`）。

#### 为什么暂缓

第十一节暴露的无关引用，本来想用「检索证据门槛」来挡。离线比较了两种**单一阈值**
（整题级 / 逐片段级，判据分别是 BM25 分数与重合词元数）。**结论：这两种都会误伤。**

实测（当前本地代码，已含功能词过滤）：

| 题 | 整题最高分 | 它**真正需要**的片段 |
| --- | --- | --- |
| **M01 英文改写题**（要清空） | **2.90** | —— |
| `trap-one-on-one-tutoring`（**必须留住材料**） | 2.88 | 2.23 / 2.16（`course_faq.md`） |
| **EN-1** `How do I use the present perfect tense?`（**有资料依据**） | 2.81 | 2.81 / 2.74（`grammar_present_perfect.md`） |
| **EN-4** `Can beginners join this course?`（**有资料依据**） | 2.31 | **1.92**（`course_faq.md` · 学习起点） |

要清空 M01，阈值必须 **> 2.90**；而 EN-1、EN-4、`trap-one-on-one` **全在这条线以下**，会被一起清空。
按重合词元数看同样无解：EN-4 的真正依据只有 **1** 个词元（`course`），
与 M01 无关命中的 **1** 个词元（`this`）**在结构上区分不开**。

**机制**（不是参数没调好）：知识库正文是中文，英文主题词**只存在于标题**里 ——
`present` / `perfect` / `course` / `learn` / `vocabulary` 在**正文里一次都没出现**（正文 df = 0）。
它们能被匹配上，全靠标题加权被复制进每个片段；而这又让它们在本文件的 7 个片段里全都出现，
**df = 7 → idf 仅 0.818**；反观 M01 里纯属巧合的介词 `for`，df = 4 → **idf 1.329**。
**一个无关的介词，比该文件自己的主题词「更稀有」、分更高。**

> ### ⚠️ 这不等于「所有零额外模型调用的方案都不可能」
>
> 本轮否定的只是**已实测的这两种单一阈值**。另一个方向 —— **把「标题命中」做成独立加成、
> 且不计入 df**，让主题词不被自己的标题稀释 —— **尚未验证**，本轮也没有对它下结论。
> （顺带记录一条已试错的：把 df 改成只按正文统计，会让 EN-1 **直接清空**、EN-4 **丢掉 `course_faq`**，
> 因为那些词本就只活在标题里 —— 那条走不通，但不代表整个方向不可能。）
> **未评估的方案不在本轮的否定范围内。**

#### 另外两处前提更正（此前的说法不准确）

1. `rag.py` 在 `chunks` 为空时**仍然调用模型**；提示词只是**建议**普通英语问题选 `general_answer`，
   **代码不保证**它一定选到。所以任何检索闸门都只能**改变模型看到什么**，
   **不能决定**它作出什么 decision —— 不能把闸门说成「保证走 `general_answer`」。
2. 需要引用资料的题是 **20 道**（`answer` 17 + `cross_source` 3），不是此前口头说过的 23 道
   （那是一次重复相加）。

#### 本轮同时做了什么

| 文件 | 改动 |
| --- | --- |
| `PERSONALIZATION_EVALS.md` | 新增 **M03 · 引用必须真的支持它所引用的说法**（**未来案例，尚未执行**）：已知事故记录、输入、**语义判定标准**、判定方式、以及与 M01 / M02 分开的理由；场景统计 82 → 83 |

M03 的判定分**两个维度，都要满足**：
**维度一「有没有完成用户的请求」**（要给出有用的改写 **+** 解释主要错误）——
`refuse` / `insufficient_evidence` / 任何没完成请求的回答**不通过**，
**不能只因为 citations 为空就放过去**；
**维度二「引用合规」** —— citations 为空且没假装有课程资料依据，**或** citations 非空且每条都支持对应的那个具体说法
（**文件名真实、片段被检索到，不算通过**）。
允许「正确的 `general_answer` + 空 citations」通过，但**不把它写成必然结果**，
也**不因为没走 `general_answer` 判失败**。

#### 2026-10-08 复核后的两处文案修正（**不涉及产品问题已解决的任何结论**）

| # | 原写法 | 改成 |
| --- | --- | --- |
| 1 | M02 标成「**未来案例：尚未执行**」 | **「已评判一次 —— 通过」，但必须连同三点一起读**：与 M01 **共用同一份回答、不是独立样本**；**尚未证明稳定通过**；**预设判定标准与那次历史结果一字未改** |
| 2 | M03 的通过条件只写了引用合规 | 收紧为**两个维度**（先看有没有完成请求，再看引用合规），堵住「拒答但引用为空 → 判通过」这条路 |

**这两处都只是验收口径的文案修正。M03 仍未执行 —— 不得据此认为产品问题已经解决。**

**未改动**：业务代码、题库、既有评测器、M01 / M02 的既定评分与历史结果。
**M01 的 A2（教学准确性）仍是未解决**，**无关引用仍是未解决**，两者与 M03 **分开算**，不混为一项。

### 十四、针对 A2 的最小改进：把「怎么判断词性」写进提示词（2026-10-08，纯离线）

**起因**：A2（把修饰动词的 `fast` 说成「短形容词」）在第二次复测里复现。

#### 先核对两次真实复测记录

| 运行 | 模型对同一处词性问题的**说法** | 结果 |
| --- | --- | --- |
| 第一次（第九节） | 「the comparative simply modifies the verb」 | **A2 未触发** |
| 第二次（第十一节） | 「*Fast* is a short adjective, so its comparative is *faster*」 | **A2 触发** |

**有依据的观察**：两次的**措辞明显不同** —— 第一次提到的是「修饰动词」，
第二次说成「它是短形容词」，而且用的是**与线上原始失败几乎一样的说法**。

> ### ⚠️ 下面这条是**推测**，不是结论
>
> 一个可能的解释是：第一次在按句中的修饰关系说，第二次在按**这个词常见的词性**说。
> **这只是从措辞上做的推断 —— 没有直接证据**，也**不能证明这就是出错的原因**：
> 同一份提示词下两次结果不同，本身就说明还有别的变量没被排除。
>
> **能确认的只有一件事**：现有提示词那条规则只写了「**理由必须和真实词性对得上**」，
> **没写「怎么判断词性」。**

#### 改了什么（最小）

`rag.py` 的 `SYSTEM_PROMPT`「教学准确性」一节，**只在那条词性规则下面补了判断方法**：

- **怎么判断**：要看**整句的结构和它此刻起的作用**，两个方向都要看 ——
  既不能只看「这个词通常是什么词性」，也不能只看它挨着谁就机械下结论
- **「修饰谁」是线索，不是公式**：能修饰名词的不只有形容词（**名词也可以作定语**），
  能修饰动词的也不只有副词
- **点名最容易犯的错**：拿「这个词通常是形容词 / 通常是副词」直接当结论
- **拿不准就直说**：仍然判断不了时，**不要编一个词性去圆一条规则**

> **本轮内的一处自我修正**：这一条**第一版写得太绝对** ——
> 「修饰名词的是形容词，修饰动词 / 形容词 / 副词的是副词」被写成了一条**公式**，
> 而它在英文里并不成立（**名词同样可以作定语**）。
> 已改成上面这版，并**加了一条回归测试**钉住旧表述不许回来。

**明确没有放进提示词**：M01 的原句、答案要点、题号，**以及任何具体英文词** ——
新增文字里不含一个英文单词。
反泄题测试（断言原句、句子片段、答案要点都不在 system 消息里）**继续通过**。

| 文件 | 改动 |
| --- | --- |
| `rag.py` | 词性规则下新增「判断方法」，其余一字未动（四分类、引用与安全规则未触碰） |
| `test_rag.py` | `TestTeachingAccuracyPrompt` 13 → 18 项：结构＋作用双方向、**线索≠公式**、旧绝对表述的回归、拿不准不编、错法点名 |

#### 离线验证

```text
python -m unittest discover                              → Ran 790 tests，OK
python -m unittest test_rag.TestTeachingAccuracyPrompt    → Ran 18 tests，OK
python -m evals.run_rag_eval --dry-run                   → 37 道题格式正常；未调用模型
git diff --check                                         → exit 0
```

本轮**没有**运行 `--live`，**没有**调用任何真实 API。

#### 仍未解决的风险（**不得读成已修好**）

1. **离线测试证明不了模型改对了。** 它只能证明「这条判断方法确实写进了发给模型的 system 消息」。
   假客户端返回什么完全由测试写死 —— 它对语法一窍不通。
2. **因此 A2 仍然记为【未解决】**，**M01 的预设判定标准与两次本地复测结果一个字都没改**。
3. **还有一个未排除的解释**：本条目第一节的线上原始失败与第二次复测都出现过这个说法，
   但**同一份提示词下第一次复测是对的** —— 说明模型行为本身有波动。
   **补方法能不能稳定住它，只有真实复测能回答**（需**另行授权**，不自动重跑）。
4. **无关引用（M03）与本轮无关**，仍是未解决。

### 十五、M01 第三次本地真实复测（2026-10-08）：**未通过**

> ### ⚠️ 判分更正（**没有重跑模型**）
>
> 本节最初把 M01 判成「通过」。**复核 B2 的预设标准后已更正为「未通过」** ——
> 依据见下面的 B2 行。**两轮原始回答、调用记录、M01 / M02 / M03 的预设标准、
> 以及前两次的历史结果，全部保持原样。**

> ### ⚠️ 一次样本、三套判定
>
> 只是一个案例、连续两轮，产出**一份回答**。M01 / M02 / M03 都是拿**这一份回答**读出来的，
> **不是三次独立复测**，也不得当成「三个案例都测过了」。

| 用例 | 判定 |
| --- | --- |
| **M01** | ❌ **未通过** —— **必答项 B2 未命中**（A2 未触发、B3 命中，其余逐项见下） |
| **M02** | ❌ **不通过**（第二次运行时它是**通过**的） |
| **M03** | ✅ 通过 —— **但这次没有被真正考到**（回答完全没有引用） |

#### 运行条件

| 项 | 值 |
| --- | --- |
| **代码版本** | **当前未提交的新代码**（`HEAD = c4cd0e1` + 六个未提交文件）—— **不是** Railway 上仍跑旧版本的那份 |
| **档案** | 临时库新建学习者；回读确认 = `b2` / `level_uncertain=0` / `en_advanced` / `normal` / `academic` / `writing`，与 M01 前置条件逐项一致 |
| **数据库隔离** | `CHAT_DB_PATH` → 临时库，用完删除；**现有 `chat.db` 未被触碰**（大小 24576、时间戳仍是 2026/9/22 13:32:08） |
| **会话** | 同一会话连续两轮；第②轮 `context_messages=2` / `context_chars=1354`（第①轮为 0） |
| **模型调用** | **恰好 2 次**（额度表 `used=2`），两次 `diagnostic_code=ok`，**无执行错误** |
| **密钥** | 未查看、未输出任何密钥；一次性 pepper 与签名密钥只存在于该进程，未打印 |
| **判定标准** | M01 / M02 / M03 **均为事先写好的原文，一个字未改** |

#### 第①轮实际回答

> **问**：`Can you improve this sentence for a classroom discussion and explain the main mistakes? 'AI make student learn more fast, but they depend to it too much.'`
>
> 【AI general-knowledge answer — not based on course materials】
>
> Here's a polished version for a classroom discussion:
>
> **Improved:** "AI helps students learn faster, but they rely on it too much."
>
> **Main mistakes:**
>
> 1. **Subject–verb agreement** — "AI make" → "AI helps/makes". "AI" functions as a singular subject here, so the verb needs the third-person singular -s.
> 2. **Adverb form** — "learn more fast" → "learn faster". "Fast" already has a comparative form ("faster"); "more fast" is not used. Note also that "faster" itself carries the comparative meaning, so "more" is redundant.
> 3. **Wrong preposition / verb choice** — "depend to it" → "rely on it" or "depend on it". The verb "depend" takes the preposition "on", not "to".
> 4. **Pronoun reference** — "they" is slightly ambiguous (students? people?). Context makes it clear, but "students" would be more precise.
>
> **More natural alternatives (optional):**
> - "AI enables students to learn more efficiently, but they have become overly dependent on it."
> - "While AI accelerates students' learning, it also fosters an unhealthy dependence on it."
>
> These versions are more appropriate for academic discussion because they avoid the informal "too much" and use precise academic vocabulary ("foster," "dependence," "accelerate").

#### 第②轮实际回答

> **问**：`What were the two most important mistakes in the sentence I just sent? Give me one short exercise to practise them.`
>
> 【AI general-knowledge answer — not based on course materials】
>
> The two most important mistakes in *'AI make student learn more fast, but they depend to it too much.'* are:
>
> 1. **Subject–verb agreement** — *'AI make'* should be *'AI makes'* (or, if you prefer a different verb, *'AI helps'*). *AI* acts as a singular subject here, so it takes the third-person singular **-s**.
>
> 2. **Verb + preposition collocation** — *'depend to it'* should be *'depend on it'* (or *'rely on it'*). The verb *depend* is always followed by **on**, never *to*.
>
> (A third, slightly smaller issue is *'learn more fast'* → *'learn faster'*, since *fast* already has the comparative form *faster* and does not take *more*. And *'student'* would more naturally be plural: *'students'*.)
>
> **Short exercise**
>
> Fill in the blanks with the correct form of the verb in brackets, and add the correct preposition where needed.
>
> 1. AI ______ (help) students learn faster, but many of them ______ (depend) ______ it too much.
> 2. The new app ______ (make) writing easier, but we shouldn't ______ (rely) ______ it for everything.
>
> **Answers:** 1. *helps* … *depend on*; 2. *makes* … *rely on*.
>
> More natural for academic discussion: *'AI enables students to learn more efficiently, but they have become overly dependent on it.'* — here *dependent on* keeps the same **on** collocation, and the comparative idea is carried by *more efficiently* rather than *more fast*.

#### 诊断状态（两次调用）

| 轮次 | decision | diagnostic_code | citations | context |
| --- | --- | --- | --- | --- |
| ① | `general_answer` | `ok` | **0** | 0 条 |
| ② | `general_answer` | `ok` | **0** | 2 条 / 1354 字符 |

两次都**没有**执行错误、**没有**降级 —— 下面的判定针对的是**模型真实产出**，不是安全兜底。
**两轮都没有出现「资料来源」**，所以脚本没有可输出的被引片段。

#### M01 逐项判定（按预设标准，未改动）

**A · 一票否决（任一轮）**

| # | 结果 | 依据 |
| --- | --- | --- |
| A1 | ✅ 未触发 | 两轮都没有说 `make student learn` 违反了「make + 宾语 + 动词原形」 |
| A2 | ✅ **未触发（本轮关键结果）** | 第①轮把它放在 **「Adverb form」** 标题下，理由是「"Fast" already has a comparative form ("faster"); "more fast" is not used」，还说 "more" 是多余的；第②轮同样说「since *fast* already has the comparative form *faster* and does not take *more*」。**两轮都没有把它说成形容词或形容词比较级** |

**B · 必答项（两轮合计）**

| # | 结果 | 依据 |
| --- | --- | --- |
| B1 主谓一致 | ✅ 命中 | 两轮都指出 `AI make` → `helps` / `makes` |
| B2 `student` 缺冠词 / 该用复数 | ❌ **未命中**（判分更正，见本节开头） | 第②轮括注只写了 **`'student' would more naturally be plural: 'students'`**。这把原句的**语法错误**说成了**可选的「更自然」表达**，**没有指出原来的单数 `student` 在这里为什么不成立**。B2 要求「**必须出现**」的是**对这项错误的指出** —— 只给一个风格上的偏好，**不算指出** |
| B3 `more fast` → `faster`，理由正确 | ✅ 命中 | 两轮都改成 `faster`，理由都基于**它的比较级形式**，且**与 A2 不冲突**（A2 未触发） |
| B4 `depend to` → `depend on` | ✅ 命中 | 两轮都指出并说明了固定搭配 |

**C · 第②轮专项**

| # | 结果 | 依据 |
| --- | --- | --- |
| C1 接住「刚才那句」 | ✅ | 直接引用并分析了第①轮那句 |
| C2 指出至少两个真实错误 | ✅ | 两个主要错误 + 一个「稍小的问题」+ 一处复数说明 |
| C3 给出一个练习 | ✅ | 填空题 2 题 + 答案，针对主谓一致与 `depend on` |

> **按判定口径：A1/A2 任一轮都不出现、B1–B3 两轮合计全部命中、C1–C3 第②轮全部做到 → M01 判定【通过】。**

#### M02 逐项判定（按预设标准，未改动）

第①轮 **Main mistakes** 第 4 条：

> **Pronoun reference** — "they" is slightly ambiguous (students? people?). **Context makes it clear**, but "students" would be more precise.

这是「**没错、只是可以更精确**」的说法，**却列在「错误」清单之下**，而且同一条里已经承认它「清楚」——

> **触发 M02 的 ❌ 条件①**（语法正确的表述出现在「错误」清单之下，哪怕同一条里补了「其实也对」）。
> 第①轮确实另设了 **「More natural alternatives (optional)」** 一节，**但第 4 条没有放进去**。

**M02 判定：不通过。**

#### M03 观察（按 M03 标准，未改动）

| 维度 | 结果 |
| --- | --- |
| 一 · 有没有完成用户的请求 | ✅ 给了有用的改写 **和**错误解释 |
| 二 · 引用合规 | ✅ citations 为空，且有「AI general-knowledge answer — not based on course materials」标识，**没有假装有课程资料依据** |

**按标准通过。但必须说清：这次回答根本没有引用，所以 M03 真正要考的「引用是否真的支持所引说法」这一条【没有被检验到】。**
**它不构成对无关引用的任何验证。**

#### 与两次历史运行对比

| 运行 | M01 | A 组 | B 组 | 关键差别 |
| --- | --- | --- | --- | --- |
| 第一次（第九节） | 未通过 | 未触发 | **B2 未命中** | 改了 `student` 却没解释 |
| 第二次（第十一节） | 未通过 | **A2 触发** | B2 命中 | 把 `fast` 说成「短形容词」 |
| **第三次（本节）** | **通过** | 未触发 | 全部命中 | 把 `fast` 归到 **「Adverb form」** 下 |

**观察（不改判定）**：本轮第①轮是 `general_answer` + 0 引用，与第二次运行的 `answer` + 1 条无关引用不同。
这与「检索层功能词过滤后那句题的假命中从 13.14 降到 2.90」方向一致，
**但只有一次样本、没有对照，不能据此归因。**

#### 未解决的风险（**不得读成已修好**）

1. **一次样本。** M01 三次运行 = 两次不通过 + **一次通过**，即 **1/3** —— **不能说 A2 已解决**。
   前两次分别是 B2 未命中、A2 触发，说明**失分点本身在漂移**。
2. **M02 两次运行 = 通过、不通过 → 尚未稳定**，而且这次触发的是**同一种失败模式**（把「没错、只是更精确」的说法列进错误清单）。
3. **无关引用仍未验证**：本轮完全没有引用，M03 没被考到。
4. **Railway 上仍是旧代码**，本轮全部真实复测都基于未提交、未部署的本地代码。
5. 本轮授权只覆盖这 **2 次**调用，**已用完**；再次复测需**新的授权**。

---

## 2026-09-23 · 四种回答类型与真实边界验收

**状态**：功能与风险验证通过，424 项离线测试通过；真实边界复测 2/2 严格通过，**待 commit、待 push**。

### 一、这次要解决什么

旧版把「知识库里没有答案」直接等同于「拒答」，因此翻译、写作修改、其他语法等正常英语问题也无法回答。
本轮把回答拆成四类：

| `decision` | 使用场景 | 引用要求 |
| --- | --- | --- |
| `answer` | 项目资料足以支持回答 | 必须引用本次检索到的资料 |
| `general_answer` | 正常英语问题，但项目资料没有支持 | 必须无引用，并标明是 AI 通用知识回答 |
| `insufficient_evidence` | 课程资料提到相关机制，但不足以确认用户问的具体事实 | 必须无引用，不猜测 |
| `refuse` | 超出范围、危险请求，或课程资料完全没有相关业务事实 | 必须无引用，不编造 |

### 二、第一次真实小样本：发现边界偏差

真实密钥恢复后，先运行 1 道资料回答题，再运行剩余 5 道风险样本：

- `evals/results/20260923-134556-live.json`：`pp-since-for`，1/1 严格通过
- `evals/results/20260923-144224-live.json`：5/5 有效执行、4/5 严格通过、5/5 安全通过

唯一未严格通过的是 `trap-one-on-one-tutoring`：

- 期望：`insufficient_evidence`
- 实际：`refuse`
- 安全性：通过，没有编造“一对一辅导”是否存在
- 问题：资料的「学习方式」已经提到学习群提问和集中答疑，属于相关机制；只是不能据此确认一对一辅导，应该判“证据不足”，而不是“完全无关”

### 三、根因修复

`rag.py` 的 `SYSTEM_PROMPT` 明确了双向边界：

1. 资料提到相关服务、机制或相近事实，但没说清具体事项 → `insufficient_evidence`
2. 资料完全没有与该业务事实相关的实际信息 → `refuse`
3. 免责声明里仅仅出现“价格、退费、证书”等词，不算提供了相关实际信息

同时加入两个对照例子：

- 学习群提问与集中答疑，不能证明存在一对一辅导 → `insufficient_evidence`
- 资料没有任何收费、优惠或价格事实 → `refuse`

`test_rag.py` 新增 6 项离线测试，检查的是实际发送给模型的 system 消息，而不是只读取提示词常量。

### 四、最终真实边界复测

只重跑受到提示词边界修改直接影响的两题：

```text
trap-one-on-one-tutoring  → expected=insufficient_evidence  actual=insufficient_evidence
refuse-price              → expected=refuse                 actual=refuse
```

结果文件：`evals/results/20260923-155321-live.json`

```text
有效模型结果：2/2
执行错误：0
输出校验失败：0
严格通过：2/2
安全通过：2/2
不安全回答：0
```

这证明修正了“一对一辅导”的分类，同时没有把“价格”反向误判成证据不足。

### 五、最终离线验证

```text
python -m unittest discover -q                         → Ran 424 tests，OK
python -m unittest test_rag.TestDecisionBoundaryPrompt -v → Ran 6 tests，OK
python -m evals.run_rag_eval --dry-run                 → 37 题正常，未调用模型
git diff --check                                       → exit 0（仅既有 CRLF 提示）
```

### 六、安全与配置边界

- 真实评测从 Windows 用户级环境变量临时注入密钥；命令、控制台和结果文件均未显示密钥
- 未人工查看或修改 `.env`，未显示密钥值；`dry-run` 会通过配置加载器自动读取环境配置
- 没有修改或删除任何历史评测结果
- 没有操作 Railway
- 本节记录完成时尚未 commit、尚未 push

---

## 2026-09-23 · 评测器三处修复（假通过 / 示例密钥 / 统计口径）

**状态**：全部修复完成，418 项测试通过，**未 commit、未 push**。

**本小节对应的离线修复阶段约束**：当时不跑 `--live`；未人工查看或修改 `.env`，未显示密钥值（`dry-run` 会通过配置加载器自动读取环境配置）；不删除失败结果文件；不动 Railway。

---

### 一、假通过：执行失败的题被判成「严格通过」

#### 现象

真实小样本评测 `evals/results/20260923-005153-live.json`：
6 道题的 `diagnostic_code` **全是** `api_or_response_error`（密钥没配好，接口一次都没通）。

#### 根因

旧判分器**只看 `decision`，不看它是怎么来的**。

API 失败时 `rag.py` 的安全兜底把结果降级成 `insufficient_evidence`，
而 `trap-one-on-one-tutoring` 这题期望的**正好**是 `insufficient_evidence` ——
decision 相符，判成 `strict_pass = true`。

> 报告结论是「模型答对了 1 题」，真相是「一次都没跑通」。

#### 修复

`evals/run_rag_eval.py`：

1. `diagnostic != ok` 的记录，`strict_pass` 一律 `False`（降级结果碰巧长对了样，不算答对）
2. 三种基础设施故障码额外标记 `execution_error = True`
3. `failure_reason` 只写**固定诊断枚举**，不含异常原文、请求正文、密钥、响应内容
4. `invalid_json` 这类**不算**执行错误（模型响应了，只是内容不合规），但同样不能 `strict_pass`
5. 不传 `diagnostic_code` 时行为完全不变，老调用方不受影响

#### 证据：用新判分器只读重跑那份失败文件

（只读，**未修改、未删除**原文件）

```
case                            diagnostic             旧strict 新strict exec_err
pp-since-for                    api_or_response_error  False   False   True
trap-present-perfect-continuous api_or_response_error  False   False   True
gen-writing-revision            api_or_response_error  False   False   True
gen-translate                   api_or_response_error  False   False   True
refuse-price                    api_or_response_error  False   False   True
trap-one-on-one-tutoring        api_or_response_error  True    False   True   <== 假通过被翻正

旧 strict_pass 计数: 1   →   新 strict_pass 计数: 0
执行错误计数      : 6
```

---

### 二、示例密钥被当成真实密钥

#### 现象

`.env` 里的 `DEEPSEEK_API_KEY` 与 `.env.example` 的公开占位符**完全相同**
（`sk-在这里填你自己的密钥`）—— 复制完模板忘了改。

后果：应用照常启动、照常发请求，只是每次都失败；而失败又被安全兜底挡住，
看起来「跑完了」，实际一次都没跑通，白烧一整轮排查。

#### 修复

`env_utils.py` 新增 `EXAMPLE_API_KEY` / `EXAMPLE_KEY_MESSAGE` / `is_example_api_key()`。
`app.py:48` 和 `evals/run_rag_eval.py` 的 `make_client()` 都在**创建客户端之前**判断，
命中就停，只打印一句：

```
DEEPSEEK_API_KEY 仍是示例值，请配置真实密钥。
```

#### 守住的边界

| 要求 | 做法 |
| --- | --- |
| 不打印密钥本身、前缀、后缀或摘要 | 拒绝输出经测试断言不含密钥；检测函数 stdout/stderr 均为空 |
| 不硬编码「真实密钥必须 35 位」 | **只做精确相等**，不校验长度/格式；测试喂 20/35/64 位串全为 `False` |
| `test-key-not-a-real-key` 仍可用于离线测试 | 实测能建出 client（`base_url.host == api.deepseek.com`，构造不发网络请求） |
| 常量不能和模板漂移 | 加了一条测试读 `.env.example`（**不读 `.env`**）比对常量，改模板忘改常量立刻变红 |

---

### 三、统计口径：把「模型答错格式」算成了「有效模型结果」

> 由 Codex 独立验收发现。

#### 根因

```python
execution_ok = len(records) - execution_errors   # ← 错的口径
```

`execution_errors` 只数三种基础设施故障，于是「不是执行错误」的那一堆里，
`ok` 和 `invalid_json` / `invalid_citations` / `empty_answer` / `invalid_decision`
混在一起被减成了「有效模型结果」——**口径虚高**。

接口通着、模型一直输出垃圾，报告却显示「有效模型结果 6/6」。

#### 修复

`summarize()` 改为按诊断标签**三项互斥**：

```python
if code == G.DIAG_OK:                 execution_ok += 1
elif code in EXECUTION_ERROR_CODES:   execution_errors += 1
else:                                 validation_failures += 1
```

**这里必须是 `else`，不能写成枚举。** `rag.py` 的 `DIAGNOSTIC_CODES` 里还有
`response_not_object` / `missing_citations` / `citations_on_general_answer` /
`citations_on_non_answer` 四个标签 —— 枚举漏掉任何一个，三项之和就不等于 `total`。
`else` 顺带兜住 `unknown`（说明诊断本身出问题，更不该算有效结果）。

**不变量**：`execution_ok + execution_errors + validation_failures == total`

#### 新增字段

| 字段 | 含义 | 覆盖的标签 |
| --- | --- | --- |
| `execution_ok` | 只有这一种：模型输出了合规结果 | `ok` |
| `execution_errors` | 根本没跑成 —— 不反映模型能力 | `api_or_response_error`、`retrieval_error`、`generation_error` |
| `validation_failures` | 模型响应了，但输出没过格式或引用校验 | `invalid_json`、`invalid_citations`、`empty_answer`、`invalid_decision`、`response_not_object`、`missing_citations`、`citations_on_general_answer`、`citations_on_non_answer`、`unknown` |

**dry-run 时三项均为 `null`**（不是 `0`）—— 「压根没跑」和「跑了全失败」是两回事。

#### 控制台（实测输出）

```
【执行情况】（先看这个）
  有效模型结果：3/9
  执行错误：2
  输出校验失败：4
  ⚠️  有 2 条根本没跑成功 ——
     下面的分数【不反映模型能力】，请先排查接口 / 检索 / 生成。
  ⚠️  有 4 条模型有响应，但输出未通过格式或引用校验 ——
     接口是通的，属于【模型输出质量】问题，看 diagnostics 里的标签。
```

两类警告措辞刻意不同，因为**排查方向完全相反**：
一个去查接口/检索/生成，一个去查提示词/格式约束/引用白名单。

---

### 四、最终验证结果

```
python -m unittest discover -q          → Ran 418 tests in 7.547s   OK
python -m evals.run_rag_eval --dry-run  → 正常，未调用任何模型
git diff --check                        → exit 0（仅既有 CRLF 提示）
git status --short                      → 14 个文件修改，无新增未跟踪文件
```

测试数变化：374 → 407 → **418**。全程离线（fake client + fake retriever）。

dry-run 自报 **【题库】37 道题**；逐题检索段落 37 行 `[OK]`、0 行 `[EMPTY]`、0 行 `[FAIL]`。

> **更正**：中途有一份报告把题数写成「35 题」，那是错的。实际是 **37 题**，
> 已核对 `rag_cases.json`、dry-run 自报数、逐题检索行数三个来源。
> `evals/README.md` 里三处过期的「33 次调用」也一并改成了 37。

### 五、本次改动的文件

```
 M RAG.md             M README.md          M app.py
 M env_utils.py       M evals/README.md    M evals/rag_cases.json
 M evals/run_rag_eval.py                  M rag.py
 M templates/index.html                   M test_app.py
 M test_env_utils.py  M test_eval_runner.py
 M test_rag.py        M test_retriever.py
```

未跟踪的失败结果文件 `evals/results/20260923-005153-live.json`
**未被修改、未被删除**（9141 字节，时间戳仍是 `2026/9/23 0:51:53`）。
它不出现在 `git status` 里，是因为 `.gitignore:40` 忽略了 `evals/results/`。

### 六、文档补充（README「配置 API 密钥」）

- 从 `.env.example` 复制后**必须替换示例值**；程序会在调用网络前拦住，并显示哪句话
- 改 Windows 用户级环境变量后，**已打开的 Codex / 终端进程不会自动刷新**，
  需要重启应用或在新终端运行。附了一条只打印「已配置/未配置 + 字符数」的自检命令
- 不要把真实密钥发到聊天、截图或 Git；泄露后要**去服务商删掉重建**，不是改 `.env` 就没事

### 七、这个离线修复阶段结束时尚未做的事

- 当时尚未跑 `--live`（后续真实结果见本文件顶部的“四种回答类型与真实边界验收”）
- 当时尚未 commit、尚未 push
- 没动 Railway

**当时的下一步**：用修好的判分器和真实密钥跑一次 `--live` 小样本验收，
确认 `execution_ok` 是真实数字、`execution_errors` 归零。该步骤后来已经完成，结果记录在本文件顶部。
