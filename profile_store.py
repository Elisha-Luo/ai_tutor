# -*- coding: utf-8 -*-
# 上面这行告诉 Python：这个文件用 UTF-8 编码，中文才不会乱码

# =====================================================================
# 学习档案的【存储层】：邀请码、学习者、会话绑定、偏好。
#
# 【为什么单独一个文件，而不是塞进 app.py】
# 这个项目已经有一条明确的规矩：一层一个文件 ——
#   retriever.py  只负责「找资料」
#   rag.py        只负责「让模型照着资料回答，并校验引用」
#   app.py        只负责「网页这一层：路由、会话、锁、额度」
# 档案的读写是第四件事，而且规则不少（邀请码摘要、白名单、会话唯一绑定）。
# 塞进 app.py 会让那个文件同时管三件事，测试也没法单独跑。
#
# 【这一层绝对不碰的东西】
#   · 不认识 Flask，不认识 request / session —— 它只收字符串、只回字典/整数
#   · 不调模型、不联网
#   · 不打印任何东西（日志归 app.py 管，而且邀请码绝不许进日志）
# =====================================================================

import hashlib      # 算 HMAC 摘要用（SHA-256）
import hmac         # 常数时间比较 + HMAC，都是标准库，不用额外装包
import secrets      # 密码学安全的随机源，用来生成邀请码
import sqlite3      # 只有一处用到：认出「摘要撞了」这个异常（见 create_invite）
from datetime import datetime, timezone   # 记时间


# ===================== 状态常量 =====================
#
# 【为什么用常量而不是到处写字符串】
# 状态会进数据库的 CHECK 约束。写成常量之后，改名字只要改这一处，
# 不会出现「代码里写 'bound'、约束里写 'bounded'」这种只有运行到才发现的不一致。

INVITE_ACTIVE = "active"        # 未使用，可以用来创建学习者
INVITE_BOUND = "bound"          # 已经被某个学习者用掉了
INVITE_REVOKED = "revoked"      # 被组织者作废

VALID_INVITE_STATUSES = (INVITE_ACTIVE, INVITE_BOUND, INVITE_REVOKED)

LEARNER_ACTIVE = "active"
LEARNER_DELETED = "deleted"

VALID_LEARNER_STATUSES = (LEARNER_ACTIVE, LEARNER_DELETED)


# ===================== 偏好的白名单 =====================
#
# 【为什么偏好必须是「代号」，不能是用户自由填的字符串】
# 两条理由：
#   1. 安全：偏好会拼进提示词。让用户随便填，提示词里就多了一段【完全由用户控制】的
#      文字，他可以塞「忽略上面的规则」这类东西。
#   2. 设计：TECH_DESIGN 第 8 节明确写了 learner_preferences 存的是
#      「选项代号」，【不存用户的原话】。
# 所以五项全部是代号，每一项都有一个白名单 + CHECK 约束。

# 【注意】下面这些代号值在设计文档里没有给出 —— 文档只给了中文名（第 3 节说
# level_code / language_mode / length_mode「CHECK 在白名单里」，但没列出白名单）。
# 这里按文档里的中文名一一对应定义，改的时候两处一起改。
VALID_LEVELS = ("a1", "a2", "b1", "b2", "c1")           # 英语水平（CEFR 大致档位）

# 语言模式：对应 PRODUCT_REQUIREMENTS 第 8 节的三种
VALID_LANGUAGE_MODES = (
    "zh_pair",       # 中英对照：英文在前，中文解释紧随其后
    "en_only",       # 全英文：只用英文，用词按水平控制
    "en_advanced",   # 全英文 + 高级表达：主动给出更地道的说法
)

# 长度偏好：对应 PRODUCT_REQUIREMENTS 第 8 节
VALID_LENGTH_MODES = ("brief", "normal", "detailed")
#   brief    —— 精简：只给结果和最关键的一句解释
#   normal   —— 标准：结果 + 解释 + 可选说法
#   detailed —— 详细：可以展开讲原理、多给例子

# 【这两个字段是本轮新增的，设计文档里【没有】对应列】
# 文档只在 PRODUCT_REQUIREMENTS 第 6 节提到「学习目标」是档案该记的内容之一，
# 「重点提升方向」则完全没有出现过。既然文档的规矩是「只存代号，不存原话」，
# 这两个也做成代号 —— 与 level_code 保持一致，也避免把用户原文拼进提示词。
VALID_GOALS = (
    "exam",          # 考试 / 升学（雅思、托福、高考……）
    "academic",      # 学业写作（作业、论文、报告）
    "daily",         # 日常交流（听说、聊天）
    "work",          # 工作 / 商务
    "interest",      # 兴趣 / 自我提升
)

VALID_FOCUS = (
    "writing",
    "speaking",
    "listening",
    "reading",
    "grammar",
    "vocabulary",
)


# ===================== 给页面用的「代号 + 中文名」 =====================
#
# 【和 rag.py 里那份对照表的区别】
#   这里 —— 给【人看】的短名字，出现在下拉框里，要短、要好懂
#   rag.py —— 给【模型看】的说明，要写成一句能执行的指令
# 两份分开是因为它们服务于两个完全不同的读者。
# 代价是「加了新代号忘了改另一处」—— 所以有一条测试专门核对两边都覆盖全了。

LEVEL_OPTIONS = (
    ("a1", "A1 入门"),
    ("a2", "A2 基础"),
    ("b1", "B1 中级"),
    ("b2", "B2 中高级"),
    ("c1", "C1 高级"),
)

LANGUAGE_OPTIONS = (
    ("zh_pair", "中英对照"),
    ("en_only", "全英文"),
    ("en_advanced", "全英文 + 高级表达"),
)

LENGTH_OPTIONS = (
    ("brief", "精简"),
    ("normal", "标准"),
    ("detailed", "详细"),
)

GOAL_OPTIONS = (
    ("exam", "考试 / 升学"),
    ("academic", "学业写作"),
    ("daily", "日常交流"),
    ("work", "工作 / 商务"),
    ("interest", "兴趣 / 自我提升"),
)

FOCUS_OPTIONS = (
    ("writing", "写作"),
    ("speaking", "口语"),
    ("listening", "听力"),
    ("reading", "阅读"),
    ("grammar", "语法"),
    ("vocabulary", "词汇"),
)



# ===================== 建表 =====================
#
# 【总原则】只加不改。
#   · 全部用 CREATE TABLE IF NOT EXISTS —— 已经存在的库上重复调用不会出错
#   · 绝不动 messages 和 api_usage 这两张老表
#   · 外键一律 ON DELETE CASCADE，删学习者时子表跟着清干净（invites 除外，
#     见下面那行的注释）
#
# 【为什么外键要写出来】
# SQLite 默认【不检查】外键，每次开连接都要 PRAGMA foreign_keys=ON。
# 这条在 app.py 里统一开，这里只负责把约束写进表结构。

def _quoted(values):
    """把一串常量拼成 SQL 的 IN 列表：'a', 'b', 'c'。

    给 CHECK 约束用。常量都是代码里写死的短串（没有引号、没有分号），
    拼进去是安全的 —— 这里拼的【不是】用户输入。
    """
    return ", ".join("'" + v + "'" for v in values)


def ensure_schema(conn):
    """把档案相关的四张表建好（已存在就什么都不做）。"""
    with conn:                                        # with conn = 事务：中间出错自动回滚
        # ---------- 邀请码 ----------
        # 【只存摘要，不存明文】code_digest 是 HMAC-SHA256 的结果。
        # 拿到数据库文件的人，没有 pepper 也算不出任何一个邀请码。
        conn.execute(
            "CREATE TABLE IF NOT EXISTS invites ("
            "    id         INTEGER PRIMARY KEY AUTOINCREMENT,"
            "    code_digest TEXT NOT NULL UNIQUE,"          # 摘要唯一 → 同一个码只会有一行
            "    status     TEXT NOT NULL CHECK (status IN (" + _quoted(VALID_INVITE_STATUSES) + ")),"
            "    created_at TEXT NOT NULL,"
            "    revoked_at TEXT,"
            # 【为什么是 SET NULL 而不是 CASCADE】删掉学习者时，这张邀请码不应该跟着消失 ——
            # 它是「谁用过」这条审计线索，留着并断开指向更合理。
            "    learner_id INTEGER REFERENCES learners(id) ON DELETE SET NULL"
            ")"
        )

        # ---------- 学习者 ----------
        # 【这里没有任何身份字段】没有姓名、没有学校、没有邮箱 —— 只有内部编号和状态。
        # 这是刻意的：这个产品不需要知道你是谁，只需要知道「这是同一个人」。
        conn.execute(
            "CREATE TABLE IF NOT EXISTS learners ("
            "    id           INTEGER PRIMARY KEY AUTOINCREMENT,"
            "    created_at   TEXT NOT NULL,"
            "    last_seen_at TEXT NOT NULL,"
            "    status       TEXT NOT NULL CHECK (status IN (" + _quoted(VALID_LEARNER_STATUSES) + "))"
            ")"
        )

        # ---------- 会话绑定 ----------
        # 【session_id 上必须有 UNIQUE】一个 session 只能属于一个学习者。
        # 少了这条约束，A 的浏览器会话就可能被改绑到 B 身上 —— 那是数据泄露级的错误。
        #
        # 【revoked_at 是干什么的 —— 这一列很关键】
        # NULL      = 绑定有效，这个浏览器能访问那位学习者的档案
        # 有值      = 绑定已失效（邀请码被作废），这个浏览器【不能】再访问档案，
        #             但这一行【故意留着】—— 因为「哪个会话属于谁」这个事实
        #             在删除全部数据时还要用（见 clear_all_data）。
        #
        # 早期版本作废时是直接 DELETE 这一行，结果把关联也一起弄丢了：
        # 用户被踢下线之后，我们就再也找不到他那些会话的聊天记录，
        # 「清空全部个人数据」就会漏掉它们。所以改成「标记失效」而不是「删除」。
        conn.execute(
            "CREATE TABLE IF NOT EXISTS learner_sessions ("
            "    id           INTEGER PRIMARY KEY AUTOINCREMENT,"
            "    learner_id   INTEGER NOT NULL REFERENCES learners(id) ON DELETE CASCADE,"
            "    session_id   TEXT NOT NULL UNIQUE,"
            "    created_at   TEXT NOT NULL,"
            "    last_seen_at TEXT NOT NULL,"
            "    revoked_at   TEXT"
            ")"
        )

        # ---------- 迁移：给老库补上 revoked_at ----------
        # CREATE TABLE IF NOT EXISTS 对【已经存在】的表什么都不做 ——
        # 所以老库不会自动多出这一列。这里显式检查一次并补上。
        # 补上之后，老数据里所有绑定都是「有效」（NULL），行为不变。
        existing_columns = {row[1] for row in
                            conn.execute("PRAGMA table_info(learner_sessions)")}
        if "revoked_at" not in existing_columns:
            conn.execute("ALTER TABLE learner_sessions ADD COLUMN revoked_at TEXT")

        # ---------- 偏好 ----------
        # 【learner_id 就是主键】一个学习者只有一行偏好。
        # 用 INSERT ... ON CONFLICT 更新，天然不会重复建行。
        conn.execute(
            "CREATE TABLE IF NOT EXISTS learner_preferences ("
            "    learner_id   INTEGER PRIMARY KEY REFERENCES learners(id) ON DELETE CASCADE,"
            "    level_code   TEXT NOT NULL CHECK (level_code IN (" + _quoted(VALID_LEVELS) + ")),"
            "    level_uncertain INTEGER NOT NULL CHECK (level_uncertain IN (0, 1)),"
            "    language_mode TEXT NOT NULL CHECK (language_mode IN (" + _quoted(VALID_LANGUAGE_MODES) + ")),"
            "    length_mode  TEXT NOT NULL CHECK (length_mode IN (" + _quoted(VALID_LENGTH_MODES) + ")),"
            "    goal_code    TEXT CHECK (goal_code IS NULL OR goal_code IN (" + _quoted(VALID_GOALS) + ")),"
            "    focus_code   TEXT CHECK (focus_code IS NULL OR focus_code IN (" + _quoted(VALID_FOCUS) + ")),"
            "    updated_at   TEXT NOT NULL"
            ")"
        )


# ===================== 小工具 =====================

def _now():
    """当前时间，形如 2026-09-24T10:30:05。和 app.py 里记消息时间用的是同一种格式。"""
    return datetime.now().isoformat(timespec="seconds")


def generate_invite_code():
    """生成一个高熵的邀请码（给组织者用）。

    【为什么用 secrets 而不是 random】
    random 是可预测的：知道种子的人能推出下一个码。secrets 是密码学安全的随机源，
    专门用来生成凭证。

    【为什么比 16 个字符还长】
    码越长越难猜。这里 24 字节的 token_urlsafe 约 32 个字符，
    配合 pepper 和限速，暴力枚举在现实中不可行。

    【为什么不带任何编号或日期】
    码里一旦能反推出「第几个用户」，泄露一个码就等于泄露了用户规模，
    也给了猜其他码的线索。
    """
    return secrets.token_urlsafe(24)


def digest_invite_code(code, pepper):
    """把邀请码算成摘要：HMAC-SHA256(key=pepper, message=code)。

    【为什么是 HMAC 而不是「每个码加随机盐再哈希」】
    加随机盐之后，同一个码每次算出来的哈希都不一样 —— 那就没法「按摘要直接查询」了。
    原始设计里这两条是矛盾的，后来改成 HMAC + 服务端 pepper：
    同一个码永远得到同一个摘要，既能直接查，又没有 pepper 就算不出来。

    【为什么不用 bcrypt/argon2 那类慢哈希】
    邀请码是【高熵随机值】，不是人选的密码 —— 没有字典可猜。
    慢哈希在这里只会让每次请求都变慢，换不到安全收益。
    防爆破靠的是「高熵 + pepper + 限速」这三条。

    【pepper 从哪来】只从环境变量读，和 API 密钥同级。
    它绝不进代码、绝不进数据库、绝不进日志。
    """
    return hmac.new(
        pepper.encode("utf-8"),          # 密钥要转成字节
        code.encode("utf-8"),            # 消息也要转成字节
        hashlib.sha256,
    ).hexdigest()


# ===================== 邀请码：登记 =====================

def create_invite(conn, code, pepper):
    """登记一张新邀请码。返回 True；如果这个码已经登记过，返回 False。

    【和 generate_invite_code 的分工】
      generate_invite_code() —— 造出一个码（一次性的，给组织者看）
      create_invite()        —— 把这个码登记进库（只存摘要）

    拆开是因为「造码」和「存码」是两件事：
    测试要能指定一个固定的码，生产要能一次生成一个随机的码。

    【为什么重复登记要返回 False，而不是报错】摘要上有 UNIQUE 约束，
    撞了说明这个码已经在库里了。这属于「输入重复」，不是「程序故障」，
    用返回值表达，调用方好处理。
    """
    digest = digest_invite_code(code, pepper)
    try:
        with conn:
            conn.execute(
                "INSERT INTO invites (code_digest, status, created_at) VALUES (?, ?, ?)",
                (digest, INVITE_ACTIVE, _now()),
            )
    except sqlite3.IntegrityError:
        return False                                  # 摘要已存在
    return True


# ===================== 登记一张「外部生成」的邀请码 =====================
#
# 【和 create_invite / generate_invite_code 的分工】
#   generate_invite_code() —— 在我们这边造一个码（老路：SSH 进去跑命令行）
#   create_invite()        —— 把指定明文码登记进库；**重复登记算失败**（返回 False）
#   register_generated_invite() —— 线上管理入口用：码由【操作者的脚本】本地生成，
#                             这里只负责登记，而且**重复登记必须是幂等的**
#
# 【为什么幂等是硬要求】
# 管理入口是走 HTTPS 的，响应可能丢：脚本发出去了、但没收到回复。
# 这时操作者【手里已经有那张码了】（本地生成的），正确的做法是拿同一张码重试。
# 如果重试会被当成「重复 / 出错」，操作者就会改生成第二张 —— 于是线上多了一张
# 谁也不认识的码。所以「同一个码再次登记」必须给一个明确、可确认、不重复发码的结果。

REGISTER_CREATED = "created"                       # 新登记成功
REGISTER_ALREADY_REGISTERED = "already_registered"  # 同一张码之前就登记过，仍然有效未绑定
REGISTER_ALREADY_BOUND = "already_bound"           # 这张码已经被某位学习者用掉了
REGISTER_REVOKED = "revoked"                       # 这张码已被作废
REGISTER_REJECTED = "rejected"                     # 数据库完整性错误，且【不是】"已存在"

VALID_REGISTER_OUTCOMES = (REGISTER_CREATED, REGISTER_ALREADY_REGISTERED,
                           REGISTER_ALREADY_BOUND, REGISTER_REVOKED,
                           REGISTER_REJECTED)


def register_generated_invite(conn, code, pepper):
    """登记一张由外部生成的邀请码。返回上面那几种状态之一（**绝不返回码本身**）。

    【只登记「全新的、未绑定的」码】这里没有任何 learner_id 参数 ——
    从结构上保证它做不到「给已有学习者补发邀请码」那件事
    （那件事需要身份核验，本项目没有，所以 reissue 是停用的）。

    【为什么 IntegrityError 不能一律当成「已登记」】
    完整性错误至少有四种来源：UNIQUE、CHECK、NOT NULL、触发器 RAISE。
    只有「按摘要查得到这一行」才是幂等成功；查不到就说明是别的完整性问题，
    必须如实报 rejected —— 否则数据库出错时我们会告诉操作者「已经登记好了」，
    而他手上那张码其实根本不在库里。
    """
    digest = digest_invite_code(code, pepper)

    try:
        with conn:
            conn.execute(
                "INSERT INTO invites (code_digest, status, created_at) VALUES (?, ?, ?)",
                (digest, INVITE_ACTIVE, _now()),
            )
        return REGISTER_CREATED
    except sqlite3.IntegrityError:
        # 走到这里不代表「已存在」—— 再查一次，用事实说话
        row = conn.execute(
            "SELECT status, learner_id FROM invites WHERE code_digest = ?", (digest,)
        ).fetchone()
        if row is None:
            return REGISTER_REJECTED          # 库里有别的完整性问题

        status, learner_id = row
        if learner_id is not None:
            return REGISTER_ALREADY_BOUND     # 已经被人用掉了
        if status == INVITE_REVOKED:
            return REGISTER_REVOKED           # 已作废（作废过的码不"复活"）
        if status == INVITE_ACTIVE:
            return REGISTER_ALREADY_REGISTERED  # 幂等：同一张码，之前登记过了
        return REGISTER_REJECTED              # 状态不在白名单里（数据坏了）


def issue_replacement_invite(conn, learner_id, pepper):
    """给一个【已经存在】的学习者补发一张新邀请码。返回新码；学习者不存在返回 None。

    > ### ⚠️ 这一轮【没有对外入口】，请先读完再调用
    >
    > 这个函数本身是能用的，但它**故意没有暴露给命令行或网页**。
    > 原因：它只需要一个 `learner_id` 就能造出「指向那个学习者」的新码，
    > 而我们**没有任何办法确认来要码的人就是本人** ——
    > 邀请码是这套设计里唯一的凭证，补发恰恰是「绕过凭证」的操作，
    > 编号又是自增的（猜都猜得到）。留一个这样的入口，
    > 等于给「冒领别人档案」开了一条路。
    >
    > 所以：`reissue` 命令已停用（输入它会得到解释）。
    > 要恢复这项能力，得先有一套身份核验方式。
    > 在那之前，**不要**把它接进任何面向用户的路径。

    【它做的事】给已有学习者补发一张新码（码丢了、或旧码作废之后用）。
    【为什么新码直接写成 bound 并指向这个 learner】
    这样用户把新码输进来时，走的是「已绑定 → 回到原来那个人」那条路，
    档案和偏好原封不动 —— 对应设计文档第 14 节的补救方式：
    「作废旧码 → 发一张新码 → 绑定到同一个 learner」。
    （如果新码是 active 状态，兑换时会给他建一个全新的 learner，档案就丢了。）
    """
    row = conn.execute("SELECT id FROM learners WHERE id = ?", (learner_id,)).fetchone()
    if row is None:
        return None                                   # 没有这个学习者，不能凭空补发

    code = generate_invite_code()
    with conn:
        conn.execute(
            "INSERT INTO invites (code_digest, status, created_at, learner_id) "
            "VALUES (?, ?, ?, ?)",
            (digest_invite_code(code, pepper), INVITE_BOUND, _now(), learner_id),
        )
    return code


def revoke_invite(conn, code, pepper, invalidate_sessions=True):
    """作废一张邀请码。返回 {"revoked": bool, "learner_id": int|None, "sessions_dropped": int}。

    【作废到底意味着什么 —— 这里必须说清楚，因为它很容易被想当然】

    只把 invites.status 改成 revoked，只能挡住「以后再用这张码输一遍」。
    **它挡不住已经进来的浏览器**：那些设备的 cookie 里还揣着 session_id，
    而 session 和 learner 的绑定关系还在 learner_sessions 表里躺着 ——
    换句话说，拿着泄露码的那个人如果已经进来过，他照样能一直看档案。

    所以真正的「作废」必须同时做第二件事：**让那些已绑定的会话失效**。
    失效之后，那些浏览器下一次请求就查不到 learner_id，档案页会把他们
    引导回「输入邀请码」——这才叫「立刻失效」。

    【「失效」是打标记，不是删行 —— 这一点是修正过的】
    早期版本直接把 learner_sessions 那一行 DELETE 掉。问题是：
    那一行同时承担两个职责 ——① 控制访问，② 记住「这个会话属于谁」。
    删掉之后第②件事也没了，于是以后「清空全部个人数据」时就找不到
    这个会话的聊天记录（用户以为删干净了，其实还躺在库里）。
    现在改成把 revoked_at 打上时间戳：访问立刻断掉，关联仍然记得。

    【为什么不动 learner 和偏好】
    因为要的是「断开进入通道」，不是「销毁数据」：
      · learner、learner_preferences 一律保留 —— 补发新码后档案还在
      · messages（聊天记录）一律不碰 —— 它是按 session_id 存的，本来就不归档案管
    失效之后，那个浏览器仍然看得到自己之前的聊天记录（那是它自己问的），
    但看不到任何档案内容 —— 因为它已经不是那位学习者了。

    【被作废的人怎么重新拿到访问权 —— 目前【没有】安全可行的办法】
    理论上可以补发一张新码（issue_replacement_invite），但那条路**已经停用**：
    我们没有身份核验手段，无法确认来要码的人就是本人，
    而补发恰恰是「绕过唯一凭证」的操作 —— 留着它等于给冒领开了门。
    所以现在作废 = 这个人**永久失去**自己的档案（数据还在库里，只是没法安全地还给谁）。
    对「码泄露、必须立刻止血」这个场景，这个代价可能是值得的；
    但如果是正常用户把码弄丢了，作废帮不上他 —— 这属于已知的上线阻塞项。
    """
    digest = digest_invite_code(code, pepper)

    with conn:                                        # 一个事务：改状态 + 断会话要么都成，要么都不成
        row = conn.execute(
            "SELECT id, code_digest, learner_id FROM invites WHERE code_digest = ?",
            (digest,),
        ).fetchone()

        # 和 redeem_invite 一样，找着了再比一次（防的是以后有人把查询改成模糊匹配）
        if row is None or not hmac.compare_digest(row[1], digest):
            return {"revoked": False, "learner_id": None, "sessions_dropped": 0}

        invite_id, _digest, learner_id = row
        conn.execute(
            "UPDATE invites SET status = ?, revoked_at = ? WHERE id = ?",
            (INVITE_REVOKED, _now(), invite_id),
        )

        dropped = 0
        if invalidate_sessions and learner_id is not None:
            # 【打标记，不删行】见上面那段说明：关联必须留着，
            # 否则以后清空全部数据时会漏掉这些会话的聊天记录。
            # 只标记「还有效」的那些，免得把旧的失效时间覆盖掉。
            cur = conn.execute(
                "UPDATE learner_sessions SET revoked_at = ? "
                "WHERE learner_id = ? AND revoked_at IS NULL",
                (_now(), learner_id),
            )
            dropped = cur.rowcount

    return {"revoked": True, "learner_id": learner_id, "sessions_dropped": dropped}


def clear_preferences(conn, learner_id):
    """清空某个学习者的【偏好】。返回 True 表示确实清掉了内容。

    【这是「清空学习档案」，不是「清空全部个人数据」—— 两者差得很远】

    清（只有这一样）：
      · learner_preferences 里他填的那一行（水平 / 目标 / 重点 / 长度 / 语言）

    不清：
      · learners 那一行 —— 他还是同一个人，不需要重新输码
      · invites —— 作废邀请码是另一个动作，这里绝不牵连
      · messages —— 聊天记录按 session_id 存，不归档案管，一个字都不动
      · learner_sessions —— 会话绑定照旧，清完立刻还能用

    清完之后：档案页显示「未设置」，下一次回答走默认偏好。
    """
    if learner_id is None:
        return False
    with conn:
        cur = conn.execute("DELETE FROM learner_preferences WHERE learner_id = ?",
                           (learner_id,))
    return cur.rowcount > 0


# ===================== 清空全部个人数据 =====================

def clear_all_data(conn, learner_id):
    """删除这位学习者在【本系统】里的全部个人数据。返回一份统计字典。

    > ### 范围与边界（说在代码前面，免得被误读）
    >
    > **删的是**：他的聊天记录、学习偏好、会话绑定、学习者身份。
    > **作废的是**：他名下所有邀请码（**不删行** —— 保留「谁用过」的痕迹）。
    > **不碰**：其他学习者、匿名访客的记录、全站 api_usage、
    >          以及**第三方模型服务商**那边可能已经保留的数据。
    > **也不等于物理擦除**：SQLite 文件里的页、备份、快照都不会因此消失。

    【必须在同一个事务里，而且要在删绑定【之前】把会话找出来】
    顺序是设计文档第 13 节定的：
      ① 找出全部会话（**含已失效的绑定** —— 少了这一步就会漏记录）
      ② 删这些会话的 messages
      ③ 删偏好
      ④ 作废他名下的邀请码
      ⑤ 删会话绑定
      ⑥ 删学习者
    ②依赖①的结果，⑤又把①的线索抹掉 —— 所以顺序错了就删不干净。
    任何一步抛异常，整体 ROLLBACK，绝不能留下「删了一半」的状态。

    【为什么用 BEGIN IMMEDIATE】和兑换邀请码同一个道理：
    立刻拿写锁，避免「另一边正在写、这边正在删」交出错乱的结果。
    """
    if learner_id is None:
        return None

    stats = {"sessions": 0, "messages": 0, "preferences": 0,
             "invites": 0, "sessions_deleted": 0, "learners": 0}

    conn.execute("BEGIN IMMEDIATE")
    try:
        # ① 先把他名下的会话全找出来（含已失效的绑定）
        sessions = session_ids_for_learner(conn, learner_id, include_revoked=True)
        stats["sessions"] = len(sessions)

        # ② 删这些会话的聊天记录。
        #    用子查询而不是把 ID 拼进 IN (...)：会话多的时候不会撞上 SQLite 的参数个数上限。
        #    注意先别删 learner_sessions —— 下面这条子查询还指着它。
        if sessions:
            stats["messages"] = conn.execute(
                "DELETE FROM messages WHERE session_id IN "
                "(SELECT session_id FROM learner_sessions WHERE learner_id = ?)",
                (learner_id,),
            ).rowcount

        # ③ 偏好
        stats["preferences"] = conn.execute(
            "DELETE FROM learner_preferences WHERE learner_id = ?", (learner_id,)
        ).rowcount

        # ④ 作废他名下的全部邀请码（改成 revoked，不删行）
        stats["invites"] = conn.execute(
            "UPDATE invites SET status = ?, revoked_at = ? "
            "WHERE learner_id = ? AND status != ?",
            (INVITE_REVOKED, _now(), learner_id, INVITE_REVOKED),
        ).rowcount

        # ⑤ 会话绑定（这时候已经没有用了）
        stats["sessions_deleted"] = conn.execute(
            "DELETE FROM learner_sessions WHERE learner_id = ?", (learner_id,)
        ).rowcount

        # ⑥ 学习者本人
        stats["learners"] = conn.execute(
            "DELETE FROM learners WHERE id = ?", (learner_id,)
        ).rowcount

        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")               # 出错必须整体撤销，不能留半截
        except Exception:
            pass
        raise

    return stats


# ===================== 邀请码：兑换 =====================

# 兑换结果。用「短代号」而不是把话写死在函数里 ——
# 对外说什么话是 app.py 的事，这一层只回答「发生了什么」。
REDEEM_INVALID = "invalid"      # 查不到这个码
REDEEM_REVOKED = "revoked"      # 码被作废了
REDEEM_EXISTING = "existing"    # 码有效，且已经绑定了学习者 → 回到那个人
REDEEM_NEW = "new"              # 码有效且没用过 → 新建学习者

# 【兑换成功、但这次会话绑不上】
# 典型场景：这个浏览器已经绑着另一个学习者了，又拿一张新码来兑。
# 它不算「码无效」，但绝不能算「成功」—— 见 redeem_and_bind 的说明。
REDEEM_BIND_REFUSED = "bind_refused"


def _redeem_locked(conn, digest):
    """兑换的核心逻辑。**在一个已经开着写事务的连接上执行，自己不提交、不回滚。**

    返回 (结果代号, learner_id 或 None)。

    【为什么把「核心逻辑」和「事务边界」拆开】
    因为现在有两个调用方，事务范围不一样：
      · redeem_invite()      —— 只兑换（老入口，测试也在用）
      · redeem_and_bind()    —— 兑换 + 绑定会话，两件事必须在【同一个】事务里
    逻辑写两份迟早会漂移，所以核心留在这里，事务由调用方包。
    """
    row = conn.execute(
        "SELECT id, code_digest, status, learner_id FROM invites WHERE code_digest = ?",
        (digest,),
    ).fetchone()

    # 【为什么查到了还要再比一次】
    # 上面是按摘要精确查的，正常情况下不存在时序问题。
    # 但万一以后有人把查询改成了 LIKE、去掉了 UNIQUE，或者加了别的匹配方式，
    # 常数时间比较就是最后一道保险 —— 它不会因为「前几个字符对了」而变快或变慢。
    if row is not None and not hmac.compare_digest(row[1], digest):
        row = None                                    # 摘要对不上 = 当作没查到

    if row is None:
        return REDEEM_INVALID, None

    invite_id, _digest, status, learner_id = row

    if status == INVITE_REVOKED:
        return REDEEM_REVOKED, None

    if learner_id is not None:
        # 已经绑过学习者了：回到原来那个人（换设备、清 cookie 后重新输码就是这条路径）
        return REDEEM_EXISTING, learner_id

    if status != INVITE_ACTIVE:
        # 防御性分支：状态不在上面几种里（理论上被 CHECK 约束挡住了，真出现说明数据坏了）
        return REDEEM_INVALID, None

    now = _now()
    cur = conn.execute(
        "INSERT INTO learners (created_at, last_seen_at, status) VALUES (?, ?, ?)",
        (now, now, LEARNER_ACTIVE),
    )
    new_learner_id = cur.lastrowid

    conn.execute(
        "UPDATE invites SET status = ?, learner_id = ? WHERE id = ?",
        (INVITE_BOUND, new_learner_id, invite_id),
    )
    return REDEEM_NEW, new_learner_id


def redeem_invite(conn, code, pepper):
    """用邀请码换一个 learner_id。返回 (结果代号, learner_id 或 None)。

    【必须在一个事务里完成「查 → 建学习者 → 回写邀请码」】
    两个人同时用同一个码（比如同一个码被转发给了两个人），
    如果不是原子的，两边都可能读到「还没被用过」，然后各自建一个学习者 ——
    一个码对应了两个人，档案就串了。
    这里用 BEGIN IMMEDIATE 立刻拿写锁，把并发的两次兑换彻底串起来。

    【未知的码绝不能「先建个号再说」】
    查不到就是查不到，直接返回 invalid。

    【注意它不含「绑定会话」】网页那边请用 redeem_and_bind()，
    否则会出现「码被消耗、学习者建好了，但这次会话没绑上」的半截状态。
    """
    digest = digest_invite_code(code, pepper)

    conn.execute("BEGIN IMMEDIATE")                   # 立刻拿写锁，串行化并发兑换
    try:
        outcome, learner_id = _redeem_locked(conn, digest)
        conn.execute("COMMIT")
        return outcome, learner_id
    except Exception:
        try:
            conn.execute("ROLLBACK")                  # 出错必须收尾，否则写锁会一直挂着
        except Exception:
            pass
        raise


# ===================== 会话 ↔ 学习者 =====================

def link_session(conn, learner_id, session_id):
    """把当前浏览器会话绑到某个学习者上。返回最终生效的 learner_id（None = 没绑成）。

    【为什么要检查再插入，而不是直接 INSERT OR REPLACE】
    这是本文件最要紧的一处安全逻辑。
    如果允许「覆盖写」，那么一个已经属于 A 的 session，只要再走一次邀请码流程
    就能被改绑到 B 身上 —— 于是 B 拿着自己的邀请码，就能看到 A 的对话和档案。
    UNIQUE(session_id) 是这条规则在数据库层的兜底：就算代码写错了，
    也不可能插进第二行。

    【三种情况，行为各不相同】
      · 从来没绑过 → 新建一行（有效）
      · 已绑过、且【有效】 → 保持原样，绝不改绑
      · 已绑过、但【已失效】（邀请码被作废过）：
          - 新来的码指向【同一个】学习者 → 把失效标记清掉（这是本人拿新码回来）
          - 新来的码指向【另一个人】     → 拒绝，返回 None（不能凭一张码抢走别人的会话）
    """
    with conn:
        bound, _ok = _bind_locked(conn, learner_id, session_id)
    return bound


def _bind_locked(conn, learner_id, session_id):
    """绑定的核心逻辑。**在一个已经开着事务的连接上执行，自己不提交。**

    返回 (这个会话最终归属的 learner_id 或 None, 本次是否算「绑到了请求的那个学习者」)。

    【为什么要返回第二个值】两个调用方要的东西不一样：
      · link_session()     —— 只关心「现在归谁」（老行为，返回第一个值）
      · redeem_and_bind()  —— 必须知道「到底绑成功了没有」，
        因为没成功的话，刚才那次兑换必须一起回滚（否则码被白白消耗掉）
    """
    now = _now()

    row = conn.execute(
        "SELECT learner_id, revoked_at FROM learner_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    # ---- 情况一：没绑过 ----
    if row is None:
        conn.execute(
            "INSERT INTO learner_sessions "
            "(learner_id, session_id, created_at, last_seen_at) VALUES (?, ?, ?, ?)",
            (learner_id, session_id, now, now),
        )
        return learner_id, True

    existing_learner, revoked_at = row

    # ---- 情况二：绑过、还有效 ----
    if revoked_at is None:
        # 已经是这个学习者了 → 相当于「本来就在」，算绑成功（重复输同一个码就是这条路）
        return existing_learner, (existing_learner == learner_id)

    # ---- 情况三：绑过、但已失效 ----
    if existing_learner != learner_id:
        # 想用一张属于别人的码，接管这个已失效的会话 —— 不允许。
        # （会话虽然失效了，但它仍然是「某个人的浏览器」，不该被另一张码认领走）
        return None, False

    conn.execute(                                # 本人拿新码回来：恢复这一行
        "UPDATE learner_sessions SET revoked_at = NULL, last_seen_at = ? WHERE session_id = ?",
        (now, session_id),
    )
    return existing_learner, True


def redeem_and_bind(conn, code, pepper, session_id):
    """兑换邀请码，并把这次会话绑上去。**两件事必须一起成功，否则一起不做。**

    > ### 这是在修一个真实的半截状态
    >
    > 旧流程是「先 redeem_invite()（自己提交），再 link_session()」。问题在于
    > 第二步是**可能失败**的：比如这个浏览器已经绑着另一个学习者了，就不允许改绑。
    > 而第一步早就提交了 —— 结果是：
    >   · 邀请码被标记成 bound（白白消耗掉了）
    >   · 新学习者建出来了，却没有任何会话能进得去（孤儿）
    >   · 浏览器那边还是匿名，但网页返回 302 并记了「邀请成功」
    >
    > 所以现在把两件事放进【同一个事务】：绑定失败 → 整体 ROLLBACK →
    > 邀请码仍然可用、不会多出孤儿学习者、网页也能如实报错。

    返回 (结果代号, learner_id 或 None)。结果代号多了一种 REDEEM_BIND_REFUSED。
    """
    digest = digest_invite_code(code, pepper)

    conn.execute("BEGIN IMMEDIATE")                   # 立刻拿写锁，把并发兑换串起来
    try:
        outcome, learner_id = _redeem_locked(conn, digest)

        if outcome in (REDEEM_INVALID, REDEEM_REVOKED):
            conn.execute("ROLLBACK")                  # 没改过任何东西，回滚只是为了收尾
            return outcome, None

        bound, ok = _bind_locked(conn, learner_id, session_id)
        if not ok:
            # 【关键的一行】这一次兑换不能留 —— 码要还能用，也不能留下孤儿学习者
            conn.execute("ROLLBACK")
            return REDEEM_BIND_REFUSED, None

        conn.execute("COMMIT")
        return outcome, bound
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


def learner_id_for_session(conn, session_id):
    """查这个会话【当前有效】地属于哪个学习者。没有绑定、或者绑定已失效 → None。

    【为什么必须带上 revoked_at IS NULL】
    这是「作废」真正生效的地方：被作废的浏览器 cookie 还在，
    但查出来是 None，于是它就成了匿名访客 —— 看不到任何档案内容。
    """
    row = conn.execute(
        "SELECT learner_id FROM learner_sessions "
        "WHERE session_id = ? AND revoked_at IS NULL",
        (session_id,),
    ).fetchone()
    return row[0] if row else None


def session_is_revoked(conn, session_id):
    """这个会话的绑定是不是【已经失效】了？（从来没绑过 → False）

    【谁在用】app.py 每次请求开头都会问一次：如果这个浏览器手里的会话已经失效，
    就必须给它换一个新的 session_id。理由见 app.py 里那段说明 ——
    不换的话，它在失效之后问的话会继续挂在这个 session_id 下，
    而那个 id 仍然关联着原来的学习者，将来清空数据时会把这些无关的消息一起删掉。
    """
    row = conn.execute(
        "SELECT revoked_at FROM learner_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return bool(row and row[0] is not None)


def session_ids_for_learner(conn, learner_id, include_revoked=True):
    """找出这位学习者名下的【全部】会话 ID。

    【为什么默认连已失效的也一起返回】
    这些行是删除全部数据时的关键：用户可能先被作废（会话失效），
    过一阵才来要求清空全部数据。如果只看「还有效的绑定」，
    那些失效会话的聊天记录就会被漏掉 —— 用户以为删干净了，其实还躺在库里。
    """
    sql = "SELECT session_id FROM learner_sessions WHERE learner_id = ?"
    if not include_revoked:
        sql += " AND revoked_at IS NULL"
    return [row[0] for row in conn.execute(sql, (learner_id,))]


def touch_learner(conn, learner_id):
    """更新「最近一次出现时间」。只记时间，不记任何身份信息。"""
    if learner_id is None:
        return
    with conn:
        conn.execute("UPDATE learners SET last_seen_at = ? WHERE id = ?",
                     (_now(), learner_id))
        conn.execute("UPDATE learner_sessions SET last_seen_at = ? WHERE learner_id = ?",
                     (_now(), learner_id))


# ===================== 偏好：读 / 写 =====================

# 默认偏好。【为什么要有默认值】
# 用户第一次进来还没填档案时，也要能正常回答 —— 这时候用一套保守的默认值：
# 不假设水平、英文为主配生词中文解释、长度正常。
DEFAULT_PREFERENCES = {
    "level_code": "b1",
    "level_uncertain": 1,              # 1 = 用户没确认过水平，让模型别太当真
    "language_mode": "zh_pair",        # 默认中英对照：初学者更需要中文解释
    "length_mode": "normal",
    "goal_code": None,
    "focus_code": None,
}


def get_preferences(conn, learner_id):
    """读某个学习者的偏好。没填过、或者没有学习者 → 返回 None。

    【返回 None 而不是默认值，是刻意的】
    「用户填过档案」和「用户没填，用的是默认」在页面上要显示得不一样：
    前者显示他填的内容，后者要提示「还没设置，去设置一下」。
    如果这里悄悄返回默认值，上层就分不清这两种情况了。
    默认值由调用方按需取（DEFAULT_PREFERENCES）。
    """
    if learner_id is None:
        return None

    row = conn.execute(
        "SELECT level_code, level_uncertain, language_mode, length_mode, goal_code, focus_code "
        "FROM learner_preferences WHERE learner_id = ?",
        (learner_id,),
    ).fetchone()
    if row is None:
        return None

    return {
        "level_code": row[0],
        "level_uncertain": row[1],
        "language_mode": row[2],
        "length_mode": row[3],
        "goal_code": row[4],
        "focus_code": row[5],
    }


def save_preferences(conn, learner_id, level_code, level_uncertain,
                     language_mode, length_mode, goal_code=None, focus_code=None):
    """写入（或覆盖）某个学习者的偏好。返回 False 表示输入不合法、什么都没写。

    【为什么非法输入要「返回 False」而不是「纠正成默认值」】
    悄悄纠正会让用户以为设置生效了，其实存的是别的东西。
    对外给一句明确的话，比默默改掉诚实得多。

    【为什么在这里再校验一遍白名单】
    数据库的 CHECK 约束是最后一道防线，但它会在【写的时候】抛异常 ——
    那对用户来说就是一个 500 错误。在这里先查一遍，就能变成一句人能看懂的话。
    两层都要有：一层给人话，一层给数据完整性。

    goal_code / focus_code 允许为空（用户可以跳过一项不填）。
    """
    if level_code not in VALID_LEVELS:
        return False
    if language_mode not in VALID_LANGUAGE_MODES:
        return False
    if length_mode not in VALID_LENGTH_MODES:
        return False
    if goal_code is not None and goal_code not in VALID_GOALS:
        return False
    if focus_code is not None and focus_code not in VALID_FOCUS:
        return False

    level_uncertain = 1 if level_uncertain else 0    # 归一成 0/1，免得写进别的值

    with conn:
        conn.execute(
            "INSERT INTO learner_preferences "
            "(learner_id, level_code, level_uncertain, language_mode, length_mode, "
            " goal_code, focus_code, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            # learner_id 是主键：冲突时更新同一行，绝不新建
            "ON CONFLICT(learner_id) DO UPDATE SET "
            "  level_code = excluded.level_code,"
            "  level_uncertain = excluded.level_uncertain,"
            "  language_mode = excluded.language_mode,"
            "  length_mode = excluded.length_mode,"
            "  goal_code = excluded.goal_code,"
            "  focus_code = excluded.focus_code,"
            "  updated_at = excluded.updated_at",
            (learner_id, level_code, level_uncertain, language_mode, length_mode,
             goal_code, focus_code, _now()),
        )
    return True


# ===================== 命令行：给组织者用的两个操作 =====================
#
# 【为什么需要一个命令行】邀请码是「组织者生成 → 手动发给用户」的。
# 没有这个入口，库里就永远不会有一张邀请码，整个功能等于用不了。
#
# 【为什么不是网页上的一个按钮】「谁能发码」是个权限问题，
# 而这一版没有管理员登录。做一个不设防的发码页面等于让任何人都能给自己发一个码。
# 命令行要能在这台机器上执行，本身就已经是一道门槛了。
#
# 用法（在 ai_tutor 文件夹里）：
#     python profile_store.py new-invite
#     python profile_store.py revoke <邀请码>
#     python profile_store.py stats

def _main(argv):
    import os                     # 这里才 import：正常导入本模块时不需要这些
    import env_utils              # 和网页共用同一份 .env 读取逻辑

    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_utils.load_dotenv(os.path.join(base_dir, ".env"))

    # 【密钥相关】只检查有没有，绝不打印内容
    pepper = os.environ.get("INVITE_CODE_PEPPER", "")
    if not pepper:
        print("没有找到 INVITE_CODE_PEPPER。请先设置环境变量，或在 .env 里加一行。")
        return 1

    # 【和 app.py 同一条拦截】只查「非空」会放过 .env.example 里那个公开的占位符 ——
    # 用它发出来的邀请码，摘要谁都能离线爆破。organizer 这条路上一样要拦住。
    if env_utils.is_example_secret(pepper, env_utils.EXAMPLE_PEPPER):
        print(env_utils.example_secret_message("INVITE_CODE_PEPPER"))
        return 1

    db_path = os.environ.get("CHAT_DB_PATH") or os.path.join(base_dir, "chat.db")
    if not os.path.isdir(os.path.dirname(os.path.abspath(db_path))):
        print("数据库所在目录不存在：" + db_path)
        return 1

    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        ensure_schema(conn)

        if not argv or argv[0] == "new-invite":
            code = generate_invite_code()
            if not create_invite(conn, code, pepper):
                print("生成时撞上了已有的码，请再执行一次。")
                return 1
            # 【只在这里、只显示这一次】之后库里只有摘要，找不回来。
            # 所以这一句是刻意的，不是漏了「不要打印敏感信息」——
            # 组织者必须看到明文才能把它发给用户。
            print("新邀请码（只显示这一次，请立刻复制发给用户）：")
            print()
            print("    " + code)
            print()
            print("库里只存了它的摘要，这个码关掉就再也看不到。")
            return 0

        if argv[0] == "revoke" and len(argv) == 2:
            result = revoke_invite(conn, argv[1], pepper)
            if not result["revoked"]:
                print("没有找到这张邀请码。")
                return 1

            # 【必须把「做了什么」说清楚】只印一句「已作废」会让人以为万事大吉，
            # 而实际上它到底断了什么、没断什么，直接决定这次处置有没有效。
            print("已作废。这张码不能再用来进入。")
            print("同时断开了 " + str(result["sessions_dropped"]) + " 个已绑定的浏览器会话"
                  "（它们下次请求就会被引导回输码页）。")
            if result["learner_id"] is not None:
                print()
                print("⚠️ 这位学习者现在没有任何入口了，而且【目前无法补回】：")
                print("   学习者编号 " + str(result["learner_id"]) + " 的档案和偏好仍然保存在库里，")
                print("   但我们没有任何身份核验手段，不能确认「来要码的人」就是本人，")
                print("   所以补发命令已停用 —— 详见 python profile_store.py reissue。")
                print()
                print("   实际影响：作废会让人【永久失去】自己的档案。")
                print("   对「码泄露」这种场景这或许正是你要的；但正常用户丢了码就没救了。")
            else:
                print()
                print("（这张码还没被任何人用过，所以没有学习者受影响。）")
            return 0

        if argv[0] == "reissue":
            # 【为什么这条命令被停掉了，而不是继续用】
            # issue_replacement_invite() 只要给一个编号就能造出「指向那个学习者」的新码。
            # 可我们【没有任何办法确认打电话来要码的人就是本人】——
            # 邀请码是这套设计里唯一的凭证，而补发恰好是「绕过凭证」的操作：
            # 谁能说出一个编号（编号还是自增的，猜都猜得到），谁就能拿到别人档案的钥匙。
            # 与其留一个「看起来能恢复、实际上是身份冒用入口」的命令，不如先关掉。
            print("这条命令已停用。")
            print()
            print("原因：本项目没有账号密码，也没有任何身份核验手段，")
            print("      无法确认「来要码的人」就是那位学习者本人。")
            print("      编号是自增的，谁都能猜 —— 能用编号补发，就等于谁都能拿到别人的档案。")
            print()
            print("数据没有丢：那位学习者的档案和偏好都还在库里。")
            print("但目前【无法安全地确认归属】，所以暂时不提供「找回原档案」这条路。")
            print("要恢复这项能力，需要先设计一套身份核验方式（见 README 的已知限制）。")
            return 1

        if argv[0] == "stats":
            # 【只报数量，绝不列码】摘要也不能打印 —— 它同样是凭证的一部分
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM invites GROUP BY status").fetchall()
            learners = conn.execute("SELECT COUNT(*) FROM learners").fetchall()
            sessions = conn.execute("SELECT COUNT(*) FROM learner_sessions").fetchone()[0]
            prefs = conn.execute("SELECT COUNT(*) FROM learner_preferences").fetchone()[0]
            print("邀请码：" + ("、".join(s + "=" + str(n) for s, n in rows) or "（还没有）"))
            print("学习者：" + str(learners[0][0]))
            print("已绑定会话：" + str(sessions))
            print("填过档案的学习者：" + str(prefs))
            # 【列编号只为排查】对照测试记录、定位「哪个编号出过问题」时用得上。
            # 它是内部自增数字，不含身份信息。
            # 【注意】不要拿它当「补发的依据」—— reissue 已停用，理由见那条命令的说明。
            ids = [str(r[0]) for r in conn.execute("SELECT id FROM learners ORDER BY id")]
            print("学习者编号：" + ("、".join(ids) or "（还没有）"))
            return 0

        print("用法：")
        print("    python profile_store.py new-invite       生成一张新邀请码")
        print("    python profile_store.py revoke <码>      作废（同时断开已绑定的会话）")
        print("    python profile_store.py stats            看统计（不显示任何码）")
        print()
        print("注意：reissue 已于本轮停用 —— 没有身份核验，不能安全地把码补发给「本人」。"
              "输入该命令会得到详细说明。")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
