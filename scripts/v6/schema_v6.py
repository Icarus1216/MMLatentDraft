"""
schema_v6.py
============
v6 continuous-CoT-with-blank latent 数据 schema 定义 + 强校验。

核心设计 (v6 vs v5):
  - 形式上是一段流畅自然语言 CoT, 不再有 <step>/<action>/<observation> 结构.
  - latent 段以 <|latent|>...<|/latent|> 包围 (与 rld/data.py 训练管线一致),
    段内由 <|pause|> 重复 K 次占位. NLDDataset.__getitem__ 会:
      (1) 把 <|pause|> 替换为 <|latent|>;
      (2) 在每个 <|latent|> 后追加 <|/latent|>.
    因此本 schema 校验时, latent 段 = "<|latent|> ... <|/latent|>", 段内 K 个 <|pause|>.
  - 每个 latent 段之前是"意图陈述句 (intent)", 之后是"锚定判断句 (anchor_phrase)".
  - **留白契约**: 删掉 latent 段后, anchor_phrase 必须出现"悬空指代 / 未定选择 /
    未解释因果", 即文本不可能由 text-only 自洽续写到正确 final answer.
  - **qualitative-only**: 不允许 cot_text/question/answer/anchor 出现精确数值
    (角度 °/deg, 长度 m/cm/inch/ft, 像素/坐标, 百分比, 以及 "thirty degrees" 这类
    文字数字+单位). 推理只能停留在定性方向 ("her left"), 序数选择 ("the leftmost"),
    关系比较 ("closer than"). 数值只能在隐空间存在.
  - **per-boundary 多次极浅思考 (v6.3)**: 每个 substep 触发一次 latent
    (一个 <|pause|>), 该 latent 仅做 k ∈ {1,2} 步的极浅演化, 用于在子结论前
    "加深关键位置的层深", 把本来可能一步答出来的子问题"提升答对的概率".
    substep 之间用自然语言桥接, 每个子问题都有自己独立的 latent 触发点.
    设计立足: PonderLM-2 证明 1 步 hidden 等价于参数加倍; 我们用 1~2 步做
    "短而频繁"的 depth boost, 避免 hidden 长链坍塌.
  - 训练侧承诺 (per-boundary 独立 stages, 与 rld/data.py + rld/model_v2.py v6.2 契约一致):
      * 每个 substep 在序列里写一个 <|pause|> (不是 K 个);
      * 该 substep 的 k_latent (= 2/3/4) 决定该 boundary 的隐空间迭代步数;
      * 该 substep 配 k_latent 个 stage 的 key tokens, 每 stage 一个内容词组,
        从 anchor_phrase / intent / latent_hint 抽取;
      * latent_thinker 在每次 boundary 触发时按当前 boundary 的 stages 做 LASER
        forward-KL 蒸馏 (温度 1.0, eta 0.6), key-token 窗口随 stage 演化收敛;
      * <|/latent|> 紧跟 <|latent|>, 由 latent_thinker.exit_token_loss 单独监督;
      * anchor_phrase 与 <answer>...</answer>: 正常 CE, 由训练侧自动加权.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter


# ============================================================
# 常量
# ============================================================

DATA_VERSION = "v6.continuous_blank_cot.5.0_waypoint_k1to3_bottleneck"

# 仍沿用 v5 的 task_type 体系 (高价值类型最适合 latent reasoning)
ALLOWED_TASK_TYPES = {
    "perspective",
    "geometry",
    "fine_grained",
    "counterfactual",
    "multi_hypothesis",
    "physical",
    "comparison",
    "spatial_relation",
    "counting",
    "detail_observation",
    "temporal",
    "ocr",
}
HIGH_VALUE_TASK_TYPES = {
    "perspective", "geometry", "fine_grained",
    "counterfactual", "multi_hypothesis",
}
MEDIUM_VALUE_TASK_TYPES = {"physical", "comparison"}
LOW_VALUE_TASK_TYPES = {
    "spatial_relation", "counting",
    "detail_observation", "temporal", "ocr",
}

# substep 数量 = <|latent|>...<|/latent|> 段数
MIN_SUBSTEPS = 2
MAX_SUBSTEPS = 5
# 每段 latent 数量 (k = 该 boundary 的 latent 演化步数; v6.5: 1~3 步).
#   k=1: 单步深化, 纯 depth boost (PonderLM-2 等价).
#   k=2: 1 步 concept drift (一次 "由 a -> b" 的概念漂移).
#   k=3: 2 步 concept drift, 用于物理/多视角空间这种确实需要 3 段 hidden 演化的题.
# 触发次数本身不限制 (= n_substeps), 只限制每次的演化深度.
ALLOWED_K_LATENT = {1, 2, 3}
MIN_K_LATENT = 1
MAX_K_LATENT = 3

# v6.5 soft K_s (key_token count per stage):
#   - Not strictly fixed at 4; allow [3, 5] to follow the natural semantic
#     length of each stage.
#   - abstract role: typically sparse, prefer 3-4 tokens.
#   - concrete role: typically denser, prefer 4-5 tokens.
#   - unified role : prefer 4 (middle).
#   - LASER / PonderLM-2 key-token window is by design "intent-natural-length",
#     not a fixed constant.
MIN_KEY_TOKENS_PER_STAGE = 3
PREFERRED_KEY_TOKENS_PER_STAGE = 4
MAX_KEY_TOKENS_PER_STAGE = 5

# 文本长度
MAX_QUESTION_WORDS = 35
MAX_ANSWER_WORDS = 25
MIN_COT_WORDS = 60
MAX_COT_WORDS = 280
MIN_HOPS = 2
MAX_HOPS = 6

# 留白契约关键词
# anchor_phrase 必须以下列开场之一开头 (或紧接其后 1~3 个词内出现)
# —— 强制使用"指代延续 / 选择落地 / 因果延伸"中的一种.
# 允许的 dash 字符 (普通 - / em-dash — / en-dash – / 双 em-dash —— 等)
_DASH = r"[\-\u2013\u2014]"

# Qualitative-only: 禁止 cot_text / question / answer / anchor_phrase 出现精确数值 + 单位.
# 命中即视为 "latent 内容泄露到 text", 拒收.
# 覆盖: 角度 (30°, 30 deg, thirty degrees), 长度 (2 m, 5 cm, 4 inches),
#       像素/坐标 (x=120, (120, 80), 120 px), 百分比 (30 %).
_NUMERIC_UNIT_RE = re.compile(
    r"""(?xi)
    (?:                                        # (a) 数字 + 单位
        \b\d+(?:\.\d+)?\s*(?:
            \u00b0 | deg\b | degrees? |
            cm\b | mm\b | km\b | metres?\b | meters?\b |
            inch(?:es)?\b | feet\b | ft\b |
            px\b | pixels?\b |
            %
        )
        |
        \b\d+(?:\.\d+)?\s*m\b                  # 单字母 m (单独处理避免误伤 "3 men")
    )
    |
    (?:                                        # (b) 文字数字 + 单位
        \b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|
            eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|
            eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|
            eighty|ninety|hundred)
        (?:[\s\-](?:one|two|three|four|five|six|seven|eight|nine))?
        \s+(?:degrees?|metres?|meters?|centim[a-z]+|inches?|feet|percent)\b
    )
    |
    (?:                                        # (c) 坐标式: x=120, y = 80, (120, 80)
        \b[xyz]\s*=\s*\d+ |
        \(\s*\d+\s*,\s*\d+\s*(?:,\s*\d+\s*)?\)
    )
    """
)

def _has_numeric_violation(text: str) -> Optional[str]:
    """qualitative-only 检查: 返回命中的子串, None 表示合规."""
    if not text:
        return None
    m = _NUMERIC_UNIT_RE.search(text)
    return m.group(0) if m else None

# 留白开场判定 (v6.1, 更宽松):
#   规则 A (opener phrase): anchor 以下列"开场短语"开头, 大小写/符号不敏感.
#     using / with / under / given / applying / whatever / within /
#     among / of / out\s+of / following / mapped / projected / relative\s+to /
#     call / call\s+(?:it|that|this) / once / from / for /
#     that / this / those / these / it /
#     [- — –]+ (the|that|this|it|those|these|its) /
#     so / therefore / thus / hence / since / because / consequently / only
#   规则 B (referent token in head): anchor 的前 14 个词中至少出现一次:
#     that / this / those / these / it / them / their / its /
#     θ / φ / Δ / λ / α / β / γ /
#     him / her / his / he / she
#   合规判定 = (规则A 命中) AND (规则B 命中).
#   如果只满足 A 不满足 B (例如 "Therefore the imminent motion is..."), 视为
#   "假留白" (没有真正复用前文 referent), 拒收.
_OPENER_PHRASES = re.compile(
    rf"^\s*(?:"
    r"using\b|with\b|under\b|given\b|applying\b|whatever\b|within\b|"
    r"among\b|of\b|out\s+of\b|following\b|mapped\b|projected\b|"
    r"relative\s+to\b|call\b|once\b|from\b|for\b|"
    r"that\b|this\b|those\b|these\b|it\b|they\b|"
    # v6.5 fix: 新增 LLM 常用的合法 BLANK opener (实质上都是 coreference / causal continuation)
    r"adding\b|combining\b|calling\b|reading\b|resolving\b|"
    r"after\s+(?:that|those|these|the|this)\b|"
    r"now\b|"
    # v6.5d E1 fix: 拓宽 anchor 句式多样性 - 增加 F2/F3/F4 骨架的 opener
    # F3 条件式: when/if (后续仍走 _REFERENT_TOKEN 校验, 不会误放无指代句)
    # F2 代词主语: 仅允许 "her own / his own / its own / their own" (明确所有格延续) 与
    #              "it then|now|already|still|just" / "they all|both|then|now" 紧凑延续形式;
    #              直接 "Her body is..." 这类裸代词 + 普通名词 不算延续, 应保留拒收.
    r"when\b|if\b|"
    r"(?:her|his|its|their)\s+own\b|"
    r"it\s+(?:then|now|already|still|just|points|lines|maps|sits|fronts|tilts|lands|opens|closes)\b|"
    r"they\s+(?:all|both|then|now)\b|"
    r"she\s+(?:then|now)\b|he\s+(?:then|now)\b|"
    rf"{_DASH}+\s*(?:that\b|this\b|the\b|it\b|those\b|these\b|its\b)|"
    r"so\b|therefore\b|thus\b|hence\b|since\b|because\b|consequently\b|only\b"
    r")",
    re.IGNORECASE,
)
_REFERENT_TOKEN = re.compile(
    r"\b(?:that|this|those|these|it|them|their|its|they"
    r"|\u03b8|\u03c6|\u0394|\u03bb|\u03b1|\u03b2|\u03b3"
    r"|him|her|his|he|she)\b"
    # v6.5 fix: \u5355\u4e2a\u5927\u5199\u7f57\u9a6c\u5b57\u6bcd / \u5e26\u4e0b\u6807\u7684\u5b57\u6bcd\u4e5f\u662f\u5408\u6cd5 referent
    # \u4f8b "Using C", "Calling those anchors L and R", "with axis A"
    # \u7528 (?-i:...) \u5173\u95ed\u5927\u5c0f\u5199\u4e0d\u654f\u611f, \u907f\u514d "a"/"i" \u8bef\u547d\u4e2d
    r"|(?-i:\b[A-Z](?:_[A-Za-z0-9]+)?\b)",
    re.IGNORECASE,
)


def _has_blank_opener(anchor: str) -> bool:
    """anchor 是否符合留白契约 (开场短语 + 前 14 词内出现指代标记)."""
    if not anchor:
        return False
    if not _OPENER_PHRASES.match(anchor):
        return False
    head_words = anchor.split()[:14]
    head = " ".join(head_words)
    return bool(_REFERENT_TOKEN.search(head))


# 兼容 schema 里旧的引用 (validate 中改成调 _has_blank_opener)
ANCHOR_OPENERS_REGEX = _OPENER_PHRASES

# 整段 CoT 中被禁止的子串 (出现即拒) —— 防止结构化标签泄漏 / 分段排版
FORBIDDEN_COT_SUBSTRINGS = (
    "subq:", "sub-q:", "subquestion", "sub-question", "sub q:",
    "step 1:", "step 2:", "step 3:", "step 4:", "step 5:",
    "<sub", "</sub", "<step", "</step",
    "phase=", "<action", "<observation",
    "\n\n",  # 双换行 = 分段, 不允许
    "1. ", "2. ", "3. ", "4. ", "5. ",  # 编号列表
    "(a)", "(b)", "(c)",
    "let me reconsider", "actually,",
    "wait,", "wait.", "hmm,", "hmm ",
    "i'm not sure", "i am not sure",
    "on second thought",
)

# anchor_phrase 中禁止出现的 hedge 词 (复用 v5 标准)
FORBIDDEN_ANCHOR_HEDGES = (
    "wait ", "wait,", "actually,", "actually ",
    "i am not sure", "i'm not sure",
    "hmm,", "hmm ",
    "let me reconsider",
    "on second thought",
    "perhaps", "maybe", "i guess",
)

# 已知顶层字段 (其他视为 unexpected, 仅记报表)
KNOWN_TOP_FIELDS = {
    "sample_id", "data_source", "data_version",
    "image_paths", "task_type", "difficulty_hops",
    "n_substeps", "latent_necessity", "non_verbal_signal",
    "question", "answer",
    "substeps", "cot_text",
    "messages",
    "origin", "original_sample_ref", "src_task_type",
    "latent_key_tokens", "num_stages", "reasoning_for_training",
}
KNOWN_SUBSTEP_FIELDS = {
    "substep_id", "intent",
    "non_verbal_signal", "latent_necessity", "k_latent",
    "is_critical", "latent_hint",
    "anchor_phrase", "leak_check_phrase",
}


# ============================================================
# 渲染 / 解析 helpers
# ============================================================

# Latent 边界 token (与 rld/data.py LATENT_TOKEN / LATENT_END_TOKEN 严格一致)
LATENT_START = "<|latent|>"
LATENT_END   = "<|/latent|>"
PAUSE_TOKEN  = "<|pause|>"
# 别名, 保留旧引用名不破坏其他模块
LATENT_TOKEN = LATENT_START
BOT_TAG = LATENT_START
EOT_TAG = LATENT_END

# latent 段: <|latent|> ... <|/latent|>
_BOT_BLOCK_RE = re.compile(r"<\|latent\|>(.*?)<\|/latent\|>", re.DOTALL)
# latent 段内的占位 token (<|pause|>): NLDDataset 会把它们 replace 为 <|latent|>
_PAUSE_TOK_RE = re.compile(r"<\|pause\|>")
# 兼容旧名: schema 内仍叫 _LATENT_TOK_RE, 但匹配的是 <|pause|>
_LATENT_TOK_RE = _PAUSE_TOK_RE
_THINK_RE = re.compile(r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$",
                       re.DOTALL)

def _word_count(s: str) -> int:
    return len(re.findall(r"\S+", s or ""))

def _strip_latent_blocks(cot_text: str) -> str:
    """删除所有 <|latent|>...<|/latent|> 段, 用单空格替换. 用于 leak-test."""
    return re.sub(r"\s*<\|latent\|>.*?<\|/latent\|>\s*", " ",
                  cot_text, flags=re.DOTALL)

def _extract_anchor_segments(cot_text: str) -> List[str]:
    """切分 cot_text, 返回每个 <|/latent|> 之后到下一个 <|latent|> 前(或文本结尾) 的片段."""
    out: List[str] = []
    spans = []
    for m in _BOT_BLOCK_RE.finditer(cot_text):
        spans.append((m.start(), m.end()))
    if not spans:
        return out
    n = len(cot_text)
    for i, (s, e) in enumerate(spans):
        nxt_start = spans[i + 1][0] if i + 1 < len(spans) else n
        seg = cot_text[e:nxt_start]
        out.append(seg.strip())
    return out


# ============================================================
# 校验
# ============================================================

def validate_v6_sample(sample: Dict) -> Tuple[bool, str]:
    if not isinstance(sample, dict):
        return False, "sample_not_dict"

    # ---- 必填顶层 ----
    q = (sample.get("question") or "").strip()
    a = (sample.get("answer") or "").strip()
    tt = (sample.get("task_type") or "").strip()
    hops = sample.get("difficulty_hops")
    n_sub = sample.get("n_substeps")
    substeps = sample.get("substeps")
    cot = sample.get("cot_text")
    if not q:
        return False, "empty_question"
    if not a:
        return False, "empty_answer"
    if tt not in ALLOWED_TASK_TYPES:
        return False, f"bad_task_type:{tt}"
    if not (isinstance(hops, int) and MIN_HOPS <= hops <= MAX_HOPS):
        return False, f"bad_difficulty_hops:{hops}"
    if not (isinstance(n_sub, int) and MIN_SUBSTEPS <= n_sub <= MAX_SUBSTEPS):
        return False, f"bad_n_substeps:{n_sub}"
    if not (isinstance(substeps, list) and len(substeps) == n_sub):
        return False, f"substeps_count_mismatch:{len(substeps) if isinstance(substeps,list) else 'NA'}"
    if not (isinstance(cot, str) and cot.strip()):
        return False, "empty_cot_text"

    # ---- 字段长度 ----
    if _word_count(q) > MAX_QUESTION_WORDS:
        return False, "question_too_long"
    if _word_count(a) > MAX_ANSWER_WORDS:
        return False, "answer_too_long"

    # ---- cot_text 顶层结构 ----
    m = _THINK_RE.match(cot.strip())
    if not m:
        return False, "cot_text_missing_think_answer_tags"
    inner_cot = m.group(1).strip()
    inner_ans = m.group(2).strip()
    if not inner_cot:
        return False, "empty_inner_cot"
    # answer tag 内必须与 top-level answer 一致 (宽松匹配)
    if not _answer_consistent(a, inner_ans):
        return False, "answer_tag_mismatch_top_answer"

    # ---- cot 词数 ----
    wc = _word_count(_strip_latent_blocks(inner_cot))
    if wc < MIN_COT_WORDS:
        return False, f"cot_too_short:{wc}"
    if wc > MAX_COT_WORDS:
        return False, f"cot_too_long:{wc}"

    # ---- 禁词检查 ----
    cot_l = inner_cot.lower()
    for bad in FORBIDDEN_COT_SUBSTRINGS:
        if bad in cot_l:
            return False, f"cot_contains_forbidden:{bad.strip()!r}"

    # ---- qualitative-only: 禁止精确数值 + 单位 ----
    # 这是为了让 latent 内容只能停留在隐空间, 不被复述到 text.
    for label, txt in (("question", q), ("answer", a),
                       ("cot_text", _strip_latent_blocks(inner_cot))):
        hit = _has_numeric_violation(txt)
        if hit:
            return False, f"{label}_numeric_violation:{hit!r}"

    # ---- <|latent|>...<|/latent|> 段数 = n_substeps ----
    bot_blocks = _BOT_BLOCK_RE.findall(inner_cot)
    if len(bot_blocks) != n_sub:
        return False, f"latent_block_count_mismatch:{len(bot_blocks)}_vs_{n_sub}"

    # ---- 每段 <|latent|>...<|/latent|> 内仅含 <|pause|> 占位和空白 ----
    for i, blk in enumerate(bot_blocks):
        # 用 <|pause|> 全部去掉后, 应该只剩空白
        residue = _PAUSE_TOK_RE.sub("", blk).strip()
        if residue:
            return False, f"latent_block{i+1}_has_non_pause:{residue[:40]!r}"
        n_lat = len(_PAUSE_TOK_RE.findall(blk))
        if not (MIN_K_LATENT <= n_lat <= MAX_K_LATENT):
            return False, f"latent_block{i+1}_bad_k_pause:{n_lat}"

    # ---- substeps 字段逐条校验 + 与 cot_text 一致 ----
    anchor_segments = _extract_anchor_segments(inner_cot)
    if len(anchor_segments) != n_sub:
        return False, f"anchor_seg_count_mismatch:{len(anchor_segments)}_vs_{n_sub}"

    seen_anchors: List[str] = []
    for i, st in enumerate(substeps):
        if not isinstance(st, dict):
            return False, f"substep{i+1}_not_dict"
        if st.get("substep_id") != i + 1:
            return False, f"substep{i+1}_bad_id:{st.get('substep_id')}"

        intent = (st.get("intent") or "").strip()
        anchor = (st.get("anchor_phrase") or "").strip()
        leak_chk = (st.get("leak_check_phrase") or "").strip()
        latent_hint = (st.get("latent_hint") or "").strip()
        nvs = (st.get("non_verbal_signal") or "").strip()
        ln = st.get("latent_necessity")
        kl = st.get("k_latent")
        is_crit = st.get("is_critical")

        if not intent:
            return False, f"substep{i+1}_empty_intent"
        if not anchor:
            return False, f"substep{i+1}_empty_anchor_phrase"
        if not leak_chk:
            return False, f"substep{i+1}_empty_leak_check_phrase"
        if not latent_hint:
            return False, f"substep{i+1}_empty_latent_hint"
        if not nvs:
            return False, f"substep{i+1}_empty_non_verbal_signal"
        if not (isinstance(ln, int) and 0 <= ln <= 3):
            return False, f"substep{i+1}_bad_latent_necessity:{ln}"
        if not (isinstance(kl, int) and kl in ALLOWED_K_LATENT):
            return False, f"substep{i+1}_bad_k_latent:{kl}"
        if not isinstance(is_crit, bool):
            return False, f"substep{i+1}_bad_is_critical:{is_crit}"

        # k_latent 与 cot_text 中实际 <|pause|> 占位数对齐 (落盘时即为 <|pause|>)
        actual_k = len(_PAUSE_TOK_RE.findall(bot_blocks[i]))
        if actual_k != kl:
            return False, f"substep{i+1}_k_pause_mismatch:{actual_k}_vs_{kl}"

        # anchor_phrase 必须与 cot_text 中第 i 段 <|/latent|> 之后的窗口高度匹配 (允许 LLM 微调措辞)
        seg = anchor_segments[i]
        if not _phrase_matches_segment(anchor, seg):
            return False, f"substep{i+1}_anchor_not_in_cot"
        # leak_check_phrase 也必须出现在 anchor 段之中 (它是 anchor 中"删 latent 后悬空"的子串)
        if not _phrase_matches_segment(leak_chk, seg, jaccard_thr=0.6, allow_substring=True):
            return False, f"substep{i+1}_leak_check_phrase_not_in_anchor_seg"

        # ---- 留白契约: anchor_phrase 必须以"指代/选择/因果"开场 ----
        if not _has_blank_opener(anchor):
            return False, f"substep{i+1}_anchor_missing_blank_opener"

        # ---- anchor 不得含 hedge ----
        anc_l = anchor.lower()
        for hedge in FORBIDDEN_ANCHOR_HEDGES:
            if hedge in anc_l:
                return False, f"substep{i+1}_anchor_contains_hedge:{hedge.strip()!r}"

        # ---- anchor 不得含精确数值 (qualitative-only) ----
        nv_hit = _has_numeric_violation(anchor)
        if nv_hit:
            return False, f"substep{i+1}_anchor_numeric_violation:{nv_hit!r}"

        # latent_hint <= 8 词
        if _word_count(latent_hint) > 8:
            return False, f"substep{i+1}_latent_hint_too_long"

        seen_anchors.append(anchor)

    # ---- top-level latent_necessity / non_verbal_signal 必填 ----
    top_ln = sample.get("latent_necessity")
    top_nvs = sample.get("non_verbal_signal")
    if not (isinstance(top_ln, int) and 0 <= top_ln <= 3):
        return False, f"top_bad_latent_necessity:{top_ln}"
    if not (isinstance(top_nvs, str) and top_nvs.strip()):
        return False, "top_empty_non_verbal_signal"

    # ---- 留白快速 self-check (静态版): 把 cot_text 中所有 <|latent|>...<|/latent|> 删掉,
    #      检查是否仍出现任意 leak_check_phrase. 若仍出现 -> 留白可能失效.
    #      注意这里只是"必要条件": leak_check_phrase 本身就应在 anchor 段, 删 latent 不会
    #      让它消失. 我们要检查的是 leak_check_phrase 在 *删了 latent 块后的全文* 中
    #      仍是悬空的 (即上文里没有定义其所指). 这一项交给后续 leak-test 脚本; schema 这里
    #      只做"leak_check_phrase 不能等于 cot 全文剥 latent 后任意一个完整子句" 的弱检查.
    #      为简单起见, 仅断言 leak_check_phrase 不为空且出现在原 cot 中.
    if any(lk not in inner_cot for lk in
           [(s.get("leak_check_phrase") or "").strip() for s in substeps]):
        return False, "leak_check_phrase_missing_in_cot"

    return True, "ok"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _phrase_matches_segment(phrase: str, segment: str,
                            jaccard_thr: float = 0.7,
                            allow_substring: bool = True) -> bool:
    """判定 phrase 是否"出现在"segment 中:
      1) 严格 substring (大小写不敏感, 空格归一) -> True
      2) token-set Jaccard >= jaccard_thr -> True
    """
    if not phrase or not segment:
        return False
    np_p = _normalize(phrase)
    np_s = _normalize(segment)
    if allow_substring and np_p in np_s:
        return True
    # 标点剥离后再试一次
    def _strip_punct(x: str) -> str:
        return re.sub(r"[^a-zA-Z0-9 \u03b8]+", " ", x).strip()
    sp_p = _strip_punct(np_p)
    sp_s = _strip_punct(np_s)
    if allow_substring and sp_p and sp_p in sp_s:
        return True
    # token-set Jaccard
    tok_p = set(re.findall(r"[a-zA-Z0-9\u03b8]+", sp_p))
    tok_s = set(re.findall(r"[a-zA-Z0-9\u03b8]+", sp_s))
    if not tok_p or not tok_s:
        return False
    j = len(tok_p & tok_s) / len(tok_p | tok_s)
    return j >= jaccard_thr


def _answer_consistent(top_ans: str, inner_ans: str) -> bool:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()
    a, b = norm(top_ans), norm(inner_ans)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        if len(short) / max(len(long), 1) >= 0.3:
            return True
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return False
    j = len(sa & sb) / len(sa | sb)
    return j >= 0.5


# ============================================================
# 渲染：assistant 文本 = sample["cot_text"] (本身已含 <think>...</answer>)
# ============================================================

def render_assistant_text(sample: Dict) -> str:
    cot = (sample.get("cot_text") or "").strip()
    return cot


# ============================================================
# Normalization (防御性: 剔除 LLM 偶发幻觉字段, 不破坏合规)
# ============================================================

def normalize_sample_inplace(sample: Dict) -> Dict[str, Any]:
    removed_top: Counter = Counter()
    removed_sub: Counter = Counter()
    if not isinstance(sample, dict):
        return {"removed_top_fields": {}, "removed_substep_fields": {}}
    # 顶层只剔除明显幻觉 (我们不删非 KNOWN, 因为有 src_task_type 等 wrapper 注入)
    # 这里不主动删顶层; 仅记录
    for k in list(sample.keys()):
        if k not in KNOWN_TOP_FIELDS:
            removed_top[k] += 1
    substeps = sample.get("substeps")
    if isinstance(substeps, list):
        for st in substeps:
            if not isinstance(st, dict):
                continue
            bad = [k for k in list(st.keys()) if k not in KNOWN_SUBSTEP_FIELDS]
            for k in bad:
                removed_sub[k] += 1
                del st[k]
    return {
        "removed_top_fields": dict(removed_top),
        "removed_substep_fields": dict(removed_sub),
    }


# ============================================================
# 软质量探查 + 统计
# ============================================================

def inspect_v6_sample(sample: Dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "latent_necessity_top": -1,
        "non_verbal_signal_top": "",
        "n_substeps": 0,
        "n_critical_substeps": 0,
        "k_latent_total": 0,
        "task_value_tier": "low",
        "low_quality": False,
    }
    if not isinstance(sample, dict):
        return out
    ln = sample.get("latent_necessity")
    if isinstance(ln, int) and 0 <= ln <= 3:
        out["latent_necessity_top"] = ln
        out["low_quality"] = ln < 2
    nvs = sample.get("non_verbal_signal")
    if isinstance(nvs, str):
        out["non_verbal_signal_top"] = nvs.strip()
    subs = sample.get("substeps") or []
    if isinstance(subs, list):
        out["n_substeps"] = len(subs)
        out["n_critical_substeps"] = sum(
            1 for s in subs if isinstance(s, dict) and s.get("is_critical"))
        out["k_latent_total"] = sum(
            int(s.get("k_latent") or 0) for s in subs if isinstance(s, dict))
    tt = (sample.get("task_type") or "").strip()
    if tt in HIGH_VALUE_TASK_TYPES:
        out["task_value_tier"] = "high"
    elif tt in MEDIUM_VALUE_TASK_TYPES:
        out["task_value_tier"] = "medium"
    else:
        out["task_value_tier"] = "low"
    return out


def stats_summary(samples: List[Dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n": len(samples),
        "task_type": Counter(),
        "difficulty_hops": Counter(),
        "n_substeps": Counter(),
        "k_latent_total": Counter(),
        "latent_necessity_top": Counter(),
        "task_value_tier": Counter(),
        "non_verbal_signal_top": Counter(),
        "low_quality_count": 0,
        "n_critical_substeps_sum": 0,
        "k_latent_total_sum": 0,
    }
    for s in samples:
        out["task_type"][s.get("task_type", "?")] += 1
        out["difficulty_hops"][int(s.get("difficulty_hops", 0))] += 1
        info = inspect_v6_sample(s)
        out["n_substeps"][info["n_substeps"]] += 1
        out["k_latent_total"][info["k_latent_total"]] += 1
        out["latent_necessity_top"][info["latent_necessity_top"]] += 1
        out["task_value_tier"][info["task_value_tier"]] += 1
        if info["low_quality"]:
            out["low_quality_count"] += 1
        out["n_critical_substeps_sum"] += info["n_critical_substeps"]
        out["k_latent_total_sum"] += info["k_latent_total"]
        nvs = info["non_verbal_signal_top"]
        if nvs:
            out["non_verbal_signal_top"][nvs] += 1
    return {k: (dict(v) if hasattr(v, "items") else v) for k, v in out.items()}


def get_data_version() -> str:
    return DATA_VERSION


# ============================================================
# 留白 leak-test (静态版, 不依赖外部 LLM)
# ============================================================

def static_leak_check(sample: Dict) -> Dict[str, Any]:
    """对单条样本做"删 latent 静态留白检查":
      - 把 inner_cot 中所有 <|latent|>...<|/latent|> 删掉 -> 得到 stripped.
      - 对每个 leak_check_phrase, 验证它仍然出现在 stripped (因为它本身在 anchor 段);
        但要求其 *上文* (即 stripped 中该 phrase 之前的内容) 中不能含有该 phrase 中的
        关键名词 (heuristic: 长度>=4 的实词). 若关键名词已在上文出现 -> 视为
        留白失败 (text-only 可推断).

    Returns:
        {
          "stripped_cot": str,
          "per_substep": [
            {"phrase": str, "ok": bool, "leaked_via": str|None}, ...
          ],
          "n_leaks": int,
        }

    注: 这是粗粒度 heuristic, 真正可靠的 leak-test 用一个 text-only LLM 续写 +
    ROUGE 度量, 留给 leak_test_v6.py.
    """
    out: Dict[str, Any] = {"stripped_cot": "", "per_substep": [], "n_leaks": 0}
    if not isinstance(sample, dict):
        return out
    cot = sample.get("cot_text") or ""
    m = _THINK_RE.match(cot.strip())
    if not m:
        return out
    inner = m.group(1)
    stripped = _strip_latent_blocks(inner)
    out["stripped_cot"] = stripped
    subs = sample.get("substeps") or []
    for i, st in enumerate(subs):
        if not isinstance(st, dict):
            continue
        phrase = (st.get("leak_check_phrase") or "").strip()
        if not phrase:
            out["per_substep"].append({"phrase": "", "ok": False,
                                        "leaked_via": "empty"})
            out["n_leaks"] += 1
            continue
        # 找 phrase 在 stripped 的位置
        idx = stripped.lower().find(phrase.lower())
        if idx < 0:
            out["per_substep"].append({"phrase": phrase, "ok": False,
                                        "leaked_via": "not_found_in_stripped"})
            out["n_leaks"] += 1
            continue
        prefix = stripped[:idx].lower()
        # 提取 phrase 中的"内容词" (长度>=4, 排除停用词)
        STOP = {"that", "this", "with", "those", "them", "their", "from",
                 "what", "when", "where", "which", "into", "onto", "upon",
                 "have", "been", "were", "will", "would", "could", "shall",
                 "than", "then", "such", "some", "more", "most", "less",
                 "much", "many", "very", "just", "also", "only", "even",
                 "they", "them", "your", "ours", "mine"}
        content_words = [
            w for w in re.findall(r"[a-zA-Z][a-zA-Z\-']{3,}", phrase.lower())
            if w not in STOP
        ]
        leaked_via = None
        # 关键名词若在上文出现且其搭配语境与 phrase 上文同义 -> 视为泄露 (粗粒度)
        for w in content_words:
            if re.search(rf"\b{re.escape(w)}\b", prefix):
                leaked_via = w
                break
        ok = leaked_via is None
        out["per_substep"].append({"phrase": phrase, "ok": ok,
                                    "leaked_via": leaked_via})
        if not ok:
            out["n_leaks"] += 1
    return out


# ============================================================
# LASER key-token 抽取 (v6 -> 训练侧契约的桥梁)
# ============================================================

# 抽 key tokens 时要排除的停用词 / 留白契约 opener / 模糊代词 / 过于通用的连接词.
# 它们出现在 anchor_phrase 但不携带视觉语义, 不该作为 latent 蒸馏目标.
_KW_STOP_WORDS = {
    # 留白 opener
    "using", "with", "under", "given", "applying", "whatever", "within",
    "among", "out", "following", "mapped", "projected", "relative",
    "call", "once", "from", "for",
    "so", "therefore", "thus", "hence", "since", "because",
    "consequently", "only", "that", "this", "those", "these",
    "it", "its", "them", "their",
    # v6.5b fix: BLANK opener -ing/-ed forms (grammar connectors, not key_tokens)
    "calling", "combining", "adding", "reading", "resolving",
    "noting", "placing", "taking", "looking", "checking",
    "finding", "turning", "moving", "fixing", "setting",
    "after", "before", "having", "making",
    # 通用连接 / 副词
    "the", "a", "an", "of", "in", "on", "at", "to", "by", "is",
    "are", "was", "were", "be", "been", "being",
    "and", "but", "or", "nor", "if", "than", "then", "as",
    "also", "very", "just", "even", "still", "now", "yet",
    "would", "could", "should", "may", "might", "can",
    "have", "has", "had", "do", "does", "did", "will", "shall",
    # 模糊量词
    "some", "any", "all", "most", "more", "less", "few", "many",
    "much", "such", "each", "every",
    # 太通用的描述
    "thing", "things", "object", "objects", "side", "sides",
    "part", "parts", "place", "places", "way", "ways",
    "one", "two", "three", "first", "second", "third",
    "him", "her", "his", "he", "she", "they", "you", "your",
    "what", "when", "where", "which", "who", "whom", "whose", "how",
    # 留白契约里"剧情常客", 防止过度对齐到指代:
    "candidate", "option", "both", "either", "neither",
}


# ============================================================
# v6.4 waypoint: helper for snake_case 拆词
# ============================================================

_NUMERAL_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "single", "pair", "few", "several",
}

# v6.5 S-1: abstract pool 动词污染过滤.
# latent_hint/intent 常以动词开头 ("estimate body yaw" / "find ground contacts"),
# 这些起首动词不是真正的语义中心 — 它们是"指令性动词", 几乎不携带视觉/几何信息.
# 把它们从 abstract pool 过滤掉, 让 hidden 学到的是 "what to look at" 而不是
# "what action to perform". 注意: 这个黑名单**仅作用于 abstract pool**, 不会影响
# concrete pool (anchor 中的 sits/maps/lands 等是落点动词, 携带强语义).
_ABSTRACT_VERB_BLACKLIST = {
    # locate / find 类
    "find", "locate", "place", "fix", "pin", "spot", "anchor", "mark",
    # compare / verify 类
    "compare", "verify", "check", "test", "match", "confirm", "validate",
    # estimate / determine 类
    "estimate", "determine", "compute", "calculate", "evaluate", "measure",
    # decide / resolve 类
    "decide", "resolve", "settle", "judge", "rule", "pick", "choose", "select",
    # infer / deduce 类
    "infer", "deduce", "derive", "conclude", "reason",
    # translate / map / transform 类
    "translate", "map", "convert", "transform", "project", "shift",
    # combine / integrate 类
    "combine", "integrate", "merge", "fuse", "synthesize",
    # ask / look / read 类 (元认知动词)
    "ask", "look", "read", "see", "view", "observe", "inspect", "examine",
    # general action 类
    "use", "apply", "do", "make", "get", "set", "put", "give", "take",
    # need / want 类
    "need", "want", "require", "must",
}


def _filter_abstract_verbs(words: List[str]) -> List[str]:
    """v6.5 S-1: 把 abstract pool 中的指令性动词过滤掉.

    输入是 _content_words_from / _split_snake 已经处理过的内容词列表;
    本函数只负责删除黑名单动词, 不改变其他 token.
    """
    if not words:
        return words
    return [w for w in words if w.lower() not in _ABSTRACT_VERB_BLACKLIST]

def _split_snake(s: str) -> List[str]:
    """v6.4: 把 non_verbal_signal 这种 snake_case / dashed 字符串拆成 content words.

    例: 'shoulder_line_alignment' -> ['shoulder', 'line', 'alignment']
        'over-the-shoulder_framing' -> ['over', 'shoulder', 'framing']  (停用词 'the' 被滤掉)
    """
    if not s:
        return []
    parts = re.split(r"[_\s\-]+", s)
    out = []
    for p in parts:
        pl = p.lower()
        if pl in _KW_STOP_WORDS:
            continue
        if len(pl) < 3:
            continue
        out.append(p)
    return out


def _is_numeral(w: str) -> bool:
    """v6.4: 识别字面数词 (concrete pool 排除它们, 因为它们对 latent 想象方向无意义)."""
    if w.lower() in _NUMERAL_WORDS:
        return True
    if any(c.isdigit() for c in w):
        return True
    return False


def _content_words_from(text: str, head_only: bool = False) -> List[str]:
    """v6.4: 标准内容词抽取 (字母 >=3, 去停用词)."""
    if not text:
        return []
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text)
    if head_only:
        words = words[:14]
    out = []
    for w in words:
        wl = w.lower()
        if wl in _KW_STOP_WORDS:
            continue
        if len(wl) < 3:
            continue
        out.append(w)
    return out


def _dedup_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for w in seq:
        wl = w.lower()
        if wl in seen:
            continue
        seen.add(wl)
        out.append(w)
    return out


def extract_key_tokens_from_anchor(
    anchor_phrase: str,
    intent: str = "",
    latent_hint: str = "",
    non_verbal_signal: str = "",
    max_tokens: int = 6,
) -> List[str]:
    """v6.4 waypoint: 从 substep 元数据中抽取一组 key tokens 作 LASER 蒸馏 anchor.

    用途: LASER DWAL 自蒸馏 (rld/latent_thinker._compute_stage_keyword_distill_loss)
          会把这些 token id 当成 hard target, 拉近 latent hidden 的 logits.

    v6.4 关键变更 (相比 v6.3):
      - 候选词来源**优先级翻转**:
          v6.3: anchor_phrase (前14) > intent > latent_hint  (= anchor 字面词主导)
          v6.4: latent_hint > non_verbal_signal (snake_case 拆) > intent > anchor (兜底)
      - 含义: latent 应该"想象什么" (路标) 比 "想完后写什么" (anchor 字面词) 更重要.
        v6.3 经常给出 [four, five, torsos] 这种数字+名词, hidden 学的是 token 表面;
        v6.4 给出 [fix, group, frontal, line], hidden 学的是"该往哪个语义方向想".

    抽取规则:
      1. abstract pool = latent_hint 内容词 + non_verbal_signal 拆词 + intent 内容词
      2. anchor 内容词 (排除字面数词) 仅作 fallback 补齐.
      3. 保序去重, 取前 max_tokens 个.

    Args:
        non_verbal_signal: snake_case 字符串 (例 'body_axis_yaw'), v6.4 新增形参.
            旧调用方未传此参数时退化为只用 anchor + intent + latent_hint, 但优先级仍翻转.
    """
    abstract = (
        _content_words_from(latent_hint)
        + _split_snake(non_verbal_signal)
        + _content_words_from(intent)
    )
    # v6.5 S-1: 同 build_latent_key_tokens, 过滤指令性动词
    abstract = _filter_abstract_verbs(abstract)
    fallback = _content_words_from(anchor_phrase, head_only=True)
    candidates = abstract + fallback
    uniq = _dedup_preserve(candidates)
    return uniq[:max_tokens]


def _pick_pool_tokens(pool: List[str],
                      role: str,
                      preferred: int = PREFERRED_KEY_TOKENS_PER_STAGE,
                      min_n: int = MIN_KEY_TOKENS_PER_STAGE,
                      max_n: int = MAX_KEY_TOKENS_PER_STAGE) -> List[str]:
    """v6.5 soft K_s: pick tokens from pool with role-aware preference.

    Rules:
      - abstract role: prefer min_n..preferred (semantically sparse, e.g. "fix axis")
      - concrete role: prefer preferred..max_n (semantically denser, e.g.
                       "torsos sit roughly flush against L")
      - unified  role: prefer exactly preferred (middle ground)
    Falls back to whatever the pool has, clipped to [min_n, max_n].
    If pool is too small (<min_n), returns the whole pool (caller handles fallback).
    """
    if not pool:
        return []
    n_avail = len(pool)
    if role == "abstract":
        target = min(preferred, max(min_n, n_avail))
    elif role == "concrete":
        target = min(max_n, max(preferred, n_avail))
    else:  # unified or unknown
        target = min(preferred, n_avail)
    target = max(min_n, min(max_n, target))
    target = min(target, n_avail)
    return pool[:target]


def build_latent_key_tokens(sample: Dict,
                            tokens_per_stage: int = PREFERRED_KEY_TOKENS_PER_STAGE
                            ) -> List[List[Dict[str, Any]]]:
    """v6.5 waypoint: per-boundary nested stages with role-aware soft K_s.

    Returns [[stage_dict, ...], ...] aligned with rld/data.py per-boundary contract.

    v6.5 changes vs v6.4:
      - k_latent now supports {1, 2, 3}.
      - K_s (tokens per stage) is no longer hard-fixed at 4: it floats in
        [MIN_KEY_TOKENS_PER_STAGE, MAX_KEY_TOKENS_PER_STAGE] = [3, 5] following
        each stage's natural semantic length, with role-aware preference.

    Stage roles:
      - k=1 -> single stage, role='unified' (abstract + concrete fused).
      - k=2 -> stage_1 role='abstract', stage_2 role='concrete'.
      - k=3 -> stage_1 role='abstract', stage_2 role='bridge',
               stage_3 role='concrete'.  bridge interleaves the two pools to
               form an intermediate semantic checkpoint.

    Args:
        tokens_per_stage: legacy positional arg, kept for backward compat.
            v6.5 uses it only as the "preferred" target; min/max are enforced
            via MIN_KEY_TOKENS_PER_STAGE / MAX_KEY_TOKENS_PER_STAGE constants.
    """
    out: List[List[Dict[str, Any]]] = []
    subs = sample.get("substeps") or []
    for i, st in enumerate(subs):
        if not isinstance(st, dict):
            continue
        kl = int(st.get("k_latent") or MIN_K_LATENT)
        kl = max(MIN_K_LATENT, min(MAX_K_LATENT, kl))
        intent = st.get("intent") or ""
        nvs = st.get("non_verbal_signal") or ""
        hint = st.get("latent_hint") or ""
        anchor = st.get("anchor_phrase") or ""

        # ---- abstract pool: "what to imagine" (latent_hint + nvs + intent) ----
        abstract_pool = _dedup_preserve(
            _content_words_from(hint)
            + _split_snake(nvs)
            + _content_words_from(intent)
        )
        # v6.5 S-1: 过滤指令性动词 (find/locate/compare/...) — 它们不是语义中心.
        # 仅作用于 abstract pool; concrete pool 不动 (anchor 中的动词是落点信息).
        abstract_pool = _filter_abstract_verbs(abstract_pool)
        # ---- concrete pool: "where it lands" (anchor content words, no numerals) ----
        concrete_pool = _dedup_preserve(
            [w for w in _content_words_from(anchor, head_only=True)
             if not _is_numeral(w)]
        )

        sub_id = st.get("substep_id", i + 1)
        preferred = tokens_per_stage

        if kl == 1:
            # unified single stage = mix of abstract + concrete (semantic dense)
            half = max(1, preferred // 2)
            tokens = _dedup_preserve(
                abstract_pool[:half] + concrete_pool[:preferred - half]
            )
            tokens = _pick_pool_tokens(tokens, "unified",
                                       preferred=preferred)
            if not tokens:
                tokens = (abstract_pool[:preferred]
                          or concrete_pool[:preferred]
                          or ["unknown"])
                tokens = tokens[:MAX_KEY_TOKENS_PER_STAGE]
            stages = [{
                "tokens": list(tokens),
                "stage_id": 1,
                "role": "unified",
                "from_substep": sub_id,
            }]
        elif kl == 2:
            # k=2: stage_1 = abstract waypoint, stage_2 = concrete landing
            s1 = _pick_pool_tokens(abstract_pool, "abstract",
                                   preferred=preferred)
            s2 = _pick_pool_tokens(concrete_pool, "concrete",
                                   preferred=preferred)
            if not s1:
                s1 = (concrete_pool[:preferred] or ["unknown"])[:MAX_KEY_TOKENS_PER_STAGE]
            if not s2:
                s2 = (abstract_pool[:preferred] or ["unknown"])[:MAX_KEY_TOKENS_PER_STAGE]
            stages = [
                {"tokens": list(s1), "stage_id": 1,
                 "role": "abstract", "from_substep": sub_id},
                {"tokens": list(s2), "stage_id": 2,
                 "role": "concrete", "from_substep": sub_id},
            ]
        else:  # kl == 3
            # k=3: stage_1 abstract -> stage_2 bridge -> stage_3 concrete
            s1 = _pick_pool_tokens(abstract_pool, "abstract",
                                   preferred=preferred)
            s3 = _pick_pool_tokens(concrete_pool, "concrete",
                                   preferred=preferred)
            # bridge: interleave the two pools (later half of abstract +
            #         earlier half of concrete) to form a smooth midpoint
            half_a = max(1, preferred // 2)
            half_c = preferred - half_a
            bridge_seed = _dedup_preserve(
                abstract_pool[half_a:half_a + half_a] +
                concrete_pool[:half_c]
            )
            if len(bridge_seed) < MIN_KEY_TOKENS_PER_STAGE:
                # fallback: concat both pools' tail/head to fill bridge
                bridge_seed = _dedup_preserve(
                    abstract_pool[-2:] + concrete_pool[:3]
                )
            s2 = _pick_pool_tokens(bridge_seed, "unified",
                                   preferred=preferred)
            # robust fallbacks
            if not s1:
                s1 = (concrete_pool[:preferred] or ["unknown"])[:MAX_KEY_TOKENS_PER_STAGE]
            if not s3:
                s3 = (abstract_pool[:preferred] or ["unknown"])[:MAX_KEY_TOKENS_PER_STAGE]
            if not s2:
                s2 = (s1[-2:] + s3[:2]) or ["unknown"]
                s2 = s2[:MAX_KEY_TOKENS_PER_STAGE]
            stages = [
                {"tokens": list(s1), "stage_id": 1,
                 "role": "abstract", "from_substep": sub_id},
                {"tokens": list(s2), "stage_id": 2,
                 "role": "bridge",   "from_substep": sub_id},
                {"tokens": list(s3), "stage_id": 3,
                 "role": "concrete", "from_substep": sub_id},
            ]
        out.append(stages)
    return out

# ============================================================
# v6 -> 训练侧契约 (rld/data.py 直接消费)
# ============================================================

def cot_to_reasoning_for_training(cot_text: str) -> str:
    """把 v6 cot_text 转换为训练侧 reasoning_for_training:
       - 去掉 <think>...</think> 和 <answer>...</answer> 包装 (训练侧自有 chat template);
       - latent 段语义保持: 每个 <|latent|>...<|/latent|> 段折叠为单个 <|pause|>.

    设计动机 (v6.2 per-boundary):
      - 数据里 substep 间是独立 latent 触发点 (N 个 substep -> N 个 <|pause|>);
      - 序列里每个 boundary 只占 1 个 <|pause|> (NLDDataset 会替换为 <|latent|>
        并在后面补 <|/latent|>);
      - 该 boundary 真正的 latent 演化步数由 latent_key_tokens[boundary] 的 stage
        数 (= k_latent) 在训练侧动态控制, 不在序列里物理占位 K 次.

    这样既保持序列长度紧凑, 又保留了"N 次独立浅思考"的语义.
    """
    if not cot_text:
        return ""
    m = _THINK_RE.match(cot_text.strip())
    inner = m.group(1).strip() if m else cot_text.strip()
    # 把每个 <|latent|>...<|/latent|> 段折叠为单个 <|pause|>
    out = re.sub(r"<\|latent\|>.*?<\|/latent\|>", PAUSE_TOKEN,
                 inner, flags=re.DOTALL)
    # 清理多余空白
    out = re.sub(r"\s+", " ", out).strip()
    return out


def to_training_record(sample: Dict,
                       tokens_per_stage: int = 4,
                       keep_meta: bool = False) -> Dict[str, Any]:
    """v6 sample -> rld/data.py v6.2 直接消费的训练记录 (slim 版).

    训练侧 rld/data.py 实际只读取 6 个字段:
        - image_path (或 image, 二选一)
        - question
        - answer
        - reasoning_for_training : 含 N 个 <|pause|> 的连续 CoT
        - latent_key_tokens      : per-boundary 嵌套 List[List[Dict]]
        - task_type              : 仅用于分布日志, 缺省 fallback 'unknown'

    其余 v6 schema 字段 (sample_id / data_version / non_verbal_signal /
    latent_necessity / n_substeps / src_task_type ...) 训练时全部不读, 因此
    默认从训练记录中剔除以减小 IO 与内存占用.

    参数:
        sample           : v6 内部 schema 单条样本 (含 substeps / cot_text)
        tokens_per_stage : 每个 stage 抽多少个 key tokens (默认 2, 与 v6.3 k_latent 上限对齐)
        keep_meta        : True 时额外保留 sample_id / n_substeps / latent_necessity /
                           num_stages / n_boundaries / data_version, 便于审计追溯
                           (训练时仍不会被使用, 只是更易做后过滤/抽样).

    返回:
        slim dict, 仅含训练所需字段 (+ 可选元信息).
    """
    img_paths = sample.get("image_paths") or []
    img = img_paths[0] if img_paths else (sample.get("image_path") or "")
    cot = sample.get("cot_text") or ""
    reasoning = cot_to_reasoning_for_training(cot)
    boundaries = build_latent_key_tokens(sample, tokens_per_stage=tokens_per_stage)

    # ---- 训练必需 6 个字段 ----
    rec: Dict[str, Any] = {
        "image_path":             img,
        "question":               sample.get("question", ""),
        "answer":                 sample.get("answer", ""),
        "reasoning_for_training": reasoning,
        "latent_key_tokens":      boundaries,
        "task_type":              sample.get("task_type", "unknown"),
    }

    # ---- 可选审计元信息 ----
    if keep_meta:
        rec["sample_id"]        = sample.get("sample_id")
        rec["data_version"]     = sample.get("data_version", DATA_VERSION)
        rec["n_substeps"]       = sample.get("n_substeps", -1)
        rec["latent_necessity"] = sample.get("latent_necessity", -1)
        rec["n_boundaries"]     = len(boundaries)
        rec["num_stages"]       = sum(len(b) for b in boundaries)

    return rec
