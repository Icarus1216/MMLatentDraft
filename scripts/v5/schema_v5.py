"""
schema_v5.py
============
v5 agentic-trajectory 数据 schema 定义 + 强校验。

数据形式：单轮 QA, assistant 一次性输出整条 trajectory。
本模块同时提供:
  - validate_v5_sample(): 程序化校验 generator 输出 dict
  - render_assistant_text(): 把结构化 trajectory 渲染成 <step>…</step> 串行文本
                              (供训练数据序列化使用)
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

from action_space import (
    validate_action,
    action_allowed_in_phase,
    ALLOWED_ACTION_TYPES,
    PHASE_ACTIONS,
)


# ============================================================
# 常量
# ============================================================

DATA_VERSION = "v5.agentic_traj.1.0"

# v5.1: 扩展 task_type, 倾向 latent-necessity 高的类型
ALLOWED_TASK_TYPES = {
    # HIGH-VALUE (latent reasoning 真正需要的)
    "perspective",
    "geometry",
    "fine_grained",
    "counterfactual",
    "multi_hypothesis",
    # MEDIUM-VALUE
    "physical",
    "comparison",
    # LOW-VALUE (兼容旧 seed; 配额由 generate_v5.py 控制)
    "spatial_relation",
    "counting",
    "detail_observation",
    "temporal",
    "ocr",
}

# task_type 价值分层 (用于上层采样配额 + 报表)
HIGH_VALUE_TASK_TYPES = {
    "perspective", "geometry", "fine_grained",
    "counterfactual", "multi_hypothesis",
}
MEDIUM_VALUE_TASK_TYPES = {"physical", "comparison"}
LOW_VALUE_TASK_TYPES = {
    "spatial_relation", "counting",
    "detail_observation", "temporal", "ocr",
}

MIN_STEPS = 4
MAX_STEPS = 8
MIN_MIDDLE_PRIVATE_STEPS = 2
MAX_QUESTION_WORDS = 35   # v5.1: 30 -> 35, 让多跳问句更易合规
MAX_ANSWER_WORDS = 25
MIN_HOPS = 3
MAX_HOPS = 7              # v5.1: 6 -> 7, 容纳极密集多跳

# v5.1: latent_necessity 评分相关
MIN_LATENT_NECESSITY = 0  # 强制下限 (0 不拒, 仅打报表)
WARN_LATENT_NECESSITY_BELOW = 2  # 低于此值视为 "低质量", 计入 quality 报表

# 已知顶层字段 (其他都视为 unexpected, 仅记录不拒)
KNOWN_TOP_FIELDS = {
    "sample_id", "data_source", "data_version",
    "image_paths", "task_type", "difficulty_hops",
    "question", "answer", "trajectory_steps",
    "origin", "original_sample_ref", "src_task_type",
    "messages",
    # v5.1 新增
    "latent_necessity", "non_verbal_signal",
}
KNOWN_STEP_FIELDS = {
    "step_id", "phase", "thought",
    "latent_hint", "action", "observation",
}

# v5.1: observation 中禁止的犹豫词 (出现即拒)
FORBIDDEN_OBS_HEDGES = (
    "wait ", "wait,", "wait.",
    "actually,", "actually ",
    "on second thought",
    "i am not sure", "i'm not sure",
    "let me reconsider", "hmm,", "hmm ",
)


def _word_count(s: str) -> int:
    return len(re.findall(r"\S+", s or ""))


# ============================================================
# 校验
# ============================================================

def validate_v5_sample(sample: Dict) -> Tuple[bool, str]:
    """对 generator 产出的 raw dict 做 hard-constraint 校验。

    Returns:
        (ok, reason)。ok=False 时 reason 描述违规字段。
    """
    if not isinstance(sample, dict):
        return False, "sample_not_dict"

    # ---- 必填顶层字段 ----
    q = (sample.get("question") or "").strip()
    a = (sample.get("answer") or "").strip()
    tt = (sample.get("task_type") or "").strip()
    hops = sample.get("difficulty_hops")
    steps = sample.get("trajectory_steps")
    if not q:
        return False, "empty_question"
    if not a:
        return False, "empty_answer"
    if tt not in ALLOWED_TASK_TYPES:
        return False, f"bad_task_type:{tt}"
    if not (isinstance(hops, int) and MIN_HOPS <= hops <= MAX_HOPS):
        return False, f"bad_difficulty_hops:{hops}"
    if not (isinstance(steps, list) and steps):
        return False, "missing_trajectory_steps"

    # ---- 字段长度 ----
    if _word_count(q) > MAX_QUESTION_WORDS:
        return False, "question_too_long"
    if _word_count(a) > MAX_ANSWER_WORDS:
        return False, "answer_too_long"

    # ---- 步数 ----
    n = len(steps)
    if not (MIN_STEPS <= n <= MAX_STEPS):
        return False, f"bad_step_count:{n}"

    # ---- 步骤逐条校验 ----
    seen_obs = []
    for i, st in enumerate(steps):
        if not isinstance(st, dict):
            return False, f"step{i+1}_not_dict"
        sid = st.get("step_id")
        if sid != i + 1:
            return False, f"step{i+1}_bad_id:{sid}"

        phase = st.get("phase", "")
        if phase not in PHASE_ACTIONS:
            return False, f"step{i+1}_bad_phase:{phase}"

        # 首末步 phase 强约束
        if i == 0 and phase != "global_glance":
            return False, f"step1_must_be_global_glance:{phase}"
        if i == n - 1 and phase != "aggregation":
            return False, f"stepN_must_be_aggregation:{phase}"
        if 0 < i < n - 1 and phase != "focused_inspection":
            return False, f"step{i+1}_middle_must_be_focused_inspection:{phase}"

        # 文本字段
        thought = (st.get("thought") or "").strip()
        latent_hint = (st.get("latent_hint") or "").strip()
        if not thought:
            return False, f"step{i+1}_empty_thought"
        if not latent_hint:
            return False, f"step{i+1}_empty_latent_hint"

        # action
        ok, reason = validate_action(st.get("action", {}))
        if not ok:
            return False, f"step{i+1}_action_invalid:{reason}"
        atype = st["action"]["type"]
        if not action_allowed_in_phase(atype, phase):
            return False, f"step{i+1}_action_phase_mismatch:{atype}/{phase}"

        # observation
        obs = st.get("observation")
        if i == n - 1:
            if obs not in (None, "", "null"):
                return False, f"stepN_observation_must_be_null"
        else:
            if not (isinstance(obs, str) and obs.strip()):
                return False, f"step{i+1}_empty_observation"
            obs_l = obs.strip().lower()
            for hedge in FORBIDDEN_OBS_HEDGES:
                if hedge in obs_l:
                    return False, f"step{i+1}_obs_contains_hedge:{hedge.strip()}"
            seen_obs.append(obs.strip())

        # 中间步骤的 latent_hint 长度检查
        if 0 < i < n - 1 and _word_count(latent_hint) > 8:
            return False, f"step{i+1}_latent_hint_too_long"

    # ---- 中间私有证据步数 ----
    n_middle = n - 2
    if n_middle < MIN_MIDDLE_PRIVATE_STEPS:
        return False, f"too_few_middle_steps:{n_middle}"

    # ---- final_answer 与 top-level answer 一致性 (宽松匹配) ----
    last_action = steps[-1]["action"]
    fav = (last_action.get("args", {}).get("value") or "").strip()
    if not fav:
        return False, "final_answer_empty_value"
    # 宽松：subset 包含 / 长度比例 ≥ 0.6
    if not _answer_consistent(a, fav):
        return False, "final_answer_inconsistent_with_top_answer"

    # ---- 中间 observation 之间不应高度重复 ----
    for i in range(len(seen_obs)):
        for j in range(i + 1, len(seen_obs)):
            if _too_similar(seen_obs[i], seen_obs[j]):
                return False, f"obs_redundant:step{i+2}~step{j+2}"

    return True, "ok"


def inspect_v5_sample(sample: Dict) -> Dict[str, Any]:
    """v5.1: 软性质量探查 (即使 schema 通过也要打报表).

    返回:
        {
          "latent_necessity": int (0..3, 缺失=-1),
          "non_verbal_signal": str,
          "low_quality": bool,           # latent_necessity < WARN_LATENT_NECESSITY_BELOW
          "unexpected_top_fields": [..], # 未声明的顶层字段
          "unexpected_step_fields": [..],# 未声明的 step 字段
          "task_value_tier": "high|medium|low",
        }
    """
    out: Dict[str, Any] = {
        "latent_necessity": -1,
        "non_verbal_signal": "",
        "low_quality": False,
        "unexpected_top_fields": [],
        "unexpected_step_fields": [],
        "task_value_tier": "low",
    }
    if not isinstance(sample, dict):
        return out

    ln = sample.get("latent_necessity")
    if isinstance(ln, int) and 0 <= ln <= 3:
        out["latent_necessity"] = ln
        out["low_quality"] = ln < WARN_LATENT_NECESSITY_BELOW
    else:
        # 缺失视为低质量 (旧 generator 兼容)
        out["low_quality"] = True

    nvs = sample.get("non_verbal_signal")
    if isinstance(nvs, str):
        out["non_verbal_signal"] = nvs.strip()

    # 未知顶层字段
    for k in sample.keys():
        if k not in KNOWN_TOP_FIELDS:
            out["unexpected_top_fields"].append(k)

    # 未知 step 字段
    bad_step_fields = set()
    for st in (sample.get("trajectory_steps") or []):
        if isinstance(st, dict):
            for k in st.keys():
                if k not in KNOWN_STEP_FIELDS:
                    bad_step_fields.add(k)
    out["unexpected_step_fields"] = sorted(bad_step_fields)

    # task_type 价值分层
    tt = (sample.get("task_type") or "").strip()
    if tt in HIGH_VALUE_TASK_TYPES:
        out["task_value_tier"] = "high"
    elif tt in MEDIUM_VALUE_TASK_TYPES:
        out["task_value_tier"] = "medium"
    else:
        out["task_value_tier"] = "low"
    return out


def _answer_consistent(top_ans: str, final_ans: str) -> bool:
    """final_answer.value 与顶层 answer 的一致性判断 (宽松).

    规则：
      - 完全相等 (忽略大小写/标点) -> ok
      - 一方包含另一方 (>= 0.5 长度) -> ok
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()
    a, b = norm(top_ans), norm(final_ans)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        if len(short) / max(len(long), 1) >= 0.3:
            return True
    # 关键词重合度 (Jaccard) ≥ 0.5
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return False
    j = len(sa & sb) / len(sa | sb)
    return j >= 0.5


def _too_similar(s1: str, s2: str) -> bool:
    """两个 observation 是否过度相似 (token Jaccard >= 0.85)."""
    def toks(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", s.lower()))
    a, b = toks(s1), toks(s2)
    if not a or not b:
        return False
    j = len(a & b) / len(a | b)
    return j >= 0.85


# ============================================================
# 渲染：结构化 -> assistant 文本
# ============================================================

def render_assistant_text(sample: Dict, latent_token: str = "<|latent|>",
                          k_per_step: int = 4) -> str:
    """把结构化 trajectory 渲染成训练用的 assistant 输出字符串。

    格式 (与 generator_v5.txt 描述一致)：
        <think>
        <step id="1" phase="global_glance">
        <thought>...</thought>
        <latent>⟨LATENT_BEGIN⟩{latent_token}*K⟨LATENT_END⟩</latent>
        <action>{json}</action>
        <observation>...</observation>
        </step>
        ...
        </think>
        <answer>...</answer>
    """
    import json as _json
    parts = ["<think>"]
    for st in sample["trajectory_steps"]:
        sid = st["step_id"]
        phase = st["phase"]
        thought = st["thought"].strip()
        latent_hint = st["latent_hint"].strip()
        action_json = _json.dumps(st["action"], ensure_ascii=False)
        obs = st.get("observation")

        parts.append(f'<step id="{sid}" phase="{phase}">')
        parts.append(f'<thought>{thought}</thought>')
        parts.append(
            f'<latent hint="{latent_hint}">'
            f'⟨LATENT_BEGIN⟩{latent_token * k_per_step}⟨LATENT_END⟩'
            f'</latent>'
        )
        parts.append(f'<action>{action_json}</action>')
        if obs not in (None, "", "null"):
            parts.append(f'<observation>{obs.strip()}</observation>')
        parts.append('</step>')
    parts.append("</think>")
    parts.append(f"<answer>{sample['answer'].strip()}</answer>")
    return "\n".join(parts)


# ============================================================
# 统计
# ============================================================

def stats_summary(samples: List[Dict]) -> Dict[str, Any]:
    from collections import Counter
    out: Dict[str, Any] = {
        "n": len(samples),
        "task_type": Counter(),
        "difficulty_hops": Counter(),
        "step_count": Counter(),
        "action_type": Counter(),
        # v5.1
        "latent_necessity": Counter(),
        "task_value_tier": Counter(),
        "low_quality_count": 0,
        "unexpected_top_fields": Counter(),
        "unexpected_step_fields": Counter(),
        "non_verbal_signal_top": Counter(),
    }
    for s in samples:
        out["task_type"][s.get("task_type", "?")] += 1
        out["difficulty_hops"][int(s.get("difficulty_hops", 0))] += 1
        steps = s.get("trajectory_steps", [])
        out["step_count"][len(steps)] += 1
        for st in steps:
            atype = st.get("action", {}).get("type", "?")
            out["action_type"][atype] += 1
        # v5.1 软性质量
        info = inspect_v5_sample(s)
        out["latent_necessity"][info["latent_necessity"]] += 1
        out["task_value_tier"][info["task_value_tier"]] += 1
        if info["low_quality"]:
            out["low_quality_count"] += 1
        for k in info["unexpected_top_fields"]:
            out["unexpected_top_fields"][k] += 1
        for k in info["unexpected_step_fields"]:
            out["unexpected_step_fields"][k] += 1
        nvs = info["non_verbal_signal"]
        if nvs:
            out["non_verbal_signal_top"][nvs] += 1
    # convert Counter to dict for JSON
    return {k: (dict(v) if hasattr(v, "items") else v) for k, v in out.items()}


def get_data_version() -> str:
    return DATA_VERSION


# ============================================================
# Normalization (防御性: 剔除 LLM 偶发幻觉字段)
# ============================================================

def normalize_sample_inplace(sample: Dict) -> Dict[str, Any]:
    """对 generator 产出的 raw dict 做防御性清洗 (in-place):

    - 删除 trajectory_steps[*] 中不在 KNOWN_STEP_FIELDS 的字段
      (例如 step_id_check / observation_public / public_hint_redacted / latent_hint_note)
    - 当 zoom_in 缺 box_norm 时, 降级为 point_at(target=...) 避免整条 trajectory 被丢
    - 顶层 KNOWN_TOP_FIELDS 之外的字段保留 (上层 generate 还会覆盖一些 meta)

    返回:
        {"removed_step_fields": dict, "downgraded_zoom_in": int}
    """
    from collections import Counter
    removed: Counter = Counter()
    downgraded_zoom = 0
    if not isinstance(sample, dict):
        return {"removed_step_fields": {}, "downgraded_zoom_in": 0}
    steps = sample.get("trajectory_steps")
    if not isinstance(steps, list):
        return {"removed_step_fields": {}, "downgraded_zoom_in": 0}
    for st in steps:
        if not isinstance(st, dict):
            continue
        # (1) 剔除未声明字段
        bad = [k for k in list(st.keys()) if k not in KNOWN_STEP_FIELDS]
        for k in bad:
            removed[k] += 1
            del st[k]
        # (2) zoom_in 缺 box_norm -> 降级为 point_at
        action = st.get("action") or {}
        if isinstance(action, dict) and action.get("type") == "zoom_in":
            args = action.get("args") or {}
            box = args.get("box_norm")
            valid_box = (
                isinstance(box, (list, tuple))
                and len(box) == 4
                and all(isinstance(v, (int, float)) for v in box)
            )
            if not valid_box:
                tgt = args.get("target") if isinstance(args, dict) else None
                if not isinstance(tgt, str) or not tgt.strip():
                    tgt = "the relevant region"
                st["action"] = {
                    "type": "point_at",
                    "args": {"target": tgt.strip()},
                }
                downgraded_zoom += 1
    return {
        "removed_step_fields": dict(removed),
        "downgraded_zoom_in": downgraded_zoom,
    }
