"""
action_space.py
===============
v5 agentic-trajectory 动作空间定义 (closed set)。

每个动作：
  - name:        动作类型字符串
  - required:    args 必须包含的字段
  - validator:   一个 (args) -> (ok: bool, reason: str) 函数

v5 单轮 QA 形式但语义是 agentic：模型在 ONE generation 内串行输出
multiple <step> blocks，每个 step 含一个 action。
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Callable

# ============================================================
# 基础 validators
# ============================================================

def _is_box_norm(b: Any) -> bool:
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except Exception:
        return False
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return False
    if (x2 - x1) < 0.04 or (y2 - y1) < 0.04:
        return False
    return True


def _check_global_glance(args: Dict) -> Tuple[bool, str]:
    if not isinstance(args, dict):
        return False, "args_not_dict"
    return True, ""


def _check_zoom_in(args: Dict) -> Tuple[bool, str]:
    if not isinstance(args, dict):
        return False, "args_not_dict"
    if "box_norm" not in args:
        return False, "missing_box_norm"
    if not _is_box_norm(args["box_norm"]):
        return False, "bad_box_norm"
    target = args.get("target", "")
    if not (isinstance(target, str) and target.strip()):
        return False, "missing_target"
    return True, ""


def _check_point_at(args: Dict) -> Tuple[bool, str]:
    if not isinstance(args, dict):
        return False, "args_not_dict"
    target = args.get("target", "")
    if not (isinstance(target, str) and target.strip()):
        return False, "missing_target"
    return True, ""


def _check_count_objects(args: Dict) -> Tuple[bool, str]:
    if not isinstance(args, dict):
        return False, "args_not_dict"
    if "region_norm" not in args or not _is_box_norm(args["region_norm"]):
        return False, "bad_region_norm"
    pred = args.get("predicate", "")
    if not (isinstance(pred, str) and pred.strip()):
        return False, "missing_predicate"
    return True, ""


def _check_compare_attributes(args: Dict) -> Tuple[bool, str]:
    if not isinstance(args, dict):
        return False, "args_not_dict"
    for k in ("obj_a", "obj_b", "attribute"):
        v = args.get(k, "")
        if not (isinstance(v, str) and v.strip()):
            return False, f"missing_{k}"
    return True, ""


def _check_final_answer(args: Dict) -> Tuple[bool, str]:
    if not isinstance(args, dict):
        return False, "args_not_dict"
    val = args.get("value", "")
    if not (isinstance(val, str) and val.strip()):
        return False, "missing_value"
    return True, ""


# ============================================================
# Action registry
# ============================================================

ACTION_REGISTRY: Dict[str, Callable[[Dict], Tuple[bool, str]]] = {
    "global_glance":      _check_global_glance,
    "zoom_in":            _check_zoom_in,
    "point_at":           _check_point_at,
    "count_objects":      _check_count_objects,
    "compare_attributes": _check_compare_attributes,
    "final_answer":       _check_final_answer,
}

# 阶段 -> 允许的 action 集合
PHASE_ACTIONS = {
    "global_glance":      {"global_glance"},
    "focused_inspection": {"zoom_in", "point_at", "count_objects", "compare_attributes"},
    "aggregation":        {"final_answer"},
}

ALLOWED_ACTION_TYPES = set(ACTION_REGISTRY.keys())


def validate_action(act: Dict) -> Tuple[bool, str]:
    """单个 action dict 校验。"""
    if not isinstance(act, dict):
        return False, "action_not_dict"
    t = act.get("type", "")
    if t not in ACTION_REGISTRY:
        return False, f"unknown_action_type:{t}"
    args = act.get("args", {})
    return ACTION_REGISTRY[t](args)


def action_allowed_in_phase(action_type: str, phase: str) -> bool:
    return action_type in PHASE_ACTIONS.get(phase, set())
