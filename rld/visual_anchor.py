"""
Visual Anchor: path B2 视觉侧 hidden state 监督 (零可学参数, 纯函数式).

设计依据 (基于训练数据真实结构):
  - 每个 stage 自带 role ∈ {abstract, bridge, unified, concrete}
  - abstract: token 偏语义 (来自 intent / latent_hint), 应靠近语言锥
  - concrete: token 偏视觉 (像素位/视觉细节), 应靠近视觉锥
  - bridge / unified: 居中

实现要点:
  1. 视觉重心 v_bar: 用 stage key tokens 的 mean embedding 作 query, 在该样本视觉
     token 序列上做 cosine top-K 检索 (K 可配, 默认 6) 取均值, 零可学参数.
  2. 文本重心 t_bar: stage key tokens 的 LM input embedding 直接取均值.
  3. slerp 过渡参考目标 r: 由 role 决定的固定 alpha (abstract=0.75, bridge/unified=0.5,
     concrete=0.25), 在 v_bar / t_bar 之间做球面线性插值.
     约定: alpha=1 对应纯文本重心 (语言锥), alpha=0 对应纯视觉重心 (视觉锥).
  4. 双 anchor margin loss: ReLU(d(h,r) - d(h,v_bar) + δ) + ReLU(d(h,r) - d(h,t_bar) + δ)
     即要求 hidden 比 v_bar / t_bar 都更靠近 r, 落在两锥之间的过渡邻域.

调用现场:
  latent_thinker.forward 末尾 (与 SW-SRS / TGVR 同级), 默认权重 0 (向后兼容);
  env NLD_VISION_LOSS_WEIGHT > 0 时启用.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- role -> alpha 调度 ---------------------------

# alpha = 文本侧权重; 1 - alpha = 视觉侧权重
_ROLE_ALPHA_TABLE: Dict[str, float] = {
    "abstract": 0.75,   # 偏语义 (intent / latent_hint), 靠近语言锥
    "bridge":   0.50,   # 居中 (k=3 中段)
    "unified":  0.50,   # 居中 (k=1 整段)
    "concrete": 0.25,   # 偏视觉 (像素/可见细节), 靠近视觉锥
}
_DEFAULT_ALPHA = 0.50  # 未知 role 走中点


def role_to_alpha(role: Optional[str]) -> float:
    """role 字符串 -> 固定 alpha (无可学参数)."""
    if role is None:
        return _DEFAULT_ALPHA
    return _ROLE_ALPHA_TABLE.get(str(role).lower().strip(), _DEFAULT_ALPHA)


# --------------------------- slerp ---------------------------

def slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """球面线性插值 (Spherical Linear Interpolation).

    sl(a, b; alpha) = sin((1-alpha)*theta)/sin(theta) * a + sin(alpha*theta)/sin(theta) * b
    其中 theta = arccos(<a,b> / (||a||*||b||)).

    参数:
      a, b: [..., D] 任意同形状; 可未归一化, 内部会做 L2 归一.
      alpha: 标量 ∈ [0, 1]; 0 -> 纯 a 方向, 1 -> 纯 b 方向.

    返回:
      r: [..., D] L2 归一化后的插值结果, 模长为 1.

    数值稳健性:
      - theta 接近 0 或 pi 时退化为 lerp + 归一.
    """
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    cos_t = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_t)            # [..., 1]
    sin_t = torch.sin(theta)             # [..., 1]
    # theta -> 0 时 sin_t -> 0; 用阈值切换到 lerp
    use_lerp = sin_t.abs() < 1e-3
    w_a = torch.where(use_lerp, torch.full_like(sin_t, 1.0 - alpha),
                      torch.sin((1.0 - alpha) * theta) / sin_t)
    w_b = torch.where(use_lerp, torch.full_like(sin_t, alpha),
                      torch.sin(alpha * theta) / sin_t)
    r = w_a * a_n + w_b * b_n
    r = F.normalize(r, dim=-1, eps=eps)
    return r


# --------------------------- 文本重心 t_bar ---------------------------

def text_centroid(
    token_ids: List[int],
    embed_tokens: nn.Module,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    excluded_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """stage key tokens 的 LM input embedding 均值, L2 归一化前的版本.

    返回 [H] 的张量; 若 token 全被排除或为空则返回 None.
    embed_tokens 通常是 inner_model.get_input_embeddings(); 对其输出做 detach
    避免视觉侧 loss 的梯度回传到 LM embedding 表 (我们只塑形 hidden, 不改 embedding).
    """
    if not token_ids:
        return None

    if excluded_ids is not None and excluded_ids.numel() > 0:
        excluded_set = set(int(x) for x in excluded_ids.tolist())
        token_ids = [int(t) for t in token_ids if int(t) not in excluded_set]
        if not token_ids:
            return None

    ids = torch.tensor(token_ids, device=device, dtype=torch.long).unique()
    if ids.numel() == 0:
        return None

    with torch.no_grad():
        embs = embed_tokens(ids).to(dtype=dtype)  # [K, H]
    centroid = embs.mean(dim=0)                    # [H]
    return centroid


# --------------------------- 视觉重心 v_bar (top-K 检索) ---------------------------

def visual_centroid_topk(
    query: torch.Tensor,            # [H], 已 detach, float32
    visual_tokens: torch.Tensor,    # [N_v, H], 已 detach, float32
    top_k: int = 6,
    eps: float = 1e-6,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """用文本 query 在视觉 token 上做 cosine top-K 检索, 返回 top-K 的均值.

    参数:
      query: stage 概念词的均值 embedding, 充当文本检索 query.
      visual_tokens: 该样本的视觉 embedding 序列 (已经过 vision encoder + projector,
                     与 LM hidden 同维度).
      top_k: 取前 K 个最相关视觉 token; K 自动夹到 [1, N_v].

    返回:
      (centroid, stats):
        centroid: [H] L2 归一化前的均值, None 表示无可用视觉 token.
        stats:    诊断指标 dict (top1_sim, mean_topk_sim).
    """
    if visual_tokens is None or visual_tokens.numel() == 0 or visual_tokens.dim() != 2:
        return None, {}
    N_v = visual_tokens.shape[0]
    if N_v < 1:
        return None, {}

    K = max(1, min(int(top_k), N_v))

    q = F.normalize(query, dim=-1, eps=eps)            # [H]
    V = F.normalize(visual_tokens, dim=-1, eps=eps)    # [N_v, H]
    sims = V @ q                                       # [N_v]

    top_sims, top_idx = torch.topk(sims, k=K, dim=0)   # [K], [K]
    selected = visual_tokens.index_select(0, top_idx)  # [K, H] (用未归一化原向量取均值)
    centroid = selected.mean(dim=0)                    # [H]

    stats = {
        "vis_top1_sim": float(top_sims[0].item()),
        "vis_topk_sim_mean": float(top_sims.mean().item()),
    }
    return centroid, stats


# --------------------------- 双 anchor margin loss ---------------------------

def dual_anchor_margin_loss(
    h: torch.Tensor,        # [H]
    r: torch.Tensor,        # [H]  (slerp 参考目标, 已归一)
    v_bar: torch.Tensor,    # [H]
    t_bar: torch.Tensor,    # [H]
    margin: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """双 anchor margin: 要求 cos(h,r) > cos(h,v) + δ 且 cos(h,r) > cos(h,t) + δ.

    用 d = 1 - cos 定义距离:
      L = ReLU(d(h,r) - d(h,v) + δ) + ReLU(d(h,r) - d(h,t) + δ)
    等价于:
      L = ReLU(cos(h,v) - cos(h,r) + δ) + ReLU(cos(h,t) - cos(h,r) + δ)

    后一种形式数值更稳定 (避免减法误差), 实现采用之.
    """
    h_n = F.normalize(h, dim=-1, eps=eps)
    r_n = F.normalize(r, dim=-1, eps=eps)
    v_n = F.normalize(v_bar, dim=-1, eps=eps)
    t_n = F.normalize(t_bar, dim=-1, eps=eps)

    cos_hr = (h_n * r_n).sum(dim=-1)
    cos_hv = (h_n * v_n).sum(dim=-1)
    cos_ht = (h_n * t_n).sum(dim=-1)

    loss = F.relu(cos_hv - cos_hr + margin) + F.relu(cos_ht - cos_hr + margin)
    return loss


# --------------------------- 端到端: 单 (b, s) 项的 vision loss ---------------------------

def compute_vision_loss_one_step(
    h_step: torch.Tensor,                   # [H], 该样本第 s 步 hidden (有梯度)
    stage_token_ids: List[int],
    stage_role: Optional[str],
    visual_tokens: torch.Tensor,            # [N_v, H], detached
    embed_tokens: nn.Module,
    top_k: int = 6,
    margin: float = 0.05,
    excluded_ids: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """对单个 (sample, stage) 项计算视觉侧 vision_loss.

    返回:
      (loss, stats); loss 为 None 表示该 step 数据不足 (跳过, 不计入 sum/count).
      stats 字段: vis_top1_sim, vis_topk_sim_mean, vis_alpha,
                 vis_cos_h_r, vis_cos_h_v, vis_cos_h_t (诊断, float).
    """
    if not stage_token_ids:
        return None, {}
    device = h_step.device

    # 文本重心 (无梯度)
    t_bar = text_centroid(
        token_ids=stage_token_ids,
        embed_tokens=embed_tokens,
        device=device,
        dtype=torch.float32,
        excluded_ids=excluded_ids,
    )
    if t_bar is None:
        return None, {}

    # 视觉重心 (无梯度): 用 t_bar 当 query, 在视觉 token 上 top-K 检索
    v_f = visual_tokens.detach().to(device=device, dtype=torch.float32)
    v_bar, vis_stats = visual_centroid_topk(
        query=t_bar.detach(),
        visual_tokens=v_f,
        top_k=top_k,
    )
    if v_bar is None:
        return None, {}

    # slerp 过渡参考 (无梯度): r = slerp(v_bar, t_bar; alpha)
    # 这里把 v_bar 当 a (alpha=0 -> 纯视觉), t_bar 当 b (alpha=1 -> 纯文本)
    alpha = role_to_alpha(stage_role)
    r = slerp(v_bar, t_bar, alpha=alpha)

    # hidden -> float32 (与重心同 dtype, 但保留计算图)
    h_f = h_step.to(dtype=torch.float32)

    # 双 anchor margin loss (有梯度, 仅经 h_f)
    loss = dual_anchor_margin_loss(
        h=h_f, r=r, v_bar=v_bar, t_bar=t_bar, margin=margin,
    )

    # 诊断 (no_grad)
    with torch.no_grad():
        h_n = F.normalize(h_f, dim=-1)
        r_n = F.normalize(r, dim=-1)
        v_n = F.normalize(v_bar, dim=-1)
        t_n = F.normalize(t_bar, dim=-1)
        diag = {
            "vis_alpha": float(alpha),
            "vis_cos_h_r": float((h_n * r_n).sum().item()),
            "vis_cos_h_v": float((h_n * v_n).sum().item()),
            "vis_cos_h_t": float((h_n * t_n).sum().item()),
        }
        diag.update(vis_stats)

    return loss, diag


# --------------------------- 端到端: forward 入口 (batched) ---------------------------

def compute_vision_loss(
    all_step_outputs: List[torch.Tensor],            # list of [B, 1, H], len=actual_steps
    stage_key_token_ids: List[List[List[int]]],      # [B][stage][token_ids]
    stage_roles: Optional[List[List[str]]],          # [B][stage]; 可为 None (走 default alpha)
    image_embeds_per_sample: List[torch.Tensor],     # list of [N_i, H], len=B
    embed_tokens: nn.Module,
    B: int,
    actual_steps: int,
    device: torch.device,
    top_k: int = 6,
    margin: float = 0.05,
    excluded_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """对当前 boundary 的所有 (b, s) 项累加 vision_loss + 聚合诊断.

    返回:
      (loss_mean, stats); loss_mean 是参与样本数的平均, 无可用项时返回 0 张量.
      stats: {vis_loss, vis_count, vis_alpha_mean, vis_cos_h_r_mean, vis_cos_h_v_mean,
              vis_cos_h_t_mean, vis_top1_sim_mean, vis_topk_sim_mean,
              vis_cos_h_v_abstract / _concrete / _bridge / _unified (按 role 分桶)}.
    """
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)

    if (
        not stage_key_token_ids
        or not image_embeds_per_sample
        or B is None or B < 1
        or actual_steps is None or actual_steps < 1
    ):
        return zero, {"vis_loss": 0.0, "vis_count": 0.0}
    if len(image_embeds_per_sample) < B:
        return zero, {"vis_loss": 0.0, "vis_count": 0.0}

    losses: List[torch.Tensor] = []
    diag_keys = [
        "vis_cos_h_r", "vis_cos_h_v", "vis_cos_h_t",
        "vis_top1_sim", "vis_topk_sim_mean", "vis_alpha",
    ]
    diag_acc: Dict[str, List[float]] = {k: [] for k in diag_keys}
    # 按 role 分桶诊断 (cos_h_v / cos_h_t)
    role_buckets: Dict[str, Dict[str, List[float]]] = {
        "abstract": {"cos_h_v": [], "cos_h_t": []},
        "bridge":   {"cos_h_v": [], "cos_h_t": []},
        "unified":  {"cos_h_v": [], "cos_h_t": []},
        "concrete": {"cos_h_v": [], "cos_h_t": []},
    }

    for b_idx in range(B):
        stages = (
            stage_key_token_ids[b_idx]
            if b_idx < len(stage_key_token_ids) else []
        )
        if not stages:
            continue
        v_b = image_embeds_per_sample[b_idx]
        if v_b is None or v_b.numel() == 0:
            continue

        S_use = min(len(stages), actual_steps)
        if S_use < 1:
            continue

        b_roles: List[Optional[str]]
        if stage_roles is not None and b_idx < len(stage_roles):
            b_roles = list(stage_roles[b_idx]) + [None] * max(0, S_use - len(stage_roles[b_idx]))
        else:
            b_roles = [None] * S_use

        for s in range(S_use):
            tok_ids = stages[s] or []
            if not tok_ids:
                continue
            if s >= len(all_step_outputs):
                continue
            h_step = all_step_outputs[s][b_idx].squeeze(0)  # [H], 有梯度

            loss_one, diag_one = compute_vision_loss_one_step(
                h_step=h_step,
                stage_token_ids=tok_ids,
                stage_role=b_roles[s] if s < len(b_roles) else None,
                visual_tokens=v_b,
                embed_tokens=embed_tokens,
                top_k=top_k,
                margin=margin,
                excluded_ids=excluded_ids,
            )
            if loss_one is None:
                continue

            losses.append(loss_one)
            for k in diag_keys:
                if k in diag_one:
                    diag_acc[k].append(diag_one[k])

            # 按 role 分桶
            role_key = (b_roles[s] or "").lower().strip()
            if role_key in role_buckets:
                if "vis_cos_h_v" in diag_one:
                    role_buckets[role_key]["cos_h_v"].append(diag_one["vis_cos_h_v"])
                if "vis_cos_h_t" in diag_one:
                    role_buckets[role_key]["cos_h_t"].append(diag_one["vis_cos_h_t"])

    stats: Dict[str, float] = {"vis_count": float(len(losses))}
    if not losses:
        stats["vis_loss"] = 0.0
        return zero, stats

    loss_mean = torch.stack(losses).mean()
    stats["vis_loss"] = float(loss_mean.detach().item())
    for k, vals in diag_acc.items():
        if vals:
            stats[f"{k}_mean"] = float(sum(vals) / len(vals))
    for role_key, sub in role_buckets.items():
        for sub_k, vals in sub.items():
            if vals:
                stats[f"vis_{sub_k}_{role_key}"] = float(sum(vals) / len(vals))

    return loss_mean, stats
