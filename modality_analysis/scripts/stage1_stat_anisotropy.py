#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 Analysis: Statistical Rigor + Anisotropy / Intrinsic Dimension
======================================================================
读取 outputs/modality_geometry_v2/embeddings.pt 中已抽取的 V/L/H 向量,
在不重新前向的前提下, 补齐 "顶会审稿人标准" 的分析:

Module A — 统计严谨化
  A1. Bootstrap 95% CI:
        modality_gap_angle (V-L, V-H, L-H)
        cone_half_angle    (V, L, H)
        intra_cosine_mean  (V, L, H)
  A2. von Mises-Fisher MLE for concentration κ
        (Banerjee-Dhillon-Ghosh-Sra 近似公式, JMLR 2005)
  A3. Permutation test:  p-value(cone_H < cone_V), p-value(gap_VL > gap_VH), etc.

Module B — 各向异性 & 内在维数
  B1. Intrinsic Dimension:
        TwoNN   (Facco et al. 2017, Sci. Rep.)
        MLE-ID  (Levina & Bickel 2005, k-NN based)
  B2. Anisotropy / Isotropy:
        top-1  PCA 方差占比 (anisotropy_pc1)
        IsoScore (Rudman et al. 2022 EMNLP, 简化版: cumulative eigenvalue)
        Participation Ratio PR = (Σλ)^2 / Σ(λ^2)
  B3. Cone-shape diagnosis:
        "薄球冠" vs "1D 管":  top-2 PCA 解释方差差异,
         与锥体 half-angle 的一致性检查

依赖: numpy, scipy, torch(仅 load), matplotlib
输入: outputs/modality_geometry_v2/embeddings.pt   (4.7 MB, 446 samples)
输出:
  modality_analysis/results/stage1_metrics.json
  modality_analysis/figures/stage1_geometry_with_CI.pdf
  modality_analysis/figures/stage1_intrinsic_dim.pdf
  modality_analysis/figures/stage1_anisotropy.pdf

Author: modality_analysis pipeline, 2026-05-07
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# ------------------------------------------------------------------
# 0. 路径
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DEFAULT_EMB = os.path.join(
    ROOT_DIR, 'outputs', 'modality_geometry_v2', 'embeddings.pt'
)
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
FIGURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==================================================================
# 1. 读取 embeddings.pt 并容错探测结构
# ==================================================================

def _collect_tensor_2d(obj, depth=0, max_depth=3):
    """递归找出 2D 张量 (N, D) 并按形状聚合."""
    out = []
    if depth > max_depth:
        return out
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 2:
            out.append(obj)
        elif obj.ndim == 3:
            # 尝试压平 batch, 若第二维很小(token) 做 mean pool
            out.append(obj.reshape(-1, obj.shape[-1]))
        return out
    if isinstance(obj, (list, tuple)):
        # list of 1-D tensors 或 2-D tensors
        if obj and all(isinstance(x, torch.Tensor) for x in obj):
            stack_dim = obj[0].ndim
            if stack_dim == 1:
                try:
                    out.append(torch.stack(obj, dim=0))
                    return out
                except Exception:
                    pass
            elif stack_dim == 2:
                try:
                    out.append(torch.cat(obj, dim=0))
                    return out
                except Exception:
                    pass
        for x in obj:
            out.extend(_collect_tensor_2d(x, depth + 1, max_depth))
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_tensor_2d(v, depth + 1, max_depth))
        return out
    return out


def load_vlh(path: str, verbose: bool = True, position: str = 'flat'):
    """
    读取 embeddings.pt, 返回 V, L, H (各自 np.ndarray of shape (N_i, D)).

    真实 schema (v2):
        v_pos : list[N] of (T, D)   T=2: 每样本 T 个位置 (e.g. PSIT step 0 / step 1)
        l_pos : list[N] of (T, D)
        h_unit: list[N] of (T, D)   已 L2 归一化
        alpha : list[N] of (T,)
        meta  : list[N]

    position:
        'flat'   -> 全展平 (N*T, D), 默认
        'pos0'   -> 仅取 t=0 (alpha=1, 视觉主导 step)
        'pos1'   -> 仅取 t=1 (alpha=0, 语言主导 step)
    """
    assert os.path.isfile(path), f"embeddings not found: {path}"
    obj = torch.load(path, map_location='cpu', weights_only=False)

    if verbose:
        print(f"[load] file: {path}")
        print(f"[load] top-level type: {type(obj).__name__}")

    # 首选: dict 且 keys 含 V/L/H 或 vision/language/hidden
    V = L = H = None
    if isinstance(obj, dict):
        if verbose:
            print(f"[load] keys: {list(obj.keys())}")
            for k, v in obj.items():
                t = type(v).__name__
                extra = ''
                if isinstance(v, torch.Tensor):
                    extra = f' shape={tuple(v.shape)} dtype={v.dtype}'
                elif isinstance(v, list):
                    extra = f' list_len={len(v)}'
                    if v and isinstance(v[0], torch.Tensor):
                        extra += f' item_shape={tuple(v[0].shape)}'
                print(f"[load]   {k}: {t}{extra}")

        # 尝试多种命名 (优先匹配真实 schema: v_pos/l_pos/h_unit)
        key_aliases = {
            'V': ['v_pos', 'V', 'v', 'vision', 'visual', 'image', 'img', 'V_emb', 'vis_emb'],
            'L': ['l_pos', 'L', 'l', 'language', 'text', 'lang', 'L_emb', 'text_emb'],
            'H': ['h_unit', 'h_pos', 'H', 'h', 'hidden', 'hs', 'state', 'H_emb', 'hidden_emb'],
        }
        def _pick(keys_try):
            """
            取出单个 V/L/H 的张量, 尊重 position 参数:
                list of (T_i, D) -> 若 position='flat' 则 cat (支持 T_i 不一致),
                                   若 'pos0' 则取 t=0 后 stack (需 T_i>=1),
                                   若 'pos1' 则取 t=1 后 stack (需 T_i>=2, 自动跳过 T_i<2 的样本).
            """
            for k in keys_try:
                if k in obj:
                    val = obj[k]
                    if isinstance(val, torch.Tensor):
                        return val
                    if isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                        if val[0].ndim == 1:
                            return torch.stack(val, dim=0)
                        elif val[0].ndim == 2:
                            if position == 'flat':
                                return torch.cat(val, dim=0)  # (sum T_i, D), 允许 T_i 不一致
                            elif position == 'pos0':
                                picked = [v[0] for v in val if v.shape[0] >= 1]
                                return torch.stack(picked, dim=0)
                            elif position == 'pos1':
                                picked = [v[1] for v in val if v.shape[0] >= 2]
                                if len(picked) == 0:
                                    raise RuntimeError(
                                        f"position='pos1' 但没有任何样本满足 T>=2 (key={k})"
                                    )
                                return torch.stack(picked, dim=0)
                            else:
                                raise ValueError(f"unknown position={position}")
                        elif val[0].ndim == 3:
                            return torch.cat([v.reshape(-1, v.shape[-1]) for v in val], dim=0)
                    if isinstance(val, np.ndarray):
                        return torch.from_numpy(val)
            return None

        V = _pick(key_aliases['V'])
        L = _pick(key_aliases['L'])
        H = _pick(key_aliases['H'])

    # 回退: 扫描全部 2D 张量, 按 N 最接近的 3 个作为 V/L/H
    if any(x is None for x in (V, L, H)):
        tensors = _collect_tensor_2d(obj)
        if verbose:
            print(f"[load][fallback] collected {len(tensors)} 2D tensors: "
                  f"{[tuple(t.shape) for t in tensors[:10]]}")
        if len(tensors) >= 3:
            # 取前 3 个
            V, L, H = tensors[0], tensors[1], tensors[2]

    if V is None or L is None or H is None:
        raise RuntimeError(
            "无法自动识别 V/L/H。请检查 embeddings.pt 内部结构并调整 key_aliases。"
        )

    V = V.detach().float().cpu().numpy()
    L = L.detach().float().cpu().numpy()
    H = H.detach().float().cpu().numpy()

    if verbose:
        print(f"[load] V shape={V.shape}, L shape={L.shape}, H shape={H.shape}")

    return V, L, H


# ==================================================================
# 2. 基础几何工具
# ==================================================================

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def mean_direction(x: np.ndarray) -> np.ndarray:
    """样本均值向量 (未归一化), 用于 cone-center / vMF MLE."""
    return x.mean(axis=0)


def cone_half_angle(x: np.ndarray) -> float:
    """cone half-angle (度) = mean arccos(cos(xi, center))."""
    c = l2_normalize(mean_direction(x))
    xn = l2_normalize(x)
    coss = np.clip(xn @ c, -1.0, 1.0)
    angles = np.degrees(np.arccos(coss))
    return float(angles.mean())


def modality_gap_angle(a: np.ndarray, b: np.ndarray) -> float:
    """mean-to-mean 角度 (度)."""
    ca = l2_normalize(mean_direction(a))
    cb = l2_normalize(mean_direction(b))
    cos = float(np.clip(ca @ cb, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def intra_cosine_mean(x: np.ndarray, max_pairs: int = 20000, rng=None) -> float:
    """随机配对的 intra cosine mean (避免 O(N^2))."""
    if rng is None:
        rng = np.random.default_rng(0)
    xn = l2_normalize(x)
    N = xn.shape[0]
    idx_i = rng.integers(0, N, size=max_pairs)
    idx_j = rng.integers(0, N, size=max_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    coss = (xn[idx_i] * xn[idx_j]).sum(axis=-1)
    return float(coss.mean())


# ==================================================================
# 3. Module A — Bootstrap / vMF MLE / Permutation
# ==================================================================

def bootstrap_ci(
    func,
    data_args: list,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng=None,
):
    """
    对 func(*data_args) 做 bootstrap CI.
    data_args: list of np.ndarray, 每个 shape (N_i, D),
               每个都用自己的 N_i 做有放回重采样.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    estimates = []
    for _ in range(n_boot):
        resamp = []
        for arr in data_args:
            n = arr.shape[0]
            idx = rng.integers(0, n, size=n)
            resamp.append(arr[idx])
        try:
            estimates.append(func(*resamp))
        except Exception:
            continue
    estimates = np.array(estimates)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(estimates, 100 * alpha))
    hi = float(np.percentile(estimates, 100 * (1 - alpha)))
    return {
        'mean': float(estimates.mean()),
        'std': float(estimates.std(ddof=1)),
        'ci_low': lo,
        'ci_high': hi,
        'n_boot': int(len(estimates)),
    }


def vmf_mle_kappa(x: np.ndarray) -> dict:
    """
    von Mises-Fisher concentration MLE
    ==================================
    Banerjee-Dhillon-Ghosh-Sra (JMLR 2005) 近似:
        κ ≈ r̄ * (d - r̄²) / (1 - r̄²)
    其中 r̄ = ||mean(x_unit)|| ∈ [0, 1], d = 维度.
    κ 越大 → 分布越集中 (等价于 cone 越窄).
    另给出 "等效 cone half-angle": acos(r̄) (度).
    """
    xn = l2_normalize(x)
    mean_vec = xn.mean(axis=0)
    r_bar = float(np.linalg.norm(mean_vec))
    d = xn.shape[1]
    r_bar_safe = min(r_bar, 1.0 - 1e-9)
    kappa = r_bar_safe * (d - r_bar_safe ** 2) / (1 - r_bar_safe ** 2)
    eff_half_angle = float(np.degrees(np.arccos(r_bar_safe)))
    return {
        'r_bar': r_bar,
        'kappa': float(kappa),
        'eff_half_angle_deg': eff_half_angle,
        'dim': int(d),
        'n': int(xn.shape[0]),
    }


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    stat_func,
    n_perm: int = 2000,
    alternative: str = 'two-sided',
    rng=None,
) -> dict:
    """
    置换检验: H0: a 和 b 可交换 (来自同一分布).
    stat_func(a, b) 返回标量 (默认用 |mean 差| 或 |几何量差|).
    对两个矩阵行做整体合并后随机划分.
    """
    if rng is None:
        rng = np.random.default_rng(123)
    obs = stat_func(a, b)
    combined = np.concatenate([a, b], axis=0)
    Na = a.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined.shape[0])
        pa = combined[perm[:Na]]
        pb = combined[perm[Na:]]
        s = stat_func(pa, pb)
        if alternative == 'two-sided':
            if abs(s) >= abs(obs):
                count += 1
        elif alternative == 'greater':
            if s >= obs:
                count += 1
        elif alternative == 'less':
            if s <= obs:
                count += 1
    p = (count + 1) / (n_perm + 1)
    return {'observed': float(obs), 'p_value': float(p), 'n_perm': int(n_perm)}


# ==================================================================
# 4. Module B — Intrinsic Dim & Anisotropy
# ==================================================================

def twonn_intrinsic_dim(
    x: np.ndarray,
    max_samples: int = 2000,
    rng=None,
) -> dict:
    """
    TwoNN ID 估计 (Facco, d'Errico, Rodriguez, Laio, Sci. Rep. 2017).
    取每个点最近 2 个邻居距离 r1, r2, 令 μ = r2/r1, 则
        d = log(0.5) / log(F^{-1}(0.5))  (大样本 median-based)
    稳健估计: d = (N-1) / sum(log μ).
    """
    if rng is None:
        rng = np.random.default_rng(7)
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    if N > max_samples:
        idx = rng.choice(N, size=max_samples, replace=False)
        x = x[idx]
        N = x.shape[0]

    # 成对距离 (OK for N<=2000, D<=4096)
    sqr = np.sum(x ** 2, axis=1, keepdims=True)
    d2 = sqr + sqr.T - 2.0 * (x @ x.T)
    d2 = np.maximum(d2, 0)
    dist = np.sqrt(d2)
    # 将自身距离设为 +inf
    np.fill_diagonal(dist, np.inf)
    dist_sorted = np.sort(dist, axis=1)  # ascending
    r1 = dist_sorted[:, 0]
    r2 = dist_sorted[:, 1]
    # 过滤退化点 (r1==0 极少见但可能)
    valid = (r1 > 1e-12) & (r2 > 1e-12) & (r2 >= r1)
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0 + 1e-12]
    log_mu = np.log(mu)
    if len(log_mu) < 5:
        return {'id_mle': None, 'id_median': None, 'n_used': int(len(log_mu))}
    # MLE 估计
    id_mle = float(len(log_mu) / log_mu.sum())
    # Median 估计: d = log(0.5) / log(F^{-1}(0.5))
    # 简化做法: log(2)/median(log_mu)
    id_median = float(np.log(2.0) / np.median(log_mu))
    return {
        'id_mle': id_mle,
        'id_median': id_median,
        'n_used': int(len(log_mu)),
        'n_samples': int(N),
    }


def mle_id_levina_bickel(
    x: np.ndarray,
    k: int = 10,
    max_samples: int = 2000,
    rng=None,
) -> dict:
    """Levina-Bickel MLE intrinsic dimension with k-NN (2005)."""
    if rng is None:
        rng = np.random.default_rng(11)
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    if N > max_samples:
        idx = rng.choice(N, size=max_samples, replace=False)
        x = x[idx]
        N = x.shape[0]

    sqr = np.sum(x ** 2, axis=1, keepdims=True)
    d2 = sqr + sqr.T - 2.0 * (x @ x.T)
    d2 = np.maximum(d2, 0)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, np.inf)
    dist_sorted = np.sort(dist, axis=1)  # (N, N-1) ascending
    # 取前 k 个近邻
    Tk = dist_sorted[:, :k]  # (N, k)
    # Levina-Bickel:  d_hat_k(x_i) = [ (1/(k-1)) * sum_{j=1..k-1} log(T_k / T_j) ]^{-1}
    log_ratio = np.log(Tk[:, -1:] / np.maximum(Tk[:, :-1], 1e-12))  # (N, k-1)
    inv_d = log_ratio.mean(axis=1)
    inv_d = inv_d[inv_d > 1e-12]
    if len(inv_d) < 5:
        return {'id_mean': None, 'id_harmonic': None, 'k': k}
    # Levina-Bickel 推荐 harmonic mean over points
    d_per_point = 1.0 / inv_d
    id_mean = float(d_per_point.mean())
    id_harmonic = float(len(d_per_point) / (1.0 / d_per_point).sum())
    return {
        'id_mean': id_mean,
        'id_harmonic': id_harmonic,
        'k': int(k),
        'n_samples': int(N),
    }


def pca_spectrum(x: np.ndarray, top_k: int = 50, center: bool = True) -> dict:
    """中心化后做 PCA, 返回前 top_k 个 eigenvalue 与相关各向异性指标."""
    x = np.asarray(x, dtype=np.float64)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    # 用 SVD 更稳
    # 为了避免 (D, D) 过大, 用 gram_matrix on samples
    N, D = x.shape
    if N <= D:
        # (N, N) gram
        gram = x @ x.T
        eigvals, _ = np.linalg.eigh(gram)
        eigvals = eigvals[::-1]
    else:
        cov = (x.T @ x) / max(N - 1, 1)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum() + 1e-12
    ratios = eigvals / total
    # anisotropy indicators
    pc1_ratio = float(ratios[0])
    pc_top2 = float(ratios[:2].sum()) if len(ratios) >= 2 else pc1_ratio
    pc_top5 = float(ratios[:5].sum()) if len(ratios) >= 5 else pc_top2
    # Participation ratio (effective rank)
    pr = float((eigvals.sum() ** 2) / ((eigvals ** 2).sum() + 1e-12))
    # IsoScore (Rudman 2022) 简化版: 1 - ||normalized_eigvals - uniform||_2^2 / 2
    # 实际 IsoScore 需做 PCA 再看 diag 协方差的方差; 这里给 simplified_iso_score
    uniform = np.ones_like(ratios) / len(ratios)
    iso_simplified = float(1.0 - 0.5 * np.sum((ratios - uniform) ** 2) / (1.0 - 1.0 / len(ratios)))
    iso_simplified = max(0.0, min(1.0, iso_simplified))
    return {
        'pc1_ratio': pc1_ratio,
        'pc_top2_ratio': pc_top2,
        'pc_top5_ratio': pc_top5,
        'participation_ratio': pr,
        'iso_simplified': iso_simplified,
        'eigvals_top': ratios[:top_k].tolist(),
        'n': int(N),
        'dim': int(D),
    }


# ==================================================================
# 5. 主流程
# ==================================================================

def run_analysis(
    emb_path: str,
    n_boot: int = 1000,
    n_perm: int = 2000,
    id_max_samples: int = 2000,
    seed: int = 42,
    position: str = 'flat',
    verbose: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)

    V, L, H = load_vlh(emb_path, verbose=verbose, position=position)

    # ---------- Module A.1 bootstrap on all scalar geometry metrics ----------
    if verbose:
        print(f"\n[A.1] Bootstrap 95% CI (n_boot={n_boot})...")
    t0 = time.time()

    ci_gap_vl = bootstrap_ci(modality_gap_angle, [V, L], n_boot=n_boot, rng=rng)
    ci_gap_vh = bootstrap_ci(modality_gap_angle, [V, H], n_boot=n_boot, rng=rng)
    ci_gap_lh = bootstrap_ci(modality_gap_angle, [L, H], n_boot=n_boot, rng=rng)

    ci_cone_v = bootstrap_ci(cone_half_angle, [V], n_boot=n_boot, rng=rng)
    ci_cone_l = bootstrap_ci(cone_half_angle, [L], n_boot=n_boot, rng=rng)
    ci_cone_h = bootstrap_ci(cone_half_angle, [H], n_boot=n_boot, rng=rng)

    ci_intra_v = bootstrap_ci(lambda x: intra_cosine_mean(x, rng=rng), [V], n_boot=n_boot, rng=rng)
    ci_intra_l = bootstrap_ci(lambda x: intra_cosine_mean(x, rng=rng), [L], n_boot=n_boot, rng=rng)
    ci_intra_h = bootstrap_ci(lambda x: intra_cosine_mean(x, rng=rng), [H], n_boot=n_boot, rng=rng)

    if verbose:
        print(f"  done in {time.time()-t0:.1f}s")

    # ---------- Module A.2 vMF MLE ----------
    if verbose:
        print("\n[A.2] von Mises-Fisher MLE...")
    vmf_v = vmf_mle_kappa(V)
    vmf_l = vmf_mle_kappa(L)
    vmf_h = vmf_mle_kappa(H)
    if verbose:
        for name, r in [('V', vmf_v), ('L', vmf_l), ('H', vmf_h)]:
            print(f"  {name}: r̄={r['r_bar']:.4f}, κ={r['kappa']:.1f}, "
                  f"eff_half_angle={r['eff_half_angle_deg']:.2f}°")

    # ---------- Module A.3 Permutation test ----------
    if verbose:
        print(f"\n[A.3] Permutation tests (n_perm={n_perm})...")
    t0 = time.time()

    def stat_cone_diff(a, b):
        return cone_half_angle(a) - cone_half_angle(b)

    def stat_gap_diff(a, b):
        # 这个用到 V/H 比较 gap,无双样本统计量. 改成 mean-angle-to-anchor 差.
        anchor = l2_normalize(np.concatenate([a, b], axis=0).mean(axis=0))
        def _mean_angle_to(x):
            xn = l2_normalize(x)
            return float(np.degrees(np.arccos(np.clip(xn @ anchor, -1, 1))).mean())
        return _mean_angle_to(a) - _mean_angle_to(b)

    perm_cone_HV = permutation_test(H, V, stat_cone_diff, n_perm=n_perm,
                                    alternative='two-sided', rng=rng)
    perm_cone_HL = permutation_test(H, L, stat_cone_diff, n_perm=n_perm,
                                    alternative='two-sided', rng=rng)
    perm_gap_VL_vs_VH = permutation_test(L, H, stat_gap_diff, n_perm=n_perm,
                                         alternative='two-sided', rng=rng)

    if verbose:
        print(f"  done in {time.time()-t0:.1f}s")
        print(f"  cone H vs V:  obs diff={perm_cone_HV['observed']:+.2f}°, p={perm_cone_HV['p_value']:.4f}")
        print(f"  cone H vs L:  obs diff={perm_cone_HL['observed']:+.2f}°, p={perm_cone_HL['p_value']:.4f}")

    # ---------- Module B.1 Intrinsic dimension ----------
    if verbose:
        print("\n[B.1] Intrinsic dimension (TwoNN + Levina-Bickel k=10)...")
    t0 = time.time()
    id_v_twonn = twonn_intrinsic_dim(V, max_samples=id_max_samples, rng=rng)
    id_l_twonn = twonn_intrinsic_dim(L, max_samples=id_max_samples, rng=rng)
    id_h_twonn = twonn_intrinsic_dim(H, max_samples=id_max_samples, rng=rng)

    id_v_lb = mle_id_levina_bickel(V, k=10, max_samples=id_max_samples, rng=rng)
    id_l_lb = mle_id_levina_bickel(L, k=10, max_samples=id_max_samples, rng=rng)
    id_h_lb = mle_id_levina_bickel(H, k=10, max_samples=id_max_samples, rng=rng)

    if verbose:
        print(f"  done in {time.time()-t0:.1f}s")
        for name, a, b in [('V', id_v_twonn, id_v_lb),
                           ('L', id_l_twonn, id_l_lb),
                           ('H', id_h_twonn, id_h_lb)]:
            print(f"  {name}: TwoNN_MLE={a['id_mle']:.2f}, TwoNN_med={a['id_median']:.2f}, "
                  f"LB_mean={b['id_mean']:.2f}, LB_harm={b['id_harmonic']:.2f}")

    # ---------- Module B.2 Anisotropy (PCA spectrum) ----------
    if verbose:
        print("\n[B.2] Anisotropy / PCA spectrum...")
    pca_v = pca_spectrum(V)
    pca_l = pca_spectrum(L)
    pca_h = pca_spectrum(H)
    if verbose:
        for name, r in [('V', pca_v), ('L', pca_l), ('H', pca_h)]:
            print(f"  {name}: pc1={r['pc1_ratio']*100:.1f}%, "
                  f"pc_top5={r['pc_top5_ratio']*100:.1f}%, "
                  f"PR={r['participation_ratio']:.1f}, "
                  f"IsoSimpl={r['iso_simplified']:.3f}")

    # ---------- 汇总 ----------
    results = {
        'meta': {
            'emb_path': os.path.abspath(emb_path),
            'position': position,
            'n_V': int(V.shape[0]),
            'n_L': int(L.shape[0]),
            'n_H': int(H.shape[0]),
            'dim': int(V.shape[1]),
            'n_boot': n_boot,
            'n_perm': n_perm,
            'id_max_samples': id_max_samples,
            'seed': seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'moduleA_statistics': {
            'bootstrap_ci': {
                'modality_gap_angle_vl': ci_gap_vl,
                'modality_gap_angle_vh': ci_gap_vh,
                'modality_gap_angle_lh': ci_gap_lh,
                'cone_half_angle_V': ci_cone_v,
                'cone_half_angle_L': ci_cone_l,
                'cone_half_angle_H': ci_cone_h,
                'intra_cos_mean_V': ci_intra_v,
                'intra_cos_mean_L': ci_intra_l,
                'intra_cos_mean_H': ci_intra_h,
            },
            'vmf_mle': {
                'V': vmf_v,
                'L': vmf_l,
                'H': vmf_h,
            },
            'permutation_tests': {
                'cone_H_vs_V_two_sided': perm_cone_HV,
                'cone_H_vs_L_two_sided': perm_cone_HL,
                'mean_angle_L_vs_H_two_sided': perm_gap_VL_vs_VH,
            },
        },
        'moduleB_anisotropy_id': {
            'intrinsic_dim': {
                'V': {'twonn': id_v_twonn, 'levina_bickel_k10': id_v_lb},
                'L': {'twonn': id_l_twonn, 'levina_bickel_k10': id_l_lb},
                'H': {'twonn': id_h_twonn, 'levina_bickel_k10': id_h_lb},
            },
            'pca_spectrum': {
                'V': pca_v,
                'L': pca_l,
                'H': pca_h,
            },
        },
    }
    return results


# ==================================================================
# 6. 可视化
# ==================================================================

def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 中文字体
    import matplotlib.font_manager as fm
    for f in ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei']:
        if any(f.lower() in fn.name.lower() for fn in fm.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [f] + plt.rcParams.get('font.sans-serif', [])
            break
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def plot_geometry_with_ci(results: dict, out_path: str):
    plt = _setup_mpl()
    bs = results['moduleA_statistics']['bootstrap_ci']
    vmf = results['moduleA_statistics']['vmf_mle']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Module A: Geometry with 95% Bootstrap CI & vMF MLE',
                 fontsize=14, fontweight='bold')

    # A) Modality gaps
    ax = axes[0]
    names = ['V-L', 'V-H', 'L-H']
    keys = ['modality_gap_angle_vl', 'modality_gap_angle_vh', 'modality_gap_angle_lh']
    means = [bs[k]['mean'] for k in keys]
    errs = [[bs[k]['mean']-bs[k]['ci_low'] for k in keys],
            [bs[k]['ci_high']-bs[k]['mean'] for k in keys]]
    bars = ax.bar(names, means, yerr=errs, capsize=6,
                  color=['#FF6B35', '#7B2FBE', '#4ECDC4'], alpha=0.85,
                  edgecolor='black')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, m+2, f'{m:.1f}°',
                ha='center', fontsize=10, fontweight='bold')
    ax.axhline(90, color='gray', linestyle=':', alpha=0.5, label='orthogonal (90°)')
    ax.set_ylabel('Angle (degrees)', fontsize=11)
    ax.set_title('Modality Gap (mean-to-mean) with 95% CI', fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # B) Cone half-angles + vMF eff angle
    ax = axes[1]
    names = ['V', 'L', 'H']
    keys = ['cone_half_angle_V', 'cone_half_angle_L', 'cone_half_angle_H']
    means = [bs[k]['mean'] for k in keys]
    errs = [[bs[k]['mean']-bs[k]['ci_low'] for k in keys],
            [bs[k]['ci_high']-bs[k]['mean'] for k in keys]]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x-w/2, means, w, yerr=errs, capsize=5,
                   color=['#FF6B35', '#4ECDC4', '#7B2FBE'], alpha=0.85,
                   edgecolor='black', label='ad-hoc half-angle')
    vmf_vals = [vmf[n]['eff_half_angle_deg'] for n in names]
    bars2 = ax.bar(x+w/2, vmf_vals, w,
                   color=['#FF6B35', '#4ECDC4', '#7B2FBE'], alpha=0.45,
                   edgecolor='black', hatch='//', label='vMF eff. half-angle')
    for i, (m, v) in enumerate(zip(means, vmf_vals)):
        ax.text(i-w/2, m+1.5, f'{m:.1f}°', ha='center', fontsize=9)
        ax.text(i+w/2, v+1.5, f'{v:.1f}°', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('Half-angle (degrees)', fontsize=11)
    ax.set_title('Cone Half-angles: ad-hoc vs vMF MLE', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # C) vMF kappa (log scale)
    ax = axes[2]
    kappas = [vmf['V']['kappa'], vmf['L']['kappa'], vmf['H']['kappa']]
    rbars = [vmf['V']['r_bar'], vmf['L']['r_bar'], vmf['H']['r_bar']]
    bars = ax.bar(['V', 'L', 'H'], kappas,
                  color=['#FF6B35', '#4ECDC4', '#7B2FBE'], alpha=0.85,
                  edgecolor='black')
    for bar, k, r in zip(bars, kappas, rbars):
        ax.text(bar.get_x()+bar.get_width()/2, k*1.05,
                f'κ={k:.0f}\nr̄={r:.3f}',
                ha='center', fontsize=9)
    ax.set_ylabel('vMF concentration κ', fontsize=11)
    ax.set_title('vMF Concentration (higher = tighter cone)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ saved: {out_path}")


def plot_intrinsic_dim(results: dict, out_path: str):
    plt = _setup_mpl()
    idr = results['moduleB_anisotropy_id']['intrinsic_dim']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = ['V', 'L', 'H']
    estimators = [
        ('TwoNN (MLE)', lambda r: r['twonn']['id_mle']),
        ('TwoNN (median)', lambda r: r['twonn']['id_median']),
        ('Levina-Bickel (mean)', lambda r: r['levina_bickel_k10']['id_mean']),
        ('Levina-Bickel (harmonic)', lambda r: r['levina_bickel_k10']['id_harmonic']),
    ]
    x = np.arange(len(names))
    w = 0.2
    colors = ['#FF6B35', '#4ECDC4', '#7B2FBE', '#FDCB6E']
    for i, (label, getter) in enumerate(estimators):
        vals = [getter(idr[n]) for n in names]
        bars = ax.bar(x + (i - 1.5)*w, vals, w, label=label,
                      color=colors[i], alpha=0.85, edgecolor='black')
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}',
                        ha='center', fontsize=8)
    dim = results['meta']['dim']
    ax.axhline(dim, color='red', linestyle=':', alpha=0.5, label=f'ambient dim={dim}')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('Intrinsic dimension', fontsize=11)
    ax.set_title(f'Module B.1: Intrinsic Dimension (ambient={dim}, n_samples={results["meta"]["n_V"]})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ saved: {out_path}")


def plot_anisotropy(results: dict, out_path: str):
    plt = _setup_mpl()
    pca = results['moduleB_anisotropy_id']['pca_spectrum']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: PCA eigenvalue spectrum
    ax = axes[0]
    colors = {'V': '#FF6B35', 'L': '#4ECDC4', 'H': '#7B2FBE'}
    for name in ['V', 'L', 'H']:
        spec = np.array(pca[name]['eigvals_top'])
        cum = np.cumsum(spec)
        ax.plot(range(1, len(cum)+1), cum, '-o',
                color=colors[name], label=f'{name}', markersize=4, linewidth=2)
    ax.set_xlabel('PC index', fontsize=11)
    ax.set_ylabel('Cumulative variance explained', fontsize=11)
    ax.set_title('Cumulative PCA Spectrum (top-50 PCs)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90%')

    # Right: anisotropy summary bars
    ax = axes[1]
    metrics = ['pc1_ratio', 'pc_top5_ratio', 'iso_simplified']
    labels = ['PC1 share', 'Top-5 share', 'IsoSimpl (↑=isotropic)']
    names = ['V', 'L', 'H']
    x = np.arange(len(metrics))
    w = 0.25
    for i, name in enumerate(names):
        vals = [pca[name][m] for m in metrics]
        bars = ax.bar(x + (i-1)*w, vals, w,
                      color=colors[name], alpha=0.85, edgecolor='black', label=name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.2f}',
                    ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Ratio', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title('Module B.2: Anisotropy Summary', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 额外：PR 注记
    pr_text = " | ".join([f"{n}: PR={pca[n]['participation_ratio']:.1f}" for n in names])
    fig.text(0.5, -0.02, f"Participation Ratio (effective rank):  {pr_text}",
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ saved: {out_path}")


# ==================================================================
# 7. CLI
# ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', default=DEFAULT_EMB,
                        help='Path to embeddings.pt with V/L/H vectors')
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--n_perm', type=int, default=2000)
    parser.add_argument('--id_max_samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--position', choices=['flat', 'pos0', 'pos1'], default='pos0',
                        help="哪个 latent-step 位置: pos0 (alpha=1, 视觉主导), "
                             "pos1 (alpha=0, 语言主导), flat (全展平, N*T 个样本)")
    parser.add_argument('--all_positions', action='store_true',
                        help='一次性跑 pos0 / pos1 / flat 3 个位置, 分别保存输出')
    parser.add_argument('--out_json', default=None,
                        help='默认: modality_analysis/results/stage1_metrics_<position>.json')
    parser.add_argument('--probe_only', action='store_true',
                        help='只打印 embeddings.pt 结构, 不跑分析')
    args = parser.parse_args()

    print("=" * 70)
    print("  Stage-1 Analysis: Statistical Rigor + Anisotropy / Intrinsic Dim")
    print("=" * 70)
    print(f"  emb_path       : {args.emb_path}")
    print(f"  n_boot         : {args.n_boot}")
    print(f"  n_perm         : {args.n_perm}")
    print(f"  id_max_samples : {args.id_max_samples}")
    print(f"  position       : {args.position}{'  (但 --all_positions 会覆盖)' if args.all_positions else ''}")
    print("=" * 70)

    if args.probe_only:
        for pos in ['flat', 'pos0', 'pos1']:
            print(f"\n--- probing position='{pos}' ---")
            V, L, H = load_vlh(args.emb_path, verbose=True, position=pos)
            print(f"Probe OK [{pos}].  V: {V.shape}, L: {L.shape}, H: {H.shape}")
        return

    positions_to_run = ['pos0', 'pos1', 'flat'] if args.all_positions else [args.position]

    for pos in positions_to_run:
        print(f"\n{'=' * 70}\n  RUN position = '{pos}'\n{'=' * 70}")
        results = run_analysis(
            args.emb_path,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            id_max_samples=args.id_max_samples,
            seed=args.seed,
            position=pos,
            verbose=True,
        )

        # 输出文件名按 position 区分
        out_json = args.out_json if (args.out_json and not args.all_positions) \
                   else os.path.join(RESULTS_DIR, f'stage1_metrics_{pos}.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ metrics written to {out_json}")

        plot_geometry_with_ci(results,
            os.path.join(FIGURES_DIR, f'stage1_geometry_with_CI_{pos}.pdf'))
        plot_intrinsic_dim(results,
            os.path.join(FIGURES_DIR, f'stage1_intrinsic_dim_{pos}.pdf'))
        plot_anisotropy(results,
            os.path.join(FIGURES_DIR, f'stage1_anisotropy_{pos}.pdf'))

        # 简要文本报告
        report_path = os.path.join(RESULTS_DIR, f'stage1_summary_{pos}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Stage-1 Analysis Summary (position={pos})\n")
            f.write("=" * 50 + "\n\n")
            bs = results['moduleA_statistics']['bootstrap_ci']
            vmf = results['moduleA_statistics']['vmf_mle']
            perm = results['moduleA_statistics']['permutation_tests']
            pca = results['moduleB_anisotropy_id']['pca_spectrum']
            idr = results['moduleB_anisotropy_id']['intrinsic_dim']

            f.write("[A.1] Bootstrap 95%% CI\n".replace("%%", "%"))
            for k, r in bs.items():
                f.write(f"  {k:30s}: {r['mean']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]\n")
            f.write("\n[A.2] vMF MLE\n")
            for n in ['V', 'L', 'H']:
                r = vmf[n]
                f.write(f"  {n}: r_bar={r['r_bar']:.4f}, kappa={r['kappa']:.2f}, "
                        f"eff_half_angle={r['eff_half_angle_deg']:.2f} deg\n")
            f.write("\n[A.3] Permutation tests\n")
            for k, r in perm.items():
                f.write(f"  {k}: observed={r['observed']:.3f}, p={r['p_value']:.4f}\n")
            f.write("\n[B.1] Intrinsic dimension\n")
            for n in ['V', 'L', 'H']:
                t = idr[n]['twonn']; l = idr[n]['levina_bickel_k10']
                f.write(f"  {n}: TwoNN_MLE={t['id_mle']:.2f}, TwoNN_med={t['id_median']:.2f}, "
                        f"LB_mean={l['id_mean']:.2f}, LB_harm={l['id_harmonic']:.2f}\n")
            f.write("\n[B.2] Anisotropy\n")
            for n in ['V', 'L', 'H']:
                r = pca[n]
                f.write(f"  {n}: PC1={r['pc1_ratio']*100:.1f}%, Top5={r['pc_top5_ratio']*100:.1f}%, "
                        f"PR={r['participation_ratio']:.2f}, IsoSimpl={r['iso_simplified']:.3f}\n")
        print(f"✅ summary written to {report_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
