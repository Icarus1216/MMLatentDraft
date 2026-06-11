#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1b: Paired Analysis — H_latent vs H_native
================================================
读取 extract_embeddings_with_native.py 的产出 (.pt), 做 *配对* 几何对比:

  差分几何 (paired):
    ΔCone_half_angle  = cone(H_latent) - cone(H_native)
    ΔID               = intrinsic_dim(H_latent) - intrinsic_dim(H_native)
    ΔPR               = participation_ratio(H_latent) - participation_ratio(H_native)
    ΔAngle_to_V       = angle(H_latent_i, V_center) - angle(H_native_i, V_center)   [per-sample]
    ΔAngle_to_L       = angle(H_latent_i, L_center) - angle(H_native_i, L_center)   [per-sample]

  每项都给出:
    - bootstrap 95% CI
    - paired permutation test p-value
    - per-sample direction (latent 是否让 H 更靠近 V 而非 L?)

  Layer-wise (可选):
    对每个选定的 layer, 单独做上面的全套比较 -> 画出 ΔCone / ΔID / Δθ_V 随层数的曲线

输入:
  --emb_path  modality_analysis/results/embeddings_native_vs_latent.pt
  --native_reference {latent_pos, answer}  # H_native 用哪个参照
  --latent_step       {0, 1, mean}         # H_latent 取第几步 (0=α≈1 视觉主导, 1=α≈0 语言主导, mean=平均)
  --layer_idx         int | 'all' | 'last' # 用哪一层; 'all' 表示跨所有抽取层画曲线

输出:
  modality_analysis/results/stage1b_paired_metrics_<step>_<layer>.json
  modality_analysis/figures/stage1b_paired_delta_<step>_<layer>.pdf
  modality_analysis/figures/stage1b_layerwise_delta_<step>.pdf   (若 --layer_idx all)

作者: modality_analysis pipeline, 2026-05-07
"""
import os
import sys
import json
import time
import argparse
from typing import Optional, List, Tuple

import numpy as np
import torch

# 复用 stage1_stat_anisotropy.py 中的工具
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from stage1_stat_anisotropy import (
    l2_normalize, cone_half_angle, modality_gap_angle,
    intra_cosine_mean, bootstrap_ci, vmf_mle_kappa,
    twonn_intrinsic_dim, mle_id_levina_bickel, pca_spectrum,
)

RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
FIGURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==================================================================
# 1. 读取新 schema 的 embeddings
# ==================================================================

def load_paired_embeddings(
    path: str,
    native_reference: str = 'latent_pos',
    latent_step: str = '0',
    layer_idx=None,
    verbose: bool = True,
):
    """
    返回 (V, L, H_latent, H_native, meta):
      每个都是 np.ndarray of shape (N, D).

    native_reference:
      'latent_pos' -> H_native_latent_pos (同 <|latent|> 触发位置的 native hidden)
      'answer'     -> H_native_answer (答案首 token 的 native hidden)
    latent_step:
      '0' | '1' | 'mean' (对 H_latent: list of Tensor (T_i, D))
    layer_idx:
      None  -> 只用 last-layer (H_latent / H_native_latent_pos 本身)
      int   -> 用 H_latent_layers[:, :, k, :] 的第 k 层
      'last'-> 最后一层 (= -1 在抽取时的索引)
    """
    assert os.path.isfile(path), f"not found: {path}"
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if verbose:
        print(f"[load] file: {path}")
        keys = list(obj.keys()) if isinstance(obj, dict) else '-'
        print(f"[load] keys: {keys}")
        if isinstance(obj, dict):
            for k in keys:
                v = obj[k]
                if isinstance(v, list):
                    print(f"[load]   {k}: list len={len(v)}"
                          f" item[0]={tuple(v[0].shape) if v and hasattr(v[0],'shape') else v[0] if v else None}")

    meta = obj.get('meta', {})
    layers_cfg = meta.get('layers', None)

    # 决定要用 last-layer 还是 layer-wise
    use_layerwise = (layer_idx is not None)
    if use_layerwise:
        assert 'H_latent_layers' in obj and len(obj['H_latent_layers']) > 0, \
            "embeddings.pt 没有 H_latent_layers; 抽取时 --layers 不能为空"
        if isinstance(layer_idx, str) and layer_idx == 'last':
            layer_k = len(layers_cfg) - 1
        else:
            layer_k = int(layer_idx)

    # V / L: 若要 layer-wise 则从 V_layers/L_layers 拿，否则用 V/L
    Vs, Ls = [], []
    H_lats, H_nats = [], []
    keep_idx = []

    N = len(obj['V'])
    for i in range(N):
        # H_latent (T_i, D) 或 (T_i, L, D)
        if use_layerwise:
            HL_i_full = obj['H_latent_layers'][i]  # [T, L, D]
            HL_i = HL_i_full[:, layer_k, :]        # [T, D]
        else:
            HL_i = obj['H_latent'][i]              # [T, D]
        if HL_i is None:
            continue

        T = HL_i.shape[0]
        # 选择 latent step
        if latent_step == '0':
            if T < 1:
                continue
            h_lat = HL_i[0]
        elif latent_step == '1':
            if T < 2:
                continue
            h_lat = HL_i[1]
        elif latent_step == 'mean':
            h_lat = HL_i.mean(dim=0)
        else:
            raise ValueError(f"unknown latent_step={latent_step}")

        # H_native
        if native_reference == 'latent_pos':
            if use_layerwise:
                HN_i_full = obj['H_native_layers'][i] if len(obj.get('H_native_layers', [])) > i else None
            else:
                HN_i_full = obj['H_native_latent_pos'][i] if len(obj.get('H_native_latent_pos', [])) > i else None
            if HN_i_full is None:
                continue
            if use_layerwise:
                HN = HN_i_full[:, layer_k, :]  # [T, D]
            else:
                HN = HN_i_full                 # [T, D]
            if latent_step == '0':
                if HN.shape[0] < 1: continue
                h_nat = HN[0]
            elif latent_step == '1':
                if HN.shape[0] < 2: continue
                h_nat = HN[1]
            else:
                h_nat = HN.mean(dim=0)
        elif native_reference == 'answer':
            if use_layerwise:
                HA = obj.get('H_native_answer_layers', [None]*N)[i]
                if HA is None: continue
                h_nat = HA[layer_k]           # [D]
            else:
                HA = obj.get('H_native_answer', [None]*N)[i]
                if HA is None: continue
                h_nat = HA                    # [D]
        else:
            raise ValueError(f"unknown native_reference={native_reference}")

        # V / L
        if use_layerwise and len(obj.get('V_layers', [])) > i:
            V_vec = obj['V_layers'][i][layer_k]
            L_vec = obj['L_layers'][i][layer_k]
        else:
            V_vec = obj['V'][i]
            L_vec = obj['L'][i]

        Vs.append(V_vec)
        Ls.append(L_vec)
        H_lats.append(h_lat)
        H_nats.append(h_nat)
        keep_idx.append(i)

    V = torch.stack(Vs, dim=0).float().numpy()     # (N, D)
    L = torch.stack(Ls, dim=0).float().numpy()
    H_latent = torch.stack(H_lats, dim=0).float().numpy()
    H_native = torch.stack(H_nats, dim=0).float().numpy()

    if verbose:
        print(f"[load] final V={V.shape}, L={L.shape}, "
              f"H_latent={H_latent.shape}, H_native={H_native.shape}   "
              f"(kept {len(keep_idx)}/{N})")

    meta_out = dict(meta)
    meta_out.update({
        'native_reference': native_reference,
        'latent_step': latent_step,
        'layer_idx': layer_idx,
        'kept_samples': len(keep_idx),
        'kept_indices': keep_idx,
    })
    return V, L, H_latent, H_native, meta_out


# ==================================================================
# 2. Paired Δ 指标
# ==================================================================

def angle_to_anchor(x: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Per-sample angle (deg) of xi to anchor direction."""
    xn = l2_normalize(x)
    a = l2_normalize(anchor)
    coss = np.clip(xn @ a, -1.0, 1.0)
    return np.degrees(np.arccos(coss))


def paired_delta_ci(
    lat_samples: np.ndarray,
    nat_samples: np.ndarray,
    anchor: np.ndarray,
    n_boot: int = 1000,
    rng=None,
):
    """
    对 per-sample Δ(angle_to_anchor) 做 bootstrap 95% CI 和 paired permutation test.
    lat_samples, nat_samples: (N, D), 配对顺序一致.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d_lat = angle_to_anchor(lat_samples, anchor)  # (N,)
    d_nat = angle_to_anchor(nat_samples, anchor)
    delta = d_lat - d_nat  # per-sample

    # bootstrap 95% CI on mean(delta)
    N = len(delta)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots.append(delta[idx].mean())
    boots = np.array(boots)
    # paired permutation test: 随机翻转每个样本的 Δ 符号, p = Pr(|mean*| >= |obs|)
    obs = float(delta.mean())
    signs = rng.choice([-1, 1], size=(n_boot, N))
    perm_means = (signs * delta[None, :]).mean(axis=1)
    p_val = (np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_boot + 1)

    return {
        'mean_delta': obs,
        'std': float(delta.std(ddof=1)),
        'ci_low': float(np.percentile(boots, 2.5)),
        'ci_high': float(np.percentile(boots, 97.5)),
        'paired_p_value': float(p_val),
        'n_samples': int(N),
        'd_lat_mean': float(d_lat.mean()),
        'd_nat_mean': float(d_nat.mean()),
    }


def cone_pr_id_for_both(
    lat_samples: np.ndarray,
    nat_samples: np.ndarray,
    n_boot: int = 500,
    id_max_samples: int = 2000,
    rng=None,
):
    """对两组分别计算 cone / PR / ID 和 Δ."""
    if rng is None:
        rng = np.random.default_rng(1)

    out = {}
    # cone
    cl = cone_half_angle(lat_samples)
    cn = cone_half_angle(nat_samples)
    out['cone_half_angle'] = {'latent': cl, 'native': cn, 'delta': cl - cn}

    # vMF kappa + half-angle
    vl = vmf_mle_kappa(lat_samples)
    vn = vmf_mle_kappa(nat_samples)
    out['vmf_kappa'] = {'latent': vl['kappa'], 'native': vn['kappa'],
                        'log_ratio': float(np.log(vl['kappa'] / max(vn['kappa'], 1e-9)))}
    out['vmf_eff_half_angle_deg'] = {'latent': vl['eff_half_angle_deg'],
                                     'native': vn['eff_half_angle_deg'],
                                     'delta': vl['eff_half_angle_deg'] - vn['eff_half_angle_deg']}

    # ID (TwoNN)
    il = twonn_intrinsic_dim(lat_samples, max_samples=id_max_samples, rng=rng)
    inn = twonn_intrinsic_dim(nat_samples, max_samples=id_max_samples, rng=rng)
    out['id_twonn_mle'] = {'latent': il['id_mle'], 'native': inn['id_mle'],
                           'delta': (il['id_mle'] or 0) - (inn['id_mle'] or 0)}

    # LB ID (harmonic)
    jl = mle_id_levina_bickel(lat_samples, k=10, max_samples=id_max_samples, rng=rng)
    jn = mle_id_levina_bickel(nat_samples, k=10, max_samples=id_max_samples, rng=rng)
    out['id_lb_harmonic'] = {'latent': jl['id_harmonic'], 'native': jn['id_harmonic'],
                             'delta': (jl['id_harmonic'] or 0) - (jn['id_harmonic'] or 0)}

    # PCA PR
    pl = pca_spectrum(lat_samples)
    pn = pca_spectrum(nat_samples)
    out['participation_ratio'] = {'latent': pl['participation_ratio'],
                                  'native': pn['participation_ratio'],
                                  'delta': pl['participation_ratio'] - pn['participation_ratio']}
    out['pc1_ratio'] = {'latent': pl['pc1_ratio'], 'native': pn['pc1_ratio'],
                        'delta': pl['pc1_ratio'] - pn['pc1_ratio']}
    out['iso_simplified'] = {'latent': pl['iso_simplified'], 'native': pn['iso_simplified'],
                             'delta': pl['iso_simplified'] - pn['iso_simplified']}

    # cone CI (bootstrap on each set separately, 比较用)
    out['cone_lat_ci'] = bootstrap_ci(cone_half_angle, [lat_samples], n_boot=n_boot, rng=rng)
    out['cone_nat_ci'] = bootstrap_ci(cone_half_angle, [nat_samples], n_boot=n_boot, rng=rng)

    return out


# ==================================================================
# 3. 单层 run_analysis (含 paired angle CI)
# ==================================================================

def run_paired_layer(V, L, H_latent, H_native,
                     n_boot: int = 1000, n_perm: int = 2000,
                     id_max_samples: int = 2000, seed: int = 42,
                     verbose: bool = True):
    rng = np.random.default_rng(seed)

    V_center = V.mean(axis=0)
    L_center = L.mean(axis=0)

    # Δ angle to V / L (paired, per-sample)
    if verbose: print("[P.1] paired Δ angle to V / L (bootstrap + permutation) ...")
    delta_V = paired_delta_ci(H_latent, H_native, V_center, n_boot=n_boot, rng=rng)
    delta_L = paired_delta_ci(H_latent, H_native, L_center, n_boot=n_boot, rng=rng)

    # cone / ID / PR
    if verbose: print("[P.2] cone / ID / PR for both sets ...")
    cone_pr_id = cone_pr_id_for_both(H_latent, H_native,
                                     n_boot=min(n_boot, 500),
                                     id_max_samples=id_max_samples,
                                     rng=rng)

    # gap 比较
    gap_VL = modality_gap_angle(V, L)
    gap_VH_lat = modality_gap_angle(V, H_latent)
    gap_VH_nat = modality_gap_angle(V, H_native)
    gap_LH_lat = modality_gap_angle(L, H_latent)
    gap_LH_nat = modality_gap_angle(L, H_native)

    result = {
        'meta': {
            'n_samples': int(H_latent.shape[0]),
            'dim': int(H_latent.shape[1]),
            'n_boot': n_boot,
            'n_perm': n_perm,
            'seed': seed,
        },
        'modality_gaps': {
            'V-L': gap_VL,
            'V-H_latent': gap_VH_lat,
            'V-H_native': gap_VH_nat,
            'L-H_latent': gap_LH_lat,
            'L-H_native': gap_LH_nat,
            'Δ(V-H) = lat - nat': gap_VH_lat - gap_VH_nat,
            'Δ(L-H) = lat - nat': gap_LH_lat - gap_LH_nat,
        },
        'paired_delta_angle_to_V': delta_V,
        'paired_delta_angle_to_L': delta_L,
        'cone_pr_id': cone_pr_id,
    }

    if verbose:
        print("\n== summary ==")
        print(f"  gap V-L          = {gap_VL:.2f}°")
        print(f"  V-H: lat={gap_VH_lat:.2f}°  nat={gap_VH_nat:.2f}°  Δ={gap_VH_lat-gap_VH_nat:+.2f}°")
        print(f"  L-H: lat={gap_LH_lat:.2f}°  nat={gap_LH_nat:.2f}°  Δ={gap_LH_lat-gap_LH_nat:+.2f}°")
        print(f"  paired Δangle→V  mean={delta_V['mean_delta']:+.3f}°  CI=[{delta_V['ci_low']:+.3f}, {delta_V['ci_high']:+.3f}]  p={delta_V['paired_p_value']:.4f}")
        print(f"  paired Δangle→L  mean={delta_L['mean_delta']:+.3f}°  CI=[{delta_L['ci_low']:+.3f}, {delta_L['ci_high']:+.3f}]  p={delta_L['paired_p_value']:.4f}")
        print(f"  cone:  lat={cone_pr_id['cone_half_angle']['latent']:.2f}°  nat={cone_pr_id['cone_half_angle']['native']:.2f}°  Δ={cone_pr_id['cone_half_angle']['delta']:+.2f}°")
        print(f"  PR:    lat={cone_pr_id['participation_ratio']['latent']:.2f}  nat={cone_pr_id['participation_ratio']['native']:.2f}  Δ={cone_pr_id['participation_ratio']['delta']:+.2f}")
        _id = cone_pr_id['id_twonn_mle']
        def _f(v, fmt='.2f'):
            if v is None:
                return 'NA'
            try:
                return format(float(v), fmt)
            except Exception:
                return 'NA'
        print(f"  ID(TwoNN MLE): lat={_f(_id['latent'])}  nat={_f(_id['native'])}  Δ={_f(_id['delta'], '+.2f')}")

    return result


# ==================================================================
# 4. 画图
# ==================================================================

def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_paired_delta(result: dict, out_path: str, title_suffix: str = ''):
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Stage-1b Paired Δ: latent − native  ({title_suffix})',
                 fontsize=13, fontweight='bold')

    # Panel A: cone / PR / ID
    ax = axes[0]
    names = ['cone (°)', 'PR', 'ID TwoNN', 'PC1 share']
    def _safe(v):
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0
    raw_lats = [result['cone_pr_id']['cone_half_angle']['latent'],
                result['cone_pr_id']['participation_ratio']['latent'],
                result['cone_pr_id']['id_twonn_mle']['latent'],
                result['cone_pr_id']['pc1_ratio']['latent']]
    raw_nats = [result['cone_pr_id']['cone_half_angle']['native'],
                result['cone_pr_id']['participation_ratio']['native'],
                result['cone_pr_id']['id_twonn_mle']['native'],
                result['cone_pr_id']['pc1_ratio']['native']]
    lats = [_safe(v) for v in raw_lats]
    nats = [_safe(v) for v in raw_nats]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, lats, w, color='#7B2FBE', alpha=0.85, label='latent', edgecolor='black')
    ax.bar(x + w/2, nats, w, color='#4ECDC4', alpha=0.85, label='native', edgecolor='black')
    for i, (lv, nv, rlv, rnv) in enumerate(zip(lats, nats, raw_lats, raw_nats)):
        ltxt = 'NA' if rlv is None else f'{lv:.2f}'
        ntxt = 'NA' if rnv is None else f'{nv:.2f}'
        ax.text(i - w/2, lv + 0.02*max(abs(lv), 1), ltxt,
                ha='center', fontsize=8)
        ax.text(i + w/2, nv + 0.02*max(abs(nv), 1), ntxt,
                ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_title('Geometry: latent vs native')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Δ angle to V / L (paired)
    ax = axes[1]
    dV = result['paired_delta_angle_to_V']
    dL = result['paired_delta_angle_to_L']
    means = [dV['mean_delta'], dL['mean_delta']]
    errs = [[dV['mean_delta'] - dV['ci_low'], dL['mean_delta'] - dL['ci_low']],
            [dV['ci_high'] - dV['mean_delta'], dL['ci_high'] - dL['mean_delta']]]
    colors = ['#FF6B35', '#4ECDC4']
    bars = ax.bar(['Δ angle→V', 'Δ angle→L'], means, yerr=errs, capsize=6,
                  color=colors, alpha=0.85, edgecolor='black')
    for bar, m, p in zip(bars, means, [dV['paired_p_value'], dL['paired_p_value']]):
        sym = '+' if m >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.3,
                f'{sym}{m:.2f}°\np={p:.3g}',
                ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Δ angle (deg), latent − native')
    ax.set_title('Paired Δ angle to V / L  (95% CI)')
    ax.grid(axis='y', alpha=0.3)

    # Panel C: gap table
    ax = axes[2]
    ax.axis('off')
    g = result['modality_gaps']
    rows = [
        ['V-L', f"{g['V-L']:.2f}°", '-'],
        ['V-H_latent', f"{g['V-H_latent']:.2f}°", f"Δ {g['Δ(V-H) = lat - nat']:+.2f}°"],
        ['V-H_native', f"{g['V-H_native']:.2f}°", ''],
        ['L-H_latent', f"{g['L-H_latent']:.2f}°", f"Δ {g['Δ(L-H) = lat - nat']:+.2f}°"],
        ['L-H_native', f"{g['L-H_native']:.2f}°", ''],
    ]
    table = ax.table(cellText=rows, colLabels=['pair', 'angle', 'Δ (lat−nat)'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title('Modality gaps (mean-to-mean)', pad=16)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ figure saved: {out_path}")


def _resolve_layer_labels(layers: List[int], num_model_layers: Optional[int] = None):
    """将 layers 解析为 (order, xs_numeric, xtick_labels), 其中 -1 会被放到序列末端并显示为 'final'."""
    if not layers:
        return [], [], []
    # 确定 -1 应被画在多大的 x：允许只要比最大非负层号大即可
    non_neg = [l for l in layers if l is not None and l >= 0]
    if num_model_layers is None:
        num_model_layers = (max(non_neg) if non_neg else 0) + 8
    xs_numeric = []
    labels = []
    for l in layers:
        if l is None or l == -1:
            xs_numeric.append(float(num_model_layers))
            labels.append('final')
        else:
            xs_numeric.append(float(l))
            labels.append(str(l))
    # 按 xs_numeric 排序
    order = np.argsort(xs_numeric)
    return order, [xs_numeric[i] for i in order], [labels[i] for i in order]


def plot_layerwise_delta(results_per_layer: List[dict], layers: List[int],
                         out_path: str, title_suffix: str = '',
                         num_model_layers: Optional[int] = None):
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f'Stage-1b Layerwise Δ: latent − native  ({title_suffix})',
                 fontsize=13, fontweight='bold')

    order, xs, xtick_labels = _resolve_layer_labels(layers, num_model_layers)
    def _reorder(seq):
        return [seq[i] for i in order]

    # Δ cone
    ax = axes[0]
    dcones = _reorder([r['cone_pr_id']['cone_half_angle']['delta'] for r in results_per_layer])
    ax.plot(xs, dcones, '-o', color='#FF6B35', linewidth=2, markersize=7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('layer'); ax.set_ylabel('Δcone (deg), lat − nat')
    ax.set_title('Δ cone half-angle'); ax.grid(alpha=0.3)
    ax.set_xticks(xs); ax.set_xticklabels(xtick_labels)

    # Δ PR / Δ ID
    ax = axes[1]
    dpr = _reorder([r['cone_pr_id']['participation_ratio']['delta'] for r in results_per_layer])
    did_raw = [r['cone_pr_id']['id_twonn_mle']['delta'] for r in results_per_layer]
    did = _reorder([0.0 if v is None else float(v) for v in did_raw])
    ax.plot(xs, dpr, '-s', color='#7B2FBE', linewidth=2, markersize=7, label='ΔPR')
    ax2 = ax.twinx()
    ax2.plot(xs, did, '-^', color='#4ECDC4', linewidth=2, markersize=7, label='ΔID (TwoNN)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('layer'); ax.set_ylabel('ΔPR', color='#7B2FBE')
    ax2.set_ylabel('ΔID', color='#4ECDC4')
    ax.set_title('Δ participation ratio / intrinsic dim')
    ax.grid(alpha=0.3)
    ax.set_xticks(xs); ax.set_xticklabels(xtick_labels)

    # Δangle to V / L
    ax = axes[2]
    dV = _reorder([r['paired_delta_angle_to_V']['mean_delta'] for r in results_per_layer])
    dL = _reorder([r['paired_delta_angle_to_L']['mean_delta'] for r in results_per_layer])
    ax.plot(xs, dV, '-o', color='#FF6B35', linewidth=2, markersize=7, label='Δθ→V')
    ax.plot(xs, dL, '-s', color='#4ECDC4', linewidth=2, markersize=7, label='Δθ→L')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('layer'); ax.set_ylabel('Δ angle (deg)')
    ax.set_title('Paired Δ angle to V / L'); ax.grid(alpha=0.3); ax.legend()
    ax.set_xticks(xs); ax.set_xticklabels(xtick_labels)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ figure saved: {out_path}")


# ==================================================================
# 5. CLI
# ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', required=True,
                        help='modality_analysis/results/embeddings_native_vs_latent.pt')
    parser.add_argument('--native_reference', default='latent_pos',
                        choices=['latent_pos', 'answer'])
    parser.add_argument('--latent_step', default='0',
                        choices=['0', '1', 'mean'])
    parser.add_argument('--layer_idx', default=None,
                        help="None (last layer), or int k, or 'all' to scan抽取的所有层")
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--n_perm', type=int, default=2000)
    parser.add_argument('--id_max_samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--probe_only', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("  Stage-1b Paired Analysis: H_latent vs H_native")
    print("=" * 70)

    # probe
    if args.probe_only:
        obj = torch.load(args.emb_path, map_location='cpu', weights_only=False)
        print("top keys:", list(obj.keys()))
        meta = obj.get('meta', {})
        print("meta:", {k: v for k, v in meta.items() if k != 'per_sample'})
        print(f"per_sample (first 3): {meta.get('per_sample', [])[:3]}")
        print(f"n lists: V={len(obj['V'])}, H_latent={len(obj['H_latent'])}, "
              f"H_native_latent_pos={len(obj.get('H_native_latent_pos', []))}, "
              f"H_native_answer={len(obj.get('H_native_answer', []))}")
        return

    # 解析 layer_idx
    if args.layer_idx is None or (isinstance(args.layer_idx, str) and args.layer_idx.lower() == 'none'):
        layer_modes = [None]  # 用 last-layer (H_latent 本身, 不走 layers)
    elif args.layer_idx == 'all':
        obj_peek = torch.load(args.emb_path, map_location='cpu', weights_only=False)
        layers_cfg = obj_peek.get('meta', {}).get('layers', None) or []
        if not layers_cfg:
            raise RuntimeError("embeddings 未抽取 layer-wise, 无法 --layer_idx all")
        layer_modes = list(range(len(layers_cfg)))
    else:
        layer_modes = [int(args.layer_idx)]

    # 运行
    all_results = []
    tag_base = f"{args.latent_step}_{args.native_reference}"

    for lm in layer_modes:
        V, L, H_latent, H_native, meta = load_paired_embeddings(
            args.emb_path, native_reference=args.native_reference,
            latent_step=args.latent_step, layer_idx=lm, verbose=True,
        )
        print(f"\n===== layer={lm} =====")
        res = run_paired_layer(
            V, L, H_latent, H_native,
            n_boot=args.n_boot, n_perm=args.n_perm,
            id_max_samples=args.id_max_samples, seed=args.seed,
            verbose=True,
        )
        # layer_idx_scan 是 layers_cfg 数组下标；layer_idx 语义为"真实层号"
        real_layer = None
        if lm is not None:
            try:
                obj_peek2 = torch.load(args.emb_path, map_location='cpu', weights_only=False)
                layers_cfg2 = obj_peek2.get('meta', {}).get('layers', None) or []
                if 0 <= lm < len(layers_cfg2):
                    real_layer = int(layers_cfg2[lm])
                else:
                    real_layer = int(lm)
            except Exception:
                real_layer = int(lm)
        res['meta'].update({'layer_idx_scan': lm,
                            'layer_idx': real_layer,
                            'native_reference': args.native_reference,
                            'latent_step': args.latent_step})
        all_results.append(res)

        # 保存单层 JSON + figure
        tag = f"{tag_base}_layer{lm}" if lm is not None else f"{tag_base}_last"
        out_json = os.path.join(RESULTS_DIR, f'stage1b_paired_metrics_{tag}.json')
        with open(out_json, 'w') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"✅ metrics saved: {out_json}")
        plot_paired_delta(res,
            os.path.join(FIGURES_DIR, f'stage1b_paired_delta_{tag}.pdf'),
            title_suffix=tag)

    # layer-wise curve
    if len(all_results) > 1:
        obj_peek = torch.load(args.emb_path, map_location='cpu', weights_only=False)
        layers_cfg = obj_peek.get('meta', {}).get('layers', None) or []
        num_model_layers = obj_peek.get('meta', {}).get('num_hidden_layers', None)
        if num_model_layers is None:
            non_neg = [l for l in layers_cfg if l is not None and l >= 0]
            num_model_layers = (max(non_neg) + 8) if non_neg else 32
        xs = layers_cfg
        tag = f"{tag_base}_all_layers"
        plot_layerwise_delta(all_results, xs,
            os.path.join(FIGURES_DIR, f'stage1b_layerwise_delta_{tag}.pdf'),
            title_suffix=tag,
            num_model_layers=num_model_layers)
        all_json = os.path.join(RESULTS_DIR, f'stage1b_layerwise_{tag}.json')
        with open(all_json, 'w') as f:
            json.dump({'layers': layers_cfg,
                       'num_model_layers': num_model_layers,
                       'per_layer': all_results},
                      f, indent=2, ensure_ascii=False)
        print(f"✅ layerwise saved: {all_json}")

    print("\nDone.")


if __name__ == '__main__':
    main()
