#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1c: Latent Task Specificity Analysis
==========================================
目的：区分两个假设 ——

  (a) Generic Visual Bias:  latent thinker 产出的 h_lat_i 朝向一个固定的
      "视觉方向"，与具体样本的 v_i 无关 ⇒ 任意 v_j 与 h_lat_i 的对齐度都差不多
  (b) Task-Specific Alignment: h_lat_i 真的编码了第 i 个样本独有的视觉
      内容 ⇒ cos(h_lat_i, v_i) 显著大于 cos(h_lat_i, v_j≠i)

方法（基于 400×400 的对齐矩阵 S[i,j] = cos(h_lat_i, v_j)）：

  1. Diagonal vs Off-diagonal mean / std / paired permutation test
     gap = mean(diag) - mean(off_diag)   (越大越 specific)

  2. Retrieval metrics:
       R@1, R@5, R@10 —— 对每个 i，看 v_i 是否排在 S[i,:] 的 top-k
       Median / Mean rank of correct match
       Mean reciprocal rank (MRR)
       Random baseline (uniform over N): R@k = k/N

  3. Specificity Score (normalized):
       z_i = (S[i,i] - mean_{j≠i} S[i,j]) / std_{j≠i} S[i,j]
       平均 z >> 0 且 CI 不过 0 ⇒ specific

  4. Per-sample direction agreement:
       对每个 i：S[i,i] 是否是 S[i,:] 的最大值（% correct pairing）

  5. 同时对 H_native_answer 做全部上面的计算作为对照。

  6. 同时对 L（语言锥）做一遍 —— 检查 latent 对 L 是否也 specific。

  7. （可选） layer-wise：每层都跑一次。

输入:
  --emb_path  modality_analysis/results/embeddings_native_vs_latent.pt
  --latent_step     {0, 1, mean}
  --native_reference {answer, latent_pos}
  --layer_idx        None | int | 'all'

输出:
  results/stage1c_specificity_<tag>.json
  figures/stage1c_specificity_<tag>.pdf
  figures/stage1c_sim_matrix_<tag>.pdf   (sanity heatmap, 可选 --save_heatmap)

作者: modality_analysis pipeline, 2026-05-08
"""
import os
import sys
import json
import argparse
from typing import Optional, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from stage1_stat_anisotropy import l2_normalize
from stage1b_pair_native_vs_latent import load_paired_embeddings

RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
FIGURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==================================================================
# 1. 核心指标
# ==================================================================

def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """S[i,j] = cos(A_i, B_j)"""
    An = l2_normalize(A.astype(np.float32))
    Bn = l2_normalize(B.astype(np.float32))
    return (An @ Bn.T).astype(np.float32)


def diag_offdiag_stats(S: np.ndarray) -> dict:
    """把对角/非对角分开，返回均值/方差/gap/paired t-like z."""
    N = S.shape[0]
    assert S.shape[0] == S.shape[1]
    diag = np.diagonal(S).copy()                       # (N,)
    mask = ~np.eye(N, dtype=bool)
    off = S[mask].reshape(N, N - 1)                    # per-row off-diag, (N, N-1)
    off_row_mean = off.mean(axis=1)                    # (N,)
    off_row_std = off.std(axis=1, ddof=1) + 1e-9       # (N,)
    spec_z = (diag - off_row_mean) / off_row_std       # (N,) "specificity score"

    return {
        'N': int(N),
        'diag_mean': float(diag.mean()),
        'diag_std': float(diag.std(ddof=1)),
        'off_mean': float(off.mean()),
        'off_std': float(off.std(ddof=1)),
        'gap': float(diag.mean() - off.mean()),
        'gap_per_sample_mean': float((diag - off_row_mean).mean()),
        'gap_per_sample_std': float((diag - off_row_mean).std(ddof=1)),
        'spec_z_mean': float(spec_z.mean()),
        'spec_z_std': float(spec_z.std(ddof=1)),
        'spec_z_median': float(np.median(spec_z)),
        'diag': diag,                                  # keep for plot/bootstrap
        'off_row_mean': off_row_mean,
    }


def retrieval_metrics(S: np.ndarray, ks=(1, 5, 10)) -> dict:
    """对每个 i，给定 S[i, :]，正确匹配是 j=i；计算 R@k / rank / MRR."""
    N = S.shape[0]
    # rank of correct match：按 S[i, :] 从大到小排序后，i 的位置（0-based，越小越好）
    order = np.argsort(-S, axis=1)                   # (N, N)
    ranks = np.zeros(N, dtype=np.int64)
    for i in range(N):
        # np.where returns tuple
        pos = np.where(order[i] == i)[0]
        ranks[i] = int(pos[0]) if len(pos) else (N - 1)
    mrr = float((1.0 / (ranks + 1.0)).mean())

    out = {
        'N': N,
        'rank_mean': float(ranks.mean()),
        'rank_median': float(np.median(ranks)),
        'rank_p25': float(np.percentile(ranks, 25)),
        'rank_p75': float(np.percentile(ranks, 75)),
        'MRR': mrr,
        'ranks': ranks,
    }
    for k in ks:
        out[f'R@{k}'] = float((ranks < k).mean())
        out[f'R@{k}_random_baseline'] = float(k) / N
        # one-sided binomial significance under H0 R@k = k/N
        # use normal approx
        p0 = k / N
        p_hat = (ranks < k).mean()
        se = (p0 * (1 - p0) / N) ** 0.5 + 1e-9
        z = (p_hat - p0) / se
        out[f'R@{k}_z_vs_random'] = float(z)
    return out


def paired_permutation_gap(
    A: np.ndarray, B: np.ndarray,
    n_perm: int = 2000, seed: int = 42,
) -> dict:
    """
    对 S = cos(A, B) 的 diag - off_diag gap 做 paired permutation test.
    H0: A 和 B 之间没有配对 (行/列置换不改变分布).
    做法: 每次随机打乱 B 的行顺序, 重新算 gap*, 比较 |gap*| >= |gap_obs|.
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    S_obs = cosine_matrix(A, B)
    gap_obs = float(np.diagonal(S_obs).mean() - S_obs[~np.eye(N, dtype=bool)].mean())

    # 高效做法: 不重复算 cos matrix, 只置换行索引的对角位置
    # perm_gap* = mean_{i} S[i, pi(i)]  - mean_{i,j: j != pi(i)} S[i, j]
    # 近似上 off_diag 均值几乎不变, 主导项是 mean S[i, pi(i)]
    # 正式起见, 直接对 S_obs 的行按置换 pi 重排列: S' = S_obs[:, pi], 然后算 gap
    gaps = []
    for _ in range(n_perm):
        pi = rng.permutation(N)
        S_p = S_obs[:, pi]
        gaps.append(float(np.diagonal(S_p).mean() - S_p[~np.eye(N, dtype=bool)].mean()))
    gaps = np.asarray(gaps)
    p_val = (np.sum(np.abs(gaps) >= abs(gap_obs)) + 1) / (n_perm + 1)

    return {
        'gap_obs': gap_obs,
        'gap_null_mean': float(gaps.mean()),
        'gap_null_std': float(gaps.std(ddof=1)),
        'gap_null_p95_abs': float(np.percentile(np.abs(gaps), 95)),
        'n_perm': n_perm,
        'paired_p_value': float(p_val),
    }


def bootstrap_gap_ci(
    S: np.ndarray, n_boot: int = 1000, seed: int = 42,
) -> dict:
    """对 gap = mean(diag) - mean(off) 做 sample-level bootstrap (resample rows)."""
    rng = np.random.default_rng(seed)
    N = S.shape[0]
    gaps = []
    diags = []
    offs = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        Sb = S[np.ix_(idx, idx)]
        dg = float(np.diagonal(Sb).mean())
        og = float(Sb[~np.eye(N, dtype=bool)].mean())
        diags.append(dg); offs.append(og); gaps.append(dg - og)
    gaps = np.asarray(gaps)
    return {
        'gap_mean': float(gaps.mean()),
        'gap_ci_low': float(np.percentile(gaps, 2.5)),
        'gap_ci_high': float(np.percentile(gaps, 97.5)),
        'diag_ci_low': float(np.percentile(diags, 2.5)),
        'diag_ci_high': float(np.percentile(diags, 97.5)),
        'off_ci_low': float(np.percentile(offs, 2.5)),
        'off_ci_high': float(np.percentile(offs, 97.5)),
        'n_boot': n_boot,
    }


# ==================================================================
# 2. 主分析: 对一组 (H, Anchor) 计算所有指标
# ==================================================================

def analyze_pair(
    H: np.ndarray, Anchor: np.ndarray, name: str = '',
    n_boot: int = 1000, n_perm: int = 2000, seed: int = 42,
    verbose: bool = True,
) -> dict:
    assert H.shape == Anchor.shape, f"{name}: H {H.shape} vs Anchor {Anchor.shape}"
    S = cosine_matrix(H, Anchor)                    # (N, N)

    stats = diag_offdiag_stats(S)
    ret = retrieval_metrics(S)
    ci = bootstrap_gap_ci(S, n_boot=n_boot, seed=seed)
    perm = paired_permutation_gap(H, Anchor, n_perm=n_perm, seed=seed)

    out = {
        'name': name,
        'diag_offdiag': {k: v for k, v in stats.items() if k not in ('diag', 'off_row_mean')},
        'retrieval': {k: v for k, v in ret.items() if k != 'ranks'},
        'ci': ci,
        'perm': perm,
    }

    if verbose:
        print(f"\n  [{name}]")
        print(f"    diag_mean = {stats['diag_mean']:+.4f}  off_mean = {stats['off_mean']:+.4f}"
              f"  gap = {stats['gap']:+.4f}  (95% CI [{ci['gap_ci_low']:+.4f}, {ci['gap_ci_high']:+.4f}])")
        print(f"    spec_z   = mean {stats['spec_z_mean']:+.3f}  median {stats['spec_z_median']:+.3f}")
        print(f"    R@1 = {ret['R@1']:.3f} (random {ret['R@1_random_baseline']:.3f}, z={ret['R@1_z_vs_random']:+.2f})")
        print(f"    R@5 = {ret['R@5']:.3f} (random {ret['R@5_random_baseline']:.3f}, z={ret['R@5_z_vs_random']:+.2f})")
        print(f"    R@10= {ret['R@10']:.3f} (random {ret['R@10_random_baseline']:.3f}, z={ret['R@10_z_vs_random']:+.2f})")
        print(f"    median rank = {ret['rank_median']:.0f} / {ret['N']}   MRR = {ret['MRR']:.4f}")
        print(f"    paired perm: gap_obs = {perm['gap_obs']:+.4f}, p = {perm['paired_p_value']:.4g}")

    # stash 一些原始数组给图
    out['_raw'] = {
        'diag': stats['diag'].tolist(),
        'off_row_mean': stats['off_row_mean'].tolist(),
        'spec_z': ((stats['diag'] - stats['off_row_mean'])
                   / (np.asarray(stats['off_row_mean']).std(ddof=1) + 1e-9)).tolist(),
    }
    return out


def verdict_a_or_b(res_latent_to_V: dict) -> str:
    """给出简易判决：(a) generic visual bias vs (b) task-specific alignment"""
    r1 = res_latent_to_V['retrieval']['R@1']
    r1_rand = res_latent_to_V['retrieval']['R@1_random_baseline']
    z = res_latent_to_V['retrieval']['R@1_z_vs_random']
    gap = res_latent_to_V['diag_offdiag']['gap']
    ci_low = res_latent_to_V['ci']['gap_ci_low']
    p = res_latent_to_V['perm']['paired_p_value']
    spec_z = res_latent_to_V['diag_offdiag']['spec_z_mean']

    strong_b = (r1 > 10 * r1_rand) and (ci_low > 0) and (p < 0.01) and (spec_z > 1.0)
    moderate_b = (r1 > 3 * r1_rand) and (ci_low > 0) and (p < 0.05)
    if strong_b:
        return "(b) STRONG task-specific alignment — latent encodes sample-specific visual info"
    if moderate_b:
        return "(b) MODERATE task-specific alignment — some sample-specific signal, but not dominant"
    if (ci_low <= 0) or (p > 0.1):
        return "(a) generic visual bias — latent direction is largely sample-independent"
    return "inconclusive — partial alignment, need more data / other probes"


# ==================================================================
# 3. 画图
# ==================================================================

def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_specificity(
    bundle: dict, out_path: str, title_suffix: str = '',
):
    plt = _setup_mpl()
    names = list(bundle.keys())
    assert len(names) >= 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Stage-1c Task Specificity  ({title_suffix})',
                 fontsize=13, fontweight='bold')

    # Panel A: diag vs off bars with CI
    ax = axes[0, 0]
    diag_m = [bundle[n]['diag_offdiag']['diag_mean'] for n in names]
    off_m = [bundle[n]['diag_offdiag']['off_mean'] for n in names]
    gap_m = [bundle[n]['ci']['gap_mean'] for n in names]
    gap_err_low = [bundle[n]['ci']['gap_mean'] - bundle[n]['ci']['gap_ci_low'] for n in names]
    gap_err_high = [bundle[n]['ci']['gap_ci_high'] - bundle[n]['ci']['gap_mean'] for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, diag_m, w, label='diag (matched)',
           color='#FF6B35', edgecolor='black', alpha=0.85)
    ax.bar(x + w/2, off_m, w, label='off-diag (mismatched)',
           color='#4ECDC4', edgecolor='black', alpha=0.85)
    for i, (d, o) in enumerate(zip(diag_m, off_m)):
        ax.text(i - w/2, d + 0.002, f'{d:+.3f}', ha='center', fontsize=9)
        ax.text(i + w/2, o + 0.002, f'{o:+.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=9)
    ax.set_ylabel('mean cosine similarity')
    ax.set_title('Matched vs Mismatched alignment')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Panel B: gap with bootstrap CI + perm p
    ax = axes[0, 1]
    ax.bar(x, gap_m, yerr=[gap_err_low, gap_err_high], capsize=6,
           color='#7B2FBE', edgecolor='black', alpha=0.85)
    for i, n in enumerate(names):
        p = bundle[n]['perm']['paired_p_value']
        g = gap_m[i]
        ax.text(i, g + max(gap_err_high[i], 0.002),
                f'gap={g:+.4f}\np={p:.3g}',
                ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=9)
    ax.set_ylabel('gap = mean(diag) - mean(off-diag)')
    ax.set_title('Specificity gap (95% CI + paired-perm p)')
    ax.grid(axis='y', alpha=0.3)

    # Panel C: R@k curve
    ax = axes[1, 0]
    for n, col in zip(names, ['#FF6B35', '#4ECDC4', '#7B2FBE', '#1F77B4', '#FFC857']):
        ks = [1, 5, 10]
        rs = [bundle[n]['retrieval'][f'R@{k}'] for k in ks]
        rand = [bundle[n]['retrieval'][f'R@{k}_random_baseline'] for k in ks]
        ax.plot(ks, rs, '-o', color=col, linewidth=2, markersize=8, label=n)
        ax.plot(ks, rand, '--', color=col, alpha=0.4)
    ax.set_xlabel('k'); ax.set_ylabel('R@k (matched recall)')
    ax.set_title('Retrieval Recall@k  (dashed = random baseline)')
    ax.set_xticks([1, 5, 10])
    ax.legend(); ax.grid(alpha=0.3)

    # Panel D: spec_z histogram
    ax = axes[1, 1]
    for n, col in zip(names, ['#FF6B35', '#4ECDC4', '#7B2FBE', '#1F77B4', '#FFC857']):
        zs = np.asarray(bundle[n]['_raw']['spec_z'])
        ax.hist(zs, bins=40, alpha=0.5, label=f'{n} (mean={zs.mean():+.2f})',
                color=col, edgecolor='black')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('spec z = (S_ii - mean off) / std off')
    ax.set_ylabel('count')
    ax.set_title('Per-sample specificity score distribution')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ figure saved: {out_path}")


def plot_sim_heatmap(S: np.ndarray, out_path: str, title: str, max_show: int = 120):
    plt = _setup_mpl()
    n = min(max_show, S.shape[0])
    Sshow = S[:n, :n]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Sshow, cmap='RdBu_r',
                   vmin=-float(np.abs(Sshow).max()),
                   vmax=float(np.abs(Sshow).max()))
    plt.colorbar(im, ax=ax, label='cosine similarity')
    ax.set_title(f'{title}  (first {n}x{n})')
    ax.set_xlabel('anchor idx j')
    ax.set_ylabel('latent/native idx i')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"✅ heatmap saved: {out_path}")


# ==================================================================
# 4. CLI
# ==================================================================

def run_one_layer(
    V, L, H_latent, H_native, *,
    n_boot: int, n_perm: int, seed: int, verbose: bool,
    save_heatmap: bool, tag: str,
):
    print("\n" + "=" * 64)
    print(f"  === {tag} ===")
    print("=" * 64)

    bundle = {}
    # 核心四个对：latent↔V, native↔V, latent↔L, native↔L
    bundle['latent↔V'] = analyze_pair(H_latent, V, 'latent↔V',
        n_boot=n_boot, n_perm=n_perm, seed=seed, verbose=verbose)
    bundle['native↔V'] = analyze_pair(H_native, V, 'native↔V',
        n_boot=n_boot, n_perm=n_perm, seed=seed, verbose=verbose)
    bundle['latent↔L'] = analyze_pair(H_latent, L, 'latent↔L',
        n_boot=n_boot, n_perm=n_perm, seed=seed, verbose=verbose)
    bundle['native↔L'] = analyze_pair(H_native, L, 'native↔L',
        n_boot=n_boot, n_perm=n_perm, seed=seed, verbose=verbose)

    verdict = verdict_a_or_b(bundle['latent↔V'])
    print(f"\n  === verdict for latent↔V ===\n  {verdict}")

    if save_heatmap:
        S = cosine_matrix(H_latent, V)
        plot_sim_heatmap(S,
            os.path.join(FIGURES_DIR, f'stage1c_sim_matrix_latentV_{tag}.pdf'),
            title=f'cos(H_latent, V)  {tag}')

    return bundle, verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', required=True)
    parser.add_argument('--native_reference', default='answer',
                        choices=['latent_pos', 'answer'])
    parser.add_argument('--latent_step', default='0',
                        choices=['0', '1', 'mean'])
    parser.add_argument('--layer_idx', default=None,
                        help="None (last layer) | int | 'all'")
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--n_perm', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_heatmap', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("  Stage-1c Task Specificity Analysis")
    print("=" * 70)

    # 解析 layer_idx
    if args.layer_idx is None or (isinstance(args.layer_idx, str) and args.layer_idx.lower() == 'none'):
        layer_modes = [None]
    elif args.layer_idx == 'all':
        obj_peek = torch.load(args.emb_path, map_location='cpu', weights_only=False)
        layers_cfg = obj_peek.get('meta', {}).get('layers', None) or []
        if not layers_cfg:
            raise RuntimeError("embeddings 未抽取 layer-wise, 无法 --layer_idx all")
        layer_modes = list(range(len(layers_cfg)))
    else:
        layer_modes = [int(args.layer_idx)]

    tag_base = f"{args.latent_step}_{args.native_reference}"
    all_layer_bundles = []

    for lm in layer_modes:
        V, L, H_latent, H_native, meta = load_paired_embeddings(
            args.emb_path, native_reference=args.native_reference,
            latent_step=args.latent_step, layer_idx=lm, verbose=True,
        )
        # 解析真实层号
        real_layer = None
        if lm is not None:
            obj_peek = torch.load(args.emb_path, map_location='cpu', weights_only=False)
            layers_cfg = obj_peek.get('meta', {}).get('layers', None) or []
            if 0 <= lm < len(layers_cfg):
                real_layer = int(layers_cfg[lm])
            else:
                real_layer = int(lm)
        tag = f"{tag_base}_layer{real_layer}" if real_layer is not None else f"{tag_base}_last"

        bundle, verdict = run_one_layer(
            V, L, H_latent, H_native,
            n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
            verbose=True, save_heatmap=args.save_heatmap, tag=tag,
        )

        out_json = os.path.join(RESULTS_DIR, f'stage1c_specificity_{tag}.json')
        # 在写入前去掉 _raw 里的 list (太长)，保留关键数组长度/统计
        _to_dump = {
            'meta': {
                'n_samples': int(H_latent.shape[0]),
                'dim': int(H_latent.shape[1]),
                'n_boot': args.n_boot,
                'n_perm': args.n_perm,
                'seed': args.seed,
                'layer_idx_scan': lm,
                'layer_idx': real_layer,
                'native_reference': args.native_reference,
                'latent_step': args.latent_step,
                'tag': tag,
            },
            'verdict_latent_to_V': verdict,
            'pairs': {},
        }
        for k, v in bundle.items():
            _to_dump['pairs'][k] = {kk: vv for kk, vv in v.items() if kk != '_raw'}
            # 保留 spec_z 数组（400 个样本不大）
            _to_dump['pairs'][k]['spec_z_per_sample'] = v['_raw']['spec_z']
        with open(out_json, 'w') as f:
            json.dump(_to_dump, f, indent=2, ensure_ascii=False)
        print(f"✅ metrics saved: {out_json}")

        # 单层图
        plot_specificity(bundle,
            os.path.join(FIGURES_DIR, f'stage1c_specificity_{tag}.pdf'),
            title_suffix=tag)

        all_layer_bundles.append({
            'layer_idx': real_layer, 'layer_scan': lm,
            'bundle': bundle, 'verdict': verdict,
        })

    # layer-wise 汇总图
    if len(all_layer_bundles) > 1:
        plt = _setup_mpl()
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # 真实层号 + final 映射
        real_layers_raw = [b['layer_idx'] for b in all_layer_bundles]
        non_neg = [l for l in real_layers_raw if l is not None and l >= 0]
        num_hidden = (max(non_neg) + 8) if non_neg else 32
        xs_numeric = []
        labels = []
        for l in real_layers_raw:
            if l is None or l == -1:
                xs_numeric.append(float(num_hidden)); labels.append('final')
            else:
                xs_numeric.append(float(l)); labels.append(str(l))
        order = list(np.argsort(xs_numeric))
        xs = [xs_numeric[i] for i in order]
        xtick_labels = [labels[i] for i in order]
        def _reorder(seq):
            return [seq[i] for i in order]

        # Δgap_latentV - Δgap_nativeV
        gap_latV = _reorder([b['bundle']['latent↔V']['diag_offdiag']['gap'] for b in all_layer_bundles])
        gap_natV = _reorder([b['bundle']['native↔V']['diag_offdiag']['gap'] for b in all_layer_bundles])
        gap_latL = _reorder([b['bundle']['latent↔L']['diag_offdiag']['gap'] for b in all_layer_bundles])
        gap_natL = _reorder([b['bundle']['native↔L']['diag_offdiag']['gap'] for b in all_layer_bundles])

        ax = axes[0]
        ax.plot(xs, gap_latV, '-o', color='#FF6B35', linewidth=2, markersize=7, label='latent↔V')
        ax.plot(xs, gap_natV, '--o', color='#FF6B35', linewidth=2, markersize=6, alpha=0.5, label='native↔V')
        ax.plot(xs, gap_latL, '-s', color='#4ECDC4', linewidth=2, markersize=7, label='latent↔L')
        ax.plot(xs, gap_natL, '--s', color='#4ECDC4', linewidth=2, markersize=6, alpha=0.5, label='native↔L')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(xtick_labels)
        ax.set_xlabel('layer'); ax.set_ylabel('gap = mean(diag) - mean(off)')
        ax.set_title('Specificity gap across layers')
        ax.legend(); ax.grid(alpha=0.3)

        # R@1
        r1_latV = _reorder([b['bundle']['latent↔V']['retrieval']['R@1'] for b in all_layer_bundles])
        r1_natV = _reorder([b['bundle']['native↔V']['retrieval']['R@1'] for b in all_layer_bundles])
        r1_rand = _reorder([b['bundle']['latent↔V']['retrieval']['R@1_random_baseline'] for b in all_layer_bundles])
        ax = axes[1]
        ax.plot(xs, r1_latV, '-o', color='#FF6B35', linewidth=2, markersize=7, label='latent↔V R@1')
        ax.plot(xs, r1_natV, '--o', color='#FF6B35', linewidth=2, markersize=6, alpha=0.5, label='native↔V R@1')
        ax.plot(xs, r1_rand, ':', color='gray', linewidth=1.5, label='random (k/N)')
        ax.set_xticks(xs); ax.set_xticklabels(xtick_labels)
        ax.set_xlabel('layer'); ax.set_ylabel('R@1')
        ax.set_title('Matched retrieval R@1 across layers')
        ax.legend(); ax.grid(alpha=0.3)

        # spec_z_mean
        sz_latV = _reorder([b['bundle']['latent↔V']['diag_offdiag']['spec_z_mean'] for b in all_layer_bundles])
        sz_natV = _reorder([b['bundle']['native↔V']['diag_offdiag']['spec_z_mean'] for b in all_layer_bundles])
        sz_latL = _reorder([b['bundle']['latent↔L']['diag_offdiag']['spec_z_mean'] for b in all_layer_bundles])
        sz_natL = _reorder([b['bundle']['native↔L']['diag_offdiag']['spec_z_mean'] for b in all_layer_bundles])
        ax = axes[2]
        ax.plot(xs, sz_latV, '-o', color='#FF6B35', linewidth=2, markersize=7, label='latent↔V')
        ax.plot(xs, sz_natV, '--o', color='#FF6B35', linewidth=2, markersize=6, alpha=0.5, label='native↔V')
        ax.plot(xs, sz_latL, '-s', color='#4ECDC4', linewidth=2, markersize=7, label='latent↔L')
        ax.plot(xs, sz_natL, '--s', color='#4ECDC4', linewidth=2, markersize=6, alpha=0.5, label='native↔L')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(xtick_labels)
        ax.set_xlabel('layer'); ax.set_ylabel('mean spec z')
        ax.set_title('Per-sample spec z across layers')
        ax.legend(); ax.grid(alpha=0.3)

        fig.suptitle(f'Stage-1c Layerwise Task Specificity  ({tag_base})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_fig = os.path.join(FIGURES_DIR, f'stage1c_layerwise_{tag_base}.pdf')
        plt.savefig(out_fig, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"✅ layerwise figure saved: {out_fig}")

        # layerwise JSON
        all_json = os.path.join(RESULTS_DIR, f'stage1c_layerwise_{tag_base}.json')
        dump_all = {
            'layers_real': real_layers_raw,
            'num_hidden_layers_est': num_hidden,
            'per_layer': []
        }
        for b in all_layer_bundles:
            slim = {'layer_idx': b['layer_idx'], 'verdict': b['verdict'], 'pairs': {}}
            for k, v in b['bundle'].items():
                slim['pairs'][k] = {kk: vv for kk, vv in v.items() if kk != '_raw'}
            dump_all['per_layer'].append(slim)
        with open(all_json, 'w') as f:
            json.dump(dump_all, f, indent=2, ensure_ascii=False)
        print(f"✅ layerwise metrics saved: {all_json}")

    print("\nDone.")


if __name__ == '__main__':
    main()
