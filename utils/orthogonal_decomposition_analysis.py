#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orthogonal_decomposition_analysis.py
======================================
实验 A: 正交分解分析 (Orthogonal Decomposition)
实验 B: 子空间主角分析 (Subspace Principal Angles)

核心目标：
  证明 "H 在对语义能力无损的子空间/方向上容纳了更多视觉信息"

实验 A 思路：
  将 H 分解为 "T 方向分量" 和 "T 正交分量"：
    H = Proj_{S_T}(H) + Proj_{S_T^⊥}(H)
  然后分别计算两个分量与 V 的相似度：
    - cos(H_proj_T, V): T 方向上的视觉信息
    - cos(H_orth_T, V): T 正交方向上的视觉信息
  如果 cos(H_orth_T, V) >> cos(H_proj_T, V)，
  则证明视觉信息主要编码在正交于语义的方向上。

实验 B 思路：
  对 H、T、V 分别做 PCA，计算子空间之间的主角 (Principal Angles)：
  - H 的前 k 个主方向与 T 子空间的重叠度
  - H 的前 k 个主方向与 V 子空间的重叠度
  - 训练后 H 的"新增维度"（dim 5→30 中的增量部分）与 V/T 的重叠
  这能量化"新增方向"的来源。

用法:
  python3 orthogonal_decomposition_analysis.py \\
      --results_dir /path/to/results_v3 \\
      --output_dir /path/to/results_v3/subspace_analysis
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches


# ============================================================
# 工具函数
# ============================================================

def normalize_rows(X):
    """L2 归一化每一行"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return X / norms


def compute_center(X):
    """计算样本集的中心（均值向量）"""
    return X.mean(axis=0)


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def pairwise_cosine_mean(X, Y):
    """计算 X 中每个样本与 Y 中每个样本的余弦相似度的均值"""
    X_norm = normalize_rows(X)
    Y_norm = normalize_rows(Y)
    cos_matrix = X_norm @ Y_norm.T
    return cos_matrix.mean()


def compute_pca_basis(X, k=None, explained_variance_ratio=0.95):
    """
    对 X 做 PCA，返回前 k 个主成分方向。
    如果 k=None，则自动选择解释 explained_variance_ratio 方差的维度数。
    
    返回:
      U: [k, d] — 前 k 个主方向（行向量）
      S: [k] — 对应的奇异值
      explained_var: [k] — 每个方向解释的方差比例
      effective_k: int — 实际使用的维度数
    """
    # 中心化
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # 方差比例
    var_explained = (S ** 2) / (S ** 2).sum()
    cumulative_var = np.cumsum(var_explained)
    
    if k is None:
        # 自动选择维度数
        k = int(np.searchsorted(cumulative_var, explained_variance_ratio) + 1)
        k = min(k, len(S))
    
    return Vt[:k], S[:k], var_explained[:k], k


def project_onto_subspace(X, basis):
    """
    将 X 投影到 basis 张成的子空间上。
    
    参数:
      X: [N, d] — 数据矩阵
      basis: [k, d] — 子空间基（行向量，假设正交归一化）
    
    返回:
      X_proj: [N, d] — X 在子空间上的投影
      X_orth: [N, d] — X 在子空间正交补上的分量
    """
    # 确保 basis 正交归一化
    Q, _ = np.linalg.qr(basis.T)  # Q: [d, k]
    
    # 投影: X_proj = X @ Q @ Q^T
    X_proj = X @ Q @ Q.T
    X_orth = X - X_proj
    
    return X_proj, X_orth


def compute_principal_angles(basis_A, basis_B):
    """
    计算两个子空间之间的主角 (Principal Angles)。
    纯 numpy 实现，不依赖 scipy。
    
    参数:
      basis_A: [k1, d] — 子空间 A 的基
      basis_B: [k2, d] — 子空间 B 的基
    
    返回:
      angles: [min(k1, k2)] — 主角（弧度），从大到小排列
    """
    # 先对 basis 做 QR 分解确保正交归一
    QA, _ = np.linalg.qr(basis_A.T)  # [d, k1]
    QB, _ = np.linalg.qr(basis_B.T)  # [d, k2]
    
    # 计算 QA^T @ QB 的奇异值
    M = QA.T @ QB  # [k1, k2]
    _, sigma, _ = np.linalg.svd(M, full_matrices=False)
    
    # 奇异值 clamp 到 [0, 1] 范围（数值稳定性）
    sigma = np.clip(sigma, 0.0, 1.0)
    
    # 主角 = arccos(sigma)，从大到小排列
    angles = np.arccos(sigma)
    angles = np.sort(angles)[::-1]  # 从大到小
    
    return angles


def compute_subspace_overlap(basis_A, basis_B):
    """
    计算子空间重叠度 (Subspace Overlap)。
    定义为: overlap = (1/k) * sum(cos^2(theta_i))
    其中 theta_i 是主角。
    
    overlap = 1 表示完全重叠，overlap = 0 表示完全正交。
    """
    angles = compute_principal_angles(basis_A, basis_B)
    cos_sq = np.cos(angles) ** 2
    overlap = cos_sq.mean()
    return overlap, angles


# ============================================================
# 实验 A: 正交分解分析
# ============================================================

def experiment_A_orthogonal_decomposition(emb_baseline, emb_ckpt, output_dir):
    """
    实验 A: 正交分解分析
    
    将 H 分解为 "T 方向分量" 和 "T 正交分量"，
    分析视觉信息主要编码在哪个分量中。
    """
    print("\n" + "=" * 70)
    print("  实验 A: 正交分解分析 (Orthogonal Decomposition)")
    print("=" * 70)
    
    results = {}
    
    for model_name, emb in [('baseline', emb_baseline), ('ckpt', emb_ckpt)]:
        print(f"\n  --- {model_name.upper()} ---")
        
        V = emb['vision']    # [N, 4096]
        T = emb['text']      # [N, 4096]
        H = emb['generated'] # [N, 4096]
        
        N, d = H.shape
        print(f"  数据: N={N}, d={d}")
        
        # ============================================================
        # Step 1: 构建 T 子空间的基
        # 使用 T 的 PCA 主方向作为 S_T 的基
        # ============================================================
        T_basis, T_S, T_var, T_k = compute_pca_basis(T, explained_variance_ratio=0.95)
        print(f"  T 子空间: k={T_k} 个主方向 (解释 95% 方差)")
        print(f"    前 5 个方向解释方差: {T_var[:5]}")
        
        # ============================================================
        # Step 2: 将 H 分解为 T 方向分量和 T 正交分量
        # ============================================================
        H_proj_T, H_orth_T = project_onto_subspace(H, T_basis)
        
        # 计算各分量的能量（范数比例）
        H_norm = np.linalg.norm(H, axis=1)
        H_proj_norm = np.linalg.norm(H_proj_T, axis=1)
        H_orth_norm = np.linalg.norm(H_orth_T, axis=1)
        
        energy_ratio_proj = (H_proj_norm ** 2).sum() / (H_norm ** 2).sum()
        energy_ratio_orth = (H_orth_norm ** 2).sum() / (H_norm ** 2).sum()
        
        print(f"\n  H 的分解:")
        print(f"    ||H_proj_T||² / ||H||² = {energy_ratio_proj:.4f} ({energy_ratio_proj*100:.1f}% 能量在 T 子空间)")
        print(f"    ||H_orth_T||² / ||H||² = {energy_ratio_orth:.4f} ({energy_ratio_orth*100:.1f}% 能量在 T⊥ 子空间)")
        
        # ============================================================
        # Step 3: 分别计算两个分量与 V 的相似度
        # ============================================================
        # 中心向量级别
        V_center = compute_center(V)
        H_center = compute_center(H)
        H_proj_center = compute_center(H_proj_T)
        H_orth_center = compute_center(H_orth_T)
        T_center = compute_center(T)
        
        cos_H_V = cosine_similarity(H_center, V_center)
        cos_Hproj_V = cosine_similarity(H_proj_center, V_center)
        cos_Horth_V = cosine_similarity(H_orth_center, V_center)
        
        cos_H_T = cosine_similarity(H_center, T_center)
        cos_Hproj_T = cosine_similarity(H_proj_center, T_center)
        cos_Horth_T = cosine_similarity(H_orth_center, T_center)
        
        print(f"\n  中心向量余弦相似度:")
        print(f"    cos(H, V)       = {cos_H_V:.4f}")
        print(f"    cos(H_proj_T, V) = {cos_Hproj_V:.4f}  (T 方向分量与 V)")
        print(f"    cos(H_orth_T, V) = {cos_Horth_V:.4f}  (T⊥ 方向分量与 V) ← 关键指标")
        print(f"    cos(H, T)       = {cos_H_T:.4f}")
        print(f"    cos(H_proj_T, T) = {cos_Hproj_T:.4f}  (T 方向分量与 T)")
        print(f"    cos(H_orth_T, T) = {cos_Horth_T:.4f}  (T⊥ 方向分量与 T)")
        
        # 逐样本级别的统计
        cos_Horth_V_samples = []
        cos_Hproj_V_samples = []
        for i in range(N):
            if np.linalg.norm(H_orth_T[i]) > 1e-8 and np.linalg.norm(V[i]) > 1e-8:
                cos_Horth_V_samples.append(cosine_similarity(H_orth_T[i], V[i]))
            if np.linalg.norm(H_proj_T[i]) > 1e-8 and np.linalg.norm(V[i]) > 1e-8:
                cos_Hproj_V_samples.append(cosine_similarity(H_proj_T[i], V[i]))
        
        cos_Horth_V_samples = np.array(cos_Horth_V_samples)
        cos_Hproj_V_samples = np.array(cos_Hproj_V_samples)
        
        print(f"\n  逐样本统计 cos(H_orth_T, V):")
        print(f"    mean={cos_Horth_V_samples.mean():.4f}, std={cos_Horth_V_samples.std():.4f}")
        print(f"    p10={np.percentile(cos_Horth_V_samples, 10):.4f}, p90={np.percentile(cos_Horth_V_samples, 90):.4f}")
        print(f"  逐样本统计 cos(H_proj_T, V):")
        print(f"    mean={cos_Hproj_V_samples.mean():.4f}, std={cos_Hproj_V_samples.std():.4f}")
        print(f"    p10={np.percentile(cos_Hproj_V_samples, 10):.4f}, p90={np.percentile(cos_Hproj_V_samples, 90):.4f}")
        
        # ============================================================
        # Step 4: 将 V 也分解，验证 V 主要在 T⊥ 方向
        # ============================================================
        V_proj_T, V_orth_T = project_onto_subspace(V, T_basis)
        V_norm = np.linalg.norm(V, axis=1)
        V_proj_norm = np.linalg.norm(V_proj_T, axis=1)
        V_orth_norm = np.linalg.norm(V_orth_T, axis=1)
        
        V_energy_proj = (V_proj_norm ** 2).sum() / (V_norm ** 2).sum()
        V_energy_orth = (V_orth_norm ** 2).sum() / (V_norm ** 2).sum()
        
        print(f"\n  V 的分解 (验证 V 主要在 T⊥):")
        print(f"    ||V_proj_T||² / ||V||² = {V_energy_proj:.4f} ({V_energy_proj*100:.1f}% 在 T 子空间)")
        print(f"    ||V_orth_T||² / ||V||² = {V_energy_orth:.4f} ({V_energy_orth*100:.1f}% 在 T⊥ 子空间)")
        
        # 保存结果
        results[model_name] = {
            'T_subspace_dim': int(T_k),
            'H_energy_in_T': float(energy_ratio_proj),
            'H_energy_in_T_orth': float(energy_ratio_orth),
            'V_energy_in_T': float(V_energy_proj),
            'V_energy_in_T_orth': float(V_energy_orth),
            'center_cos_H_V': float(cos_H_V),
            'center_cos_Hproj_V': float(cos_Hproj_V),
            'center_cos_Horth_V': float(cos_Horth_V),
            'center_cos_H_T': float(cos_H_T),
            'center_cos_Hproj_T': float(cos_Hproj_T),
            'center_cos_Horth_T': float(cos_Horth_T),
            'sample_cos_Horth_V_mean': float(cos_Horth_V_samples.mean()),
            'sample_cos_Horth_V_std': float(cos_Horth_V_samples.std()),
            'sample_cos_Hproj_V_mean': float(cos_Hproj_V_samples.mean()),
            'sample_cos_Hproj_V_std': float(cos_Hproj_V_samples.std()),
        }
    
    # ============================================================
    # 对比分析
    # ============================================================
    print("\n" + "=" * 70)
    print("  实验 A 对比总结")
    print("=" * 70)
    
    b = results['baseline']
    c = results['ckpt']
    
    print(f"\n  {'指标':<30} {'Baseline':>10} {'Trained':>10} {'Δ':>10}")
    print(f"  {'-'*60}")
    print(f"  {'cos(H_orth_T, V) [中心]':<30} {b['center_cos_Horth_V']:>10.4f} {c['center_cos_Horth_V']:>10.4f} {c['center_cos_Horth_V']-b['center_cos_Horth_V']:>+10.4f}")
    print(f"  {'cos(H_proj_T, V) [中心]':<30} {b['center_cos_Hproj_V']:>10.4f} {c['center_cos_Hproj_V']:>10.4f} {c['center_cos_Hproj_V']-b['center_cos_Hproj_V']:>+10.4f}")
    print(f"  {'cos(H_orth_T, V) [样本均值]':<30} {b['sample_cos_Horth_V_mean']:>10.4f} {c['sample_cos_Horth_V_mean']:>10.4f} {c['sample_cos_Horth_V_mean']-b['sample_cos_Horth_V_mean']:>+10.4f}")
    print(f"  {'cos(H_proj_T, V) [样本均值]':<30} {b['sample_cos_Hproj_V_mean']:>10.4f} {c['sample_cos_Hproj_V_mean']:>10.4f} {c['sample_cos_Hproj_V_mean']-b['sample_cos_Hproj_V_mean']:>+10.4f}")
    print(f"  {'H 能量在 T 子空间':<30} {b['H_energy_in_T']:>10.4f} {c['H_energy_in_T']:>10.4f} {c['H_energy_in_T']-b['H_energy_in_T']:>+10.4f}")
    print(f"  {'H 能量在 T⊥ 子空间':<30} {b['H_energy_in_T_orth']:>10.4f} {c['H_energy_in_T_orth']:>10.4f} {c['H_energy_in_T_orth']-b['H_energy_in_T_orth']:>+10.4f}")
    print(f"  {'cos(H_proj_T, T) [中心]':<30} {b['center_cos_Hproj_T']:>10.4f} {c['center_cos_Hproj_T']:>10.4f} {c['center_cos_Hproj_T']-b['center_cos_Hproj_T']:>+10.4f}")
    
    # 核心结论
    print(f"\n  📋 核心结论:")
    delta_orth = c['center_cos_Horth_V'] - b['center_cos_Horth_V']
    delta_proj = c['center_cos_Hproj_V'] - b['center_cos_Hproj_V']
    if delta_orth > delta_proj and delta_orth > 0.05:
        print(f"    ✅ cos(H_orth_T, V) 增量 ({delta_orth:+.4f}) >> cos(H_proj_T, V) 增量 ({delta_proj:+.4f})")
        print(f"    ✅ 视觉信息主要编码在 T⊥ 方向（正交于语义子空间）")
        print(f"    → 证明: Latent Reasoning 在不损害语义能力的子空间上容纳了视觉信息")
    else:
        print(f"    cos(H_orth_T, V) Δ={delta_orth:+.4f}, cos(H_proj_T, V) Δ={delta_proj:+.4f}")
        print(f"    需要进一步分析...")
    
    # 保存结果
    results_file = os.path.join(output_dir, 'experiment_A_orthogonal_decomposition.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ 结果已保存: {results_file}")
    
    return results


# ============================================================
# 实验 B: 子空间主角分析
# ============================================================

def experiment_B_principal_angles(emb_baseline, emb_ckpt, output_dir):
    """
    实验 B: 子空间主角分析
    
    对 H、T、V 分别做 PCA，计算子空间之间的主角。
    特别关注：训练后 H 的"新增维度"与 V/T 的重叠度。
    """
    print("\n" + "=" * 70)
    print("  实验 B: 子空间主角分析 (Principal Angles)")
    print("=" * 70)
    
    results = {}
    
    for model_name, emb in [('baseline', emb_baseline), ('ckpt', emb_ckpt)]:
        print(f"\n  --- {model_name.upper()} ---")
        
        V = emb['vision']
        T = emb['text']
        H = emb['generated']
        
        # ============================================================
        # Step 1: 对 H、T、V 分别做 PCA
        # ============================================================
        # 使用固定维度以便对比
        k_H = 30   # H 的前 30 个主方向
        k_T = 10   # T 的前 10 个主方向
        k_V = 15   # V 的前 15 个主方向
        
        H_basis, H_S, H_var, _ = compute_pca_basis(H, k=k_H)
        T_basis, T_S, T_var, _ = compute_pca_basis(T, k=k_T)
        V_basis, V_S, V_var, _ = compute_pca_basis(V, k=k_V)
        
        # 有效维度（参与率）
        _, H_S_full, H_var_full, _ = compute_pca_basis(H, k=min(200, H.shape[0]))
        eff_dim_H = np.exp(-np.sum(H_var_full * np.log(H_var_full + 1e-10)))
        
        print(f"  PCA 维度:")
        print(f"    H: 前 {k_H} 个方向, 累计方差 = {H_var.sum():.4f}, 有效维度 = {eff_dim_H:.1f}")
        print(f"    T: 前 {k_T} 个方向, 累计方差 = {T_var.sum():.4f}")
        print(f"    V: 前 {k_V} 个方向, 累计方差 = {V_var.sum():.4f}")
        
        # ============================================================
        # Step 2: 计算子空间之间的主角和重叠度
        # ============================================================
        # H vs T
        overlap_H_T, angles_H_T = compute_subspace_overlap(H_basis, T_basis)
        # H vs V
        overlap_H_V, angles_H_V = compute_subspace_overlap(H_basis, V_basis)
        # T vs V
        overlap_T_V, angles_T_V = compute_subspace_overlap(T_basis, V_basis)
        
        print(f"\n  子空间重叠度 (Subspace Overlap):")
        print(f"    overlap(H, T) = {overlap_H_T:.4f}  (H 与 T 子空间的重叠)")
        print(f"    overlap(H, V) = {overlap_H_V:.4f}  (H 与 V 子空间的重叠)")
        print(f"    overlap(T, V) = {overlap_T_V:.4f}  (T 与 V 子空间的重叠)")
        
        print(f"\n  主角统计 (度):")
        print(f"    H-T: min={np.degrees(angles_H_T.min()):.1f}°, max={np.degrees(angles_H_T.max()):.1f}°, mean={np.degrees(angles_H_T.mean()):.1f}°")
        print(f"    H-V: min={np.degrees(angles_H_V.min()):.1f}°, max={np.degrees(angles_H_V.max()):.1f}°, mean={np.degrees(angles_H_V.mean()):.1f}°")
        print(f"    T-V: min={np.degrees(angles_T_V.min()):.1f}°, max={np.degrees(angles_T_V.max()):.1f}°, mean={np.degrees(angles_T_V.mean()):.1f}°")
        
        # ============================================================
        # Step 3: 分析 H 的"新增维度" (训练后 effective_dim 从 ~6 增加到 ~30)
        # 将 H 的主方向分为"原有方向"(前6个)和"新增方向"(第7-30个)
        # ============================================================
        k_original = 6   # baseline 的有效维度约为 5.7
        k_new_start = k_original
        k_new_end = k_H
        
        H_basis_original = H_basis[:k_original]   # 前 6 个主方向
        H_basis_new = H_basis[k_new_start:k_new_end]  # 第 7-30 个主方向
        
        if H_basis_new.shape[0] > 0:
            # 新增方向与 V 的重叠
            overlap_Hnew_V, angles_Hnew_V = compute_subspace_overlap(H_basis_new, V_basis)
            # 新增方向与 T 的重叠
            overlap_Hnew_T, angles_Hnew_T = compute_subspace_overlap(H_basis_new, T_basis)
            # 原有方向与 V 的重叠
            overlap_Horig_V, angles_Horig_V = compute_subspace_overlap(H_basis_original, V_basis)
            # 原有方向与 T 的重叠
            overlap_Horig_T, angles_Horig_T = compute_subspace_overlap(H_basis_original, T_basis)
            
            print(f"\n  H 主方向分层分析 (原有 dim 1-{k_original} vs 新增 dim {k_original+1}-{k_new_end}):")
            print(f"    overlap(H_original, V) = {overlap_Horig_V:.4f}")
            print(f"    overlap(H_original, T) = {overlap_Horig_T:.4f}")
            print(f"    overlap(H_new, V)      = {overlap_Hnew_V:.4f}  ← 新增方向与 V 的重叠")
            print(f"    overlap(H_new, T)      = {overlap_Hnew_T:.4f}  ← 新增方向与 T 的重叠")
            
            if overlap_Hnew_V > overlap_Hnew_T:
                print(f"    ✅ 新增方向更偏向 V (视觉): {overlap_Hnew_V:.4f} > {overlap_Hnew_T:.4f}")
            else:
                print(f"    新增方向更偏向 T (文本): {overlap_Hnew_T:.4f} > {overlap_Hnew_V:.4f}")
        else:
            overlap_Hnew_V = overlap_Hnew_T = 0
            overlap_Horig_V = overlap_Horig_T = 0
            angles_Hnew_V = angles_Hnew_T = np.array([])
            angles_Horig_V = angles_Horig_T = np.array([])
        
        # ============================================================
        # Step 4: 逐方向分析 — H 的每个主方向与 V/T 的对齐度
        # ============================================================
        direction_alignment_V = []
        direction_alignment_T = []
        
        for i in range(min(k_H, H_basis.shape[0])):
            h_dir = H_basis[i:i+1]  # [1, d]
            # 与 V 子空间的对齐: 投影到 V 子空间后的范数
            proj_on_V = h_dir @ V_basis.T  # [1, k_V]
            align_V = np.linalg.norm(proj_on_V)
            # 与 T 子空间的对齐
            proj_on_T = h_dir @ T_basis.T  # [1, k_T]
            align_T = np.linalg.norm(proj_on_T)
            
            direction_alignment_V.append(float(align_V))
            direction_alignment_T.append(float(align_T))
        
        print(f"\n  H 各主方向与 V/T 子空间的对齐度 (前 10 个方向):")
        print(f"    {'方向':<6} {'align(V)':>10} {'align(T)':>10} {'偏向':>8}")
        for i in range(min(10, len(direction_alignment_V))):
            bias = 'V' if direction_alignment_V[i] > direction_alignment_T[i] else 'T'
            print(f"    PC{i+1:<4} {direction_alignment_V[i]:>10.4f} {direction_alignment_T[i]:>10.4f} {bias:>8}")
        
        # 保存结果
        results[model_name] = {
            'effective_dim_H': float(eff_dim_H),
            'overlap_H_T': float(overlap_H_T),
            'overlap_H_V': float(overlap_H_V),
            'overlap_T_V': float(overlap_T_V),
            'overlap_H_original_V': float(overlap_Horig_V),
            'overlap_H_original_T': float(overlap_Horig_T),
            'overlap_H_new_V': float(overlap_Hnew_V),
            'overlap_H_new_T': float(overlap_Hnew_T),
            'principal_angles_H_T_deg': [float(np.degrees(a)) for a in angles_H_T],
            'principal_angles_H_V_deg': [float(np.degrees(a)) for a in angles_H_V],
            'direction_alignment_V': direction_alignment_V,
            'direction_alignment_T': direction_alignment_T,
            'H_var_explained': [float(v) for v in H_var],
        }
    
    # ============================================================
    # 对比分析
    # ============================================================
    print("\n" + "=" * 70)
    print("  实验 B 对比总结")
    print("=" * 70)
    
    b = results['baseline']
    c = results['ckpt']
    
    print(f"\n  {'指标':<35} {'Baseline':>10} {'Trained':>10} {'Δ':>10}")
    print(f"  {'-'*65}")
    print(f"  {'effective_dim(H)':<35} {b['effective_dim_H']:>10.1f} {c['effective_dim_H']:>10.1f} {c['effective_dim_H']-b['effective_dim_H']:>+10.1f}")
    print(f"  {'overlap(H, V)':<35} {b['overlap_H_V']:>10.4f} {c['overlap_H_V']:>10.4f} {c['overlap_H_V']-b['overlap_H_V']:>+10.4f}")
    print(f"  {'overlap(H, T)':<35} {b['overlap_H_T']:>10.4f} {c['overlap_H_T']:>10.4f} {c['overlap_H_T']-b['overlap_H_T']:>+10.4f}")
    print(f"  {'overlap(H_new[7-30], V)':<35} {b['overlap_H_new_V']:>10.4f} {c['overlap_H_new_V']:>10.4f} {c['overlap_H_new_V']-b['overlap_H_new_V']:>+10.4f}")
    print(f"  {'overlap(H_new[7-30], T)':<35} {b['overlap_H_new_T']:>10.4f} {c['overlap_H_new_T']:>10.4f} {c['overlap_H_new_T']-b['overlap_H_new_T']:>+10.4f}")
    
    # 核心结论
    print(f"\n  📋 核心结论:")
    if c['overlap_H_new_V'] > c['overlap_H_new_T']:
        print(f"    ✅ 训练后 H 的新增方向 (PC7-30) 与 V 的重叠度 ({c['overlap_H_new_V']:.4f})")
        print(f"       > 与 T 的重叠度 ({c['overlap_H_new_T']:.4f})")
        print(f"    → 证明: effective_dim 的增加主要来自'视觉方向'的扩展")
    
    # 保存结果
    results_file = os.path.join(output_dir, 'experiment_B_principal_angles.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ 结果已保存: {results_file}")
    
    return results


# ============================================================
# 可视化
# ============================================================

def visualize_results(results_A, results_B, output_dir):
    """生成实验 A 和 B 的可视化图表"""
    
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # ============================================================
    # 图 1: 正交分解对比 (实验 A)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax_idx, (model_name, label) in enumerate([('baseline', 'Baseline'), ('ckpt', 'Trained (ckpt-1200)')]):
        ax = axes[ax_idx]
        r = results_A[model_name]
        
        # 柱状图: cos(H_proj_T, V) vs cos(H_orth_T, V)
        categories = ['cos(H_proj_T, V)\n(T方向→V)', 'cos(H_orth_T, V)\n(T⊥方向→V)',
                     'cos(H_proj_T, T)\n(T方向→T)', 'cos(H_orth_T, T)\n(T⊥方向→T)']
        values = [r['center_cos_Hproj_V'], r['center_cos_Horth_V'],
                 r['center_cos_Hproj_T'], r['center_cos_Horth_T']]
        colors = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4']
        hatches = ['', '///', '', '///']
        
        bars = ax.bar(range(len(categories)), values, color=colors, 
                     edgecolor='black', linewidth=0.8, alpha=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel('Cosine Similarity', fontsize=11)
        ax.set_title(f'{label}', fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.1, max(values) * 1.3 + 0.05)
        
        # 标注数值
        for i, v in enumerate(values):
            ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('实验 A: 正交分解分析\nH 在 T 方向 vs T⊥ 方向上与 V/T 的相似度', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'fig_A1_orthogonal_decomposition.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 图 2: 能量分布对比 (实验 A)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax_idx, (model_name, label) in enumerate([('baseline', 'Baseline'), ('ckpt', 'Trained')]):
        ax = axes[ax_idx]
        r = results_A[model_name]
        
        # 饼图: H 的能量分布
        sizes = [r['H_energy_in_T'] * 100, r['H_energy_in_T_orth'] * 100]
        labels_pie = [f"在 S_T 中\n({sizes[0]:.1f}%)", f"在 S_T⊥ 中\n({sizes[1]:.1f}%)"]
        colors_pie = ['#4ECDC4', '#A78BFA']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels_pie, colors=colors_pie,
                                          autopct='', startangle=90,
                                          textprops={'fontsize': 11})
        ax.set_title(f'{label}\nH 能量分布', fontsize=12, fontweight='bold')
    
    plt.suptitle('实验 A: H 的能量在 T 子空间 vs T⊥ 子空间的分布', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'fig_A2_energy_distribution.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 图 3: 子空间重叠度对比 (实验 B)
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics = ['overlap(H, V)', 'overlap(H, T)', 'overlap(H_new, V)', 'overlap(H_new, T)']
    baseline_vals = [results_B['baseline']['overlap_H_V'], results_B['baseline']['overlap_H_T'],
                    results_B['baseline']['overlap_H_new_V'], results_B['baseline']['overlap_H_new_T']]
    ckpt_vals = [results_B['ckpt']['overlap_H_V'], results_B['ckpt']['overlap_H_T'],
                results_B['ckpt']['overlap_H_new_V'], results_B['ckpt']['overlap_H_new_T']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
                  color='#95a5a6', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, ckpt_vals, width, label='Trained (ckpt-1200)', 
                  color='#A78BFA', edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('子空间对', fontsize=12)
    ax.set_ylabel('Subspace Overlap', fontsize=12)
    ax.set_title('实验 B: 子空间重叠度对比\n(overlap = mean(cos²(principal_angles)))', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(baseline_vals), max(ckpt_vals)) * 1.3)
    
    # 标注数值
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.4f}',
               ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.4f}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'fig_B1_subspace_overlap.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 图 4: H 各主方向与 V/T 的对齐度 (实验 B)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax_idx, (model_name, label) in enumerate([('baseline', 'Baseline'), ('ckpt', 'Trained')]):
        ax = axes[ax_idx]
        r = results_B[model_name]
        
        align_V = r['direction_alignment_V']
        align_T = r['direction_alignment_T']
        n_dirs = len(align_V)
        
        x = np.arange(1, n_dirs + 1)
        ax.plot(x, align_V, 'o-', color='#FF6B6B', label='align(PC_i, V)', 
               markersize=4, linewidth=1.5)
        ax.plot(x, align_T, 's-', color='#4ECDC4', label='align(PC_i, T)', 
               markersize=4, linewidth=1.5)
        
        # 标记"新增方向"区域
        ax.axvspan(7, n_dirs, alpha=0.1, color='#A78BFA', label='新增方向 (PC7+)')
        ax.axvline(x=6.5, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('H 的主方向 (PC index)', fontsize=11)
        ax.set_ylabel('与 V/T 子空间的对齐度', fontsize=11)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(0.5, n_dirs + 0.5)
    
    plt.suptitle('实验 B: H 各主方向与 V/T 子空间的对齐度\n(对齐度 = ||Proj_{S}(PC_i)||)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'fig_B2_direction_alignment.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 图 5: 综合论证图 — 正交分解示意图
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制概念示意图
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 坐标轴
    ax.annotate('', xy=(3.2, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(0, 3.2), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(3.3, -0.1, r'$\mathcal{S}_T$ (语义方向)', fontsize=12, color='#4ECDC4')
    ax.text(-0.4, 3.3, r'$\mathcal{S}_T^\perp$ (正交方向)', fontsize=12, color='#A78BFA', rotation=90)
    
    # Baseline H 向量 (主要在 T 方向)
    b_A = results_A['baseline']
    c_A = results_A['ckpt']
    
    # 归一化用于示意
    ax.annotate('', xy=(2.0, 0.3), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#95a5a6'))
    ax.text(2.1, 0.4, r'$H_{baseline}$', fontsize=12, color='#95a5a6', fontweight='bold')
    
    # Trained H 向量 (T 方向 + T⊥ 方向)
    ax.annotate('', xy=(1.8, 2.2), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#A78BFA'))
    ax.text(1.9, 2.3, r'$H_{trained}$', fontsize=12, color='#A78BFA', fontweight='bold')
    
    # V 向量 (主要在 T⊥ 方向)
    ax.annotate('', xy=(0.5, 2.5), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#FF6B6B'))
    ax.text(0.6, 2.6, r'$V$', fontsize=12, color='#FF6B6B', fontweight='bold')
    
    # T 向量 (在 T 方向)
    ax.annotate('', xy=(2.5, 0.2), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#4ECDC4'))
    ax.text(2.6, 0.3, r'$T$', fontsize=12, color='#4ECDC4', fontweight='bold')
    
    # 投影虚线
    ax.plot([1.8, 1.8], [0, 2.2], '--', color='#A78BFA', alpha=0.5, linewidth=1)
    ax.plot([0, 1.8], [2.2, 2.2], '--', color='#A78BFA', alpha=0.5, linewidth=1)
    
    # 标注
    ax.text(1.8, -0.2, r'$Proj_{\mathcal{S}_T}(H)$', fontsize=10, ha='center', color='#4ECDC4')
    ax.text(-0.3, 2.2, r'$Proj_{\mathcal{S}_T^\perp}(H)$', fontsize=10, ha='right', color='#A78BFA')
    
    # 添加数值标注
    textbox = (f"核心数据:\n"
              f"cos(H_orth, V): {b_A['center_cos_Horth_V']:.4f} → {c_A['center_cos_Horth_V']:.4f}\n"
              f"cos(H_proj, V): {b_A['center_cos_Hproj_V']:.4f} → {c_A['center_cos_Hproj_V']:.4f}\n"
              f"H 能量在 T⊥: {b_A['H_energy_in_T_orth']*100:.1f}% → {c_A['H_energy_in_T_orth']*100:.1f}%")
    ax.text(0.5, 0.8, textbox, fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           verticalalignment='top')
    
    ax.set_title('正交分解示意图: H 在语义无损子空间上编码视觉信息', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'fig_AB_conceptual_diagram.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 图 6: 方差解释比例对比
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax_idx, (model_name, label) in enumerate([('baseline', 'Baseline'), ('ckpt', 'Trained')]):
        ax = axes[ax_idx]
        var_exp = results_B[model_name]['H_var_explained']
        
        x = np.arange(1, len(var_exp) + 1)
        ax.bar(x, var_exp, color='#A78BFA', edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.axvline(x=6.5, color='red', linestyle='--', alpha=0.7, label='原有/新增分界')
        
        ax.set_xlabel('主成分 (PC index)', fontsize=11)
        ax.set_ylabel('方差解释比例', fontsize=11)
        ax.set_title(f'{label} — H 的 PCA 方差分布\n(effective_dim={results_B[model_name]["effective_dim_H"]:.1f})', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(0.5, 30.5)
    
    plt.suptitle('实验 B: H 的 PCA 方差分布对比\n(训练后方差更均匀分布 → 有效维度增加)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'fig_B3_variance_distribution.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✅ 可视化图表已保存到: {figures_dir}/")
    print(f"    - fig_A1_orthogonal_decomposition.png")
    print(f"    - fig_A2_energy_distribution.png")
    print(f"    - fig_B1_subspace_overlap.png")
    print(f"    - fig_B2_direction_alignment.png")
    print(f"    - fig_B3_variance_distribution.png")
    print(f"    - fig_AB_conceptual_diagram.png")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='正交分解 & 子空间主角分析')
    parser.add_argument('--results_dir', type=str, 
                       default='./modality_manifold_analysis/results_v3',
                       help='包含 embeddings_baseline.pt 和 embeddings_ckpt.pt 的目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: results_dir/subspace_analysis)')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'subspace_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  正交分解 & 子空间主角分析")
    print("  (证明: H 在语义无损的子空间上容纳了更多视觉信息)")
    print("=" * 70)
    
    # 加载 embeddings
    import torch
    
    baseline_path = os.path.join(args.results_dir, 'embeddings_baseline.pt')
    ckpt_path = os.path.join(args.results_dir, 'embeddings_ckpt.pt')
    
    print(f"\n  加载 embeddings:")
    print(f"    Baseline: {baseline_path}")
    print(f"    Ckpt:     {ckpt_path}")
    
    emb_baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
    emb_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    for name, emb in [('baseline', emb_baseline), ('ckpt', emb_ckpt)]:
        print(f"    {name}:")
        for k, v in emb.items():
            print(f"      {k}: shape={v.shape}")
    
    # 运行实验 A
    results_A = experiment_A_orthogonal_decomposition(emb_baseline, emb_ckpt, args.output_dir)
    
    # 运行实验 B
    results_B = experiment_B_principal_angles(emb_baseline, emb_ckpt, args.output_dir)
    
    # 生成可视化
    print("\n" + "=" * 70)
    print("  生成可视化图表")
    print("=" * 70)
    visualize_results(results_A, results_B, args.output_dir)
    
    # ============================================================
    # 最终综合结论
    # ============================================================
    print("\n" + "=" * 70)
    print("  📋 综合结论")
    print("=" * 70)
    
    b_A = results_A['baseline']
    c_A = results_A['ckpt']
    b_B = results_B['baseline']
    c_B = results_B['ckpt']
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  实验 A: 正交分解                                                │
  ├─────────────────────────────────────────────────────────────────┤
  │  cos(H_orth_T, V): {b_A['center_cos_Horth_V']:.4f} → {c_A['center_cos_Horth_V']:.4f} (Δ={c_A['center_cos_Horth_V']-b_A['center_cos_Horth_V']:+.4f})  │
  │  cos(H_proj_T, V): {b_A['center_cos_Hproj_V']:.4f} → {c_A['center_cos_Hproj_V']:.4f} (Δ={c_A['center_cos_Hproj_V']-b_A['center_cos_Hproj_V']:+.4f})  │
  │  H 能量在 T⊥:     {b_A['H_energy_in_T_orth']*100:.1f}% → {c_A['H_energy_in_T_orth']*100:.1f}%                          │
  ├─────────────────────────────────────────────────────────────────┤
  │  实验 B: 子空间主角                                              │
  ├─────────────────────────────────────────────────────────────────┤
  │  overlap(H, V):       {b_B['overlap_H_V']:.4f} → {c_B['overlap_H_V']:.4f} (Δ={c_B['overlap_H_V']-b_B['overlap_H_V']:+.4f})  │
  │  overlap(H_new, V):   {b_B['overlap_H_new_V']:.4f} → {c_B['overlap_H_new_V']:.4f} (Δ={c_B['overlap_H_new_V']-b_B['overlap_H_new_V']:+.4f})  │
  │  overlap(H_new, T):   {b_B['overlap_H_new_T']:.4f} → {c_B['overlap_H_new_T']:.4f} (Δ={c_B['overlap_H_new_T']-b_B['overlap_H_new_T']:+.4f})  │
  │  effective_dim(H):    {b_B['effective_dim_H']:.1f} → {c_B['effective_dim_H']:.1f}                          │
  └─────────────────────────────────────────────────────────────────┘
    """)
    
    # 保存综合结果
    summary = {
        'experiment_A': results_A,
        'experiment_B': results_B,
        'conclusion': {
            'delta_cos_Horth_V': float(c_A['center_cos_Horth_V'] - b_A['center_cos_Horth_V']),
            'delta_cos_Hproj_V': float(c_A['center_cos_Hproj_V'] - b_A['center_cos_Hproj_V']),
            'delta_overlap_H_V': float(c_B['overlap_H_V'] - b_B['overlap_H_V']),
            'delta_overlap_Hnew_V': float(c_B['overlap_H_new_V'] - b_B['overlap_H_new_V']),
            'visual_info_in_orthogonal_subspace': bool(
                (c_A['center_cos_Horth_V'] - b_A['center_cos_Horth_V']) > 
                (c_A['center_cos_Hproj_V'] - b_A['center_cos_Hproj_V'])
            ),
        }
    }
    
    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ 综合结果已保存: {summary_file}")
    print(f"  ✅ 所有图表已保存: {os.path.join(args.output_dir, 'figures')}/")


if __name__ == '__main__':
    main()
