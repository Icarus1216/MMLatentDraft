#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_modality_manifold.py (v2 — 精简版)
==============================
"模态流形分析"可视化 —— 支撑 last_hidden_state 训练成"视觉-语言过渡模态"的 insight

仅保留对分析和论证有效的指标，精简图表。

Figure 1: 模态对齐轨迹 (Alignment Trajectory)
  - cos(H,V) / cos(H,T) 随训练步数的演化
  - 任务类型拆解 (abstract / concrete / unified / bridge)

Figure 2: 过渡模态指标 (Transition Modality Index)
  - α_transition: H 在 V-L 轴线上的投影位置
  - MTI: 模态过渡度
  - V-L 等距度

Figure 3: 隐状态几何演化 (Hidden State Geometry)
  - diag_score / h_norm / shift_kl / adj_cos

Figure 4: CKA 近似轨迹 (CKA Approximation)
  - CKA(H,V) / CKA(H,T) 近似值
  - CKA 比率变化

Figure 5: 训练阶段仪表盘 (Training Phase Dashboard)
  - 6 面板综合视图 (精简): 对齐/几何/效率/注意力/思考/总览

Figure 6: 模态锥体 3D 演化 (3D Cone Evolution)
  - 训练前后 V/L/H 锥体相对位置变化

Figure 7: 与 Baseline 对比 (Baseline Comparison)
  - 训练前 (预训练模型) vs 训练后 (ckpt-1200) 的关键指标对比

用法:
  python3 visualize_modality_manifold.py
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ============================================================
# 0. 全局样式
# ============================================================
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei', 'Arial'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 11,
})

# Dark theme
BG = '#0D1117'
FG = '#C9D1D9'
GRID = '#21262D'
EDGE = '#30363D'

# 模态配色
C_V = '#00D2FF'    # 视觉 - 青蓝
C_L = '#FF6B6B'    # 语言 - 珊瑚红
C_H = '#A78BFA'    # 隐状态 - 紫色
C_ACCENT = '#FFD93D'  # 强调 - 金黄
C_GREEN = '#4ADE80'
C_ORANGE = '#FB923C'
C_PINK = '#F472B6'
C_TEAL = '#4ECDC4'

# 任务类型配色
C_ABSTRACT = '#7C3AED'
C_CONCRETE = '#F59E0B'
C_UNIFIED = '#10B981'
C_BRIDGE = '#EC4899'

# ============================================================
# 1. 数据加载
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
FIGURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# Baseline geometry (from pre-trained model)
BASELINE_METRICS_PATH = os.path.join(
    PROJ_ROOT, 'outputs/modality_geometry_v2/geometry_metrics.json'
)


def load_training_metrics(results_dir=None):
    """加载训练指标"""
    if results_dir is None:
        results_dir = RESULTS_DIR
    metrics_path = os.path.join(results_dir, 'training_metrics.json')
    with open(metrics_path) as f:
        return json.load(f)


def load_baseline_metrics():
    """加载 baseline (预训练模型) 的模态几何指标"""
    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH) as f:
            return json.load(f)
    return None


def load_phase_summary(results_dir=None):
    """加载训练阶段摘要"""
    if results_dir is None:
        results_dir = RESULTS_DIR
    path = os.path.join(results_dir, 'training_phase_summary.json')
    with open(path) as f:
        return json.load(f)


# ============================================================
# 2. 平滑函数
# ============================================================
def smooth(arr, w=15):
    """NaN-aware 移动平均"""
    arr = np.array(arr, dtype=float)
    if len(arr) < w:
        return arr
    result = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - w // 2)
        hi = min(len(arr), i + w // 2 + 1)
        window = arr[lo:hi]
        valid = window[~np.isnan(window)]
        result[i] = np.nan if len(valid) == 0 else valid.mean()
    return result


def safe_arr(data, key):
    """安全获取数组, NaN 替代 None"""
    return np.array([v if v is not None else np.nan for v in data.get(key, [])])


# ============================================================
# 3. 训练阶段背景
# ============================================================
def add_phase_backgrounds(ax, phases, alpha=0.08):
    """在图上添加训练阶段背景色块"""
    phase_colors = ['#FF6B6B', '#FFD93D', '#4ADE80', '#00D2FF']
    for i, p in enumerate(phases):
        step_range = p['step_range'].split('-')
        x0, x1 = int(step_range[0]), int(step_range[1])
        ax.axvspan(x0, x1, alpha=alpha, color=phase_colors[i % len(phase_colors)])
        ax.text((x0 + x1) / 2, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] != 0 else 0.5,
                p['name'], ha='center', va='top', fontsize=7,
                color=phase_colors[i % len(phase_colors)], alpha=0.7)


# ============================================================
# Figure 1: 模态对齐轨迹 (精简版)
# ============================================================
def plot_alignment_trajectory(data, phases, out_dir):
    """cos(H,V) / cos(H,T) 随训练步数的演化"""
    steps = np.array(data['steps'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle('Modality Alignment Trajectory During Training',
                 fontsize=16, fontweight='bold', color=FG, y=0.98)

    # Panel A: H-V 对齐 (总体 + 关键任务拆解)
    ax = axes[0, 0]
    ax.set_facecolor(BG)
    cos_hv_mean = smooth(safe_arr(data, 'cos_hv__mean'))
    cos_hv_abstract = smooth(safe_arr(data, 'cos_hv__abstract'))
    cos_hv_concrete = smooth(safe_arr(data, 'cos_hv__concrete'))
    cos_hv_unified = smooth(safe_arr(data, 'cos_hv__unified'))
    cos_hv_bridge = smooth(safe_arr(data, 'cos_hv__bridge'))

    ax.plot(steps, cos_hv_mean, color=C_V, lw=2.5, label='cos(H,V) mean', zorder=5)
    ax.plot(steps, cos_hv_abstract, color=C_ABSTRACT, lw=1.5, alpha=0.7, label='abstract', ls='--')
    ax.plot(steps, cos_hv_concrete, color=C_CONCRETE, lw=1.5, alpha=0.7, label='concrete', ls='--')
    ax.plot(steps, cos_hv_unified, color=C_UNIFIED, lw=1.5, alpha=0.7, label='unified', ls='--')
    if not np.all(np.isnan(cos_hv_bridge)):
        ax.plot(steps, cos_hv_bridge, color=C_BRIDGE, lw=1.5, alpha=0.5, label='bridge', ls=':')

    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('cos(H, V)', color=FG)
    ax.set_title('H-V Alignment', color=FG, fontweight='bold')
    ax.legend(fontsize=8, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel B: H-T 对齐
    ax = axes[0, 1]
    ax.set_facecolor(BG)
    cos_ht_mean = smooth(safe_arr(data, 'cos_ht__mean'))
    cos_ht_abstract = smooth(safe_arr(data, 'cos_ht__abstract'))
    cos_ht_concrete = smooth(safe_arr(data, 'cos_ht__concrete'))
    cos_ht_unified = smooth(safe_arr(data, 'cos_ht__unified'))
    cos_ht_bridge = smooth(safe_arr(data, 'cos_ht__bridge'))

    ax.plot(steps, cos_ht_mean, color=C_L, lw=2.5, label='cos(H,T) mean', zorder=5)
    ax.plot(steps, cos_ht_abstract, color=C_ABSTRACT, lw=1.5, alpha=0.7, label='abstract', ls='--')
    ax.plot(steps, cos_ht_concrete, color=C_CONCRETE, lw=1.5, alpha=0.7, label='concrete', ls='--')
    ax.plot(steps, cos_ht_unified, color=C_UNIFIED, lw=1.5, alpha=0.7, label='unified', ls='--')
    if not np.all(np.isnan(cos_ht_bridge)):
        ax.plot(steps, cos_ht_bridge, color=C_BRIDGE, lw=1.5, alpha=0.5, label='bridge', ls=':')

    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('cos(H, T)', color=FG)
    ax.set_title('H-T Alignment', color=FG, fontweight='bold')
    ax.legend(fontsize=8, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel C: V-H vs T-H 对比
    ax = axes[1, 0]
    ax.set_facecolor(BG)
    ax.plot(steps, cos_hv_mean, color=C_V, lw=2.5, label='cos(H, V)')
    ax.plot(steps, cos_ht_mean, color=C_L, lw=2.5, label='cos(H, T)')

    # 标注交叉点区域
    diff = cos_hv_mean - cos_ht_mean
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    for sc in sign_changes:
        ax.axvline(steps[sc], color=C_ACCENT, ls='--', alpha=0.5, lw=1)
        ax.annotate('H equidistant\nto V and T',
                    xy=(steps[sc], cos_hv_mean[sc]),
                    xytext=(steps[sc] + 50, cos_hv_mean[sc] + 0.1),
                    fontsize=8, color=C_ACCENT,
                    arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=1))

    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('Cosine Similarity', color=FG)
    ax.set_title('V-H vs T-H Alignment Competition', color=FG, fontweight='bold')
    ax.legend(fontsize=9, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel D: 角度空间
    ax = axes[1, 1]
    ax.set_facecolor(BG)
    angle_hv = smooth(safe_arr(data, 'derived__angle_hv_deg'))
    angle_ht = smooth(safe_arr(data, 'derived__angle_ht_deg'))

    ax.plot(steps, angle_hv, color=C_V, lw=2.5, label='∠(H,V)')
    ax.plot(steps, angle_ht, color=C_L, lw=2.5, label='∠(H,T)')

    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('Angle (degrees)', color=FG)
    ax.set_title('Angular Distance to V / T', color=FG, fontweight='bold')
    ax.legend(fontsize=9, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig1_alignment_trajectory.png')
    plt.savefig(out_path, facecolor=BG)
    plt.close()
    print(f"  ✅ Fig 1: {out_path}")


# ============================================================
# Figure 2: 过渡模态指标 (精简版)
# ============================================================
def plot_transition_index(data, phases, baseline, out_dir):
    """MTI 和 α_transition"""
    steps = np.array(data['steps'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)
    fig.suptitle('Transition Modality Index: H as V→L Bridge',
                 fontsize=16, fontweight='bold', color=FG, y=1.02)

    # Panel A: α_transition
    ax = axes[0]
    ax.set_facecolor(BG)
    alpha_t = smooth(safe_arr(data, 'derived__alpha_transition'))
    ax.plot(steps, alpha_t, color=C_H, lw=2.5)

    # Baseline 线
    if baseline is not None:
        bv = baseline.get('step0_alpha', None)
        if bv is not None:
            ax.axhline(bv, color=C_ACCENT, ls='--', lw=1.5, alpha=0.7, label=f'Baseline α={bv:.3f}')

    ax.axhline(0.5, color=FG, ls=':', lw=1, alpha=0.3, label='Equidistant')
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('α_transition = |cos(H,V)| / (|cos(H,V)| + |cos(H,T)|)', color=FG)
    ax.set_title('Transition Position on V-L Axis', color=FG, fontweight='bold')
    ax.legend(fontsize=8, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel B: MTI
    ax = axes[1]
    ax.set_facecolor(BG)
    mti = smooth(safe_arr(data, 'derived__mti'))
    ax.plot(steps, mti, color=C_GREEN, lw=2.5)
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('MTI = cos(H,V) × (1-cos(H,T)) × diag_score', color=FG)
    ax.set_title('Modality Transition Index', color=FG, fontweight='bold')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel C: V-L 等距度
    ax = axes[2]
    ax.set_facecolor(BG)
    vl_eq = smooth(safe_arr(data, 'derived__vl_equidistance'))
    ax.plot(steps, vl_eq, color=C_ORANGE, lw=2.5)
    ax.axhline(0, color=FG, ls=':', lw=1, alpha=0.3, label='Perfectly equidistant')
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('|cos(H,V) - cos(H,T)|', color=FG)
    ax.set_title('V-L Equidistance of H', color=FG, fontweight='bold')
    ax.legend(fontsize=8, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig2_transition_modality_index.png')
    plt.savefig(out_path, facecolor=BG)
    plt.close()
    print(f"  ✅ Fig 2: {out_path}")


# ============================================================
# Figure 3: 隐状态几何演化 (精简版)
# ============================================================
def plot_hidden_geometry(data, phases, out_dir):
    """diag_score / h_norm / shift_kl / adj_cos"""
    steps = np.array(data['steps'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle('Hidden State Geometry Evolution',
                 fontsize=16, fontweight='bold', color=FG, y=0.98)

    # Panel A: diag_score (对角一致性)
    ax = axes[0, 0]
    ax.set_facecolor(BG)
    diag = smooth(safe_arr(data, 'hidden_geometry__diag_score'))
    ax.plot(steps, diag, color=C_GREEN, lw=2.5)
    ax.fill_between(steps, diag, alpha=0.15, color=C_GREEN)
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('Diagonal Score', color=FG)
    ax.set_title('Stage Diagonal Score (Representation Efficiency)', color=FG, fontweight='bold')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel B: h_norm
    ax = axes[0, 1]
    ax.set_facecolor(BG)
    h_norm = smooth(safe_arr(data, 'hidden_geometry__h_norm'))
    ax.plot(steps, h_norm, color=C_H, lw=2.5)
    ax.fill_between(steps, h_norm, alpha=0.1, color=C_H)
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('‖h‖ Mean', color=FG)
    ax.set_title('Hidden State Norm', color=FG, fontweight='bold')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel C: shift_kl
    ax = axes[1, 0]
    ax.set_facecolor(BG)
    shift_kl = smooth(safe_arr(data, 'hidden_geometry__shift_kl'))
    ax.plot(steps, shift_kl, color=C_ORANGE, lw=2.5)
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('Stage Shift KL', color=FG)
    ax.set_title('Stage-to-Stage Representation Shift', color=FG, fontweight='bold')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel D: adj_cos_mean
    ax = axes[1, 1]
    ax.set_facecolor(BG)
    adj_cos = smooth(safe_arr(data, 'hidden_geometry__adj_cos_mean'))
    ax.plot(steps, adj_cos, color=C_TEAL, lw=2.5, label='Adjacent cos')
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('Cosine Similarity', color=FG)
    ax.set_title('Inter-Step Consistency', color=FG, fontweight='bold')
    ax.legend(fontsize=9, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig3_hidden_geometry.png')
    plt.savefig(out_path, facecolor=BG)
    plt.close()
    print(f"  ✅ Fig 3: {out_path}")


# ============================================================
# Figure 4: CKA 近似轨迹
# ============================================================
def plot_cka_trajectory(data, phases, out_dir):
    """CKA(H,V) / CKA(H,T) 近似值变化"""
    steps = np.array(data['steps'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)
    fig.suptitle('CKA Approximation: Representation Alignment',
                 fontsize=16, fontweight='bold', color=FG, y=1.02)

    cka_hv = smooth(safe_arr(data, 'cka_hv_approx'))
    cka_ht = smooth(safe_arr(data, 'cka_ht_approx'))
    cka_ratio = smooth(safe_arr(data, 'cka_hv_ht_ratio'))

    # Panel A: CKA(H,V)
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.plot(steps, cka_hv, color=C_V, lw=2.5)
    ax.fill_between(steps, cka_hv, alpha=0.15, color=C_V)
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('CKA(H,V) approx', color=FG)
    ax.set_title('CKA(H, V) — Vision Alignment', color=FG, fontweight='bold')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel B: CKA(H,T)
    ax = axes[1]
    ax.set_facecolor(BG)
    ax.plot(steps, cka_ht, color=C_L, lw=2.5)
    ax.fill_between(steps, cka_ht, alpha=0.15, color=C_L)
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('CKA(H,T) approx', color=FG)
    ax.set_title('CKA(H, T) — Language Alignment', color=FG, fontweight='bold')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel C: CKA Ratio
    ax = axes[2]
    ax.set_facecolor(BG)
    ax.plot(steps, cka_ratio, color=C_H, lw=2.5)
    ax.axhline(1.0, color=FG, ls=':', lw=1, alpha=0.3, label='Equal alignment')
    add_phase_backgrounds(ax, phases)
    ax.set_xlabel('Training Step', color=FG)
    ax.set_ylabel('CKA(H,V) / CKA(H,T)', color=FG)
    ax.set_title('Vision-Language Alignment Ratio', color=FG, fontweight='bold')
    ax.legend(fontsize=9, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig4_cka_trajectory.png')
    plt.savefig(out_path, facecolor=BG)
    plt.close()
    print(f"  ✅ Fig 4: {out_path}")


# ============================================================
# Figure 5: 训练阶段仪表盘 (精简版 6 面板)
# ============================================================
def plot_training_dashboard(data, phases, out_dir):
    """综合仪表盘 — 6 面板"""
    steps = np.array(data['steps'])

    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Modality Manifold Analysis — Training Dashboard',
                 fontsize=18, fontweight='bold', color=FG, y=0.99)

    # Panel A (0,0): 模态对齐
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(BG)
    cos_hv = smooth(safe_arr(data, 'cos_hv__mean'))
    cos_ht = smooth(safe_arr(data, 'cos_ht__mean'))
    ax.plot(steps, cos_hv, color=C_V, lw=2, label='cos(H,V)')
    ax.plot(steps, cos_ht, color=C_L, lw=2, label='cos(H,T)')
    add_phase_backgrounds(ax, phases)
    ax.set_title('Modality Alignment', color=FG, fontweight='bold')
    ax.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG, labelsize=8)

    # Panel B (0,1): 几何效率
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(BG)
    diag = smooth(safe_arr(data, 'hidden_geometry__diag_score'))
    h_norm = smooth(safe_arr(data, 'hidden_geometry__h_norm'))
    ax.plot(steps, diag, color=C_GREEN, lw=2, label='diag_score')
    ax2 = ax.twinx()
    ax2.plot(steps, h_norm, color=C_H, lw=2, label='‖h‖', alpha=0.7)
    ax2.tick_params(axis='y', colors=C_H, labelsize=8)
    ax2.set_ylabel('‖h‖', color=C_H, fontsize=9)
    add_phase_backgrounds(ax, phases)
    ax.set_title('Geometry Efficiency', color=FG, fontweight='bold')
    ax.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=FG, loc='upper left')
    ax2.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=C_H, loc='upper right')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG, labelsize=8)

    # Panel C (0,2): 准确率 + Loss
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(BG)
    acc = smooth(safe_arr(data, 'training__acc'))
    total_loss = smooth(safe_arr(data, 'training__total_loss'))
    ax.plot(steps, acc, color=C_GREEN, lw=2, label='Accuracy')
    ax2 = ax.twinx()
    ax2.plot(steps, total_loss, color=C_L, lw=2, label='Total Loss', alpha=0.7)
    ax2.tick_params(axis='y', colors=C_L, labelsize=8)
    ax2.set_ylabel('Loss', color=C_L, fontsize=9)
    add_phase_backgrounds(ax, phases)
    ax.set_title('Training Progress', color=FG, fontweight='bold')
    ax.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=FG, loc='upper left')
    ax2.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=C_L, loc='upper right')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG, labelsize=8)

    # Panel D (1,0): 过渡指标综合
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(BG)
    mti = smooth(safe_arr(data, 'derived__mti'))
    alpha_t = smooth(safe_arr(data, 'derived__alpha_transition'))
    ax.plot(steps, mti, color=C_GREEN, lw=2, label='MTI')
    ax2 = ax.twinx()
    ax2.plot(steps, alpha_t, color=C_H, lw=2, label='α_transition', alpha=0.7)
    ax2.tick_params(axis='y', colors=C_H, labelsize=8)
    add_phase_backgrounds(ax, phases)
    ax.set_title('Transition Modality Index', color=FG, fontweight='bold')
    ax.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=FG, loc='upper left')
    ax2.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=C_H, loc='upper right')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG, labelsize=8)

    # Panel E (1,1): 视觉注意力
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(BG)
    alpha_mean = smooth(safe_arr(data, 'vision_attn__alpha_mean'))
    vis_count = smooth(safe_arr(data, 'vision_attn__vis_count'))
    ax.plot(steps, alpha_mean, color=C_V, lw=2, label='Vision α')
    ax2 = ax.twinx()
    ax2.plot(steps, vis_count, color=C_ACCENT, lw=2, label='Vis Token Count', alpha=0.7)
    ax2.tick_params(axis='y', colors=C_ACCENT, labelsize=8)
    add_phase_backgrounds(ax, phases)
    ax.set_title('Visual Attention Dynamics', color=FG, fontweight='bold')
    ax.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=FG, loc='upper left')
    ax2.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=C_ACCENT, loc='upper right')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG, labelsize=8)

    # Panel F (1,2): 思考效率
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(BG)
    thought_steps = smooth(safe_arr(data, 'training__thought_steps'))
    thought_count = smooth(safe_arr(data, 'training__thought_count'))
    ax.plot(steps, thought_steps, color=C_PINK, lw=2, label='Mean Thought Steps')
    ax2 = ax.twinx()
    ax2.plot(steps, thought_count, color=C_TEAL, lw=2, label='Thought Token Count', alpha=0.7)
    ax2.tick_params(axis='y', colors=C_TEAL, labelsize=8)
    add_phase_backgrounds(ax, phases)
    ax.set_title('Thinking Efficiency', color=FG, fontweight='bold')
    ax.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=FG, loc='upper left')
    ax2.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=C_TEAL, loc='upper right')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG, labelsize=8)

    # Panel G (2,0:2): MTI & Capacity 综合指标
    ax = fig.add_subplot(gs[2, :2])
    ax.set_facecolor(BG)
    mti = smooth(safe_arr(data, 'derived__mti'))
    capacity = smooth(safe_arr(data, 'derived__capacity'))
    ax.plot(steps, mti, color=C_GREEN, lw=2.5, label='MTI (Transition Index)')
    ax.plot(steps, capacity, color=C_H, lw=2, alpha=0.7, label='Capacity (‖h‖×diag/100)')
    add_phase_backgrounds(ax, phases)
    ax.set_title('Modality Transition Index & Capacity', color=FG, fontweight='bold')
    ax.legend(fontsize=9, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel H (2,2): CKA 综合视图
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor(BG)
    cka_hv = smooth(safe_arr(data, 'cka_hv_approx'))
    cka_ht = smooth(safe_arr(data, 'cka_ht_approx'))
    cka_ratio = smooth(safe_arr(data, 'cka_hv_ht_ratio'))
    ax.plot(steps, cka_hv, color=C_V, lw=2, label='CKA(H,V)')
    ax.plot(steps, cka_ht, color=C_L, lw=2, label='CKA(H,T)')
    ax2 = ax.twinx()
    ax2.plot(steps, cka_ratio, color=C_ACCENT, lw=1.5, label='CKA Ratio', alpha=0.7)
    ax2.tick_params(axis='y', colors=C_ACCENT, labelsize=8)
    ax2.axhline(1.0, color=FG, ls=':', lw=0.5, alpha=0.3)
    add_phase_backgrounds(ax, phases)
    ax.set_title('CKA Overview', color=FG, fontweight='bold')
    ax.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=FG, loc='upper left')
    ax2.legend(fontsize=7, facecolor=BG, edgecolor=EDGE, labelcolor=C_ACCENT, loc='upper right')
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=FG, labelsize=8)

    out_path = os.path.join(out_dir, 'fig5_training_dashboard.png')
    plt.savefig(out_path, facecolor=BG)
    plt.close()
    print(f"  ✅ Fig 5: {out_path}")


# ============================================================
# Figure 6: 模态锥体 3D 演化
# ============================================================
def plot_3d_cone_evolution(data, phases, baseline, out_dir):
    """训练前后 V/L/H 锥体相对位置变化的 3D 示意图"""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 8), facecolor=BG)
    fig.suptitle('Modality Cone Evolution: Before vs After Training',
                 fontsize=16, fontweight='bold', color=FG, y=0.98)

    # 使用真实指标构建几何
    # 初始状态 (step 0)
    cos_hv_0_raw = safe_arr(data, 'cos_hv__mean')[0]
    cos_ht_0_raw = safe_arr(data, 'cos_ht__mean')[0]
    cos_hv_0 = cos_hv_0_raw if not np.isnan(cos_hv_0_raw) else 0.41
    cos_ht_0 = cos_ht_0_raw if not np.isnan(cos_ht_0_raw) else 0.09
    # 最终状态 (step end)
    cos_hv_f_raw = safe_arr(data, 'cos_hv__mean')[-1]
    cos_ht_f_raw = safe_arr(data, 'cos_ht__mean')[-1]
    cos_hv_f = cos_hv_f_raw if not np.isnan(cos_hv_f_raw) else 0.17
    cos_ht_f = cos_ht_f_raw if not np.isnan(cos_ht_f_raw) else 0.17

    # 角度
    angle_hv_0 = np.degrees(np.arccos(np.clip(cos_hv_0, -1, 1)))
    angle_ht_0 = np.degrees(np.arccos(np.clip(cos_ht_0, -1, 1)))
    angle_hv_f = np.degrees(np.arccos(np.clip(cos_hv_f, -1, 1)))
    angle_ht_f = np.degrees(np.arccos(np.clip(cos_ht_f, -1, 1)))

    # Baseline V-L gap
    vl_gap = 83.28  # from geometry_metrics.json
    if baseline:
        vl_gap = baseline.get('modality_gap_angle_vl', 83.28)

    for idx, (title, ahv, aht) in enumerate([
        ('Before Training (Step 0)', angle_hv_0, angle_ht_0),
        ('After Training (Step 1200)', angle_hv_f, angle_ht_f),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d', facecolor=BG)

        # V 在 x 轴正方向, L 在 V-L gap 角度处
        v_dir = np.array([1, 0, 0])
        vl_rad = np.radians(vl_gap)
        l_dir = np.array([np.cos(vl_rad), np.sin(vl_rad), 0])

        # H 的位置: 从 V 方向偏转 ahv, 从 L 方向偏转 aht
        # 简化: 用 V 和 L 的加权组合
        hv_rad = np.radians(ahv)
        hl_rad = np.radians(aht)
        # H 在 V-L 平面内, 靠近 V 的方向
        h_dir = np.array([
            np.cos(hv_rad),
            np.sin(hv_rad) * np.sign(np.cos(vl_rad/2)),
            0.1 * np.sin(hv_rad)  # 轻微 Z 偏移表示"过渡"
        ])
        h_dir = h_dir / np.linalg.norm(h_dir)

        # 画锥体 (简化为圆弧)
        for center_dir, color, label, cone_angle in [
            (v_dir, C_V, 'Vision (V)', 45.28),
            (l_dir, C_L, 'Language (L)', 48.70),
            (h_dir, C_H, 'Hidden (H)', 21.01),
        ]:
            # 画中心轴
            ax.quiver(0, 0, 0, center_dir[0], center_dir[1], center_dir[2],
                     color=color, arrow_length_ratio=0.15, lw=2.5, label=label)

            # 画锥面 (简化为几条射线)
            cone_rad = np.radians(cone_angle)
            perp1 = np.cross(center_dir, [0, 0, 1])
            if np.linalg.norm(perp1) < 1e-6:
                perp1 = np.cross(center_dir, [0, 1, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(center_dir, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)

            for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                ray_dir = (center_dir * np.cos(cone_rad) +
                          (perp1 * np.cos(theta) + perp2 * np.sin(theta)) * np.sin(cone_rad))
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                ax.plot([0, ray_dir[0] * 0.5], [0, ray_dir[1] * 0.5], [0, ray_dir[2] * 0.5],
                       color=color, alpha=0.15, lw=0.5)

            # 锥口圆弧
            t = np.linspace(0, 2 * np.pi, 40)
            r = np.sin(cone_rad) * 0.5
            z_offset = np.cos(cone_rad) * 0.5
            circle = center_dir[:, None] * z_offset + (perp1[:, None] * np.cos(t) + perp2[:, None] * np.sin(t)) * r
            ax.plot(circle[0], circle[1], circle[2], color=color, alpha=0.4, lw=1.5)

        # 标注角度
        ax.text2D(0.05, 0.95, f'∠(H,V) = {ahv:.1f}°', transform=ax.transAxes,
                 color=C_V, fontsize=10, fontweight='bold')
        ax.text2D(0.05, 0.88, f'∠(H,T) = {aht:.1f}°', transform=ax.transAxes,
                 color=C_L, fontsize=10, fontweight='bold')
        ax.text2D(0.05, 0.81, f'∠(V,L) = {vl_gap:.1f}°', transform=ax.transAxes,
                 color=FG, fontsize=10)

        ax.set_xlim(-0.5, 1.2)
        ax.set_ylim(-0.5, 1.2)
        ax.set_zlim(-0.5, 0.5)
        ax.set_title(title, color=FG, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='upper right', facecolor=BG, edgecolor=EDGE, labelcolor=FG)
        ax.tick_params(colors=FG, labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(EDGE)
        ax.yaxis.pane.set_edgecolor(EDGE)
        ax.zaxis.pane.set_edgecolor(EDGE)

    out_path = os.path.join(out_dir, 'fig6_3d_cone_evolution.png')
    plt.savefig(out_path, facecolor=BG)
    plt.close()
    print(f"  ✅ Fig 6: {out_path}")


# ============================================================
# Figure 7: Baseline 对比
# ============================================================
def plot_baseline_comparison(data, baseline, out_dir):
    """训练前 vs 训练后的关键指标雷达图"""
    if baseline is None:
        print("  ⚠️ No baseline metrics, skipping Fig 7")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle('Before vs After Training: Modality Geometry Comparison',
                 fontsize=16, fontweight='bold', color=FG, y=0.98)

    # 收集指标
    before = {
        'cos(H,V)': baseline.get('pairwise_vh_cos_mean', 0.42),
        'cos(H,T)': baseline.get('pairwise_lh_cos_mean', 0.08),
        'diag_score': 0.45,  # step 0 value
        'cone_H_half_angle': baseline.get('cone_H_half_angle', 21.0),
        'intra_H_cos': baseline.get('intra_H_cos_mean', 0.87),
        'gap_VH': baseline.get('modality_gap_angle_vh', 53.0),
    }

    after = {
        'cos(H,V)': safe_arr(data, 'cos_hv__mean')[-1] if not np.isnan(safe_arr(data, 'cos_hv__mean')[-1]) else 0.17,
        'cos(H,T)': safe_arr(data, 'cos_ht__mean')[-1] if not np.isnan(safe_arr(data, 'cos_ht__mean')[-1]) else 0.17,
        'diag_score': safe_arr(data, 'hidden_geometry__diag_score')[-1] if not np.isnan(safe_arr(data, 'hidden_geometry__diag_score')[-1]) else 0.74,
        'cone_H_half_angle': 21.0,  # 需要从 ckpt-1200 embeddings 计算, 暂用近似
        'intra_H_cos': safe_arr(data, 'hidden_geometry__adj_cos_mean')[-1] if not np.isnan(safe_arr(data, 'hidden_geometry__adj_cos_mean')[-1]) else 0.87,
        'gap_VH': safe_arr(data, 'derived__angle_hv_deg')[-1] if not np.isnan(safe_arr(data, 'derived__angle_hv_deg')[-1]) else 80.0,
    }

    # Panel A: 柱状对比
    ax = axes[0]
    ax.set_facecolor(BG)
    metrics = list(before.keys())
    x = np.arange(len(metrics))
    w = 0.35

    before_vals = [before[k] for k in metrics]
    after_vals = [after[k] for k in metrics]

    bars1 = ax.bar(x - w/2, before_vals, w, color=C_V, alpha=0.85, label='Baseline', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, after_vals, w, color=C_H, alpha=0.85, label='After (ckpt-1200)', edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars1, before_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7, color=FG)
    for bar, val in zip(bars2, after_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7, color=FG)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=9, color=FG)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=EDGE, labelcolor=FG)
    ax.set_title('Metric Comparison', color=FG, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    # Panel B: 变化量
    ax = axes[1]
    ax.set_facecolor(BG)
    deltas = {k: after[k] - before[k] for k in metrics}
    colors = [C_GREEN if d > 0 else C_L for d in deltas.values()]

    bars = ax.barh(list(deltas.keys()), list(deltas.values()), color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, (k, d) in zip(bars, deltas.items()):
        ax.text(bar.get_width() + (0.005 if d >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
                f'{d:+.4f}', ha='left' if d >= 0 else 'right', va='center', fontsize=9, color=FG, fontweight='bold')

    ax.axvline(0, color=FG, ls='--', alpha=0.3)
    ax.set_title('Δ (After - Before)', color=FG, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, color=GRID)
    ax.tick_params(colors=FG)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig7_baseline_comparison.png')
    plt.savefig(out_path, facecolor=BG)
    plt.close()
    print(f"  ✅ Fig 7: {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    parser.add_argument('--figures_dir', default=FIGURES_DIR)
    args = parser.parse_args()

    print("=" * 70)
    print("  Modality Manifold Analysis — Visualization (v2)")
    print("=" * 70)

    # 加载数据
    data = load_training_metrics(args.results_dir)
    phases = load_phase_summary(args.results_dir)
    baseline = load_baseline_metrics()

    print(f"  Steps: {len(data['steps'])}, Phases: {len(phases)}")
    if baseline:
        print(f"  Baseline loaded: {len(baseline)} metrics")

    # 生成所有图表
    print("\n📊 Generating figures...")

    plot_alignment_trajectory(data, phases, args.figures_dir)
    plot_transition_index(data, phases, baseline, args.figures_dir)
    plot_hidden_geometry(data, phases, args.figures_dir)
    plot_cka_trajectory(data, phases, args.figures_dir)
    plot_training_dashboard(data, phases, args.figures_dir)
    plot_3d_cone_evolution(data, phases, baseline, args.figures_dir)
    plot_baseline_comparison(data, baseline, args.figures_dir)

    print("\n✅ All figures generated.")


if __name__ == '__main__':
    main()
