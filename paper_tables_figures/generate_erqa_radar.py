#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_erqa_radar.py
======================
Generate a publication-quality radar chart for ERQA sub-task
performance comparison between baseline (Qwen3-VL-8B-Instruct)
and LatentDraft.

Produces:
  - fig_erqa_radar.pdf / fig_erqa_radar.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ============================================================
# Style Configuration (Academic Paper Style)
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Color palette
COLOR_BASE = '#D35400'     # Baseline - Warm Orange
COLOR_OURS = '#2980B9'     # Ours - Strong Blue
COLOR_BASE_FILL = '#F5CBA7'
COLOR_OURS_FILL = '#AED6F1'
COLOR_GRID = '#8E9296'
COLOR_BG = '#FFFFFF'
COLOR_LABEL = '#1A252F'    # Darker label color
COLOR_TICK = '#4A5568'     # Darker tick label color


def make_radar_chart(output_dir):
    """Generate ERQA sub-task radar chart with polished aesthetics."""

    categories = [
        'Action\nReasoning',
        'Multi-view\nReasoning',
        'Other',
        'Pointing',
        'Spatial\nReasoning',
        'State\nEstimation',
        'Task\nReasoning',
        'Trajectory\nReasoning',
    ]

    baseline_scores = [44.44, 35.14, 28.57, 44.12, 41.67, 47.27, 50.00, 40.91]
    ours_scores     = [47.22, 29.73, 35.71, 61.76, 47.62, 52.73, 57.89, 43.94]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    baseline_scores_closed = baseline_scores + baseline_scores[:1]
    ours_scores_closed = ours_scores + ours_scores[:1]

    # ---- Create figure ----
    fig = plt.figure(figsize=(5.5, 5.5), facecolor=COLOR_BG)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(COLOR_BG)

    # ---- Grid styling ----
    ax.set_ylim(0, 75)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(['20', '40', '60'], fontsize=8, color=COLOR_TICK)
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.5, alpha=0.5)

    # ---- Spine styling ----
    ax.spines['polar'].set_visible(False)

    # ---- Draw radial grid circles manually for cleaner look ----
    for r in [20, 40, 60]:
        circle = plt.Circle((0, 0), r, transform=ax.transData + ax.transAxes,
                             fill=False, edgecolor=COLOR_GRID, linewidth=0.5, linestyle='--')
        # Use ax's own grid instead

    # ---- Baseline polygon ----
    ax.plot(angles, baseline_scores_closed, 'o-', color=COLOR_BASE, linewidth=1.8,
            markersize=5, markeredgewidth=1.2, markeredgecolor='white',
            label='Qwen3-VL-8B-Instruct', zorder=3)
    ax.fill(angles, baseline_scores_closed, color=COLOR_BASE_FILL, alpha=0.35, zorder=2)

    # ---- Ours polygon (on top) ----
    ax.plot(angles, ours_scores_closed, 's-', color=COLOR_OURS, linewidth=2.4,
            markersize=5.5, markeredgewidth=1.2, markeredgecolor='white',
            label='LatentDraft (Ours)', zorder=4)
    ax.fill(angles, ours_scores_closed, color=COLOR_OURS_FILL, alpha=0.4, zorder=2)

    # ---- Axis labels ----
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='normal', color=COLOR_LABEL)

    # ---- Value annotations ----
    for i in range(N):
        angle = angles[i]
        # Baseline value
        ax.text(angle, baseline_scores[i] + 3.5, f'{baseline_scores[i]:.1f}',
                ha='center', va='center', fontsize=8, fontweight='bold', color=COLOR_BASE,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        # Ours value
        ax.text(angle, ours_scores[i] + 3.5, f'{ours_scores[i]:.1f}',
                ha='center', va='center', fontsize=8, fontweight='bold', color=COLOR_OURS,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ---- Legend ----
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                       fontsize=9, framealpha=0.95, edgecolor='#BDC3C7',
                       fancybox=True, shadow=False)
    legend.get_frame().set_linewidth(0.5)

    # ---- No title ----

    # ---- Save ----
    plt.tight_layout()
    pdf_path = os.path.join(output_dir, 'fig_erqa_radar.pdf')
    png_path = os.path.join(output_dir, 'fig_erqa_radar.png')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    plt.close()
    print(f"  Radar chart saved: {pdf_path}")
    print(f"  Radar chart saved: {png_path}")


if __name__ == '__main__':
    output_dir = os.path.dirname(os.path.abspath(__file__))
    make_radar_chart(output_dir)
