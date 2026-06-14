#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_paper_figures.py
===========================
Generate publication-quality figures for the Modality Manifold Analysis chapter.

Produces:
  Main Figures (for paper body):
    - Figure A: Combined panel showing (a) modality gap angles, (b) orthogonal decomposition
    - Figure B: Per-sample cosine similarity distributions with statistical tests

  Supplementary Figures (for appendix):
    - Figure S1: t-SNE manifold visualization
    - Figure S2: PCA variance distribution and effective dimensionality
    - Figure S3: Per-direction alignment analysis
    - Figure S4: Subspace overlap comparison

Usage:
  python3 visualize_paper_figures.py \
      --results_dir /path/to/results_v3 \
      --subspace_dir /path/to/results_subspace \
      --output_dir /path/to/paper_figures
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ============================================================
# Style Configuration (Academic Paper Style)
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'grid.alpha': 0.3,
})

# Color palette
COLOR_V = '#D64541'    # Vision - Red
COLOR_T = '#2ECC71'    # Text - Green
COLOR_H = '#8E44AD'    # Hidden/Latent - Purple
COLOR_BASE = '#7F8C8D'  # Baseline - Gray
COLOR_TRAIN = '#3498DB'  # Trained - Blue
COLOR_ORTH = '#E67E22'   # Orthogonal component - Orange


def load_all_data(results_dir, subspace_dir):
    """Load all analysis results."""
    data = {}
    
    # Metrics
    with open(os.path.join(results_dir, 'metrics_baseline.json')) as f:
        data['metrics_baseline'] = json.load(f)
    with open(os.path.join(results_dir, 'metrics_ckpt.json')) as f:
        data['metrics_ckpt'] = json.load(f)
    
    # Subspace analysis
    with open(os.path.join(subspace_dir, 'experiment_A_orthogonal_decomposition.json')) as f:
        data['exp_A'] = json.load(f)
    with open(os.path.join(subspace_dir, 'experiment_B_principal_angles.json')) as f:
        data['exp_B'] = json.load(f)
    
    # Embeddings (for t-SNE and distributions)
    emb_baseline_path = os.path.join(results_dir, 'embeddings_baseline.pt')
    emb_ckpt_path = os.path.join(results_dir, 'embeddings_ckpt.pt')
    if os.path.exists(emb_baseline_path):
        data['emb_baseline'] = torch.load(emb_baseline_path, map_location='cpu', weights_only=False)
    if os.path.exists(emb_ckpt_path):
        data['emb_ckpt'] = torch.load(emb_ckpt_path, map_location='cpu', weights_only=False)
    
    return data


# ============================================================
# MAIN FIGURE 1: Modality Gap & Orthogonal Decomposition
# (Combined 2-panel figure for paper body)
# ============================================================

def figure_main_1(data, output_dir):
    """
    Main Figure 1: Two-panel figure showing:
      (a) Modality gap angles (triangle diagram) - before/after training
      (b) Orthogonal decomposition bar chart - key evidence
    """
    fig = plt.figure(figsize=(7.5, 3.5))  # Slightly larger to avoid overlap
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.40)
    
    mb = data['metrics_baseline']
    mc = data['metrics_ckpt']
    ea = data['exp_A']
    
    # ============================================================
    # Panel (a): Modality Gap Angles (Triangle Diagram)
    # ============================================================
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(-0.5, 3.0)
    ax_a.set_ylim(-0.4, 2.8)
    ax_a.set_aspect('equal')
    ax_a.axis('off')
    
    # Triangle vertices - spread out more to avoid label overlap
    # Baseline (left triangle)
    bx, by = 0.0, 0.0  # offset
    # V at bottom-left, T at bottom-right, H at top
    v_pos_b = (bx + 0.0, by + 0.0)
    t_pos_b = (bx + 1.2, by + 0.0)
    h_pos_b = (bx + 0.6, by + 1.8)
    
    # Draw baseline triangle
    for p1, p2 in [(v_pos_b, t_pos_b), (v_pos_b, h_pos_b), (t_pos_b, h_pos_b)]:
        ax_a.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color='#BDC3C7', linewidth=1.2, zorder=1)
    
    # Trained (right triangle, overlaid with solid lines)
    v_pos_t = v_pos_b
    t_pos_t = t_pos_b
    # H moves closer (lower angle = closer)
    h_pos_t = (bx + 0.6, by + 1.25)  # visually closer
    
    for p1, p2 in [(v_pos_t, h_pos_t), (t_pos_t, h_pos_t)]:
        ax_a.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=COLOR_H, linewidth=1.8, zorder=2)
    ax_a.plot([v_pos_t[0], t_pos_t[0]], [v_pos_t[1], t_pos_t[1]], '-', color='#BDC3C7', linewidth=1.2, zorder=1)
    
    # Node markers
    for pos, color, label in [(v_pos_b, COLOR_V, r'$\mathbf{V}$'), 
                               (t_pos_b, COLOR_T, r'$\mathbf{T}$'),
                               (h_pos_b, COLOR_BASE, r'$\mathbf{H}_{pre}$'),
                               (h_pos_t, COLOR_H, r'$\mathbf{H}_{post}$')]:
        ax_a.plot(*pos, 'o', color=color, markersize=8, zorder=5)
    
    # Labels - positioned further from nodes to avoid overlap
    ax_a.text(v_pos_b[0] - 0.15, v_pos_b[1] - 0.18, r'$\mathbf{V}$', fontsize=10, 
             color=COLOR_V, fontweight='bold', ha='center')
    ax_a.text(t_pos_b[0] + 0.15, t_pos_b[1] - 0.18, r'$\mathbf{T}$', fontsize=10, 
             color=COLOR_T, fontweight='bold', ha='center')
    ax_a.text(h_pos_b[0] - 0.3, h_pos_b[1] + 0.08, r'$\mathbf{H}_{pre}$', fontsize=9, 
             color=COLOR_BASE, ha='right')
    ax_a.text(h_pos_t[0] + 0.25, h_pos_t[1] + 0.08, r'$\mathbf{H}_{post}$', fontsize=9, 
             color=COLOR_H, ha='left')
    
    # Angle annotations
    angle_hv_pre = mb['center_angle_generated_vision']
    angle_ht_pre = mb['center_angle_generated_text']
    angle_hv_post = mc['center_angle_generated_vision']
    angle_ht_post = mc['center_angle_generated_text']
    angle_tv = mb['center_angle_text_vision']
    
    # Angle labels on edges - spread apart to avoid overlap
    # Left edge (H-V): pre label further left, post label closer to center
    ax_a.text(-0.22, 1.0, f'{angle_hv_pre:.1f}\u00b0', fontsize=7, color=COLOR_BASE, 
             ha='right', rotation=70, style='italic')
    ax_a.text(0.05, 0.5, f'{angle_hv_post:.1f}\u00b0', fontsize=7, color=COLOR_H, 
             ha='right', rotation=62, fontweight='bold')
    # Right edge (H-T): pre label further right, post label closer to center
    ax_a.text(1.42, 1.0, f'{angle_ht_pre:.1f}\u00b0', fontsize=7, color=COLOR_BASE, 
             ha='left', rotation=-70, style='italic')
    ax_a.text(1.15, 0.5, f'{angle_ht_post:.1f}\u00b0', fontsize=7, color=COLOR_H, 
             ha='left', rotation=-62, fontweight='bold')
    ax_a.text(0.6, -0.25, f'{angle_tv:.1f}\u00b0 (unchanged)', fontsize=7, color='#7F8C8D', ha='center')
    
    # Arrow showing H movement
    ax_a.annotate('', xy=h_pos_t, xytext=h_pos_b,
                 arrowprops=dict(arrowstyle='->', color=COLOR_H, lw=1.5, 
                               connectionstyle='arc3,rad=0.2'))
    
    # Delta annotations - placed at top with more spacing
    delta_hv = angle_hv_post - angle_hv_pre
    delta_ht = angle_ht_post - angle_ht_pre
    ax_a.text(0.6, 2.55, f'$\\Delta\\angle(\\mathbf{{H}},\\mathbf{{V}})={delta_hv:.1f}°$', 
             fontsize=8, ha='center', color=COLOR_V, fontweight='bold')
    ax_a.text(0.6, 2.30, f'$\\Delta\\angle(\\mathbf{{H}},\\mathbf{{T}})={delta_ht:.1f}°$', 
             fontsize=8, ha='center', color=COLOR_T)
    
    ax_a.set_title('(a) Modality Gap Reduction', fontsize=10, fontweight='bold', pad=5)
    
    # ============================================================
    # Panel (b): Orthogonal Decomposition Bar Chart
    # ============================================================
    ax_b = fig.add_subplot(gs[1])
    
    # Data
    categories = [
        r'$\cos(\mathrm{Proj}_{\mathcal{S}_T}(\mathbf{H}), \mathbf{V})$',
        r'$\cos(\mathrm{Proj}_{\mathcal{S}_T^\perp}(\mathbf{H}), \mathbf{V})$',
    ]
    baseline_vals = [ea['baseline']['center_cos_Hproj_V'], ea['baseline']['center_cos_Horth_V']]
    trained_vals = [ea['ckpt']['center_cos_Hproj_V'], ea['ckpt']['center_cos_Horth_V']]
    
    x = np.arange(len(categories))
    width = 0.32
    
    bars1 = ax_b.bar(x - width/2, baseline_vals, width, label='Baseline',
                    color=COLOR_BASE, edgecolor='black', linewidth=0.6, alpha=0.85)
    bars2 = ax_b.bar(x + width/2, trained_vals, width, label='Post-training', 
                    color=COLOR_H, edgecolor='black', linewidth=0.6, alpha=0.85)
    
    # Value annotations
    for bar in bars1:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=7.5, color=COLOR_BASE)
    for bar in bars2:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=7.5, color=COLOR_H, fontweight='bold')
    
    # Delta annotation with arrow
    delta_orth = trained_vals[1] - baseline_vals[1]
    delta_proj = trained_vals[0] - baseline_vals[0]
    ax_b.annotate(f'$\\Delta$=+{delta_orth:.3f}', 
                 xy=(1 + width/2, trained_vals[1]),
                 xytext=(1.5, trained_vals[1] * 0.7),
                 fontsize=8, color=COLOR_H, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLOR_H, lw=1.0))
    ax_b.annotate(f'$\\Delta$={delta_proj:.3f}', 
                 xy=(0 + width/2, trained_vals[0] + 0.01),
                 xytext=(0.4, 0.08),
                 fontsize=7.5, color='#7F8C8D',
                 arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=0.8))
    
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(categories, fontsize=8.5)
    ax_b.set_ylabel('Cosine Similarity with $\\mathbf{V}$', fontsize=9)
    ax_b.set_ylim(0, max(trained_vals) * 1.35)
    ax_b.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_b.set_title('(b) Orthogonal Decomposition', fontsize=10, fontweight='bold', pad=5)
    
    # Add text box with key insight
    textstr = (f'Visual info encoded in $\\mathcal{{S}}_T^\\perp$:\n'
               f'  $\\Delta\\cos_{{\\perp}}$ = +{delta_orth:.3f}\n'
               f'  $\\Delta\\cos_{{\\parallel}}$ = {delta_proj:.3f}')
    props = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='#BDC3C7')
    ax_b.text(0.98, 0.98, textstr, transform=ax_b.transAxes, fontsize=7,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.savefig(os.path.join(output_dir, 'fig_main1_modality_gap_and_decomposition.pdf'), 
               format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_main1_modality_gap_and_decomposition.png'), 
               format='png', bbox_inches='tight')
    plt.close()
    print("  ✅ Main Figure 1 saved: fig_main1_modality_gap_and_decomposition.pdf/png")


# ============================================================
# MAIN FIGURE 2: Per-Sample Cosine Similarity Distributions
# ============================================================

def figure_main_2(data, output_dir):
    """
    Main Figure 2: Violin/box plots showing per-sample cosine similarity distributions.
    Three panels: cos(H,V), cos(H,T), cos(T,V)
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
    
    emb_b = data.get('emb_baseline')
    emb_c = data.get('emb_ckpt')
    
    if emb_b is None or emb_c is None:
        print("  ⚠️ Embeddings not found, skipping Main Figure 2")
        return
    
    def cos_sim_samples(X, Y):
        """Compute per-sample cosine similarity."""
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
        return (X_norm * Y_norm).sum(axis=1)
    
    pairs = [
        ('cos($\\mathbf{H}$, $\\mathbf{V}$)', 'generated', 'vision'),
        ('cos($\\mathbf{H}$, $\\mathbf{T}$)', 'generated', 'text'),
        ('cos($\\mathbf{T}$, $\\mathbf{V}$)', 'text', 'vision'),
    ]
    
    for idx, (title, key1, key2) in enumerate(pairs):
        ax = axes[idx]
        
        cos_baseline = cos_sim_samples(emb_b[key1], emb_b[key2])
        cos_trained = cos_sim_samples(emb_c[key1], emb_c[key2])
        
        # Violin plot
        parts_b = ax.violinplot([cos_baseline], positions=[0], showmeans=True, 
                               showextrema=True, widths=0.6)
        parts_t = ax.violinplot([cos_trained], positions=[1], showmeans=True, 
                               showextrema=True, widths=0.6)
        
        # Style violins
        for pc in parts_b['bodies']:
            pc.set_facecolor(COLOR_BASE)
            pc.set_alpha(0.6)
        for pc in parts_t['bodies']:
            pc.set_facecolor(COLOR_H)
            pc.set_alpha(0.6)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            if partname in parts_b:
                parts_b[partname].set_color(COLOR_BASE)
                parts_b[partname].set_linewidth(1.0)
            if partname in parts_t:
                parts_t[partname].set_color(COLOR_H)
                parts_t[partname].set_linewidth(1.0)
        
        # Mean annotations - offset to avoid overlapping with violin body
        mu_b = cos_baseline.mean()
        mu_t = cos_trained.mean()
        # Place mu labels to the side of violins
        ax.text(-0.38, mu_b, f'$\\mu$={mu_b:.3f}', ha='center', fontsize=6.5, color=COLOR_BASE, va='center')
        ax.text(1.38, mu_t, f'$\\mu$={mu_t:.3f}', ha='center', fontsize=6.5, color=COLOR_H, fontweight='bold', va='center')
        
        # Delta - place above the plot area using axes coordinates
        delta = mu_t - mu_b
        if abs(delta) > 0.01:
            ax.text(0.5, 0.95, f'$\\Delta$={delta:+.3f}', ha='center', fontsize=7.5, 
                   color=COLOR_H if delta > 0 else COLOR_BASE, fontweight='bold',
                   transform=ax.transAxes, va='top')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre', 'Post'], fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4, linewidth=0.5)
        
        if idx == 0:
            ax.set_ylabel('Cosine Similarity', fontsize=9)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_BASE, alpha=0.6, label='Baseline'),
        mpatches.Patch(facecolor=COLOR_H, alpha=0.6, label='Post-training'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 1.08), framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_main2_cosine_distributions.pdf'), 
               format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_main2_cosine_distributions.png'), 
               format='png', bbox_inches='tight')
    plt.close()
    print("  ✅ Main Figure 2 saved: fig_main2_cosine_distributions.pdf/png")


# ============================================================
# MAIN FIGURE 3: Dimensionality Expansion & Direction Alignment
# ============================================================

def figure_main_3(data, output_dir):
    """
    Main Figure 3: Two-panel figure showing:
      (a) PCA variance distribution (effective dim expansion)
      (b) Per-direction alignment with V vs T subspaces
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    eb = data['exp_B']
    
    # ============================================================
    # Panel (a): PCA Variance Distribution
    # ============================================================
    ax = axes[0]
    
    var_b = np.array(eb['baseline']['H_var_explained'])
    var_t = np.array(eb['ckpt']['H_var_explained'])
    n_pcs = len(var_b)
    x = np.arange(1, n_pcs + 1)
    
    ax.bar(x - 0.2, var_b, 0.38, label=f'Pre (eff. dim={eb["baseline"]["effective_dim_H"]:.1f})', 
           color=COLOR_BASE, alpha=0.75, edgecolor='none')
    ax.bar(x + 0.2, var_t, 0.38, label=f'Post (eff. dim={eb["ckpt"]["effective_dim_H"]:.1f})', 
           color=COLOR_H, alpha=0.75, edgecolor='none')
    
    # Effective dim boundary
    eff_dim_b = eb['baseline']['effective_dim_H']
    ax.axvline(x=eff_dim_b, color=COLOR_BASE, linestyle='--', alpha=0.6, linewidth=0.8)
    ax.text(eff_dim_b + 0.3, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.3, 
           f'eff. dim\n(pre)', fontsize=6.5, color=COLOR_BASE, va='top')
    
    ax.set_xlabel('Principal Component Index', fontsize=9)
    ax.set_ylabel('Variance Explained', fontsize=9)
    ax.set_title('(a) PCA Variance Distribution of $\\mathbf{H}$', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
    ax.set_xlim(0.5, 30.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    
    # ============================================================
    # Panel (b): Per-Direction Alignment
    # ============================================================
    ax = axes[1]
    
    align_V_t = np.array(eb['ckpt']['direction_alignment_V'])
    align_T_t = np.array(eb['ckpt']['direction_alignment_T'])
    n_dirs = len(align_V_t)
    x = np.arange(1, n_dirs + 1)
    
    ax.plot(x, align_V_t, '-', color=COLOR_V, linewidth=1.3, marker='o', markersize=2.5,
           label='Alignment with $\\mathcal{S}_V$')
    ax.plot(x, align_T_t, '-', color=COLOR_T, linewidth=1.3, marker='s', markersize=2.5,
           label='Alignment with $\\mathcal{S}_T$')
    
    # Shade "new directions" region
    k_orig = 6
    ax.axvspan(k_orig + 0.5, n_dirs + 0.5, alpha=0.08, color=COLOR_H, zorder=0)
    ax.axvline(x=k_orig + 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)
    ax.text(k_orig + 1, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.12, 
           'New dims', fontsize=7, color=COLOR_H, va='top', style='italic')
    
    # Mean lines for new directions
    mean_V_new = np.mean(align_V_t[k_orig:])
    mean_T_new = np.mean(align_T_t[k_orig:])
    ax.axhline(y=mean_V_new, color=COLOR_V, linestyle=':', alpha=0.5, linewidth=0.7, 
              xmin=(k_orig+0.5)/n_dirs, xmax=1.0)
    ax.axhline(y=mean_T_new, color=COLOR_T, linestyle=':', alpha=0.5, linewidth=0.7,
              xmin=(k_orig+0.5)/n_dirs, xmax=1.0)
    
    ax.set_xlabel('PC Index of $\\mathbf{H}$ (Post-training)', fontsize=9)
    ax.set_ylabel('Subspace Alignment', fontsize=9)
    ax.set_title('(b) Direction Alignment (Post-training)', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
    ax.set_xlim(0.5, n_dirs + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_main3_dimensionality_and_alignment.pdf'), 
               format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_main3_dimensionality_and_alignment.png'), 
               format='png', bbox_inches='tight')
    plt.close()
    print("  ✅ Main Figure 3 saved: fig_main3_dimensionality_and_alignment.pdf/png")


# ============================================================
# SUPPLEMENTARY FIGURE S1: t-SNE Manifold Visualization
# ============================================================

def figure_supp_1(data, output_dir):
    """Supplementary Figure S1: t-SNE visualization of modality manifolds."""
    from sklearn.manifold import TSNE
    
    emb_b = data.get('emb_baseline')
    emb_c = data.get('emb_ckpt')
    
    if emb_b is None or emb_c is None:
        print("  ⚠️ Embeddings not found, skipping Supp Figure S1")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    
    for ax_idx, (emb, title) in enumerate([(emb_b, 'Baseline'), (emb_c, 'Post-training')]):
        ax = axes[ax_idx]
        
        V = emb['vision']
        T = emb['text']
        H = emb['generated']
        
        # Combine and run t-SNE
        all_data = np.vstack([V, T, H])
        n_v, n_t, n_h = len(V), len(T), len(H)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        embedded = tsne.fit_transform(all_data)
        
        # Split back
        emb_V = embedded[:n_v]
        emb_T = embedded[n_v:n_v+n_t]
        emb_H = embedded[n_v+n_t:]
        
        # Plot
        ax.scatter(emb_V[:, 0], emb_V[:, 1], c=COLOR_V, s=12, alpha=0.5, label='$\\mathbf{V}$ (Vision)', zorder=2)
        ax.scatter(emb_T[:, 0], emb_T[:, 1], c=COLOR_T, s=12, alpha=0.5, label='$\\mathbf{T}$ (Text)', zorder=2)
        ax.scatter(emb_H[:, 0], emb_H[:, 1], c=COLOR_H, s=12, alpha=0.6, label='$\\mathbf{H}$ (Latent)', zorder=3)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9, markerscale=1.5)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    
    plt.suptitle('Supplementary Figure S1: t-SNE Visualization of Modality Manifolds', 
                fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_supp1_tsne_manifold.pdf'), 
               format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_supp1_tsne_manifold.png'), 
               format='png', bbox_inches='tight')
    plt.close()
    print("  ✅ Supp Figure S1 saved: fig_supp1_tsne_manifold.pdf/png")


# ============================================================
# SUPPLEMENTARY FIGURE S2: Orthogonal Decomposition Detail
# ============================================================

def figure_supp_2(data, output_dir):
    """
    Supplementary Figure S2: Detailed orthogonal decomposition analysis.
    Shows both cos with V and cos with T for both components.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ea = data['exp_A']
    
    for ax_idx, (model_name, title) in enumerate([('baseline', 'Baseline'), ('ckpt', 'Post-training')]):
        ax = axes[ax_idx]
        r = ea[model_name]
        
        # 4 bars: proj→V, orth→V, proj→T, orth→T
        categories = [
            r'$\mathrm{Proj}_{\mathcal{S}_T}$' + '\n' + r'$\rightarrow \mathbf{V}$',
            r'$\mathrm{Proj}_{\mathcal{S}_T^\perp}$' + '\n' + r'$\rightarrow \mathbf{V}$',
            r'$\mathrm{Proj}_{\mathcal{S}_T}$' + '\n' + r'$\rightarrow \mathbf{T}$',
            r'$\mathrm{Proj}_{\mathcal{S}_T^\perp}$' + '\n' + r'$\rightarrow \mathbf{T}$',
        ]
        values = [r['center_cos_Hproj_V'], r['center_cos_Horth_V'],
                 r['center_cos_Hproj_T'], r['center_cos_Horth_T']]
        colors = [COLOR_V, COLOR_ORTH, COLOR_T, COLOR_ORTH]
        
        bars = ax.bar(range(len(categories)), values, color=colors, 
                     edgecolor='black', linewidth=0.5, alpha=0.8, width=0.7)
        
        # Hatch for orthogonal components
        bars[1].set_hatch('///')
        bars[3].set_hatch('///')
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=7.5)
        ax.set_ylabel('Cosine Similarity', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4, linewidth=0.5)
        
        # Value annotations
        for i, v in enumerate(values):
            ax.text(i, v + 0.003, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
        
        ymax = max(values) * 1.3 + 0.02
        ax.set_ylim(-0.02, ymax)
    
    plt.suptitle('Supplementary Figure S2: Orthogonal Decomposition — Cosine Similarity of Components', 
                fontsize=9.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_supp2_orthogonal_detail.pdf'), 
               format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_supp2_orthogonal_detail.png'), 
               format='png', bbox_inches='tight')
    plt.close()
    print("  ✅ Supp Figure S2 saved: fig_supp2_orthogonal_detail.pdf/png")


# ============================================================
# SUPPLEMENTARY FIGURE S3: Direction Alignment Comparison
# ============================================================

def figure_supp_3(data, output_dir):
    """
    Supplementary Figure S3: Direction alignment comparison (pre vs post).
    Shows how each PC direction of H aligns with V and T subspaces.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    eb = data['exp_B']
    
    for ax_idx, (model_name, title) in enumerate([('baseline', 'Baseline'), ('ckpt', 'Post-training')]):
        ax = axes[ax_idx]
        
        align_V = np.array(eb[model_name]['direction_alignment_V'])
        align_T = np.array(eb[model_name]['direction_alignment_T'])
        n_dirs = len(align_V)
        x = np.arange(1, n_dirs + 1)
        
        ax.plot(x, align_V, '-o', color=COLOR_V, linewidth=1.2, markersize=2.5,
               label='Align($\\mathrm{PC}_i$, $\\mathcal{S}_V$)')
        ax.plot(x, align_T, '-s', color=COLOR_T, linewidth=1.2, markersize=2.5,
               label='Align($\\mathrm{PC}_i$, $\\mathcal{S}_T$)')
        
        # Shade new directions
        k_orig = 6
        ax.axvspan(k_orig + 0.5, n_dirs + 0.5, alpha=0.08, color=COLOR_H, zorder=0)
        ax.axvline(x=k_orig + 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)
        
        ax.set_xlabel('PC Index', fontsize=9)
        ax.set_ylabel('Alignment Magnitude', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
        ax.set_xlim(0.5, n_dirs + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    
    plt.suptitle('Supplementary Figure S3: Per-Direction Alignment with Vision/Text Subspaces', 
                fontsize=9.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_supp3_direction_alignment_comparison.pdf'), 
               format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_supp3_direction_alignment_comparison.png'), 
               format='png', bbox_inches='tight')
    plt.close()
    print("  ✅ Supp Figure S3 saved: fig_supp3_direction_alignment_comparison.pdf/png")


# ============================================================
# SUPPLEMENTARY FIGURE S4: Subspace Overlap & Energy Distribution
# ============================================================

def figure_supp_4(data, output_dir):
    """
    Supplementary Figure S4: Subspace overlap and energy distribution.
    (a) Subspace overlap comparison
    (b) Energy distribution in T vs T-perp
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    eb = data['exp_B']
    ea = data['exp_A']
    
    # ============================================================
    # Panel (a): Subspace Overlap
    # ============================================================
    ax = axes[0]
    
    metrics = ['$\\mathbf{H}$ vs $\\mathcal{S}_V$', 
               '$\\mathbf{H}$ vs $\\mathcal{S}_T$',
               '$\\mathbf{H}_{new}$ vs $\\mathcal{S}_V$', 
               '$\\mathbf{H}_{new}$ vs $\\mathcal{S}_T$']
    baseline_vals = [eb['baseline']['overlap_H_V'], eb['baseline']['overlap_H_T'],
                    eb['baseline']['overlap_H_new_V'], eb['baseline']['overlap_H_new_T']]
    ckpt_vals = [eb['ckpt']['overlap_H_V'], eb['ckpt']['overlap_H_T'],
                eb['ckpt']['overlap_H_new_V'], eb['ckpt']['overlap_H_new_T']]
    
    x = np.arange(len(metrics))
    width = 0.32
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Pre', 
                  color=COLOR_BASE, edgecolor='black', linewidth=0.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, ckpt_vals, width, label='Post', 
                  color=COLOR_H, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Value annotations
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.0003, f'{h:.4f}',
               ha='center', va='bottom', fontsize=6, color=COLOR_BASE)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.0003, f'{h:.4f}',
               ha='center', va='bottom', fontsize=6, color=COLOR_H)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=6.5, rotation=15, ha='right')
    ax.set_ylabel('Subspace Overlap', fontsize=9)
    ax.set_title('(a) Subspace Overlap', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    
    # ============================================================
    # Panel (b): Energy Distribution
    # ============================================================
    ax = axes[1]
    
    # Stacked bar: energy in S_T vs S_T_perp
    models = ['Baseline', 'Post-training']
    energy_T = [ea['baseline']['H_energy_in_T'] * 100, ea['ckpt']['H_energy_in_T'] * 100]
    energy_Tperp = [ea['baseline']['H_energy_in_T_orth'] * 100, ea['ckpt']['H_energy_in_T_orth'] * 100]
    
    x = np.arange(len(models))
    width = 0.5
    
    ax.bar(x, energy_Tperp, width, label=r'In $\mathcal{S}_T^\perp$ (orthogonal)', 
           color=COLOR_ORTH, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x, energy_T, width, bottom=0, label=r'In $\mathcal{S}_T$ (semantic)', 
           color=COLOR_T, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Annotate energy percentages - avoid overlap for small values
    for i in range(len(models)):
        ax.text(i, energy_Tperp[i] / 2, f'{energy_Tperp[i]:.1f}%', 
               ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        # Place small S_T percentage below the bar with an arrow
        ax.annotate(f'{energy_T[i]:.1f}%', 
                   xy=(i, energy_T[i]), xytext=(i + 0.3, 12),
                   fontsize=7, color=COLOR_T, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=COLOR_T, lw=0.8),
                   ha='center', va='bottom')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Energy (%)', fontsize=9)
    ax.set_title(r'(b) $\|\mathbf{H}\|^2$ Distribution', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7.5, loc='center right', framealpha=0.9)
    ax.set_ylim(0, 105)
    
    plt.suptitle('Supplementary Figure S4: Subspace Overlap & Energy Distribution', 
                fontsize=9.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_supp4_overlap_and_energy.pdf'), 
               format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_supp4_overlap_and_energy.png'), 
               format='png', bbox_inches='tight')
    plt.close()
    print("  ✅ Supp Figure S4 saved: fig_supp4_overlap_and_energy.pdf/png")


# ============================================================
# SUPPLEMENTARY TABLE: Summary Metrics
# ============================================================

def generate_summary_table(data, output_dir):
    """Generate a LaTeX-ready summary table of all key metrics."""
    mb = data['metrics_baseline']
    mc = data['metrics_ckpt']
    ea = data['exp_A']
    eb = data['exp_B']
    
    table_content = r"""
\begin{table}[h]
\centering
\caption{Modality Manifold Geometry: Pre- vs. Post-training Comparison}
\label{tab:modality_geometry}
\begin{tabular}{lccr}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Post-training} & \textbf{$\Delta$} \\
\midrule
\multicolumn{4}{l}{\textit{Modality Gap (Cosine Similarity)}} \\
$\cos(\mathbf{H}, \mathbf{V})$ & %.3f & \textbf{%.3f} & +%.3f \\
$\cos(\mathbf{H}, \mathbf{T})$ & %.3f & \textbf{%.3f} & +%.3f \\
$\cos(\mathbf{T}, \mathbf{V})$ & %.3f & %.3f & %.3f \\
\midrule
\multicolumn{4}{l}{\textit{Orthogonal Decomposition}} \\
$\cos(\mathrm{Proj}_{\mathcal{S}_T^\perp}(\mathbf{H}), \mathbf{V})$ & %.3f & \textbf{%.3f} & +%.3f \\
$\cos(\mathrm{Proj}_{\mathcal{S}_T}(\mathbf{H}), \mathbf{V})$ & %.3f & %.3f & %.3f \\
$\|\mathbf{H}\|^2$ in $\mathcal{S}_T^\perp$ (\%%) & %.1f & %.1f & +%.1f \\
\midrule
\multicolumn{4}{l}{\textit{Dimensionality}} \\
Effective dim($\mathbf{H}$) & %.1f & \textbf{%.1f} & +%.1f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        mb['center_cos_generated_vision'], mc['center_cos_generated_vision'],
        mc['center_cos_generated_vision'] - mb['center_cos_generated_vision'],
        mb['center_cos_generated_text'], mc['center_cos_generated_text'],
        mc['center_cos_generated_text'] - mb['center_cos_generated_text'],
        mb['center_cos_text_vision'], mc['center_cos_text_vision'],
        mc['center_cos_text_vision'] - mb['center_cos_text_vision'],
        ea['baseline']['center_cos_Horth_V'], ea['ckpt']['center_cos_Horth_V'],
        ea['ckpt']['center_cos_Horth_V'] - ea['baseline']['center_cos_Horth_V'],
        ea['baseline']['center_cos_Hproj_V'], ea['ckpt']['center_cos_Hproj_V'],
        ea['ckpt']['center_cos_Hproj_V'] - ea['baseline']['center_cos_Hproj_V'],
        ea['baseline']['H_energy_in_T_orth'] * 100, ea['ckpt']['H_energy_in_T_orth'] * 100,
        (ea['ckpt']['H_energy_in_T_orth'] - ea['baseline']['H_energy_in_T_orth']) * 100,
        eb['baseline']['effective_dim_H'], eb['ckpt']['effective_dim_H'],
        eb['ckpt']['effective_dim_H'] - eb['baseline']['effective_dim_H'],
    )
    
    table_file = os.path.join(output_dir, 'table_modality_geometry.tex')
    with open(table_file, 'w') as f:
        f.write(table_content)
    print(f"  ✅ LaTeX table saved: {table_file}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures for modality manifold analysis')
    parser.add_argument('--results_dir', type=str, 
                       default='./modality_manifold_analysis/results_v3',
                       help='Directory containing metrics and embeddings')
    parser.add_argument('--subspace_dir', type=str, 
                       default='./modality_manifold_analysis/results_subspace',
                       help='Directory containing subspace analysis results')
    parser.add_argument('--output_dir', type=str, 
                       default='./modality_manifold_analysis/paper_figures',
                       help='Output directory for paper figures')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Generating Paper Figures: Modality Manifold Analysis")
    print("=" * 60)
    print(f"  Results dir:  {args.results_dir}")
    print(f"  Subspace dir: {args.subspace_dir}")
    print(f"  Output dir:   {args.output_dir}")
    
    # Load data
    print("\n  Loading data...")
    data = load_all_data(args.results_dir, args.subspace_dir)
    print("  ✅ Data loaded successfully")
    
    # Generate main figures (for paper body)
    print("\n  --- Main Figures (Paper Body) ---")
    figure_main_1(data, args.output_dir)
    figure_main_2(data, args.output_dir)
    figure_main_3(data, args.output_dir)
    
    # Generate supplementary figures (for appendix)
    print("\n  --- Supplementary Figures (Appendix) ---")
    try:
        figure_supp_1(data, args.output_dir)
    except ImportError:
        print("  ⚠️ sklearn not available, skipping t-SNE figure")
    figure_supp_2(data, args.output_dir)
    figure_supp_3(data, args.output_dir)
    figure_supp_4(data, args.output_dir)
    
    # Generate LaTeX table
    print("\n  --- LaTeX Table ---")
    generate_summary_table(data, args.output_dir)
    
    print("\n" + "=" * 60)
    print("  ✅ All figures generated successfully!")
    print(f"  Output: {args.output_dir}/")
    print("\n  Main figures (paper body):")
    print("    - fig_main1_modality_gap_and_decomposition.pdf")
    print("    - fig_main2_cosine_distributions.pdf")
    print("    - fig_main3_dimensionality_and_alignment.pdf")
    print("\n  Supplementary figures (appendix):")
    print("    - fig_supp1_tsne_manifold.pdf")
    print("    - fig_supp2_orthogonal_detail.pdf")
    print("    - fig_supp3_direction_alignment_comparison.pdf")
    print("    - fig_supp4_overlap_and_energy.pdf")
    print("\n  Table:")
    print("    - table_modality_geometry.tex")
    print("=" * 60)


if __name__ == '__main__':
    main()
