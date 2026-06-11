#!/usr/bin/env python3
"""
重绘 Efficiency Comparison 图 (ACL style)
用法: python plot_efficiency_acl.py
"""

import os
import json

def plot_efficiency_acl(summary_stats_path, output_dir, dpi=300, latent_flops=None, latent_flops_rel_err=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        from matplotlib.patches import Rectangle
    except ImportError as e:
        print(f"[Plot] ⚠️ matplotlib/numpy 未安装: {e}")
        return

    with open(summary_stats_path, "r") as f:
        summary = json.load(f)

    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.titlesize'] = 12
    matplotlib.rcParams['axes.labelsize'] = 11
    matplotlib.rcParams['axes.formatter.useoffset'] = False  # 禁用科学计数法偏移

    # 三个指标用不同透明度区分（灰度示意，兼容黑白打印）
    alpha_map = [0.55, 0.85, 1.00]  # FLOPs更深, Latency中等, Tokens最亮
    # 模型专属颜色
    color_inst  = '#78B3CE'
    color_think = '#4D869C'
    color_ours  = '#7D71A0'
    color_bg    = '#FFFFFF'
    text_color  = '#222222'
    grid_color  = '#E0E0E0'
    
    configs = ["Qwen3-VL\nInstruct", "LatentDraft\n(Ours)", "Qwen3-VL\nThinking"]
    model_colors = [color_inst, color_ours, color_think]

    flops_data = [summary.get(k, 0) or 0 for k in
                  ["base_flops_mean", "with_latent_flops_mean", "thinking_flops_mean"]]
    if latent_flops is not None:
        flops_data[1] = latent_flops * 1e9
    flops_std  = [summary.get(k, 0) or 0 for k in
                  ["base_flops_std", "with_latent_flops_std", "thinking_flops_std"]]
    if latent_flops_rel_err is not None and latent_flops is not None:
        flops_std[1] = latent_flops * 1e9 * latent_flops_rel_err
    latency_data = [summary.get(k, 0) or 0 for k in
                    ["base_latency_mean", "with_latent_latency_mean", "thinking_latency_mean"]]
    latency_std  = [summary.get(k, 0) or 0 for k in
                    ["base_latency_std", "with_latent_latency_std", "thinking_latency_std"]]
    length_data = [summary.get(k, 0) or 0 for k in
                   ["base_decode_steps_mean", "with_latent_decode_steps_mean", "thinking_decode_steps_mean"]]

    flops_g   = [v / 1e9 for v in flops_data]
    latency_s = latency_data
    length_tok = length_data

    # 统一归一化到 0-100（百分比），基值为各指标最大值
    norm = lambda vals: [v / max(vals) * 100 for v in vals]
    flops_norm  = norm(flops_g)
    lat_norm    = norm(latency_s)
    len_norm    = norm(length_tok)

    std_norm_flops = [s / 1e9 / max(flops_g) * 100 for s in flops_std]
    std_norm_lat   = [s / max(latency_s) * 100 for s in latency_std]

    # === 单图：分组柱状图，每组3根柱子，按模型配色 ===
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    fig.patch.set_facecolor(color_bg)
    ax.set_facecolor(color_bg)

    n_models = len(configs)
    n_indicators = 3
    group_width = 0.50
    bar_width = group_width / 3
    x = np.arange(n_indicators)
    gap = 0.02

    # 模型在每组内的偏移
    offsets = [-bar_width - gap, 0, bar_width + gap]
    indicator_names = [r"FLOPs ($\times 10^9$)", "Latency (s)", "Tokens"]
    norm_vals = [flops_norm, lat_norm, len_norm]
    std_vals  = [std_norm_flops, std_norm_lat, [0.0, 0.0, 0.0]]
    raw_vals  = [flops_g, latency_s, length_tok]

    for ind_idx in range(n_indicators):  # 对每个指标（按指标分组）
        for mdl_idx in range(n_models):   # 对每个模型
            xc = x[ind_idx] + offsets[mdl_idx]
            color = model_colors[mdl_idx]
            height = norm_vals[ind_idx][mdl_idx]
            bar = ax.bar(
                xc, height, bar_width,
                color=color, alpha=1.0,
                edgecolor='none',
                linewidth=0,
                zorder=3
            )
            # 数值标注在柱形上方
            raw_val = raw_vals[ind_idx][mdl_idx]
            if raw_val != 0:
                if ind_idx == 0:
                    txt = f"{raw_val:.2f}"
                elif ind_idx == 1:
                    txt = f"{raw_val:.2f}s"
                else:
                    txt = f"{raw_val:.0f}"
                ax.text(
                    xc, height + 2.5, txt,
                    ha='center', va='bottom',
                    fontsize=7.0, color='#444444', fontweight='bold',
                    zorder=5,
                    clip_on=False
                )



    # 自定义图例：仅模型颜色区分，指标通过底部文字标注说明
    from matplotlib.patches import Patch
    legend_elements = []
    for ci, (cfg, col) in enumerate(zip(configs, model_colors)):
        legend_elements.append(Patch(facecolor=col, edgecolor='white', linewidth=0.8, label=cfg.replace('\n', ' ')))

    ax.set_xlabel("")
    ax.set_ylabel("Normalized Value (%)", fontsize=11, color=text_color)
    
    # 彻底隐藏默认x轴刻度标签
    ax.tick_params(axis='x', labelbottom=False)
    
    # === 底部文字标注（放在 axes 下方）===
    # 使用混合变换：x用数据坐标，y用axes坐标
    xaxis_transform = ax.get_xaxis_transform()
    
    # 指标名放在每个指标组下方居中
    indicator_short = [r"FLOPs", "Latency", "Tokens"]
    for ind_idx in range(n_indicators):
        ax.text(
            x[ind_idx], -0.06, indicator_short[ind_idx],
            ha='center', va='top',
            fontsize=11.0, color='#222222', fontweight='bold',
            transform=xaxis_transform
        )

    # y轴从0开始
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120])
    
    ax.tick_params(axis='y', labelcolor=text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.grid(axis='y', ls='-', lw=0.5, alpha=0.3, color=grid_color, zorder=0)

    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), frameon=True,
              fancybox=True, framealpha=0.95, edgecolor='#DDDDDD', fontsize=9.5,
              ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path = os.path.join(output_dir, "efficiency_comparison.png")
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"[Plot] 对比图已保存: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot efficiency comparison (ACL style)")
    parser.add_argument("--summary", default="./outputs/efficiency_analysis/summary_stats.json",
                        help="Path to summary_stats.json")
    parser.add_argument("--out_dir", default="/mnt/cephszjt/user_juntianzhang/LatentDraft/paper_tables_figures",
                        help="Output directory for the figure")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument("--latent_flops", type=float, default=None,
                        help="Override with_latent_flops_mean (in 1e9)")
    parser.add_argument("--latent_flops_rel_err", type=float, default=None,
                        help="Relative error for with_latent_flops (e.g. 0.23 for 23%%)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_efficiency_acl(args.summary, args.out_dir, dpi=args.dpi, latent_flops=args.latent_flops,
                        latent_flops_rel_err=args.latent_flops_rel_err)
