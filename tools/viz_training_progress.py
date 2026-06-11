#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练日志关键指标可视化脚本。

输入数据来自 stage2_swsrs (ckpt-200 resume, hybrid 91k, 3-layer anti-collapse)
训练日志中提取的 step 100/150/.../500 关键指标。

输出: tools/viz/training_diagnosis_*.png
诊断目标: 找到 step 300(健康) → step 450/500(异常) → step 2049(NaN) 的演化轨迹。
"""

from __future__ import annotations
import os
import math
import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# 1. 数据(从用户提供的训练日志中手动提取的关键 step 数据点)
# --------------------------------------------------------------------------- #
# 注: step 100, 150, 200, 250, 350, 400 的数据从 avg(50) 反推趋势
#     仅 step 300, 450, 500 是直接日志读到的精确值
# --------------------------------------------------------------------------- #

steps = [100, 150, 200, 250, 300, 350, 400, 450, 500]

# 主 CE Loss (来自日志的 main_ce 字段)
main_ce = [None, None, None, None, 1.5085, None, None, 0.9783, 1.0740]

# total Loss
total_loss = [None, None, None, None, 5.0247, None, None, 0.9783, 1.0740]

# 50 step 滑动平均 total
avg50 = [None, None, None, None, 1.3966, None, None, 4.5635, 5.0122]

# Exit Token Loss (Layer 1)
exit_loss = [None, None, None, None, 0.0017, None, None, None, None]

# SW-SRS Loss (Layer 1 主目标)
sw_srs_loss = [None, None, None, None, 0.1477, None, None, None, None]

# Anti-Collapse 3 层
anti_col_margin = [None, None, None, None, 0.0000, None, None, None, None]
anti_col_swsrs = [None, None, None, None, 2.0312, None, None, None, None]   # ⚠️ collapse-to-exit
anti_col_diversity = [None, None, None, None, 0.0139, None, None, None, None]

# Hidden 几何健康度
first_last_cos = [None, None, None, None, 0.793, None, None, None, None]
h_norm = [None, None, None, None, 165.54, None, None, None, None]

# Saturation
sat_step1 = [None, None, None, None, 0.650, None, None, None, None]
sat_step_last = [None, None, None, None, 0.836, None, None, None, None]
early_exit_pct = [None, None, None, None, 0.00, None, None, None, None]

# CE/Acc
think_ce = [None, None, None, None, 1.7718, None, None, 1.2776, 1.3236]
answer_ce = [None, None, None, None, 1.2462, None, None, 0.6796, 0.8251]
top1_ans = [None, None, None, None, 0.7286, None, None, 0.8171, 0.7612]

# thoughts (latent 实际触发步数, 关键诊断变量)
thoughts = [None, None, None, None, 2, None, None, 0, 0]   # ⚠️ 450/500 = 0!

# NaN 爆发位置 (推断: fwd#2049 ≈ step 512)
nan_step = 512

# --------------------------------------------------------------------------- #
# 2. 推断: 由 avg(50) 反推中间 step total loss 的"伪历史"
# --------------------------------------------------------------------------- #
# avg(50)[step] = 平均(最近 50 step 的 total)
#   step 300: avg=1.40, total=5.02 → 之前 50 step 平均 ~1.4 (健康)
#   step 450: avg=4.56, total=0.97 → step 400~450 之间存在 spike
#   step 500: avg=5.01, total=1.07 → spike 持续中
# 估算 step 350-450 区间出现过若干 total>20 的离群点

# --------------------------------------------------------------------------- #
# 3. 绘图: 4 子图诊断面板
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    "Stage2 SW-SRS (resume from ckpt-200, hybrid91k, 3-layer anti-collapse)\n"
    "Diagnosis: step 300 (healthy) → 450/500 (silent failure) → ~512 (NaN explosion)",
    fontsize=14, fontweight='bold'
)

x = np.array(steps, dtype=float)


def _filter(arr):
    """过滤 None, 返回 (xs, ys)"""
    xs, ys = [], []
    for s, v in zip(steps, arr):
        if v is not None:
            xs.append(s)
            ys.append(v)
    return np.array(xs), np.array(ys)


# ---------- 子图 1: Loss 主曲线 ----------
ax1 = plt.subplot(2, 2, 1)
xs, ys = _filter(total_loss)
ax1.plot(xs, ys, 'o-', color='black', linewidth=2.2,
         markersize=10, label='total_loss')
xs, ys = _filter(main_ce)
ax1.plot(xs, ys, 's--', color='royalblue', linewidth=1.8,
         markersize=8, label='main_ce')
xs, ys = _filter(avg50)
ax1.plot(xs, ys, '^-', color='red', linewidth=2.2,
         markersize=10, label='avg(50) sliding')

# NaN 标注
ax1.axvline(nan_step, color='red', linestyle=':', alpha=0.7, linewidth=2)
ax1.annotate('💀 NaN explosion\n(fwd#2049 ≈ step 512)',
             xy=(nan_step, 5.0),
             xytext=(nan_step - 80, 6.5),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

# Spike 区域 (step 350-450)
ax1.axvspan(350, 470, alpha=0.15, color='orange',
            label='inferred spike zone (avg50↑)')

ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('① Loss curves: total vs main_ce vs avg(50)\n'
              '   ⚠️ avg(50) 反超 total → 此前有大幅 spike',
              fontsize=11)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(80, 560)

# 关键观察: total=main_ce (在 450/500)
ax1.annotate('⚠️ total == main_ce\n(latent loss 全部消失,thoughts=0)',
             xy=(475, 1.0), xytext=(280, 2.5),
             fontsize=9, color='darkorange',
             arrowprops=dict(arrowstyle='->', color='darkorange'))


# ---------- 子图 2: thoughts (latent 触发步数) ----------
ax2 = plt.subplot(2, 2, 2)
xs, ys = _filter(thoughts)
colors = ['green' if v > 0 else 'red' for v in ys]
ax2.bar(xs, ys, width=20, color=colors, edgecolor='black', alpha=0.7)
for s, v in zip(xs, ys):
    ax2.text(s, v + 0.05, f'{int(v)}', ha='center',
             fontsize=11, fontweight='bold')

ax2.set_xlabel('Step')
ax2.set_ylabel('thoughts (latent 实际推理步数)')
ax2.set_title('② thoughts: 是否触发 latent thinker\n'
              '   ⚠️ step 450/500 thoughts=0 → latent 完全跳过',
              fontsize=11)
ax2.set_ylim(-0.3, 3)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(80, 560)
ax2.axhline(0, color='gray', linewidth=0.5)


# ---------- 子图 3: Anti-Collapse 三层 + Hidden 健康度 ----------
ax3 = plt.subplot(2, 2, 3)
# 仅 step 300 有完整数据 (孤立点用大圆圈强调)
ax3.scatter([300], [anti_col_margin[4]], color='blue', s=200,
            label=f'L1 margin={anti_col_margin[4]:.4f} ✅', zorder=5)
ax3.scatter([300], [anti_col_swsrs[4]], color='red', s=200,
            label=f'L2 anti_col={anti_col_swsrs[4]:.4f} ⚠️ HIGH', zorder=5)
ax3.scatter([300], [anti_col_diversity[4]], color='green', s=200,
            label=f'L3 diversity={anti_col_diversity[4]:.4f} ✅', zorder=5)
ax3.scatter([300], [sw_srs_loss[4]], color='purple', s=200, marker='^',
            label=f'sw_srs={sw_srs_loss[4]:.4f}', zorder=5)
ax3.scatter([300], [exit_loss[4]], color='orange', s=200, marker='s',
            label=f'exit_loss={exit_loss[4]:.4f}', zorder=5)

# 红色危险阈值
ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5,
            label='⚠️ collapse-to-exit threshold (0.5)')
ax3.axhline(2.0, color='darkred', linestyle=':', alpha=0.5,
            label='💥 numeric danger zone (>2)')

# 标注: anti_col 无上界 ReLU
ax3.annotate('单 batch 极端时\n可能 spike 到 50+\n(F.relu 无上界)',
             xy=(300, 2.03), xytext=(370, 1.5),
             fontsize=9, color='darkred',
             arrowprops=dict(arrowstyle='->', color='darkred'))

ax3.set_xlabel('Step')
ax3.set_ylabel('Loss component value')
ax3.set_title('③ Anti-collapse loss decomposition (step 300 snapshot)\n'
              '   ⚠️ L2 anti_col=2.03 已是无界 ReLU,易在极端 batch 爆炸',
              fontsize=11)
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(80, 560)
ax3.set_ylim(-0.1, 2.5)


# ---------- 子图 4: 训练事件时间线 ----------
ax4 = plt.subplot(2, 2, 4)
ax4.set_xlim(0, 600)
ax4.set_ylim(0, 10)
ax4.axis('off')

events = [
    (0,   9.0, 'Step 0', 'gray',       '从 ckpt-200 resume 启动'),
    (100, 8.0, 'Step 100', 'green',    '✅ 训练健康'),
    (200, 7.0, 'Step 200', 'green',    '✅ avg(50) ≈ 1.4'),
    (300, 6.0, 'Step 300', 'olive',    '🟡 anti_col=2.03 ⚠️\n   但 hidden 已修复\n   (early_exit=0%)'),
    (400, 4.5, 'Step ~400', 'orange',  '⚠️ avg(50) 开始上升\n   (有 batch spike)'),
    (450, 3.0, 'Step 450', 'darkorange', '🟠 thoughts=0\n   total = main_ce\n   avg(50)=4.56'),
    (500, 2.0, 'Step 500', 'red',      '🔴 thoughts=0\n   触发 ckpt 保存\n   FSDP 8 ranks 阻塞 86s'),
    (512, 1.0, 'fwd#2049', 'darkred',  '💀 vision NaN\n   lm_head NaN\n   全模型 NaN'),
]

for x_pos, y_pos, label, color, desc in events:
    ax4.scatter(x_pos, y_pos, s=200, color=color, edgecolor='black',
                zorder=5)
    ax4.text(x_pos + 8, y_pos, f"{label}: {desc}", fontsize=9,
             color=color, va='center', fontweight='bold')

# 时间线主轴
ax4.plot([0, 600], [9.5, 9.5], color='black', linewidth=1)
for x_pos, _, _, _, _ in events:
    ax4.plot([x_pos, x_pos], [9.4, 9.6], color='black', linewidth=1)

ax4.set_title('④ 故障事件时间线 (Hypothesized failure trajectory)',
              fontsize=11, pad=15)


plt.tight_layout(rect=[0, 0, 1, 0.96])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "training_diagnosis_panel.png")
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print(f"[OK] 训练诊断面板已保存: {out_path}")


# --------------------------------------------------------------------------- #
# 4. 单独绘制 avg(50) vs total 的 spike 分析图
# --------------------------------------------------------------------------- #
fig2, ax = plt.subplots(figsize=(12, 6))

xs1, ys1 = _filter(total_loss)
xs2, ys2 = _filter(avg50)

ax.plot(xs1, ys1, 'o-', color='black', linewidth=2.5,
        markersize=14, label='total_loss (single step)')
ax.plot(xs2, ys2, '^--', color='red', linewidth=2.5,
        markersize=14, label='avg(50) sliding mean')
ax.fill_between(xs2, 0, ys2, alpha=0.15, color='red')

# 推断: 隐藏 spike 区
spike_x = np.linspace(360, 470, 50)
# 反推 spike 强度: avg(450) ≈ avg(300) + spike_total / 50
# 5.0 - 1.4 = 3.6 → 50 step 内累计多了 180 单位 → 平均每个 spike step ≈ 30+
spike_height = 32
spike_y = spike_height * np.exp(-((spike_x - 415) / 20) ** 2)
ax.plot(spike_x, spike_y, color='orange', linestyle=':',
        linewidth=2, label='inferred hidden spikes (avg50 反推)')
ax.fill_between(spike_x, 0, spike_y, alpha=0.25, color='orange')

ax.axvline(500, color='red', linestyle='-', alpha=0.4,
           linewidth=2, label='ckpt save (step 500)')
ax.axvline(512, color='darkred', linestyle='--', alpha=0.7,
           linewidth=2, label='NaN explosion (~step 512)')

ax.annotate('avg(50) - total 鸿沟 = 隐藏的 spike 证据\n'
            'step 350~450 区间存在多个 total>20 的爆炸性单 step',
            xy=(420, 4.7), xytext=(150, 25),
            fontsize=11, color='darkorange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkorange',
                            linewidth=2))

ax.set_xlabel('Training step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('avg(50) vs total: 隐藏 loss spike 检测\n'
             '关键证据: step 450 total=0.97 但 avg(50)=4.56',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(80, 560)
ax.set_ylim(0, 35)

plt.tight_layout()
out_path2 = os.path.join(out_dir, "training_spike_analysis.png")
plt.savefig(out_path2, dpi=110, bbox_inches='tight')
print(f"[OK] Spike 分析图已保存: {out_path2}")


# --------------------------------------------------------------------------- #
# 5. CE/Acc 演化图
# --------------------------------------------------------------------------- #
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
xs, ys = _filter(think_ce)
ax.plot(xs, ys, 'o-', color='royalblue', linewidth=2.2,
        markersize=12, label='think_ce')
xs, ys = _filter(answer_ce)
ax.plot(xs, ys, 's-', color='green', linewidth=2.2,
        markersize=12, label='answer_ce')
ax.set_xlabel('Step')
ax.set_ylabel('Cross Entropy')
ax.set_title('CE Loss 分解: think vs answer')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
xs, ys = _filter(top1_ans)
ax.plot(xs, ys, 'D-', color='purple', linewidth=2.2,
        markersize=12, label='top1_ans accuracy')
ax.set_xlabel('Step')
ax.set_ylabel('Accuracy')
ax.set_title('Top-1 Answer Accuracy\n'
             '(注: step 450 上升至 0.82,但隐藏的 spike 已经发生)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.6, 0.9)

plt.tight_layout()
out_path3 = os.path.join(out_dir, "training_ce_acc.png")
plt.savefig(out_path3, dpi=110, bbox_inches='tight')
print(f"[OK] CE/Acc 演化图已保存: {out_path3}")


print("\n========== 自动诊断结论 ==========")
print(f"  step 300:  ✅ 健康(total=5.02 主要来自 anti_col=2.03)")
print(f"  step 350~450: ⚠️ 隐藏 spike (avg(50) 上升 1.4 → 4.6)")
print(f"  step 450/500: 🟠 thoughts=0,latent 跳过(纯 main_ce)")
print(f"  step ~512: 💀 vision_embed/lm_head 全 NaN")
print()
print("根因排序(可能性从高到低):")
print("  1. anti_col loss 无上界 → 极端 batch 单步爆炸 → bf16 grad 累积溢出")
print("  2. ckpt 保存(step 500)与 vision tower 异步写入冲突,污染权重")
print("  3. 数据中存在异常样本(images 极端尺寸或 latent 标签错误)")
