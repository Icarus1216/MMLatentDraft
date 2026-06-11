---
title: Modality Analysis — Geometry-based Analysis Section
---

# Modality Analysis 目录

**目的**：为 *Adaptive Latent Thinker* 论文补充一个符合顶会（ICLR / NeurIPS / CVPR）标准的
"模态几何分析" 小节（analysis section，约 5-8 页正文 + 附录）。

本目录**独立于主代码**，所有分析脚本与产物均自包含，读已有 `outputs/` 中的权重/embedding 产物，
不修改训练代码。

## 目录结构

```
modality_analysis/
├── README.md                               # ← 本文件, 路线图 + 复现命令
├── scripts/
│   ├── stage1_stat_anisotropy.py           # Stage-1: 模块 A + 模块 B (单组几何)
│   ├── run_stage1.sh                       # Stage-1 启动 (含 conda)
│   ├── extract_embeddings_with_native.py   # [NEW] 抽取 latent+native 配对 hidden (含 5 层 layer-wise)
│   ├── run_extract_embeddings.sh           # [NEW] 抽取脚本启动器
│   ├── stage1b_pair_native_vs_latent.py    # [NEW] Stage-1b: latent vs native 配对对比
│   ├── run_stage1b.sh                      # [NEW] Stage-1b 启动器
│   ├── stage2_layerwise_ckpt.py            # (待建) Stage-2: 多 ckpt 对比
│   ├── run_stage2.sh                       # (待建)
│   ├── stage3_probe_causal.py              # (待建) Stage-3: 因果干预
│   └── run_stage3.sh                       # (待建)
├── results/                                # JSON / TXT 指标产物
└── figures/                                # PDF 图 (论文可直接用)
```

## 总体路线（3 阶段）

| 阶段 | 模块 | 目标 | 输入 | 产物 | 预计耗时 |
|---|---|---|---|---|---|
| **1** | A + B | 统计严谨化 + 内在维数 / 各向异性 | 已有 `embeddings.pt` | 3 张 PDF + 1 JSON + 1 TXT | 20 min |
| **1b** | Paired Δ | **latent vs native 配对几何差** | 新抽 `embeddings_native_vs_latent.pt` | 1-5 张 PDF + 配对 JSON | 1.5-2 h（含新抽取）|
| **2** | C + D + E | Layer-wise + 多 ckpt 对比 + Task 鲁棒性 | base / s1 / s2-{100,200,300,400} 6 ckpt + 4 benchmark | 6 张 PDF + 关联 JSON | 1.5 天 GPU |
| **3** | F + G | Linear probe + Causal intervention | stage2-400 + 最优 ckpt | 2 张 PDF + causal 日志 | 2 天 GPU |

## Stage-1b（新增）— Latent vs Native 配对分析

**方法学动机**：Stage-1 报告的 `H cone=16.8°, ID=10, PR=3` 可能来自 LLM 本身的各向异性（Gao 2019），
不能单独归因于 latent thinker。Stage-1b 通过**同一输入、两次前向**得到配对样本，
允许做 paired bootstrap / paired permutation test，计算 **Δcone / ΔPR / ΔID / Δθ_V**。

### 抽取 (需 1 张 GPU, 500 样本约 1-2 h)

```bash
bash modality_analysis/scripts/run_extract_embeddings.sh \
    --num_samples 500 \
    --layers "0,8,16,24,-1"
# 产物: modality_analysis/results/embeddings_native_vs_latent.pt
```

该脚本对每个样本:
- Latent 通路: prefill → 遇 `<|latent|>` → 调 `model.latent_thinker(...)` → 记录每步 `thought_output` + 饱和度 α
- Native 通路: 复用同一 prefill cache → 遇 `<|latent|>` 当普通 token → 记录同一 cache_position 处的 LLM 原生 hidden
- V/L: 从 prefill hidden 对 image token / text token 做 mean pool
- 所有 hidden 同时抽取 5 层 layer-wise (0, 8, 16, 24, last)

### 配对分析

```bash
# 最常用: 用 latent step 0 vs native (在 <|latent|> 触发位置), last layer
bash modality_analysis/scripts/run_stage1b.sh

# latent 后半 step (α≈0) vs answer-start native
bash modality_analysis/scripts/run_stage1b.sh --latent_step 1 --native_reference answer

# 扫描所有层 → 得 layer-wise Δ 曲线
bash modality_analysis/scripts/run_stage1b.sh --layer_idx all
```

### 产出

| 文件 | 内容 |
|---|---|
| `results/stage1b_paired_metrics_<step>_<native>_<layer>.json` | 每个组合的所有数值 |
| `results/stage1b_layerwise_<step>_<native>_all_layers.json` | layer-wise 全量 |
| `figures/stage1b_paired_delta_*.pdf` | 单层: cone/PR/ID 对比 + paired Δangle→V/L + gap 表 |
| `figures/stage1b_layerwise_delta_*.pdf` | 多层曲线: Δcone / ΔPR / ΔID / Δθ 随层数变化 |

### 核心配对指标

- **Δcone = cone(H_latent) − cone(H_native)**
  - 若 <0: latent step 进一步压缩表示（negative finding, 论文重要）
  - 若 >0: latent step 扩展了思维空间（positive finding）
- **Δθ_V (paired, per-sample) + 95% CI + paired permutation p-value**
  - 若 mean < 0 且 p<0.05: latent step 显著把 H 拉近 V
  - 若 mean > 0 且 p<0.05: latent step 反而远离 V

## Stage-1 模块清单（本次交付）

### Module A — Statistical Rigor

| 指标 | 方法 | 参考 |
|---|---|---|
| A.1 | Bootstrap 95% CI (n=1000) on gap / cone / intra-cos | standard |
| A.2 | von Mises-Fisher MLE for concentration κ | Banerjee-Dhillon-Ghosh-Sra, JMLR 2005 |
| A.3 | Permutation test (n=2000) for cone_H vs cone_V, etc. | standard |

### Module B — Anisotropy / Intrinsic Dimension

| 指标 | 方法 | 参考 |
|---|---|---|
| B.1 | TwoNN intrinsic dimension | Facco et al., Sci. Rep. 2017 |
| B.1 | Levina-Bickel MLE ID (k=10) | Levina & Bickel, NeurIPS 2005 |
| B.2 | PCA spectrum / PC1 share / IsoScore | Rudman et al., EMNLP 2022 |
| B.2 | Participation Ratio (effective rank) | Gao et al., 2017 |
| B.3 | Cone-shape diagnosis ("球冠" vs "1D 管") | our analysis |

## 使用方法

### 先探测 embeddings.pt 结构（推荐第一步）

```bash
bash modality_analysis/scripts/run_stage1.sh --probe
```

如果自动识别 V/L/H 失败，查看打印输出后在 `stage1_stat_anisotropy.py` 的
`key_aliases` 中补充键名即可。

### 跑完整 Stage-1

```bash
bash modality_analysis/scripts/run_stage1.sh
```

产物：

- `results/stage1_metrics.json`：所有数值 + 95% CI
- `results/stage1_summary.txt`：人类可读摘要
- `figures/stage1_geometry_with_CI.pdf`：几何量带置信区间的柱状图（Fig. 1 候选）
- `figures/stage1_intrinsic_dim.pdf`：内在维数对比图（Fig. 2 候选）
- `figures/stage1_anisotropy.pdf`：各向异性与 PCA 谱图（Fig. 3 候选）

### 自定义参数

```bash
bash modality_analysis/scripts/run_stage1.sh --n_boot 2000 --n_perm 5000
```

## 与现有产物的对应关系

| 现有产物 | 本目录对应 |
|---|---|
| `outputs/modality_geometry_v2/geometry_metrics.json` | 被 `stage1_metrics.json` 替代（含 CI + vMF + ID） |
| `outputs/modality_geometry_v2/modality_geometry_*.png` | 被 `figures/stage1_*.pdf` 替代（矢量 + 错误条） |
| `outputs/training_curves/stage2/visualize_3d_modality_dynamics.py` | 保留为 "训练动力学" 可视化, Stage-2 会扩展多 ckpt 版本 |

## 已知限制（Stage-1 后需要 Stage-2 补全）

- Stage-1 仅看 **last layer** 的 hidden state（ckpt-400 单个）→ Stage-2 做 layer-wise + 多 ckpt
- Stage-1 仅看 **单数据集 (ERQA)** → Stage-2 扩展到 MMStar / VSI / BLINK
- Stage-1 无因果证据 → Stage-3 做 activation patching

## 论文写作建议段落规划

```
Section X. Geometric Analysis of Adaptive Latent Thinker
  X.1 Preliminaries (modality cone, gap, vMF)
  X.2 Static geometry with statistical guarantees           [Stage-1 Module A]
  X.3 Intrinsic dimensionality and representation collapse  [Stage-1 Module B]
  X.4 Training dynamics across checkpoints                  [Stage-2 Module D]
  X.5 Layer-wise evolution                                  [Stage-2 Module C]
  X.6 Task-conditioned geometry                             [Stage-2 Module E]
  X.7 Mechanistic probes                                    [Stage-3 Module F]
  X.8 Causal intervention                                   [Stage-3 Module G]
```
