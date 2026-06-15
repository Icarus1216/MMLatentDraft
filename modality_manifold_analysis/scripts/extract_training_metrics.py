#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_training_metrics.py (v2 — 精简版)
===========================
从 trainer_state.json 提取训练过程中的模态几何指标，
仅保留对分析和论证有效的指标。

产出:
  - modality_manifold_analysis/results/training_metrics.json
  - modality_manifold_analysis/results/training_phase_summary.json

指标分类:
  【核心保留】
  1. 模态对齐: cos_hv__mean, cos_ht__mean
  2. 过渡指标: derived__alpha_transition, derived__mti,
               derived__angle_hv_deg, derived__angle_ht_deg,
               derived__vl_equidistance, derived__capacity
  3. 几何效率: hidden_geometry__diag_score, hidden_geometry__h_norm,
               hidden_geometry__shift_kl, hidden_geometry__adj_cos_mean
  4. CKA: cka_hv_approx, cka_ht_approx, cka_hv_ht_ratio
  5. 训练信号: training__acc, training__total_loss

  【辅助保留 — 任务拆解】(fig1 需要)
  6. cos_hv__abstract, cos_hv__concrete, cos_hv__unified, cos_hv__bridge
  7. cos_ht__abstract, cos_ht__concrete, cos_ht__unified, cos_ht__bridge

  【辅助保留 — 思考效率】(fig5 Panel H 需要)
  8. training__thought_steps, training__thought_count

  【辅助保留 — 视觉注意力】(fig5 Panel F 需要)
  9. vision_attn__alpha_mean, vision_attn__vis_count

  【已移除】
  - training__main_ce, training__think_ce, training__answer_ce
  - sw_srs__q_entropy_first, sw_srs__q_entropy_last, sw_srs__q_entropy_mean
  - sw_srs__topk_hit_ratio, sw_srs__sw_srs_loss
  - tgvr__attn_entropy, tgvr__cos_h_v_first, tgvr__cos_h_v_last,
    tgvr__cos_h_v_mean, tgvr__topk_recall, tgvr__v_pos_norm
  - vision_attn__vis_loss, vision_attn__cos_h_r_mean,
    vision_attn__top1_sim, vision_attn__topk_sim_mean
  - hidden_geometry__first_last_cos, hidden_geometry__sat_step1

用法:
  python3 extract_training_metrics.py
  python3 extract_training_metrics.py --ckpt_path /path/to/trainer_state.json
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ============================================================
# 0. 路径
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_CKPT_PATH = os.path.join(
    PROJ_ROOT,
    'outputs/<RUN_NAME>/checkpoint-1200/trainer_state.json'
)


# ============================================================
# 1. 提取训练曲线指标 (精简版)
# ============================================================
def extract_metrics(ckpt_path: str, verbose: bool = True):
    """从 trainer_state.json 提取对分析和论证有效的指标"""
    with open(ckpt_path) as f:
        data = json.load(f)

    logs = data['log_history']
    metric_entries = [e for e in logs if 'nld/vis_cos_h_v_mean' in e]

    if verbose:
        print(f"[load] {len(metric_entries)} metric entries from {ckpt_path}")

    # ==========================================================
    # 保留的指标组 (精简后)
    # ==========================================================

    # 【核心】模态对齐: H 与 V / H 与 T 的余弦相似度
    cos_hv = {
        'mean':    'nld/vis_cos_h_v_mean',
        'unified': 'nld/vis_cos_h_v_unified',
        'abstract': 'nld/vis_cos_h_v_abstract',
        'concrete': 'nld/vis_cos_h_v_concrete',
        'bridge':  'nld/vis_cos_h_v_bridge',
    }

    cos_ht = {
        'mean':    'nld/vis_cos_h_t_mean',
        'unified': 'nld/vis_cos_h_t_unified',
        'abstract': 'nld/vis_cos_h_t_abstract',
        'concrete': 'nld/vis_cos_h_t_concrete',
        'bridge':  'nld/vis_cos_h_t_bridge',
    }

    # 【核心】隐状态几何
    hidden_geometry = {
        'h_norm':       'collapse/h_norm_mean',
        'diag_score':   'collapse/h_stage_diag_score',
        'adj_cos_mean': 'collapse/h_adj_cos_mean',
        'shift_kl':     'collapse/h_stage_shift_kl',
    }

    # 【核心】CKA 近似
    cka = {
        'cka_hv_approx':    'cka_hv_approx_placeholder',  # 需要从 embeddings 计算
        'cka_ht_approx':    'cka_ht_approx_placeholder',
        'cka_hv_ht_ratio':  'cka_ratio_placeholder',
    }

    # 【核心】训练信号
    training = {
        'acc':           'acc/top1_answer',
        'total_loss':    'loss/total',
        'thought_steps': 'nld/num_thought_steps_mean',
        'thought_count': 'nld/thought_count',
    }

    # 【辅助】视觉注意力
    vision_attn = {
        'alpha_mean':   'nld/vis_alpha_mean',
        'vis_count':    'nld/vis_count',
    }

    # 组合所有指标组
    metric_groups = {
        'cos_hv': cos_hv,
        'cos_ht': cos_ht,
        'hidden_geometry': hidden_geometry,
        'training': training,
        'vision_attn': vision_attn,
    }

    # 提取步骤
    steps = [e.get('step', i) for i, e in enumerate(metric_entries)]

    # 提取所有指标
    result = {'steps': steps}
    for group_name, key_map in metric_groups.items():
        for short_name, json_key in key_map.items():
            values = []
            for e in metric_entries:
                v = e.get(json_key, None)
                values.append(v)
            result[f'{group_name}__{short_name}'] = values

    # ==========================================================
    # 【核心】派生指标
    # ==========================================================

    # 过渡模态指标: H 在 V-L 连线上的投影位置
    cos_hv_mean = np.array([v if v is not None else np.nan for v in result['cos_hv__mean']])
    cos_ht_mean = np.array([v if v is not None else np.nan for v in result['cos_ht__mean']])

    denom = np.abs(cos_hv_mean) + np.abs(cos_ht_mean)
    denom = np.where(denom == 0, 1e-8, denom)
    alpha_transition = np.abs(cos_hv_mean) / denom
    result['derived__alpha_transition'] = alpha_transition.tolist()

    # 模态分离度: 角度差
    angle_hv = np.degrees(np.arccos(np.clip(cos_hv_mean, -1, 1)))
    angle_ht = np.degrees(np.arccos(np.clip(cos_ht_mean, -1, 1)))
    result['derived__angle_hv_deg'] = angle_hv.tolist()
    result['derived__angle_ht_deg'] = angle_ht.tolist()
    result['derived__angle_diff'] = (angle_hv - angle_ht).tolist()

    # V-L 等距度
    result['derived__vl_equidistance'] = (np.abs(cos_hv_mean - cos_ht_mean)).tolist()

    # 表示能力
    h_norm = np.array([v if v is not None else np.nan for v in result['hidden_geometry__h_norm']])
    diag = np.array([v if v is not None else np.nan for v in result['hidden_geometry__diag_score']])
    result['derived__capacity'] = (h_norm * diag / 100).tolist()

    # MTI: 模态过渡度
    # MTI = cos(H,V) × (1-cos(H,T)) × diag_score
    dist_to_text = 1 - cos_ht_mean
    mti = cos_hv_mean * dist_to_text * diag
    result['derived__mti'] = mti.tolist()

    # CKA 近似计算
    cka_hv = np.abs(cos_hv_mean) * diag
    cka_ht = np.abs(cos_ht_mean) * diag
    result['cka_hv_approx'] = cka_hv.tolist()
    result['cka_ht_approx'] = cka_ht.tolist()
    result['cka_hv_ht_ratio'] = (cka_hv / np.maximum(cka_ht, 1e-8)).tolist()

    return result, metric_entries


# ============================================================
# 2. 训练阶段分析
# ============================================================
def analyze_training_phases(result: dict, n_phases: int = 4):
    """自动划分训练阶段并计算统计摘要"""
    steps = np.array(result['steps'])
    n = len(steps)
    phase_size = n // n_phases

    phases = []
    for i in range(n_phases):
        lo = i * phase_size
        hi = (i + 1) * phase_size if i < n_phases - 1 else n
        idx = slice(lo, hi)

        phase = {
            'name': f'Phase {chr(65+i)}',
            'step_range': f'{steps[lo]}-{steps[min(hi-1, n-1)]}',
            'n_steps': hi - lo,
        }

        # 仅保留核心指标的均值
        key_metrics = [
            'cos_hv__mean', 'cos_ht__mean',
            'hidden_geometry__diag_score', 'hidden_geometry__h_norm',
            'hidden_geometry__shift_kl', 'hidden_geometry__adj_cos_mean',
            'derived__alpha_transition', 'derived__mti',
            'derived__angle_hv_deg', 'derived__angle_ht_deg',
            'derived__capacity',
            'cka_hv_approx', 'cka_ht_approx', 'cka_hv_ht_ratio',
            'vision_attn__alpha_mean', 'vision_attn__vis_count',
            'training__acc', 'training__total_loss',
            'training__thought_steps', 'training__thought_count',
        ]

        for k in key_metrics:
            vals = result.get(k, [])
            if len(vals) >= hi:
                arr = np.array([v if v is not None else np.nan for v in vals[idx]])
                phase[f'{k}__mean'] = float(np.nanmean(arr))
                phase[f'{k}__std'] = float(np.nanstd(arr))

        phases.append(phase)

    return phases


# ============================================================
# 3. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default=DEFAULT_CKPT_PATH)
    parser.add_argument('--output_dir', default=RESULTS_DIR)
    args = parser.parse_args()

    print("=" * 70)
    print("  Modality Manifold Analysis: Training Metrics Extraction (v2)")
    print("=" * 70)

    # 1. 提取指标
    result, raw_entries = extract_metrics(args.ckpt_path)
    print(f"  ✅ Extracted {len(result['steps'])} steps, {len(result)-1} metric series")

    # 2. 训练阶段分析
    phases = analyze_training_phases(result, n_phases=4)
    print(f"  ✅ Analyzed {len(phases)} training phases")

    # 3. 保存
    out_metrics = os.path.join(args.output_dir, 'training_metrics.json')
    with open(out_metrics, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved: {out_metrics}")

    out_phases = os.path.join(args.output_dir, 'training_phase_summary.json')
    with open(out_phases, 'w') as f:
        json.dump(phases, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved: {out_phases}")

    # 打印关键发现
    print("\n" + "=" * 70)
    print("  Key Findings")
    print("=" * 70)
    steps = result['steps']
    cos_hv = result['cos_hv__mean']
    cos_ht = result['cos_ht__mean']
    diag = result['hidden_geometry__diag_score']
    acc = result['training__acc']

    print(f"  cos(H,V):  {cos_hv[0]:.4f} → {cos_hv[-1]:.4f}  (Δ={cos_hv[-1]-cos_hv[0]:+.4f})")
    print(f"  cos(H,T):  {cos_ht[0]:.4f} → {cos_ht[-1]:.4f}  (Δ={cos_ht[-1]-cos_ht[0]:+.4f})")
    print(f"  diag_score: {diag[0]:.4f} → {diag[-1]:.4f}  (Δ={diag[-1]-diag[0]:+.4f})")
    print(f"  accuracy:   {acc[0]:.4f} → {acc[-1]:.4f}  (Δ={acc[-1]-acc[0]:+.4f})")

    # 训练阶段变化
    for p in phases:
        print(f"\n  {p['name']} ({p['step_range']}):")
        for k in ['cos_hv__mean__mean', 'cos_ht__mean__mean',
                   'hidden_geometry__diag_score__mean',
                   'derived__alpha_transition__mean',
                   'derived__mti__mean',
                   'training__acc__mean']:
            if k in p:
                print(f"    {k}: {p[k]:.4f}")

    print("\n✅ Done.")


if __name__ == '__main__':
    main()
