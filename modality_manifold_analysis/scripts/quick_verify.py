#!/usr/bin/env python3
"""快速验证 H 在训练过程中的模态位置变化"""
import json, numpy as np

RESULTS = '/mnt/cephszjt/user_juntianzhang/LatentDraft/modality_manifold_analysis/results'

with open(f'{RESULTS}/training_metrics.json') as f:
    data = json.load(f)
with open(f'{RESULTS}/training_phase_summary.json') as f:
    phases = json.load(f)
with open('/mnt/cephszjt/user_juntianzhang/LatentDraft/outputs/modality_geometry_v2/geometry_metrics.json') as f:
    baseline = json.load(f)

cos_hv = [v for v in data['cos_hv__mean'] if v is not None]
cos_ht = [v for v in data['cos_ht__mean'] if v is not None]
alpha = [v for v in data['derived__alpha_transition'] if v is not None]
vl_eq = [v for v in data['derived__vl_equidistance'] if v is not None]
angle_hv = [v for v in data['derived__angle_hv_deg'] if v is not None]
angle_ht = [v for v in data['derived__angle_ht_deg'] if v is not None]
mti = [v for v in data['derived__mti'] if v is not None]
diag = [v for v in data['hidden_geometry__diag_score'] if v is not None]
capacity = [v for v in data['derived__capacity'] if v is not None]

vl_gap = baseline.get('modality_gap_angle_vl', 83.28)

print("=" * 70)
print("  验证论断: 过渡模态是'H比训练前更接近V，同时不远离T'")
print("=" * 70)
print()
print("--- 核心指标变化 ---")
print(f"  cos(H,V): {cos_hv[0]:.4f} → {cos_hv[-1]:.4f}  (Δ={cos_hv[-1]-cos_hv[0]:+.4f}, {abs(cos_hv[-1]-cos_hv[0])/cos_hv[0]*100:.0f}%↓)")
print(f"  cos(H,T): {cos_ht[0]:.4f} → {cos_ht[-1]:.4f}  (Δ={cos_ht[-1]-cos_ht[0]:+.4f}, {abs(cos_ht[-1]-cos_ht[0])/abs(cos_ht[0])*100:.0f}%↑)")
print(f"  α_transition: {alpha[0]:.4f} → {alpha[-1]:.4f}  (Δ={alpha[-1]-alpha[0]:+.4f})")
print(f"  V-L等距度: {vl_eq[0]:.4f} → {vl_eq[-1]:.4f}  (Δ={vl_eq[-1]-vl_eq[0]:+.4f})")
print(f"  ∠(H,V): {angle_hv[0]:.1f}° → {angle_hv[-1]:.1f}°")
print(f"  ∠(H,T): {angle_ht[0]:.1f}° → {angle_ht[-1]:.1f}°")
print(f"  ∠(V,L): {vl_gap:.1f}° (基准)")
print(f"  MTI: {mti[0]:.4f} → {mti[-1]:.4f}")
print(f"  diag_score: {diag[0]:.4f} → {diag[-1]:.4f}")
print(f"  capacity: {capacity[0]:.4f} → {capacity[-1]:.4f}")

print()
print("--- 分阶段数据 ---")
for p in phases:
    print(f"\n  {p['name']} ({p['step_range']}):")
    for k in ['cos_hv__mean__mean', 'cos_ht__mean__mean', 'derived__alpha_transition__mean',
               'derived__mti__mean', 'derived__angle_hv_deg__mean', 'derived__angle_ht_deg__mean',
               'hidden_geometry__diag_score__mean', 'derived__capacity__mean']:
        v = p.get(k)
        if v is not None:
            print(f"    {k.split('__')[-1]}: {v:.4f}")

print()
print("=" * 70)
print("  论断验证结论")
print("=" * 70)
print()
print("  1. \"H比训练前更接近V\" — ⚠️ 不完全成立")
print(f"     cos(H,V) 从 {cos_hv[0]:.4f} 降至 {cos_hv[-1]:.4f} (↓{abs(cos_hv[-1]-cos_hv[0])/cos_hv[0]*100:.0f}%)")
print(f"     这意味着 H 与 V 的相似度降低, H 在嵌入空间中远离了 V")
print(f"     但从 α_transition 角度: α={alpha[-1]:.4f} > 0.5, H 在 V-L 轴上的投影仍更靠近 V 侧")
print(f"     从锥体角度: ∠(H,V)={angle_hv[-1]:.1f}° 远小于 ∠(H,T)={angle_ht[-1]:.1f}°")
print(f"     所以 H 虽然远离 V, 但相对位置仍更靠近 V 而非 T")
print()
print("  2. \"同时不远离T\" — ✅ 成立")
print(f"     cos(H,T) 从 {cos_ht[0]:.4f} 升至 {cos_ht[-1]:.4f} (↑{abs(cos_ht[-1]-cos_ht[0])/abs(cos_ht[0])*100:.0f}%)")
print(f"     H 与 T 的相似度增加, H 在靠近 T")
print(f"     ∠(H,T) 从 {angle_ht[0]:.1f}° 收窄到 {angle_ht[-1]:.1f}°, 确认 H 靠近 T")
print()
print("  3. 正确的综合表述:")
print("     RL训练将 H 从「偏向视觉的隐状态」塑造为「视觉-语言过渡模态」:")
print(f"     - H 从 V 侧向 V-L 中间移动 (α: {alpha[0]:.2f} → {alpha[-1]:.2f})")
print(f"     - H 与 T 的对齐增强 (cos_HT: {cos_ht[0]:.4f} → {cos_ht[-1]:.4f})")
print(f"     - H 与 V 的距离增大但相对位置仍偏 V (cos_HV↓, α>0.5)")
print(f"     - 表示能力提升 (diag: {diag[0]:.4f} → {diag[-1]:.4f}, capacity: {capacity[0]:.4f} → {capacity[-1]:.4f})")
print()
print("  4. 用户论断的修正建议:")
print("     将 \"H比训练前更接近V\" 修正为:")
print("     \"H在V-L空间中的投影位置仍偏向V侧, 但从V侧向V-L等距点移动\"")
print("     即 H 不是更接近 V, 而是 H 从偏向V的位置向中间移动,")
print("     同时保持与T的对齐增强")
