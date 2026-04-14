#!/usr/bin/env python3
"""Batch 2 10K 数据质量检查脚本"""
import json
import sys
from collections import Counter

print("=" * 60, flush=True)
print("  Batch 2 10K 数据质量报告", flush=True)
print("=" * 60, flush=True)

# 加载数据
print("\n加载 Batch 2...", flush=True)
b2 = json.load(open('data/nld_phase1/vcr_latent_cot_v3_stage1_batch2_10k.json'))
print(f"  Batch 2 样本数: {len(b2)}", flush=True)

print("\n加载 Batch 1...", flush=True)
b1 = json.load(open('data/nld_phase1/vcr_latent_cot_v3_stage1_10k.json'))
print(f"  Batch 1 样本数: {len(b1)}", flush=True)

# === 图片重叠检查 ===
print("\n🔍 图片重叠检查", flush=True)
b1_images = set(d['image'] for d in b1)
b2_images = set(d['image'] for d in b2)
overlap = b1_images & b2_images
print(f"  Batch 1 唯一图片: {len(b1_images)}", flush=True)
print(f"  Batch 2 唯一图片: {len(b2_images)}", flush=True)
print(f"  重叠图片: {len(overlap)} {'✅' if len(overlap)==0 else '❌'}", flush=True)
print(f"  合计唯一图片: {len(b1_images | b2_images)}", flush=True)

# === 任务类型分布 ===
print("\n📋 任务类型分布对比", flush=True)
b1_tasks = Counter(d['task_type'] for d in b1)
b2_tasks = Counter(d['task_type'] for d in b2)
all_tasks = sorted(set(list(b1_tasks.keys()) + list(b2_tasks.keys())))
header = f"  {'任务类型':<25} {'Batch1':>7} {'B1%':>6} {'Batch2':>7} {'B2%':>6} {'合计':>7} {'合计%':>6}"
print(header, flush=True)
print("  " + "-" * 68, flush=True)
for t in all_tasks:
    b1c = b1_tasks.get(t, 0)
    b2c = b2_tasks.get(t, 0)
    total = b1c + b2c
    b1p = b1c / len(b1) * 100
    b2p = b2c / len(b2) * 100
    tp = total / (len(b1) + len(b2)) * 100
    print(f"  {t:<25} {b1c:>7} {b1p:>5.1f}% {b2c:>7} {b2p:>5.1f}% {total:>7} {tp:>5.1f}%", flush=True)

b2_vals = list(b2_tasks.values())
combined_vals = [b1_tasks.get(t, 0) + b2_tasks.get(t, 0) for t in all_tasks]
print(f"\n  Batch 2 均衡度 (min/max): {min(b2_vals)/max(b2_vals):.3f}", flush=True)
print(f"  合并后均衡度 (min/max): {min(combined_vals)/max(combined_vals):.3f}", flush=True)

# === 字段完整性 ===
print("\n✅ 字段完整性检查 (Batch 2)", flush=True)
fields = ['image', 'question', 'answer', 'task_type', 'latent_text', 'reasoning_for_training', 'latent_key_tokens']
all_ok = True
for f in fields:
    missing = sum(1 for d in b2 if f not in d or not d[f])
    status = "✅" if missing == 0 else "❌"
    print(f"  {f}: 缺失 {missing} {status}", flush=True)
    if missing > 0:
        all_ok = False

# === Latent key tokens 统计 ===
print("\n🔑 Latent Key Tokens 统计 (Batch 2)", flush=True)
all_tokens = []
stage_counts = []
for d in b2:
    stages = d.get('latent_key_tokens', [])
    stage_counts.append(len(stages))
    for s in stages:
        if isinstance(s, dict):
            all_tokens.extend(s.get('tokens', []))

token_freq = Counter(all_tokens)
print(f"  总 tokens: {len(all_tokens)}", flush=True)
print(f"  唯一 tokens: {len(token_freq)}", flush=True)
print(f"  唯一率: {len(token_freq)/max(len(all_tokens),1)*100:.1f}%", flush=True)
print(f"  平均 stages/样本: {sum(stage_counts)/len(stage_counts):.1f}", flush=True)
print(f"  stages 范围: {min(stage_counts)} - {max(stage_counts)}", flush=True)

# Stage token 长度分布
token_lens = Counter()
for d in b2:
    for s in d.get('latent_key_tokens', []):
        if isinstance(s, dict):
            token_lens[len(s.get('tokens', []))] += 1
print(f"\n  Token 长度分布:", flush=True)
for l in sorted(token_lens.keys()):
    print(f"    {l} tokens: {token_lens[l]} ({token_lens[l]/sum(token_lens.values())*100:.1f}%)", flush=True)

# === Latent 内容主题分析 ===
print("\n🧠 Latent 内容主题对比", flush=True)
def count_themes(data):
    themes = Counter()
    for d in data:
        lt = d.get('latent_text', '').lower()
        if 'rotat' in lt: themes['rotation'] += 1
        if 'light' in lt or 'shadow' in lt: themes['light_shadow'] += 1
        if 'gravity' in lt or 'balance' in lt or 'torque' in lt: themes['physics'] += 1
        if 'occlu' in lt: themes['occlusion'] += 1
        if 'trajectory' in lt or 'path' in lt: themes['trajectory'] += 1
        if 'perspect' in lt or 'viewpoint' in lt: themes['perspective'] += 1
        if 'depth' in lt or '3d' in lt: themes['3d_depth'] += 1
        if 'emotion' in lt or 'feel' in lt or 'express' in lt: themes['emotion'] += 1
        if 'social' in lt or 'interact' in lt or 'gesture' in lt: themes['social'] += 1
        if 'texture' in lt or 'material' in lt or 'surface' in lt: themes['material'] += 1
        if 'causal' in lt or 'cause' in lt or 'effect' in lt: themes['causal'] += 1
    return themes

b1_themes = count_themes(b1)
b2_themes = count_themes(b2)
all_themes = sorted(set(list(b1_themes.keys()) + list(b2_themes.keys())))
print(f"  {'主题':<15} {'Batch1':>8} {'Batch2':>8} {'变化':>8}", flush=True)
print(f"  " + "-" * 42, flush=True)
for t in all_themes:
    b1c = b1_themes.get(t, 0)
    b2c = b2_themes.get(t, 0)
    change = '+' if b2c > b1c else ('-' if b2c < b1c else '=')
    print(f"  {t:<15} {b1c:>8} {b2c:>8} {change:>8}", flush=True)

# === reasoning_for_training 中 <|pause|> 检查 ===
print("\n⏸️  <|pause|> Token 检查 (Batch 2)", flush=True)
has_pause = sum(1 for d in b2 if '<|pause|>' in d.get('reasoning_for_training', ''))
print(f"  含 <|pause|> 的样本: {has_pause}/{len(b2)} ({has_pause/len(b2)*100:.1f}%)", flush=True)

# 平均 pause 数量
pause_counts = []
for d in b2:
    pc = d.get('reasoning_for_training', '').count('<|pause|>')
    pause_counts.append(pc)
print(f"  平均 <|pause|> 数/样本: {sum(pause_counts)/len(pause_counts):.1f}", flush=True)
print(f"  <|pause|> 范围: {min(pause_counts)} - {max(pause_counts)}", flush=True)

# === 样本长度统计 ===
print("\n📏 文本长度统计 (Batch 2)", flush=True)
latent_lens = [len(d.get('latent_text', '')) for d in b2]
reasoning_lens = [len(d.get('reasoning_for_training', '')) for d in b2]
print(f"  latent_text 平均长度: {sum(latent_lens)/len(latent_lens):.0f} 字符", flush=True)
print(f"  latent_text 范围: {min(latent_lens)} - {max(latent_lens)}", flush=True)
print(f"  reasoning_for_training 平均长度: {sum(reasoning_lens)/len(reasoning_lens):.0f} 字符", flush=True)

# === 总结 ===
print("\n" + "=" * 60, flush=True)
print("  📊 总结", flush=True)
print("=" * 60, flush=True)
print(f"  Batch 1: {len(b1)} 条", flush=True)
print(f"  Batch 2: {len(b2)} 条", flush=True)
print(f"  合计: {len(b1)+len(b2)} 条", flush=True)
print(f"  图片无重叠: {'✅' if len(overlap)==0 else '❌'}", flush=True)
print(f"  字段完整: {'✅' if all_ok else '❌'}", flush=True)
print(f"  <|pause|> 覆盖率: {has_pause/len(b2)*100:.1f}%", flush=True)
print(f"  合并后任务均衡度: {min(combined_vals)/max(combined_vals):.3f}", flush=True)
print("=" * 60, flush=True)
