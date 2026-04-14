#!/usr/bin/env python3
"""分析 Stage 1 生成数据的质量"""
import json
import sys
import random
from collections import Counter

DATA_PATH = "/mnt/cephszjt/user_juntianzhang/LatentDraft/data/nld_phase1/vcr_latent_cot_v3_stage1_10k.json"

print(f"Loading data from {DATA_PATH} ...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)

print(f"\n{'='*60}")
print(f"  Stage 1 数据质量分析报告")
print(f"{'='*60}")
print(f"\n总样本数: {len(data)}")

# 1. 任务类型分布
task_dist = Counter(d['task_type'] for d in data)
print(f"\n--- 任务类型分布 ---")
for task, cnt in sorted(task_dist.items(), key=lambda x: -x[1]):
    bar = '█' * int(cnt / len(data) * 50)
    print(f"  {task:30s}: {cnt:5d} ({cnt/len(data)*100:5.1f}%) {bar}")

# 2. reasoning_full 长度统计
rf_lens = [len(d['reasoning_full']) for d in data]
print(f"\n--- reasoning_full 长度 (字符) ---")
print(f"  平均: {sum(rf_lens)/len(rf_lens):.0f}")
print(f"  最小: {min(rf_lens)}")
print(f"  最大: {max(rf_lens)}")
print(f"  中位数: {sorted(rf_lens)[len(rf_lens)//2]}")
# 长度分布
len_buckets = Counter()
for l in rf_lens:
    if l < 500: len_buckets['<500'] += 1
    elif l < 1000: len_buckets['500-1000'] += 1
    elif l < 1500: len_buckets['1000-1500'] += 1
    elif l < 2000: len_buckets['1500-2000'] += 1
    elif l < 3000: len_buckets['2000-3000'] += 1
    else: len_buckets['3000+'] += 1
print(f"  长度分布:")
for b in ['<500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '3000+']:
    cnt = len_buckets.get(b, 0)
    print(f"    {b:12s}: {cnt:5d} ({cnt/len(data)*100:5.1f}%)")

# 3. latent_text 长度统计
lt_lens = [len(d['latent_text']) for d in data]
print(f"\n--- latent_text 长度 (字符) ---")
print(f"  平均: {sum(lt_lens)/len(lt_lens):.0f}")
print(f"  最小: {min(lt_lens)}")
print(f"  最大: {max(lt_lens)}")
print(f"  中位数: {sorted(lt_lens)[len(lt_lens)//2]}")

# 4. latent_position 分布
positions = [d['latent_position'] for d in data]
print(f"\n--- latent_position 分布 (理想范围 0.3-0.7) ---")
print(f"  平均: {sum(positions)/len(positions):.3f}")
print(f"  最小: {min(positions):.3f}")
print(f"  最大: {max(positions):.3f}")
pos_buckets = Counter()
for p in positions:
    if p < 0.2: pos_buckets['0.0-0.2'] += 1
    elif p < 0.3: pos_buckets['0.2-0.3'] += 1
    elif p < 0.4: pos_buckets['0.3-0.4'] += 1
    elif p < 0.5: pos_buckets['0.4-0.5'] += 1
    elif p < 0.6: pos_buckets['0.5-0.6'] += 1
    elif p < 0.7: pos_buckets['0.6-0.7'] += 1
    elif p < 0.8: pos_buckets['0.7-0.8'] += 1
    else: pos_buckets['0.8-1.0'] += 1
for b in ['0.0-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-1.0']:
    cnt = pos_buckets.get(b, 0)
    bar = '█' * int(cnt / len(data) * 80)
    print(f"  {b}: {cnt:5d} ({cnt/len(data)*100:5.1f}%) {bar}")
# 理想范围内的比例
ideal = sum(1 for p in positions if 0.3 <= p <= 0.7)
print(f"  理想范围(0.3-0.7)内: {ideal} ({ideal/len(data)*100:.1f}%)")

# 5. 认知阶段数分布
stage_dist = Counter(d['num_stages'] for d in data)
print(f"\n--- 认知阶段数分布 ---")
for ns, cnt in sorted(stage_dist.items()):
    bar = '█' * int(cnt / len(data) * 50)
    print(f"  {ns} stages: {cnt:5d} ({cnt/len(data)*100:5.1f}%) {bar}")

# 6. 每样本 latent_key_tokens 数量
token_counts = [d['num_latent_tokens'] for d in data]
print(f"\n--- 每样本 latent_key_tokens 数量 ---")
print(f"  平均: {sum(token_counts)/len(token_counts):.1f}")
print(f"  最小: {min(token_counts)}")
print(f"  最大: {max(token_counts)}")

# 7. question 和 answer 长度
q_lens = [len(d['question']) for d in data]
a_lens = [len(d['answer']) for d in data]
print(f"\n--- question 长度 (字符) ---")
print(f"  平均: {sum(q_lens)/len(q_lens):.0f}, 最小: {min(q_lens)}, 最大: {max(q_lens)}")
print(f"\n--- answer 长度 (字符) ---")
print(f"  平均: {sum(a_lens)/len(a_lens):.0f}, 最小: {min(a_lens)}, 最大: {max(a_lens)}")

# 8. Top stage names
all_stage_names = []
for d in data:
    for stage in d.get('latent_key_tokens', []):
        if isinstance(stage, dict):
            all_stage_names.append(stage.get('stage', 'unknown'))
stage_name_freq = Counter(all_stage_names)
print(f"\n--- Top-20 认知阶段名称 ---")
for name, cnt in stage_name_freq.most_common(20):
    print(f"  {name:35s}: {cnt}")

# 9. Top key tokens
all_tokens = []
for d in data:
    for stage in d.get('latent_key_tokens', []):
        if isinstance(stage, dict):
            all_tokens.extend(stage.get('tokens', []))
token_freq = Counter(all_tokens)
print(f"\n--- Top-30 高频 key tokens ---")
print(f"  总 tokens: {len(all_tokens)}, 唯一 tokens: {len(token_freq)}")
for token, cnt in token_freq.most_common(30):
    print(f"  {token:35s}: {cnt}")

# 10. 图片去重检查
images = [d['image'] for d in data]
unique_images = set(images)
print(f"\n--- 图片去重检查 ---")
print(f"  总样本: {len(data)}")
print(f"  唯一图片: {len(unique_images)}")
if len(images) != len(unique_images):
    dup_count = len(images) - len(unique_images)
    print(f"  ⚠️ 重复图片: {dup_count}")

# 11. 抽样展示 3 个样本
print(f"\n{'='*60}")
print(f"  随机抽样 3 个样本")
print(f"{'='*60}")
random.seed(42)
samples = random.sample(data, min(3, len(data)))
for i, s in enumerate(samples):
    print(f"\n--- 样本 {i+1} ---")
    print(f"  图片: {s['image']}")
    print(f"  任务类型: {s['task_type']}")
    print(f"  问题: {s['question'][:150]}...")
    print(f"  答案: {s['answer'][:150]}...")
    print(f"  reasoning_full 长度: {len(s['reasoning_full'])} 字符")
    print(f"  latent_text 长度: {len(s['latent_text'])} 字符")
    print(f"  latent_position: {s['latent_position']}")
    print(f"  阶段数: {s['num_stages']}")
    print(f"  latent_key_tokens:")
    for stage in s['latent_key_tokens']:
        if isinstance(stage, dict):
            print(f"    [{stage['stage']}]: {stage['tokens']}")

# 12. 质量问题检测
print(f"\n{'='*60}")
print(f"  潜在质量问题检测")
print(f"{'='*60}")
issues = Counter()
for d in data:
    # reasoning 太短
    if len(d['reasoning_full']) < 300:
        issues['reasoning_full < 300 chars'] += 1
    # latent_text 太短
    if len(d['latent_text']) < 50:
        issues['latent_text < 50 chars'] += 1
    # latent_text 太长 (可能标记了太多内容)
    if len(d['latent_text']) > 2000:
        issues['latent_text > 2000 chars'] += 1
    # position 太靠前
    if d['latent_position'] < 0.15:
        issues['latent_position < 0.15'] += 1
    # position 太靠后
    if d['latent_position'] > 0.85:
        issues['latent_position > 0.85'] += 1
    # answer 太长
    if len(d['answer']) > 400:
        issues['answer > 400 chars'] += 1
    # stages 太少
    if d['num_stages'] < 2:
        issues['num_stages < 2'] += 1
    # tokens 太少
    if d['num_latent_tokens'] < 3:
        issues['num_latent_tokens < 3'] += 1
    # 检查 reasoning_for_training 是否包含 <|pause|>
    if '<|pause|>' not in d.get('reasoning_for_training', ''):
        issues['missing <|pause|> in training'] += 1

if issues:
    for issue, cnt in sorted(issues.items(), key=lambda x: -x[1]):
        severity = '⚠️' if cnt / len(data) > 0.05 else 'ℹ️'
        print(f"  {severity} {issue}: {cnt} ({cnt/len(data)*100:.1f}%)")
else:
    print(f"  ✅ 未发现明显质量问题")

print(f"\n{'='*60}")
print(f"  分析完成!")
print(f"{'='*60}")
