#!/usr/bin/env python3
"""
从 266K 样本中筛选 50K 高质量训练数据

筛选策略:
  1. 步骤数多样性 — 高步骤数样本对 draft controller 学习更有价值，上采样
  2. 来源均衡   — 4 个来源按比例采样，避免单一来源过拟合
  3. think_chain 质量 — 过滤过短/过长的，保留中间段
  4. 题型平衡   — 选择题/开放题均衡

目标: 50,000 样本
"""

import json
import random
import argparse
import os
from collections import defaultdict, Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="从 266K 中筛选 50K 高质量样本")
    p.add_argument("--input", type=str,
                   default="/mnt/cephszjt/user_juntianzhang/LatentDraft/data/rld_mmathcot_filtered_checked.json",
                   help="输入 JSON 路径")
    p.add_argument("--output", type=str,
                   default="/mnt/cephszjt/user_juntianzhang/LatentDraft/data/rld_50k_selected.json",
                   help="输出 JSON 路径")
    p.add_argument("--target", type=int, default=50000, help="目标样本数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def get_source_dir(image_path: str) -> str:
    """从图片路径提取数据来源目录"""
    parts = image_path.split('/')
    if 'MMathCoT-1M' in parts:
        idx = parts.index('MMathCoT-1M')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return 'unknown'


def is_choice_question(question: str) -> bool:
    """判断是否为选择题"""
    return 'Choices' in question or 'choices' in question


def compute_quality_score(sample: dict) -> float:
    """
    计算样本质量分数 (0~1)，用于排序筛选

    评分维度:
      - think_chain 长度适中性 (最优区间 300-800 字符)
      - 步骤数 (越多越好，对 controller 训练价值越大)
      - question 长度合理性 (太短可能信息不足)
    """
    tc = sample.get('think_chain', '')
    tc_len = len(tc)
    steps = tc.count('</step>')
    q_len = len(sample.get('question', ''))

    score = 0.0

    # ---- think_chain 长度分 (0~0.3) ----
    # 最优区间: 300-800 字符
    if 300 <= tc_len <= 800:
        score += 0.3
    elif 200 <= tc_len < 300:
        score += 0.15
    elif 800 < tc_len <= 1200:
        score += 0.2
    elif tc_len > 1200:
        score += 0.1
    else:
        score += 0.05  # < 200

    # ---- 步骤数分 (0~0.4) ----
    # 高步骤数对 draft controller 学习更有价值
    # 4步=0.15, 5步=0.25, 6步=0.30, 7步=0.35, 8+=0.40
    step_scores = {4: 0.15, 5: 0.25, 6: 0.30, 7: 0.35}
    score += step_scores.get(steps, 0.40 if steps >= 8 else 0.10)

    # ---- question 长度分 (0~0.15) ----
    if 50 <= q_len <= 500:
        score += 0.15
    elif 30 <= q_len < 50:
        score += 0.10
    elif q_len > 500:
        score += 0.08
    else:
        score += 0.05

    # ---- think_chain 步骤连贯性 (0~0.15) ----
    # 检查是否有 <think> 开头和合理的 step 结构
    if tc.startswith('<think>') and steps >= 4:
        score += 0.15
    elif steps >= 4:
        score += 0.10
    else:
        score += 0.05

    return score


def select_samples(data: list, target: int, seed: int) -> list:
    """
    分层采样策略:

    Step 1: 硬过滤 — 去掉低质量样本
    Step 2: 按 (来源 × 步骤数 × 题型) 分桶
    Step 3: 在每个桶内按质量分数排序，优先选高分样本
    Step 4: 配额分配，高步骤数桶获得更多配额
    """
    random.seed(seed)

    # ========== Step 1: 硬过滤 ==========
    filtered = []
    reject_reasons = Counter()

    for sample in data:
        tc = sample.get('think_chain', '')
        tc_len = len(tc)
        steps = tc.count('</step>')
        q_len = len(sample.get('question', ''))

        # 过滤条件
        if tc_len < 150:
            reject_reasons['tc_too_short'] += 1
            continue
        if tc_len > 2000:
            reject_reasons['tc_too_long'] += 1
            continue
        if steps < 4:
            reject_reasons['too_few_steps'] += 1
            continue
        if q_len < 15:
            reject_reasons['q_too_short'] += 1
            continue
        if not sample.get('image'):
            reject_reasons['no_image'] += 1
            continue
        # 跳过文件存在性检查 (数据已经预过滤过, 检查 266K 文件太慢)

        filtered.append(sample)

    print(f"\n[Step 1] 硬过滤: {len(data)} → {len(filtered)} (去掉 {len(data) - len(filtered)})")
    for reason, cnt in reject_reasons.most_common():
        print(f"  去掉 {reason}: {cnt}")

    # ========== Step 2: 分桶 ==========
    # 维度: (来源, 步骤数分组, 是否选择题)
    def get_step_group(steps):
        if steps <= 4:
            return '4'
        elif steps == 5:
            return '5'
        elif steps == 6:
            return '6'
        elif steps == 7:
            return '7'
        else:
            return '8+'

    buckets = defaultdict(list)
    for sample in filtered:
        src = get_source_dir(sample['image'])
        steps = sample['think_chain'].count('</step>')
        step_grp = get_step_group(steps)
        q_type = 'choice' if is_choice_question(sample['question']) else 'open'
        key = (src, step_grp, q_type)
        buckets[key].append(sample)

    print(f"\n[Step 2] 分桶: {len(buckets)} 个桶")

    # 打印桶分布
    print(f"\n  {'来源':<25} {'步骤':>5} {'题型':>8} {'数量':>8}")
    print(f"  {'─' * 50}")
    for key in sorted(buckets.keys()):
        src, step_grp, q_type = key
        print(f"  {src:<25} {step_grp:>5} {q_type:>8} {len(buckets[key]):>8}")

    # ========== Step 3: 质量排序 ==========
    for key in buckets:
        # 计算质量分数并按分数降序排序
        for sample in buckets[key]:
            sample['_quality_score'] = compute_quality_score(sample)
        buckets[key].sort(key=lambda x: -x['_quality_score'])

    # ========== Step 4: 配额分配 ==========
    # 策略: 
    #   - 来源均衡: 4个来源各 ~25%
    #   - 步骤数上采样: 高步骤数获得更多配额
    #   - 题型: 选择题/开放题 ≈ 55:45 (略偏选择题，因为原始数据选择题多)
    
    # 步骤数权重 (上采样高步骤数)
    step_weights = {
        '4': 1.0,   # 原始 43.7% → 目标 ~20%
        '5': 1.5,   # 原始 31.9% → 目标 ~25%
        '6': 3.0,   # 原始 15.0% → 目标 ~25%
        '7': 5.0,   # 原始 6.1%  → 目标 ~18%
        '8+': 8.0,  # 原始 3.3%  → 目标 ~12%
    }

    # 计算每个桶的原始权重
    bucket_weights = {}
    for key, samples in buckets.items():
        src, step_grp, q_type = key
        w = step_weights.get(step_grp, 1.0) * len(samples)
        bucket_weights[key] = w

    total_weight = sum(bucket_weights.values())

    # 按权重分配配额
    bucket_quotas = {}
    allocated = 0
    for key in sorted(bucket_weights.keys()):
        quota = int(target * bucket_weights[key] / total_weight)
        # 不超过桶的实际大小
        quota = min(quota, len(buckets[key]))
        bucket_quotas[key] = quota
        allocated += quota

    # 剩余配额分配给高步骤数的桶
    remaining = target - allocated
    if remaining > 0:
        # 按步骤数优先级分配剩余
        priority_keys = sorted(
            [k for k in buckets if len(buckets[k]) > bucket_quotas.get(k, 0)],
            key=lambda k: -step_weights.get(k[1], 1.0)
        )
        for key in priority_keys:
            if remaining <= 0:
                break
            can_add = len(buckets[key]) - bucket_quotas[key]
            add = min(can_add, remaining)
            bucket_quotas[key] += add
            remaining -= add

    print(f"\n[Step 4] 配额分配 (目标={target}):")
    print(f"\n  {'来源':<25} {'步骤':>5} {'题型':>8} {'桶大小':>8} {'配额':>8} {'采样率':>8}")
    print(f"  {'─' * 65}")
    total_quota = 0
    for key in sorted(bucket_quotas.keys()):
        src, step_grp, q_type = key
        quota = bucket_quotas[key]
        bsize = len(buckets[key])
        rate = quota / bsize * 100 if bsize > 0 else 0
        total_quota += quota
        print(f"  {src:<25} {step_grp:>5} {q_type:>8} {bsize:>8} {quota:>8} {rate:>7.1f}%")
    print(f"  {'─' * 65}")
    print(f"  {'总计':<25} {'':>5} {'':>8} {len(filtered):>8} {total_quota:>8}")

    # ========== Step 5: 采样 ==========
    selected = []
    for key, quota in bucket_quotas.items():
        bucket_samples = buckets[key][:quota]  # 已按质量分数排序，取前 quota 个
        selected.extend(bucket_samples)

    # 打乱顺序
    random.shuffle(selected)

    # 移除临时字段
    for s in selected:
        s.pop('_quality_score', None)

    return selected


def print_final_stats(selected: list):
    """打印最终筛选结果的统计"""
    print(f"\n{'=' * 70}")
    print(f"📊 最终筛选结果统计")
    print(f"{'=' * 70}")
    print(f"  总样本数: {len(selected)}")

    # 步骤数分布
    step_counts = [s['think_chain'].count('</step>') for s in selected]
    sc = Counter(step_counts)
    print(f"\n  步骤数分布:")
    for k in sorted(sc.keys()):
        print(f"    steps={k}: {sc[k]:>6} ({sc[k] / len(selected) * 100:>5.1f}%)")

    # 来源分布
    sources = Counter(get_source_dir(s['image']) for s in selected)
    print(f"\n  来源分布:")
    for k, v in sources.most_common():
        print(f"    {k}: {v:>6} ({v / len(selected) * 100:>5.1f}%)")

    # 题型分布
    n_choice = sum(1 for s in selected if is_choice_question(s['question']))
    n_open = len(selected) - n_choice
    print(f"\n  题型分布:")
    print(f"    选择题: {n_choice:>6} ({n_choice / len(selected) * 100:.1f}%)")
    print(f"    开放题: {n_open:>6} ({n_open / len(selected) * 100:.1f}%)")

    # think_chain 长度分布
    tc_lens = [len(s['think_chain']) for s in selected]
    print(f"\n  think_chain 长度:")
    print(f"    min={min(tc_lens)}, max={max(tc_lens)}, "
          f"mean={sum(tc_lens) / len(tc_lens):.0f}, "
          f"median={sorted(tc_lens)[len(tc_lens) // 2]}")

    # 估算训练步数
    effective_bs = 32  # 1 * 4 * 8
    total_steps = len(selected) // effective_bs
    print(f"\n  预估训练步数 (bs=32): {total_steps} steps/epoch")
    print(f"  预估训练时间 (3s/step): {total_steps * 3 / 3600:.1f}h")
    print(f"{'=' * 70}")


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"📂 加载数据: {args.input}")
    print(f"  (文件较大，加载中...)", flush=True)
    import time
    t0 = time.time()
    with open(args.input) as f:
        data = json.load(f)
    print(f"  加载完成，耗时 {time.time() - t0:.1f}s")
    print(f"  原始样本数: {len(data)}", flush=True)

    # 筛选
    selected = select_samples(data, target=args.target, seed=args.seed)

    # 统计
    print_final_stats(selected)

    # 保存
    print(f"\n💾 保存到: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"  文件大小: {file_size_mb:.1f}MB")
    print(f"\n✅ 完成！")

    # 打印使用方法
    print(f"\n📝 使用方法:")
    print(f"  修改 configs/rld_train.yaml 中的 train_json 为:")
    print(f"    train_json: \"{args.output}\"")


if __name__ == '__main__':
    main()
