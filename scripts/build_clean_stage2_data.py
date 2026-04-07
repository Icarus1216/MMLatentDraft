#!/usr/bin/env python3
"""
构建 Stage 2 训练集 — 简化版

直接从 rld_mmathcot_filtered_checked.json (266K) 中采样 80K，
将旧格式 (<think>...</step>...</think>) 转换为新格式 (Step N: ...)，
输出为 Stage 2 训练集。

格式转换:
  旧: "<think>\nstep1\n</step>\nstep2\n</step>\n</think>\nAnswer: D"
  新: "Step 1: step1\nStep 2: step2\nFinal Answer: D"

用法:
  python scripts/build_clean_stage2_data.py \
    --input data/rld_mmathcot_filtered_checked.json \
    --output data/rld_stage2_clean.json \
    --size 80000 --seed 42
"""

import argparse
import json
import os
import re
import random
from collections import Counter


def convert_old_to_new(think_chain: str, answer: str) -> str:
    """
    旧格式 → 新格式
    
    输入: "<think>\nstep1\n</step>\nstep2\n</step>\n</think>\nAnswer: D"
    输出: "Step 1: step1\nStep 2: step2\nFinal Answer: D"
    """
    text = think_chain

    # 1) 剥离 <think>...</think> 及其后面的 Answer
    end = text.rfind('</think>')
    if end != -1:
        text = text[:end]
    if text.startswith('<think>'):
        text = text[len('<think>'):]
    text = text.strip('\n').strip()

    # 2) 按 </step> 分割
    parts = text.split('</step>')
    steps = [p.strip() for p in parts if p.strip()]

    if not steps:
        return f"Step 1: Analyzing the problem.\nFinal Answer: {answer}"

    # 3) 重新编号
    lines = []
    for i, s in enumerate(steps, 1):
        if s.startswith('<step>'):
            s = s[len('<step>'):].strip()
        lines.append(f"Step {i}: {s}")

    lines.append(f"Final Answer: {answer}")
    return "\n".join(lines)


def count_steps_old(tc: str) -> int:
    """统计旧格式步骤数"""
    return tc.count('</step>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/rld_mmathcot_filtered_checked.json")
    parser.add_argument("--output", default="data/rld_stage2_clean.json")
    parser.add_argument("--size", type=int, default=80000)
    parser.add_argument("--min_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print(f"📦 从原始数据采样 {args.size} 条 → Stage 2 训练集")
    print("=" * 60)

    # 加载
    print(f"\n[1/3] 加载 {args.input} ...")
    with open(args.input) as f:
        raw = json.load(f)
    print(f"  总样本: {len(raw)}")

    # 筛选: 有 image、有 think_chain、步骤数在范围内
    valid = []
    skip = Counter()
    for item in raw:
        if not item.get('image'):
            skip['no_image'] += 1
            continue
        tc = item.get('think_chain', '')
        if not tc or '<think>' not in tc:
            skip['no_think_chain'] += 1
            continue
        n = count_steps_old(tc)
        if n < args.min_steps:
            skip['too_few_steps'] += 1
            continue
        if n > args.max_steps:
            skip['too_many_steps'] += 1
            continue
        valid.append(item)
    del raw

    print(f"  有效样本: {len(valid)}")
    print(f"  跳过: {dict(skip)}")

    # 采样
    print(f"\n[2/3] 随机采样 {args.size} 条...")
    if len(valid) < args.size:
        print(f"  ⚠️ 有效样本不足，使用全部 {len(valid)} 条")
        selected = valid
    else:
        selected = random.sample(valid, args.size)
    del valid

    # 转换格式
    print(f"\n[3/3] 格式转换 (旧→新) 并保存...")
    output = []
    for item in selected:
        new_tc = convert_old_to_new(item['think_chain'], item['answer'])
        output.append({
            'image': item['image'],
            'question': item['question'],
            'answer': item['answer'],
            'think_chain': new_tc,
            'think_chain_status': 'valid',
            'think_chain_source': 'dataset_converted',
        })

    random.shuffle(output)

    # 统计
    step_counts = []
    for d in output:
        n = len(re.findall(r'(?:^|\n)Step \d+:', d['think_chain']))
        step_counts.append(n)
    dist = Counter(step_counts)

    print(f"\n{'=' * 60}")
    print(f"📊 最终统计: {len(output)} 条样本")
    print(f"  步骤数分布:")
    for k in sorted(dist):
        print(f"    {k} 步: {dist[k]} ({dist[k]/len(output)*100:.1f}%)")

    # 抽样验证
    print(f"\n  抽样验证 (3 条):")
    for i in range(min(3, len(output))):
        tc = output[i]['think_chain']
        print(f"  --- 样本 {i} ---")
        print(f"  {tc[:200]}...")
        print()

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, ensure_ascii=False)

    sz = os.path.getsize(args.output) / (1024 * 1024)
    print(f"💾 已保存: {args.output} ({sz:.1f} MB)")


if __name__ == '__main__':
    main()
