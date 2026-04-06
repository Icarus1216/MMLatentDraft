#!/usr/bin/env python3
"""
预处理脚本: 静态混合 wrong_cot + correct_cot 生成最终训练集

功能:
  1. 加载拒绝采样后的 wrong_cot 数据 (新格式: Step N:)
  2. 加载原始数据集的正确 CoT (旧格式: <think>...</step>...</think>)
  3. 从原始数据中选择拒绝采样 **未覆盖** 的样本 (不同的 image+question)
  4. 将旧格式 correct_cot 转换为新格式 (Step N:)
  5. 按指定比例混合, 输出最终训练集 JSON

使用方式:
  python scripts/prepare_training_data.py \
    --wrong_cot_json /path/to/rld_50k_rejection_sampled_new_format.json \
    --correct_cot_json /path/to/rld_50k_selected.json \
    --output_json /path/to/rld_training_mixed.json \
    --correct_cot_ratio 0.3 \
    --max_steps 14 \
    --hard_sample_max_ratio 0.22 \
    --seed 42

输出格式 (每个样本):
  {
    "image": "/path/to/image.jpg",
    "question": "...",
    "answer": "D",
    "think_chain": "Step 1: ...\nStep 2: ...\n",
    "think_chain_status": "valid",
    "think_chain_source": "wrong_cot" | "dataset_converted",
    "rejection_sampling_meta": {...}  // 仅 wrong_cot 样本有
  }
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter


# ====== 格式转换工具函数 ======

def count_steps(think_chain: str) -> int:
    """统计推理链中的步骤数 (兼容新旧格式)"""
    if not think_chain:
        return 0
    # 新格式: 统计 "Step N:" 的数量
    step_n_count = len(re.findall(r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]', think_chain, re.IGNORECASE))
    if step_n_count > 0:
        return step_n_count
    # 旧格式: 统计 </step> 的数量
    return think_chain.count('</step>')


def convert_old_format_to_new(think_chain: str) -> str:
    """
    将旧格式 (<think>...</step>...</think>) 转换为新格式 (Step N: ...)
    
    旧格式: "<think>\nstep1_content\n</step>\nstep2_content\n</step>\n</think>\nAnswer: xxx"
    新格式: "Step 1: step1_content\nStep 2: step2_content\n"
    
    返回值只包含推理步骤部分, 不包含 "Final Answer:" 行。
    """
    text = think_chain
    
    # 1) 剥离 <think>...</think> 包裹及其后面的内容 (如 "\nAnswer: 12")
    think_end_pos = text.rfind('</think>')
    if think_end_pos != -1:
        text = text[:think_end_pos]
    if text.startswith('<think>'):
        text = text[len('<think>'):]
    text = text.strip('\n')
    
    # 2) 按 </step> 分割为步骤
    parts = text.split('</step>')
    steps = [p.strip() for p in parts if p.strip()]
    
    if not steps:
        return "Step 1: Analyzing the image.\n"
    
    # 3) 重新编号为 "Step N: ..." 格式
    cot_parts = []
    for i, step_text in enumerate(steps, 1):
        cot_parts.append(f"Step {i}: {step_text}")
    return "\n".join(cot_parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="静态混合 wrong_cot + correct_cot 生成训练集")
    parser.add_argument("--wrong_cot_json", type=str, required=True,
                        help="拒绝采样后的 wrong_cot 数据 (新格式)")
    parser.add_argument("--correct_cot_json", type=str, required=True,
                        help="原始数据集的正确 CoT 数据 (旧格式)")
    parser.add_argument("--output_json", type=str, required=True,
                        help="输出的混合训练集 JSON 路径")
    parser.add_argument("--correct_cot_ratio", type=float, default=0.3,
                        help="correct_cot 样本占最终数据的目标比例 (默认 0.3)")
    parser.add_argument("--max_steps", type=int, default=14,
                        help="最大步骤数过滤阈值 (默认 14)")
    parser.add_argument("--hard_sample_max_ratio", type=float, default=0.22,
                        help="pass_rate=0 困难样本的最大比例 (默认 0.22, 设为 1.0 则不下采样)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"=" * 60)
    print(f"📦 训练数据预处理: 静态混合 wrong_cot + correct_cot")
    print(f"=" * 60)

    # ====== 1. 加载 wrong_cot 数据 ======
    print(f"\n[1/5] 加载 wrong_cot 数据: {args.wrong_cot_json}")
    with open(args.wrong_cot_json, 'r') as f:
        wrong_cot_data = json.load(f)
    print(f"  原始样本数: {len(wrong_cot_data)}")

    # 基本字段检查: 移除无 image 字段的样本
    wrong_cot_data = [item for item in wrong_cot_data if item.get('image', '')]
    print(f"  有效样本数 (有 image): {len(wrong_cot_data)}")

    # ====== 2. 过滤超高段数样本 ======
    print(f"\n[2/5] 过滤超高段数样本 (>{args.max_steps} 步)...")
    before = len(wrong_cot_data)
    wrong_cot_data = [
        item for item in wrong_cot_data
        if count_steps(item.get('think_chain', '')) <= args.max_steps
    ]
    filtered = before - len(wrong_cot_data)
    if filtered > 0:
        print(f"  ⚠️ 已过滤 {filtered} 个超高段数样本")
    print(f"  保留: {len(wrong_cot_data)}")

    # ====== 3. P0: 下采样 pass_rate=0 的困难样本 ======
    if args.hard_sample_max_ratio < 1.0:
        print(f"\n[3/5] 下采样 pass_rate=0 困难样本 (目标比例 ≤{args.hard_sample_max_ratio:.0%})...")
        hard_samples = [item for item in wrong_cot_data
                        if item.get('rejection_sampling_meta', {}).get('pass_rate', -1) == 0]
        non_hard_samples = [item for item in wrong_cot_data
                            if item.get('rejection_sampling_meta', {}).get('pass_rate', -1) != 0]
        
        if hard_samples:
            max_hard = int(len(non_hard_samples) * args.hard_sample_max_ratio / max(1.0 - args.hard_sample_max_ratio, 0.01))
            if len(hard_samples) > max_hard:
                hard_samples = random.sample(hard_samples, max_hard)
                print(f"  🎯 pass_rate=0 下采样: {len(hard_samples) + (len(wrong_cot_data) - len(non_hard_samples) - len(hard_samples))} → {max_hard}")
            else:
                print(f"  ✅ pass_rate=0 样本 ({len(hard_samples)}) 已在目标比例内")
            wrong_cot_data = non_hard_samples + hard_samples
            random.shuffle(wrong_cot_data)
        else:
            print(f"  ✅ 无 pass_rate=0 样本")
        print(f"  下采样后: {len(wrong_cot_data)}")
    else:
        print(f"\n[3/5] 跳过困难样本下采样 (hard_sample_max_ratio=1.0)")

    # ====== 4. 加载 correct_cot 数据并选择未覆盖样本 ======
    print(f"\n[4/5] 加载 correct_cot 数据并选择拒绝采样未覆盖的样本...")
    print(f"  correct_cot 数据: {args.correct_cot_json}")
    
    # 建立 wrong_cot 的 (image, question) 集合
    existing_keys = set()
    for item in wrong_cot_data:
        existing_keys.add((item.get('image', ''), item.get('question', '')))
    print(f"  wrong_cot 唯一 (image, question) 对: {len(existing_keys)}")

    with open(args.correct_cot_json, 'r') as f:
        cc_raw_data = json.load(f)
    print(f"  原始 correct_cot 样本数: {len(cc_raw_data)}")

    # 选择拒绝采样未覆盖的样本, 并转换格式
    correct_cot_candidates = []
    skipped_invalid = 0
    skipped_overlap = 0
    skipped_too_many_steps = 0
    
    for cc_item in cc_raw_data:
        tc = cc_item.get('think_chain', '')
        tc_status = cc_item.get('think_chain_status', '')
        
        # 有效性检查
        if not (tc and tc_status == 'valid' and '<think>' in tc):
            skipped_invalid += 1
            continue
        
        key = (cc_item.get('image', ''), cc_item.get('question', ''))
        
        # 排除已有的 wrong_cot 样本
        if key in existing_keys:
            skipped_overlap += 1
            continue
        
        # 超高段数过滤
        if count_steps(tc) > args.max_steps:
            skipped_too_many_steps += 1
            continue
        
        # 转换旧格式为新格式
        new_format_tc = convert_old_format_to_new(tc)
        
        cc_sample = {
            'image': cc_item['image'],
            'question': cc_item['question'],
            'answer': cc_item['answer'],
            'think_chain': new_format_tc,
            'think_chain_status': 'valid',
            'think_chain_source': 'dataset_converted',
        }
        correct_cot_candidates.append(cc_sample)
    
    del cc_raw_data  # 释放内存
    
    print(f"  跳过 (无效/无 think_chain): {skipped_invalid}")
    print(f"  跳过 (与 wrong_cot 重叠): {skipped_overlap}")
    print(f"  跳过 (超高段数 >{args.max_steps}): {skipped_too_many_steps}")
    print(f"  ✅ 未覆盖的 correct_cot 候选: {len(correct_cot_candidates)}")

    # ====== 5. 按比例混合 ======
    print(f"\n[5/5] 按比例混合 (correct_cot 目标比例: {args.correct_cot_ratio:.0%})...")
    
    num_wrong = len(wrong_cot_data)
    # num_cc / (num_wrong + num_cc) = ratio → num_cc = num_wrong * ratio / (1 - ratio)
    target_cc = int(num_wrong * args.correct_cot_ratio / max(1.0 - args.correct_cot_ratio, 0.01))
    target_cc = min(target_cc, len(correct_cot_candidates))
    
    if target_cc > 0:
        if target_cc < len(correct_cot_candidates):
            selected_cc = random.sample(correct_cot_candidates, target_cc)
        else:
            selected_cc = correct_cot_candidates
        
        mixed_data = wrong_cot_data + selected_cc
        random.shuffle(mixed_data)
        
        actual_ratio = len(selected_cc) / len(mixed_data)
        print(f"  wrong_cot: {num_wrong}")
        print(f"  correct_cot: {len(selected_cc)} (从 {len(correct_cot_candidates)} 个候选中选取)")
        print(f"  混合后总样本数: {len(mixed_data)}")
        print(f"  实际 correct_cot 比例: {actual_ratio:.1%}")
    else:
        mixed_data = wrong_cot_data
        print(f"  ⚠️ 无可用的 correct_cot 候选, 仅使用 wrong_cot")
        print(f"  总样本数: {len(mixed_data)}")

    # ====== 统计信息 ======
    print(f"\n{'=' * 60}")
    print(f"📊 最终训练集统计")
    print(f"{'=' * 60}")
    
    src_counts = Counter(d.get('think_chain_source', 'unknown') for d in mixed_data)
    for src, cnt in src_counts.most_common():
        print(f"  {src}: {cnt} ({cnt/len(mixed_data)*100:.1f}%)")
    
    # 步骤数分布
    step_counts = [count_steps(d.get('think_chain', '')) for d in mixed_data]
    step_dist = Counter(step_counts)
    print(f"\n  步骤数分布:")
    for n_steps in sorted(step_dist.keys()):
        cnt = step_dist[n_steps]
        print(f"    {n_steps} 步: {cnt} ({cnt/len(mixed_data)*100:.1f}%)")
    
    # pass_rate 分布 (仅 wrong_cot)
    wrong_items = [d for d in mixed_data if d.get('think_chain_source') == 'wrong_cot']
    if wrong_items:
        pr_values = [d.get('rejection_sampling_meta', {}).get('pass_rate', -1) for d in wrong_items]
        pr_zero = sum(1 for p in pr_values if p == 0)
        pr_positive = sum(1 for p in pr_values if p > 0)
        print(f"\n  wrong_cot pass_rate 分布:")
        print(f"    pass_rate=0: {pr_zero} ({pr_zero/len(wrong_items)*100:.1f}%)")
        print(f"    pass_rate>0: {pr_positive} ({pr_positive/len(wrong_items)*100:.1f}%)")

    # ====== 写入输出文件 ======
    print(f"\n💾 写入: {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(mixed_data, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(args.output_json) / (1024 * 1024)
    print(f"  文件大小: {file_size:.1f} MB")
    print(f"\n✅ 完成!")


if __name__ == '__main__':
    main()
