#!/usr/bin/env python3
"""
从 MMathCoT-1M 原始数据集中抽取训练集未使用的样本作为测试集。

用法:
    python scripts/prepare_test_data.py \
        --train_json rld_training_mixed.json \
        --source_jsonl data/MMathCoT-1M/train.jsonl \
        --image_root data/MMathCoT-1M \
        --num_samples 50 \
        --output test_unseen_50.json
"""

import argparse
import json
import os
import random
import re
import sys


def extract_image_basenames(train_json_path: str) -> set:
    """从训练集 JSON 中提取所有图片的 basename 集合 (快速去重)"""
    print(f"[1/3] 提取训练集图片列表: {train_json_path}")
    basenames = set()
    with open(train_json_path, 'r') as f:
        data = json.load(f)
    for item in data:
        img = item.get('image', '')
        basenames.add(os.path.basename(img))
    print(f"  训练集样本数: {len(data)}, 唯一图片: {len(basenames)}")
    return basenames


def sample_unseen(
    source_jsonl: str,
    image_root: str,
    train_basenames: set,
    num_samples: int = 50,
    seed: int = 42,
) -> list:
    """从原始数据集中随机抽取训练集未使用的样本"""
    print(f"[2/3] 扫描原始数据集: {source_jsonl}")
    candidates = []
    total = 0
    skipped_in_train = 0
    skipped_no_image = 0
    skipped_no_answer = 0

    with open(source_jsonl, 'r') as f:
        for line in f:
            total += 1
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            image_url = item.get('image_url', '')
            basename = os.path.basename(image_url)

            # 跳过训练集中已有的
            if basename in train_basenames:
                skipped_in_train += 1
                continue

            # 检查图片是否存在
            full_path = os.path.join(image_root, image_url)
            if not os.path.exists(full_path):
                skipped_no_image += 1
                continue

            # 提取答案 (格式: "†Answer: X")
            output = item.get('output', '')
            answer_match = re.search(r'†Answer:\s*(.+?)$', output, re.MULTILINE)
            if not answer_match:
                skipped_no_answer += 1
                continue

            gt_answer = answer_match.group(1).strip()

            # 提取问题 (去掉 instruction 前缀)
            instruction = item.get('instruction', '')
            # 常见前缀: "you are given a math problem image, please solve the problem step by step. \nQuestion:"
            question = instruction
            if '\nQuestion:' in instruction:
                question = instruction.split('\nQuestion:', 1)[1].strip()
            elif 'Question:' in instruction:
                question = instruction.split('Question:', 1)[1].strip()

            candidates.append({
                'image': full_path,
                'question': question,
                'answer': gt_answer,
                'image_url': image_url,  # 保留原始相对路径
            })

            # 收集足够多的候选后提前退出 (加速)
            if len(candidates) >= num_samples * 10:
                break

    print(f"  原始数据扫描: {total} 条")
    print(f"  跳过 (在训练集中): {skipped_in_train}")
    print(f"  跳过 (图片不存在): {skipped_no_image}")
    print(f"  跳过 (无答案): {skipped_no_answer}")
    print(f"  候选样本: {len(candidates)}")

    # 随机抽样
    random.seed(seed)
    if len(candidates) > num_samples:
        samples = random.sample(candidates, num_samples)
    else:
        samples = candidates[:num_samples]

    print(f"  最终抽取: {len(samples)} 条")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str,
                        default='rld_training_mixed.json')
    parser.add_argument('--source_jsonl', type=str,
                        default='data/MMathCoT-1M/train.jsonl')
    parser.add_argument('--image_root', type=str,
                        default='data/MMathCoT-1M')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--output', type=str, default='test_unseen_50.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. 提取训练集图片
    train_basenames = extract_image_basenames(args.train_json)

    # 2. 抽取未见样本
    samples = sample_unseen(
        source_jsonl=args.source_jsonl,
        image_root=args.image_root,
        train_basenames=train_basenames,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # 3. 保存
    with open(args.output, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\n[3/3] ✅ 测试集已保存到: {args.output}")
    print(f"  样本数: {len(samples)}")

    # 预览
    for i, s in enumerate(samples[:3]):
        print(f"\n  样本 {i+1}:")
        print(f"    图片: {s['image_url']}")
        print(f"    问题: {s['question'][:80]}...")
        print(f"    答案: {s['answer']}")


if __name__ == '__main__':
    main()
