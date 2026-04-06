#!/usr/bin/env python3
"""
Stage 2 数据合并脚本

合并 Stage 1 训练数据和 Stage 2 checked 数据，基于 (image, question) 去重。
Stage 2 数据优先保留（因为经过了 rejection sampling 质量更高）。

用法:
    python scripts/merge_stage2_data.py \
        --stage1_data data/rld_50k_newformat.json \
        --stage2_data rld_training_stage2_checked.json \
        --output data/rld_stage2_merged.json
"""

import json
import argparse
import os
from pathlib import Path


def normalize_image_path(path: str) -> str:
    """标准化图片路径，去掉前缀差异以便比较"""
    # 去掉可能的绝对路径前缀，只保留相对部分
    # 例如 "/mnt/cephszjt/user_juntianzhang/LatentDraft/data/MMathCoT-1M/xxx" -> "MMathCoT-1M/xxx"
    # 或 "data/MMathCoT-1M/xxx" -> "MMathCoT-1M/xxx"
    path = path.strip()
    
    # 常见前缀列表
    prefixes = [
        "/mnt/cephszjt/user_juntianzhang/LatentDraft/data/",
        "/mnt/cephszjt/user_juntianzhang/LatentDraft/",
        "data/",
    ]
    for prefix in prefixes:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    
    return path


def normalize_question(question: str) -> str:
    """标准化问题文本，去掉多余空白"""
    return " ".join(question.split()).strip()


def make_dedup_key(sample: dict) -> str:
    """生成去重键: (标准化图片路径, 标准化问题)"""
    img = normalize_image_path(sample.get("image", ""))
    q = normalize_question(sample.get("question", ""))
    return f"{img}|||{q}"


def main():
    parser = argparse.ArgumentParser(description="合并 Stage 1 和 Stage 2 训练数据")
    parser.add_argument("--stage1_data", type=str, 
                        default="data/rld_50k_newformat.json",
                        help="Stage 1 训练数据路径")
    parser.add_argument("--stage2_data", type=str,
                        default="rld_training_stage2_checked.json",
                        help="Stage 2 checked 数据路径")
    parser.add_argument("--output", type=str,
                        default="data/rld_stage2_merged.json",
                        help="合并后的输出路径")
    args = parser.parse_args()

    print(f"📂 Stage 1 数据: {args.stage1_data}")
    print(f"📂 Stage 2 数据: {args.stage2_data}")
    print(f"📂 输出路径: {args.output}")
    print()

    # 加载数据
    print("⏳ 加载 Stage 1 数据...")
    with open(args.stage1_data, 'r') as f:
        stage1_data = json.load(f)
    print(f"  Stage 1: {len(stage1_data)} 条样本")

    print("⏳ 加载 Stage 2 数据...")
    with open(args.stage2_data, 'r') as f:
        stage2_data = json.load(f)
    print(f"  Stage 2: {len(stage2_data)} 条样本")
    print()

    # 去重: Stage 2 数据优先
    seen_keys = set()
    merged = []

    # 先添加 Stage 2 数据（优先保留）
    stage2_dupes = 0
    for sample in stage2_data:
        key = make_dedup_key(sample)
        if key in seen_keys:
            stage2_dupes += 1
            continue
        seen_keys.add(key)
        merged.append(sample)
    
    print(f"✅ Stage 2 保留: {len(merged)} 条 (内部去重 {stage2_dupes} 条)")

    # 再添加 Stage 1 数据（去掉与 Stage 2 重复的）
    stage1_added = 0
    stage1_dupes = 0
    for sample in stage1_data:
        key = make_dedup_key(sample)
        if key in seen_keys:
            stage1_dupes += 1
            continue
        seen_keys.add(key)
        merged.append(sample)
        stage1_added += 1

    print(f"✅ Stage 1 新增: {stage1_added} 条 (与 Stage 2 重复 {stage1_dupes} 条)")
    print()
    print(f"📊 合并结果:")
    print(f"  总样本数: {len(merged)}")
    print(f"  来自 Stage 2: {len(merged) - stage1_added}")
    print(f"  来自 Stage 1: {stage1_added}")

    # 统计数据来源分布
    source_counts = {}
    for sample in merged:
        src = sample.get("think_chain_source", sample.get("data_source", "unknown"))
        source_counts[src] = source_counts.get(src, 0) + 1
    print(f"\n📊 数据来源分布:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} ({cnt/len(merged)*100:.1f}%)")

    # 保存
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    print(f"\n💾 保存到 {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(merged, f, ensure_ascii=False)
    
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"✅ 保存完成! 文件大小: {file_size:.1f} MB")


if __name__ == "__main__":
    main()
