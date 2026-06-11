#!/usr/bin/env python3
"""
FlagEval/ERQA 数据集下载与准备脚本

数据集说明:
  - 来源: https://huggingface.co/datasets/FlagEval/ERQA
  - 任务: 具身推理问答 (Embodied Reasoning QA)
  - 内容: 400个测试样本，涵盖空间推理和真实世界场景知识
  - 格式: 多图像VQA (每个问题可能关联多张图像)

用法:
  python3 prepare_erqa.py
  
  # 指定保存目录
  python3 prepare_erqa.py --save_dir /path/to/save
  
  # 使用镜像
  python3 prepare_erqa.py --mirror
"""

import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="下载并准备 FlagEval/ERQA 数据集")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/erqa",
        help="数据集保存目录 (默认: ./data/erqa)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace 缓存目录 (默认: 自动检测)"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 HuggingFace 镜像站 (hf-mirror.com)"
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="跳过图像保存，仅生成元数据"
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="图像保存格式 (默认: jpg)"
    )
    return parser.parse_args()


def setup_environment(args):
    """设置环境变量"""
    # 设置缓存目录
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
    else:
        # 尝试使用可写的缓存目录
        default_cache = os.path.expanduser("~/.cache/huggingface")
        if not os.access(os.path.dirname(default_cache), os.W_OK):
            # 回退到工作区目录
            workspace_cache = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".cache", "huggingface"
            )
            os.makedirs(workspace_cache, exist_ok=True)
            os.environ["HF_HOME"] = workspace_cache
            print(f"📁 缓存目录: {workspace_cache}")
    
    # 设置镜像
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("🌐 使用 HuggingFace 镜像站: hf-mirror.com")


def download_dataset(args):
    """下载数据集"""
    from datasets import load_dataset
    
    cache_dir = args.cache_dir
    if not cache_dir:
        cache_dir = os.environ.get("HF_HOME", None)
        if cache_dir:
            cache_dir = os.path.join(cache_dir, "datasets")
    
    print("📥 正在下载 FlagEval/ERQA 数据集...")
    ds = load_dataset(
        "FlagEval/ERQA",
        split="test",
        cache_dir=cache_dir
    )
    print(f"✅ 下载完成! 样本数: {len(ds)}")
    print(f"   特征: {list(ds.features.keys())}")
    return ds


def analyze_dataset(ds):
    """分析数据集统计信息"""
    print("\n" + "=" * 60)
    print("📊 数据集统计信息")
    print("=" * 60)
    
    # Question Type 分布
    types = []
    img_counts = []
    answer_lengths = []
    question_lengths = []
    
    for i in range(len(ds)):
        sample = ds[i]
        types.append(sample["question_type"])
        img_counts.append(len(sample["images"]))
        answer_lengths.append(len(sample["answer"]))
        question_lengths.append(len(sample["question"]))
    
    type_counts = Counter(types)
    img_count_dist = Counter(img_counts)
    
    print(f"\n总样本数: {len(ds)}")
    
    print(f"\n📋 Question Type 分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(c / len(ds) * 40)
        print(f"  {t:30s}: {c:4d} ({c/len(ds)*100:5.1f}%) {bar}")
    
    print(f"\n🖼️  每样本图像数分布:")
    for k, v in sorted(img_count_dist.items()):
        print(f"  {k} 张图像: {v:4d} 个样本 ({v/len(ds)*100:.1f}%)")
    
    print(f"\n📝 文本长度统计:")
    print(f"  问题长度: min={min(question_lengths)}, max={max(question_lengths)}, "
          f"avg={sum(question_lengths)/len(question_lengths):.1f}")
    print(f"  答案长度: min={min(answer_lengths)}, max={max(answer_lengths)}, "
          f"avg={sum(answer_lengths)/len(answer_lengths):.1f}")
    
    return type_counts, img_count_dist


def save_dataset(ds, args, type_counts, img_count_dist):
    """保存数据集到本地"""
    save_dir = Path(args.save_dir)
    img_dir = save_dir / "images"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 正在保存数据集到: {save_dir}")
    
    metadata = []
    total_images = 0
    
    for i in range(len(ds)):
        sample = ds[i]
        
        # 保存图像
        img_paths = []
        if not args.skip_images:
            for j, img in enumerate(sample["images"]):
                img_filename = f'{sample["question_id"]}_img{j}.{args.image_format}'
                img_path = img_dir / img_filename
                if not img_path.exists():
                    if args.image_format == "jpg":
                        img.save(str(img_path), quality=95)
                    else:
                        img.save(str(img_path))
                img_paths.append(f'images/{img_filename}')
                total_images += 1
        
        metadata.append({
            "question_id": sample["question_id"],
            "question": sample["question"],
            "question_type": sample["question_type"],
            "answer": sample["answer"],
            "visual_indices": sample["visual_indices"],
            "image_paths": img_paths,
            "num_images": len(sample["images"])
        })
        
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(ds)} ({(i+1)/len(ds)*100:.0f}%)")
    
    # 保存元数据 JSONL
    meta_path = save_dir / "erqa_test.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 保存统计信息
    stats = {
        "dataset": "FlagEval/ERQA",
        "split": "test",
        "num_samples": len(metadata),
        "total_images": total_images,
        "question_types": dict(type_counts),
        "images_per_sample": dict(img_count_dist),
        "save_dir": str(save_dir.absolute()),
        "image_format": args.image_format
    }
    stats_path = save_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return meta_path, stats_path, total_images


def show_samples(ds, n=3):
    """展示样本示例"""
    print(f"\n" + "=" * 60)
    print(f"🔍 样本示例 (前{n}个)")
    print("=" * 60)
    
    for i in range(min(n, len(ds))):
        sample = ds[i]
        print(f"\n--- 样本 {i} ---")
        print(f"  ID:           {sample['question_id']}")
        print(f"  类型:         {sample['question_type']}")
        print(f"  问题:         {sample['question'][:200]}")
        print(f"  答案:         {sample['answer'][:200]}")
        print(f"  图像数:       {len(sample['images'])}")
        print(f"  visual_idx:   {sample['visual_indices']}")
        if sample['images']:
            img = sample['images'][0]
            print(f"  首张图尺寸:   {img.size} (WxH)")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🚀 FlagEval/ERQA 数据集准备工具")
    print("=" * 60)
    print(f"  保存目录: {args.save_dir}")
    print(f"  图像格式: {args.image_format}")
    print(f"  跳过图像: {args.skip_images}")
    print()
    
    # 设置环境
    setup_environment(args)
    
    # 下载数据集
    ds = download_dataset(args)
    
    # 展示样本
    show_samples(ds)
    
    # 分析统计
    type_counts, img_count_dist = analyze_dataset(ds)
    
    # 保存到本地
    meta_path, stats_path, total_images = save_dataset(
        ds, args, type_counts, img_count_dist
    )
    
    # 最终总结
    print("\n" + "=" * 60)
    print("✅ 数据集准备完成!")
    print("=" * 60)
    print(f"  📄 元数据文件:   {meta_path}")
    print(f"  📊 统计信息:     {stats_path}")
    if not args.skip_images:
        print(f"  🖼️  图像目录:     {Path(args.save_dir) / 'images'}")
        print(f"  🖼️  图像总数:     {total_images}")
    print(f"  📝 样本总数:     {len(ds)}")
    print()
    print("目录结构:")
    print(f"  {args.save_dir}/")
    print(f"  ├── erqa_test.jsonl      # 元数据 (每行一个JSON)")
    print(f"  ├── dataset_stats.json   # 统计信息")
    if not args.skip_images:
        print(f"  └── images/              # 图像文件")
        print(f"      ├── <qid>_img0.{args.image_format}")
        print(f"      ├── <qid>_img1.{args.image_format}")
        print(f"      └── ...")
    print()
    print("JSONL 格式说明:")
    print('  {"question_id": "...", "question": "...", "question_type": "...",')
    print('   "answer": "...", "visual_indices": [...], "image_paths": [...],')
    print('   "num_images": N}')


if __name__ == "__main__":
    main()
