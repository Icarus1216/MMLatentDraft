#!/usr/bin/env python3
"""
RLD 数据预过滤脚本

一次性检查所有样本的图片存在性，输出过滤后的 JSON 文件。
后续训练可直接使用过滤后的文件，跳过动态检查，节省 ~4 分钟/次启动时间。

使用方法:
    python scripts/prefilter_data.py \
        --input data/rld_mmathcot_filtered.json \
        --output data/rld_mmathcot_filtered_checked.json

    # 也可以覆盖原文件 (谨慎使用)
    python scripts/prefilter_data.py \
        --input data/rld_mmathcot_filtered.json \
        --inplace
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_image_exists(item):
    """检查单个样本的图片是否存在"""
    img = item.get('image', '')
    if not img:
        return item, 'no_image'
    if not os.path.exists(img):
        return item, 'missing'
    return item, 'valid'


def prefilter_data(input_path: str, output_path: str, num_workers: int = 32):
    """
    预过滤数据集，移除图片缺失的样本
    
    Args:
        input_path: 输入 JSON 文件路径
        output_path: 输出 JSON 文件路径
        num_workers: 并行检查的线程数 (os.path.exists 是 IO 操作，多线程有效)
    """
    print(f"📂 读取数据: {input_path}")
    t0 = time.time()
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    print(f"   加载 {len(raw_data)} 个样本 ({time.time() - t0:.1f}s)")

    print(f"\n🔍 检查图片存在性 (线程数: {num_workers})...")
    t0 = time.time()

    valid_data = []
    missing_count = 0
    no_image_count = 0
    missing_paths = []  # 记录前几个缺失路径，方便调试

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_image_exists, item): i for i, item in enumerate(raw_data)}
        
        done_count = 0
        total = len(raw_data)
        for future in as_completed(futures):
            item, status = future.result()
            done_count += 1
            
            if status == 'valid':
                valid_data.append(item)
            elif status == 'missing':
                missing_count += 1
                if len(missing_paths) < 10:
                    missing_paths.append(item.get('image', ''))
            else:
                no_image_count += 1
            
            # 进度显示
            if done_count % 50000 == 0 or done_count == total:
                print(f"   已检查: {done_count}/{total} ({done_count/total:.1%})")

    elapsed = time.time() - t0
    print(f"   检查完成 ({elapsed:.1f}s)")

    # 统计报告
    print(f"\n📊 过滤统计:")
    print(f"   原始样本数:     {len(raw_data)}")
    print(f"   有效样本数:     {len(valid_data)}")
    print(f"   图片缺失:       {missing_count}")
    if no_image_count > 0:
        print(f"   无图像字段:     {no_image_count}")
    print(f"   过滤比例:       {(missing_count + no_image_count) / max(len(raw_data), 1):.1%}")

    if missing_paths:
        print(f"\n   缺失图片路径示例 (前 {len(missing_paths)} 个):")
        for p in missing_paths:
            print(f"     - {p}")

    # 保存
    print(f"\n💾 保存到: {output_path}")
    t0 = time.time()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(valid_data, f, ensure_ascii=False)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   保存完成 ({time.time() - t0:.1f}s, {file_size:.1f}MB)")
    print(f"\n✅ 预过滤完成！后续训练请使用: {output_path}")
    print(f"   并在配置中设置 skip_image_check: true 以跳过运行时检查")

    return len(valid_data), missing_count + no_image_count


def main():
    parser = argparse.ArgumentParser(description="RLD 数据预过滤: 检查图片存在性")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径 (默认: 在输入文件名后加 _checked)")
    parser.add_argument("--inplace", action="store_true", help="直接覆盖原文件 (谨慎)")
    parser.add_argument("--num_workers", type=int, default=32, help="并行检查线程数 (默认: 32)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)

    if args.inplace:
        output_path = input_path
    elif args.output:
        output_path = args.output
    else:
        # 默认: input_name_checked.json
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_checked{p.suffix}")

    valid_count, filtered_count = prefilter_data(input_path, output_path, args.num_workers)
    
    if filtered_count == 0:
        print(f"\n💡 没有需要过滤的样本，所有图片均存在。")


if __name__ == "__main__":
    main()
