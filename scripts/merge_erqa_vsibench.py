"""
合并 ERQA + VSI-Bench 训练数据
- ERQA: 400 条 (单图, image_path 字段)
- VSI-Bench: 1162 条 (多图, image_paths 字段)
- 输出: 1562 条混合训练集, 随机打乱
"""

import json
import random
import os
from collections import Counter

def main():
    random.seed(42)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 加载 ERQA 数据
    erqa_path = os.path.join(base_dir, "data/erqa/erqa_latent_cot_v2.json")
    print(f"📂 加载 ERQA: {erqa_path}")
    with open(erqa_path) as f:
        erqa_data = json.load(f)
    print(f"   ✅ {len(erqa_data)} 条")
    
    # 加载 VSI-Bench 数据
    vsibench_path = os.path.join(base_dir, "data/vsibench/vsibench_direction_route_latent_cot.json")
    print(f"📂 加载 VSI-Bench: {vsibench_path}")
    with open(vsibench_path) as f:
        vsibench_data = json.load(f)
    print(f"   ✅ {len(vsibench_data)} 条")
    
    # 为每条数据添加 source 标记 (方便后续分析)
    for d in erqa_data:
        d['source'] = 'erqa'
    for d in vsibench_data:
        d['source'] = 'vsibench'
    
    # 合并
    merged = erqa_data + vsibench_data
    print(f"\n📊 合并前统计:")
    print(f"   ERQA:      {len(erqa_data)} 条")
    print(f"   VSI-Bench: {len(vsibench_data)} 条")
    print(f"   总计:      {len(merged)} 条")
    
    # 打乱
    random.shuffle(merged)
    
    # 统计
    print(f"\n📊 合并后统计:")
    source_dist = Counter(d['source'] for d in merged)
    for k, v in source_dist.most_common():
        print(f"   {k}: {v} ({v/len(merged)*100:.1f}%)")
    
    task_dist = Counter(d.get('task_type', '?') for d in merged)
    print(f"\n   task_type 分布:")
    for k, v in task_dist.most_common():
        print(f"     {k}: {v} ({v/len(merged)*100:.1f}%)")
    
    # 图片格式统计
    single_img = sum(1 for d in merged if 'image_path' in d and 'image_paths' not in d)
    multi_img = sum(1 for d in merged if 'image_paths' in d)
    print(f"\n   单图样本: {single_img}")
    print(f"   多图样本: {multi_img}")
    
    # pause 统计
    pause_counts = [d['reasoning_for_training'].count('<|pause|>') for d in merged]
    print(f"\n   pause 分布: min={min(pause_counts)}, max={max(pause_counts)}, avg={sum(pause_counts)/len(pause_counts):.2f}")
    
    # 保存
    output_dir = os.path.join(base_dir, "data/erqa_vsibench_merged")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "erqa_vsibench_merged_1562.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n✅ 保存到: {output_path}")
    print(f"   文件大小: {file_size:.2f} MB")
    print(f"   总样本数: {len(merged)}")


if __name__ == "__main__":
    main()
