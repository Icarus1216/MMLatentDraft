#!/usr/bin/env python3
"""
构建纯 correct_cot 的 Stage 2 训练集

目标: 用 correct_cot 替换所有 wrong_cot 样本，生成 ~80K 纯干净训练数据。

数据源:
  1. rld_training_stage2_checked.json (59K) — 已有 correct_cot (新格式) + wrong_cot (新格式)
  2. rld_50k_newformat.json (50K) — Stage 1 数据 (新格式, 全部 dataset_converted)
  3. rld_mmathcot_filtered_checked.json (266K) — 原始数据池 (旧格式, 需转换)

策略:
  Step 1: 从 stage2_checked 中保留 correct_cot + dataset_converted (丢弃 wrong_cot)
  Step 2: 从 stage1 数据中保留与 step1 不重复的样本
  Step 3: 从 266K 原始数据池中补充新的 correct_cot (旧格式→新格式转换)
  Step 4: 去重合并，目标 ~80K

格式转换:
  旧格式: "<think>\nstep1_content\n</step>\nstep2_content\n</step>\n</think>\nAnswer: xxx"
  新格式: "Step 1: step1_content\nStep 2: step2_content\n"
  (不含 "Final Answer:" 行，answer 由 data.py 在运行时拼接)

用法:
  python scripts/build_clean_stage2_data.py \
    --stage2_checked rld_training_stage2_checked.json \
    --stage1_data data/rld_50k_newformat.json \
    --raw_pool data/rld_mmathcot_filtered_checked.json \
    --output data/rld_stage2_clean.json \
    --target_size 80000 \
    --max_steps 14 \
    --seed 42
"""

import argparse
import json
import os
import re
import random
from collections import Counter


# ====== 格式转换工具 ======

def count_steps(think_chain: str) -> int:
    """统计推理链中的步骤数 (兼容新旧格式)"""
    if not think_chain:
        return 0
    # 新格式: "Step N:"
    step_n_count = len(re.findall(r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]', think_chain, re.IGNORECASE))
    if step_n_count > 0:
        return step_n_count
    # 旧格式: "</step>"
    return think_chain.count('</step>')


def is_new_format(think_chain: str) -> bool:
    """判断是否为新格式 (Step N: ...)"""
    return bool(re.search(r'(?:^|\n)\s*Step\s+\d+\s*:', think_chain))


def convert_old_to_new_format(think_chain: str) -> str:
    """
    将旧格式转换为新格式。
    
    旧格式: "<think>\nstep1_content\n</step>\nstep2_content\n</step>\n</think>\nAnswer: xxx"
    新格式: "Step 1: step1_content\nStep 2: step2_content\n"
    
    返回值只包含推理步骤部分，不包含 "Final Answer:" 行。
    """
    text = think_chain
    
    # 1) 剥离 <think>...</think> 包裹及其后面的内容 (如 "\nAnswer: 12")
    think_end_pos = text.rfind('</think>')
    if think_end_pos != -1:
        text = text[:think_end_pos]
    if text.startswith('<think>'):
        text = text[len('<think>'):]
    text = text.strip('\n').strip()
    
    # 2) 按 </step> 分割为步骤
    parts = text.split('</step>')
    steps = [p.strip() for p in parts if p.strip()]
    
    if not steps:
        return "Step 1: Analyzing the image.\n"
    
    # 3) 重新编号为 "Step N: ..." 格式
    cot_parts = []
    for i, step_text in enumerate(steps, 1):
        # 清理步骤文本中可能残留的标签
        step_text = step_text.strip()
        if step_text.startswith('<step>'):
            step_text = step_text[len('<step>'):].strip()
        cot_parts.append(f"Step {i}: {step_text}")
    
    return "\n".join(cot_parts) + "\n"


def extract_answer_from_old_format(think_chain: str) -> str:
    """从旧格式中提取 answer (</think> 后面的 Answer: xxx)"""
    think_end_pos = think_chain.rfind('</think>')
    if think_end_pos == -1:
        return ""
    after = think_chain[think_end_pos + len('</think>'):].strip()
    # 匹配 "Answer: xxx" 或 "answer: xxx"
    m = re.match(r'(?:Answer|answer)\s*:\s*(.+)', after)
    if m:
        return m.group(1).strip()
    return after.strip()


def validate_converted_cot(cot_text: str, min_steps: int = 2, max_steps: int = 14) -> bool:
    """验证转换后的 CoT 是否有效"""
    if not cot_text or len(cot_text) < 20:
        return False
    n_steps = count_steps(cot_text)
    if n_steps < min_steps or n_steps > max_steps:
        return False
    # 检查是否有 "Step 1:"
    if not re.search(r'Step\s+1\s*:', cot_text):
        return False
    return True


# ====== 去重工具 ======

def normalize_image_path(path: str) -> str:
    """标准化图片路径"""
    path = path.strip()
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


def make_dedup_key(sample: dict) -> str:
    """生成去重键: (标准化图片路径, 标准化问题前100字符)"""
    img = normalize_image_path(sample.get("image", ""))
    q = " ".join(sample.get("question", "").split()).strip()[:100]
    return f"{img}|||{q}"


def main():
    parser = argparse.ArgumentParser(description="构建纯 correct_cot 的 Stage 2 训练集")
    parser.add_argument("--stage2_checked", type=str,
                        default="rld_training_stage2_checked.json",
                        help="Stage 2 checked 数据 (含 correct_cot + wrong_cot)")
    parser.add_argument("--stage1_data", type=str,
                        default="data/rld_50k_newformat.json",
                        help="Stage 1 训练数据 (新格式)")
    parser.add_argument("--raw_pool", type=str,
                        default="data/rld_mmathcot_filtered_checked.json",
                        help="原始数据池 (266K, 旧格式)")
    parser.add_argument("--output", type=str,
                        default="data/rld_stage2_clean.json",
                        help="输出路径")
    parser.add_argument("--target_size", type=int, default=80000,
                        help="目标样本数 (默认 80000)")
    parser.add_argument("--max_steps", type=int, default=14,
                        help="最大步骤数 (默认 14)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    
    print("=" * 70)
    print("🧹 构建纯 correct_cot Stage 2 训练集")
    print("=" * 70)
    print(f"  目标样本数: {args.target_size}")
    print(f"  最大步骤数: {args.max_steps}")
    print()

    # ====== Step 1: 从 stage2_checked 中保留 correct 样本 ======
    print("[1/5] 加载 Stage 2 checked 数据...")
    with open(args.stage2_checked, 'r') as f:
        stage2_data = json.load(f)
    print(f"  总样本数: {len(stage2_data)}")
    
    # 统计来源分布
    s2_sources = Counter(d.get('think_chain_source', 'unknown') for d in stage2_data)
    for src, cnt in s2_sources.most_common():
        print(f"    {src}: {cnt}")
    
    # 保留 correct_cot 和 dataset_converted，丢弃 wrong_cot
    correct_sources = {'correct_cot', 'dataset_converted', 'free_reasoning', 'corrected_free_reasoning'}
    stage2_correct = []
    stage2_wrong_count = 0
    for item in stage2_data:
        src = item.get('think_chain_source', 'unknown')
        if src in correct_sources:
            # 验证步骤数
            n_steps = count_steps(item.get('think_chain', ''))
            if n_steps <= args.max_steps and n_steps >= 2:
                stage2_correct.append(item)
            # else: 跳过超高/超低步骤数
        else:
            stage2_wrong_count += 1
    
    print(f"  ✅ 保留 correct 样本: {len(stage2_correct)}")
    print(f"  ❌ 丢弃 wrong_cot 样本: {stage2_wrong_count}")
    del stage2_data
    
    # ====== Step 2: 加载 Stage 1 数据 ======
    print(f"\n[2/5] 加载 Stage 1 数据...")
    with open(args.stage1_data, 'r') as f:
        stage1_data = json.load(f)
    print(f"  总样本数: {len(stage1_data)}")
    
    # ====== Step 3: 合并 Stage 2 correct + Stage 1，去重 ======
    print(f"\n[3/5] 合并 Stage 2 correct + Stage 1 (去重)...")
    seen_keys = set()
    merged = []
    
    # Stage 2 correct 优先
    for item in stage2_correct:
        key = make_dedup_key(item)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(item)
    s2_kept = len(merged)
    print(f"  Stage 2 correct 保留: {s2_kept}")
    
    # Stage 1 补充
    s1_added = 0
    for item in stage1_data:
        key = make_dedup_key(item)
        if key not in seen_keys:
            n_steps = count_steps(item.get('think_chain', ''))
            if n_steps <= args.max_steps and n_steps >= 2:
                seen_keys.add(key)
                merged.append(item)
                s1_added += 1
    print(f"  Stage 1 新增: {s1_added}")
    print(f"  当前总数: {len(merged)}")
    del stage1_data, stage2_correct
    
    # ====== Step 4: 从原始数据池补充 ======
    need_more = args.target_size - len(merged)
    if need_more > 0:
        print(f"\n[4/5] 从原始数据池补充 {need_more} 条样本...")
        print(f"  加载 {args.raw_pool} (可能需要较长时间)...")
        with open(args.raw_pool, 'r') as f:
            raw_pool = json.load(f)
        print(f"  原始数据池: {len(raw_pool)} 条")
        
        # 筛选有效的 correct_cot 候选
        candidates = []
        stats = {
            'invalid_status': 0,
            'no_think_chain': 0,
            'duplicate': 0,
            'too_many_steps': 0,
            'too_few_steps': 0,
            'conversion_failed': 0,
            'valid': 0,
        }
        
        for item in raw_pool:
            # 基本有效性检查
            tc = item.get('think_chain', '')
            tc_status = item.get('think_chain_status', '')
            if tc_status != 'valid':
                stats['invalid_status'] += 1
                continue
            if not tc:
                stats['no_think_chain'] += 1
                continue
            if not item.get('image', ''):
                continue
            
            # 去重
            key = make_dedup_key(item)
            if key in seen_keys:
                stats['duplicate'] += 1
                continue
            
            # 格式转换 (旧格式 → 新格式)
            if is_new_format(tc):
                new_tc = tc
            elif '<think>' in tc:
                new_tc = convert_old_to_new_format(tc)
            else:
                stats['conversion_failed'] += 1
                continue
            
            # 验证转换后的 CoT
            if not validate_converted_cot(new_tc, min_steps=2, max_steps=args.max_steps):
                n = count_steps(new_tc)
                if n > args.max_steps:
                    stats['too_many_steps'] += 1
                elif n < 2:
                    stats['too_few_steps'] += 1
                else:
                    stats['conversion_failed'] += 1
                continue
            
            # 构建新格式样本
            # 确保 think_chain 不包含 "Final Answer:" (由 data.py 运行时拼接)
            # 新格式的 think_chain 应以 "\n" 结尾，不含 "Final Answer:"
            if 'Final Answer:' in new_tc:
                # 去掉 Final Answer 行
                lines = new_tc.split('\n')
                lines = [l for l in lines if not l.strip().startswith('Final Answer')]
                new_tc = '\n'.join(lines)
                if not new_tc.endswith('\n'):
                    new_tc += '\n'
            
            new_item = {
                'image': item['image'],
                'question': item['question'],
                'answer': item['answer'],
                'think_chain': new_tc,
                'think_chain_status': 'valid',
                'think_chain_source': 'dataset_converted',  # 标记为 dataset_converted (参与 CoT loss)
                'data_source': item.get('data_source', 'mmathcot_1m'),
            }
            
            candidates.append(new_item)
            seen_keys.add(key)
            stats['valid'] += 1
        
        del raw_pool
        
        print(f"  筛选统计:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
        print(f"  有效候选: {len(candidates)}")
        
        # 随机采样
        if len(candidates) > need_more:
            random.shuffle(candidates)
            selected = candidates[:need_more]
            print(f"  随机采样: {need_more} 条")
        else:
            selected = candidates
            print(f"  ⚠️ 候选不足，全部使用: {len(selected)} 条")
        
        merged.extend(selected)
        print(f"  补充后总数: {len(merged)}")
    else:
        print(f"\n[4/5] 无需从原始数据池补充 (当前 {len(merged)} ≥ 目标 {args.target_size})")
    
    # ====== Step 5: 打乱并保存 ======
    print(f"\n[5/5] 打乱并保存...")
    random.shuffle(merged)
    
    # 最终统计
    print(f"\n{'=' * 70}")
    print(f"📊 最终训练集统计")
    print(f"{'=' * 70}")
    print(f"  总样本数: {len(merged)}")
    
    src_counts = Counter(d.get('think_chain_source', 'unknown') for d in merged)
    print(f"\n  数据来源分布:")
    for src, cnt in src_counts.most_common():
        print(f"    {src}: {cnt} ({cnt/len(merged)*100:.1f}%)")
    
    # 验证: 确认没有 wrong_cot
    wrong_count = sum(1 for d in merged if d.get('think_chain_source') == 'wrong_cot')
    if wrong_count > 0:
        print(f"\n  ⚠️ 警告: 仍有 {wrong_count} 条 wrong_cot 样本!")
    else:
        print(f"\n  ✅ 确认: 0 条 wrong_cot 样本 (100% 干净数据)")
    
    # 步骤数分布
    step_counts = [count_steps(d.get('think_chain', '')) for d in merged]
    step_dist = Counter(step_counts)
    print(f"\n  步骤数分布:")
    for n_steps in sorted(step_dist.keys()):
        cnt = step_dist[n_steps]
        print(f"    {n_steps} 步: {cnt} ({cnt/len(merged)*100:.1f}%)")
    
    # 验证格式: 抽样检查
    print(f"\n  格式验证 (抽样 10 条):")
    sample_indices = random.sample(range(len(merged)), min(10, len(merged)))
    format_ok = 0
    for idx in sample_indices:
        tc = merged[idx].get('think_chain', '')
        has_step1 = bool(re.search(r'Step\s+1\s*:', tc))
        has_old_format = '<think>' in tc or '</step>' in tc
        has_final_answer_in_tc = 'Final Answer:' in tc
        ok = has_step1 and not has_old_format and not has_final_answer_in_tc
        if ok:
            format_ok += 1
        else:
            print(f"    ❌ 样本 {idx}: step1={has_step1}, old_fmt={has_old_format}, final_in_tc={has_final_answer_in_tc}")
            print(f"       think_chain[:200] = {tc[:200]}")
    print(f"    格式正确: {format_ok}/{len(sample_indices)}")
    
    # 保存
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    print(f"\n💾 保存到 {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(merged, f, ensure_ascii=False)
    
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"✅ 保存完成! 文件大小: {file_size:.1f} MB")
    print(f"\n📋 下一步: 更新 configs/rld_train_stage2.yaml 中的 train_json 路径为:")
    print(f"   {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
