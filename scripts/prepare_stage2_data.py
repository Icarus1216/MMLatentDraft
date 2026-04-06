#!/usr/bin/env python3
"""
Stage 2 数据准备脚本

从 MMathCoT-1M 原始数据集中提取 correct CoT 样本，转换为训练格式，
并与现有训练集中的 wrong_cot 样本混合。

数据构成 (目标 ~100K):
├── 70K correct_cot (MMathCoT-1M 转换, think+answer 全监督)
├── 20K wrong_cot (基座 rejection sampling 失败的样本, answer-only 监督)
└── 10K 基座自生成 correct_cot (pass_rate>0 的 rejection sampling 结果, 最高质量)

用法:
    python scripts/prepare_stage2_data.py \
        --source_jsonl data/MMathCoT-1M/train.jsonl \
        --image_root data/MMathCoT-1M \
        --existing_train rld_training_mixed.json \
        --output rld_training_stage2.json \
        --num_correct_cot 70000 \
        --num_wrong_cot 20000 \
        --num_self_correct 10000 \
        --seed 42
"""

import argparse
import json
import os
import re
import random
import sys
from collections import Counter


def normalize_answer(answer: str) -> str:
    """标准化答案格式 (与 data.py 中的 _normalize_answer 一致)"""
    if not answer:
        return answer
    text = answer.strip()
    # LaTeX 包裹去除
    text = re.sub(r'\\\((.+?)\\\)', r'\1', text)
    if text.startswith('$') and text.endswith('$') and len(text) > 2:
        text = text[1:-1].strip()
    text = re.sub(r'\\\[(.+?)\\\]', r'\1', text)
    text = text.replace('\\pi', 'π')
    text = text.replace('\\sqrt', '√')
    text = text.replace('\\times', '×')
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = text.replace('\\,', ' ')
    # Markdown
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # 选择题
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in 'ABCDE':
        return stripped.upper()
    m = re.match(r'^([A-Ea-e])\s*[:.]\s*.+', stripped)
    if m:
        return m.group(1).upper()
    # 数值尾部零
    m_num = re.match(r'^(-?[\d]+\.[\d]+)(.*)', text)
    if m_num:
        num_str = m_num.group(1)
        suffix = m_num.group(2)
        cleaned = num_str.rstrip('0').rstrip('.')
        text = cleaned + suffix
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def convert_mmathcot_output_to_steps(output_text: str) -> tuple:
    """
    将 MMathCoT-1M 的 output 字段转换为 "Step N: ..." 格式的推理链
    
    输入格式: "Step 1: ... Step 2: ... †Answer: X"
    输出: (cot_text, final_answer)
      cot_text: "Step 1: ...\nStep 2: ...\n"
      final_answer: "X"
    """
    text = output_text.strip()
    
    # 提取 †Answer: 后面的答案
    final_answer = ""
    answer_match = re.search(r'†Answer:\s*(.+?)$', text, re.MULTILINE)
    if answer_match:
        final_answer = answer_match.group(1).strip()
        text = text[:answer_match.start()].strip()
    
    # 检查是否已经有 "Step N:" 格式
    if re.search(r'Step\s+\d+\s*[:.]', text, re.IGNORECASE):
        # 已经是 Step N: 格式，直接使用
        # 确保每个 Step 独占一行
        lines = []
        current_step = ""
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if re.match(r'Step\s+\d+\s*[:.]', line, re.IGNORECASE):
                if current_step:
                    lines.append(current_step)
                current_step = line
            else:
                if current_step:
                    current_step += " " + line
                else:
                    current_step = line
        if current_step:
            lines.append(current_step)
        
        cot_text = "\n".join(lines) + "\n"
    else:
        # 不是 Step N: 格式，按句子分割并添加编号
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [text]
        
        lines = []
        for i, sent in enumerate(sentences, 1):
            lines.append(f"Step {i}: {sent}")
        cot_text = "\n".join(lines) + "\n"
    
    return cot_text, final_answer


def extract_question(instruction: str) -> str:
    """从 MMathCoT-1M 的 instruction 字段提取问题"""
    question = instruction
    # 常见前缀模式
    for prefix in [
        "you are given a math problem image, please solve the problem step by step.\nQuestion:",
        "you are given a math problem image, please solve the problem step by step. \nQuestion:",
        "Question:",
    ]:
        if prefix.lower() in question.lower():
            idx = question.lower().index(prefix.lower())
            question = question[idx + len(prefix):].strip()
            break
    return question


def load_existing_training_data(path: str) -> list:
    """加载现有训练集"""
    print(f"[1/4] 加载现有训练集: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    
    # 统计来源分布
    sources = Counter(d.get('think_chain_source', 'unknown') for d in data)
    print(f"  总样本数: {len(data)}")
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt} ({cnt/len(data)*100:.1f}%)")
    
    return data


def extract_correct_cot_from_mmathcot(
    source_jsonl: str,
    image_root: str,
    existing_images: set,
    num_samples: int = 70000,
    seed: int = 42,
    skip_image_check: bool = False,
) -> list:
    """
    从 MMathCoT-1M 中提取 correct CoT 样本
    
    策略:
    - 排除已在训练集中的图片 (避免数据泄漏)
    - 只保留有完整 CoT 和答案的样本
    - 检查图片是否存在
    - 转换为训练格式
    """
    print(f"\n[2/4] 从 MMathCoT-1M 提取 correct CoT 样本...")
    print(f"  源文件: {source_jsonl}")
    print(f"  目标数量: {num_samples}")
    
    candidates = []
    total = 0
    skipped_in_train = 0
    skipped_no_image = 0
    skipped_no_answer = 0
    skipped_too_short = 0
    
    with open(source_jsonl, 'r') as f:
        for line in f:
            total += 1
            if total % 50000 == 0:
                print(f"  已扫描 {total} 条, 候选 {len(candidates)} 条...", flush=True)
            
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            image_url = item.get('image_url', '')
            basename = os.path.basename(image_url)
            
            # 排除训练集中已有的图片
            if basename in existing_images:
                skipped_in_train += 1
                continue
            
            # 检查图片是否存在 (可跳过以加速)
            full_path = os.path.join(image_root, image_url)
            if not skip_image_check and not os.path.exists(full_path):
                skipped_no_image += 1
                continue
            
            # 提取答案
            output = item.get('output', '')
            answer_match = re.search(r'†Answer:\s*(.+?)$', output, re.MULTILINE)
            if not answer_match:
                skipped_no_answer += 1
                continue
            
            # 转换 CoT 格式
            cot_text, raw_answer = convert_mmathcot_output_to_steps(output)
            
            # 检查 CoT 质量: 至少 2 步
            num_steps = len(re.findall(r'Step\s+\d+\s*[:.]', cot_text, re.IGNORECASE))
            if num_steps < 2:
                skipped_too_short += 1
                continue
            
            # 提取问题
            question = extract_question(item.get('instruction', ''))
            
            # 标准化答案
            final_answer = normalize_answer(raw_answer)
            
            # 构造训练格式
            think_chain = cot_text + f"Final Answer: {final_answer}"
            
            candidates.append({
                'image': full_path,
                'question': question,
                'answer': final_answer,
                'think_chain': think_chain,
                'think_chain_status': 'valid',
                'think_chain_source': 'correct_cot',  # 标记为 correct_cot
                'data_source': 'mmathcot_1m',
            })
            
            # 收集足够多候选后提前退出
            if len(candidates) >= num_samples * 3:
                break
    
    print(f"  扫描完成: {total} 条")
    print(f"  跳过 (在训练集中): {skipped_in_train}")
    print(f"  跳过 (图片不存在): {skipped_no_image}")
    print(f"  跳过 (无答案): {skipped_no_answer}")
    print(f"  跳过 (步骤太少): {skipped_too_short}")
    print(f"  候选样本: {len(candidates)}")
    
    # 随机抽样
    random.seed(seed)
    if len(candidates) > num_samples:
        samples = random.sample(candidates, num_samples)
    else:
        samples = candidates
        print(f"  ⚠️ 候选不足 {num_samples}, 使用全部 {len(candidates)} 条")
    
    print(f"  最终提取: {len(samples)} 条 correct_cot")
    return samples


def extract_subsets_from_existing(
    existing_data: list,
    num_wrong_cot: int = 20000,
    num_self_correct: int = 10000,
    seed: int = 42,
) -> tuple:
    """
    从现有训练集中提取子集:
    - wrong_cot: 基座推理错误的样本 (answer-only 监督)
    - self_correct: 基座自生成的正确 CoT (dataset_converted, 最高质量)
    """
    print(f"\n[3/4] 从现有训练集提取子集...")
    
    wrong_cot_pool = [d for d in existing_data if d.get('think_chain_source') == 'wrong_cot']
    self_correct_pool = [d for d in existing_data if d.get('think_chain_source') == 'dataset_converted']
    
    print(f"  wrong_cot 池: {len(wrong_cot_pool)}")
    print(f"  dataset_converted (self_correct) 池: {len(self_correct_pool)}")
    
    random.seed(seed)
    
    # 抽取 wrong_cot
    if len(wrong_cot_pool) > num_wrong_cot:
        wrong_cot_samples = random.sample(wrong_cot_pool, num_wrong_cot)
    else:
        wrong_cot_samples = wrong_cot_pool
        print(f"  ⚠️ wrong_cot 不足 {num_wrong_cot}, 使用全部 {len(wrong_cot_pool)} 条")
    
    # 抽取 self_correct (dataset_converted)
    if len(self_correct_pool) > num_self_correct:
        self_correct_samples = random.sample(self_correct_pool, num_self_correct)
    else:
        self_correct_samples = self_correct_pool
        print(f"  ⚠️ dataset_converted 不足 {num_self_correct}, 使用全部 {len(self_correct_pool)} 条")
    
    print(f"  抽取 wrong_cot: {len(wrong_cot_samples)}")
    print(f"  抽取 self_correct: {len(self_correct_samples)}")
    
    return wrong_cot_samples, self_correct_samples


def main():
    parser = argparse.ArgumentParser(description="Stage 2 数据准备")
    parser.add_argument('--source_jsonl', type=str,
                        default='data/MMathCoT-1M/train.jsonl',
                        help='MMathCoT-1M 原始数据路径')
    parser.add_argument('--image_root', type=str,
                        default='data/MMathCoT-1M',
                        help='图片根目录')
    parser.add_argument('--existing_train', type=str,
                        default='rld_training_mixed.json',
                        help='现有训练集 JSON 路径')
    parser.add_argument('--output', type=str,
                        default='rld_training_stage2.json',
                        help='输出文件路径')
    parser.add_argument('--num_correct_cot', type=int, default=70000,
                        help='从 MMathCoT-1M 提取的 correct CoT 数量')
    parser.add_argument('--num_wrong_cot', type=int, default=20000,
                        help='从现有训练集提取的 wrong_cot 数量')
    parser.add_argument('--num_self_correct', type=int, default=10000,
                        help='从现有训练集提取的 self_correct (dataset_converted) 数量')
    parser.add_argument('--skip_image_check', action='store_true',
                        help='跳过图片存在性检查 (网络文件系统上 os.path.exists 很慢)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 1. 加载现有训练集
    existing_data = load_existing_training_data(args.existing_train)
    
    # 提取现有训练集的图片集合 (用于去重)
    existing_images = set()
    for item in existing_data:
        img = item.get('image', '')
        if img:
            existing_images.add(os.path.basename(img))
    print(f"  现有训练集唯一图片: {len(existing_images)}")
    
    # 2. 从 MMathCoT-1M 提取 correct CoT
    correct_cot_samples = extract_correct_cot_from_mmathcot(
        source_jsonl=args.source_jsonl,
        image_root=args.image_root,
        existing_images=existing_images,
        num_samples=args.num_correct_cot,
        seed=args.seed,
        skip_image_check=args.skip_image_check,
    )
    
    # 3. 从现有训练集提取子集
    wrong_cot_samples, self_correct_samples = extract_subsets_from_existing(
        existing_data,
        num_wrong_cot=args.num_wrong_cot,
        num_self_correct=args.num_self_correct,
        seed=args.seed,
    )
    
    # 4. 混合并保存
    print(f"\n[4/4] 混合数据并保存...")
    all_samples = correct_cot_samples + wrong_cot_samples + self_correct_samples
    random.seed(args.seed)
    random.shuffle(all_samples)
    
    # 统计
    sources = Counter(d.get('think_chain_source', 'unknown') for d in all_samples)
    print(f"  总样本数: {len(all_samples)}")
    for src, cnt in sources.most_common():
        supervised = "think+answer" if src in ('correct_cot', 'dataset_converted', 'free_reasoning', 'corrected_free_reasoning') else "answer-only"
        print(f"    {src}: {cnt} ({cnt/len(all_samples)*100:.1f}%) → {supervised} 监督")
    
    with open(args.output, 'w') as f:
        json.dump(all_samples, f, ensure_ascii=False)
    
    print(f"\n✅ Stage 2 训练数据已保存到: {args.output}")
    print(f"  总样本数: {len(all_samples)}")
    
    # 预估 token 利用率
    n_supervised = sum(1 for d in all_samples if d.get('think_chain_source') in 
                       ('correct_cot', 'dataset_converted', 'free_reasoning', 'corrected_free_reasoning'))
    print(f"  think+answer 全监督样本: {n_supervised} ({n_supervised/len(all_samples)*100:.1f}%)")
    print(f"  预估 token 利用率: ~{n_supervised/len(all_samples)*80 + (1-n_supervised/len(all_samples))*5:.0f}%")
    print(f"  (Stage 1 token 利用率: ~4.5%)")


if __name__ == '__main__':
    main()
