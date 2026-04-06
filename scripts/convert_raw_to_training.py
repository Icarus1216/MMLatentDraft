#!/usr/bin/env python3
"""
从 raw_outputs.json 转换为新格式训练数据

输入: rld_50k_rejection_sampled_raw_outputs.json
  - 每个样本有 4 个 outputs, 每个 output 有 raw_text (Step N: 格式)
  
输出: rld_50k_training_new_format.json
  - 每个样本有 think_chain (Step N: 格式), think_chain_source, rejection_sampling_meta 等

转换逻辑:
  1. 从 raw_text 中提取推理链 (已经是 Step N: 格式, 无需转换)
  2. 按答案正确性分为 correct / wrong
  3. 选择最佳 wrong_cot + 保留 correct_cot
  4. 输出新格式训练数据

与 rejection_sampling.py 的区别:
  - 不需要 vLLM 推理, 直接从已有的 raw outputs 转换
  - 输出新格式 (Step N:), 不再包裹 <think>...</think>
"""

import json
import re
import argparse
import time
from typing import List, Dict, Optional, Tuple
from collections import Counter


# ============================================================
# 复用 rejection_sampling.py 中的工具函数
# ============================================================

def parse_raw_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从 raw_text 中提取推理链和最终答案
    
    raw_text 格式: "Step 1: ...\nStep 2: ...\nFinal Answer: ..."
    
    Returns:
        (cot_text, gen_answer): 推理链文本 (Step N: 格式) 和最终答案
    """
    if not text or not text.strip():
        return None, None
    
    # 提取 Final Answer
    fa_patterns = [
        r'(?:#+\s*)?(?:\*\*)?(?:✅\s*)?Final\s*Answer\s*:?\s*(?:\*\*)?\s*(.+?)(?:\n|$)',
        r'(?:The\s+)?(?:correct\s+)?(?:answer|choice)\s+is\s*:?\s*(?:\*\*)?([^\n*]+)',
        r'(?:^|\n)\s*(?:\*\*)?Answer\s*:?\s*(?:\*\*)?\s*(.+?)(?:\n|$)',
    ]
    
    gen_answer = None
    answer_pos = len(text)
    
    for pat in fa_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            gen_answer = m.group(1).strip()
            answer_pos = m.start()
            # 清理 markdown 格式
            gen_answer = re.sub(r'\*+$', '', gen_answer).strip()
            gen_answer = re.sub(r'^\*+', '', gen_answer).strip()
            gen_answer = re.sub(r'^-+\s*', '', gen_answer).strip()
            gen_answer = re.sub(r'\s*-+$', '', gen_answer).strip()
            break
    
    # 提取推理部分 (Final Answer 之前)
    reasoning_text = text[:answer_pos].strip()
    if not reasoning_text:
        return None, gen_answer
    
    # 检测 Step N: 格式并重新编号
    step_pattern = r'(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?(?:Step\s*\d+)\s*[:.]\s*(?:\*\*)?'
    step_splits = re.split(step_pattern, reasoning_text, flags=re.IGNORECASE)
    steps = [s.strip() for s in step_splits if s and s.strip()]
    
    if len(steps) >= 2:
        # 重新编号为连续的 Step N: 格式
        cot_parts = [f"Step {i}: {s}" for i, s in enumerate(steps, 1)]
        cot_text = "\n".join(cot_parts) + "\n"
        return cot_text, gen_answer
    
    # 无法按 Step N: 分割, 尝试按段落
    para_splits = re.split(r'\n\s*(?:---+|\n)\s*\n', reasoning_text)
    paras = [p.strip() for p in para_splits if p and p.strip() and len(p.strip()) > 15]
    if len(paras) >= 2:
        cot_parts = [f"Step {i}: {p}" for i, p in enumerate(paras, 1)]
        cot_text = "\n".join(cot_parts) + "\n"
        return cot_text, gen_answer
    
    # 按行合并
    lines = [l.strip() for l in reasoning_text.split('\n') if l.strip()]
    if len(lines) >= 3:
        chunk_size = max(2, len(lines) // 4)
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        if len(chunks) >= 2:
            cot_parts = [f"Step {i}: {c}" for i, c in enumerate(chunks, 1)]
            cot_text = "\n".join(cot_parts) + "\n"
            return cot_text, gen_answer
    
    return None, gen_answer


def count_steps(think_content: str) -> int:
    """统计推理链中的步骤数"""
    if not think_content:
        return 0
    return len(re.findall(r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]', think_content, re.IGNORECASE))


def validate_cot(cot_text: str, min_steps: int = 4, max_steps: int = 10,
                 min_chars_per_step: int = 10, max_cot_chars: int = 1500) -> Tuple[bool, str]:
    """验证推理链格式合规性"""
    if not cot_text or not cot_text.strip():
        return False, "空内容"
    if len(cot_text) > max_cot_chars:
        return False, f"超长: {len(cot_text)} > {max_cot_chars}"
    
    num_steps = count_steps(cot_text)
    if num_steps < min_steps:
        return False, f"步骤不足: {num_steps} < {min_steps}"
    if num_steps > max_steps:
        return False, f"步骤过多: {num_steps} > {max_steps}"
    
    # 检查每步内容长度
    step_pattern = r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]'
    step_splits = re.split(step_pattern, cot_text, flags=re.IGNORECASE)
    non_empty = [s.strip() for s in step_splits if s.strip()]
    for i, step in enumerate(non_empty):
        if len(step) < min_chars_per_step:
            return False, f"step {i+1} 过短: {len(step)} < {min_chars_per_step}"
    
    return True, "合规"


def check_answer_match(generated_answer: str, gt_answer: str) -> bool:
    """宽松检查答案匹配"""
    if not generated_answer or not gt_answer:
        return False
    
    def normalize(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    gen_norm = normalize(generated_answer)
    gt_norm = normalize(gt_answer)
    
    if not gen_norm or not gt_norm:
        return False
    
    # 完全匹配
    if gen_norm == gt_norm:
        return True
    
    # 子串匹配
    if gt_norm in gen_norm or gen_norm in gt_norm:
        return True
    
    # 选择题字母匹配
    gen_letter = re.match(r'^([a-e])\b', gen_norm)
    gt_letter = re.match(r'^([a-e])\b', gt_norm)
    if gen_letter and gt_letter and gen_letter.group(1) == gt_letter.group(1):
        return True
    
    # 数值匹配
    def extract_number(text):
        m = re.search(r'-?[\d]+\.?[\d]*', text)
        return float(m.group()) if m else None
    
    gen_num = extract_number(gen_norm)
    gt_num = extract_number(gt_norm)
    if gen_num is not None and gt_num is not None:
        if abs(gen_num - gt_num) < 0.01:
            return True
    
    return False


def select_best_wrong_cot(wrong_cots: List[Dict], correct_cots: List[Dict],
                          max_cot_chars: int = 1500) -> Optional[Dict]:
    """从多个错误 CoT 中选择最有训练价值的一个"""
    if not wrong_cots:
        return None
    
    short_cots = [c for c in wrong_cots if len(c["cot_text"]) <= max_cot_chars]
    candidates = short_cots if short_cots else wrong_cots
    
    if correct_cots:
        avg_correct_steps = sum(count_steps(c["cot_text"]) for c in correct_cots) / len(correct_cots)
    else:
        avg_correct_steps = 7.0
    
    def score(cot):
        steps = count_steps(cot["cot_text"])
        content_len = len(cot["cot_text"])
        s = 0.0
        # 步骤数接近正确 CoT
        step_diff = abs(steps - avg_correct_steps)
        s += 0.4 * max(0, 1.0 - step_diff / 5.0)
        # 步骤数适中
        if 4 <= steps <= 10:
            s += 0.3
        elif steps <= 15:
            s += 0.15
        # 长度简洁
        if 100 <= content_len <= 500:
            s += 0.3
        elif content_len <= 1000:
            s += 0.2
        elif content_len <= max_cot_chars:
            s += 0.1
        else:
            s -= 0.2
        return s
    
    scored = [(score(c), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def select_best_correct_cot(correct_cots: List[Dict], max_cot_chars: int = 1500) -> Optional[Dict]:
    """从多个正确 CoT 中选择最佳的一个"""
    if not correct_cots:
        return None
    
    short_cots = [c for c in correct_cots if len(c["cot_text"]) <= max_cot_chars]
    candidates = short_cots if short_cots else correct_cots
    
    def score(cot):
        steps = count_steps(cot["cot_text"])
        content_len = len(cot["cot_text"])
        s = 0.0
        if 4 <= steps <= 10:
            s += 0.5
        elif steps <= 15:
            s += 0.25
        if 100 <= content_len <= 500:
            s += 0.5
        elif content_len <= 1000:
            s += 0.3
        elif content_len <= max_cot_chars:
            s += 0.1
        else:
            s -= 0.2
        return s
    
    scored = [(score(c), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


# ============================================================
# 主转换逻辑
# ============================================================

def convert_raw_outputs(
    raw_data: List[Dict],
    min_steps: int = 4,
    max_steps: int = 10,
    max_cot_chars: int = 1500,
    image_to_question: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict], Dict]:
    """
    从 raw outputs 转换为新格式训练数据
    
    Args:
        raw_data: raw_outputs.json 中的数据列表
        min_steps: 最少步骤数
        max_steps: 最多步骤数
        max_cot_chars: CoT 最大字符数
        image_to_question: image路径 -> 完整question 的映射 (用于修复截断的question)
    
    Returns:
        (training_data, stats): 训练数据列表和统计信息
    """
    training_data = []
    stats = {
        "total_samples": len(raw_data),
        "has_both": 0,          # 有正确也有错误 (理想)
        "all_correct": 0,       # 全部正确
        "all_wrong": 0,         # 全部错误
        "all_invalid": 0,       # 全部无效
        "paired_output": 0,     # 最终输出数
        "question_restored": 0,  # question 从原始数据集恢复的数量
        "question_not_found": 0, # 在原始数据集中找不到的数量
        "pass_rate_dist": [],
        "step_count_dist": Counter(),
    }
    
    for entry in raw_data:
        gt_answer = entry.get("gt_answer", "")
        image = entry.get("image", "")
        question = entry.get("question", "")
        outputs = entry.get("outputs", [])
        
        # 从原始数据集恢复完整 question (raw outputs 中被截断为 200 字符)
        if image_to_question and image in image_to_question:
            full_question = image_to_question[image]
            if len(full_question) > len(question):
                question = full_question
                stats["question_restored"] += 1
        elif image_to_question:
            stats["question_not_found"] += 1
        
        correct_cots = []
        wrong_cots = []
        invalid_count = 0
        
        for out in outputs:
            raw_text = out.get("raw_text", "")
            if not raw_text:
                invalid_count += 1
                continue
            
            # 超长直接丢弃
            if len(raw_text) > max_cot_chars * 2:
                invalid_count += 1
                continue
            
            # 解析推理链和答案
            cot_text, gen_answer = parse_raw_text(raw_text)
            
            if cot_text is None:
                invalid_count += 1
                continue
            
            # 验证格式
            is_valid, reason = validate_cot(
                cot_text, min_steps=min_steps, max_steps=max_steps,
                max_cot_chars=max_cot_chars
            )
            
            if not is_valid:
                invalid_count += 1
                continue
            
            cot_info = {
                "cot_text": cot_text,
                "gen_answer": gen_answer or "",
                "sample_idx": out.get("sample_idx", -1),
            }
            
            is_correct = check_answer_match(gen_answer or "", gt_answer)
            if is_correct:
                correct_cots.append(cot_info)
            else:
                wrong_cots.append(cot_info)
        
        valid_count = len(correct_cots) + len(wrong_cots)
        if valid_count == 0:
            stats["all_invalid"] += 1
            continue
        
        pass_rate = len(correct_cots) / valid_count
        stats["pass_rate_dist"].append(pass_rate)
        
        if len(correct_cots) > 0 and len(wrong_cots) > 0:
            # ✅ 理想情况: 有正确也有错误
            stats["has_both"] += 1
            
            best_wrong = select_best_wrong_cot(wrong_cots, correct_cots, max_cot_chars)
            best_correct = select_best_correct_cot(correct_cots, max_cot_chars)
            
            paired_item = {
                "image": image,
                "question": question,
                "answer": gt_answer,
                "think_chain": best_wrong["cot_text"],       # 新格式: Step N:
                "think_chain_status": "valid",
                "think_chain_source": "wrong_cot",
                "rejection_sampling_meta": {
                    "num_samples": len(outputs),
                    "num_valid": valid_count,
                    "num_correct": len(correct_cots),
                    "num_wrong": len(wrong_cots),
                    "num_invalid": invalid_count,
                    "pass_rate": round(pass_rate, 4),
                    "wrong_gen_answer": best_wrong["gen_answer"][:100],
                    "wrong_num_steps": count_steps(best_wrong["cot_text"]),
                },
            }
            
            if best_correct:
                paired_item["correct_cot"] = best_correct["cot_text"]  # 新格式
            
            training_data.append(paired_item)
            stats["paired_output"] += 1
            stats["step_count_dist"][count_steps(best_wrong["cot_text"])] += 1
        
        elif len(correct_cots) == valid_count:
            stats["all_correct"] += 1
        
        elif len(wrong_cots) == valid_count:
            # 全部错误: 仍然输出
            stats["all_wrong"] += 1
            
            best_wrong = select_best_wrong_cot(wrong_cots, [], max_cot_chars)
            
            paired_item = {
                "image": image,
                "question": question,
                "answer": gt_answer,
                "think_chain": best_wrong["cot_text"],
                "think_chain_status": "valid",
                "think_chain_source": "wrong_cot",
                "rejection_sampling_meta": {
                    "num_samples": len(outputs),
                    "num_valid": valid_count,
                    "num_correct": 0,
                    "num_wrong": len(wrong_cots),
                    "num_invalid": invalid_count,
                    "pass_rate": 0.0,
                    "wrong_gen_answer": best_wrong["gen_answer"][:100],
                    "wrong_num_steps": count_steps(best_wrong["cot_text"]),
                },
            }
            
            training_data.append(paired_item)
            stats["paired_output"] += 1
            stats["step_count_dist"][count_steps(best_wrong["cot_text"])] += 1
    
    return training_data, stats


def print_stats(stats: Dict):
    """打印统计信息"""
    print(f"\n{'='*70}")
    print(f"📊 转换统计")
    print(f"{'='*70}")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  有正确+错误 (理想): {stats['has_both']}")
    print(f"  全部正确 (跳过): {stats['all_correct']}")
    print(f"  全部错误 (保留): {stats['all_wrong']}")
    print(f"  全部无效 (跳过): {stats['all_invalid']}")
    print(f"  最终输出: {stats['paired_output']}")
    print(f"  question 已恢复: {stats.get('question_restored', 0)}")
    print(f"  question 未找到: {stats.get('question_not_found', 0)}")
    
    if stats['pass_rate_dist']:
        pr = stats['pass_rate_dist']
        print(f"\n  Pass Rate 分布:")
        print(f"    mean={sum(pr)/len(pr):.3f}, min={min(pr):.3f}, max={max(pr):.3f}")
        pr_bins = Counter()
        for p in pr:
            if p == 0:
                pr_bins['0.0'] += 1
            elif p <= 0.25:
                pr_bins['0.01-0.25'] += 1
            elif p <= 0.5:
                pr_bins['0.26-0.50'] += 1
            elif p <= 0.75:
                pr_bins['0.51-0.75'] += 1
            else:
                pr_bins['0.76-1.0'] += 1
        for k in sorted(pr_bins.keys()):
            print(f"    {k}: {pr_bins[k]}")
    
    if stats['step_count_dist']:
        print(f"\n  步骤数分布:")
        for k in sorted(stats['step_count_dist'].keys()):
            cnt = stats['step_count_dist'][k]
            print(f"    {k} steps: {cnt} ({cnt/max(stats['paired_output'],1)*100:.1f}%)")


def verify_output_format(training_data: List[Dict], num_samples: int = 5):
    """验证输出格式正确性"""
    print(f"\n{'='*70}")
    print(f"🔍 格式验证 (前 {num_samples} 个样本)")
    print(f"{'='*70}")
    
    all_ok = True
    for i, item in enumerate(training_data[:num_samples]):
        tc = item['think_chain']
        has_think_tag = '<think>' in tc
        has_step_delim = '</step>' in tc
        has_step_n = bool(re.search(r'Step\s+\d+\s*[:.]', tc, re.I))
        num_steps = count_steps(tc)
        ends_with_newline = tc.endswith('\n')
        
        ok = has_step_n and not has_think_tag and not has_step_delim and ends_with_newline
        status = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        
        print(f"\n  {status} 样本 {i}:")
        print(f"    steps={num_steps}, has_step_n={has_step_n}, "
              f"has_think_tag={has_think_tag}, has_step_delim={has_step_delim}")
        print(f"    source={item['think_chain_source']}, "
              f"pass_rate={item['rejection_sampling_meta']['pass_rate']}")
        print(f"    think_chain (first 200): {repr(tc[:200])}")
        
        # 检查 correct_cot 格式
        cc = item.get('correct_cot', '')
        if cc:
            cc_ok = bool(re.search(r'Step\s+\d+\s*[:.]', cc, re.I)) and '<think>' not in cc
            cc_status = "✅" if cc_ok else "❌"
            if not cc_ok:
                all_ok = False
            print(f"    {cc_status} correct_cot (first 200): {repr(cc[:200])}")
    
    print(f"\n  {'✅ 所有样本格式正确!' if all_ok else '❌ 存在格式问题!'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="从 raw outputs 转换为新格式训练数据")
    parser.add_argument("--input", type=str,
                        default="/mnt/cephszjt/user_juntianzhang/LatentDraft/data/rejection_sampling/rld_50k_rejection_sampled_raw_outputs.json",
                        help="输入 raw outputs JSON 路径")
    parser.add_argument("--output", type=str,
                        default="/mnt/cephszjt/user_juntianzhang/LatentDraft/data/rejection_sampling/rld_50k_rejection_sampled_new_format.json",
                        help="输出训练数据 JSON 路径")
    parser.add_argument("--min-steps", type=int, default=4, help="最少步骤数")
    parser.add_argument("--max-steps", type=int, default=10, help="最多步骤数")
    parser.add_argument("--max-cot-chars", type=int, default=1500, help="CoT 最大字符数")
    parser.add_argument("--orig-dataset", type=str,
                        default="/mnt/cephszjt/user_juntianzhang/LatentDraft/data/rld_mmathcot_filtered_checked.json",
                        help="原始数据集路径 (用于恢复被截断的 question)")
    args = parser.parse_args()
    
    # 加载原始数据集，构建 image -> question 映射
    image_to_question = None
    if args.orig_dataset:
        print(f"📂 加载原始数据集: {args.orig_dataset}")
        t0 = time.time()
        with open(args.orig_dataset, 'r') as f:
            orig_data = json.load(f)
        image_to_question = {}
        for item in orig_data:
            img = item.get("image", "")
            q = item.get("question", "")
            if img and q:
                image_to_question[img] = q
        print(f"  加载完成，耗时 {time.time() - t0:.1f}s")
        print(f"  构建 image->question 映射: {len(image_to_question)} 条")
        del orig_data  # 释放内存
    
    print(f"\n📂 加载 raw outputs: {args.input}")
    t0 = time.time()
    with open(args.input, 'r') as f:
        raw_data = json.load(f)
    print(f"  加载完成，耗时 {time.time() - t0:.1f}s")
    print(f"  总样本数: {len(raw_data)}")
    
    # 转换
    print(f"\n🔄 开始转换...")
    t0 = time.time()
    training_data, stats = convert_raw_outputs(
        raw_data,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        max_cot_chars=args.max_cot_chars,
        image_to_question=image_to_question,
    )
    print(f"  转换完成，耗时 {time.time() - t0:.1f}s")
    
    # 统计
    print_stats(stats)
    
    # 格式验证
    verify_output_format(training_data)
    
    # 保存
    print(f"\n💾 保存到: {args.output}")
    t0 = time.time()
    with open(args.output, 'w') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    import os
    file_size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"  保存完成，耗时 {time.time() - t0:.1f}s")
    print(f"  文件大小: {file_size_mb:.1f}MB")
    print(f"  样本数: {len(training_data)}")
    
    print(f"\n✅ 转换完成!")
    print(f"\n📝 使用方法:")
    print(f"  修改 configs/rld_train.yaml 中的 train_json 为:")
    print(f"    train_json: \"{args.output}\"")


if __name__ == '__main__':
    main()
