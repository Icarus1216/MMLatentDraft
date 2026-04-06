#!/usr/bin/env python3
"""
数据格式转换脚本: 将旧格式 think_chain 转换为新格式

旧格式 (rld_50k_selected.json):
  think_chain: "<think>\nstep1\n</step>\nstep2\n</step>\n</think>\nAnswer: D"
  
新格式 (训练代码期望):
  think_chain: "Step 1: step1\nStep 2: step2\nFinal Answer: D"
  think_chain_status: "valid"
  think_chain_source: "dataset_converted"

用法:
  python scripts/convert_data_format.py \
    --input data/rld_50k_selected.json \
    --output data/rld_50k_newformat.json
"""

import json
import re
import argparse
from pathlib import Path


THINK_START = "<think>"
THINK_END = "</think>"
STEP_DELIMITER = "</step>"


def convert_think_chain(think_chain: str, answer: str) -> str:
    """
    将旧格式 think_chain 转换为新格式
    
    旧格式: "<think>\nstep1\n</step>\nstep2\n</step>\n</think>\nAnswer: D"
    新格式: "Step 1: step1\nStep 2: step2\nFinal Answer: D"
    
    Args:
        think_chain: 旧格式的推理链
        answer: GT answer (用于 Final Answer 行)
    
    Returns:
        新格式的推理链文本
    """
    if not think_chain:
        return f"Step 1: Analyzing the image.\nFinal Answer: {answer}"
    
    text = think_chain
    
    # 1) 检测是否已经是新格式 (包含 "Step N:" 但不包含 <think>)
    if THINK_START not in text and re.search(r'Step\s+\d+\s*[:.]', text, re.IGNORECASE):
        # 已经是新格式，确保有 Final Answer
        if not re.search(r'Final\s+Answer\s*:', text, re.IGNORECASE):
            text = text.rstrip() + f"\nFinal Answer: {answer}"
        return text
    
    # 2) 旧格式: 剥离 <think>...</think> 包裹
    think_end_pos = text.rfind(THINK_END)
    after_think = ""
    if think_end_pos != -1:
        after_think = text[think_end_pos + len(THINK_END):].strip()
        text = text[:think_end_pos]
    if text.startswith(THINK_START):
        text = text[len(THINK_START):]
    text = text.strip('\n')
    
    # 3) 按 </step> 分割为步骤
    parts = text.split(STEP_DELIMITER)
    steps = [p.strip() for p in parts if p.strip()]
    
    if not steps:
        return f"Step 1: Analyzing the image.\nFinal Answer: {answer}"
    
    # 4) 重新编号为 "Step N: ..." 格式
    cot_parts = []
    for i, step_text in enumerate(steps, 1):
        # 清理步骤文本中可能残留的标签
        step_text = step_text.replace(THINK_START, '').replace(THINK_END, '').strip()
        if step_text:
            cot_parts.append(f"Step {i}: {step_text}")
    
    if not cot_parts:
        cot_parts = ["Step 1: Analyzing the image."]
    
    # 5) 拼接 Final Answer
    # 优先使用 answer 字段，如果 after_think 中有 "Answer: xxx" 也可以参考
    final_answer = answer.strip()
    
    result = "\n".join(cot_parts) + f"\nFinal Answer: {final_answer}"
    return result


def normalize_answer(answer: str) -> str:
    """
    Answer 标准化 (与 data.py 中的 _normalize_answer 保持一致)
    """
    if not answer:
        return answer
    
    text = answer.strip()
    
    # LaTeX 包裹去除
    text = re.sub(r'\\\((.+?)\\\)', r'\1', text)
    if text.startswith('$') and text.endswith('$') and len(text) > 2:
        text = text[1:-1].strip()
    text = re.sub(r'\\\[(.+?)\\\]', r'\1', text)
    
    # 常见 LaTeX 命令简化
    text = text.replace('\\pi', 'π')
    text = text.replace('\\sqrt', '√')
    text = text.replace('\\times', '×')
    text = text.replace('\\div', '÷')
    text = text.replace('\\pm', '±')
    text = text.replace('\\leq', '≤')
    text = text.replace('\\geq', '≥')
    text = text.replace('\\neq', '≠')
    text = text.replace('\\approx', '≈')
    text = text.replace('\\infty', '∞')
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = text.replace('\\,', ' ')
    
    # Markdown 格式清理
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # 选择题格式标准化
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in 'ABCDE':
        return stripped.upper()
    m = re.match(r'^([A-Ea-e])\s*[:.]\s*.+', stripped)
    if m:
        return m.group(1).upper()
    
    # 数值格式标准化
    m_num = re.match(r'^(-?[\d]+\.[\d]+)(.*)', text)
    if m_num:
        num_str = m_num.group(1)
        suffix = m_num.group(2)
        cleaned = num_str.rstrip('0').rstrip('.')
        text = cleaned + suffix
    
    # 多余空白清理
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def count_steps(think_chain: str) -> int:
    """统计新格式中的步骤数"""
    return len(re.findall(r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]', think_chain, re.IGNORECASE))


def main():
    parser = argparse.ArgumentParser(description="转换训练数据格式: 旧格式 → 新格式")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 文件路径")
    parser.add_argument("--normalize_answer", action="store_true", default=False,
                        help="是否对 answer 字段做标准化 (默认不做, 由 data.py 运行时处理)")
    args = parser.parse_args()
    
    print(f"📂 读取输入文件: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)
    print(f"   总样本数: {len(data)}")
    
    # 统计旧格式
    old_format_count = sum(1 for item in data if THINK_START in item.get('think_chain', ''))
    new_format_count = len(data) - old_format_count
    print(f"   旧格式 (<think>...</step>...</think>): {old_format_count}")
    print(f"   新格式 (Step N: ...): {new_format_count}")
    
    # 转换
    converted = []
    error_count = 0
    step_count_stats = {}
    
    for i, item in enumerate(data):
        try:
            think_chain = item.get('think_chain', '')
            answer = item.get('answer', '')
            
            # 转换 think_chain 为新格式
            new_chain = convert_think_chain(think_chain, answer)
            
            # 可选: 标准化 answer
            if args.normalize_answer:
                answer = normalize_answer(answer)
            
            # 构建新样本
            new_item = {
                "image": item.get("image", ""),
                "question": item.get("question", ""),
                "answer": answer,
                "think_chain": new_chain,
                "think_chain_status": "valid",
                "think_chain_source": item.get("think_chain_source", "dataset_converted"),
            }
            
            # 保留其他可能有用的字段
            for key in ["rejection_sampling_meta", "correct_cot", "original_answer"]:
                if key in item:
                    new_item[key] = item[key]
            
            # 统计步骤数
            n_steps = count_steps(new_chain)
            step_count_stats[n_steps] = step_count_stats.get(n_steps, 0) + 1
            
            converted.append(new_item)
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"   ⚠️ 样本 {i} 转换失败: {e}")
            # 保留原样本
            converted.append(item)
    
    print(f"\n✅ 转换完成:")
    print(f"   成功: {len(converted) - error_count}")
    print(f"   失败: {error_count}")
    
    # 步骤数分布
    print(f"\n📊 步骤数分布:")
    for n_steps in sorted(step_count_stats.keys()):
        count = step_count_stats[n_steps]
        bar = "█" * min(count // 500, 50)
        print(f"   {n_steps:2d} steps: {count:6d} {bar}")
    
    # 验证: 展示前3个转换后的样本
    print(f"\n🔍 转换后样本预览:")
    for i in range(min(3, len(converted))):
        item = converted[i]
        print(f"\n--- 样本 {i} ---")
        print(f"  answer: {item['answer']}")
        print(f"  think_chain_source: {item['think_chain_source']}")
        chain = item['think_chain']
        if len(chain) > 300:
            print(f"  think_chain: {chain[:300]}...")
        else:
            print(f"  think_chain: {chain}")
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存到: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    # 文件大小
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   文件大小: {size_mb:.1f} MB")
    print(f"\n🎉 完成! 可以将 configs/rld_train.yaml 中的 train_json 改为: {args.output}")


if __name__ == "__main__":
    main()
