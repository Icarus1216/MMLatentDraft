#!/usr/bin/env python3
"""
Qwen3-VL 基座模型推理脚本 (不加载 RLD, 作为对比基线)

使用与 RLD 推理完全相同的 system prompt、数据和评估方式,
唯一区别是不加载 RLD Controller / ReadoutAdapter。

使用方法:
    python scripts/inference_baseline.py \
        --model_path /path/to/qwen3-vl \
        --test_file test_unseen_50.json \
        --output_file test_results_baseline.json \
        --max_new_tokens 1024
"""

import argparse
import json
import os
import sys
import time
import re
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[Baseline] ⚠️ qwen_vl_utils 未安装")
    process_vision_info = None

# 与 RLD 训练/推理完全相同的 system prompt
SYSTEM_PROMPT = """You are a visual reasoning assistant. Think step by step. Be concise.

Rules:
1. Use numbered steps: "Step 1:", "Step 2:", etc.
2. Each step should focus on one reasoning action (observe, calculate, deduce, etc.).
3. Do NOT add introductions, greetings, summaries, or filler words.
4. End with "Final Answer:" on its own line. This is MANDATORY.

Example:
Step 1: The triangle has base 6 and height 8.
Step 2: Area = 0.5 × 6 × 8 = 24.
Final Answer: 24 square units."""


class BaselineInference:
    """Qwen3-VL 基座推理引擎 (无 RLD)"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
    ):
        self.device = device
        print(f"[Baseline] 初始化...")
        print(f"  - Model: {model_path}")

        self.processor = AutoProcessor.from_pretrained(model_path)

        print(f"[Baseline] 加载 Qwen3-VL...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.model.to(device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Baseline] 模型参数: {total_params / 1e9:.2f}B")
        print(f"[Baseline] ✅ 初始化完成")

    @torch.no_grad()
    def infer(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> dict:
        """单张图像推理"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if process_vision_info:
            image_inputs, _ = process_vision_info(messages)
        else:
            image_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
        ).to(self.device)

        t0 = time.time()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        elapsed = time.time() - t0

        prompt_len = inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0, prompt_len:]
        num_generated = generated_tokens.shape[0]
        full_output = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=False
        )

        # 解析 CoT 和 Final Answer (与 RLD 推理脚本完全一致)
        cot = None
        answer = None

        if "Final Answer:" in full_output:
            fa_idx = full_output.rfind("Final Answer:")
            cot = full_output[:fa_idx].strip()
            answer_part = full_output[fa_idx + len("Final Answer:"):].strip()
            answer = answer_part.replace("<|im_end|>", "").strip()
        elif "</think>" in full_output:
            think_part, answer_part = full_output.split("</think>", 1)
            cot = think_part.replace("<think>", "").strip()
            answer = answer_part.replace("<|im_end|>", "").strip()
        else:
            answer = full_output.replace("<|im_end|>", "").strip()

        if cot:
            cot = cot.replace("</step>", "").replace("<think>", "").replace("</think>", "").strip()

        return {
            'full_output': full_output.replace("<|im_end|>", "").strip(),
            'cot': cot,
            'answer': answer,
            'num_tokens': num_generated,
            'time_s': round(elapsed, 2),
        }


def evaluate_results(results: list) -> dict:
    """评估推理结果 (与 RLD 推理脚本完全一致)"""
    correct = 0
    total = len(results)

    for r in results:
        gt = r['gt_answer'].strip().upper()
        pred_raw = (r.get('answer') or '').strip()

        pred = ''
        for ch in pred_raw:
            if ch.upper() in 'ABCDEFGH':
                pred = ch.upper()
                break
        if not pred and pred_raw:
            pred = pred_raw.split()[0] if pred_raw.split() else pred_raw

        match = (pred == gt)
        correct += int(match)
        r['pred_letter'] = pred
        r['correct'] = match

    accuracy = correct / total if total > 0 else 0
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_pct': f"{accuracy * 100:.1f}%",
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Baseline Inference (No RLD)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True, help="测试集 JSON 文件")
    parser.add_argument("--output_file", type=str, default="test_results_baseline.json")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--no_sample", action="store_true", help="确定性解码 (greedy)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    do_sample = not args.no_sample

    # 加载测试数据
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    print(f"\n📋 测试集: {len(test_data)} 条样本")
    print(f"📌 模式: Qwen3-VL 基座 (无 RLD)")
    print(f"📌 解码: {'Greedy' if not do_sample else f'Sampling (T={args.temperature})'}")

    # 初始化推理引擎
    engine = BaselineInference(
        model_path=args.model_path,
        device=args.device,
    )

    # 逐条推理
    results = []
    for i, item in enumerate(test_data):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(test_data)}] 图片: {item.get('image_url', os.path.basename(item['image']))}")
        print(f"  问题: {item['question'][:100]}...")
        print(f"  GT答案: {item['answer']}")

        try:
            result = engine.infer(
                image_path=item['image'],
                question=item['question'],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=do_sample,
            )
            print(f"\n  🔍 完整 CoT:")
            if result['cot']:
                for line in result['cot'].split('\n'):
                    print(f"    {line}")
            else:
                print(f"    (无 CoT)")
            print(f"\n  📝 最终答案: {result['answer']}")
            print(f"  ⏱️ 耗时: {result['time_s']}s, 生成 {result['num_tokens']} tokens")

            results.append({
                'image': item['image'],
                'image_url': item.get('image_url', ''),
                'question': item['question'],
                'gt_answer': item['answer'],
                'answer': result['answer'],
                'cot': result['cot'],
                'full_output': result['full_output'],
                'num_tokens': result['num_tokens'],
                'time_s': result['time_s'],
            })
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'image': item['image'],
                'question': item['question'],
                'gt_answer': item['answer'],
                'answer': None,
                'cot': None,
                'full_output': None,
                'error': str(e),
            })

    # 评估
    print(f"\n{'='*70}")
    print("📊 评估结果 (Qwen3-VL 基座, 无 RLD):")
    metrics = evaluate_results(results)
    print(f"  总样本: {metrics['total']}")
    print(f"  正确数: {metrics['correct']}")
    print(f"  准确率: {metrics['accuracy_pct']}")

    print(f"\n  逐条结果:")
    for i, r in enumerate(results):
        status = '✅' if r.get('correct') else '❌'
        pred = r.get('pred_letter', '?')
        gt = r.get('gt_answer', '?')
        print(f"    {status} [{i+1}] GT={gt} Pred={pred}")

    # 保存
    output = {
        'metrics': metrics,
        'config': {
            'model': 'Qwen3-VL-8B-Instruct (baseline, no RLD)',
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'do_sample': do_sample,
        },
        'results': results,
    }
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
