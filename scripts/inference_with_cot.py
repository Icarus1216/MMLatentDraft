#!/usr/bin/env python3
"""
RLD 推理脚本 (带完整 CoT 输出)

使用方法:
    # 批量推理 (输出完整 CoT)
    python scripts/inference_with_cot.py \
        --model_path /path/to/qwen3-vl \
        --rld_checkpoint ./outputs/rld_top8_rerun_768_stage1/model \
        --test_file test_unseen_50.json \
        --output_file test_results_cot.json \
        --max_new_tokens 1024
"""

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor
from rld.model import RLDModel
from rld.data import RLD_SYSTEM_PROMPT

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[RLD Inference] ⚠️ qwen_vl_utils 未安装")
    process_vision_info = None


class RLDInferenceWithCoT:
    """RLD 推理引擎 (输出完整 CoT)"""

    def __init__(
        self,
        model_path: str,
        rld_checkpoint: str,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        hidden_size: int = 4096,
        d_z: int = 768,
        num_evidence_slots: int = 16,
        num_draft_slots: int = 16,
        num_trace_slots: int = 16,
        total_layers: int = 36,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list = None,
        lora_layers_from: int = None,
    ):
        self.device = device
        self.system_prompt = RLD_SYSTEM_PROMPT
        print(f"[RLD Inference] 初始化...")
        print(f"  - Model: {model_path}")
        print(f"  - RLD Checkpoint: {rld_checkpoint}")
        if use_lora:
            _modules_str = lora_target_modules or ['q_proj', 'v_proj']
            _layers_str = f'L{lora_layers_from}~L35' if lora_layers_from else 'L18~L35'
            print(f"  - LoRA: r={lora_r}, alpha={lora_alpha}, modules={_modules_str}, layers={_layers_str}")

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.model = RLDModel(
            model_path=model_path,
            hidden_size=hidden_size,
            d_z=d_z,
            num_evidence_slots=num_evidence_slots,
            num_draft_slots=num_draft_slots,
            num_trace_slots=num_trace_slots,
            total_layers=total_layers,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.model.set_processor(self.processor)

        # Stage 2: 先配置 LoRA 结构，再加载权重
        if use_lora:
            # 解析 LoRA 层范围
            _lora_layers = None
            if lora_layers_from is not None:
                _lora_layers = list(range(lora_layers_from, total_layers))
            
            self.model.setup_lora(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,  # 推理时不需要 dropout
                target_modules=lora_target_modules,
                lora_layers=_lora_layers,
            )

        if os.path.exists(rld_checkpoint):
            self.model.load_pretrained(rld_checkpoint)

        self.model.to(device)
        self.model.eval()
        print("[RLD Inference] ✅ 初始化完成")

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
        """
        单张图像推理，返回完整 CoT + 最终答案
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
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
            pixel_values=inputs['pixel_values'],
            image_grid_thw=inputs['image_grid_thw'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
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

        # 解析 CoT 和 Final Answer
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

        # 清理 CoT 中的特殊标记
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
    """评估推理结果"""
    correct = 0
    total = len(results)
    details = []

    for r in results:
        gt = r['gt_answer'].strip().upper()
        pred_raw = (r.get('answer') or '').strip()

        # 尝试多种匹配方式
        pred = ''
        # 1. 直接取第一个大写字母
        for ch in pred_raw:
            if ch.upper() in 'ABCDEFGH':
                pred = ch.upper()
                break
        # 2. 如果答案是数字
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
    parser = argparse.ArgumentParser(description="RLD Inference with Full CoT Output")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--rld_checkpoint", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True, help="测试集 JSON 文件")
    parser.add_argument("--output_file", type=str, default="test_results_cot.json")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--no_sample", action="store_true", help="确定性解码 (greedy)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_lora", action="store_true", help="启用 LoRA (Stage 2 模型)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_target_modules", type=str, default=None,
                        help="LoRA 目标模块 (逗号分隔, 如 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')")
    parser.add_argument("--lora_layers_from", type=int, default=None,
                        help="LoRA 起始层 (如 9 表示 L9~L35)")
    args = parser.parse_args()

    do_sample = not args.no_sample

    # 加载测试数据
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    print(f"\n📋 测试集: {len(test_data)} 条样本")

    # 初始化推理引擎
    # 解析 LoRA target modules
    _lora_target_modules = None
    if args.lora_target_modules:
        _lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

    engine = RLDInferenceWithCoT(
        model_path=args.model_path,
        rld_checkpoint=args.rld_checkpoint,
        device=args.device,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=_lora_target_modules,
        lora_layers_from=args.lora_layers_from,
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
    print("📊 评估结果:")
    metrics = evaluate_results(results)
    print(f"  总样本: {metrics['total']}")
    print(f"  正确数: {metrics['correct']}")
    print(f"  准确率: {metrics['accuracy_pct']}")

    # 逐条结果
    print(f"\n  逐条结果:")
    for i, r in enumerate(results):
        status = '✅' if r.get('correct') else '❌'
        pred = r.get('pred_letter', '?')
        gt = r.get('gt_answer', '?')
        print(f"    {status} [{i+1}] GT={gt} Pred={pred}")

    # 保存
    output = {
        'metrics': metrics,
        'results': results,
    }
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
