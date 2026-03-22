#!/usr/bin/env python3
"""
RLD 推理脚本

使用方法:
    # 单张图像问答
    python scripts/inference.py \
        --model_path /path/to/qwen3-vl \
        --rld_checkpoint ./outputs/rld_train/model \
        --image /path/to/image.jpg \
        --question "What is shown in this image?"
    
    # 批量推理
    python scripts/inference.py \
        --model_path /path/to/qwen3-vl \
        --rld_checkpoint ./outputs/rld_train/model \
        --batch_file /path/to/queries.json \
        --output_file ./results.json
"""

import argparse
import json
import os
import sys
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


class RLDInference:
    """RLD 推理引擎"""

    def __init__(
        self,
        model_path: str,
        rld_checkpoint: str,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        # 模型参数
        hidden_size: int = 4096,
        # RLD 参数
        d_z: int = 512,
        num_evidence_slots: int = 16,
        num_draft_slots: int = 16,
        num_trace_slots: int = 16,
        total_layers: int = 36,
        # System prompt 控制
        use_system_prompt: bool = True,
        system_prompt: str = None,
    ):
        self.device = device
        self.use_system_prompt = use_system_prompt
        self.system_prompt = system_prompt or RLD_SYSTEM_PROMPT
        print(f"[RLD Inference] 初始化...")
        print(f"  - Model: {model_path}")
        print(f"  - RLD Checkpoint: {rld_checkpoint}")

        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # 加载模型
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

        # 加载 RLD controller 权重
        if os.path.exists(rld_checkpoint):
            self.model.load_pretrained(rld_checkpoint)

        self.model.to(device)
        self.model.eval()
        print("[RLD Inference] ✅ 初始化完成")

    def infer(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        单张图像推理
        
        Args:
            image_path: 图像路径
            question: 问题文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样参数
            do_sample: 是否采样
        
        Returns:
            answer: 生成的回答文本
        """
        # 构造消息 (注入与训练一致的 system prompt)
        messages = []
        if self.use_system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ]})

        # 处理输入
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

        # 生成
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

        # 解码 (只取生成部分)
        prompt_len = inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0, prompt_len:]
        answer = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        # 清理 </step> delimiter
        answer = answer.replace("</step>", "").strip()

        return answer

    def batch_infer(
        self,
        queries: list,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> list:
        """
        批量推理
        
        Args:
            queries: [{"image": path, "question": text}, ...]
        
        Returns:
            results: [{"image": path, "question": text, "answer": text}, ...]
        """
        results = []
        for i, q in enumerate(queries):
            print(f"[{i+1}/{len(queries)}] 处理: {q['question'][:50]}...")
            answer = self.infer(
                image_path=q['image'],
                question=q['question'],
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            results.append({
                "image": q['image'],
                "question": q['question'],
                "answer": answer,
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="RLD Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen3-VL 模型路径")
    parser.add_argument("--rld_checkpoint", type=str, required=True, help="RLD controller 权重路径")
    parser.add_argument("--image", type=str, default=None, help="图像路径")
    parser.add_argument("--question", type=str, default=None, help="问题文本")
    parser.add_argument("--batch_file", type=str, default=None, help="批量查询 JSON 文件")
    parser.add_argument("--output_file", type=str, default=None, help="结果输出文件")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    engine = RLDInference(
        model_path=args.model_path,
        rld_checkpoint=args.rld_checkpoint,
        device=args.device,
    )

    if args.batch_file:
        # 批量推理
        with open(args.batch_file, 'r') as f:
            queries = json.load(f)
        results = engine.batch_infer(
            queries, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ 结果已保存到 {args.output_file}")
        else:
            for r in results:
                print(f"\nQ: {r['question']}")
                print(f"A: {r['answer']}")
    elif args.image and args.question:
        # 单张推理
        answer = engine.infer(
            image_path=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {answer}")
    else:
        parser.print_help()
        print("\n请提供 --image + --question 或 --batch_file")


if __name__ == "__main__":
    main()
