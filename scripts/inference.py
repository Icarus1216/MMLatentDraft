#!/usr/bin/env python3
"""
NLD 推理脚本 (Native Latent Draft)

使用方法:
    # 单张图像问答
    python scripts/inference.py \
        --model_path /path/to/qwen3-vl \
        --nld_checkpoint ./outputs/nld_train/model \
        --image /path/to/image.jpg \
        --question "What is shown in this image?"
    
    # 批量推理
    python scripts/inference.py \
        --model_path /path/to/qwen3-vl \
        --nld_checkpoint ./outputs/nld_train/model \
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
from rld.model_v2 import NLDModel
from rld.data import NLD_SYSTEM_PROMPT, LATENT_TOKEN

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[NLD Inference] ⚠️ qwen_vl_utils 未安装")
    process_vision_info = None


class NLDInference:
    """NLD 推理引擎"""

    def __init__(
        self,
        model_path: str,
        nld_checkpoint: str,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        # 模型参数
        hidden_size: int = 4096,
        total_layers: int = 36,
        # NLD 参数
        max_think_steps: int = 6,
        # System prompt 控制
        use_system_prompt: bool = True,
        system_prompt: str = None,
    ):
        self.device = device
        self.use_system_prompt = use_system_prompt
        self.system_prompt = system_prompt or NLD_SYSTEM_PROMPT
        print(f"[NLD Inference] 初始化...")
        print(f"  - Model: {model_path}")
        print(f"  - NLD Checkpoint: {nld_checkpoint}")

        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # 注册 <|latent|> 特殊 token
        num_added = self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [LATENT_TOKEN]}
        )
        latent_token_id = self.processor.tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        print(f"  - <|latent|> token id: {latent_token_id} (新增 {num_added} 个)")

        # 加载模型
        self.model = NLDModel(
            model_path=model_path,
            hidden_size=hidden_size,
            total_layers=total_layers,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            max_think_steps=max_think_steps,
        )
        
        # 调整 embedding 大小
        self.model.base_model.resize_token_embeddings(len(self.processor.tokenizer))
        self.model.set_processor(self.processor)

        # 加载 NLD 权重
        if os.path.exists(nld_checkpoint):
            self.model.load_pretrained(nld_checkpoint)

        self.model.to(device)
        self.model.eval()
        print("[NLD Inference] ✅ 初始化完成")

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
        # 构造消息
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
        full_output = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=False
        )

        # 后处理: 用 "Final Answer:" 分割 reasoning 和 answer
        # 生成内容格式: "{reasoning}<|latent|>{conclusion}\nFinal Answer:\n{answer}<|im_end|>"
        # 清理 <|latent|> token (隐空间思考标记，不显示)
        full_output = full_output.replace(LATENT_TOKEN, " [latent thinking] ")
        
        if "Final Answer:" in full_output:
            fa_idx = full_output.rfind("Final Answer:")
            reasoning_part = full_output[:fa_idx].strip()
            answer_part = full_output[fa_idx + len("Final Answer:"):].strip()
            answer = answer_part.replace("<|im_end|>", "").strip()
        else:
            # 模型未输出 "Final Answer:"，取全部输出作为答案
            answer = full_output.replace("<|im_end|>", "").strip()

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
    parser = argparse.ArgumentParser(description="NLD Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen3-VL 模型路径")
    parser.add_argument("--nld_checkpoint", type=str, required=True, help="NLD 模型权重路径")
    parser.add_argument("--image", type=str, default=None, help="图像路径")
    parser.add_argument("--question", type=str, default=None, help="问题文本")
    parser.add_argument("--batch_file", type=str, default=None, help="批量查询 JSON 文件")
    parser.add_argument("--output_file", type=str, default=None, help="结果输出文件")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    engine = NLDInference(
        model_path=args.model_path,
        nld_checkpoint=args.nld_checkpoint,
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
