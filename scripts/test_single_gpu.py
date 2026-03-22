#!/usr/bin/env python3
"""
单卡快速验证脚本 — 无需 DeepSpeed，2-3 分钟内发现 CUDA 错误

用法:
    # 在 Docker 内, conda activate colamem 后:
    python scripts/test_single_gpu.py --config configs/rld_train.yaml

    # 指定 GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_single_gpu.py --config configs/rld_train.yaml

    # 开启 CUDA 同步调试 (更精确的错误定位，但更慢):
    CUDA_LAUNCH_BLOCKING=1 python scripts/test_single_gpu.py --config configs/rld_train.yaml

功能:
    1. 加载模型和真实数据 (1 个样本)
    2. 执行 forward + backward
    3. 打印每个段的详细信息 (cache 形状, position 范围, attention_mask 维度)
    4. 检查梯度是否正常 (无 NaN/Inf)
    5. 如果有错误，立即报告具体位置
"""

import os
import sys
import argparse
import yaml
import torch
import traceback
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="单卡快速验证")
    parser.add_argument("--config", type=str, default="configs/rld_train.yaml")
    parser.add_argument("--sample_idx", type=int, default=0, help="测试第几个样本")
    parser.add_argument("--num_samples", type=int, default=2, help="测试几个样本")
    parser.add_argument("--check_backward", action="store_true", default=True, help="是否测试 backward")
    parser.add_argument("--verbose", action="store_true", default=True, help="打印详细信息")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    data_config = config['data']
    rld_config = config.get('rld', {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # ====== 1. 加载模型 ======
    print("=" * 60)
    print("[1/4] 加载模型...")
    print("=" * 60)

    from transformers import AutoProcessor, set_seed
    from rld.model import RLDModel
    from rld.data import RLDDataset, RLDCollator

    set_seed(42)
    processor = AutoProcessor.from_pretrained(model_config['model_path'])

    model = RLDModel(
        model_path=model_config['model_path'],
        hidden_size=model_config.get('hidden_size', 2048),
        d_z=rld_config.get('d_z', 512),
        num_evidence_slots=rld_config.get('num_evidence_slots', 16),
        num_draft_slots=rld_config.get('num_draft_slots', 16),
        num_trace_slots=rld_config.get('num_trace_slots', 16),
        total_layers=model_config.get('total_layers', 48),
        torch_dtype=getattr(torch, model_config.get('torch_dtype', 'bfloat16')),
        attn_implementation=model_config.get('attn_implementation', 'flash_attention_2'),
        lambda_div=rld_config.get('lambda_div', 0.01),
    )
    model.set_processor(processor)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   可训练参数: {trainable / 1e6:.2f}M / 总参数: {total / 1e9:.2f}B")
    print()

    # ====== 2. 加载数据 ======
    print("=" * 60)
    print("[2/4] 加载数据...")
    print("=" * 60)

    # 尝试加载 debug 数据 (更小)
    debug_json = "data/rld_mmathcot_filtered_debug.json"
    train_json = data_config['train_json']
    json_path = debug_json if os.path.exists(debug_json) else train_json
    print(f"   数据文件: {json_path}")

    dataset = RLDDataset(
        json_path=json_path,
        processor=processor,
        auto_split_steps=data_config.get('auto_split_steps', True),
        use_system_prompt=data_config.get('use_system_prompt', True),
        system_prompt=data_config.get('system_prompt', None),
        max_seq_len=data_config.get('max_seq_len', 4096),
        skip_image_check=data_config.get('skip_image_check', False),
    )

    collator = RLDCollator(
        processor=processor,
        step_delimiter_ids=model.step_delimiter_ids,
    )

    print(f"   数据集大小: {len(dataset)}")
    print()

    # ====== 3. 测试 forward + backward ======
    print("=" * 60)
    print("[3/4] 测试 forward + backward...")
    print("=" * 60)
    print()

    num_test = min(args.num_samples, len(dataset))
    all_passed = True

    for sample_i in range(args.sample_idx, args.sample_idx + num_test):
        print(f"--- 样本 {sample_i} ---")

        try:
            # 获取单个样本并组 batch (batch_size=1)
            item = dataset[sample_i]
            batch = collator([item])

            # 移到 GPU
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v

            seq_len = inputs['input_ids'].shape[1]
            prompt_lens = inputs.get('prompt_lens', [0])
            step_boundaries = inputs.get('step_boundaries', [[]])

            print(f"   序列长度: {seq_len}")
            print(f"   prompt 长度: {prompt_lens}")
            print(f"   step 边界: {step_boundaries}")

            # labels 中有效 token 数
            if inputs.get('labels') is not None:
                valid_tokens = (inputs['labels'] != -100).sum().item()
                print(f"   有效 label token 数: {valid_tokens}")

            # 图像信息
            if inputs.get('pixel_values') is not None:
                print(f"   pixel_values shape: {inputs['pixel_values'].shape}")
            if inputs.get('image_grid_thw') is not None:
                print(f"   image_grid_thw: {inputs['image_grid_thw']}")

            print()

            # ---- Forward ----
            print(f"   🔄 Forward...")
            model.train()
            torch.cuda.synchronize()

            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs.get('labels'),
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                step_boundaries=inputs.get('step_boundaries'),
                prompt_lens=inputs.get('prompt_lens'),
            )

            torch.cuda.synchronize()
            loss = outputs['loss']
            main_loss = outputs.get('main_loss', torch.tensor(0.0))
            div_loss = outputs.get('div_loss', torch.tensor(0.0))

            print(f"   ✅ Forward 成功!")
            print(f"      loss = {loss.item():.6f}")
            print(f"      main_loss = {main_loss.item():.6f}")
            print(f"      div_loss = {div_loss.item():.6f}")

            # 检查 loss 是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   ❌ 警告: loss 为 NaN 或 Inf！这会导致 backward 异常。")
                all_passed = False

            # ---- Backward ----
            if args.check_backward and loss.requires_grad:
                print(f"   🔄 Backward...")
                loss.backward()
                torch.cuda.synchronize()
                print(f"   ✅ Backward 成功!")

                # 检查梯度
                has_nan_grad = False
                has_zero_grad = True
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"   ❌ 参数 {name} 梯度包含 NaN!")
                            has_nan_grad = True
                        if torch.isinf(param.grad).any():
                            print(f"   ❌ 参数 {name} 梯度包含 Inf!")
                            has_nan_grad = True
                        if param.grad.abs().max() > 0:
                            has_zero_grad = False

                if has_nan_grad:
                    print(f"   ❌ 存在 NaN/Inf 梯度 — 这会导致分布式训练 hang!")
                    all_passed = False
                elif has_zero_grad:
                    print(f"   ⚠️  所有梯度为零 — Controller 可能没有收到梯度信号")
                else:
                    print(f"   ✅ 梯度检查通过! (无 NaN/Inf，梯度非零)")

                # 清零梯度
                model.zero_grad()

            # 显存使用
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
                print(f"   📊 峰值显存: {mem_alloc:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

            print()

        except Exception as e:
            print(f"   ❌ 错误: {e}")
            traceback.print_exc()
            all_passed = False
            print()

    # ====== 4. 总结 ======
    print("=" * 60)
    print("[4/4] 总结")
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过! forward + backward 正常，可以启动分布式训练。")
    else:
        print("❌ 存在错误! 请先修复上述问题再启动分布式训练。")
        print("   提示: 使用 CUDA_LAUNCH_BLOCKING=1 重新运行可以获得更精确的错误位置。")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
