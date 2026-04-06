#!/usr/bin/env python3
"""
RLD 训练脚本

使用方法:
    # 单卡
    python scripts/train.py --config configs/rld_train.yaml
    
    # 多卡 (PyTorch DDP, 默认推荐)
    torchrun --nproc_per_node=8 scripts/train.py --config configs/rld_train.yaml

    # 多卡 (DeepSpeed ZeRO-2, 可选)
    deepspeed --num_gpus=8 scripts/train.py --config configs/rld_train.yaml --use_deepspeed
"""

import os
import sys
import argparse
import yaml
import torch
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    TrainingArguments,
    AutoProcessor,
    set_seed,
)

from rld.model import RLDModel
from rld.data import RLDDataset, RLDCollator
from rld.trainer import RLDTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="RLD Training Script")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖输出目录")
    parser.add_argument("--learning_rate", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--use_deepspeed", action="store_true", default=False,
                        help="使用 DeepSpeed ZeRO-2 (默认使用 PyTorch DDP)")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local rank")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    return local_rank in (-1, 0)


def main():
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    rld_config = config.get('rld', {})
    seed = config.get('seed', 42)

    set_seed(seed)
    is_main = _is_main_process()

    # ====== 判断分布式后端 ======
    # 优先级: 命令行 --use_deepspeed > yaml 中的 deepspeed_config
    ds_config_path = None
    if args.use_deepspeed:
        # 命令行显式指定使用 DeepSpeed
        ds_config_path = training_config.get('deepspeed_config', 'configs/ds_config_zero2.json')
        if not os.path.exists(ds_config_path):
            # 尝试备用路径
            ds_config_path = os.path.join(os.path.dirname(args.config), 'ds_config_zero2.json')
    elif training_config.get('deepspeed_config') and not training_config['deepspeed_config'].startswith('#'):
        # yaml 中未注释的 deepspeed_config (向后兼容)
        _yaml_ds = training_config['deepspeed_config']
        if os.path.exists(_yaml_ds):
            ds_config_path = _yaml_ds

    use_deepspeed = ds_config_path is not None and os.path.exists(ds_config_path)

    if is_main:
        if use_deepspeed:
            print(f"📦 分布式后端: DeepSpeed ZeRO-2 ({ds_config_path})")
        else:
            print(f"📦 分布式后端: PyTorch DDP (torchrun)")

    # ====== 1. 加载 Processor ======
    if is_main:
        print("\n[1/5] 加载 Processor...")
    processor = AutoProcessor.from_pretrained(model_config['model_path'])

    # 限制图片分辨率: 避免 ViT 编码器处理超高分辨率图片 (默认 max_pixels=12845056 会导致
    # vision encoder 单张图耗时 30~40s, 降到 ~1003520 后仅需 1~2s)
    _min_pixels = data_config.get('min_pixels', 3136)
    _max_pixels = data_config.get('max_pixels', 1003520)
    processor.image_processor.min_pixels = _min_pixels
    processor.image_processor.max_pixels = _max_pixels
    if is_main:
        print(f"  图片分辨率限制: min_pixels={_min_pixels}, max_pixels={_max_pixels}")

    # ====== 2. 创建数据集 ======
    if is_main:
        print("\n[2/5] 创建数据集...")
    train_dataset = RLDDataset(
        json_path=data_config['train_json'],
        processor=processor,
        auto_split_steps=data_config.get('auto_split_steps', True),
        use_system_prompt=data_config.get('use_system_prompt', True),
        system_prompt=data_config.get('system_prompt', None),
        max_seq_len=data_config.get('max_seq_len', 4096),
        skip_image_check=data_config.get('skip_image_check', False),
        hard_sample_max_ratio=data_config.get('hard_sample_max_ratio', 0.22),
    )
    if is_main:
        print(f"训练样本数: {len(train_dataset)}")

    eval_dataset = None
    if data_config.get('eval_json'):
        eval_dataset = RLDDataset(
            json_path=data_config['eval_json'],
            processor=processor,
            auto_split_steps=data_config.get('auto_split_steps', True),
            use_system_prompt=data_config.get('use_system_prompt', True),
            system_prompt=data_config.get('system_prompt', None),
            max_seq_len=data_config.get('max_seq_len', 4096),
            skip_image_check=data_config.get('skip_image_check', False),
            hard_sample_max_ratio=1.0,  # eval 不下采样
        )
        if is_main:
            print(f"评估样本数: {len(eval_dataset)}")

    # ====== 3. 创建模型 ======
    if is_main:
        print("\n[3/5] 创建 RLD 模型...")

    # 检查是否使用 ZeRO-3 (用于打印参数统计等)
    use_zero3 = False
    if use_deepspeed and ds_config_path:
        with open(ds_config_path) as f:
            ds_cfg = json.load(f)
        zero_stage = ds_cfg.get('zero_optimization', {}).get('stage', 0)
        use_zero3 = (zero_stage == 3)

    model_kwargs = dict(
        model_path=model_config['model_path'],
        hidden_size=model_config.get('hidden_size', 4096),
        d_z=rld_config.get('d_z', 768),
        num_evidence_slots=rld_config.get('num_evidence_slots', 16),
        num_draft_slots=rld_config.get('num_draft_slots', 16),
        num_trace_slots=rld_config.get('num_trace_slots', 16),
        total_layers=model_config.get('total_layers', 36),
        torch_dtype=getattr(torch, model_config.get('torch_dtype', 'bfloat16')),
        attn_implementation=model_config.get('attn_implementation', 'flash_attention_2'),
        lambda_div=rld_config.get('lambda_div', 0.01),
        max_scale=rld_config.get('max_scale', 0.3),
        selective_injection=rld_config.get('selective_injection', True),
        use_trace_updater=rld_config.get('use_trace_updater', True),
        use_bidirectional_reflection=rld_config.get('use_bidirectional_reflection', False),
    )

    # 注意: 不使用 deepspeed.zero.Init() 手动包裹模型创建。
    # HuggingFace Trainer 在 ZeRO-3 下会自动处理参数分片。
    # from_pretrained 内部会检测 DeepSpeed 环境并正确加载模型。
    model = RLDModel(**model_kwargs)
    model.set_processor(processor)

    # ====== Stage 2: 配置 LoRA (必须在 load_pretrained 之前, 避免结构不兼容) ======
    lora_config = rld_config.get('lora', None)
    skip_lora_loading = rld_config.get('skip_lora_loading', False)
    if lora_config and lora_config.get('enabled', False):
        # 解析 LoRA 层范围: 支持 layers (精确列表) 或 layers_from (起始层)
        lora_layers = lora_config.get('layers', None)
        if lora_layers is None and lora_config.get('layers_from') is not None:
            # layers_from: 从指定层到最后一层 (例如 layers_from=9 → [9,10,...,35])
            layers_from = lora_config['layers_from']
            total_layers = model_config.get('total_layers', 36)
            lora_layers = list(range(layers_from, total_layers))
            if is_main:
                print(f"[RLD Stage 2] LoRA 层范围: L{layers_from}~L{total_layers-1} ({len(lora_layers)} 层)")
        
        model.setup_lora(
            lora_r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.05),
            target_modules=lora_config.get('target_modules', None),
            lora_layers=lora_layers,
        )

    # ====== Stage 2: 从 Stage 1 checkpoint 加载 Controller + PrefixKVProjector ======
    resume_rld = rld_config.get('resume_from_rld_checkpoint', None)
    if resume_rld and os.path.exists(resume_rld):
        model.load_pretrained(resume_rld, skip_lora=skip_lora_loading)
        if is_main:
            if skip_lora_loading:
                print(f"[RLD Stage 2] ✅ 从 Stage 1 加载 Controller + PrefixKVProjector (跳过旧 LoRA): {resume_rld}")
            else:
                print(f"[RLD Stage 2] ✅ 从 Stage 1 checkpoint 加载: {resume_rld}")

    if is_main:
        # ZeRO-3 下 p.numel() 返回分片大小，需要特殊处理
        if use_zero3:
            try:
                import deepspeed
                trainable = sum(
                    p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                    for p in model.parameters() if p.requires_grad
                )
                total = sum(
                    p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                    for p in model.parameters()
                )
            except ImportError:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
        else:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable / 1e6:.2f}M / 总参数: {total / 1e9:.2f}B "
              f"({trainable / total * 100:.2f}%)")

    # ====== 4. 创建 Collator ======
    collator = RLDCollator(
        processor=processor,
        step_delimiter_ids=model.step_delimiter_ids,
    )

    # ====== 5. 训练参数 ======
    if is_main:
        print("\n[4/5] 配置训练参数...")

    output_dir = args.output_dir or training_config['output_dir']
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs or training_config['num_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=args.learning_rate or training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 100),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        bf16=model_config.get('torch_dtype', 'bfloat16') == 'bfloat16',
        fp16=model_config.get('torch_dtype', '') == 'float16',
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=training_config.get('logging_steps', 20),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=training_config.get('eval_steps', 500) if eval_dataset else None,
        dataloader_num_workers=training_config.get('num_workers', 4),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        # 分布式后端: DDP 或 DeepSpeed (由 --use_deepspeed 参数控制)
        deepspeed=ds_config_path if use_deepspeed else None,
        # DDP 配置: RLD 的计算图是数据依赖的 (不同样本 step 数量不同),
        # 当某些样本没有 step boundary 时, trace_updater/reflection/draft_updater 等
        # 模块不会被调用, 其参数不参与 loss → DDP reduction 失败。
        # 此外 streaming_accumulator 仅用于推理, 训练时永远不参与 forward。
        # 必须设置 find_unused_parameters=True 让 DDP 自动处理。
        ddp_find_unused_parameters=True,
        report_to=["tensorboard"],
        seed=seed,
        ddp_timeout=1800,  # DDP 超时 (默认)
    )

    # ====== 6. 创建 Trainer ======
    if is_main:
        print("\n[5/5] 创建 Trainer...")
    # LoRA 学习率: 从 training 配置中读取, 如果未配置则使用 controller_lr * 0.3
    _lora_lr = training_config.get('lora_lr', None)
    
    trainer = RLDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        controller_lr=training_config.get('controller_lr', 1e-4),
        lora_lr=_lora_lr,
        monitor_every_n_steps=training_config.get('monitor_every_n_steps', 50),
    )

    # ====== 7. 训练 ======
    if is_main:
        print("\n" + "=" * 80)
        print("🚀 开始 RLD 训练...")
        print("=" * 80 + "\n")

    resume_from = args.resume_from_checkpoint or training_config.get('resume_from_checkpoint')
    if resume_from and not os.path.exists(resume_from):
        resume_from = None

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # ====== 8. 保存 ======
    if is_main:
        print("\n" + "=" * 80)
        print("💾 保存模型...")
        print("=" * 80)

    final_dir = os.path.join(output_dir, 'model')
    # ZeRO-3 下所有进程都需要参与 save (GatheredParameters gather 操作)
    model.save_pretrained(final_dir)

    # 同步所有进程
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # 保存 processor
    if _is_main_process():
        processor.save_pretrained(final_dir)

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if is_main:
        print(f"\n✅ 模型已保存到 {final_dir}")
        print("\n🎉 训练完成！")


if __name__ == "__main__":
    main()
