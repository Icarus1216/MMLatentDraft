#!/usr/bin/env python3
"""
NLD 训练脚本 (Native Latent Draft)

使用方法:
    # 单卡
    python scripts/train_nld.py --config configs/nld_train_phase1.yaml
    
    # 多卡 (FSDP, 推荐)
    torchrun --nproc_per_node=8 scripts/train_nld.py --config configs/nld_train_phase1.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import json
from pathlib import Path

# NCCL 超时设置: segment-wise forward 各 rank 耗时差异大, 需要更长的超时
os.environ.setdefault('NCCL_TIMEOUT', '3600')  # 60 分钟 (与 ddp_timeout 一致)
os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '3600')  # watchdog 超时

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    TrainingArguments,
    AutoProcessor,
    set_seed,
)

from rld.model_v2 import NLDModel
from rld.data import NLDDataset, NLDCollator, LATENT_TOKEN, LATENT_END_TOKEN
from rld.trainer_nld import NLDTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="NLD Training Script")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖输出目录")
    parser.add_argument("--learning_rate", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
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
    nld_config = config.get('nld', {})
    seed = config.get('seed', 42)

    set_seed(seed)
    is_main = _is_main_process()

    # ====== 1. 加载 Processor ======
    if is_main:
        print("\n[1/5] 加载 Processor...")
    processor = AutoProcessor.from_pretrained(model_config['model_path'])

    _min_pixels = data_config.get('min_pixels', 3136)
    _max_pixels = data_config.get('max_pixels', 1003520)
    processor.image_processor.min_pixels = _min_pixels
    processor.image_processor.max_pixels = _max_pixels
    if is_main:
        print(f"  图片分辨率限制: min_pixels={_min_pixels}, max_pixels={_max_pixels}")

    # ====== 1.5 注册 <|latent|> 和 <|/latent|> 特殊 token ======
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [LATENT_TOKEN, LATENT_END_TOKEN]}
    )
    latent_token_id = processor.tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    latent_end_token_id = processor.tokenizer.convert_tokens_to_ids(LATENT_END_TOKEN)
    if is_main:
        print(f"  注册特殊 token: {LATENT_TOKEN} → id={latent_token_id}, "
              f"{LATENT_END_TOKEN} → id={latent_end_token_id} (新增 {num_added} 个)")

    # ====== 2. 创建数据集 ======
    if is_main:
        print("\n[2/5] 创建数据集...")
    train_dataset = NLDDataset(
        json_path=data_config['train_json'],
        processor=processor,
        image_base_dir=data_config.get('image_base_dir', None),
        use_system_prompt=data_config.get('use_system_prompt', True),
        system_prompt=data_config.get('system_prompt', None),
        max_seq_len=data_config.get('max_seq_len', 4096),
        skip_image_check=data_config.get('skip_image_check', False),
    )
    if is_main:
        print(f"训练样本数: {len(train_dataset)}")

    eval_dataset = None
    if data_config.get('eval_json'):
        eval_dataset = NLDDataset(
            json_path=data_config['eval_json'],
            processor=processor,
            image_base_dir=data_config.get('image_base_dir', None),
            use_system_prompt=data_config.get('use_system_prompt', True),
            max_seq_len=data_config.get('max_seq_len', 4096),
            skip_image_check=data_config.get('skip_image_check', False),
        )
        if is_main:
            print(f"评估样本数: {len(eval_dataset)}")

    # ====== 3. 创建模型 ======
    if is_main:
        print("\n[3/5] 创建 NLD 模型...")

    # 将 Key Token Decoding Loss 权重设置为环境变量 (NLDModel 通过环境变量读取)
    if 'key_token_weight' in nld_config:
        os.environ['NLD_KEY_TOKEN_WEIGHT'] = str(nld_config['key_token_weight'])

    model = NLDModel(
        model_path=model_config['model_path'],
        hidden_size=model_config.get('hidden_size', 4096),
        total_layers=model_config.get('total_layers', 36),
        torch_dtype=getattr(torch, model_config.get('torch_dtype', 'bfloat16')),
        attn_implementation=model_config.get('attn_implementation', 'flash_attention_2'),
        max_think_steps=nld_config.get('max_think_steps', 5),
        think_layer_start=nld_config.get('think_layer_start', 0),
        think_layer_end=nld_config.get('think_layer_end', 36),
        saturation_exit_threshold=nld_config.get('saturation_exit_threshold', 0.99),
    )
    
    # 调整 embedding 大小以适配新增的特殊 token
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    if is_main:
        print(f"[NLD] Embedding 已调整为 {len(processor.tokenizer)} tokens (含 {LATENT_TOKEN})")
    
    model.set_processor(processor)

    # ====== 加载预训练权重 (如果有) ======
    resume_nld = nld_config.get('resume_from_checkpoint', None)
    if resume_nld and os.path.exists(resume_nld):
        model.load_pretrained(resume_nld)
        if is_main:
            print(f"[NLD] ✅ 从 checkpoint 加载: {resume_nld}")

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable / 1e6:.2f}M / 总参数: {total / 1e9:.2f}B "
              f"({trainable / total * 100:.2f}%)")

    # ====== 4. 创建 Collator ======
    collator = NLDCollator(
        processor=processor,
        latent_token_id=latent_token_id,
    )

    # ====== 5. 训练参数 ======
    if is_main:
        print("\n[4/5] 配置训练参数...")

    output_dir = args.output_dir or training_config['output_dir']
    
    # FSDP 配置 (替代 DeepSpeed, 完全兼容 KV cache + 多次 forward)
    fsdp_strategy = training_config.get('fsdp', '')
    fsdp_config_path = training_config.get('fsdp_config', None)
    fsdp_config = {}
    if fsdp_config_path and os.path.exists(fsdp_config_path):
        with open(fsdp_config_path, 'r') as f:
            fsdp_config = json.load(f)
        if is_main:
            print(f"[NLD] FSDP 配置: {fsdp_config_path}")
    elif fsdp_strategy:
        if is_main:
            print(f"[NLD] FSDP 策略: {fsdp_strategy}")
    
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
        ddp_find_unused_parameters=False,  # FSDP 模式下不需要且不兼容
        fsdp=fsdp_strategy if fsdp_strategy else "",
        fsdp_config=fsdp_config if fsdp_config else None,
        report_to=["tensorboard"],
        seed=seed,
        ddp_timeout=3600,  # 增大超时: segment-wise forward 各 rank 耗时差异大
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
    )

    # ====== 6. 创建 Trainer ======
    if is_main:
        print("\n[5/5] 创建 Trainer...")
    
    _thinker_lr = training_config.get('thinker_lr', 1e-4)
    _vlm_lr = training_config.get('vlm_lr', None)
    
    trainer = NLDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        thinker_lr=_thinker_lr,
        vlm_lr=_vlm_lr,
        monitor_every_n_steps=training_config.get('monitor_every_n_steps', 50),
    )

    # ====== 7. 训练 ======
    if is_main:
        print("\n" + "=" * 80)
        print("🚀 开始 NLD 训练 (Native Latent Draft)...")
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
    model.save_pretrained(final_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if _is_main_process():
        processor.save_pretrained(final_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if is_main:
        print(f"\n✅ 模型已保存到 {final_dir}")
        print("\n🎉 训练完成！")


if __name__ == "__main__":
    main()
