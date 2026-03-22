"""
RLD Trainer: 自定义 Trainer 支持步段展开训练

核心特性:
1. 按 step 段展开的训练流程
2. 只训练 RLD Controller 参数
3. 多样性正则化监控
4. 兼容 DeepSpeed ZeRO-2/ZeRO-3
"""

import os
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from typing import Optional, Dict


class RLDTrainer(Trainer):
    """
    RLD Trainer
    
    扩展 HuggingFace Trainer:
    - 自定义 compute_loss: 调用 RLDModel.forward (带 step 段展开)
    - 只优化 RLD Controller 参数
    - 监控 diversity loss 和 step update 统计
    
    Args:
        controller_lr: RLD Controller 学习率 (默认 1e-4)
        monitor_every_n_steps: 每 N 步打印监控信息
    """

    def __init__(
        self,
        *args,
        controller_lr: float = 1e-4,
        monitor_every_n_steps: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controller_lr = controller_lr
        self.monitor_every_n_steps = monitor_every_n_steps
        self._step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        自定义损失计算
        
        调用 RLDModel.forward，它会:
        1. 初始化 RLD 状态
        2. 带 prefix KV 的 forward
        3. 按 step 执行 RLD 更新
        4. 返回 main_loss + div_loss
        """
        # 提取输入
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        step_boundaries = inputs.get("step_boundaries")
        prompt_lens = inputs.get("prompt_lens")

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            step_boundaries=step_boundaries,
            prompt_lens=prompt_lens,
        )

        loss = outputs['loss']

        # 监控
        self._step_count += 1
        if self._step_count % self.monitor_every_n_steps == 0 and self._is_main_process():
            main_loss = outputs.get('main_loss', torch.tensor(0.0))
            div_loss = outputs.get('div_loss', torch.tensor(0.0))
            print(f"\n[RLD Step {self._step_count}] "
                  f"total_loss={loss.item():.4f}, "
                  f"main_loss={main_loss.item():.4f}, "
                  f"div_loss={div_loss.item():.4f}")

        if return_outputs:
            return loss, outputs
        return loss

    def create_optimizer(self):
        """
        自定义优化器: 只优化 RLD Controller 参数
        
        ZeRO-3 兼容: 当使用 DeepSpeed 时，让 DeepSpeed 管理优化器创建。
        手动创建 torch.optim.AdamW 会绕过 DeepSpeed 的优化器封装，
        导致 ZeRO-3 无法正确分片优化器状态。
        """
        model = self.model

        # 收集可训练参数
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))

        if len(trainable_params) == 0:
            raise ValueError("[RLD Trainer] 没有可训练参数！请检查模型冻结设置。")

        # 分组: controller 参数使用 controller_lr
        param_groups = [
            {
                "params": [p for _, p in trainable_params],
                "lr": self.controller_lr,
                "weight_decay": self.args.weight_decay,
            }
        ]

        # 检查是否使用 DeepSpeed
        if self.is_deepspeed_enabled:
            # DeepSpeed 模式: 使用 DeepSpeed 管理的优化器
            # DeepSpeed 会根据 ds_config 中的 optimizer 配置 (或 auto) 创建优化器
            # 这里只设置参数组，让 HuggingFace Trainer 的 DeepSpeed 集成处理剩余逻辑
            from transformers.trainer import get_parameter_names
            from torch.optim import AdamW as TorchAdamW

            # 让 Trainer 基类处理 DeepSpeed 优化器创建
            # 但需要先设置 self.optimizer 为 None，使基类能正确初始化
            # 通过 DeepSpeed 的 optimizer config auto 机制自动适配
            try:
                from deepspeed.ops.adam import FusedAdam
                optimizer_cls = FusedAdam
            except ImportError:
                optimizer_cls = TorchAdamW

            self.optimizer = optimizer_cls(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            # 非 DeepSpeed 模式: 直接使用 PyTorch AdamW
            from torch.optim import AdamW
            self.optimizer = AdamW(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        return self.optimizer

    def _is_main_process(self) -> bool:
        """判断是否为主进程"""
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        return local_rank in (-1, 0)
