"""
NLD Trainer: 自定义 Trainer 支持 Native Latent Draft 训练

核心特性:
1. 差异化学习率: NativeLatentThinker lr vs VLM 基座 lr
2. 监控 NLD 特有指标: thought_count, mean_think_steps, ponder_cost 等
3. 兼容 FSDP / DDP 分布式训练
"""

import os
import logging
import time
import torch
from transformers import Trainer
from typing import Optional, Dict, List
from collections import defaultdict

logging.getLogger("transformers.trainer").setLevel(logging.WARNING)


class NLDTrainer(Trainer):
    """
    NLD Trainer
    
    扩展 HuggingFace Trainer:
    - 自定义 compute_loss: 调用 NLDModel.forward
    - 差异化学习率: NativeLatentThinker vs VLM 基座
    - 监控 NLD 特有指标
    - 兼容 FSDP / DDP 分布式训练
    """

    def __init__(
        self,
        *args,
        thinker_lr: float = 1e-4,
        vlm_lr: float = None,
        monitor_every_n_steps: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.thinker_lr = thinker_lr
        self.vlm_lr = vlm_lr  # VLM 基座学习率 (全量微调)
        self.monitor_every_n_steps = monitor_every_n_steps
        self._step_count = 0
        self._fwd_count = 0
        self._train_start_time = None
        self._sample_stats_accum = defaultdict(list)
        self._recent_losses = []
        self._loss_window_size = 50

    def _is_main_process(self):
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', -1)))
        return local_rank in (-1, 0)

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """覆盖 Trainer.log(): 只写 TensorBoard"""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if start_time is not None:
            logs["step_time"] = round(start_time, 4)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        
        for callback in self.callback_handler.callbacks:
            cb_name = type(callback).__name__
            if cb_name in ("PrinterCallback", "ProgressCallback", "DefaultFlowCallback"):
                continue
            if hasattr(callback, 'on_log'):
                callback.on_log(self.args, self.state, self.control, logs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """自定义损失计算"""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        step_boundaries = inputs.get("step_boundaries")
        prompt_lens = inputs.get("prompt_lens")
        loss_weight_mask = inputs.get("loss_weight_mask")
        latent_think_steps = inputs.get("latent_think_steps")
        stage_concept_token_ids = inputs.get("stage_concept_token_ids")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            step_boundaries=step_boundaries,
            prompt_lens=prompt_lens,
            loss_weight_mask=loss_weight_mask,
            latent_think_steps=latent_think_steps,
            stage_concept_token_ids=stage_concept_token_ids,
        )

        loss = outputs['loss']

        # 首次 forward 验证计算图
        if not hasattr(self, '_grad_path_verified'):
            self._grad_path_verified = True
            if self._is_main_process():
                _has_grad_fn = loss.grad_fn is not None
                print(f"\n  🔬 计算图验证 (首次 forward):")
                print(f"    loss.grad_fn: {'✅ 存在' if _has_grad_fn else '❌ None'}")
                if _has_grad_fn:
                    print(f"    loss.grad_fn 类型: {type(loss.grad_fn).__name__}")
                print()

        self._fwd_count += 1
        self._pending_draft_metrics = outputs.get('draft_metrics', {})
        self._pending_main_loss = outputs.get('main_loss', torch.tensor(0.0))
        self._pending_total_loss = loss

        # 累积样本级统计
        sample_meta = inputs.get('sample_meta', {})
        if sample_meta:
            for key, values in sample_meta.items():
                if isinstance(values, list):
                    self._sample_stats_accum[key].extend(values)
                else:
                    self._sample_stats_accum[key].append(values)

        if labels is not None:
            _valid_tokens = (labels != -100).sum().item()
            _total_tokens = labels.numel()
            self._sample_stats_accum['valid_tokens'].append(_valid_tokens)
            self._sample_stats_accum['total_tokens'].append(_total_tokens)

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """重写 training_step"""
        _t_start = time.time()
        _rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))

        if self._train_start_time is None:
            self._train_start_time = _t_start

        # 调用父类 training_step
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 更新 step 计数 (在梯度累积完成后)
        if (self._fwd_count % self.args.gradient_accumulation_steps == 0):
            self._step_count += 1

            # 记录 loss 到滑动窗口
            _total_loss_val = self._pending_total_loss.item() if isinstance(self._pending_total_loss, torch.Tensor) else self._pending_total_loss
            self._recent_losses.append(_total_loss_val)
            if len(self._recent_losses) > self._loss_window_size:
                self._recent_losses = self._recent_losses[-self._loss_window_size:]

            # 定期打印监控信息
            if self._is_main_process() and self._step_count % self.monitor_every_n_steps == 0:
                self._print_monitor_info()

            # 写入 TensorBoard
            if self._is_main_process():
                log_dict = {}
                _main_loss = self._pending_main_loss.item() if isinstance(self._pending_main_loss, torch.Tensor) else self._pending_main_loss
                log_dict['loss/total'] = _total_loss_val
                log_dict['loss/main_ce'] = _main_loss

                # NLD 特有指标
                draft_metrics = getattr(self, '_pending_draft_metrics', {})
                for k, v in draft_metrics.items():
                    if isinstance(v, (int, float)):
                        log_dict[k] = v

                self.log(log_dict)

            # 清空累积器
            self._sample_stats_accum.clear()

        return loss

    def _print_monitor_info(self):
        """打印监控信息"""
        _elapsed = time.time() - self._train_start_time if self._train_start_time else 0
        _hours = _elapsed / 3600
        _steps_per_hour = self._step_count / max(_hours, 0.001)

        # 估算 ETA
        total_steps = self.state.max_steps if self.state.max_steps > 0 else 0
        remaining_steps = total_steps - self._step_count
        eta_hours = remaining_steps / max(_steps_per_hour, 1)

        _total_loss = self._pending_total_loss.item() if isinstance(self._pending_total_loss, torch.Tensor) else 0
        _main_loss = self._pending_main_loss.item() if isinstance(self._pending_main_loss, torch.Tensor) else 0

        # 滑动窗口 loss 统计
        _avg_loss = sum(self._recent_losses) / max(len(self._recent_losses), 1)

        draft_metrics = getattr(self, '_pending_draft_metrics', {})
        _thought_count = draft_metrics.get('nld/thought_count', 0)
        _mean_think_steps = draft_metrics.get('nld/mean_think_steps', 0)
        _num_steps = draft_metrics.get('nld/num_steps', 0)

        print(f"\n📊 [Step {self._step_count}/{total_steps}] "
              f"({_hours:.1f}h, {_steps_per_hour:.0f} steps/h, ETA {eta_hours:.1f}h)")
        print(f"  Loss: total={_total_loss:.4f}, main_ce={_main_loss:.4f}, "
              f"avg({len(self._recent_losses)})={_avg_loss:.4f}")
        print(f"  NLD: thoughts={_thought_count:.0f}, steps={_num_steps:.0f}, "
              f"mean_think_steps={_mean_think_steps:.2f}")

        # 辅助 loss 监控
        _comp_loss = draft_metrics.get('nld/complementarity_loss', None)
        _probe_loss = draft_metrics.get('nld/visual_probe_loss', None)
        _exit_token_loss = draft_metrics.get('nld/exit_token_loss', None)
        _key_token_loss = draft_metrics.get('nld/key_token_loss', None)
        _aux_parts = []
        if _comp_loss is not None:
            _aux_parts.append(f"comp={_comp_loss:.4f}")
        if _probe_loss is not None:
            _aux_parts.append(f"vis_probe={_probe_loss:.4f}")
        if _exit_token_loss is not None:
            _aux_parts.append(f"exit_token={_exit_token_loss:.4f}")
        if _key_token_loss is not None:
            _aux_parts.append(f"key_token={_key_token_loss:.4f}")
        if _aux_parts:
            print(f"  Aux Loss: {', '.join(_aux_parts)}")

        # Per-region loss
        _think_ce = draft_metrics.get('loss/think_ce', None)
        _answer_ce = draft_metrics.get('loss/answer_ce', None)
        if _think_ce is not None or _answer_ce is not None:
            print(f"  Region CE: think={_think_ce:.4f}" if _think_ce else "", end="")
            print(f", answer={_answer_ce:.4f}" if _answer_ce else "")

        # Top-1 准确率
        _top1 = draft_metrics.get('acc/top1_overall', None)
        _top1_answer = draft_metrics.get('acc/top1_answer', None)
        if _top1 is not None:
            print(f"  Accuracy: top1={_top1:.4f}", end="")
            if _top1_answer is not None:
                print(f", top1_answer={_top1_answer:.4f}", end="")
            print()

        # 样本统计
        valid_tokens = self._sample_stats_accum.get('valid_tokens', [])
        if valid_tokens:
            _avg_valid = sum(valid_tokens) / len(valid_tokens)
            print(f"  Tokens: avg_valid={_avg_valid:.0f}")

        print(flush=True)

    def create_optimizer(self):
        """创建差异化学习率的优化器 (全量微调, FSDP/DDP 兼容)"""
        model = self.model
        
        # 分组参数
        thinker_params = []
        vlm_params_decay = []
        vlm_params_no_decay = []
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "RMSNorm"]
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'latent_thinker' in name or 'thought_injector' in name or 'visual_probe' in name:
                thinker_params.append(param)
            else:
                if any(nd in name for nd in no_decay):
                    vlm_params_no_decay.append(param)
                else:
                    vlm_params_decay.append(param)
        
        _vlm_lr = self.vlm_lr if self.vlm_lr is not None else self.args.learning_rate
        
        # 构建参数组
        optimizer_grouped_parameters = []
        
        if thinker_params:
            optimizer_grouped_parameters.append({
                'params': thinker_params,
                'lr': self.thinker_lr,
                'weight_decay': self.args.weight_decay,
            })
        
        if vlm_params_decay:
            optimizer_grouped_parameters.append({
                'params': vlm_params_decay,
                'lr': _vlm_lr,
                'weight_decay': self.args.weight_decay,
            })
        
        if vlm_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': vlm_params_no_decay,
                'lr': _vlm_lr,
                'weight_decay': 0.0,
            })
        
        if self._is_main_process():
            print(f"\n[NLD Optimizer] 参数组 (全量微调, FSDP/DDP):")
            print(f"  NativeLatentThinker: {len(thinker_params)} params, lr={self.thinker_lr}")
            print(f"  VLM (decay): {len(vlm_params_decay)} params, lr={_vlm_lr}")
            print(f"  VLM (no_decay): {len(vlm_params_no_decay)} params, lr={_vlm_lr}")
            print()
        
        optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        return self.optimizer
