"""
RLD Trainer: 自定义 Trainer 支持步段展开训练

核心特性:
1. 按 step 段展开的训练流程
2. 只训练 RLD Controller 参数
3. 多样性正则化监控
4. 兼容 PyTorch DDP (默认) 和 DeepSpeed ZeRO-2/ZeRO-3 (可选)
"""

import os
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from typing import Optional, Dict
from collections import defaultdict


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
        self._train_start_time = None  # 首次 step 时记录，用于计算 ETA
        # 梯度捕获: 通过 backward hook 在梯度实际产生时记录
        # - DDP 模式: hook 和 param.grad 都可靠，hook 优先 (更及时)
        # - DeepSpeed ZeRO-2: reduce_scatter 后 param.grad 不可靠，必须用 hook
        self._grad_norms = {}  # {param_name: grad_norm_squared}
        self._grad_hooks_registered = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        自定义损失计算
        
        调用 RLDModel.forward，它会:
        1. 初始化 RLD 状态
        2. 带 prefix KV 的 forward
        3. 按 step 执行 RLD 更新
        4. 返回 main_loss + div_loss + draft_metrics
        
        注意: 不在此方法内调用 self.log()，避免与 Trainer 内部状态冲突。
        指标暂存到 self._pending_log_dict，在 training_step 中统一记录。
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

        # ====== 首次 forward 验证计算图: 确保梯度路径完整 ======
        if not hasattr(self, '_grad_path_verified'):
            self._grad_path_verified = True
            if self._is_main_process():
                _has_grad_fn = loss.grad_fn is not None
                _div_has_grad = outputs.get('div_loss', torch.tensor(0.0)).grad_fn is not None if isinstance(outputs.get('div_loss'), torch.Tensor) else False
                print(f"\n  🔬 计算图验证 (首次 forward):")
                print(f"    loss.grad_fn: {'✅ 存在' if _has_grad_fn else '❌ None (梯度无法回传!)'}")
                print(f"    div_loss.grad_fn: {'✅ 存在' if _div_has_grad else '⚠️ None'}")
                if _has_grad_fn:
                    print(f"    loss.grad_fn 类型: {type(loss.grad_fn).__name__}")
                print()

        # ====== Draft 状态监控指标 → 暂存 (不在 compute_loss 中调用 self.log) ======
        self._step_count += 1
        draft_metrics = outputs.get('draft_metrics', {})

        # 暂存日志指标，在 training_step 中统一记录
        self._pending_log_dict = None
        self._pending_console_output = None

        # 每 logging_steps 准备 draft 指标
        if draft_metrics and self._step_count % max(self.args.logging_steps, 1) == 0:
            log_dict = {}
            for key, value in draft_metrics.items():
                if isinstance(value, (int, float)):
                    log_dict[key] = value
            # main_loss 和 div_loss 也记录到 TensorBoard
            log_dict['loss/main_loss'] = outputs.get('main_loss', torch.tensor(0.0)).item()
            log_dict['loss/div_loss'] = outputs.get('div_loss', torch.tensor(0.0)).item()
            log_dict['loss/total_loss'] = loss.item()
            self._pending_log_dict = log_dict

        # 每 monitor_every_n_steps 准备详细监控信息
        if self._step_count % self.monitor_every_n_steps == 0 and self._is_main_process():
            main_loss = outputs.get('main_loss', torch.tensor(0.0))
            div_loss = outputs.get('div_loss', torch.tensor(0.0))
            lines = []
            lines.append(f"\n{'='*70}")
            lines.append(f"[RLD Step {self._step_count}] Loss 概览")
            lines.append(f"  total_loss={loss.item():.4f}, "
                  f"main_loss={main_loss.item():.4f}, "
                  f"div_loss={div_loss.item():.4f}")

            if draft_metrics:
                lines.append(f"\n  📊 Draft 状态监控:")
                # 有效秩
                if 'draft/Zd_effective_rank' in draft_metrics:
                    lines.append(f"    Z_d 有效秩: {draft_metrics['draft/Zd_effective_rank']:.2f}")
                if 'draft/Ze_effective_rank' in draft_metrics:
                    lines.append(f"    Z_e 有效秩: {draft_metrics['draft/Ze_effective_rank']:.2f}")
                if 'draft/T_effective_rank' in draft_metrics:
                    lines.append(f"    T 轨迹有效秩: {draft_metrics['draft/T_effective_rank']:.2f}")
                # 槽间相似度
                if 'draft/Zd_slot_cosim' in draft_metrics:
                    cosim = draft_metrics['draft/Zd_slot_cosim']
                    status = "✅" if cosim < 0.5 else ("⚠️" if cosim < 0.8 else "❌坍塌")
                    lines.append(f"    Z_d 槽间余弦相似度: {cosim:.4f} {status}")
                # 段间变化率
                if 'draft/Zd_relative_delta' in draft_metrics:
                    delta = draft_metrics['draft/Zd_relative_delta']
                    status = "✅" if 0.05 <= delta <= 0.3 else ("⚠️过小" if delta < 0.05 else "⚠️过大")
                    lines.append(f"    Z_d 段间变化率: {delta:.4f} {status}")
                # 段间 prefix 余弦相似度
                if 'draft/prefix_inter_seg_cosim' in draft_metrics:
                    cosim = draft_metrics['draft/prefix_inter_seg_cosim']
                    status = "✅" if 0.7 <= cosim <= 0.95 else ("⚠️不变" if cosim > 0.95 else "⚠️剧变")
                    lines.append(f"    段间 prefix 余弦相似度: {cosim:.4f} {status}")
                # Evidence Gate β
                if 'draft/beta_mean' in draft_metrics:
                    lines.append(f"    Evidence Gate β: "
                          f"mean={draft_metrics['draft/beta_mean']:.4f}, "
                          f"std={draft_metrics['draft/beta_std']:.4f}, "
                          f"range=[{draft_metrics.get('draft/beta_min', 0):.4f}, "
                          f"{draft_metrics.get('draft/beta_max', 0):.4f}]")
                # 范数
                if 'draft/Zd_norm' in draft_metrics:
                    lines.append(f"    Z_d 范数: {draft_metrics['draft/Zd_norm']:.4f}")

                # ====== 多层 Draft 注入效果 ======
                lines.append(f"\n  🎯 多层 Draft 注入效果:")
                if 'draft/inject_norm_mean' in draft_metrics:
                    inject_norm = draft_metrics['draft/inject_norm_mean']
                    inject_ratio = draft_metrics.get('draft/inject_ratio', 0.0)
                    status = "✅生效" if inject_norm > 1e-6 else "⚠️极小"
                    lines.append(f"    注入修正量 L2 范数: {inject_norm:.6f} {status}")
                    lines.append(f"    修正/原始比: {inject_ratio:.6f}")
                # 各层 scale 汇总
                _scale_lines = []
                for key in sorted(draft_metrics.keys()):
                    if key.startswith('draft/readout_scale_L'):
                        layer_num = key.split('_L')[-1]
                        _scale_lines.append(f"L{layer_num}={draft_metrics[key]:.4f}")
                if _scale_lines:
                    lines.append(f"    各层 Readout Scale: {', '.join(_scale_lines)}")
                if 'draft/num_steps' in draft_metrics:
                    lines.append(f"    Step 数量: {int(draft_metrics['draft/num_steps'])}")

            lines.append(f"{'='*70}")
            self._pending_console_output = "\n".join(lines)

        # 暂存 outputs 供 training_step 后收集梯度指标
        self._last_outputs = outputs

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        重写 training_step: 在 backward 后记录日志和收集 controller 梯度指标
        """
        # ===== 跨 rank 时间戳: 检测 forward/backward 阶段是否有 rank 卡住 =====
        import time as _time
        _t_start = _time.time()
        _rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))

        # 首次 step 记录训练开始时间
        if self._train_start_time is None:
            self._train_start_time = _t_start

        # 确保 backward hook 已注册 (首次调用时注册, 后续幂等)
        self._register_grad_hooks(model)

        loss = super().training_step(model, inputs, num_items_in_batch)

        _t_end = _time.time()
        _duration = _t_end - _t_start
        # 每 10 步由 rank 0 打印耗时 + 训练进度; 异常耗时 (>60s) 所有 rank 打印
        _is_rank0 = (_rank == 0)
        _should_print = (
            _duration > 60  # 异常: 超过 60s, 所有 rank 打印
            or (_is_rank0 and self._step_count % 10 == 0)  # 每 10 步, 仅 rank 0
        )
        if _should_print:
            _warn = " ⚠️⚠️⚠️" if _duration > 60 else ""
            # 计算训练时间和 ETA
            _elapsed = _t_end - self._train_start_time
            _progress_info = ""
            if _is_rank0 and self.state.max_steps > 0:
                _pct = self._step_count / self.state.max_steps * 100
                _eta = _elapsed / max(self._step_count, 1) * (self.state.max_steps - self._step_count)
                _progress_info = (
                    f" | 进度: {self._step_count}/{self.state.max_steps} ({_pct:.1f}%)"
                    f" | 已用: {self._format_time(_elapsed)}"
                    f" | 剩余: {self._format_time(_eta)}"
                )
            print(f"[rank {_rank}] step={self._step_count} 耗时={_duration:.1f}s{_warn}{_progress_info}", flush=True)

        # 输出 compute_loss 中暂存的日志 (避免在 compute_loss 中调用 self.log)
        if getattr(self, '_pending_log_dict', None) is not None:
            self.log(self._pending_log_dict)
            self._pending_log_dict = None

        if getattr(self, '_pending_console_output', None) is not None:
            print(self._pending_console_output)
            self._pending_console_output = None

        # 每 monitor_every_n_steps 收集梯度范数
        if self._step_count % self.monitor_every_n_steps == 0:
            grad_metrics = self._collect_grad_metrics(model)
            if grad_metrics:
                # 记录到 TensorBoard
                self.log(grad_metrics)
                # 控制台打印
                if self._is_main_process():
                    # 区分 controller 和 readout_adapter 的指标
                    ctrl_metrics = {k: v for k, v in sorted(grad_metrics.items())
                                    if not k.startswith('grad/readout')}
                    readout_metrics = {k: v for k, v in sorted(grad_metrics.items())
                                       if k.startswith('grad/readout')}

                    print(f"\n  🔧 Controller 梯度监控 (via {'hook' if self._grad_hooks_registered else 'param.grad'}):")
                    _has_any_grad = False
                    for key, value in sorted(ctrl_metrics.items()):
                        if value == 0.0:
                            print(f"    {key}: {value:.6f} ❌无梯度")
                        else:
                            print(f"    {key}: {value:.6f} ✅")
                            _has_any_grad = True
                    if readout_metrics:
                        print(f"  🔧 ReadoutAdapter 梯度监控:")
                        for key, value in sorted(readout_metrics.items()):
                            if value == 0.0:
                                print(f"    {key}: {value:.6f} ❌无梯度")
                            else:
                                print(f"    {key}: {value:.6f} ✅")
                                _has_any_grad = True
                    if not _has_any_grad:
                        print(f"  ⚠️  所有模块梯度为 0, 可能原因:")
                        if self.is_deepspeed_enabled:
                            print(f"      - DeepSpeed ZeRO: reduce_scatter 后 param.grad 不可靠")
                            print(f"      - backward hook 未触发 (检查参数是否在计算图中)")
                        else:
                            print(f"      - DDP 模式: 参数可能未在 loss 的计算图中")
                        print(f"      - 检查 loss.grad_fn 是否存在 (首次 forward 的 🔬 验证信息)")
                    print()

        return loss

    def _register_grad_hooks(self, model):
        """
        注册 backward hook 在梯度产生时捕获范数。
        
        兼容 DDP 和 DeepSpeed ZeRO-2:
        - DDP: param.grad 在 backward 后可靠，但 hook 能更及时地捕获
        - DeepSpeed ZeRO-2: reduce_scatter 后 param.grad 不可靠，hook 是唯一可靠方式
        """
        if self._grad_hooks_registered:
            return

        inner_model = model
        if hasattr(model, 'module'):
            inner_model = model.module

        # 为 controller 和 readout_adapter 的所有可训练参数注册 hook
        for prefix, module in [('ctrl', getattr(inner_model, 'controller', None)),
                                ('readout', getattr(inner_model, 'readout_adapter', None))]:
            if module is None:
                continue
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                full_name = f"{prefix}/{name}"
                # 使用 register_post_accumulate_grad_hook (PyTorch 2.1+)
                # 如果不可用，退回 register_hook
                if hasattr(param, 'register_post_accumulate_grad_hook'):
                    def _make_hook(pname):
                        def _hook(p):
                            if p.grad is not None:
                                self._grad_norms[pname] = p.grad.data.float().norm(2).item() ** 2
                        return _hook
                    param.register_post_accumulate_grad_hook(_make_hook(full_name))
                else:
                    def _make_hook(pname):
                        def _hook(grad):
                            if grad is not None:
                                self._grad_norms[pname] = grad.data.float().norm(2).item() ** 2
                        return _hook
                    param.register_hook(_make_hook(full_name))

        self._grad_hooks_registered = True

    def _collect_grad_metrics(self, model) -> Dict[str, float]:
        """
        收集 controller + readout_adapter 各模块的梯度 L2 范数。
        
        优先使用 backward hook 捕获的梯度 (兼容 DeepSpeed ZeRO-2),
        如果 hook 没有数据则回退到直接检查 param.grad。
        
        Returns:
            grad_metrics: {'grad/module_name': grad_norm, ...}
        """
        # 获取内部模型 (可能被 DeepSpeed/DDP 包裹)
        inner_model = model
        if hasattr(model, 'module'):
            inner_model = model.module

        module_grads = defaultdict(float)
        module_counts = defaultdict(int)

        # ---- 方法 1: 使用 hook 捕获的梯度 ----
        if self._grad_norms:
            for full_name, grad_norm_sq in self._grad_norms.items():
                # full_name 格式: "ctrl/evidence_resampler.layers.0.q_proj.weight"
                prefix, param_name = full_name.split('/', 1)
                module_name = param_name.split('.')[0]
                display_name = f"{prefix}/{module_name}" if prefix == 'readout' else module_name
                module_grads[display_name] += grad_norm_sq
                module_counts[display_name] += 1
            # 清空 hook 数据 (下次 backward 会重新填充)
            self._grad_norms.clear()
        else:
            # ---- 方法 2: 回退到直接检查 param.grad (DDP 模式下可靠) ----
            for prefix, module in [('ctrl', getattr(inner_model, 'controller', None)),
                                    ('readout', getattr(inner_model, 'readout_adapter', None))]:
                if module is None:
                    continue
                for name, param in module.named_parameters():
                    if not param.requires_grad:
                        continue
                    module_name = name.split('.')[0]
                    display_name = f"{prefix}/{module_name}" if prefix == 'readout' else module_name
                    if param.grad is not None:
                        grad_norm = param.grad.data.float().norm(2).item()
                        module_grads[display_name] += grad_norm ** 2
                        module_counts[display_name] += 1
                    else:
                        if display_name not in module_grads:
                            module_grads[display_name] = 0.0

        # 转换为 L2 范数
        grad_metrics = {}
        for module_name in sorted(module_grads.keys()):
            grad_val = module_grads[module_name]
            if module_counts[module_name] > 0:
                grad_metrics[f'grad/{module_name}'] = grad_val ** 0.5
            else:
                grad_metrics[f'grad/{module_name}'] = 0.0

        return grad_metrics

    def create_optimizer(self):
        """
        自定义优化器: 只优化 RLD Controller 参数
        
        支持两种模式:
        - DDP 模式 (默认): 直接使用 PyTorch AdamW
        - DeepSpeed 模式: 使用 FusedAdam (如可用) 或 PyTorch AdamW
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

        if self.is_deepspeed_enabled:
            # DeepSpeed 模式: 优先使用 FusedAdam (性能更优)
            from torch.optim import AdamW as TorchAdamW
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
            # DDP 模式 (默认): 直接使用 PyTorch AdamW
            from torch.optim import AdamW
            self.optimizer = AdamW(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        return self.optimizer

    @staticmethod
    def _format_time(seconds: float) -> str:
        """将秒数格式化为可读的时间字符串"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s:02d}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h{m:02d}m{s:02d}s"

    def _is_main_process(self) -> bool:
        """判断是否为主进程"""
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        return local_rank in (-1, 0)
