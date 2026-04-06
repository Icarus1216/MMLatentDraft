"""
RLD Trainer: 自定义 Trainer 支持步段展开训练

核心特性:
1. 按 step 段展开的训练流程
2. 只训练 RLD Controller 参数
3. 多样性正则化监控
4. 兼容 PyTorch DDP (默认) 和 DeepSpeed ZeRO-2/ZeRO-3 (可选)
"""

import os
import logging
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from typing import Optional, Dict, List
from collections import defaultdict

# 抑制 HF Trainer 的 console 字典打印 (保留 TensorBoard 写入)
# Trainer 内部通过 logging.getLogger(__name__) 打印 {'loss': ..., 'grad_norm': ...}
logging.getLogger("transformers.trainer").setLevel(logging.WARNING)


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
    rank_monitor_every_n_steps: 每 N 步打印秩快照 (比详细监控更频繁)
    """

    def __init__(
        self,
        *args,
        controller_lr: float = 1e-4,
        lora_lr: float = None,
        monitor_every_n_steps: int = 50,
        rank_monitor_every_n_steps: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controller_lr = controller_lr
        self.lora_lr = lora_lr  # Stage 2: LoRA 学习率 (None 表示不使用 LoRA)
        self.monitor_every_n_steps = monitor_every_n_steps
        self.rank_monitor_every_n_steps = rank_monitor_every_n_steps
        self._step_count = 0        # optimizer step 计数 (与 Trainer tqdm 一致)
        self._fwd_count = 0         # forward pass 计数 (含梯度累积的中间 forward)
        self._train_start_time = None  # 首次 step 时记录，用于计算 ETA
        # 梯度捕获: 通过 backward hook 在梯度实际产生时记录
        # - DDP 模式: hook 和 param.grad 都可靠，hook 优先 (更及时)
        # - DeepSpeed ZeRO-2: reduce_scatter 后 param.grad 不可靠，必须用 hook
        self._grad_norms = {}  # {param_name: grad_norm_squared}
        self._grad_hooks_registered = False

        # ====== 样本级统计累加器 (跨梯度累积步聚合) ======
        self._sample_stats_accum = defaultdict(list)  # 累积梯度累积期间的样本统计
        # 滑动窗口: 记录最近 N 个 optimizer step 的 loss, 用于检测 loss 震荡/发散
        self._recent_losses = []  # 最近的 total_loss 值
        self._recent_main_losses = []  # 最近的 main_loss 值
        self._loss_window_size = 50  # 滑动窗口大小

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """
        覆盖 Trainer.log(): 只写 TensorBoard, 不触发 console 字典打印。
        
        HF Trainer 默认的 log() 会触发 on_log callback, 导致 PrinterCallback
        在 console 打印完整的 log_dict 字典 (几十行), 非常冗余。
        我们已经有自己的 📊 精简摘要行, 不需要 HF 的字典打印。
        """
        # 保留 Trainer 内部的 epoch 等状态更新
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if start_time is not None:
            logs["step_time"] = round(start_time, 4)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        
        # 直接调用各 integration callback 的 on_log (TensorBoard 等),
        # 但跳过 PrinterCallback / ProgressCallback 的 console 字典打印
        for callback in self.callback_handler.callbacks:
            cb_name = type(callback).__name__
            # 只允许 TensorBoard / WandB 等 integration callback
            if cb_name in ("PrinterCallback", "ProgressCallback", "DefaultFlowCallback"):
                continue
            if hasattr(callback, 'on_log'):
                callback.on_log(self.args, self.state, self.control, logs)

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
        loss_weight_mask = inputs.get("loss_weight_mask")  # P2: per-token loss 权重

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            step_boundaries=step_boundaries,
            prompt_lens=prompt_lens,
            loss_weight_mask=loss_weight_mask,
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
        self._fwd_count += 1
        draft_metrics = outputs.get('draft_metrics', {})

        # 暂存最新的 draft_metrics 和 loss (在梯度累积期间可能被多次覆盖,
        # 最终 training_step 中使用的是最后一次 forward 的值)
        self._pending_draft_metrics = draft_metrics
        self._pending_main_loss = outputs.get('main_loss', torch.tensor(0.0))
        self._pending_div_loss = outputs.get('div_loss', torch.tensor(0.0))
        self._pending_commit_loss = outputs.get('commit_loss', torch.tensor(0.0))
        self._pending_grounding_loss = outputs.get('grounding_loss', torch.tensor(0.0))
        self._pending_total_loss = loss

        # 暂存 outputs 供 training_step 后收集梯度指标
        self._last_outputs = outputs

        # ====== 累积样本级元信息 (跨梯度累积步) ======
        sample_meta = inputs.get('sample_meta', {})
        if sample_meta:
            for key, values in sample_meta.items():
                if isinstance(values, list):
                    self._sample_stats_accum[key].extend(values)
                else:
                    self._sample_stats_accum[key].append(values)

        # 累积有效 token 统计 (从 labels 直接计算)
        if labels is not None:
            _valid_tokens = (labels != -100).sum().item()
            _total_tokens = labels.numel()
            self._sample_stats_accum['valid_tokens'].append(_valid_tokens)
            self._sample_stats_accum['total_tokens'].append(_total_tokens)
            self._sample_stats_accum['seq_lens'].append(labels.shape[-1])

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

        # ★ 首次 step 启用 anomaly detection, 获取 dtype 不匹配的精确位置
        _use_anomaly = (self._fwd_count <= 1)
        if _use_anomaly:
            with torch.autograd.set_detect_anomaly(True):
                loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs, num_items_in_batch)

        # 同步 optimizer step 计数: 使用 Trainer 内部的 global_step
        # global_step 在 optimizer.step() 后才递增，与 tqdm 进度条完全一致
        self._step_count = self.state.global_step

        _t_end = _time.time()
        _duration = _t_end - _t_start
        # 耗时日志: 仅在 optimizer step 边界打印, 避免梯度累积中间步重复输出
        # 异常耗时 (>60s) 所有 rank 立即打印 (不等 optimizer step 边界)
        _is_rank0 = (_rank == 0)
        _grad_accum_check = self.args.gradient_accumulation_steps
        _is_at_optim_boundary = (self._fwd_count % _grad_accum_check == 0)
        _should_print = (
            _duration > 60  # 异常: 超过 60s, 所有 rank 打印
            or (_is_rank0 and _is_at_optim_boundary and self._step_count % 100 == 0)  # 每 100 个 optimizer step, 仅 rank 0
        )
        if _should_print:
            _warn = " ⚠️⚠️⚠️" if _duration > 60 else ""
            # 计算训练时间和 ETA
            _elapsed = _t_end - self._train_start_time
            _progress_info = ""
            if _is_rank0 and self.state.max_steps > 0 and self._step_count > 0:
                _pct = self._step_count / self.state.max_steps * 100
                _eta = _elapsed / self._step_count * (self.state.max_steps - self._step_count)
                _progress_info = (
                    f" | 进度: {self._step_count}/{self.state.max_steps} ({_pct:.1f}%)"
                    f" | 已用: {self._format_time(_elapsed)}"
                    f" | 剩余: {self._format_time(_eta)}"
                )
            elif _is_rank0 and self._step_count == 0:
                # step=0 时 ETA 无法准确估算, 只显示已用时间
                _progress_info = f" | 进度: 0/{self.state.max_steps} (0.0%) | 已用: {self._format_time(_elapsed)} | 剩余: 估算中..."
            print(f"[rank {_rank}] step={self._step_count} 耗时={_duration:.1f}s{_warn}{_progress_info}", flush=True)

        # ====== 在 optimizer step 边界统一处理日志 (避免梯度累积中间步触发) ======
        # 检查是否刚完成一个完整的 optimizer step
        _grad_accum = self.args.gradient_accumulation_steps
        _is_optim_step = (self._fwd_count % _grad_accum == 0)

        if _is_optim_step:
            # 准备 draft 日志指标
            draft_metrics = getattr(self, '_pending_draft_metrics', {})
            _main_loss_val = self._pending_main_loss.item() if hasattr(self._pending_main_loss, 'item') else 0.0
            _total_loss_val = self._pending_total_loss.item() if hasattr(self._pending_total_loss, 'item') else 0.0

            # 更新滑动窗口 loss 历史
            self._recent_losses.append(_total_loss_val)
            self._recent_main_losses.append(_main_loss_val)
            if len(self._recent_losses) > self._loss_window_size:
                self._recent_losses = self._recent_losses[-self._loss_window_size:]
                self._recent_main_losses = self._recent_main_losses[-self._loss_window_size:]

            if draft_metrics and self._step_count % max(self.args.logging_steps, 1) == 0:
                log_dict = {}
                for key, value in draft_metrics.items():
                    if isinstance(value, (int, float)):
                        log_dict[key] = value
                log_dict['loss/main_loss'] = _main_loss_val
                log_dict['loss/div_loss'] = self._pending_div_loss.item() if hasattr(self._pending_div_loss, 'item') else 0.0
                log_dict['loss/commit_loss'] = self._pending_commit_loss.item() if hasattr(self._pending_commit_loss, 'item') else 0.0
                log_dict['loss/grounding_loss'] = self._pending_grounding_loss.item() if hasattr(self._pending_grounding_loss, 'item') else 0.0
                log_dict['loss/total_loss'] = _total_loss_val

                # ====== 新增: 样本级统计指标记录到 TensorBoard ======
                _accum = self._sample_stats_accum
                if _accum.get('think_chain_sources'):
                    sources = _accum['think_chain_sources']
                    total_samples = len(sources)
                    if total_samples > 0:
                        _n_correct = sum(1 for s in sources if s in ('dataset_converted', 'free_reasoning', 'corrected_free_reasoning', 'correct_cot'))
                        _n_wrong = sum(1 for s in sources if s == 'wrong_cot')
                        log_dict['data/correct_cot_ratio'] = _n_correct / total_samples
                        log_dict['data/wrong_cot_ratio'] = _n_wrong / total_samples
                        log_dict['data/batch_size_effective'] = float(total_samples)
                if _accum.get('valid_tokens'):
                    log_dict['data/valid_tokens_per_sample'] = sum(_accum['valid_tokens']) / max(len(_accum['valid_tokens']), 1)
                    log_dict['data/total_tokens_per_sample'] = sum(_accum['total_tokens']) / max(len(_accum['total_tokens']), 1)
                    log_dict['data/token_utilization'] = sum(_accum['valid_tokens']) / max(sum(_accum['total_tokens']), 1)
                if _accum.get('num_supervised_think_tokens'):
                    log_dict['data/supervised_think_tokens_mean'] = sum(_accum['num_supervised_think_tokens']) / max(len(_accum['num_supervised_think_tokens']), 1)
                if _accum.get('seq_lens'):
                    log_dict['data/seq_len_mean'] = sum(_accum['seq_lens']) / max(len(_accum['seq_lens']), 1)
                    log_dict['data/seq_len_max'] = max(_accum['seq_lens'])

                # 学习率记录
                if self.optimizer is not None:
                    for i, pg in enumerate(self.optimizer.param_groups):
                        log_dict[f'optim/lr_group{i}'] = pg['lr']

                # Loss 趋势指标 (滑动窗口)
                if len(self._recent_losses) >= 10:
                    _recent_half = len(self._recent_losses) // 2
                    _first_half_mean = sum(self._recent_losses[:_recent_half]) / _recent_half
                    _second_half_mean = sum(self._recent_losses[_recent_half:]) / (len(self._recent_losses) - _recent_half)
                    log_dict['loss/trend_ratio'] = _second_half_mean / max(_first_half_mean, 1e-8)  # <1 表示下降, >1 表示上升
                    # Loss 标准差 (衡量震荡程度)
                    _mean = sum(self._recent_losses) / len(self._recent_losses)
                    _var = sum((x - _mean) ** 2 for x in self._recent_losses) / len(self._recent_losses)
                    log_dict['loss/std'] = _var ** 0.5

                self.log(log_dict)

                # ====== 精简 console 摘要 (每 logging_steps 打印一行) ======
                if self._is_main_process():
                    _summary_parts = [f"step={self._step_count}"]
                    _summary_parts.append(f"loss={_total_loss_val:.3f}")
                    _summary_parts.append(f"ce={_main_loss_val:.3f}")
                    _commit = log_dict.get('loss/commit_loss', 0)
                    _ground = log_dict.get('loss/grounding_loss', 0)
                    if _commit > 0:
                        _summary_parts.append(f"commit={_commit:.3f}")
                    if _ground > 0:
                        _summary_parts.append(f"ground={_ground:.3f}")
                    _t1 = log_dict.get('acc/top1_overall', -1)
                    _a1 = log_dict.get('acc/top1_answer', -1)
                    if _t1 >= 0:
                        _summary_parts.append(f"top1={_t1*100:.1f}%")
                    if _a1 >= 0:
                        _summary_parts.append(f"ans={_a1*100:.1f}%")
                    _rank_d = log_dict.get('draft/Zd_effective_rank', -1)
                    if _rank_d >= 0:
                        _summary_parts.append(f"rank={_rank_d:.1f}")
                    _t_rank = log_dict.get('rank/T_rank_last', -1)
                    if _t_rank >= 0:
                        _summary_parts.append(f"T_rank={_t_rank:.1f}")
                    _rank_trend = log_dict.get('rank/Zd_rank_trend', None)
                    if _rank_trend is not None:
                        _trend_icon = "⬆" if _rank_trend > 0.5 else ("⬇" if _rank_trend < -0.5 else "➡")
                        _summary_parts.append(f"rankΔ={_rank_trend:+.1f}{_trend_icon}")
                    _cosim = log_dict.get('draft/Zd_slot_cosim', -1)
                    if _cosim >= 0:
                        _summary_parts.append(f"cosim={_cosim:.3f}")
                    print(f"  📊 {' | '.join(_summary_parts)}")

                    # 第二行: 隐空间流监控信号
                    _rv_parts = []
                    _zd_sc = log_dict.get('draft/Zd_Sc_cosim', -1)
                    if _zd_sc >= 0:
                        _rv_parts.append(f"Zd↔Sc={_zd_sc:.3f}")
                    _zd_norm = log_dict.get('draft/Zd_norm', -1)
                    if _zd_norm >= 0:
                        _rv_parts.append(f"Zd_norm={_zd_norm:.2f}")
                    _sc_norm = log_dict.get('draft/Sc_norm', -1)
                    if _sc_norm >= 0:
                        _rv_parts.append(f"Sc_norm={_sc_norm:.2f}")
                    _delta = log_dict.get('draft/Zd_relative_delta', -1)
                    if _delta >= 0:
                        _rv_parts.append(f"Δ={_delta:.3f}")
                    _ss_mean = log_dict.get('draft/slot_scale_mean', -1)
                    if _ss_mean >= -0.5:  # slot_scale 可能为负, 用 -0.5 作为无效判断
                        _ss_max = log_dict.get('draft/slot_scale_max', 0)
                        _rv_parts.append(f"γ={_ss_mean:.3f}(max={_ss_max:.3f})")
                    _ema_alpha = log_dict.get('draft/trace_ema_alpha_mean', -1)
                    if _ema_alpha >= 0:
                        _rv_parts.append(f"α={_ema_alpha:.3f}")
                    if _rv_parts:
                        print(f"  🌊 {' | '.join(_rv_parts)}")

                    # 第三行: slot 坍塌早期预警 (cosim 过高 + rank 过低)
                    _warn_parts = []
                    if _cosim >= 0 and _cosim > 0.95:
                        _warn_parts.append(f"cosim={_cosim:.3f}>0.95")
                    if _rank_d >= 0 and _rank_d < 8:
                        _warn_parts.append(f"rank={_rank_d:.1f}<8")
                    _zd_norm_val = log_dict.get('draft/Zd_norm', -1)
                    if _zd_norm_val > 50:
                        _warn_parts.append(f"Zd_norm={_zd_norm_val:.1f}>50")
                    _rank_trend_val = log_dict.get('rank/Zd_rank_trend', None)
                    if _rank_trend_val is not None and _rank_trend_val < -2.0:
                        _warn_parts.append(f"rankΔ={_rank_trend_val:+.1f} (秩快速下降!)")
                    _cosim_trend_val = log_dict.get('rank/Zd_cosim_trend', None)
                    if _cosim_trend_val is not None and _cosim_trend_val > 0.1:
                        _warn_parts.append(f"cosimΔ={_cosim_trend_val:+.3f} (趋向坍塌!)")
                    if _warn_parts:
                        print(f"  ⚠️ slot 坍塌风险: {' | '.join(_warn_parts)} — 检查 diversity loss 是否生效")

                    # 秩快照 (每 rank_monitor_every_n_steps 步打印, 比详细监控更频繁)
                    if self._step_count > 0 and self._step_count % self.rank_monitor_every_n_steps == 0:
                        _rank_snap_parts = []
                        _r_init = log_dict.get('rank/Zd_init_rank', -1)
                        _r_first = log_dict.get('rank/Zd_rank_first', -1)
                        _r_last = log_dict.get('rank/Zd_rank_last', -1)
                        _r_trend = log_dict.get('rank/Zd_rank_trend', None)
                        _c_first = log_dict.get('rank/Zd_cosim_first', -1)
                        _c_last = log_dict.get('rank/Zd_cosim_last', -1)
                        _t_rank = log_dict.get('rank/T_rank_last', -1)
                        if _r_init >= 0:
                            _rank_snap_parts.append(f"init={_r_init:.1f}")
                        if _r_first >= 0:
                            _rank_snap_parts.append(f"step1={_r_first:.1f}")
                        if _r_last >= 0:
                            _rank_snap_parts.append(f"last={_r_last:.1f}")
                        if _r_trend is not None:
                            _trend_icon = "⬆" if _r_trend > 0.5 else ("⬇" if _r_trend < -0.5 else "➡")
                            _rank_snap_parts.append(f"Δ={_r_trend:+.1f}{_trend_icon}")
                        if _c_first >= 0 and _c_last >= 0:
                            _rank_snap_parts.append(f"cosim:{_c_first:.3f}→{_c_last:.3f}")
                        if _t_rank >= 0:
                            _rank_snap_parts.append(f"T={_t_rank:.1f}")
                        if _rank_snap_parts:
                            print(f"  🔍 秩快照: {' | '.join(_rank_snap_parts)}")

            # 保存样本统计副本 (供详细监控打印使用)
            _sample_stats_snapshot = dict(self._sample_stats_accum)
            # 清空样本统计累加器
            self._sample_stats_accum = defaultdict(list)

            # 准备详细监控信息
            if self._step_count > 0 and self._step_count % self.monitor_every_n_steps == 0 and self._is_main_process():
                main_loss = getattr(self, '_pending_main_loss', torch.tensor(0.0))
                div_loss = getattr(self, '_pending_div_loss', torch.tensor(0.0))
                commit_loss = getattr(self, '_pending_commit_loss', torch.tensor(0.0))
                grounding_loss = getattr(self, '_pending_grounding_loss', torch.tensor(0.0))
                total_loss = getattr(self, '_pending_total_loss', torch.tensor(0.0))
                lines = []
                lines.append(f"\n{'='*70}")
                lines.append(f"[RLD Step {self._step_count}] Loss 概览 (fwd#{self._fwd_count})")
                lines.append(f"  total_loss={total_loss.item():.4f}, "
                      f"main_loss={main_loss.item():.4f}, "
                      f"div_loss={div_loss.item():.4f}, "
                      f"commit_loss={commit_loss.item():.4f}, "
                      f"grounding_loss={grounding_loss.item():.4f}")

                # ====== 新增: Loss 趋势分析 ======
                if len(self._recent_losses) >= 10:
                    _recent_half = len(self._recent_losses) // 2
                    _first_half_mean = sum(self._recent_losses[:_recent_half]) / _recent_half
                    _second_half_mean = sum(self._recent_losses[_recent_half:]) / (len(self._recent_losses) - _recent_half)
                    _trend = _second_half_mean / max(_first_half_mean, 1e-8)
                    _mean = sum(self._recent_losses) / len(self._recent_losses)
                    _var = sum((x - _mean) ** 2 for x in self._recent_losses) / len(self._recent_losses)
                    _std = _var ** 0.5
                    if _trend > 1.1:
                        _trend_status = "❌ 上升 (可能发散)"
                    elif _trend > 1.02:
                        _trend_status = "⚠️ 微升"
                    elif _trend < 0.9:
                        _trend_status = "✅ 快速下降"
                    elif _trend < 0.98:
                        _trend_status = "✅ 稳步下降"
                    else:
                        _trend_status = "➡️ 平稳"
                    lines.append(f"  📈 Loss 趋势 (最近{len(self._recent_losses)}步): "
                                 f"前半均值={_first_half_mean:.4f} → 后半均值={_second_half_mean:.4f} "
                                 f"(比值={_trend:.4f}) {_trend_status}")
                    lines.append(f"     Loss 标准差={_std:.4f} "
                                 f"({'⚠️ 震荡较大' if _std > 0.3 else '✅ 稳定'})")

                # ====== 新增: 学习率信息 ======
                if self.optimizer is not None:
                    _lr_info = []
                    for i, pg in enumerate(self.optimizer.param_groups):
                        name = pg.get('name', f'group{i}')
                        _lr_info.append(f"{name}={pg['lr']:.2e}")
                    lines.append(f"  📐 学习率: {', '.join(_lr_info)}")

                # ====== 新增: 样本级统计 ======
                if _sample_stats_snapshot:
                    lines.extend(self._format_sample_monitor(_sample_stats_snapshot))

                if draft_metrics:
                    lines.extend(self._format_draft_monitor(draft_metrics))
                lines.append(f"{'='*70}")
                print("\n".join(lines))

        # 每 monitor_every_n_steps 收集梯度范数 (在 optimizer step 边界)
        if _is_optim_step and self._step_count > 0 and self._step_count % self.monitor_every_n_steps == 0:
            grad_metrics = self._collect_grad_metrics(model)
            if grad_metrics:
                # 记录到 TensorBoard
                self.log(grad_metrics)
                # 控制台打印
                if self._is_main_process():
                    # 区分 controller、prefix_kv_projector 和 lora 的指标
                    ctrl_metrics = {k: v for k, v in sorted(grad_metrics.items())
                                    if not k.startswith('grad/readout') and not k.startswith('grad/lora')}
                    readout_metrics = {k: v for k, v in sorted(grad_metrics.items())
                                       if k.startswith('grad/readout')}
                    lora_metrics = {k: v for k, v in sorted(grad_metrics.items())
                                    if k.startswith('grad/lora')}

                    print(f"\n  🔧 Controller 梯度监控 (via {'hook' if self._grad_hooks_registered else 'param.grad'}):")
                    _has_any_grad = False
                    for key, value in sorted(ctrl_metrics.items()):
                        if value == 0.0:
                            print(f"    {key}: {value:.6f} ❌无梯度")
                        else:
                            print(f"    {key}: {value:.6f} ✅")
                            _has_any_grad = True
                    if readout_metrics:
                        print(f"  🔧 PrefixKVProjector 梯度监控:")
                        for key, value in sorted(readout_metrics.items()):
                            if value == 0.0:
                                print(f"    {key}: {value:.6f} ❌无梯度")
                            else:
                                print(f"    {key}: {value:.6f} ✅")
                                _has_any_grad = True
                    if lora_metrics:
                        print(f"  🔧 LoRA 梯度监控 (Stage 2):")
                        # LoRA 参数较多，聚合显示
                        lora_total_norm = sum(v for v in lora_metrics.values()) ** 0.5
                        lora_nonzero = sum(1 for v in lora_metrics.values() if v > 0)
                        print(f"    LoRA 总梯度范数: {lora_total_norm:.6f} "
                              f"({lora_nonzero}/{len(lora_metrics)} 参数有梯度) "
                              f"{'✅' if lora_nonzero > 0 else '❌无梯度'}")
                        _has_any_grad = _has_any_grad or lora_nonzero > 0
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

        # 为 controller 和 prefix_kv_projector 的所有可训练参数注册 hook
        for prefix, module in [('ctrl', getattr(inner_model, 'controller', None)),
                                ('readout', getattr(inner_model, 'prefix_kv_projector', None))]:
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

        # Stage 2: 为 LoRA 参数注册 hook
        for name, param in inner_model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_' in name:
                full_name = f"lora/{name.split('.')[-2]}_{name.split('.')[-1]}"
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
    收集 controller + prefix_kv_projector 各模块的梯度 L2 范数。
        
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
                                    ('readout', getattr(inner_model, 'prefix_kv_projector', None))]:
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
            
            # Stage 2: 收集 LoRA 参数梯度
            for name, param in inner_model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'lora_' in name:
                    display_name = "lora/aggregated"
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
        自定义优化器: 支持多学习率分组
        
        Stage 1: 只有 Controller + ReadoutAdapter (单一学习率)
        Stage 2: Controller + ReadoutAdapter + LoRA (三组学习率)
          - Controller/Adapter: controller_lr (较小, 从 Stage 1 微调)
          - LoRA: lora_lr (最小, 防止灾难性遗忘)
        
        支持两种模式:
        - DDP 模式 (默认): 直接使用 PyTorch AdamW
        - DeepSpeed 模式: 使用 FusedAdam (如可用) 或 PyTorch AdamW
        """
        model = self.model

        # 收集可训练参数并分组
        # 获取内部模型 (可能被 DeepSpeed/DDP 包裹)
        inner_model = model
        if hasattr(model, 'module'):
            inner_model = model.module

        controller_params = []
        adapter_params = []
        lora_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # 判断参数属于哪个模块
            if 'controller' in name and 'prefix_kv_projector' not in name:
                controller_params.append(param)
            elif 'prefix_kv_projector' in name:
                adapter_params.append(param)
            elif 'lora_' in name or 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
            else:
                other_params.append(param)

        if len(controller_params) + len(adapter_params) + len(lora_params) + len(other_params) == 0:
            raise ValueError("[RLD Trainer] 没有可训练参数！请检查模型冻结设置。")

        # 构建参数分组
        param_groups = []

        # Controller 参数组
        if controller_params:
            param_groups.append({
                "params": controller_params,
                "lr": self.controller_lr,
                "weight_decay": self.args.weight_decay,
                "name": "controller",
            })

        # ReadoutAdapter 参数组 (与 Controller 相同学习率)
        if adapter_params:
            param_groups.append({
                "params": adapter_params,
                "lr": self.controller_lr,
                "weight_decay": self.args.weight_decay,
                "name": "prefix_kv_projector",
            })

        # LoRA 参数组 (更小的学习率)
        if lora_params:
            _lora_lr = self.lora_lr if self.lora_lr is not None else self.controller_lr * 0.3
            param_groups.append({
                "params": lora_params,
                "lr": _lora_lr,
                "weight_decay": self.args.weight_decay,
                "name": "lora",
            })

        # 其他参数 (如果有)
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": self.controller_lr,
                "weight_decay": self.args.weight_decay,
                "name": "other",
            })

        if self._is_main_process():
            print(f"\n  🔧 优化器参数分组:")
            for pg in param_groups:
                n_params = sum(p.numel() for p in pg['params'])
                print(f"    {pg.get('name', 'unnamed')}: {n_params:,} 参数, lr={pg['lr']:.2e}")
            print()

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

    def _format_draft_monitor(self, draft_metrics: dict) -> list:
        """
        格式化 Draft 状态监控信息为行列表。
        
        精简版: 聚焦核心健康指标, 去掉旧架构冗余指标 (inject_weight 分桶等),
        新增双向检索-验证模式的专属监控。
        """
        lines = []
        lines.append(f"\n  📊 Draft 状态监控:")

        # ---- 核心健康指标 (一行汇总) ----
        _health_parts = []
        if 'draft/Zd_effective_rank' in draft_metrics:
            _health_parts.append(f"rank={draft_metrics['draft/Zd_effective_rank']:.2f}")
        if 'rank/Zd_init_rank' in draft_metrics:
            _health_parts.append(f"init_rank={draft_metrics['rank/Zd_init_rank']:.2f}")
        if 'draft/Zd_slot_cosim' in draft_metrics:
            cosim = draft_metrics['draft/Zd_slot_cosim']
            status = "✅" if cosim < 0.5 else ("⚠️" if cosim < 0.8 else "❌坍塌")
            _health_parts.append(f"cosim={cosim:.4f}{status}")
        if 'draft/Zd_norm' in draft_metrics:
            _health_parts.append(f"norm={draft_metrics['draft/Zd_norm']:.2f}")
        if 'draft/num_steps' in draft_metrics:
            _health_parts.append(f"steps={int(draft_metrics['draft/num_steps'])}")
        if _health_parts:
            lines.append(f"    Z_d: {', '.join(_health_parts)}")

        # ---- 方案 D: 隐空间流监控 ----
        # LayerScale: per-slot 可学习缩放因子 γ_i
        if 'draft/slot_scale_mean' in draft_metrics:
            _ss_mean = draft_metrics['draft/slot_scale_mean']
            _ss_max = draft_metrics['draft/slot_scale_max']
            _ss_min = draft_metrics['draft/slot_scale_min']
            _ss_std = draft_metrics['draft/slot_scale_std']
            lines.append(f"    LayerScale γ: mean={_ss_mean:.4f}, range=[{_ss_min:.4f}, {_ss_max:.4f}], std={_ss_std:.4f}")

        # TraceEMA: per-slot 可学习记忆保留率 α_i
        if 'draft/trace_ema_alpha_mean' in draft_metrics:
            _a_mean = draft_metrics['draft/trace_ema_alpha_mean']
            _a_max = draft_metrics['draft/trace_ema_alpha_max']
            _a_min = draft_metrics['draft/trace_ema_alpha_min']
            _a_std = draft_metrics['draft/trace_ema_alpha_std']
            lines.append(f"    TraceEMA α: mean={_a_mean:.4f}, range=[{_a_min:.4f}, {_a_max:.4f}], std={_a_std:.4f}")

        if 'draft/Zd_Sc_cosim' in draft_metrics:
            _zd_sc = draft_metrics['draft/Zd_Sc_cosim']
            if _zd_sc > 0.95:
                _zd_status = "⚠️ Z_d≈S_c (流未产生差异化信号)"
            elif _zd_sc > 0.7:
                _zd_status = "✅ 适度差异 (流生效)"
            elif _zd_sc > 0.3:
                _zd_status = "➡️ 较大差异 (强融合信号)"
            else:
                _zd_status = "⚠️ Z_d 与 S_c 差异过大"
            lines.append(f"    Z_d↔S_c 余弦相似度: {_zd_sc:.4f} {_zd_status}")
        if 'draft/Sc_norm' in draft_metrics:
            _sc_norm = draft_metrics['draft/Sc_norm']
            _sc_rank = draft_metrics.get('draft/Sc_effective_rank', -1)
            _sc_cosim = draft_metrics.get('draft/Sc_slot_cosim', -1)
            _sc_parts = [f"范数={_sc_norm:.2f}"]
            if _sc_rank >= 0:
                _sc_rank_status = "✅" if _sc_rank > 3.0 else "⚠️"
                _sc_parts.append(f"rank={_sc_rank:.2f}{_sc_rank_status}")
            if _sc_cosim >= 0:
                _sc_parts.append(f"cosim={_sc_cosim:.4f}")
            lines.append(f"    S_c: {', '.join(_sc_parts)}")

        # Z_e / T 有效秩 (辅助, 仅旧模式下有 Z_e 指标)
        _aux_parts = []
        if 'draft/Ze_effective_rank' in draft_metrics:
            _aux_parts.append(f"Z_e rank={draft_metrics['draft/Ze_effective_rank']:.2f}")
        if 'draft/T_effective_rank' in draft_metrics:
            _aux_parts.append(f"T rank={draft_metrics['draft/T_effective_rank']:.2f}")
        if 'rank/T_rank_last' in draft_metrics:
            _aux_parts.append(f"T rank(last)={draft_metrics['rank/T_rank_last']:.2f}")
        if 'draft/Zd_relative_delta' in draft_metrics:
            delta = draft_metrics['draft/Zd_relative_delta']
            status = "✅" if 0.05 <= delta <= 0.3 else ("⚠️过小" if delta < 0.05 else "⚠️过大")
            _aux_parts.append(f"Δ={delta:.4f}{status}")
        if _aux_parts:
            lines.append(f"    辅助: {', '.join(_aux_parts)}")

        # ====== 秩演化追踪 (per-step) ======
        _per_step_ranks = draft_metrics.get('rank/_per_step_ranks', [])
        if _per_step_ranks:
            lines.append(f"\n  📊 秩演化追踪 ({len(_per_step_ranks)} steps):")
            # 每步的秩和 cosim
            _step_strs = []
            for i, r in enumerate(_per_step_ranks):
                _rank_icon = "✅" if r['Zd_rank'] >= 8 else ("⚠️" if r['Zd_rank'] >= 4 else "❌")
                _step_strs.append(
                    f"step{i+1}: rank={r['Zd_rank']:.1f}{_rank_icon} cosim={r['Zd_cosim']:.3f} T_rank={r['T_rank']:.1f} norm={r['Zd_norm']:.1f}"
                )
            for s in _step_strs:
                lines.append(f"    {s}")
            # 秩趋势汇总
            _zd_ranks = [r['Zd_rank'] for r in _per_step_ranks]
            _trend = _zd_ranks[-1] - _zd_ranks[0]
            if _trend < -2.0:
                _trend_status = "❌ 秩快速下降 (坍塌风险!)"
            elif _trend < -0.5:
                _trend_status = "⚠️ 秩缓慢下降"
            elif _trend > 0.5:
                _trend_status = "✅ 秩增长 (表达丰富化)"
            else:
                _trend_status = "➡️ 秩稳定"
            _init_rank = draft_metrics.get('rank/Zd_init_rank', -1)
            _init_str = f", init_rank={_init_rank:.1f}" if _init_rank >= 0 else ""
            lines.append(f"    趋势: {_zd_ranks[0]:.1f}→{_zd_ranks[-1]:.1f} (Δ={_trend:+.1f}) {_trend_status}{_init_str}")

        # ====== 方案 D: 隐空间流监控 ======
        if 'draft/visual_proj_num_tokens' in draft_metrics:
            lines.append(f"\n  🌊 隐空间流 (LatentDraftFlow):")
            _vp_n = int(draft_metrics['draft/visual_proj_num_tokens'])
            _vp_norm = draft_metrics.get('draft/visual_proj_norm', 0.0)
            lines.append(f"    visual_hidden_proj: {_vp_n} tokens, norm={_vp_norm:.4f}")
            if 'draft/Zd_Sc_cosim' in draft_metrics:
                _zd_sc = draft_metrics['draft/Zd_Sc_cosim']
                if _zd_sc > 0.95:
                    _zd_status = "⚠️ Z_d≈S_c (流未产生差异化信号)"
                elif _zd_sc > 0.7:
                    _zd_status = "✅ 适度差异 (流生效)"
                elif _zd_sc > 0.3:
                    _zd_status = "➡️ 较大差异 (强融合信号)"
                else:
                    _zd_status = "⚠️ Z_d 与 S_c 差异过大"
                lines.append(f"    Z_d↔S_c 余弦相似度: {_zd_sc:.4f} {_zd_status}")
            if 'draft/Sc_norm' in draft_metrics:
                _sc_norm = draft_metrics['draft/Sc_norm']
                _sc_rank = draft_metrics.get('draft/Sc_effective_rank', -1)
                _sc_cosim = draft_metrics.get('draft/Sc_slot_cosim', -1)
                _sc_parts = [f"范数={_sc_norm:.2f}"]
                if _sc_rank >= 0:
                    _sc_rank_status = "✅" if _sc_rank > 3.0 else "⚠️"
                    _sc_parts.append(f"rank={_sc_rank:.2f}{_sc_rank_status}")
                if _sc_cosim >= 0:
                    _sc_parts.append(f"cosim={_sc_cosim:.4f}")
                lines.append(f"    S_c: {', '.join(_sc_parts)}")

        # ====== Prefix KV 注入效果 (精简) ======
        if 'draft/adapted_hidden_norm' in draft_metrics:
            lines.append(f"\n  🎯 注入效果: adapted_hidden_norm={draft_metrics['draft/adapted_hidden_norm']:.2f}")

        # ====== Per-token Loss 分解 (think vs answer, 精简为一行) ======
        if 'loss/think_ce' in draft_metrics and 'loss/answer_ce' in draft_metrics:
            _think_ce = draft_metrics['loss/think_ce']
            _think_cnt = int(draft_metrics.get('loss/think_token_count', 0))
            _answer_ce = draft_metrics['loss/answer_ce']
            _answer_cnt = int(draft_metrics.get('loss/answer_token_count', 0))
            _ratio = _answer_ce / max(_think_ce, 1e-8)
            if _ratio > 2.0:
                _status = "⚠️答案不确定"
            elif _ratio < 0.5:
                _status = "✅答案学习良好"
            else:
                _status = "➡️"
            lines.append(f"\n  📝 CE分解: think={_think_ce:.4f}({_think_cnt}t) answer={_answer_ce:.4f}({_answer_cnt}t) ratio={_ratio:.2f} {_status}")
        elif 'loss/think_ce' in draft_metrics:
            lines.append(f"\n  📝 Think CE: {draft_metrics['loss/think_ce']:.4f} ({int(draft_metrics.get('loss/think_token_count', 0))}t)")

        # ====== Top-K 准确率 (精简为一行) ======
        if 'acc/top1_overall' in draft_metrics:
            _t1 = draft_metrics.get('acc/top1_overall', 0)
            _t1_think = draft_metrics.get('acc/top1_think', -1)
            _a1 = draft_metrics.get('acc/top1_answer', -1)
            _a5 = draft_metrics.get('acc/top5_answer', -1)
            _parts = [f"overall={_t1*100:.1f}%"]
            if _t1_think >= 0:
                _parts.append(f"think={_t1_think*100:.1f}%")
            if _a1 >= 0:
                _status = "✅" if _a1 > 0.5 else ("⚠️" if _a1 > 0.2 else "❌")
                _parts.append(f"answer={_a1*100:.1f}%{_status}")
                if _a5 >= 0:
                    _parts.append(f"ans@5={_a5*100:.1f}%")
            lines.append(f"  🎯 Top-1: {', '.join(_parts)}")

        # [Contrastive Draft Learning: 已弃用]

        return lines

    def _format_sample_monitor(self, sample_stats: dict) -> list:
        """
        格式化样本级统计信息为行列表。
        
        包括: 样本类型分布、有效 token 统计、序列长度分布等。
        """
        lines = []
        lines.append(f"\n  📋 样本级统计 (本轮梯度累积):")

        # 样本类型分布
        sources = sample_stats.get('think_chain_sources', [])
        if sources:
            from collections import Counter
            src_counts = Counter(sources)
            total = len(sources)
            lines.append(f"    样本总数: {total}")
            for src, cnt in src_counts.most_common():
                pct = cnt / total * 100
                lines.append(f"      {src}: {cnt} ({pct:.1f}%)")

        # 有效 token 统计
        valid_tokens = sample_stats.get('valid_tokens', [])
        total_tokens = sample_stats.get('total_tokens', [])
        if valid_tokens and total_tokens:
            _valid_sum = sum(valid_tokens)
            _total_sum = sum(total_tokens)
            _utilization = _valid_sum / max(_total_sum, 1) * 100
            lines.append(f"    有效 token: {_valid_sum}/{_total_sum} (利用率={_utilization:.1f}%)")
            lines.append(f"    每样本有效 token: {_valid_sum / len(valid_tokens):.1f}")

        # Think 监督 token 统计
        sup_think = sample_stats.get('num_supervised_think_tokens', [])
        if sup_think:
            _sup_mean = sum(sup_think) / len(sup_think)
            _sup_nonzero = sum(1 for x in sup_think if x > 0)
            lines.append(f"    Think 监督 token 均值: {_sup_mean:.1f} "
                         f"(有监督的样本: {_sup_nonzero}/{len(sup_think)})")

        # Answer 权重统计
        answer_weights = sample_stats.get('answer_weights', [])
        if answer_weights:
            _aw_mean = sum(answer_weights) / len(answer_weights)
            _aw_max = max(answer_weights)
            _aw_min = min(answer_weights)
            lines.append(f"    Answer 权重: 均值={_aw_mean:.2f}, 范围=[{_aw_min:.2f}, {_aw_max:.2f}]")

        # 序列长度统计
        seq_lens = sample_stats.get('seq_lens', [])
        if seq_lens:
            _sl_mean = sum(seq_lens) / len(seq_lens)
            _sl_max = max(seq_lens)
            _sl_min = min(seq_lens)
            lines.append(f"    序列长度: 均值={_sl_mean:.0f}, 范围=[{_sl_min}, {_sl_max}]")

        return lines

    def _is_main_process(self) -> bool:
        """判断是否为主进程"""
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        return local_rank in (-1, 0)
