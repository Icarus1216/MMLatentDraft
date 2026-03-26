"""
RLD Controller: 编排整个 Reflective Latent Draft 的生命周期

核心职责:
1. Prefill 阶段: 从视觉 token 初始化冻结证据 Z^e，初始化草稿 Z^d_0 和轨迹 T_0
2. Step 边界更新: 执行 S_c → T_c → G_c → Z^d_c 的完整更新链路
   - T_c 只用于计算 G_c 和提供全局方向 bias，不再作为 DraftUpdater 的 KV
   - TraceUpdater 和 ReflectionModule 均采用 2 层 CA
3. Draft 通过 ReadoutAdapter 注入 frozen hidden states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from .modules import (
    CrossAttentionBlock,
    EvidenceResampler,
    StepResampler,
    TraceUpdater,
    ReflectionModule,
    DraftUpdater,
    ResidualFlowDraftUpdater,
    DiversityRegularizer,
    RMSNorm,
)


class RLDController(nn.Module):
    """
    RLD 控制器
    
    维护三个核心状态:
    - Z^e: 冻结视觉证据槽 [B, K_e, d_z] (推理过程中不更新)
    - Z^d: 可擦写草稿槽 [B, K_d, d_z] (每个 step 边界更新)
    - T: 累计推理轨迹 [B, K_t, d_z] (每个 step 边界更新)
    
    Args:
    hidden_size: Qwen3-VL 的隐藏维度 (4096 for 8B)
        d_z: controller 空间维度 (512)
        num_evidence_slots: 证据槽数量 K_e (默认 16)
        num_draft_slots: 草稿槽数量 K_d (默认 16)
        num_trace_slots: 轨迹槽数量 K_t (默认 16)
        total_layers: 基座模型总层数 (36 for Qwen3-VL-8B)
        num_heads: cross-attention 头数 (默认 8)
        evidence_layers: evidence resampler 的层数 (默认 2)
        use_gate: [已废弃] 保留用于兼容旧配置
        lambda_div: 多样性正则化权重 (默认 0.01)
        max_steps: 最大 step 数, 用于 ResidualFlowDraftUpdater 的 Δt 计算 (默认 14)
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        d_z: int = 768,
        num_evidence_slots: int = 16,
        num_draft_slots: int = 16,
        num_trace_slots: int = 16,
        total_layers: int = 36,
        num_heads: int = 8,
        evidence_layers: int = 2,
        use_gate: bool = True,  # [已废弃] 保留用于兼容旧配置, 不再使用
        lambda_div: float = 0.01,
        max_steps: int = 14,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.K_e = num_evidence_slots
        self.K_d = num_draft_slots
        self.K_t = num_trace_slots
        self.total_layers = total_layers
        self.lambda_div = lambda_div
        self.max_steps = max_steps

        # ====== 核心模块 ======

        # 1. 证据重采样器: V → Z^e
        self.evidence_resampler = EvidenceResampler(
            hidden_size=hidden_size,
            d_z=d_z,
            num_evidence_slots=num_evidence_slots,
            num_heads=num_heads,
            num_layers=evidence_layers,
        )

        # 2. 步摘要重采样器: H^step → S_c
        self.step_resampler = StepResampler(
            hidden_size=hidden_size,
            d_z=d_z,
            num_trace_slots=num_trace_slots,
            num_heads=num_heads,
        )

        # 3. 轨迹更新器: [T_{c-1}; S_c] → T_c (2 层 CA，防止长链推理记忆退化)
        self.trace_updater = TraceUpdater(
            d_z=d_z,
            num_trace_slots=num_trace_slots,
            num_heads=num_heads,
            num_layers=2,
        )

        # 4. 回看-验证模块: (T_c, Z^e) → G_c (2 层 CA，retrieval + verification)
        self.reflection = ReflectionModule(
            d_z=d_z,
            num_heads=num_heads,
            num_layers=2,
        )

        # 5. 草稿更新器 (Neural ODE / Residual Flow): (Z^d, T_c, G_c, t) → Z^d_{c+1}
        #    替代旧版门控 DraftUpdater, 去掉 sigmoid 衰减, 用残差步进
        self.draft_updater = ResidualFlowDraftUpdater(
            d_z=d_z,
            num_heads=num_heads,
            max_steps=max_steps,
        )

        # ====== 初始化相关 ======

        # 可学习的草稿初始状态 (正交初始化, 保证 slot 间余弦相似度 ≈ 0)
        _draft_init_data = torch.empty(num_draft_slots, d_z)
        nn.init.orthogonal_(_draft_init_data)
        # 缩放范数到与 Z_e 同量级 (~1.0), 保证残差连接后正交差异不被 CA 共同分量淹没
        self.draft_init = nn.Parameter(_draft_init_data.unsqueeze(0))  # [1, K_d, d_z], 每行范数≈1.0

        # Per-slot 草稿条件化: 每个 draft slot 通过 cross-attention 从 Z_e 获取独立信息
        # (替代旧版的 Z_e.mean 广播, 消除 slot 坍塌的根因)
        self.draft_init_ca = CrossAttentionBlock(
            d_model=d_z,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )

        # 多样性正则化 (threshold=0.3: 允许 slot 间有适度相似性, 避免过度正则化)
        self.diversity_reg = DiversityRegularizer(threshold=0.3)

    def prefill(
        self,
        visual_tokens: torch.Tensor,   # [B, L_v, hidden_size]
    ) -> Dict[str, torch.Tensor]:
        """
        Prefill 阶段: 初始化所有外挂状态
        
        Args:
            visual_tokens: 来自视觉编码器的 token 序列
        
        Returns:
            state: {
                'Z_e': [B, K_e, d_z],    # 冻结证据槽
                'Z_d': [B, K_d, d_z],    # 初始草稿
                'T': [B, K_t, d_z],      # 初始轨迹 (零)
                'step_count': 0,
            }
        """
        B = visual_tokens.shape[0]
        device = visual_tokens.device
        dtype = visual_tokens.dtype

        # 1. 生成冻结证据
        Z_e = self.evidence_resampler(visual_tokens)  # [B, K_e, d_z]

        # 2. 初始化草稿 (per-slot 条件化: 每个 slot 从 Z_e 获取独立信息)
        draft_init = self.draft_init.expand(B, -1, -1).to(device=device, dtype=dtype)
        # Per-slot cross-attention: Q=draft_init (正交), KV=Z_e
        # 每个 draft slot 的 query 向量不同 → attention 分布不同 → 获取不同的证据信息
        Z_d = self.draft_init_ca(query=draft_init, key_value=Z_e)  # [B, K_d, d_z]

        # 3. 初始化轨迹为零
        T = torch.zeros(B, self.K_t, self.d_z, device=device, dtype=dtype)

        return {
            'Z_e': Z_e,
            'Z_d': Z_d,
            'T': T,
            'step_count': 0,
        }

    def step_update(
        self,
        state: Dict[str, torch.Tensor],
        step_hidden_states: torch.Tensor,   # [B, L_c, hidden_size]
        update_mask: Optional[torch.Tensor] = None,  # [B] bool, P0-3: per-sample update mask
    ) -> Dict[str, torch.Tensor]:
        """
        Step 边界更新: 执行完整的 "摘要 → 轨迹 → 回看 → 草稿" 更新链路
        
        信息流设计:
        - T_c 只用于计算 G_c（作为 query 去检索证据）+ 提供全局方向 bias
        - T_c 不再作为 DraftUpdater 的 KV（G_c 已是 T_c 的信息超集）
        - DraftUpdater 的 KV 只有 G_c，T_c 通过 additive bias 注入推理方向
        
        P0-3: 支持 per-sample update_mask。
        当 batch > 1 时，只有 update_mask[b] == True 的样本执行更新，
        其他样本保持状态不变。
        
        Args:
            state: 当前状态字典
            step_hidden_states: 当前 step 的最后层 hidden states
            update_mask: [B] bool tensor, True 表示该样本在此位置遇到了 delimiter
                         如果为 None，所有样本都更新
        
        Returns:
            state: 更新后的状态字典
        """
        Z_e = state['Z_e']
        Z_d = state['Z_d']
        T_prev = state['T']

        # 1. 步摘要: H^step → S_c
        S_c = self.step_resampler(step_hidden_states)  # [B, K_t, d_z]

        # 2. 累计轨迹更新 (2层CA): [T_{c-1}; S_c] → T_c
        T_c = self.trace_updater(T_prev, S_c)  # [B, K_t, d_z]

        # 3. 回看-验证 (2层CA): (T_c, Z^e) → G_c
        G_c = self.reflection(T_c, Z_e)  # [B, K_t, d_z]

        # 4. 草稿更新: Residual Flow step
        #    传入 step_progress, 让 updater 感知推理阶段
        step_progress = state['step_count'] / max(self.max_steps, 1)
        Z_d_new = self.draft_updater(Z_d, T_c, G_c, step_progress=step_progress)  # [B, K_d, d_z]

        # P0-3: 根据 update_mask 选择性更新
        if update_mask is not None:
            # update_mask: [B] bool → [B, 1, 1] 用于广播
            mask = update_mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            # 未触发的样本保持原状态
            Z_d_new = mask * Z_d_new + (1.0 - mask) * Z_d
            T_c = mask * T_c + (1.0 - mask) * T_prev

        new_state = {
            'Z_e': Z_e,  # 不变
            'Z_d': Z_d_new,
            'T': T_c,
            'step_count': state['step_count'] + 1,
        }

        # ====== 监控指标 (detach，不影响梯度) ======
        with torch.no_grad():
            # Z_d 段间变化率
            Zd_delta = (Z_d_new.detach() - Z_d.detach()).norm(dim=-1).mean()
            Zd_base = Z_d.detach().norm(dim=-1).mean().clamp(min=1e-8)
            new_state['_monitor'] = {
                'Zd_abs_delta': Zd_delta.item(),
                'Zd_relative_delta': (Zd_delta / Zd_base).item(),
                # Z_d 槽间余弦相似度 (反映是否坍塌)
                'Zd_slot_cosim': self._slot_cosine_similarity(Z_d_new.detach()),
                # Z_d 有效秩 (反映表达多样性)
                'Zd_effective_rank': self._effective_rank(Z_d_new.detach()),
                # T 轨迹有效秩
                'T_effective_rank': self._effective_rank(T_c.detach()),
                # Z_d 槽的 L2 范数均值 (反映 draft 信号强度)
                'Zd_norm': Z_d_new.detach().norm(dim=-1).mean().item(),
            }

        return new_state

    def scan_steps(
        self,
        state: Dict[str, torch.Tensor],
        step_summaries: List[torch.Tensor],  # list of [B, K_t, d_z], 长度 = C
        update_masks: List[torch.Tensor],    # list of [B] bool, 长度 = C
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        沿 step 维度扫描: 固定步数循环得到所有 step 的 Z_d_c
        
        这是重构的核心方法: 将 step_update 从散落在段循环里改为一个独立的 for 循环。
        所有 rank 执行相同次数 (C_max)，用 update_mask pad 保证跨 rank 一致。
        
        Args:
            state: 初始 RLD 状态 (来自 prefill)
            step_summaries: list of [B, K_t, d_z], 每个 step 的摘要 S_c
            update_masks: list of [B] bool, 每个 step 的 per-sample 更新掩码
        
        Returns:
            final_state: 最终的 RLD 状态
            all_Z_d: list of [B, K_d, d_z], 每个 step 结束后的 draft state
                      长度 = C + 1 (包含初始 Z_d_0)
        """
        all_Z_d = [state['Z_d']]  # 初始 Z_d_0

        for c in range(len(step_summaries)):
            S_c = step_summaries[c]
            mask = update_masks[c]

            # 复用 step_update 的核心逻辑
            Z_e = state['Z_e']
            Z_d = state['Z_d']
            T_prev = state['T']

            # 1. 累计轨迹更新 (2层CA): [T_{c-1}; S_c] → T_c
            T_c = self.trace_updater(T_prev, S_c)

            # 2. 回看-验证 (2层CA): (T_c, Z^e) → G_c
            G_c = self.reflection(T_c, Z_e)

            # 3. 草稿更新: Residual Flow step
            #    传入 step_progress = c / max(C, 1), 让 updater 感知推理阶段
            step_progress = c / max(len(step_summaries), 1)
            Z_d_new = self.draft_updater(Z_d, T_c, G_c, step_progress=step_progress)

            # 4. 根据 update_mask 选择性更新
            if mask is not None:
                float_mask = mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                Z_d_new = float_mask * Z_d_new + (1.0 - float_mask) * Z_d
                T_c = float_mask * T_c + (1.0 - float_mask) * T_prev

            state = {
                'Z_e': Z_e,
                'Z_d': Z_d_new,
                'T': T_c,
                'step_count': state['step_count'] + 1,
            }

            all_Z_d.append(Z_d_new)

        return state, all_Z_d

    def compute_diversity_loss(
        self,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算多样性正则化损失
        
        同时对 Z^d 和 Z^e 进行去相关正则化:
        - Z_e 是坍塌的源头，必须从源头治理
        - Z_d 是直接被使用的 slot，也需要正则化
        """
        Z_d = state['Z_d']
        Z_e = state['Z_e']
        # Z_d 和 Z_e 各占一半权重
        loss_d = self.diversity_reg(Z_d)
        loss_e = self.diversity_reg(Z_e)
        loss = (loss_d + loss_e) * self.lambda_div
        return loss

    @torch.no_grad()
    def compute_draft_metrics(
        self,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        计算 draft 状态的综合监控指标
        
        Args:
            state: 当前 RLD 状态
        
        Returns:
            metrics: 指标字典 (所有值都是 float，已 detach)
        """
        metrics = {}
        Z_d = state['Z_d'].detach()
        Z_e = state['Z_e'].detach()

        # 1. Z_d 有效秩
        metrics['draft/Zd_effective_rank'] = self._effective_rank(Z_d)

        # 2. Z_d 槽间余弦相似度 (越低越好，高表示坍塌)
        metrics['draft/Zd_slot_cosim'] = self._slot_cosine_similarity(Z_d)

        # 3. Z_d L2 范数
        metrics['draft/Zd_norm'] = Z_d.norm(dim=-1).mean().item()

        # 4. Z_e 有效秩
        metrics['draft/Ze_effective_rank'] = self._effective_rank(Z_e)

        # 5. step_update 中收集的指标
        if '_monitor' in state:
            mon = state['_monitor']
            metrics['draft/Zd_abs_delta'] = mon['Zd_abs_delta']
            metrics['draft/Zd_relative_delta'] = mon['Zd_relative_delta']

        # 6. step_count
        metrics['draft/step_count'] = float(state['step_count'])

        return metrics

    @torch.no_grad()
    def _slot_cosine_similarity(self, Z: torch.Tensor) -> float:
        """
        计算 Z 各 slot 之间的平均余弦相似度 (off-diagonal)
        Z: [B, K, d_z]
        返回: 平均余弦相似度 (标量)
        """
        Z_norm = F.normalize(Z, p=2, dim=-1)  # [B, K, d_z]
        # gram: [B, K, K]
        gram = torch.bmm(Z_norm, Z_norm.transpose(1, 2))
        K = Z.shape[1]
        mask = ~torch.eye(K, dtype=torch.bool, device=Z.device).unsqueeze(0)
        offdiag = gram[mask.expand_as(gram)].view(Z.shape[0], -1)
        return offdiag.abs().mean().item()

    @torch.no_grad()
    def _effective_rank(self, Z: torch.Tensor) -> float:
        """
        计算 Z 的有效秩 (基于归一化奇异值的香农熵)
        Z: [B, K, d_z]
        
        有效秩 = exp(-Σ p_i * log(p_i))
        其中 p_i = σ_i / Σ σ_j 是归一化的奇异值分布
        
        有效秩越高 → 信息利用的维度越多 → 表达越丰富
        有效秩 ≈ 1 → 所有 slot 坍塌到一条线上
        有效秩 ≈ K → 所有 slot 完全独立
        """
        # 对 batch 取平均 Z_mean: [K, d_z]
        Z_mean = Z.float().mean(dim=0)  # [K, d_z]
        # SVD
        try:
            _, S, _ = torch.svd(Z_mean)  # S: [min(K, d_z)]
            # 归一化
            S = S / S.sum().clamp(min=1e-8)
            # 过滤零值
            S = S[S > 1e-8]
            # 香农熵
            entropy = -(S * S.log()).sum()
            effective_rank = entropy.exp().item()
        except Exception:
            effective_rank = 0.0
        return effective_rank

    def get_num_trainable_params(self) -> int:
        """返回可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_param_summary(self):
        """打印参数统计"""
        total = self.get_num_trainable_params()
        print(f"\n{'='*60}")
        print(f"🔧 RLD Controller 参数统计")
        print(f"{'='*60}")
        
        module_params = {}
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                module_params[name] = params
                print(f"  {name}: {params:,} ({params/1e6:.2f}M)")
        
        # 直接参数
        direct_params = sum(
            p.numel() for n, p in self.named_parameters()
            if p.requires_grad and '.' not in n
        )
        if direct_params > 0:
            print(f"  [direct params]: {direct_params:,} ({direct_params/1e6:.2f}M)")
        
        print(f"{'─'*60}")
        print(f"  总计: {total:,} ({total/1e6:.2f}M)")
        print(f"{'='*60}\n")
