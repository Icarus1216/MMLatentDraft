"""
RLD Controller: 编排整个 Reflective Latent Draft 的生命周期

核心职责:
1. Prefill 阶段: 从视觉 token 初始化冻结证据 Z^e，初始化草稿 Z^d_0 和轨迹 T_0
2. Step 边界更新: 执行 S_c → T_c → G_c → Z^d_c 的完整更新链路
   - T_c 只用于计算 G_c 和提供全局方向 bias，不再作为 DraftUpdater 的 KV
   - TraceUpdater 和 ReflectionModule 均采用 2 层 CA
3. KV Prefix 生成: 将 [Z_e_gated; Z^d_c] 投影到 embedding 空间
   - Z_e 经过 EvidenceGate 自适应门控，根据推理进度动态调节各 slot 的通过量
   - 推理早期 β ≈ 0.9 (几乎全通过)，后期无用 slot 自动衰减
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

from .modules import (
    EvidenceResampler,
    StepResampler,
    TraceUpdater,
    ReflectionModule,
    DraftUpdater,
    EvidenceGate,
    EmbeddingProjector,
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
        use_gate: 草稿更新是否使用门控 (默认 True)
        lambda_div: 多样性正则化权重 (默认 0.01)
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        d_z: int = 512,
        num_evidence_slots: int = 16,
        num_draft_slots: int = 16,
        num_trace_slots: int = 16,
        total_layers: int = 36,
        num_heads: int = 8,
        evidence_layers: int = 2,
        use_gate: bool = True,
        lambda_div: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.K_e = num_evidence_slots
        self.K_d = num_draft_slots
        self.K_t = num_trace_slots
        self.K_p = num_evidence_slots + num_draft_slots  # prefix 总长度
        self.total_layers = total_layers
        self.lambda_div = lambda_div

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

        # 5. 草稿更新器: (Z^d, G_c) → Z^d_{c+1}，T_c 仅提供全局方向 bias
        self.draft_updater = DraftUpdater(
            d_z=d_z,
            num_heads=num_heads,
            use_gate=use_gate,
        )

        # 6. 自适应证据门控: 根据推理进度动态调节 Z_e 各 slot 的通过量
        # 初始 β ≈ 0.9 (几乎全通过)，推理后期自动衰减无用的 Z_e slot
        self.evidence_gate = EvidenceGate(
            d_z=d_z,
            num_evidence_slots=num_evidence_slots,
            max_steps=32,
        )

        # 7. Embedding 投影器: Z^p → prefix embeddings (方案 A)
        # 将 [Z_e_gated; Z^d] 投影到 hidden_size 维度，作为虚拟 prefix token embedding
        # 梯度自然连通，无需 hook 覆盖 KV cache
        self.embedding_projector = EmbeddingProjector(
            d_z=d_z,
            hidden_size=hidden_size,

        )

        # ====== 初始化相关 ======

        # 可学习的草稿初始状态 (会在 prefill 时被条件化)
        self.draft_init = nn.Parameter(torch.randn(1, num_draft_slots, d_z) * 0.02)

        # 草稿初始条件化: 用证据 Z^e 条件化初始草稿
        self.draft_init_ca = nn.Sequential(
            RMSNorm(d_z),
            nn.Linear(d_z, d_z, bias=False),
        )

        # 多样性正则化
        self.diversity_reg = DiversityRegularizer(threshold=0.1)

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

        # 2. 初始化草稿 (用证据条件化)
        draft_init = self.draft_init.expand(B, -1, -1).to(device=device, dtype=dtype)
        # 用 Z_e 的 mean pooled 向量给初始草稿加偏置
        evidence_ctx = self.draft_init_ca(Z_e.mean(dim=1, keepdim=True))  # [B, 1, d_z]
        Z_d = draft_init + evidence_ctx.expand_as(draft_init)

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

        # 4. 草稿更新: KV=G_c, T_c 仅提供全局方向 bias
        Z_d_new = self.draft_updater(Z_d, T_c, G_c)  # [B, K_d, d_z]

        # P0-3: 根据 update_mask 选择性更新
        if update_mask is not None:
            # update_mask: [B] bool → [B, 1, 1] 用于广播
            mask = update_mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            # 未触发的样本保持原状态
            Z_d_new = mask * Z_d_new + (1.0 - mask) * Z_d
            T_c = mask * T_c + (1.0 - mask) * T_prev

        return {
            'Z_e': Z_e,  # 不变
            'Z_d': Z_d_new,
            'T': T_c,
            'step_count': state['step_count'] + 1,
        }

    def get_prefix_embeds(
        self,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        生成 prefix embeddings (方案 A - Embedding 注入 + 自适应证据门控)
        
        将 [Z_e_gated; Z^d] 投影到 hidden_size 维度，作为虚拟 prefix token 的 embedding。
        其中 Z_e_gated = β ⊙ Z_e，β 由 EvidenceGate 根据当前推理进度动态计算。
        
        这些 embedding 直接拼接到 inputs_embeds 前面，参与完整的 Transformer forward，
        梯度自然连通，无需 hook 覆盖 KV cache。
        
        Args:
            state: RLD 状态字典
        
        Returns:
            prefix_embeds: [B, K_p, hidden_size] 可直接拼接到 inputs_embeds 前面
        """
        Z_e = state['Z_e']
        Z_d = state['Z_d']
        step_count = state['step_count']

        # 自适应证据门控: 根据推理进度调节 Z_e 各 slot 的通过量
        Z_e_gated, _beta = self.evidence_gate(Z_e, Z_d, step_count)

        # 拼接 prefix: [Z_e_gated; Z^d]
        Z_p = torch.cat([Z_e_gated, Z_d], dim=1)  # [B, K_e + K_d, d_z]

        # 投影到 hidden_size 维度
        return self.embedding_projector(Z_p)  # [B, K_p, hidden_size]

    def compute_diversity_loss(
        self,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算多样性正则化损失
        
        对 Z^d 和轨迹 queries 进行去相关正则化
        """
        Z_d = state['Z_d']
        loss = self.diversity_reg(Z_d) * self.lambda_div
        return loss

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
