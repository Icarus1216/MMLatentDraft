"""
RLD 核心模块

实现白皮书中的所有基础组件:
1. CrossAttentionBlock: 通用 cross-attention + LN + MLP
2. EvidenceResampler: 将可变长视觉 token 压缩到 K_e=16 个证据槽
3. StepResampler: 将当前 step 的 hidden states 压缩到 K_t=16 个摘要槽
4. TraceUpdater: 累计式推理轨迹更新 (T_{c-1}, S_c) -> T_c (2层CA, 防止长链记忆退化)
5. ReflectionModule: 回看-验证 (T_c, Z^e) -> G_c (2层CA, retrieval + verification)
6. DraftUpdater: 用 G_c 更新可擦写草稿 Z^d, T_c 仅提供全局方向 bias
7. EvidenceGate: 自适应证据门控，根据推理进度动态调节 Z_e 各 slot 的通过量
8. EmbeddingProjector: 将 draft prefix 投影到 hidden_size 维度的 embedding 空间 (方案 A)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """RMSNorm，与 Qwen3.5 内部使用的归一化方式一致"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class CrossAttentionBlock(nn.Module):
    """
    标准 Cross-Attention Block: MultiHeadCrossAttention + LN + MLP
    
    用于所有需要 cross-attention 的地方:
    - Evidence Resampler (Q=learnable, KV=visual tokens)
    - Step Resampler (Q=learnable, KV=step hidden states)
    - Trace Updater (Q=learnable, KV=[T_{c-1}; S_c])
    - Reflection (Q=T_c, KV=Z^e)
    - Draft Updater (Q=Z^d, KV=[T_c; G_c])
    
    Args:
        d_model: 模型维度 (d_z)
        num_heads: 注意力头数
        mlp_ratio: MLP 隐藏层倍数
        dropout: dropout 概率
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        # Cross-Attention
        self.q_norm = RMSNorm(d_model)
        self.kv_norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # MLP (FFN)
        mlp_hidden = int(d_model * mlp_ratio)
        self.ffn_norm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, mlp_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, mlp_hidden, bias=False)
        self.down_proj = nn.Linear(mlp_hidden, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,        # [B, L_q, D]
        key_value: torch.Tensor,     # [B, L_kv, D]
    ) -> torch.Tensor:
        """
        Returns:
            output: [B, L_q, D]
        """
        # 统一 dtype（DeepSpeed ZeRO 下权重可能是 bf16，输入可能是 fp32）
        param_dtype = self.q_proj.weight.dtype
        query = query.to(param_dtype)
        key_value = key_value.to(param_dtype)

        B, L_q, D = query.shape
        _, L_kv, _ = key_value.shape

        # Pre-norm
        q = self.q_norm(query)
        kv = self.kv_norm(key_value)

        # 投影 Q, K, V
        q = self.q_proj(q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # 合并头并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        attn_output = self.o_proj(attn_output)

        # 残差连接
        hidden = query + attn_output

        # FFN: SwiGLU + 残差
        residual = hidden
        hidden = self.ffn_norm(hidden)
        hidden = self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))
        hidden = residual + hidden

        return hidden


class EvidenceResampler(nn.Module):
    """
    视觉证据重采样器
    
    将可变长度的视觉 token 序列 V ∈ R^{L_v × 2048} 压缩为
    固定长度的证据槽 Z^e ∈ R^{K_e × d_z}
    
    采用 Perceiver/Flamingo 风格的 learnable queries cross-attention。
    
    Args:
        hidden_size: 基座模型隐藏维度 (2560 for Qwen3.5-4B)
        d_z: controller 空间维度 (512)
        num_evidence_slots: 证据槽数量 K_e (默认 16)
        num_heads: cross-attention 头数
        num_layers: cross-attention 层数
    """

    def __init__(
        self,
hidden_size: int = 2560,
        d_z: int = 512,
        num_evidence_slots: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_slots = num_evidence_slots
        self.d_z = d_z

        # 从 hidden_size 投影到 d_z
        self.proj_v = nn.Linear(hidden_size, d_z, bias=False)

        # 可学习的 evidence queries
        self.evidence_queries = nn.Parameter(torch.randn(1, num_evidence_slots, d_z) * 0.02)

        # Cross-attention 层
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model=d_z, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: [B, L_v, hidden_size] 来自视觉编码器的 token 序列
        
        Returns:
            Z_e: [B, K_e, d_z] 冻结证据槽（推理过程中不更新）
        """
        B = visual_tokens.shape[0]

        # 统一 dtype（DeepSpeed ZeRO 下权重可能是 bf16，输入可能是 fp32）
        visual_tokens = visual_tokens.to(self.proj_v.weight.dtype)

        # 投影到 controller 空间
        v_hat = self.proj_v(visual_tokens)  # [B, L_v, d_z]

        # 扩展 queries 到 batch
        queries = self.evidence_queries.expand(B, -1, -1)  # [B, K_e, d_z]

        # 多层 cross-attention
        for layer in self.layers:
            queries = layer(query=queries, key_value=v_hat)

        return queries  # [B, K_e, d_z]


class StepResampler(nn.Module):
    """
    步摘要重采样器
    
    将当前 step 的 hidden states H^{step}_c ∈ R^{L_c × 2048}
    压缩为固定长度摘要 S_c ∈ R^{K_t × d_z}
    
    Args:
        hidden_size: 基座模型隐藏维度 (2560)
        d_z: controller 空间维度 (512)
        num_trace_slots: 轨迹槽数量 K_t (默认 16)
        num_heads: cross-attention 头数
    """

    def __init__(
        self,
hidden_size: int = 2560,
        d_z: int = 512,
        num_trace_slots: int = 16,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_slots = num_trace_slots
        self.d_z = d_z

        # 从 hidden_size 投影到 d_z
        self.proj_h = nn.Linear(hidden_size, d_z, bias=False)

        # 可学习的 step queries
        self.step_queries = nn.Parameter(torch.randn(1, num_trace_slots, d_z) * 0.02)

        # 单层 cross-attention (step 摘要不需要太深)
        self.ca_block = CrossAttentionBlock(d_model=d_z, num_heads=num_heads)

    def forward(self, step_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_hidden_states: [B, L_c, hidden_size] 当前 step 的最后层 hidden states
        
        Returns:
            S_c: [B, K_t, d_z] 当前 step 的摘要
        """
        B = step_hidden_states.shape[0]

        # 统一 dtype（DeepSpeed ZeRO 下权重可能是 bf16，输入可能是 fp32）
        step_hidden_states = step_hidden_states.to(self.proj_h.weight.dtype)

        # 投影
        h_hat = self.proj_h(step_hidden_states)  # [B, L_c, d_z]

        # 扩展 queries
        queries = self.step_queries.expand(B, -1, -1)  # [B, K_t, d_z]

        # Cross-attention
        S_c = self.ca_block(query=queries, key_value=h_hat)

        return S_c  # [B, K_t, d_z]


class TraceUpdater(nn.Module):
    """
    累计式推理轨迹更新器 (2 层 CA)
    
    将 "旧记忆 T_{c-1} + 新摘要 S_c" 重新压回固定长度 T_c:
      T_c = Resample_t([T_{c-1}; S_c])
    
    这与 Compressive Transformer 的思想一致：
    把历史信息不断压缩进固定容量的记忆中。
    
    使用 2 层 cross-attention：
    - 第 1 层：粗选（确定 T_prev 中哪些历史信息值得保留）
    - 第 2 层：精融（将选中的历史和新摘要融合到 K_t 个 slot 中）
    
    Args:
        d_z: controller 空间维度
        num_trace_slots: 轨迹槽数量 K_t (默认 16)
        num_heads: cross-attention 头数
        num_layers: cross-attention 层数 (默认 2)
    """

    def __init__(
        self,
        d_z: int = 512,
        num_trace_slots: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_slots = num_trace_slots
        self.d_z = d_z

        # 可学习的 trace queries
        self.trace_queries = nn.Parameter(torch.randn(1, num_trace_slots, d_z) * 0.02)

        # 多层 Cross-attention: 从 [T_{c-1}; S_c] (长度 2*K_t=32) 中重采样
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model=d_z, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        T_prev: torch.Tensor,   # [B, K_t, d_z] 上一步的轨迹记忆
        S_c: torch.Tensor,       # [B, K_t, d_z] 当前步的摘要
    ) -> torch.Tensor:
        """
        Returns:
            T_c: [B, K_t, d_z] 更新后的累计轨迹记忆
        """
        B = T_prev.shape[0]

        # 拼接旧记忆和新摘要: [B, 2*K_t, d_z]
        combined = torch.cat([T_prev, S_c], dim=1)

        # 用 learnable queries 多层重采样
        queries = self.trace_queries.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(query=queries, key_value=combined)

        return queries  # [B, K_t, d_z]


class ReflectionModule(nn.Module):
    """
    回看-验证模块 (核心操作, 2 层 CA)
    
    用累计轨迹 T_c 作为 query，去冻结的视觉证据 Z^e 中检索:
      G_c = CA_layers(Q=T_c, K=Z^e, V=Z^e)
    
    解释:
    - T_c 是"目前为止我们在推理什么/相信什么"的 latent
    - 对 Z^e 做 cross-attn 相当于:
      "用当前推理作为 query 去图像证据里找相关区域/数字/关系的 latent 支撑"
    - G_c 是 grounded trace (被视觉证据支撑/对齐后的推理表示)
    
    使用 2 层 cross-attention 实现 "retrieval then verification" 范式：
    - 第 1 层：初步检索相关证据（"图中有哪些与当前推理相关的数字/关系？"）
    - 第 2 层：基于第 1 层的结果做精细验证（"这些数字/关系是否支持当前推理？"）
    
    Args:
        d_z: controller 空间维度
        num_heads: cross-attention 头数
        num_layers: cross-attention 层数 (默认 2)
    """

    def __init__(self, d_z: int = 512, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model=d_z, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        T_c: torch.Tensor,   # [B, K_t, d_z]
        Z_e: torch.Tensor,   # [B, K_e, d_z]
    ) -> torch.Tensor:
        """
        Returns:
            G_c: [B, K_t, d_z] grounded trace (证据支撑的推理表示)
        """
        G_c = T_c
        for layer in self.layers:
            G_c = layer(query=G_c, key_value=Z_e)
        return G_c


class DraftUpdater(nn.Module):
    """
    草稿更新器 (G_c 作为唯一 KV + T_c additive bias)
    
    用 G_c (证据校验后的推理) 更新可擦写草稿 Z^d:
      1. U_c = CA(Q=Z^d_c, K=G_c, V=G_c)   — 主要信息来源
      2. t_bias = Linear(MeanPool(T_c))       — 全局推理方向偏置
      3. Z^d_{c+1} = LN((1-α_c) * Z^d_c + α_c * MLP(U_c + t_bias))
    
    设计动机:
    - G_c 已包含 T_c 的核心信息 (通过 ReflectionModule 的 residual 连接)
    - T_c 不再作为 KV 拼接 (避免单层 CA 无法区分两个信源的问题)
    - T_c 通过 additive bias 注入全局推理方向，信号不互相干扰
    
    其中 α_c 是门控系数，防止每步过写。
    
    Args:
        d_z: controller 空间维度
        num_heads: cross-attention 头数
        use_gate: 是否使用门控 (推荐 True)
    """

    def __init__(
        self,
        d_z: int = 512,
        num_heads: int = 8,
        use_gate: bool = True,
    ):
        super().__init__()
        self.d_z = d_z
        self.use_gate = use_gate

        # Cross-attention: Q=Z^d, KV=G_c (只用证据校验后的推理)
        self.ca_block = CrossAttentionBlock(d_model=d_z, num_heads=num_heads)

        # T_c 全局方向 bias: MeanPool → Linear → broadcast
        self.trace_bias_proj = nn.Sequential(
            RMSNorm(d_z),
            nn.Linear(d_z, d_z, bias=False),
        )

        # MLP for update
        self.update_mlp = nn.Sequential(
            nn.Linear(d_z, d_z * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_z * 4, d_z, bias=False),
        )

        # 输出归一化
        self.out_norm = RMSNorm(d_z)

        # 门控
        if use_gate:
            self.gate_proj = nn.Linear(d_z, 1, bias=True)
            # 初始化 bias 使初始 sigmoid ≈ 0.1 (保守更新)
            nn.init.constant_(self.gate_proj.bias, -2.2)

        self._init_trace_bias()

    def _init_trace_bias(self):
        """小范围初始化 trace bias 投影，减少初始干扰"""
        for module in self.trace_bias_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)

    def forward(
        self,
        Z_d: torch.Tensor,    # [B, K_d, d_z] 当前草稿
        T_c: torch.Tensor,    # [B, K_t, d_z] 累计轨迹 (仅提供方向 bias)
        G_c: torch.Tensor,    # [B, K_t, d_z] grounded trace (主 KV)
    ) -> torch.Tensor:
        """
        Returns:
            Z_d_new: [B, K_d, d_z] 更新后的草稿
        """
        # 统一 dtype（DeepSpeed ZeRO 下权重可能是 bf16，输入可能是 fp32）
        param_dtype = self.ca_block.q_proj.weight.dtype
        Z_d = Z_d.to(param_dtype)
        T_c = T_c.to(param_dtype)
        G_c = G_c.to(param_dtype)

        # Cross-attention: 旧草稿只从 G_c (证据校验后的推理) 中读取
        U_c = self.ca_block(query=Z_d, key_value=G_c)  # [B, K_d, d_z]

        # T_c 全局方向 bias: mean pooling → 投影 → broadcast 到每个 slot
        t_bias = self.trace_bias_proj(T_c.mean(dim=1, keepdim=True))  # [B, 1, d_z]
        U_c = U_c + t_bias  # 加性注入推理方向

        # MLP
        update = self.update_mlp(U_c)  # [B, K_d, d_z]

        if self.use_gate:
            # 门控: α_c = sigmoid(W_α · Pool(U_c))
            # 对每个 slot 独立计算门控
            alpha = torch.sigmoid(self.gate_proj(U_c))  # [B, K_d, 1]
            Z_d_new = self.out_norm((1.0 - alpha) * Z_d + alpha * update)
        else:
            Z_d_new = self.out_norm(Z_d + update)

        return Z_d_new


class EvidenceGate(nn.Module):
    """
    自适应证据门控
    
    根据当前推理状态（Z_d 的全局摘要 + 推理步数）动态调节 Z_e 各 slot 的通过量。
    
    核心思想:
    - 推理早期：β ≈ 0.9，Z_e 几乎全量通过（模型需要所有视觉证据）
    - 推理后期：无用的 Z_e slot 的 β 学习降低，释放 attention budget
    - 不同 slot 独立门控：模型可以选择性保留关键证据、衰减无关证据
    
    β = σ(GateNet([Z_d_mean; step_emb]))
    Z_e' = β ⊙ Z_e
    
    初始化策略: bias = +2.2 使 σ(2.2) ≈ 0.9，初始时几乎全部通过
    
    设计动机:
    - Z_e 在整个推理过程中冻结不变，但不同推理阶段对证据的需求不同
    - 后期推理步骤中无用的 Z_e slot 会占用 attention budget（注意力稀释）
    - 门控可以让模型自适应地衰减无关 slot，释放 attention 给真正需要的 token
    - 被衰减到接近零的 slot，其位置编码不一致问题也随之消失
    
    Args:
        d_z: controller 空间维度
        num_evidence_slots: 证据槽数量 K_e
        max_steps: 最大推理步数（用于步数嵌入）
    """

    def __init__(
        self,
        d_z: int = 512,
        num_evidence_slots: int = 16,
        max_steps: int = 32,
    ):
        super().__init__()
        self.d_z = d_z
        self.K_e = num_evidence_slots

        # 步数嵌入 (positional encoding for step count)
        self.step_embedding = nn.Embedding(max_steps + 1, d_z // 4)

        # 门控网络: [Z_d_mean(d_z) + step_emb(d_z//4)] → K_e 个门控值
        gate_input_dim = d_z + d_z // 4
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, d_z // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_z // 4, num_evidence_slots, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化: bias = +2.2 使 sigmoid ≈ 0.9 (几乎全通过), 权重小范围初始化减少随机扰动"""
        nn.init.constant_(self.gate_net[-1].bias, 2.2)
        for m in self.gate_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    def forward(
        self,
        Z_e: torch.Tensor,     # [B, K_e, d_z] 冻结证据
        Z_d: torch.Tensor,     # [B, K_d, d_z] 当前草稿
        step_count: int,        # 当前推理步数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z_e: [B, K_e, d_z] 冻结证据
            Z_d: [B, K_d, d_z] 当前草稿
            step_count: int 当前推理步数
        
        Returns:
            Z_e_gated: [B, K_e, d_z] 门控后的证据
            beta: [B, K_e, 1] 门控值 (可用于监控/可视化)
        """
        B = Z_e.shape[0]
        device = Z_e.device

        # Z_d 的全局摘要
        z_d_mean = Z_d.mean(dim=1)  # [B, d_z]

        # 步数嵌入 (clamp 到 max_steps)
        step_idx = torch.tensor(
            min(step_count, self.step_embedding.num_embeddings - 1),
            device=device,
        )
        step_emb = self.step_embedding(step_idx)  # [d_z//4]
        step_emb = step_emb.unsqueeze(0).expand(B, -1)  # [B, d_z//4]

        # 拼接上下文，并统一 dtype（step_embedding 输出 float32，需对齐到 gate_net 权重 dtype）
        context = torch.cat([z_d_mean, step_emb], dim=-1)  # [B, d_z + d_z//4]
        context = context.to(self.gate_net[0].weight.dtype)

        # 计算门控
        beta = torch.sigmoid(self.gate_net(context))  # [B, K_e]
        beta = beta.unsqueeze(-1)  # [B, K_e, 1]

        # 门控 Z_e
        Z_e_gated = Z_e * beta

        return Z_e_gated, beta


class EmbeddingProjector(nn.Module):
    """
    Embedding 投影器: 将 draft prefix 投影到 hidden_size 维度的 embedding 空间
    
    核心思路 (方案 A - Embedding 注入):
    将 Z^p = [Z^e; Z^d] ∈ R^{K_p × d_z} 投影到 hidden_size 维度，
    作为「虚拟 prefix token 的 embedding」拼接到 inputs_embeds 前面。
    整个 Transformer forward 会自然处理这些 prefix embedding，
    梯度天然连通，无需 hook 覆盖 KV cache。
    
    优点:
    1. 梯度自然连通: prefix embedding 参与整个 forward，CE Loss 梯度直达
    2. 实现简洁: 无需 hook、无需手动展开 decoder 层
    3. 兼容性好: 与 flash attention / mRoPE 天然兼容
    
    4B Dense 优化: 使用全秩投影 (不再低秩分解)
      prefix_embeds = RMSNorm(MLP(Z_p))
      MLP: d_z → hidden_size (直接投影, 参数量仅 ~1.3M)
    
    Args:
        d_z: controller 空间维度 (512)
        hidden_size: 基座模型隐藏维度 (2560 for Qwen3.5-4B)
    """

    def __init__(
        self,
        d_z: int = 512,
        hidden_size: int = 2560,
    ):
        super().__init__()
        self.d_z = d_z
        self.hidden_size = hidden_size

        # 4B 优化: 全秩投影 (不再低秩分解)
        # d_z(512) → hidden_size(2560), 参数量 = 512*2560 = 1.3M, 可忽略
        self.proj = nn.Linear(d_z, hidden_size, bias=False)

        # 归一化 (使输出尺度与基座 embedding 匹配)
        self.out_norm = RMSNorm(hidden_size)

        # 可学习的缩放因子 (初始化为小值，减少初始干扰)
        self.scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        """小范围初始化，让初始注入对模型的干扰最小"""
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(
        self,
        Z_p: torch.Tensor,   # [B, K_p, d_z]
    ) -> torch.Tensor:
        """
        将 draft prefix 投影到 hidden_size 维度
        
        Args:
            Z_p: [B, K_p, d_z] draft prefix 表示 (= [Z^e; Z^d])
        
        Returns:
            prefix_embeds: [B, K_p, hidden_size] 可直接拼接到 inputs_embeds 前面
        """
        # 统一 dtype（DeepSpeed ZeRO 下权重可能是 bf16，输入可能是 fp32）
        Z_p = Z_p.to(self.proj.weight.dtype)

        # 全秩投影: d_z → hidden_size
        h = self.proj(Z_p)              # [B, K_p, hidden_size]
        h = self.out_norm(h)             # [B, K_p, hidden_size]
        h = h * self.scale               # 缩放，减少初始干扰
        return h


class DiversityRegularizer(nn.Module):
    """
    多样性正则化 (防止草稿槽坍塌)
    
    L_div = Σ_{i≠j} max(0, cos(z_i, z_j) - δ)
    
    Args:
        threshold: 相似度阈值 δ (默认 0.1)
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: [B, K, d_z] latent 序列
        
        Returns:
            loss: 标量
        """
        # 在 batch 维取均值
        B, K, D = Z.shape
        Z_norm = F.normalize(Z, p=2, dim=-1)  # [B, K, D]

        # 计算 gram 矩阵: [B, K, K]
        gram = torch.bmm(Z_norm, Z_norm.transpose(1, 2))

        # 提取 off-diagonal
        # mask 需要 expand 到 [B, K, K] 以匹配 gram 的 batch 维度 (boolean indexing 不支持广播)
        mask = ~torch.eye(K, dtype=torch.bool, device=Z.device).unsqueeze(0).expand(B, -1, -1)
        offdiag = gram[mask].view(B, -1)  # [B, K*(K-1)]

        # Hinge loss
        penalty = torch.relu(offdiag.abs() - self.threshold)
        loss = (penalty ** 2).mean()

        return loss
