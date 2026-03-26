"""
RLD 核心模块

实现白皮书中的所有基础组件:
1. CrossAttentionBlock: 通用 cross-attention + LN + MLP
2. EvidenceResampler: 将可变长视觉 token 压缩到 K_e=16 个证据槽
3. StepResampler: 将当前 step 的 hidden states 压缩到 K_t=16 个摘要槽
4. TraceUpdater: 累计式推理轨迹更新 (T_{c-1}, S_c) -> T_c (2层CA, 防止长链记忆退化)
5. ReflectionModule: 回看-验证 (T_c, Z^e) -> G_c (2层CA, retrieval + verification)
6. DraftUpdater: 用 G_c 更新可擦写草稿 Z^d, T_c 仅提供全局方向 bias
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 调试开关: 仅主进程 + RLD_DEBUG=1 时打印多层注入调试信息
def _should_debug_print():
    if os.environ.get('RLD_DEBUG', '0') != '1':
        return False
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '-1'))) in (-1, 0)

_DEBUG_MULTILAYER = None  # 延迟初始化


class RMSNorm(nn.Module):
    """RMSNorm，与 Qwen3-VL 内部使用的归一化方式一致"""

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
        d_model: int = 768,
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
        hidden_size: 基座模型隐藏维度 (4096 for Qwen3-VL-8B)
        d_z: controller 空间维度 (512)
        num_evidence_slots: 证据槽数量 K_e (默认 16)
        num_heads: cross-attention 头数
        num_layers: cross-attention 层数
    """

    def __init__(
        self,
hidden_size: int = 4096,
d_z: int = 768,
        num_evidence_slots: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_slots = num_evidence_slots
        self.d_z = d_z

        # 从 hidden_size 投影到 d_z
        self.proj_v = nn.Linear(hidden_size, d_z, bias=False)

        # 可学习的 evidence queries (正交初始化, 防止 Z_e 源头坍塌)
        _eq_data = torch.empty(num_evidence_slots, d_z)
        nn.init.orthogonal_(_eq_data)
        _eq_data = _eq_data * (0.5 / _eq_data.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        self.evidence_queries = nn.Parameter(_eq_data.unsqueeze(0))  # [1, K_e, d_z]

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
        hidden_size: 基座模型隐藏维度 (4096)
        d_z: controller 空间维度 (512)
        num_trace_slots: 轨迹槽数量 K_t (默认 16)
        num_heads: cross-attention 头数
    """

    def __init__(
        self,
hidden_size: int = 4096,
d_z: int = 768,
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
d_z: int = 768,
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

    def __init__(self, d_z: int = 768, num_heads: int = 8, num_layers: int = 2):
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
    [已弃用] 旧版门控草稿更新器，保留用于向后兼容和权重加载。
    新代码请使用 ResidualFlowDraftUpdater。
    """

    def __init__(
        self,
d_z: int = 768,
        num_heads: int = 8,
        use_gate: bool = True,
    ):
        super().__init__()
        self.d_z = d_z
        self.use_gate = use_gate
        self.ca_block = CrossAttentionBlock(d_model=d_z, num_heads=num_heads)
        self.trace_bias_proj = nn.Sequential(
            RMSNorm(d_z),
            nn.Linear(d_z, d_z, bias=False),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(d_z, d_z * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_z * 4, d_z, bias=False),
        )
        if use_gate:
            self.gate_proj = nn.Linear(d_z, 1, bias=True)
            nn.init.constant_(self.gate_proj.bias, -2.2)
        self._init_trace_bias()

    def _init_trace_bias(self):
        for module in self.trace_bias_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)

    def forward(self, Z_d, T_c, G_c):
        param_dtype = self.ca_block.q_proj.weight.dtype
        Z_d = Z_d.to(param_dtype)
        T_c = T_c.to(param_dtype)
        G_c = G_c.to(param_dtype)
        U_c = self.ca_block(query=Z_d, key_value=G_c)
        t_bias = self.trace_bias_proj(T_c.mean(dim=1, keepdim=True))
        U_c = U_c + t_bias
        update = self.update_mlp(U_c)
        if self.use_gate:
            alpha = torch.sigmoid(self.gate_proj(U_c))
            Z_d_new = (1.0 - alpha) * Z_d + alpha * update
        else:
            Z_d_new = Z_d + update
        max_norm = 43.0  # √(768/512) × 35 ≈ 43, 适配 d_z=768
        slot_norms = Z_d_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clamped_scale = (max_norm / slot_norms).clamp(max=1.0)
        Z_d_new = Z_d_new * clamped_scale
        return Z_d_new


class ResidualFlowDraftUpdater(nn.Module):
    """
    Neural ODE / Residual Flow 草稿更新器
    
    核心公式:
      v = v_θ(Z^d_c, T_c, G_c, t=c/C)
      Z^d_{c+1} = Z^d_c + Δt · v
    
    其中 v_θ (velocity field) 内部通过 cross-attention 让 Z_d 从 G_c 读取
    "推理与视觉相互印证"的信息，再注入时间嵌入和轨迹方向 bias。
    
    相比旧版 DraftUpdater 的改进:
    1. 去掉 sigmoid 门控 → 消除对 G_c 信号 90% 的衰减
    2. 残差步进 Z_d + Δt·v → 梯度路径无饱和区，∂Z_new/∂Z_d = I + Δt·∂v/∂Z_d
    3. 时间嵌入 t=c/C → 让网络感知推理阶段（早期粗略印证，后期精细校验）
    4. 信息流拓扑不变: ReflectionModule → G_c → velocity field，"相互印证"完全保留
    
    Args:
        d_z: controller 空间维度 (512)
        num_heads: cross-attention 头数 (8)
        max_steps: 最大 step 数 C，用于计算 Δt = 1/C (默认 14)
    """

    def __init__(
        self,
d_z: int = 768,
        num_heads: int = 8,
        max_steps: int = 14,
    ):
        super().__init__()
        self.d_z = d_z
        self.max_steps = max_steps

        # ====== 时间嵌入: 正弦位置编码 + MLP 投影到 d_z ======
        # 用 64 维正弦频率编码 step 进度 t ∈ [0, 1]
        self.time_embed_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, d_z, bias=False),
            nn.SiLU(),
            nn.Linear(d_z, d_z, bias=False),
        )
        # 小范围初始化，避免初始时间嵌入过大
        for m in self.time_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # ====== Velocity Field v_θ 的核心组件 ======

        # 1. Cross-attention: Q=Z^d, KV=G_c (从 grounded trace 读取印证信息)
        self.ca_block = CrossAttentionBlock(d_model=d_z, num_heads=num_heads)

        # 2. T_c 全局方向 bias: MeanPool → Linear → broadcast
        self.trace_bias_proj = nn.Sequential(
            RMSNorm(d_z),
            nn.Linear(d_z, d_z, bias=False),
        )

        # 3. Velocity MLP: 融合 CA 输出 + 时间嵌入 + 轨迹 bias → velocity
        self.velocity_mlp = nn.Sequential(
            RMSNorm(d_z),
            nn.Linear(d_z, d_z * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_z * 4, d_z, bias=False),
        )
        # velocity MLP 输出初始化为小值，确保初始 velocity ≈ 0（稳定训练起步）
        nn.init.normal_(self.velocity_mlp[-1].weight, std=0.01)

        self._init_trace_bias()

    def _init_trace_bias(self):
        """小范围初始化 trace bias 投影，减少初始干扰"""
        for module in self.trace_bias_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)

    def _sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        正弦时间嵌入 (类似 Transformer 位置编码)
        
        Args:
            t: [B] 或 [B, 1] 标量时间值 ∈ [0, 1]
        
        Returns:
            emb: [B, time_embed_dim] 时间嵌入向量
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)  # [B]
        
        half_dim = self.time_embed_dim // 2
        # 频率: exp(-log(10000) * i / (half_dim - 1)), i = 0, ..., half_dim-1
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / max(half_dim - 1, 1)
        )  # [half_dim]
        # 外积: [B, half_dim]
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * math.pi
        # 拼接 sin 和 cos: [B, time_embed_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

    def forward(
        self,
        Z_d: torch.Tensor,       # [B, K_d, d_z] 当前草稿
        T_c: torch.Tensor,       # [B, K_t, d_z] 累计轨迹 (提供方向 bias)
        G_c: torch.Tensor,       # [B, K_t, d_z] grounded trace (主 KV)
        step_progress: float = 0.0,  # t = c/C ∈ [0, 1], 当前推理进度
    ) -> torch.Tensor:
        """
        Residual Flow 更新: Z^d_{c+1} = Z^d_c + Δt · v_θ(Z^d_c, T_c, G_c, t)
        
        Returns:
            Z_d_new: [B, K_d, d_z] 更新后的草稿
        """
        B = Z_d.shape[0]
        param_dtype = self.ca_block.q_proj.weight.dtype
        Z_d = Z_d.to(param_dtype)
        T_c = T_c.to(param_dtype)
        G_c = G_c.to(param_dtype)

        # ====== 1. 时间嵌入 ======
        t_val = torch.tensor([step_progress], device=Z_d.device, dtype=param_dtype).expand(B)
        t_emb_raw = self._sinusoidal_time_embedding(t_val)  # [B, time_embed_dim]
        t_emb = self.time_mlp(t_emb_raw)  # [B, d_z]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, d_z], 广播到所有 slot

        # ====== 2. Cross-attention: Z_d 从 G_c 读取印证信息 ======
        # G_c 是 ReflectionModule(T_c, Z_e) 的输出，承载了"推理×视觉交叉验证"
        U_c = self.ca_block(query=Z_d, key_value=G_c)  # [B, K_d, d_z]

        # ====== 3. T_c 全局方向 bias ======
        t_bias = self.trace_bias_proj(T_c.mean(dim=1, keepdim=True))  # [B, 1, d_z]

        # ====== 4. 融合: CA输出 + 时间嵌入 + 轨迹bias ======
        # 加性融合: U_c 已经是 Z_d 的残差 (CA block 内部有残差连接)
        # 时间嵌入和轨迹 bias 作为条件调制
        combined = U_c + t_emb + t_bias  # [B, K_d, d_z]

        # ====== 5. Velocity MLP → v_θ ======
        velocity = self.velocity_mlp(combined)  # [B, K_d, d_z]

        # ====== 6. 残差步进: Z_d + Δt · v ======
        # Δt = 1 / max_steps, 确保多步累积后总位移在合理范围
        dt = 1.0 / self.max_steps
        Z_d_new = Z_d + dt * velocity

        # ====== 7. 软范数裁剪 (防止累积爆炸，保留 slot 间范数差异) ======
        max_norm = 43.0  # √(768/512) × 35 ≈ 43, 适配 d_z=768
        slot_norms = Z_d_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, K_d, 1]
        clamped_scale = (max_norm / slot_norms).clamp(max=1.0)
        Z_d_new = Z_d_new * clamped_scale

        return Z_d_new


class DiversityRegularizer(nn.Module):
    """
    多样性正则化 (防止草稿槽坍塌)
    
    L_div = Σ_{i≠j} max(0, cos(z_i, z_j) - δ)
    
    Args:
        threshold: 相似度阈值 δ (默认 0.3, 允许 slot 间有适度相似性)
    """

    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: [B, K, d_z] latent 序列
        
        Returns:
            loss: 标量
        """
        B, K, D = Z.shape
        Z_norm = F.normalize(Z, p=2, dim=-1)  # [B, K, D]

        # 计算 gram 矩阵: [B, K, K]
        gram = torch.bmm(Z_norm, Z_norm.transpose(1, 2))

        # 提取 off-diagonal
        mask = ~torch.eye(K, dtype=torch.bool, device=Z.device).unsqueeze(0).expand(B, -1, -1)
        offdiag = gram[mask].view(B, -1)  # [B, K*(K-1)]

        # 带阈值的余弦相似度惩罚: 允许 slot 间有适度相似性 (< threshold)
        # 只有超过阈值的部分才受惩罚, 避免过度正则化破坏模型表达能力
        # 梯度: 2*(|cosim| - threshold) * sign(cosim), 比旧版 hinge 梯度更强
        excess = (offdiag.abs() - self.threshold).clamp(min=0.0)
        loss = (excess ** 2).mean()

        return loss


class DraftReadoutAdapter(nn.Module):
    """
    Draft Readout Adapter (版本 A: 最稳、最简单)
    
    对每个 token 的 frozen hidden h_t，让它读当前 step 的 draft Z_d:
      a_t = CrossAttn(Q=h_t, K=Z_d, V=Z_d)
      h_t_adapted = h_t + scale * W_o(a_t)
    
    最后再过原来的 lm_head 得到 logits:
      logits_t = lm_head(h_t_adapted)
    
    设计动机:
    - Draft 通过 cross-attention 直接影响 token 分布
    - 训练时只需要一次 base forward
    - 没有 cache 改写、没有多次 forward
    - Prefix-Tuning (Li & Liang, ACL 2021) 证明：小的连续适配器
      可以强力调控冻结模型的输出
    
    Args:
        hidden_size: 基座模型隐藏维度 (3584 for Qwen3-VL-8B)
        d_z: controller 空间维度 (512)
        num_heads: cross-attention 头数
    """

    def __init__(
        self,
        hidden_size: int = 3584,
d_z: int = 768,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.num_heads = num_heads
        self.head_dim = d_z // num_heads
        assert d_z % num_heads == 0

        # 将 hidden_size 的 h_t 投影到 d_z 空间作为 query
        self.q_proj_down = nn.Linear(hidden_size, d_z, bias=False)

        # Cross-attention: Q=h_t(d_z), KV=Z_d(d_z)
        self.q_norm = RMSNorm(d_z)
        self.kv_norm = RMSNorm(d_z)
        self.q_proj = nn.Linear(d_z, d_z, bias=False)
        self.k_proj = nn.Linear(d_z, d_z, bias=False)
        self.v_proj = nn.Linear(d_z, d_z, bias=False)
        self.o_proj = nn.Linear(d_z, d_z, bias=False)

        # 将 cross-attention 输出从 d_z 投影回 hidden_size
        self.out_proj_up = nn.Linear(d_z, hidden_size, bias=False)

        # 输出归一化
        self.out_norm = RMSNorm(hidden_size)

        # 方案 A: 放大缩放因子初始值，避免三重投影链导致信号衰减过大
        # 从 0.1 → 1.0，配合 std=0.02 使初始修正量从 ~10^{-7} 提升到 ~10^{-4}
        self.scale = nn.Parameter(torch.tensor(1.0))

        # 自适应范数比约束参数: 基座越不确定 → 允许越大的修正量
        # base_ratio: 基座 100% 确定时的最低修正上限 (inject_weight ≈ 0.05)
        # max_ratio: 基座完全不确定时的最高修正上限 (inject_weight ≈ 1.0)
        self._adaptive_ratio_base = 0.08   # 8%: 基座确定时仅允许微弱修正
        self._adaptive_ratio_max = 0.30    # 30%: 基座犯错时允许充分修正

        self._init_weights()

    def _init_weights(self):
        """方案 A: 放大初始化，确保梯度信号有效传播 (bf16 安全)"""
        nn.init.normal_(self.q_proj_down.weight, std=0.02)
        nn.init.normal_(self.out_proj_up.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, hidden_size] 冻结 base model 的 hidden
        draft_states: torch.Tensor,    # [B, T, K_d, d_z] 每个 token 对应的 draft state
        inject_weight: torch.Tensor = None,  # [B, T, 1] token 级注入权重 (选择性注入)
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, hidden_size] — 冻结 base model 最后一层的 hidden
            draft_states: [B, T, K_d, d_z] — 每个 token 位置对应的 step-level draft
            inject_weight: [B, T, 1] — 可选, token 级注入权重 (0~1)
                           基座越确定的 token 权重越小, 越不确定的权重越大
                           None 时等价于全 1.0 (不做选择性注入)
        
        Returns:
            adapted_hidden: [B, T, hidden_size] — 适配后的 hidden (可直接过 lm_head)
        """
        B, T, H = hidden_states.shape
        K_d = draft_states.shape[2]
        param_dtype = self.q_proj.weight.dtype

        hidden_states = hidden_states.to(param_dtype)
        draft_states = draft_states.to(param_dtype)

        # 1. 投影 Q: [B, T, hidden_size] → [B, T, d_z]
        q = self.q_proj_down(hidden_states)  # [B, T, d_z]

        # 2. Pre-norm
        q = self.q_norm(q)
        # draft_states: [B, T, K_d, d_z] → reshape for batch cross-attention
        # 将 B*T 作为 batch 维
        q_flat = q.reshape(B * T, 1, self.d_z)  # [B*T, 1, d_z]
        kv_flat = draft_states.reshape(B * T, K_d, self.d_z)  # [B*T, K_d, d_z]
        kv_flat = self.kv_norm(kv_flat)

        # 3. Q, K, V 投影
        q_heads = self.q_proj(q_flat).view(B * T, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = self.k_proj(kv_flat).view(B * T, K_d, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = self.v_proj(kv_flat).view(B * T, K_d, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. Scaled Dot-Product Attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_heads.dtype)
        attn_output = torch.matmul(attn_weights, v_heads)  # [B*T, num_heads, 1, head_dim]

        # 5. 合并头并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * T, 1, self.d_z)
        attn_output = self.o_proj(attn_output)  # [B*T, 1, d_z]
        attn_output = attn_output.view(B, T, self.d_z)  # [B, T, d_z]

        # 6. 投影回 hidden_size 并残差连接
        adaptation = self.out_proj_up(attn_output)  # [B, T, hidden_size]
        adaptation = self.out_norm(adaptation)

        # Scale 软约束: 使用 clamp 防止极端值, 但主要靠 L2 正则化 (在 loss 中)
        _max_scale = getattr(self, '_max_scale', None)
        if _max_scale is not None:
            clamped_scale = self.scale.clamp(max=_max_scale)
        else:
            clamped_scale = self.scale

        # 修正量 = scale * adaptation
        delta = clamped_scale * adaptation  # [B, T, hidden_size]

        # ====== 自适应范数比约束: 基座越不确定 → 允许越大的修正量 ======
        # 旧方案: 固定 15% 上限, 与选择性注入串联叠加导致实际修正量被过度压缩
        # 新方案: max_ratio_per_token = base_ratio + (max_ratio - base_ratio) × inject_weight
        #   - 基座确定 (inject_weight ≈ 0.05): max_ratio ≈ 8%, 非常保守
        #   - 基座不确定 (inject_weight ≈ 0.95): max_ratio ≈ 29%, 充分修正
        # 两个约束协同工作: 选择性注入决定"要不要修", 范数比约束决定"修多少"
        _base_ratio = getattr(self, '_adaptive_ratio_base', 0.08)
        _max_ratio = getattr(self, '_adaptive_ratio_max', 0.30)

        with torch.no_grad():
            delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, T, 1]
            hidden_norm = hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, T, 1]
            ratio = delta_norm / hidden_norm  # [B, T, 1]

            if inject_weight is not None:
                # 自适应上限: 根据 inject_weight 线性插值
                # inject_weight: [B, T, 1], 范围 [0.05, 1.0]
                _iw = inject_weight.to(delta.dtype)
                adaptive_max_ratio = _base_ratio + (_max_ratio - _base_ratio) * _iw  # [B, T, 1]
            else:
                # 无选择性注入时, 使用保守的固定上限 (介于 base 和 max 之间)
                adaptive_max_ratio = (_base_ratio + _max_ratio) / 2.0  # 标量 19%

            # 只对超过自适应上限的 token 做缩放, 不影响正常范围内的修正
            shrink = (adaptive_max_ratio / ratio).clamp(max=1.0)  # [B, T, 1], ≤1.0
        delta = delta * shrink  # 保留梯度 (shrink 是 no_grad 的常量)

        # ====== 选择性注入: 在 adapter 内部直接做 token 级加权 ======
        # inject_weight 由 model.py 根据基座置信度计算并传入
        # 基座越确定的 token → 权重越小 → 修正量越小 (源头控制, 而非事后衰减)
        # 注意: 自适应范数比约束已经考虑了 inject_weight, 这里的乘法是额外的源头控制
        # 两者协同: 范数比约束限制"修正量上限", 选择性注入控制"实际注入比例"
        #
        # ★ 使用 sqrt(inject_weight) 缓和二次方衰减:
        # 范数比约束已经用了 inject_weight 做自适应上限, 如果选择性注入再乘一次原始值,
        # 实际修正量 ∝ inject_weight² (二次方衰减), 导致中等不确定度的 token 修正量过小。
        # 例如 inject_weight=0.1 时: 原方案实际修正 ≈ 1%, 改用 sqrt 后 ≈ 3.2%
        # 这让 draft 在基座"有点犹豫"的位置也能提供有意义的辅助。
        if inject_weight is not None:
            delta = delta * inject_weight.to(delta.dtype).sqrt()  # [B, T, hidden_size]

        adapted_hidden = hidden_states + delta

        return adapted_hidden


class MultiLayerDraftReadout(nn.Module):
    """
    多层 Draft Readout (方案 A+C)
    
    在多个 Transformer 中间层 + 最后一层分别放置 DraftReadoutAdapter，
    将各层的修正量叠加到最终 hidden states 上。
    
    设计动机:
    - 单层 readout 只在最后一层做修正，对中间表示没有引导能力
    - 多层 readout 从不同深度的 hidden states 提取信息，提供更丰富的梯度路径
    - 浅层 adapter 的 scale 更小 (渐进式影响)，避免初始干扰叠加过大
    - 仍然只需一次 base model forward (output_hidden_states=True)
    
    默认选取层索引: [L//4, L//2, 3L//4, L-1]
    对于 36 层模型: [9, 18, 27, 35]
    
    Args:
        hidden_size: 基座模型隐藏维度
        d_z: controller 空间维度
        num_heads: cross-attention 头数
        total_layers: 基座模型总层数
        readout_layer_indices: 要放置 readout adapter 的层索引列表
                               如果为 None，自动选取 [L//4, L//2, 3L//4, L-1]
    """

    def __init__(
        self,
        hidden_size: int = 3584,
d_z: int = 768,
        num_heads: int = 8,
        total_layers: int = 36,
        readout_layer_indices: list = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.total_layers = total_layers

        # 确定要放置 readout adapter 的层索引
        if readout_layer_indices is None:
            self.layer_indices = [
                total_layers // 4,      # 浅层 (第 9 层)
                total_layers // 2,      # 中层 (第 18 层)
                3 * total_layers // 4,  # 深层 (第 27 层)
                total_layers - 1,       # 最后一层 (第 35 层)
            ]
        else:
            self.layer_indices = sorted(readout_layer_indices)

        self.num_readout_layers = len(self.layer_indices)

        # 为每个层创建独立的 DraftReadoutAdapter
        self.adapters = nn.ModuleList([
            DraftReadoutAdapter(
                hidden_size=hidden_size,
                d_z=d_z,
                num_heads=num_heads,
            )
            for _ in range(self.num_readout_layers)
        ])

        # 渐进式 scale 初始化: 浅层小、深层大
        # 例如 4 层: [0.1, 0.3, 0.5, 1.0]
        scale_values = self._compute_progressive_scales()
        for i, adapter in enumerate(self.adapters):
            adapter.scale = nn.Parameter(torch.tensor(scale_values[i]))

    def _compute_progressive_scales(self) -> list:
        """
        计算渐进式 scale 初始值
        
        浅层的 hidden states 语义信息不如深层丰富，给较小的 scale；
        深层（最后一层）给最大的 scale。
        
        策略: 线性递增，从 0.1 到 1.0
        """
        n = self.num_readout_layers
        if n == 1:
            return [1.0]
        # 线性插值: 0.1, ..., 1.0
        return [0.1 + 0.9 * i / (n - 1) for i in range(n)]

    @property
    def scale(self):
        """返回最后一层 adapter 的 scale (兼容旧接口的监控代码)"""
        return self.adapters[-1].scale

    def forward(
        self,
        all_hidden_states: tuple,       # tuple of [B, T, hidden_size], 来自 output_hidden_states=True
        last_hidden_state: torch.Tensor, # [B, T, hidden_size], 最后一层的 hidden states
        draft_states: torch.Tensor,      # [B, T, K_d, d_z] 每个 token 对应的 draft state
    ) -> torch.Tensor:
        """
        多层 readout: 从各层提取修正量并叠加到最后一层的 hidden states 上
        
        公式:
            adapted_hidden = h_L + Σ_l adapter_l(h_l, Z_d)
        
        注意: 每个 adapter 内部已经包含了 scale * adaptation 的计算，
        这里的叠加是在 hidden_size 空间上的残差叠加。
        
        Args:
            all_hidden_states: tuple of [B, T, H], Transformer 各层的 hidden states
                               索引 0 是 embedding 层输出, 索引 1~L 是各 decoder 层输出
            last_hidden_state: [B, T, H], 最后一层 hidden (= all_hidden_states[-1])
            draft_states: [B, T, K_d, d_z] token 级 draft state
        
        Returns:
            adapted_hidden: [B, T, hidden_size] 适配后的 hidden
        """
        B, T, H = last_hidden_state.shape
        param_dtype = self.adapters[0].q_proj.weight.dtype
        adapted_hidden = last_hidden_state.to(param_dtype)

        # 调试: 延迟初始化 + 计数器控制打印频率
        global _DEBUG_MULTILAYER
        if _DEBUG_MULTILAYER is None:
            _DEBUG_MULTILAYER = _should_debug_print()
        if not hasattr(self, '_debug_fwd_count'):
            self._debug_fwd_count = 0
        self._debug_fwd_count += 1
        _do_print = _DEBUG_MULTILAYER and (self._debug_fwd_count <= 3 or self._debug_fwd_count % 50 == 0)

        if _do_print:
            print("\n" + "─" * 70)
            print(f"🔍 [MultiLayerDraftReadout] forward #{self._debug_fwd_count}")
            print(f"   输入: last_hidden_state={list(last_hidden_state.shape)}, "
                  f"draft_states={list(draft_states.shape)}")
            print(f"   Readout 层索引: {self.layer_indices} (共 {self.num_readout_layers} 层)")
            _original_norm = last_hidden_state.detach().float().norm(dim=-1).mean().item()
            print(f"   原始 hidden L2 范数均值: {_original_norm:.4f}")

        _layer_debug_info = []  # 收集每层的调试信息

        for i, (layer_idx, adapter) in enumerate(zip(self.layer_indices, self.adapters)):
            # all_hidden_states 的索引: 0=embedding, 1=layer0, ..., L=layerL-1
            # 所以 layer_idx 对应 all_hidden_states[layer_idx + 1]
            h_l = all_hidden_states[layer_idx + 1]  # [B, T, H]

            if layer_idx == self.total_layers - 1:
                # 最后一层: adapter 内部做 h + scale * adaptation
                # 但我们需要的是纯 adaptation (不含原始 h)
                # 因为 adapted_hidden 已经是 last_hidden_state 了
                adaptation_l = adapter(h_l, draft_states) - h_l.to(param_dtype)
            else:
                # 中间层: adapter 返回 h_l + scale * adaptation
                # 我们只需要 scale * adaptation 部分
                adaptation_l = adapter(h_l, draft_states) - h_l.to(param_dtype)

            adapted_hidden = adapted_hidden + adaptation_l

            # 收集每层调试信息 (detach, 不影响梯度)
            if _do_print:
                with torch.no_grad():
                    _adapt_norm = adaptation_l.detach().float().norm(dim=-1).mean().item()
                    _h_l_norm = h_l.detach().float().norm(dim=-1).mean().item()
                    _scale_val = adapter.scale.detach().item()
                    _ratio = _adapt_norm / max(_original_norm, 1e-8)
                    _layer_debug_info.append({
                        'layer_idx': layer_idx,
                        'scale': _scale_val,
                        'h_l_norm': _h_l_norm,
                        'adaptation_norm': _adapt_norm,
                        'ratio_to_original': _ratio,
                    })

        if _do_print:
            _total_adapt_norm = (adapted_hidden - last_hidden_state.to(param_dtype)).detach().float().norm(dim=-1).mean().item()
            print(f"\n   📊 各层 Adapter 注入详情:")
            _hdr = "   %6s | %8s | %10s | %12s | %12s" % ("层索引", "scale", "h_l范数", "修正量范数", "修正/原始比")
            print(_hdr)
            _sep = "   " + "─" * 6 + " | " + "─" * 8 + " | " + "─" * 10 + " | " + "─" * 12 + " | " + "─" * 12
            print(_sep)
            for info in _layer_debug_info:
                print(f"   L{info['layer_idx']:>4d} | {info['scale']:>8.4f} | {info['h_l_norm']:>10.4f} | "
                      f"{info['adaptation_norm']:>12.6f} | {info['ratio_to_original']:>12.6f}")
            print(_sep)
            print(f"   总修正量 L2 范数均值: {_total_adapt_norm:.6f}")
            print(f"   总修正/原始比: {_total_adapt_norm / max(_original_norm, 1e-8):.6f}")
            # 验证修正量非零 (证明 draft 确实在影响 CoT)
            if _total_adapt_norm > 1e-8:
                print("   ✅ 多层 Draft 注入生效: 修正量非零, 正在影响 CoT 推理")
            else:
                print("   ⚠️ 多层 Draft 注入量极小, 可能尚未学到有效修正")
            # 检查层间修正量的差异性 (证明不同层确实提供了不同的信息)
            if len(_layer_debug_info) > 1:
                norms = [info['adaptation_norm'] for info in _layer_debug_info]
                _norm_std = (sum((n - sum(norms)/len(norms))**2 for n in norms) / len(norms)) ** 0.5
                _diversity_tag = "✅ 各层差异化" if _norm_std > 1e-6 else "⚠️ 各层同质化"
                print(f"   层间修正量标准差: {_norm_std:.6f} ({_diversity_tag})")
            print("   " + "─" * 70)

        return adapted_hidden

    def forward_single_token(
        self,
        hidden_states: torch.Tensor,  # [B, 1, hidden_size] 当前 token 的最后一层 hidden
        draft_states: torch.Tensor,    # [B, 1, K_d, d_z] 当前 token 的 draft state
    ) -> torch.Tensor:
        """
        推理时的单 token readout (仅使用最后一层 adapter)
        
        推理时无法获取中间层 hidden states (KV cache 模式下不保存)，
        因此退化为只使用最后一层的 readout adapter。
        
        这是合理的近似:
        - 训练时多层 readout 主要帮助梯度传播和参数学习
        - 推理时最后一层的 adapter 已经从训练中学到了足够的调制能力
        - 中间层 adapter 的 scale 较小，缺失的影响有限
        
        Args:
            hidden_states: [B, 1, H] 最后一层的 hidden
            draft_states: [B, 1, K_d, d_z] draft state
        
        Returns:
            adapted_hidden: [B, 1, H]
        """
        # 使用最后一层的 adapter (scale 最大，影响最显著)
        return self.adapters[-1](hidden_states, draft_states)


class InSituDraftInjector(nn.Module):
    """
    In-Situ Draft 注入器 (Top-K Rerun 方案)
    
    核心思想: 在 Pass 2 中重算最后 K 层 decoder layers，在指定注入点插入 DraftReadoutAdapter。
    修正后的 hidden state 继续参与后续层的 self-attention（KV 被真正改写）。
    
    这解决了 post-hoc 方案的根本问题:
    - post-hoc: h̃_t = h_t + adapter(h_t, Z_d), 但 h_{t+1} 从未见过 h̃_t
    - in-situ: 修正后的 h̃_t 进入后续层的 self-attention KV, h_{t+1} 能看到修正后的上下文
    
    Phase 1: K=1 (仅最后一层 rerun + 注入), 验证管线通畅
    Phase 2: K=8 (重算最后 8 层, 注入 L28+L35), 真正因果一致
    Phase 3: K=8, 3 注入点 (L28+L31+L35), 完整方案
    
    训练时: Pass 1 (no_grad, 36层) → Controller → Pass 2 (有梯度, 最后K层 rerun + adapter注入)
    推理时: register_forward_hook 在注入层自动修正 hidden (KV cache 自然包含修正值)
    
    Args:
        hidden_size: 基座模型隐藏维度 (4096 for Qwen3-VL-8B)
        d_z: controller 空间维度 (512)
        num_heads: cross-attention 头数
        total_layers: 基座模型总层数 (36)
        rerun_k: 重算的层数 K (Phase 1: K=1, Phase 2+: K=8)
        injection_offsets: 在 rerun 的 K 层中的注入偏移量列表
                           Phase 1: [0] (仅最后一层)
                           Phase 2: [0, K-1]
                           Phase 3: [0, K//2-1, K-1]
    """

    def __init__(
        self,
        hidden_size: int = 4096,
d_z: int = 768,
        num_heads: int = 8,
        total_layers: int = 36,
        rerun_k: int = 1,
        injection_offsets: list = None,
        max_scale: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.total_layers = total_layers
        self.rerun_k = rerun_k
        self.rerun_start = total_layers - rerun_k  # Phase 1: 35, Phase 2+: 28
        self.max_scale = max_scale  # Scale 上界约束

        # 注入点: 在重算的 K 层中选取
        if injection_offsets is None:
            if rerun_k == 1:
                injection_offsets = [0]  # 仅最后一层
            else:
                injection_offsets = [0, rerun_k - 1]  # 首尾两层
        self.injection_offsets = injection_offsets
        self.injection_layer_indices = [self.rerun_start + off for off in injection_offsets]

        # 为每个注入层创建独立的 DraftReadoutAdapter
        self.adapters = nn.ModuleDict({
            str(idx): DraftReadoutAdapter(hidden_size, d_z, num_heads)
            for idx in self.injection_layer_indices
        })

        # 渐进式 scale 初始化: 从较小值开始，让 adapter 逐步学习修正量
        # 旧值 [0.3, 0.65, 1.0] 导致修正/原始比高达 0.83，严重干扰基座模型
        # 新值 [0.10, 0.15, 0.20] 确保初始修正量约为原始 hidden 的 10%~20%
        n = len(self.injection_layer_indices)
        if n == 1:
            scales = [0.15]
        else:
            scales = [0.10 + 0.10 * i / (n - 1) for i in range(n)]
        for i, idx in enumerate(self.injection_layer_indices):
            self.adapters[str(idx)].scale = nn.Parameter(torch.tensor(scales[i]))
            # 将 max_scale 传递给每个 adapter, 用于 forward 中的 clamp
            self.adapters[str(idx)]._max_scale = max_scale

        # 调试计数器
        self._debug_fwd_count = 0

    @property
    def scale(self):
        """返回最后一个注入点 adapter 的 scale (兼容旧接口的监控代码)"""
        last_idx = str(self.injection_layer_indices[-1])
        return self.adapters[last_idx].scale

    @property
    def layer_indices(self):
        """兼容 MultiLayerDraftReadout 的接口"""
        return self.injection_layer_indices

    def inject(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,    # [B, T, hidden_size]
        Z_d_expanded: torch.Tensor,     # [B, T, K_d, d_z]
        inject_weight: torch.Tensor = None,  # [B, T, 1] token 级注入权重 (选择性注入)
    ) -> torch.Tensor:
        """
        在指定层执行 in-situ adapter 注入 (支持源头选择性注入)
        
        如果 layer_idx 不是注入层, 直接返回原始 hidden_states (无修改)。
        如果是注入层, 返回 adapter(hidden_states, Z_d_expanded, inject_weight)。
        
        inject_weight 由 model.py 根据基座置信度计算:
        - 基座确定的 token → 权重小 → adapter 修正量被压缩 (源头控制)
        - 基座不确定的 token → 权重大 → adapter 充分修正
        
        Args:
            layer_idx: 当前 decoder layer 的索引 (0~35)
            hidden_states: [B, T, hidden_size] 当前层的输出
            Z_d_expanded: [B, T, K_d, d_z] token 级 draft state
            inject_weight: [B, T, 1] 可选, token 级注入权重
        
        Returns:
            hidden_states: [B, T, hidden_size] (注入后的, 或原始的)
        """
        key = str(layer_idx)
        if key in self.adapters:
            return self.adapters[key](hidden_states, Z_d_expanded, inject_weight=inject_weight)
        return hidden_states

    def inject_single_token(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,    # [B, 1, hidden_size]
        draft_states: torch.Tensor,     # [B, 1, K_d, d_z]
        inject_weight: torch.Tensor = None,  # [B, 1, 1] token 级注入权重 (推理时选择性注入)
    ) -> torch.Tensor:
        """
        推理时的单 token in-situ 注入 (用于 forward hook)
        
        支持选择性注入: 推理时也可传入 inject_weight, 保证训推一致。
        
        Args:
            layer_idx: decoder layer 索引
            hidden_states: [B, 1, hidden_size]
            draft_states: [B, 1, K_d, d_z]
            inject_weight: [B, 1, 1] 可选, token 级注入权重
        
        Returns:
            hidden_states: [B, 1, hidden_size]
        """
        key = str(layer_idx)
        if key in self.adapters:
            return self.adapters[key](hidden_states, draft_states, inject_weight=inject_weight)
        return hidden_states

    def forward(
        self,
        all_hidden_states: tuple,
        last_hidden_state: torch.Tensor,
        draft_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        兼容 MultiLayerDraftReadout 的 forward 接口 (用于不需要 rerun 的场景)
        
        注意: 在 Top-K Rerun 模式下, 这个方法通常不会被调用。
        真正的注入发生在 model.py 的 Pass 2 循环中通过 self.inject() 调用。
        
        此方法仅作为 fallback / 兼容层保留。
        """
        B, T, H = last_hidden_state.shape
        param_dtype = self.adapters[str(self.injection_layer_indices[-1])].q_proj.weight.dtype
        adapted_hidden = last_hidden_state.to(param_dtype)

        for layer_idx in self.injection_layer_indices:
            adapter = self.adapters[str(layer_idx)]
            # 注入: 提取纯 adaptation 量 (adapter 内部做了 h + scale*adapt, 所以减去 h)
            if all_hidden_states is not None and all_hidden_states[layer_idx + 1] is not None:
                h_l = all_hidden_states[layer_idx + 1]
            else:
                h_l = last_hidden_state
            adaptation_l = adapter(h_l, draft_states) - h_l.to(param_dtype)
            adapted_hidden = adapted_hidden + adaptation_l

        return adapted_hidden

    def forward_single_token(
        self,
        hidden_states: torch.Tensor,    # [B, 1, hidden_size]
        draft_states: torch.Tensor,     # [B, 1, K_d, d_z]
        inject_weight: torch.Tensor = None,  # [B, 1, 1] token 级注入权重 (推理时选择性注入)
    ) -> torch.Tensor:
        """
        推理时的单 token forward (兼容 MultiLayerDraftReadout 的接口)
        
        在推理路径使用 forward hook 时, 此方法不会被直接调用。
        但为了 prefill 阶段的兼容性 (还没注册 hook 时), 保留此接口。
        支持选择性注入: 传入 inject_weight 保证训推一致。
        """
        # 使用最后一个注入点的 adapter
        last_idx = str(self.injection_layer_indices[-1])
        return self.adapters[last_idx](hidden_states, draft_states, inject_weight=inject_weight)
