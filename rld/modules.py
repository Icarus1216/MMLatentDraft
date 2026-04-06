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
    
    输出经过 RMSNorm 归一化, 防止 CA 残差连接导致范数爆炸。
    (训练 400 步后 S_c 范数从 ~27 爆炸到 ~35000, 根因是 CA 内部
     query + attn_output + FFN 的残差叠加。加 RMSNorm 后范数稳定在 ~sqrt(d_z)≈27)
    
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

        # 可学习的 step queries (正交初始化, 确保 16 个 query 方向完全分化)
        # 旧版 randn*0.02 范数太小 (~0.55), softmax 无法区分方向 → S_c 有效秩 ~1.4
        # 正交初始化: 每行范数 ≈ 1.0, 行间严格正交 → 初始有效秩 ≈ num_trace_slots
        _step_queries_data = torch.empty(num_trace_slots, d_z)
        nn.init.orthogonal_(_step_queries_data)
        self.step_queries = nn.Parameter(_step_queries_data.unsqueeze(0))  # [1, K_t, d_z]

        # 单层 cross-attention (step 摘要不需要太深)
        self.ca_block = CrossAttentionBlock(d_model=d_z, num_heads=num_heads)

        # 输出归一化: 防止 CA 残差连接导致 S_c 范数爆炸
        # CA 内部: output = query + attn(query, kv) + FFN(...)  → 范数随训练不断增长
        # 加 RMSNorm 后: S_c 范数稳定在 ~sqrt(d_z) ≈ 27, 不会随训练爆炸
        self.output_norm = RMSNorm(d_z)

    def forward(self, step_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_hidden_states: [B, L_c, hidden_size] 当前 step 的最后层 hidden states
        
        Returns:
            S_c: [B, K_t, d_z] 当前 step 的摘要 (已归一化, 范数 ~sqrt(d_z))
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

        # 输出归一化: 防止范数爆炸
        S_c = self.output_norm(S_c)

        return S_c  # [B, K_t, d_z]


class StreamingTraceAccumulator(nn.Module):
    """
    流式轨迹累积器: 每 token 增量更新 running summary S_running
    
    用 GRU Cell 替代 StepResampler 的 batch Cross-Attention:
    1. 每次只处理 1 个 token，CA 的 KV 只有 1 行，退化为线性变换
    2. GRU 天然支持流式输入，无需缓存历史 hidden
    3. 参数量极小：3 * d_z * d_z ≈ 1.8M (d_z=768)
    
    设计:
    - 每个 slot 独立的 GRU cell (共享参数)
    - 输入: 当前 token 的 hidden state h_t (投影到 d_z)
    - 输出: 更新后的 running summary S_running
    
    只在句末 (句号/换行/冒号等标点) 计算 CommitGate 分数,
    保持句内连贯性, 避免句中 commit 打断推理流。
    
    Args:
        hidden_size: 基座模型隐藏维度 (4096 for Qwen3-VL-8B)
        d_z: controller 空间维度 (768)
        num_trace_slots: 轨迹槽数量 K_t (默认 16)
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        d_z: int = 768,
        num_trace_slots: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.num_slots = num_trace_slots

        # 从 hidden_size 投影到 d_z (与 StepResampler 共享投影维度)
        self.proj_h = nn.Linear(hidden_size, d_z, bias=False)

        # GRU Cell: 所有 slot 共享参数, 每个 slot 独立更新
        # GRU 参数量: 3 * (d_z * d_z + d_z * d_z) = 6 * d_z^2 ≈ 3.5M
        self.gru = nn.GRUCell(d_z, d_z)

        # 初始化: 小范围初始化 GRU 权重, 确保初始 S_running 变化平缓
        self._init_weights()

    def _init_weights(self):
        """小范围初始化, 确保初始 GRU 输出接近输入 (近似恒等映射)"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        h_t: torch.Tensor,          # [B, 1, hidden_size] 当前 token 的 hidden state
        S_running: torch.Tensor,     # [B, K_t, d_z] 当前的 running summary
    ) -> torch.Tensor:
        """
        单 token 增量更新 S_running
        
        Args:
            h_t: [B, 1, hidden_size] 当前 token 的 hidden state
            S_running: [B, K_t, d_z] 当前的 running summary
        
        Returns:
            S_running_new: [B, K_t, d_z] 更新后的 running summary
        """
        B, K, D = S_running.shape
        param_dtype = self.proj_h.weight.dtype
        h_t = h_t.to(param_dtype)
        S_running = S_running.to(param_dtype)

        # 投影到 d_z 空间
        h_proj = self.proj_h(h_t.squeeze(1))  # [B, d_z]

        # 广播到所有 slot: 每个 slot 接收相同的输入, 但有不同的隐状态
        h_expand = h_proj.unsqueeze(1).expand(B, K, D)  # [B, K, d_z]

        # GRU: 每个 slot 独立更新 (共享 GRU 参数)
        S_flat = S_running.reshape(B * K, D)      # [B*K, d_z]
        h_flat = h_expand.reshape(B * K, D)        # [B*K, d_z]
        S_new_flat = self.gru(h_flat, S_flat)      # [B*K, d_z]
        S_running_new = S_new_flat.reshape(B, K, D)

        return S_running_new

    def forward_batch(
        self,
        hidden_states: torch.Tensor,   # [B, L, hidden_size] 多 token 的 hidden states
        S_running: torch.Tensor,        # [B, K_t, d_z] 初始 running summary
    ) -> torch.Tensor:
        """
        批量处理多个 token (训练时使用, 沿序列维度逐 token 展开 GRU)
        
        Args:
            hidden_states: [B, L, hidden_size] 一段 token 的 hidden states
            S_running: [B, K_t, d_z] 初始 running summary
        
        Returns:
            S_running_final: [B, K_t, d_z] 处理完所有 token 后的 running summary
        """
        B, L, H = hidden_states.shape
        K, D = S_running.shape[1], S_running.shape[2]
        param_dtype = self.proj_h.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        S_running = S_running.to(param_dtype)

        # 投影所有 token 到 d_z 空间
        h_proj = self.proj_h(hidden_states)  # [B, L, d_z]

        # 沿序列维度逐 token 展开 GRU
        S_current = S_running
        for t in range(L):
            h_t = h_proj[:, t, :]  # [B, d_z]
            h_expand = h_t.unsqueeze(1).expand(B, K, D)  # [B, K, d_z]
            S_flat = S_current.reshape(B * K, D)
            h_flat = h_expand.reshape(B * K, D)
            S_new_flat = self.gru(h_flat, S_flat)
            S_current = S_new_flat.reshape(B, K, D)

        return S_current


class CommitGate(nn.Module):
    """
    学习型 Commit 门控: 决定何时将 S_running commit 到完整的 draft 更新链路
    
    灵感来自 Adaptive Computation Time (ACT) 的 halting score:
    - 不累积 halting probability, 而是直接输出 commit_score ∈ (0,1)
    - commit_score > τ 时触发 commit
    - τ 是可调超参 (训练时用 teacher-forced boundary 做监督)
    
    只在句末计算 commit_score, 保持句内连贯性:
    - 句末标记: 句号(。.)、换行(\\n)、冒号(:)、分号(;) 等
    - 句内 token 不计算 commit_score, 直接跳过
    - 这避免了句中 commit 打断推理流的问题
    
    输入信号 (3 个 d_z 维向量拼接):
    1. S_running 的变化率 (delta_S): 信息饱和度指标
    2. S_running 与 Z_d 的差异 (staleness): draft 是否过时
    3. S_running 自身的信息量 (S_info): 当前累积的信息量
    
    Args:
        d_z: controller 空间维度 (768)
        commit_threshold: commit 阈值 τ (默认 0.5)
        min_tokens_between_commits: 两次 commit 之间的最小 token 数 (默认 8)
        max_tokens_between_commits: 两次 commit 之间的最大 token 数 (默认 128)
    """

    def __init__(
        self,
        d_z: int = 768,
        commit_threshold: float = 0.5,
        min_tokens_between_commits: int = 8,
        max_tokens_between_commits: int = 128,
    ):
        super().__init__()
        self.d_z = d_z
        self.commit_threshold = commit_threshold
        self.min_tokens = min_tokens_between_commits
        self.max_tokens = max_tokens_between_commits

        # Commit score 预测头: 3*d_z → d_z → 1
        self.score_head = nn.Sequential(
            RMSNorm(d_z * 3),
            nn.Linear(d_z * 3, d_z, bias=False),
            nn.SiLU(),
            nn.Linear(d_z, 1, bias=True),
        )
        # 初始 bias 偏低 (-2.0), 训练初期倾向于不 commit (保守策略)
        # sigmoid(-2.0) ≈ 0.12, 远低于默认阈值 0.5
        nn.init.constant_(self.score_head[-1].bias, -2.0)
        # 小范围初始化权重, 确保初始输出稳定
        nn.init.normal_(self.score_head[1].weight, std=0.02)
        nn.init.normal_(self.score_head[-1].weight, std=0.02)

    def forward(
        self,
        S_running: torch.Tensor,    # [B, K_t, d_z] 当前 running summary
        S_prev: torch.Tensor,       # [B, K_t, d_z] 上次 commit 时的 S_running
        Z_d: torch.Tensor,          # [B, K_d, d_z] 当前 draft state
    ) -> torch.Tensor:
        """
        计算 commit score
        
        Args:
            S_running: [B, K_t, d_z] 当前 running summary
            S_prev: [B, K_t, d_z] 上次 commit 时的 S_running (用于计算变化率)
            Z_d: [B, K_d, d_z] 当前 draft state (用于计算 staleness)
        
        Returns:
            commit_score: [B, 1] ∈ (0, 1), 越高越应该 commit
        """
        param_dtype = self.score_head[1].weight.dtype
        S_running = S_running.to(param_dtype)
        S_prev = S_prev.to(param_dtype)
        Z_d = Z_d.to(param_dtype)

        # 信号 1: S_running 的变化率 (信息饱和度)
        # 变化大 → 新信息多 → 可能需要 commit
        delta_S = (S_running - S_prev).mean(dim=1)  # [B, d_z]

        # 信号 2: S_running 与 Z_d 的差异 (draft 过时程度)
        # 差异大 → draft 过时 → 需要 commit 更新
        staleness = (S_running.mean(dim=1) - Z_d.mean(dim=1))  # [B, d_z]

        # 信号 3: S_running 自身的信息量
        S_info = S_running.mean(dim=1)  # [B, d_z]

        # 拼接 3 个信号
        combined = torch.cat([delta_S, staleness, S_info], dim=-1)  # [B, 3*d_z]

        # 预测 commit score
        commit_score = torch.sigmoid(self.score_head(combined))  # [B, 1]

        return commit_score

    def should_commit(
        self,
        commit_score: torch.Tensor,    # [B, 1]
        token_count: int,               # 自上次 commit 以来的 token 数
        is_sentence_end: torch.Tensor,  # [B] bool, 当前 token 是否为句末标记
    ) -> torch.Tensor:
        """
        综合判断是否应该 commit (推理时使用)
        
        触发条件 (满足任一):
        1. 句末 + commit_score > τ + token_count >= min_tokens
        2. token_count >= max_tokens (硬上限, 防止永不 commit)
        
        Args:
            commit_score: [B, 1] commit 分数
            token_count: 自上次 commit 以来的 token 数
            is_sentence_end: [B] bool, 当前 token 是否为句末标记
        
        Returns:
            should_commit: [B] bool, 是否应该 commit
        """
        B = commit_score.shape[0]
        device = commit_score.device

        # 条件 1: 句末 + 分数超阈值 + 最小间隔
        gate_trigger = (
            is_sentence_end
            & (commit_score.squeeze(-1) > self.commit_threshold)
            & (token_count >= self.min_tokens)
        )

        # 条件 2: 硬上限
        hard_trigger = torch.full((B,), token_count >= self.max_tokens,
                                  dtype=torch.bool, device=device)

        return gate_trigger | hard_trigger


class TraceUpdater(nn.Module):
    """
    [已弃用] 累计式推理轨迹更新器 (2 层 CA)
    
    保留用于加载旧 checkpoint, forward 中不再使用。
    新代码请使用 TraceEMA。
    
    问题: 2 层 CA 从 [T_prev; S_c] (32 tokens) 中重采样到 16 slots,
    这是一个 16→16 的方阵映射, 天然倾向秩坍塌 (T_rank=1.0)。
    
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


class TraceEMA(nn.Module):
    """
    EMA 轨迹更新器: 用 per-slot 可学习 α 做指数移动平均
    
    核心公式:
      T_c = α_i · T_{c-1} + (1 - α_i) · S_c
    
    其中 α_i ∈ (0, 1) 是 per-slot 可学习的记忆保留率:
    - α_i 大 → 保留更多历史 (长期记忆)
    - α_i 小 → 更多采纳当前 S_c (短期响应)
    - 每个 slot 独立学习自己的 α_i → 不同 slot 可以有不同的时间尺度
    
    设计动机 (替代 TraceUpdater 的 2 层 CA):
    - TraceUpdater 的 2 层 CA 从 [T_prev; S_c] (32 tokens) 重采样到 16 slots
      → 16→16 的方阵映射天然倾向秩坍塌 (T_rank=1.0)
    - EMA 是逐 slot 独立更新, 天然保持 slot 独立性:
      每个 slot 的更新只依赖自己的 T_prev[i] 和 S_c[i], 不会被其他 slot 污染
    - 参数量极小: 只有 16 个可学习的 α_i (vs TraceUpdater 的 ~2.4M)
    - 无超参数, 训练稳定
    
    初始化策略:
    - α_logit 初始化为 1.0 → sigmoid(1.0) ≈ 0.73
    - 训练初期偏向保留历史 (α≈0.73), 随训练自动调节
    - 这比 α=0.5 (等权) 更合理: 历史轨迹是多步累积的, 应该有更高的惯性
    
    Args:
        num_trace_slots: 轨迹槽数量 K_t (默认 16)
    """

    def __init__(self, num_trace_slots: int = 16):
        super().__init__()
        self.num_slots = num_trace_slots

        # per-slot 可学习的记忆保留率 α_i
        # sigmoid(1.0) ≈ 0.73: 训练初期偏向保留历史
        self.alpha_logit = nn.Parameter(torch.ones(1, num_trace_slots, 1) * 1.0)  # [1, K_t, 1]

    def forward(
        self,
        T_prev: torch.Tensor,   # [B, K_t, d_z] 上一步的轨迹记忆
        S_c: torch.Tensor,       # [B, K_t, d_z] 当前步的摘要
    ) -> torch.Tensor:
        """
        EMA 更新: T_c = α · T_prev + (1 - α) · S_c
        
        Returns:
            T_c: [B, K_t, d_z] 更新后的累计轨迹记忆
        """
        alpha = torch.sigmoid(self.alpha_logit)  # [1, K_t, 1], ∈ (0, 1)
        T_c = alpha * T_prev + (1.0 - alpha) * S_c
        return T_c  # [B, K_t, d_z]


class ReflectionModule(nn.Module):
    """
    回看-验证模块 (核心操作, 2 层 CA + ConsistencyHead)
    
    用累计轨迹 T_c 作为 query，去冻结的视觉证据 Z^e 中检索:
      G_c = CA_layers(Q=T_c, K=Z^e, V=Z^e)
    
    解释:
    - T_c 是"目前为止我们在推理什么/相信什么"的 latent
    - 对 Z^e 做 cross-attn 相当于:
      "用当前推理作为 query 去图像证据里找相关区域/数字/关系的 latent 支撑"
    - G_c 是 grounded trace (被视觉证据支撑/对齐后的推理表示)
    
    ConsistencyHead (新增):
    - 对比 T_c (纯推理轨迹) 和 G_c (被证据校验后的推理)
    - 差异 T_c - G_c 的语义: "推理中没有被视觉证据支撑的部分"
    - 差异越大 → 推理越偏离证据 → consistency_score 越低
    - consistency_score ∈ (0, 1), 用于下游 inject_weight 融合
    
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
        self.d_z = d_z
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model=d_z, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # ====== ConsistencyHead: 从 T_c - G_c 对比信号中提取一致性分数 ======
        # 输入: T_c - G_c 的 slot 级均值 [B, d_z]
        # 输出: consistency_score [B, 1] ∈ (0, 1)
        # 参数量: d_z * (d_z/4) + (d_z/4) * 1 ≈ 148K (d_z=768)
        self.consistency_head = nn.Sequential(
            RMSNorm(d_z),
            nn.Linear(d_z, d_z // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_z // 4, 1, bias=True),
        )
        # 初始 bias=0.85 → sigmoid(0.85) ≈ 0.70
        # 训练初期 consistency_score ≈ 0.7 (偏高), 不影响下游 inject_weight
        nn.init.constant_(self.consistency_head[-1].bias, 0.85)
        # 小范围初始化权重, 确保初始输出稳定
        nn.init.normal_(self.consistency_head[1].weight, std=0.02)
        nn.init.normal_(self.consistency_head[-1].weight, std=0.02)

    def forward(
        self,
        T_c: torch.Tensor,   # [B, K_t, d_z]
        Z_e: torch.Tensor,   # [B, K_e, d_z]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            G_c: [B, K_t, d_z] grounded trace (证据支撑的推理表示)
            consistency_score: [B, 1] ∈ (0, 1) 推理-证据一致性分数
                高分 → 推理与证据一致, 低分 → 推理偏离证据
        """
        G_c = T_c
        for layer in self.layers:
            G_c = layer(query=G_c, key_value=Z_e)

        # ====== ConsistencyHead: 计算推理-证据一致性分数 ======
        # 对比信号: T_c - G_c (推理中未被证据支撑的部分)
        # G_c = T_c + Δ (CA 内部有残差连接), 所以 T_c - G_c ≈ -Δ
        # 差异越大 → 推理越偏离证据 → consistency_score 越低
        contrast = (T_c - G_c).mean(dim=1)  # [B, d_z] slot 级均值
        consistency_score = torch.sigmoid(self.consistency_head(contrast))  # [B, 1]

        return G_c, consistency_score


class BidirectionalReflection(nn.Module):
    """
    双向检索-验证模块 v4 (二次检索验证 + 自适应纠错)
    
    核心思想: 去掉 16 slots 的 EvidenceResampler 压缩, 直接从原始 visual_hidden_proj
    (~500 个 token, 768 维) 中进行检索, 然后通过二次检索验证生成纠错残差。
    
    方向 1 (推理→图像 检索): CA(Q=S_c, KV=visual_hidden_proj) → R_c
        使用完整的 CrossAttentionBlock (含残差+FFN), 让 R_c = S_c + 视觉证据增量
        R_c 保留了 S_c 的推理信息, 同时融入了从图像中检索到的证据
    
    方向 2 (二次检索验证): CA(Q=R_c, KV=visual_hidden_proj) → V_c
        核心: 用融合了图像信息的 R_c 再次检索图像, 得到 V_c
        如果 R_c 正确融合了图像信息 → V_c ≈ R_c (二次检索结果一致)
        如果 R_c 融合了错误信息 → V_c ≠ R_c (二次检索发现不一致)
        diff = V_c - R_c 是真正的验证信号, 而非 v3 中的 R_c - S_c (仅反映检索量)
    
    一致性信号: 用 V_c 和 R_c 的差异驱动
        V_c 是二次检索结果, R_c 是一次检索结果, 差异 = 一次检索的不一致程度
        consistency_score 同时用于:
        1. 内部: 调制 gate_correct (纠错残差的力度)
        2. 外部: 调制 gc_inject_gate (G_c vs Z_d_init 的融合比例)
    
    v3 的问题 (基于真实数据分析):
    - diff = R_c - S_c 是第一层 CA 的增量 Δ_visual, 不区分"确认"和"矛盾"
    - Step 2 "FE=10" 正确引用图像 → Δ_visual 大 → cs 低 → 误判为"偏离"
    - ConsistencyHead 实际测量的是"图像中有多少相关信息", 而非"推理是否正确"
    
    v4 改进:
    - 第二层改为 R_c 对 visual_hidden_proj 做二次检索 → V_c
    - diff = V_c - R_c: 二次检索差异, 真正反映一次检索的一致性
    - 正确引用图像时: V_c ≈ R_c → diff 小 → cs 高 → 不纠错 ✅
    - 错误引用图像时: V_c ≠ R_c → diff 大 → cs 低 → 强纠错 ✅
    - 参数量从 ~1.2M 增至 ~2.4M (多一个 CA), 但验证信号语义正确
    
    Args:
        d_z: controller 空间维度 (768)
        num_heads: cross-attention 头数 (8)
    """

    def __init__(self, d_z: int = 768, num_heads: int = 8):
        super().__init__()
        self.d_z = d_z
        self.num_heads = num_heads
        self.head_dim = d_z // num_heads

        # 方向 1: 推理→图像 (一次检索) — 完整 CrossAttentionBlock (含残差+FFN)
        # Q=S_c [B, 16, 768], KV=visual_hidden_proj [B, ~500, 768]
        # R_c = S_c + attn(S_c, VH) + FFN, 保留推理信息 + 融入视觉证据
        self.retrieval_ca = CrossAttentionBlock(d_model=d_z, num_heads=num_heads)

        # ====== 层间归一化: 防止两层 CA 残差叠加导致范数爆炸 ======
        # CrossAttentionBlock 的残差连接 (R_c = S_c + Δ1 + FFN1) 会让范数持续增长
        # 不加归一化时, 训练 400 步后 R_c norm 从 ~27 爆炸到 ~25000
        # 导致 ConsistencyHead 输入饱和 → cs ≈ 0 恒定 → 纠错失去自适应能力
        # → G_c 被 output_norm 压缩后 slot 方向趋同 → Z_d 完全坍塌 (rank=1)
        self.inter_norm = RMSNorm(d_z)

        # ====== 方向 2: 二次检索验证 (Re-retrieval Verification) ======
        # 核心思想: 用 R_c (融合了图像信息的推理状态) 再次检索图像 → V_c
        # 如果一次检索正确: R_c 中的图像信息与原图一致 → V_c ≈ R_c → diff 小
        # 如果一次检索错误: R_c 中的图像信息与原图矛盾 → V_c ≠ R_c → diff 大
        # diff = V_c - R_c 是真正的验证信号
        self.verify_ca = CrossAttentionBlock(d_model=d_z, num_heads=num_heads)

        # 纠错 MLP: 将二次检索差异 (V_c - R_c) 转化为纠错残差
        self.correction_norm = RMSNorm(d_z)
        self.correction_mlp = nn.Sequential(
            nn.Linear(d_z, d_z * 2, bias=False),  # 差异 → 高维空间
            nn.SiLU(),
            nn.Linear(d_z * 2, d_z, bias=False),  # 高维 → 纠错残差
        )
        # 初始化: 让纠错残差初始时接近零, 避免训练初期干扰
        nn.init.normal_(self.correction_mlp[0].weight, std=0.02)
        nn.init.normal_(self.correction_mlp[2].weight, std=0.01)  # 输出层更小的初始化

        # 纠错力度的基础门控 (与 consistency_score 联合决定最终力度)
        # sigmoid(0.0) = 0.5, 初始时纠错力度适中
        self.correction_gate_logit = nn.Parameter(torch.tensor(0.0))

        # [兼容旧 checkpoint] 保留 mix_gate 参数, 但不再用于 forward
        self.mix_gate = nn.Parameter(torch.tensor(0.85))

        # 输出归一化: 防止范数爆炸
        self.output_norm = RMSNorm(d_z)

        # ConsistencyHead: 从 V_c 和 R_c 的差异中提取一致性分数
        # V_c 是二次检索结果, R_c 是一次检索结果
        # 差异大 → 一次检索不一致 (推理偏离) → consistency_score 低
        # 差异小 → 一次检索一致 (推理正确) → consistency_score 高
        # v4: 基于 V_c - R_c (二次检索差异), 而非 v3 的 S_c - R_c (一次检索增量)
        self.consistency_head = nn.Sequential(
            RMSNorm(d_z),
            nn.Linear(d_z, d_z // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_z // 4, 1, bias=True),
        )
        nn.init.constant_(self.consistency_head[-1].bias, 0.85)
        nn.init.normal_(self.consistency_head[1].weight, std=0.02)
        nn.init.normal_(self.consistency_head[-1].weight, std=0.02)

    def _compute_correction(
        self,
        R_c: torch.Tensor,          # [B, K_t, d_z] 一次检索结果 (含视觉证据)
        V_c: torch.Tensor,          # [B, K_t, d_z] 二次检索结果 (验证信号)
        consistency_score: torch.Tensor,  # [B, 1] 一致性分数
    ) -> torch.Tensor:
        """
        二次检索验证: 从 V_c 和 R_c 的差异中生成纠错残差
        
        核心逻辑:
        1. diff = V_c - R_c: 二次检索差异 (真正的验证信号)
           - diff 大 → 一次检索不一致 (推理偏离) → 需要强纠错
           - diff 小 → 一次检索一致 (推理正确) → 几乎不纠错
        2. Δ_correct = MLP(diff): 将差异转化为纠错残差
        3. gate_correct = base_gate * (1 - cs): 自适应纠错力度
           - cs 低 (检索不一致) → gate_correct 大 → 强纠错
           - cs 高 (检索一致) → gate_correct 小 → 弱纠错
        
        Returns:
            correction: [B, K_t, d_z] 纠错后的结果 (R_c + gate_correct * Δ_correct)
        """
        param_dtype = self.correction_mlp[0].weight.dtype
        R_c = R_c.to(param_dtype)
        V_c = V_c.to(param_dtype)

        # 差异向量: 二次检索与一次检索的不一致程度
        # V_c ≈ R_c 时 diff 小 (一次检索正确, 二次确认)
        # V_c ≠ R_c 时 diff 大 (一次检索有误, 二次发现矛盾)
        diff = V_c - R_c  # [B, K_t, d_z]
        diff_normed = self.correction_norm(diff)  # 归一化, 防止差异过大时梯度爆炸

        # MLP 将差异转化为纠错残差
        delta_correct = self.correction_mlp(diff_normed)  # [B, K_t, d_z]

        # 自适应纠错力度: base_gate * (1 - consistency_score)
        # cs 低 → 检索不一致 → 纠错力度大
        # cs 高 → 检索一致 → 纠错力度小 (保持 R_c 不变)
        base_gate = torch.sigmoid(self.correction_gate_logit)  # 标量 ∈ (0, 1)
        # consistency_score: [B, 1] → [B, 1, 1] 用于广播
        adaptive_gate = base_gate * (1.0 - consistency_score.unsqueeze(-1))  # [B, 1, 1]

        # 纠错: R_c + 自适应力度 * 纠错残差
        corrected = R_c + adaptive_gate * delta_correct  # [B, K_t, d_z]

        # 修正诊断信息 (detach, 不影响梯度)
        correction_diag = {
            'diff_norm': diff.detach().norm(dim=-1).mean().item(),           # V_c - R_c 差异范数 (越大=检索越不一致)
            'delta_norm': delta_correct.detach().norm(dim=-1).mean().item(), # MLP 输出的纠错残差范数
            'adaptive_gate': adaptive_gate.detach().mean().item(),           # 实际纠错力度 = base_gate * (1-cs)
            'base_gate': base_gate.item(),                                   # 可学习的基础门控
        }
        return corrected, correction_diag

    def forward(
        self,
        S_c: torch.Tensor,         # [B, K_t, d_z] 当前推理状态 (step 摘要或轨迹)
        visual_kv: torch.Tensor,    # [B, L_v, d_z] 投影后的完整视觉 hidden
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        双向检索-验证 v4 (二次检索验证 + 自适应纠错)
        
        信号流:
        1. R_c = CA(S_c, VH): 一次检索, 从图像中检索与推理相关的证据
        2. V_c = CA(R_c, VH): 二次检索, 用融合了图像信息的 R_c 再次检索图像
        3. consistency_score = ConsistencyHead(V_c - R_c): 检测二次检索一致性
           - V_c ≈ R_c → 一次检索正确 → cs 高
           - V_c ≠ R_c → 一次检索有误 → cs 低
        4. Δ_correct = MLP(V_c - R_c): 从二次检索差异中提取纠错残差
        5. G_c = RMSNorm(R_c + adaptive_gate * Δ_correct)
           其中 adaptive_gate = base_gate * (1 - consistency_score)
        
        Args:
            S_c: [B, K_t, d_z] 当前推理状态
            visual_kv: [B, L_v, d_z] 投影后的完整视觉 hidden (~500 tokens)
        
        Returns:
            G_c: [B, K_t, d_z] 二次检索验证后的 grounded trace (归一化后)
            consistency_score: [B, 1] ∈ (0, 1) 二次检索一致性分数
        """
        # 方向 1: 一次检索 — 推理→图像 (从 ~500 个原始 token 中精准检索)
        # R_c = S_c + attn(S_c, VH) + FFN, 保留推理信息 + 融入视觉证据
        R_c = self.retrieval_ca(query=S_c, key_value=visual_kv)  # [B, K_t, d_z]

        # 层间归一化: 防止两层 CA 残差叠加导致范数爆炸
        # R_c 经过 RMSNorm 后范数稳定在 ~sqrt(d_z) ≈ 27, 不会随训练增长
        R_c_normed = self.inter_norm(R_c)  # [B, K_t, d_z]

        # 方向 2: 二次检索验证 — 用归一化后的 R_c 再次检索图像
        # R_c_normed 已融合了图像证据且范数稳定, 用它再查图像:
        # - 如果一次检索正确 (如 "FE=10" 与图像一致): R_c 中的信息与图像吻合
        #   → 二次检索得到相似结果 → V_c ≈ R_c_normed → diff 小
        # - 如果一次检索有误 (如 "FE=8" 但图像标注 10): R_c 中有矛盾信息
        #   → 二次检索发现不一致 → V_c ≠ R_c_normed → diff 大
        V_c = self.verify_ca(query=R_c_normed, key_value=visual_kv)  # [B, K_t, d_z]

        # ConsistencyHead: 用 V_c 和 R_c_normed 的差异驱动
        # V_c - R_c_normed 是二次检索的增量, 反映一次检索结果的可靠性
        # 差异大 → 一次检索不可靠 → consistency_score 低 → 纠错信号强
        # 差异小 → 一次检索可靠 → consistency_score 高 → 保持不变
        # 注意: ConsistencyHead 内部第一层就是 RMSNorm, 所以 contrast 的范数已被控制
        contrast = (V_c - R_c_normed).mean(dim=1)  # [B, d_z]
        consistency_score = torch.sigmoid(self.consistency_head(contrast))  # [B, 1]

        # 纠错: 用二次检索差异生成纠错残差, consistency_score 调制纠错力度
        # cs 低 → 检索不一致 → 强纠错; cs 高 → 检索一致 → 弱纠错
        # 注意: 使用归一化后的 R_c_normed 和 V_c, 确保 _compute_correction 中的
        # diff = V_c - R_c_normed 范数稳定, 不会因范数爆炸导致 MLP 输出失控
        G_c_raw, correction_diag = self._compute_correction(R_c_normed, V_c, consistency_score)  # [B, K_t, d_z]

        # 输出归一化: 防止范数爆炸
        G_c = self.output_norm(G_c_raw)

        # 补充检索阶段的诊断信息
        correction_diag['R_c_norm'] = R_c.detach().norm(dim=-1).mean().item()           # 归一化前的 R_c 范数 (可能增长)
        correction_diag['R_c_normed_norm'] = R_c_normed.detach().norm(dim=-1).mean().item()  # 归一化后的 R_c 范数 (应稳定 ~27)
        correction_diag['V_c_norm'] = V_c.detach().norm(dim=-1).mean().item()

        return G_c, consistency_score, correction_diag


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
        max_norm = math.sqrt(self.d_z / 512) * 35  # 根据 d_z 动态计算
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
        max_norm = math.sqrt(self.d_z / 512) * 35  # 根据 d_z 动态计算
        slot_norms = Z_d_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, K_d, 1]
        clamped_scale = (max_norm / slot_norms).clamp(max=1.0)
        Z_d_new = Z_d_new * clamped_scale

        return Z_d_new


class CurrentDominantDraftRefiner(nn.Module):
    """
    Current-Dominant Draft Refiner: 以当前 G_c 为主体的 draft 更新器
    
    设计动机:
    - 旧方案 ResidualFlowDraftUpdater: Z_d_{c+1} = Z_d_c + Δt·v(Z_d_c, G_c, T_c, t)
      优点: per-slot CA 让 slot 分化; 缺点: 残差步进让历史 Z_d 稀释当前 G_c 的修正信号
    - 方案 B compute_prefix_source: prefix = gate·G_c + (1-gate)·Z_d_init
      优点: 当前 G_c 信号强; 缺点: 标量 gate 无法产生 per-slot 差异 → rank 坍塌
    
    本方案: 取两者之长
    - 以 G_c 为主体 (不是 Z_d_prev + 增量), 确保当前步的纠错信号不被历史稀释
    - 用 per-slot cross-attention 从 Z_d_prev 中选择性读取历史上下文 → slot 分化
    - 用 Z_d_init 提供 per-slot 身份偏置, 保持 slot 间的结构差异
    
    核心公式:
      history_ctx = CA(Q=G_c, KV=Z_d_prev)  # per-slot 从历史中读取不同信息
      slot_bias = Linear(Z_d_init)           # per-slot 身份偏置
      Z_d_new = output_norm(G_c + α · (history_ctx - G_c) + β · slot_bias)
    
    其中:
    - α: 可学习标量, 初始 ~0.1, 控制历史信息注入量 (小值 → 当前 G_c 主导)
    - β: 可学习标量, 初始 ~0.1, 控制 slot 身份偏置强度
    - CA 内部: 每个 G_c slot 作为独立 query, 从 Z_d_prev 的 16 个 slot 中
      检索不同的历史信息 → 不同 slot 获得不同的 history_ctx → slot 分化
    
    与旧方案的关键区别:
    - 旧方案: Z_d 是主体, G_c 是增量 → 历史累积, 当前信号被稀释
    - 本方案: G_c 是主体, Z_d_prev 是辅助 → 当前信号主导, 历史仅提供上下文
    
    Args:
        d_z: controller 空间维度 (768)
        num_heads: cross-attention 头数 (8)
    """

    def __init__(self, d_z: int = 768, num_heads: int = 8):
        super().__init__()
        self.d_z = d_z
        self.num_heads = num_heads
        self.head_dim = d_z // num_heads

        # ====== Per-slot 历史上下文读取 ======
        # Q=G_c [B, K_d, d_z], KV=Z_d_prev [B, K_d, d_z]
        # 每个 G_c slot 独立从 Z_d_prev 中检索不同的历史信息
        # 使用轻量 attention (无 FFN), 减少参数量
        self.q_norm = RMSNorm(d_z)
        self.kv_norm = RMSNorm(d_z)
        self.q_proj = nn.Linear(d_z, d_z, bias=False)
        self.k_proj = nn.Linear(d_z, d_z, bias=False)
        self.v_proj = nn.Linear(d_z, d_z, bias=False)
        self.o_proj = nn.Linear(d_z, d_z, bias=False)
        # o_proj 小初始化: 初始时 CA 输出 ≈ 0, 不干扰 G_c
        nn.init.normal_(self.o_proj.weight, std=0.01)

        # ====== Slot 身份偏置 ======
        # 从 Z_d_init 投影出 per-slot 偏置, 保持 slot 间的结构差异
        # 即使 G_c 的 slot 方向趋同, slot_bias 也能提供分化信号
        self.slot_bias_proj = nn.Sequential(
            RMSNorm(d_z),
            nn.Linear(d_z, d_z, bias=False),
        )
        # 小初始化: 初始时 slot_bias 不过度干扰
        nn.init.normal_(self.slot_bias_proj[1].weight, std=0.02)

        # ====== 可学习混合系数 ======
        # α: 历史上下文注入量, sigmoid(-2.2) ≈ 0.1
        self._history_alpha_logit = nn.Parameter(torch.tensor(-2.2))
        # β: slot 身份偏置强度, sigmoid(-2.2) ≈ 0.1
        self._slot_bias_beta_logit = nn.Parameter(torch.tensor(-2.2))

        # ====== 输出归一化 ======
        self.output_norm = RMSNorm(d_z)

    def forward(
        self,
        G_c: torch.Tensor,          # [B, K_d, d_z] 当前 step 的 grounded trace (主体)
        Z_d_prev: torch.Tensor,      # [B, K_d, d_z] 上一步的 draft state (历史上下文)
        Z_d_init: torch.Tensor,      # [B, K_d, d_z] prefill 阶段的视觉基线 (slot 身份)
        consistency_score: Optional[torch.Tensor] = None,  # [B, 1] 一致性分数 (可选)
    ) -> torch.Tensor:
        """
        Current-Dominant Draft Refinement
        
        信号流:
        1. history_ctx = CA(Q=G_c, KV=Z_d_prev): per-slot 从历史中读取上下文
        2. slot_bias = Linear(Z_d_init): per-slot 身份偏置
        3. Z_d_new = output_norm(G_c + α·(history_ctx - G_c) + β·slot_bias)
        
        当 consistency_score 低 (推理偏离) 时:
        - 减小 α (更少依赖可能有误的历史)
        - 增大 β (更多依赖稳定的视觉基线)
        
        Args:
            G_c: [B, K_d, d_z] 当前 grounded trace
            Z_d_prev: [B, K_d, d_z] 上一步的 draft state
            Z_d_init: [B, K_d, d_z] 视觉基线
            consistency_score: [B, 1] 一致性分数, None 时使用固定系数
        
        Returns:
            Z_d_new: [B, K_d, d_z] 更新后的 draft state
        """
        param_dtype = self.q_proj.weight.dtype
        G_c = G_c.to(param_dtype)
        Z_d_prev = Z_d_prev.to(param_dtype)
        Z_d_init = Z_d_init.to(param_dtype)

        B, K_d, D = G_c.shape

        # ====== 1. Per-slot 历史上下文读取 (轻量 CA, 无 FFN) ======
        q = self.q_norm(G_c)
        kv = self.kv_norm(Z_d_prev)

        q = self.q_proj(q).view(B, K_d, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K_d, D_h]
        k = self.k_proj(kv).view(B, K_d, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, K_d, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, K_d, D)
        history_delta = self.o_proj(attn_output)  # [B, K_d, d_z], per-slot 历史增量

        # ====== 2. Slot 身份偏置 ======
        slot_bias = self.slot_bias_proj(Z_d_init)  # [B, K_d, d_z]

        # ====== 3. 可学习混合系数 ======
        alpha = torch.sigmoid(self._history_alpha_logit)  # 标量 ∈ (0, 1), 初始 ~0.1
        beta = torch.sigmoid(self._slot_bias_beta_logit)   # 标量 ∈ (0, 1), 初始 ~0.1

        # ====== 4. Consistency-aware 调制 (可选) ======
        if consistency_score is not None:
            # cs 低 (推理偏离) → 减小 α (历史可能有误), 增大 β (依赖稳定基线)
            cs = consistency_score.detach().unsqueeze(-1)  # [B, 1, 1]
            alpha_effective = alpha * cs.clamp(min=0.05)   # cs=0 时 α 降到 5% 的基础值
            beta_effective = beta * (2.0 - cs).clamp(max=1.5)  # cs=0 时 β 放大到 1.5 倍
        else:
            alpha_effective = alpha
            beta_effective = beta

        # ====== 5. 融合: G_c 为主体 + 历史增量 + slot 偏置 ======
        Z_d_new = G_c + alpha_effective * history_delta + beta_effective * slot_bias

        # ====== 6. 输出归一化 ======
        Z_d_new = self.output_norm(Z_d_new)

        return Z_d_new


class LatentDraftFlow(nn.Module):
    """
    隐空间流 (Latent Draft Flow): 将 latent draft 视为连续演化的隐空间信号流
    
    设计理念:
    - 摒弃"一致性检验"范式 (ConsistencyHead、二次检索验证、自适应纠错 MLP 等)
    - 把 Z_d 看成在隐空间中流动的状态, 每一步通过与当前推理状态和视觉信息的交互来演化
    - 类似 Neural ODE / Flow Matching: Z_d 是一个连续演化的流, 不是离散的"检验→修正"
    
    核心信号流 (两步 CA, 简洁直接):
      Stage 1 — 历史融合: 当前推理状态从历史 draft 中读取上下文
        H_c = CA(Q=S_c, KV=Z_d_prev)
      
      Stage 2 — 视觉锚定: 融合后的状态从视觉信息中锚定/修正
        Z_d_new = CA(Q=H_c, KV=visual_hidden)
      
      Stage 3 — 归一化
        Z_d_new = RMSNorm(Z_d_new)
    
    与旧架构的关键区别:
    - 旧架构: S_c → BidirectionalReflection(二次检索+ConsistencyHead+纠错MLP) → G_c
              → CurrentDominantDraftRefiner(G_c+α·CA(G_c,Z_d_prev)+β·bias) → Z_d_new
              信号流长, 有 3 个 ConsistencyHead, 2 个纠错 MLP, 多个门控标量
    - 新架构: S_c → CA(S_c, Z_d_prev) → CA(H_c, VH) → LayerScale 残差 → Z_d_new
              信号流短, 无一致性分数, 无纠错 MLP, 无门控标量
    
    为什么这个设计更好:
    1. 去掉了所有"伪一致性"信号: 不再用 CA 增量冒充一致性
    2. 信号流更短更直接: S_c → 两步 CA → Z_d_new
    3. Z_d 真正成为"流": 每一步都是前一步 Z_d 与当前推理+视觉交互的结果
    4. slot 分化天然保证: 两步 CA 中每个 slot 独立做 attention
    5. 参数量更少: ~2.4M (两个轻量 CA) vs ~8.6M (BidirectionalReflection + DraftRefiner)
    
    LayerScale 更新策略 (替代固定 Δt 残差步进):
    - 借鉴 CaiT/DeiT-III 的 LayerScale 技术
    - per-slot 可学习缩放因子 γ_i, 零初始化 → 恒等路径
    - Z_d_new = Z_d_prev + γ_i · (flow_target - Z_d_prev)
    - 优势: 无超参数, per-slot 差异化, 训练自动调节更新幅度
    - 秩坍塌防护: γ_i=0 时 Z_d_new=Z_d_prev, 保持正交初始化的 slot 多样性
    
    Args:
        d_z: controller 空间维度 (768)
        num_heads: cross-attention 头数 (8)
        num_draft_slots: draft slot 数量 K_d (默认 16, 用于 LayerScale 参数维度)
        max_steps: [已废弃, 保留用于兼容旧 checkpoint] 旧版残差步进的 Δt = 1/C
    """

    def __init__(self, d_z: int = 768, num_heads: int = 8, num_draft_slots: int = 16, max_steps: int = 7):
        super().__init__()
        self.d_z = d_z
        self.num_heads = num_heads
        self.head_dim = d_z // num_heads
        self.max_steps = max_steps  # [已废弃] 保留用于兼容旧 checkpoint
        self.num_draft_slots = num_draft_slots

        # ====== Stage 1: 历史融合 CA (轻量, 无 FFN) ======
        # Q=S_c [B, K_d, d_z], KV=Z_d_prev [B, K_d, d_z]
        # 每个 S_c slot 独立从 Z_d_prev 中读取不同的历史信息 → slot 分化
        self.history_q_norm = RMSNorm(d_z)
        self.history_kv_norm = RMSNorm(d_z)
        self.history_q_proj = nn.Linear(d_z, d_z, bias=False)
        self.history_k_proj = nn.Linear(d_z, d_z, bias=False)
        self.history_v_proj = nn.Linear(d_z, d_z, bias=False)
        self.history_o_proj = nn.Linear(d_z, d_z, bias=False)

        # ====== Stage 2: 视觉锚定 CA (完整 CrossAttentionBlock, 含 FFN) ======
        # Q=H_c [B, K_d, d_z], KV=visual_hidden [B, ~500, d_z]
        # 从 ~500 个视觉 token 中检索与当前推理相关的证据, 锚定/修正 draft
        # 使用完整 CA (含 FFN): 视觉信息量大 (~500 tokens), 需要更强的表达能力
        self.visual_ca = CrossAttentionBlock(d_model=d_z, num_heads=num_heads)

        # ====== 层间归一化: 防止两步 CA 残差叠加导致范数爆炸 ======
        self.inter_norm = RMSNorm(d_z)

        # ====== 输出归一化 ======
        self.output_norm = RMSNorm(d_z)

        # ====== LayerScale: per-slot 可学习缩放因子 (替代固定 Δt 残差步进) ======
        # 初始值 0.1: 训练初期 Z_d_new = Z_d_prev + 0.1·velocity (温和更新)
        # 每个 slot 独立学习自己的更新幅度, 避免全局 dt 的一刀切
        # 灵感来源: CaiT (Touvron et al., ICCV 2021), DeiT-III (Touvron et al., ECCV 2022)
        #
        # 为什么不用零初始化:
        #   零初始化时 Z_d_new = Z_d_prev (恒等路径), draft 完全不更新
        #   训练 400 步后 γ 均值仅 -0.0034, 说明零初始化太保守, draft 更新信号太弱
        #   CaiT 原文推荐浅层网络用 1e-4, 但我们的 LatentDraftFlow 只有 2 个 CA stage
        #   且 velocity = flow_target - Z_d_prev 已经是有意义的方向, 不需要从零开始探索
        #   0.1 让 draft 从训练一开始就有 10% 的有效更新, 加速收敛
        self.slot_scale = nn.Parameter(torch.ones(1, num_draft_slots, 1) * 0.1)  # [1, K_d, 1]

        # ====== 初始化策略 ======
        # history CA 的 o_proj 小初始化: 初始时历史融合输出 ≈ 0, S_c 主导
        nn.init.normal_(self.history_o_proj.weight, std=0.01)

    def _history_cross_attention(
        self,
        query: torch.Tensor,     # [B, K_d, d_z]
        key_value: torch.Tensor,  # [B, K_d, d_z]
    ) -> torch.Tensor:
        """
        轻量 Cross-Attention (无 FFN, 无残差连接)
        
        与完整 CrossAttentionBlock 的区别:
        - 无 FFN: 历史 draft 只有 16 个 slot, 不需要 FFN 的额外表达能力
        - 无残差连接: 输出是纯 attention 结果, 由调用方决定如何融合
        
        Returns:
            attn_output: [B, K_d, d_z] 纯 attention 输出 (无残差)
        """
        B, L_q, D = query.shape
        _, L_kv, _ = key_value.shape

        q = self.history_q_norm(query)
        kv = self.history_kv_norm(key_value)

        q = self.history_q_proj(q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.history_k_proj(kv).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.history_v_proj(kv).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        attn_output = self.history_o_proj(attn_output)

        return attn_output

    def forward(
        self,
        S_c: torch.Tensor,              # [B, K_d, d_z] 当前推理状态 (step 摘要或轨迹)
        Z_d_prev: torch.Tensor,          # [B, K_d, d_z] 上一步的 draft state (历史流)
        visual_hidden: torch.Tensor,     # [B, L_v, d_z] 投影后的完整视觉 hidden (~500 tokens)
    ) -> torch.Tensor:
        """
        隐空间流演化 (残差步进): S_c + Z_d_prev + visual_hidden → Z_d_new
        
        核心公式:
          flow_target = CA_full(Q=RMSNorm(S_c + CA_light(S_c, Z_d_prev)), KV=visual_hidden)
          velocity = flow_target - Z_d_prev
          Z_d_new = Z_d_prev + γ_i · velocity    (γ_i: per-slot 可学习, 零初始化)
        
        信号流:
        1. history_delta = CA(Q=S_c, KV=Z_d_prev): 从历史 draft 中读取上下文
        2. H_c = S_c + history_delta: 融合当前推理与历史上下文
        3. H_c_normed = RMSNorm(H_c): 层间归一化, 防止范数爆炸
        4. flow_target = CA(Q=H_c_normed, KV=visual_hidden): 从视觉中锚定/修正
        5. velocity = flow_target - Z_d_prev: 计算速度场
        6. Z_d_new = Z_d_prev + γ_i · velocity: LayerScale 残差 (恒等路径保持 slot 多样性)
        7. 软范数裁剪: 防止多步累积后范数爆炸
        
        关键设计:
        - LayerScale: per-slot 可学习缩放因子 γ_i, 初始化为 0.1 → 温和更新路径
        - 恒等路径: 当 γ_i→0 时 Z_d_new→Z_d_prev, slot 多样性通过 Z_d_prev 的正交基底保留
        - 雅可比矩阵: ∂Z_new/∂Z_prev = I + γ_i·J, 主对角线为 1, 无梯度消失
        - 自适应步长: 每个 slot 独立学习更新幅度, 无需手动调 dt 超参数
        
        Args:
            S_c: [B, K_d, d_z] 当前推理状态
            Z_d_prev: [B, K_d, d_z] 上一步的 draft state
            visual_hidden: [B, L_v, d_z] 投影后的完整视觉 hidden
        
        Returns:
            Z_d_new: [B, K_d, d_z] 演化后的 draft state
        """
        param_dtype = self.history_q_proj.weight.dtype
        S_c = S_c.to(param_dtype)
        Z_d_prev = Z_d_prev.to(param_dtype)
        visual_hidden = visual_hidden.to(param_dtype)

        # ====== Stage 1: 历史融合 ======
        # S_c 从历史 draft 中读取上下文 (per-slot 独立 attention → slot 分化)
        history_delta = self._history_cross_attention(S_c, Z_d_prev)  # [B, K_d, d_z]
        H_c = S_c + history_delta  # 残差融合: 当前推理 + 历史上下文

        # ====== 层间归一化 ======
        H_c = self.inter_norm(H_c)

        # ====== Stage 2: 视觉锚定 ======
        # 从 ~500 个视觉 token 中检索与当前推理相关的证据
        # 完整 CrossAttentionBlock (含 FFN + 残差): 视觉信息量大, 需要更强的表达能力
        flow_target = self.visual_ca(query=H_c, key_value=visual_hidden)  # [B, K_d, d_z]

        # ====== Stage 3: LayerScale 残差 (per-slot 可学习步长, 防止 slot 坍塌) ======
        # velocity = flow_target - Z_d_prev: 从当前位置指向目标位置的速度场
        # Z_d_new = Z_d_prev + γ_i · velocity: γ_i 初始化为 0.1, 训练自动调节
        # 初始时 Z_d_new = Z_d_prev + 0.1·velocity → 温和更新, 保持 slot 多样性
        # 与固定 dt=1/C 的区别: 每个 slot 独立学习更新幅度, 无超参数
        velocity = flow_target - Z_d_prev
        Z_d_new = Z_d_prev + self.slot_scale * velocity

        # ====== Stage 4: 软范数裁剪 (防止多步累积后范数爆炸, 保留 slot 间范数差异) ======
        max_norm = math.sqrt(self.d_z / 512) * 35  # 根据 d_z 动态计算, d_z=768 时 max_norm≈42.9
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


class PrefixKVProjector(nn.Module):
    """
    Prefix KV 投影器: 将 Z_d [B, K_d, d_z] 投影为注入层的 prefix K/V
    
    核心设计:
    - 每个注入层有独立的 K/V 投影头 (不共享参数)
    - 输出格式与 Qwen3-VL 的 GQA 对齐: K/V shape = [B, num_kv_heads, N_p, head_dim]
    - 不施加 RoPE (位置无关的全局控制信号, 等效于 position=0)
    - K 投影后施加 RMSNorm (与基座 Qwen3VLTextAttention 的 k_norm 对齐)
    - prefix KV 作为虚拟 token, 与 real token 在 attention 中平等竞争
      (不使用 scale 门控, 训练稳定性通过初始化策略保证: K 小范围初始化, V 零初始化)
    
    flash_attention_2 兼容性:
    - prefix KV 通过 DynamicCache.update() 拼接到真实 KV 前面
    - flash_attn 原生支持 KV_len > Q_len (这就是 KV cache 的标准用法)
    - causal mask: Q[i] 可以看到 K[j] where j <= i + N_p (prefix 对所有 Q 可见)
    - 不需要自定义 attention_mask, 不需要修改 position_ids
    
    参数量估算 (d_z=768, num_kv_heads=8, head_dim=128, 3 注入层):
    - 每层: K_proj(768→1024) + V_proj(768→1024) + K_norm(128) = ~1.57M
    - 3 层总计: ~4.7M
    
    Args:
        d_z: controller 空间维度 (768)
        num_kv_heads: GQA 的 KV 头数 (Qwen3-VL-8B: 8)
        head_dim: 每个头的维度 (Qwen3-VL-8B: 128)
        injection_layers: 注入层索引列表 (如 [18, 26, 35])
    """

    def __init__(
        self,
        d_z: int = 768,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        injection_layers: list = None,
    ):
        super().__init__()
        if injection_layers is None:
            injection_layers = [18, 26, 35]
        self.d_z = d_z
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_dim = num_kv_heads * head_dim  # Qwen3-VL-8B: 8 * 128 = 1024
        self.injection_layers = sorted(injection_layers)

        # 每个注入层独立的 K/V 投影
        self.k_projs = nn.ModuleDict({
            str(l): nn.Linear(d_z, self.kv_dim, bias=False)
            for l in self.injection_layers
        })
        self.v_projs = nn.ModuleDict({
            str(l): nn.Linear(d_z, self.kv_dim, bias=False)
            for l in self.injection_layers
        })

        # K 投影后施加 RMSNorm (与基座 Qwen3VLTextAttention 的 k_norm 对齐)
        # 基座的 k_norm 是 per-head 的 RMSNorm(head_dim)
        self.k_norms = nn.ModuleDict({
            str(l): RMSNorm(head_dim)
            for l in self.injection_layers
        })

        # 虚拟 token 初始化: K 小范围初始化 + V 零初始化
        # 这样初始时 prefix V = 0, 即使被 attend 到也不影响输出
        # 随训练逐步学到有意义的 V 值 (类似 LoRA 的 B 矩阵零初始化)
        self._init_weights()

    def _init_weights(self):
        """虚拟 token 初始化策略:
        - K 投影: 小范围正态初始化 (std=0.02), 让 prefix 在 attention 中有合理的竞争力
        - V 投影: 零初始化, 初始时 prefix 对输出无影响, 随训练逐步学到有意义的值
        这样不需要 scale 门控, prefix 天然从"无影响"状态开始, 逐步增强
        """
        for proj in self.k_projs.values():
            nn.init.normal_(proj.weight, std=0.02)
        for proj in self.v_projs.values():
            nn.init.zeros_(proj.weight)

    def forward(
        self,
        Z_d: torch.Tensor,     # [B, K_d, d_z]
        layer_idx: int,
    ) -> tuple:
        """
        将 Z_d 投影为指定层的 prefix K/V
        
        Args:
            Z_d: [B, K_d, d_z] 当前 draft state
            layer_idx: 注入层索引
        
        Returns:
            prefix_k: [B, num_kv_heads, K_d, head_dim]
            prefix_v: [B, num_kv_heads, K_d, head_dim]
        """
        B, K_d, _ = Z_d.shape
        l_str = str(layer_idx)
        input_dtype = Z_d.dtype  # 记录输入 dtype (通常是 bfloat16)

        # dtype 对齐: 将输入 cast 到参数 dtype 进行计算, 输出再 cast 回输入 dtype
        # Accelerate bf16 混合精度会将可训练参数 upcast 到 float32 (master weights),
        # 不应修改参数 dtype (会破坏 Accelerate 的精度管理), 而是在输入/输出处 cast
        proj_dtype = self.k_projs[l_str].weight.dtype
        Z_d_compute = Z_d.to(proj_dtype) if Z_d.dtype != proj_dtype else Z_d

        # 投影到 KV 空间
        k = self.k_projs[l_str](Z_d_compute)  # [B, K_d, kv_dim]
        v = self.v_projs[l_str](Z_d_compute)  # [B, K_d, kv_dim]

        # Reshape 为 [B, K_d, num_kv_heads, head_dim]
        k = k.view(B, K_d, self.num_kv_heads, self.head_dim)
        v = v.view(B, K_d, self.num_kv_heads, self.head_dim)

        # K norm (与基座 Qwen3VLTextAttention 的 k_norm 对齐)
        k = self.k_norms[l_str](k)

        # 转置为 [B, num_kv_heads, K_d, head_dim] (与 DynamicCache 格式对齐)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # 输出 cast 回输入 dtype: 确保 prefix KV 与基座模型 KV cache dtype 一致
        # (计算在 float32 精度下完成, 输出转回 bfloat16 与基座模型兼容)
        if k.dtype != input_dtype:
            k = k.to(input_dtype)
            v = v.to(input_dtype)

        return k, v

    def get_all_prefix_kvs(
        self,
        Z_d: torch.Tensor,     # [B, K_d, d_z]
    ) -> dict:
        """
        一次性计算所有注入层的 prefix K/V
        
        Returns:
            prefix_kvs: {layer_idx: (prefix_k, prefix_v)} 字典
                prefix_k: [B, num_kv_heads, K_d, head_dim]
                prefix_v: [B, num_kv_heads, K_d, head_dim]
        """
        result = {}
        for layer_idx in self.injection_layers:
            result[layer_idx] = self.forward(Z_d, layer_idx)
        return result
