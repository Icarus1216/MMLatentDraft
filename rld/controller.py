"""RLD Controller: 编排整个 Reflective Latent Draft 的生命周期

核心职责:
1. Prefill 阶段: 从视觉 token 初始化冻结证据 Z^e，初始化 Z_d_init (视觉基线) 和轨迹 T_0
2. Step 边界更新: 执行 S_c → T_c → G_c → prefix_source 的完整更新链路
   - G_c 直接作为 prefix KV 的来源信号 (取消 DraftUpdater 残差步进)
   - prefix_source = gate * G_c + (1 - gate) * Z_d_init (门控融合)
   - Z_d_init 提供稳定的视觉基线, G_c 提供即时纠错信号
3. prefix_source 通过 PrefixKVProjector 注入基座模型

简化模式 (use_trace_updater=False):
  - 去掉 TraceUpdater 的累计轨迹压缩, 直接用 S_c 作为 ReflectionModule 的 query
  - 优势: 更短的梯度路径, 更精准的检索 query, 更少的参数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from .modules import (
    CrossAttentionBlock,
    EvidenceResampler,
    StepResampler,
    StreamingTraceAccumulator,
    CommitGate,
    TraceUpdater,
    TraceEMA,
    ReflectionModule,
    BidirectionalReflection,
    DraftUpdater,
    ResidualFlowDraftUpdater,
    CurrentDominantDraftRefiner,
    LatentDraftFlow,
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
        lambda_div: 多样性正则化权重 (默认 0.01)
        max_steps: 最大 step 数, 用于 LatentDraftFlow 残差步进的 Δt = 1/max_steps 计算 (默认 7, 与实际 step 数上界匹配)
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
        lambda_div: float = 0.01,
        max_steps: int = 7,
        use_trace_updater: bool = True,
        use_bidirectional_reflection: bool = False,
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
        self.use_trace_updater = use_trace_updater
        self.use_bidirectional_reflection = use_bidirectional_reflection

        # ====== 核心模块 ======

        # 1. 证据重采样器: V → Z^e (旧模式保留用于兼容)
        #    双向检索模式下仍保留参数用于加载旧 checkpoint, 但 forward 中不使用
        self.evidence_resampler = EvidenceResampler(
            hidden_size=hidden_size,
            d_z=d_z,
            num_evidence_slots=num_evidence_slots,
            num_heads=num_heads,
            num_layers=evidence_layers,
        )

        # 1b. 视觉投影层: visual_tokens (4096) → visual_hidden_proj (d_z)
        #     双向检索模式下使用, 替代 EvidenceResampler 的压缩
        #     保留完整的 ~500 个视觉 token, 只做维度投影
        self.proj_v = nn.Linear(hidden_size, d_z, bias=False)

        # 2. 步摘要重采样器: H^step → S_c
        self.step_resampler = StepResampler(
            hidden_size=hidden_size,
            d_z=d_z,
            num_trace_slots=num_trace_slots,
            num_heads=num_heads,
        )

        # 3. 轨迹更新器
        #    当 use_trace_updater=False 时, 跳过此模块, 直接用 S_c 作为 flow query
        #    3a. [已弃用] TraceUpdater (2 层 CA): 16→16 方阵映射导致 T_rank=1.0 坍塌
        #        保留参数用于加载旧 checkpoint, forward 中不再使用
        self.trace_updater = TraceUpdater(
            d_z=d_z,
            num_trace_slots=num_trace_slots,
            num_heads=num_heads,
            num_layers=2,
        )
        #    3b. TraceEMA: per-slot 可学习 α 做 EMA 融合
        #        T_c = α·T_prev + (1-α)·S_c, 逐 slot 独立更新, 天然保持 slot 多样性
        #        参数量: 仅 16 个 (vs TraceUpdater 的 ~2.4M)
        self.trace_ema = TraceEMA(num_trace_slots=num_trace_slots)

        # 4. 回看-验证模块
        #    旧模式: ReflectionModule (单向, KV=Z_e 16 slots)
        #    新模式: BidirectionalReflection (双向, KV=visual_hidden_proj ~500 tokens)
        self.reflection = ReflectionModule(
            d_z=d_z,
            num_heads=num_heads,
            num_layers=2,
        )
        # 双向检索-验证模块 (新模式)
        self.bidirectional_reflection = BidirectionalReflection(
            d_z=d_z,
            num_heads=num_heads,
        )

        # 5. 草稿更新器 (保留用于加载旧 checkpoint, forward 中不再使用)
        self.draft_updater = ResidualFlowDraftUpdater(
            d_z=d_z,
            num_heads=num_heads,
            max_steps=max_steps,
        )

        # 5b. [兼容旧 checkpoint] 保留方案 C 的 draft_refiner, forward 中不再使用
        self.draft_refiner = CurrentDominantDraftRefiner(
            d_z=d_z,
            num_heads=num_heads,
        )

        # 5c. Latent Draft Flow (方案 D): 隐空间流 + LayerScale 残差
        #     摒弃一致性检验范式, 把 Z_d 看成连续演化的隐空间信号流
        #     flow_target = CA(RMSNorm(S_c + CA(S_c, Z_d_prev)), visual_hidden)
        #     Z_d_new = Z_d_prev + γ_i · (flow_target - Z_d_prev)  [γ_i: per-slot 可学习, 零初始化]
        self.latent_draft_flow = LatentDraftFlow(
            d_z=d_z,
            num_heads=num_heads,
            num_draft_slots=num_draft_slots,
            max_steps=max_steps,
        )

        # [兼容旧 checkpoint] 保留方案 B/C 的门控参数, forward 中不再使用
        self._gc_inject_gate_logit = nn.Parameter(torch.tensor(0.0))
        self._cs_gate_boost_logit = nn.Parameter(torch.tensor(0.0))

        # ====== 初始化相关 ======

        # 可学习的草稿初始状态 (正交初始化, 保证 slot 间余弦相似度 ≈ 0)
        _draft_init_data = torch.empty(num_draft_slots, d_z)
        nn.init.orthogonal_(_draft_init_data)
        # 方案 B: Z_d_init 作为固定视觉基线, 与 G_c 门控融合
        self.draft_init = nn.Parameter(_draft_init_data.unsqueeze(0))  # [1, K_d, d_z], 每行范数≈1.0

        # Per-slot 草稿条件化: 每个 draft slot 通过 cross-attention 从 Z_e 获取独立信息
        # (替代旧版的 Z_e.mean 广播, 消除 slot 坍塌的根因)
        self.draft_init_ca = CrossAttentionBlock(
            d_model=d_z,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )

        # ====== 流式轨迹累积 + 学习型 Commit 门控 ======

        # 6. 流式轨迹累积器: 每 token 增量更新 S_running (替代推理时的 step_hidden_buffers)
        self.streaming_accumulator = StreamingTraceAccumulator(
            hidden_size=hidden_size,
            d_z=d_z,
            num_trace_slots=num_trace_slots,
        )

        # 7. 学习型 Commit 门控: 决定何时触发完整的 draft 更新链路
        #    只在句末计算 commit_score, 保持句内连贯性
        self.commit_gate = CommitGate(
            d_z=d_z,
            commit_threshold=0.5,
            min_tokens_between_commits=8,
            max_tokens_between_commits=128,
        )

        # 多样性正则化 (threshold=0.3: 允许 slot 间有适度相似性, 避免过度正则化)
        self.diversity_reg = DiversityRegularizer(threshold=0.3)

    def compute_prefix_source(
        self,
        S_c: torch.Tensor,           # [B, K_d, d_z] 当前推理状态 (step 摘要或轨迹)
        Z_d_prev: torch.Tensor,      # [B, K_d, d_z] 上一步的 draft state (历史流)
        visual_hidden: torch.Tensor,  # [B, L_v, d_z] 投影后的完整视觉 hidden
    ) -> torch.Tensor:
        """
        方案 D: Latent Draft Flow (隐空间流)
        
        摒弃一致性检验范式, 把 Z_d 看成连续演化的隐空间信号流。
        
        核心信号流:
          Stage 1 — 历史融合: H_c = CA(Q=S_c, KV=Z_d_prev)
          Stage 2 — 视觉锚定: Z_d_new = CA(Q=H_c, KV=visual_hidden)
          Stage 3 — 归一化: Z_d_new = RMSNorm(Z_d_new)
        
        Args:
            S_c: [B, K_d, d_z] 当前推理状态
            Z_d_prev: [B, K_d, d_z] 上一步的 draft state
            visual_hidden: [B, L_v, d_z] 投影后的完整视觉 hidden
        
        Returns:
            prefix_source: [B, K_d, d_z] 用于 PrefixKVProjector 的输入
        """
        return self.latent_draft_flow(S_c, Z_d_prev, visual_hidden)

    def prefill(
        self,
        visual_tokens: torch.Tensor,   # [B, L_v, hidden_size]
    ) -> Dict[str, torch.Tensor]:
        """
        Prefill 阶段: 初始化所有外挂状态
        
        双向检索模式下:
        - 不再使用 EvidenceResampler 压缩为 16 slots
        - 直接用 proj_v 将 visual_tokens 投影到 d_z 维, 缓存完整的 visual_hidden_proj
        - Z_d_init 从 visual_hidden_proj 中获取信息, 作为固定视觉基线
        
        方案 B: Z_d_init 在整个推理过程中保持不变, 与每步的 G_c 门控融合
        
        Args:
            visual_tokens: 来自视觉编码器的 token 序列
        
        Returns:
            state: {
                'Z_e': [B, K_e, d_z],              # 冻结证据槽 (旧模式) 或占位 (新模式)
                'visual_hidden_proj': [B, L_v, d_z], # 投影后的完整视觉 hidden (新模式)
                'Z_d_init': [B, K_d, d_z],          # 视觉基线 (方案 B, 不随 step 变化)
                'Z_d': [B, K_d, d_z],               # 当前 prefix_source (初始 = Z_d_init)
                'G_c': [B, K_d, d_z],               # 最新的 G_c (初始 = Z_d_init)
                'T': [B, K_t, d_z],                 # 初始轨迹 (零)
                'step_count': 0,
            }
        """
        B = visual_tokens.shape[0]
        device = visual_tokens.device
        dtype = visual_tokens.dtype

        if self.use_bidirectional_reflection:
            # ====== 双向检索模式: 保留完整视觉信息 ======
            # 投影到 d_z 维, 缓存完整的 ~500 个 token (而非压缩到 16 slots)
            visual_tokens_cast = visual_tokens.to(self.proj_v.weight.dtype)
            visual_hidden_proj = self.proj_v(visual_tokens_cast)  # [B, L_v, d_z]

            # 双向模式下不再调用 EvidenceResampler, Z_e 用零张量占位 (仅保持接口兼容)
            Z_e = torch.zeros(B, self.K_e, self.d_z, device=device, dtype=dtype)

            # Z_d_init: 从完整 visual_hidden_proj 中获取信息, 作为固定视觉基线
            draft_init = self.draft_init.expand(B, -1, -1).to(device=device, dtype=dtype)
            Z_d_init = self.draft_init_ca(query=draft_init, key_value=visual_hidden_proj)  # [B, K_d, d_z]
        else:
            # ====== 旧模式: EvidenceResampler 压缩为 16 slots ======
            visual_hidden_proj = None
            Z_e = self.evidence_resampler(visual_tokens)  # [B, K_e, d_z]

            draft_init = self.draft_init.expand(B, -1, -1).to(device=device, dtype=dtype)
            Z_d_init = self.draft_init_ca(query=draft_init, key_value=Z_e)  # [B, K_d, d_z]

        # 3. 初始化轨迹为零
        T = torch.zeros(B, self.K_t, self.d_z, device=device, dtype=dtype)

        # 4. 初始化流式累积状态
        S_running = torch.zeros(B, self.K_t, self.d_z, device=device, dtype=dtype)
        S_prev = torch.zeros(B, self.K_t, self.d_z, device=device, dtype=dtype)

        return {
            'Z_e': Z_e,
            'visual_hidden_proj': visual_hidden_proj,  # [B, L_v, d_z] 或 None
            'Z_d_init': Z_d_init,                      # [B, K_d, d_z] 固定视觉基线 (方案 B)
            'Z_d': Z_d_init,                            # [B, K_d, d_z] 当前 prefix_source (初始 = Z_d_init)
            'G_c': Z_d_init,                            # [B, K_d, d_z] 最新 G_c (初始 = Z_d_init)
            'T': T,
            'step_count': 0,
            'S_running': S_running,     # 流式累积的 running summary
            'S_prev': S_prev,           # 上次 commit 时的 S_running 快照
            'tokens_since_commit': 0,   # 自上次 commit 以来的 token 数
        }

    def streaming_accumulate(
        self,
        state: Dict[str, torch.Tensor],
        h_t: torch.Tensor,   # [B, 1, hidden_size] 当前 token 的 hidden state
    ) -> Dict[str, torch.Tensor]:
        """
        每 token 增量更新 S_running (极轻量, ~0.1ms)
        
        不触发完整的 draft 更新链路, 只更新 running summary。
        
        Args:
            state: 当前 RLD 状态
            h_t: [B, 1, hidden_size] 当前 token 的 hidden state
        
        Returns:
            state: 更新后的状态 (只修改 S_running 和 tokens_since_commit)
        """
        S_running = state['S_running']
        S_running_new = self.streaming_accumulator(h_t, S_running)
        
        state = dict(state)  # 浅拷贝, 避免修改原 state
        state['S_running'] = S_running_new
        state['tokens_since_commit'] = state.get('tokens_since_commit', 0) + 1
        return state

    def compute_commit_score(
        self,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算 commit score (只在句末调用)
        
        方案 B: 用当前 prefix_source (state['Z_d']) 作为 CommitGate 的 draft 输入
        
        Args:
            state: 当前 RLD 状态
        
        Returns:
            commit_score: [B, 1] ∈ (0, 1)
        """
        return self.commit_gate(
            S_running=state['S_running'],
            S_prev=state['S_prev'],
            Z_d=state['Z_d'],  # 方案 B: 这里是 prefix_source, 语义上仍是 "当前 draft 状态"
        )

    def commit_update(
        self,
        state: Dict[str, torch.Tensor],
        update_mask: Optional[torch.Tensor] = None,  # [B] bool
    ) -> Dict[str, torch.Tensor]:
        """
        Commit 更新: 用 S_running 作为 S_c, 执行完整的更新链路
        
        方案 D: Latent Draft Flow (隐空间流)
        信息流: S_running → TraceUpdater → LatentDraftFlow(S_c, Z_d_prev, visual_hidden)
        
        与 step_update 的区别:
        - step_update 接收 raw hidden states, 需要先过 StepResampler
        - commit_update 直接使用已累积的 S_running, 跳过 StepResampler
        
        Args:
            state: 当前 RLD 状态 (必须包含 S_running)
            update_mask: [B] bool, True 表示该样本需要 commit
        
        Returns:
            state: 更新后的状态
        """
        Z_e = state['Z_e']
        Z_d_init = state['Z_d_init']
        prefix_source_prev = state['Z_d']  # 当前 prefix_source (即 Z_d_prev)
        T_prev = state['T']
        S_running = state['S_running']
        visual_hidden_proj = state.get('visual_hidden_proj')

        if self.use_trace_updater:
            T_c = self.trace_ema(T_prev, S_running)
            flow_query = T_c
        else:
            T_c = S_running
            flow_query = S_running

        # 方案 D: Latent Draft Flow
        # flow_query 作为 S_c, prefix_source_prev 作为 Z_d_prev
        if visual_hidden_proj is not None:
            prefix_source_new = self.compute_prefix_source(
                S_c=flow_query,
                Z_d_prev=prefix_source_prev,
                visual_hidden=visual_hidden_proj,
            )
        else:
            # 旧模式回退: 用 Z_e 作为视觉信息源
            prefix_source_new = self.compute_prefix_source(
                S_c=flow_query,
                Z_d_prev=prefix_source_prev,
                visual_hidden=Z_e,  # [B, K_e, d_z] 作为 visual_hidden 的替代
            )

        # 根据 update_mask 选择性更新
        if update_mask is not None:
            mask = update_mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            prefix_source_new = mask * prefix_source_new + (1.0 - mask) * prefix_source_prev
            T_c = mask * T_c + (1.0 - mask) * T_prev

        new_state = {
            'Z_e': Z_e,
            'visual_hidden_proj': visual_hidden_proj,  # 透传
            'Z_d_init': Z_d_init,                      # 透传
            'Z_d': prefix_source_new,                  # 方案 D: latent draft flow 输出
            'T': T_c,
            'step_count': state['step_count'] + 1,
            # Commit 后: S_prev 更新为当前 S_running, S_running 做 soft reset
            'S_running': S_running * 0.1,  # soft reset, 保留少量历史
            'S_prev': S_running.clone().detach(),
            'tokens_since_commit': 0,
        }

        # 监控指标
        with torch.no_grad():
            ps_delta = (prefix_source_new.detach() - prefix_source_prev.detach()).norm(dim=-1).mean()
            ps_base = prefix_source_prev.detach().norm(dim=-1).mean().clamp(min=1e-8)
            new_state['_monitor'] = {
                'Zd_abs_delta': ps_delta.item(),
                'Zd_relative_delta': (ps_delta / ps_base).item(),
                'Zd_slot_cosim': self._slot_cosine_similarity(prefix_source_new.detach()),
                'Zd_effective_rank': self._effective_rank(prefix_source_new.detach()),
                'T_effective_rank': self._effective_rank(T_c.detach()),
                'Zd_norm': prefix_source_new.detach().norm(dim=-1).mean().item(),
            }

        return new_state


    def step_update(
        self,
        state: Dict[str, torch.Tensor],
        step_hidden_states: torch.Tensor,   # [B, L_c, hidden_size]
        update_mask: Optional[torch.Tensor] = None,  # [B] bool, per-sample update mask
    ) -> Dict[str, torch.Tensor]:
        """
        Step 边界更新: 执行完整的 "摘要 → 轨迹 → 隐空间流" 更新链路
        
        方案 D: Latent Draft Flow (隐空间流)
        信息流: S_c → T_c → LatentDraftFlow(T_c, Z_d_prev, visual_hidden) → Z_d_new
        
        Args:
            state: 当前状态字典
            step_hidden_states: 当前 step 的最后层 hidden states
            update_mask: [B] bool tensor, True 表示该样本在此位置遇到了 delimiter
        
        Returns:
            state: 更新后的状态字典
        """
        Z_e = state['Z_e']
        Z_d_init = state['Z_d_init']
        prefix_source_prev = state['Z_d']  # 当前 prefix_source (即 Z_d_prev)
        T_prev = state['T']
        visual_hidden_proj = state.get('visual_hidden_proj')

        # 1. 步摘要: H^step → S_c
        S_c = self.step_resampler(step_hidden_states)  # [B, K_t, d_z]

        # 2. 累计轨迹更新 (TraceEMA: per-slot 可学习 α)
        if self.use_trace_updater:
            T_c = self.trace_ema(T_prev, S_c)
            flow_query = T_c
        else:
            T_c = S_c
            flow_query = S_c

        # 3. 方案 D: Latent Draft Flow
        if visual_hidden_proj is not None:
            prefix_source_new = self.compute_prefix_source(
                S_c=flow_query,
                Z_d_prev=prefix_source_prev,
                visual_hidden=visual_hidden_proj,
            )
        else:
            prefix_source_new = self.compute_prefix_source(
                S_c=flow_query,
                Z_d_prev=prefix_source_prev,
                visual_hidden=Z_e,
            )

        # 根据 update_mask 选择性更新
        if update_mask is not None:
            mask = update_mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            prefix_source_new = mask * prefix_source_new + (1.0 - mask) * prefix_source_prev
            T_c = mask * T_c + (1.0 - mask) * T_prev

        new_state = {
            'Z_e': Z_e,
            'visual_hidden_proj': visual_hidden_proj,  # 透传
            'Z_d_init': Z_d_init,                      # 透传
            'Z_d': prefix_source_new,                  # 方案 D: latent draft flow 输出
            'T': T_c,
            'step_count': state['step_count'] + 1,
        }

        # ====== 监控指标 (detach，不影响梯度) ======
        with torch.no_grad():
            ps_delta = (prefix_source_new.detach() - prefix_source_prev.detach()).norm(dim=-1).mean()
            ps_base = prefix_source_prev.detach().norm(dim=-1).mean().clamp(min=1e-8)
            new_state['_monitor'] = {
                'Zd_abs_delta': ps_delta.item(),
                'Zd_relative_delta': (ps_delta / ps_base).item(),
                'Zd_slot_cosim': self._slot_cosine_similarity(prefix_source_new.detach()),
                'Zd_effective_rank': self._effective_rank(prefix_source_new.detach()),
                'T_effective_rank': self._effective_rank(T_c.detach()),
                'Zd_norm': prefix_source_new.detach().norm(dim=-1).mean().item(),
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
        
        方案 D: Latent Draft Flow (隐空间流)
        信息流: S_c → T_c → LatentDraftFlow(T_c, Z_d_prev, visual_hidden) → Z_d_new
        
        这是重构的核心方法: 将 step_update 从散落在段循环里改为一个独立的 for 循环。
        所有 rank 执行相同次数 (C_max)，用 update_mask pad 保证跨 rank 一致。
        
        Args:
            state: 初始 RLD 状态 (来自 prefill)
            step_summaries: list of [B, K_t, d_z], 每个 step 的摘要 S_c
            update_masks: list of [B] bool, 每个 step 的 per-sample 更新掩码
        
        Returns:
            final_state: 最终的 RLD 状态 (额外包含 '_all_T_c' 用于对比学习)
            all_Z_d: list of [B, K_d, d_z], 每个 step 结束后的 draft state
                      长度 = C + 1 (包含初始 Z_d_0)
        """
        all_Z_d = [state['Z_d']]  # 初始 prefix_source (= Z_d_init)
        all_T_c = []  # 保存每步的 T_c (用于对比学习)
        all_commit_scores = []  # 保存每步的 commit_score (用于 commit loss)

        visual_hidden_proj = state.get('visual_hidden_proj')

        for c in range(len(step_summaries)):
            S_c = step_summaries[c]
            mask = update_masks[c]

            Z_e = state['Z_e']
            Z_d_init = state['Z_d_init']
            prefix_source_prev = state['Z_d']  # 当前 prefix_source (即 Z_d_prev)
            T_prev = state['T']

            # 1. 累计轨迹更新 (TraceEMA: per-slot 可学习 α)
            if self.use_trace_updater:
                T_c = self.trace_ema(T_prev, S_c)
                flow_query = T_c
            else:
                T_c = S_c  # 简化模式: 直接用 S_c
                flow_query = S_c

            # 保存 T_c (用于对比学习)
            all_T_c.append(T_c)

            # 2. 计算 commit_score (训练时用于 commit loss 监督)
            S_prev = state.get('_last_S_c', torch.zeros_like(S_c))
            commit_score = self.commit_gate(S_running=S_c, S_prev=S_prev, Z_d=prefix_source_prev)
            all_commit_scores.append(commit_score)  # [B, 1]

            # 3. 方案 D: Latent Draft Flow
            if visual_hidden_proj is not None:
                prefix_source_new = self.compute_prefix_source(
                    S_c=flow_query,
                    Z_d_prev=prefix_source_prev,
                    visual_hidden=visual_hidden_proj,
                )
            else:
                prefix_source_new = self.compute_prefix_source(
                    S_c=flow_query,
                    Z_d_prev=prefix_source_prev,
                    visual_hidden=Z_e,
                )

            # 4. 根据 update_mask 选择性更新
            if mask is not None:
                float_mask = mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                prefix_source_new = float_mask * prefix_source_new + (1.0 - float_mask) * prefix_source_prev
                T_c = float_mask * T_c + (1.0 - float_mask) * T_prev

            state = {
                'Z_e': Z_e,
                'visual_hidden_proj': visual_hidden_proj,  # 透传
                'Z_d_init': Z_d_init,                      # 透传
                'Z_d': prefix_source_new,                  # 方案 D: latent draft flow 输出
                'T': T_c,
                'step_count': state['step_count'] + 1,
                '_last_S_c': S_c.detach(),  # 保存当前 S_c 供下一步 commit_gate 使用
            }

            all_Z_d.append(prefix_source_new)

        # 将 T_c, commit_scores 列表保存到 state 中
        state['_all_T_c'] = all_T_c
        state['_all_commit_scores'] = all_commit_scores  # 用于 commit loss

        return state, all_Z_d

    def compute_contrastive_targets(
        self,
        state: Dict[str, torch.Tensor],
        correct_step_summaries: List[torch.Tensor],  # list of [B, K_t, d_z], 长度 = C_correct
        correct_update_masks: List[torch.Tensor],     # list of [B] bool, 长度 = C_correct
    ) -> Dict[str, torch.Tensor]:
        """
        从 correct_cot 的 step summaries 中提取对比学习目标
        
        方案 D: 不再需要 BidirectionalReflection 产生 G_c,
        对比学习目标只需要 T_c (轨迹状态)。
        
        注意: 此方法不更新 Z_d (correct path 不需要 draft 更新),
        只需要 T_c 作为对比目标。
        
        Args:
            state: 初始 RLD 状态 (与 wrong path 共享同一个 prefill 状态)
            correct_step_summaries: correct_cot 的 step summaries
            correct_update_masks: correct_cot 的 per-sample update masks
        
        Returns:
            contrastive_targets: {
                'T_correct_final': [B, K_t, d_z],  # 正确轨迹的最终状态
                'all_T_correct': list of [B, K_t, d_z],  # 每步的 T_correct
            }
        """
        T_prev = torch.zeros_like(state['T'])  # 从零开始 (与 wrong path 的 prefill 一致)
        
        all_T_correct = []
        
        for c in range(len(correct_step_summaries)):
            S_c = correct_step_summaries[c]
            mask = correct_update_masks[c]
            
            # 1. 累计轨迹更新 (TraceEMA: per-slot 可学习 α)
            if self.use_trace_updater:
                T_c = self.trace_ema(T_prev, S_c)
            else:
                T_c = S_c  # 简化模式: 直接用 S_c
            
            # 2. 根据 update_mask 选择性更新
            if mask is not None:
                float_mask = mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                T_c = float_mask * T_c + (1.0 - float_mask) * T_prev
            
            all_T_correct.append(T_c)
            T_prev = T_c
        
        # 如果没有 step summaries, 返回零向量
        if len(correct_step_summaries) == 0:
            T_correct_final = torch.zeros_like(state['T'])
        else:
            T_correct_final = T_prev  # 最后一步的 T_c
        
        return {
            'T_correct_final': T_correct_final.detach(),
            'all_T_correct': [t.detach() for t in all_T_correct],
        }

    def compute_diversity_loss(
        self,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算多样性正则化损失
        
        方案 B: 对 prefix_source (state['Z_d']) 做去相关正则化
        旧模式: 同时对 Z^d 和 Z^e 进行去相关正则化
        双向检索模式: 只对 prefix_source 做正则化
        """
        prefix_source = state['Z_d']  # 方案 B: 这里是 prefix_source
        loss_d = self.diversity_reg(prefix_source)
        
        if self.use_bidirectional_reflection:
            loss = loss_d * self.lambda_div
        else:
            Z_e = state['Z_e']
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
        prefix_source = state['Z_d'].detach()  # 方案 D: latent draft flow 输出

        # 1. prefix_source 有效秩
        metrics['draft/Zd_effective_rank'] = self._effective_rank(prefix_source)

        # 2. prefix_source 槽间余弦相似度 (越低越好，高表示坍塌)
        metrics['draft/Zd_slot_cosim'] = self._slot_cosine_similarity(prefix_source)

        # 3. prefix_source L2 范数
        metrics['draft/Zd_norm'] = prefix_source.norm(dim=-1).mean().item()

        # 4. Z_e 有效秩 (仅旧模式, 双向模式下 Z_e 是零占位符)
        if not self.use_bidirectional_reflection:
            Z_e = state['Z_e'].detach()
            metrics['draft/Ze_effective_rank'] = self._effective_rank(Z_e)

        # 4b. 双向检索模式: visual_hidden_proj 的统计
        visual_hidden_proj = state.get('visual_hidden_proj')
        if visual_hidden_proj is not None:
            metrics['draft/visual_proj_num_tokens'] = float(visual_hidden_proj.shape[1])
            metrics['draft/visual_proj_norm'] = visual_hidden_proj.norm(dim=-1).mean().item()

        # 5. step_update 中收集的指标
        if '_monitor' in state:
            mon = state['_monitor']
            metrics['draft/Zd_abs_delta'] = mon.get('Zd_abs_delta', 0.0)
            metrics['draft/Zd_relative_delta'] = mon.get('Zd_relative_delta', 0.0)

        # 6. step_count
        metrics['draft/step_count'] = float(state['step_count'])
        metrics['draft/num_steps'] = float(state['step_count'])

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
        print(f"  📌 方案 D: Latent Draft Flow (隐空间流 + LayerScale)")
        print(f"     摒弃一致性检验范式, 把 Z_d 看成连续演化的隐空间信号流")
        print(f"     信号流: flow_target = CA(RMSNorm(S_c + CA(S_c, Z_d_prev)), VH)")
        print(f"             Z_d_new = Z_d_prev + γ_i · (flow_target - Z_d_prev)  [γ_i: per-slot LayerScale, 零初始化]")
        print(f"     (BidirectionalReflection/DraftRefiner/旧方案参数保留用于加载旧 checkpoint, forward 中不使用)")
        if not self.use_trace_updater:
            print(f"  📌 简化模式: TraceEMA 已禁用, 直接用 S_c 作为 flow query")
            print(f"     (TraceUpdater/TraceEMA 参数仍保留用于兼容旧 checkpoint 加载)")
        else:
            print(f"  📌 轨迹模式: TraceEMA (per-slot 可学习 α EMA 融合)")
            print(f"     T_c = α·T_prev + (1-α)·S_c, 逐 slot 独立更新, 天然保持 slot 多样性")
            print(f"     (旧版 TraceUpdater 2层CA 参数保留用于兼容旧 checkpoint 加载)")
        print(f"  📌 S_c 范数控制: StepResampler 输出经 RMSNorm 归一化, 范数稳定在 ~sqrt(d_z)≈27")
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
