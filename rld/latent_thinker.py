"""
NativeLatentThinker: 基于 VLM 原生隐空间的自适应多步思考模块 (COCONUT 等价)

核心思想:
  在每个 step boundary 处，取最后一个 token 的 hidden state 作为 "thought token"，
  在隐空间中做多步步进演化 (recurrence)，等价于 COCONUT 的连续思考过程：
  - 历史信息全部通过 KV cache 提供 (thought token 通过 attention 看到所有历史)
  - 每步: thought_token → text_model() 整体调用 (全部 36 层 + norm) → new_hidden
  - 多步: new_hidden 作为下一步的输入 → 再过一遍全部层 (recurrence)
  最终输出: 全部步的 hidden states [B, num_steps, H]，作为 prefix 注入下一段自然语言推理

  与 COCONUT 的等价性:
  - COCONUT: last hidden → 过 VLM 全部层 → 新 hidden (不解码) → 循环
  - NLD: last hidden → text_model() (全部 36 层 + norm, with KV cache) → 新 hidden → 循环
  - 两者都: 把 Transformer 当 RNN cell 用，hidden state 循环反馈

  关键设计:
  - 单个 thought token: 不需要 "last N hidden"，历史信息全在 KV cache 中
  - thought token 只是一个 "查询探针"，通过 attention 访问所有历史
  - 多步演化 = 加深 Transformer 层数 (recurrence = depth)
  - KV 累积: 每一步 thought 的 KV 自动追加到 cache 中，
    后续步骤能通过 attention 看到之前所有 thought + 全部历史 token
  - 统一序列: thought tokens 作为特殊 token 永久存在于推理序列中，
    隐式推理 + 自然语言 CoT 构成完整的、连续的推理过程

  方案 B (梯度可回传, 训推一致):
  - 训练和推理都使用局部 cache (clone detached global cache)
    thought KV 写入局部 cache，不修改 global_cache
    thinker 完成后，thought_output 作为 prefix embedding 注入下一个 segment/token
    下一个 segment 的 text_model() 把 thought prefix + segment tokens 的 KV 一起写入 global_cache
  - 训练时: 梯度路径: CE loss → Segment hidden → inputs_embeds → thought_output → thinker → VLM
  - 推理时: thought_output 作为 prefix 再过一遍 text_model() 写入主 cache
  - 训推行为完全一致: thought tokens 在 global_cache 中的 KV 都是 text_model(thought_output) 产生的

  统一 forward 架构:
  - 训练和推理都使用 text_model() 整体调用 (而非手动逐层 forward)
  - KV cache 由 DynamicCache 自动管理
  - gradient checkpointing 由 text_model 内部正确处理
  - 代码简洁，训推一致

信号流:
  Step boundary 处:
  last_token_hidden [B, 1, H] (最后一个 token 的 hidden state, 已过 norm)
  → + step_embedding (标识当前是第几步隐空间演化)
  → text_model() 整体调用 (全部 36 层 + norm, with KV cache)
    (通过 KV cache attend 到所有历史 token + 之前所有 thought)
  → 输出 hidden [B, 1, H] (已过 norm)
  → KV 写入局部 cache (训练时) 或全局 cache (推理时)
  → 作为下一步的输入 (recurrence) 或退出
  → 退出判断: 退出 token 预测 OR 饱和信号 (推理时)
  → output_norm (额外 norm, 用于 prefix embedding)
  → 收集全部步的 hidden states → thought_output [B, num_steps, H]
  → 全部步的 hidden states 作为下一个 segment 的 prefix embeddings (训练和推理都一样)
  → prefix 过 text_model() 写入 global_cache (训推一致)
  
  Hidden States 监督:
  - Key Token Loss (Soft Concept Anchoring): hidden → lm_head (frozen) → logit distribution
    与 key tokens 的 embedding 语义邻域构成的 soft target 做 KL 散度对齐
  - Exit Token Loss: 最后一步 hidden → lm_head (frozen) → 应预测 <|/latent|>
  - Visual Probe: hidden → probe MLP → 应恢复视觉特征方向
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from transformers.cache_utils import DynamicCache


def compute_saturation(
    prev_hidden: torch.Tensor,
    curr_hidden: torch.Tensor,
) -> torch.Tensor:
    """
    计算相邻两步 hidden state 的饱和度 (余弦相似度)
    
    饱和度 ≈ 1.0 表示 hidden state 不再变化 (充分条件: 可以退出)
    饱和度 ≈ 0.0 表示 hidden state 仍在剧烈变化
    
    Args:
        prev_hidden: [B, 1, H] 上一步的 hidden state
        curr_hidden: [B, 1, H] 当前步的 hidden state
    
    Returns:
        saturation: [B] 饱和度 ∈ [-1, 1]，通常 ∈ [0.8, 1.0]
    """
    prev = prev_hidden.squeeze(1)  # [B, H]
    curr = curr_hidden.squeeze(1)  # [B, H]
    return F.cosine_similarity(prev, curr, dim=-1)  # [B]


class NativeLatentThinker(nn.Module):
    """
    原生隐空间思考器: COCONUT 等价的隐空间推理 (单 token recurrence, 全层)
    
    核心设计:
    1. 单个 thought token — 最后一个 token 的 hidden state 作为 "查询探针"
    2. 历史信息全部通过 KV cache 提供 — thought token 通过 attention 访问所有历史
    3. 多步演化 = recurrence — 输出 hidden 作为下一步输入，再过一遍全部层
    4. 不过 lm_head 解码 — 纯隐空间步进
    5. 复用 VLM 的权重 — 利用 VLM 已有的推理能力
    
    统一 forward 架构:
    - 使用 text_model() 整体调用替代手动逐层 forward
    - KV cache 由 DynamicCache 自动管理
    - gradient checkpointing 由 text_model 内部正确处理
    - 训推一致: 训练和推理使用完全相同的 forward 路径
    
    退出策略 (简化版, 无 ExitGate):
    - 训练时: 固定执行 target_steps 步 (由数据中 latent_key_tokens 的 stage 数决定)
    - 推理时: 退出 token 预测 OR 饱和信号 (双保险) OR max_think_steps 硬上限
    
    Hidden States 监督:
    - Key Token Loss (Soft Concept Anchoring): hidden → lm_head (frozen) → logit distribution
      与 key tokens 的 embedding 语义邻域构成的 soft target 做 KL 散度对齐
    - Exit Token Loss: 最后一步 hidden → lm_head (frozen) → 应预测 <|/latent|>
    - lm_head 在这两个 loss 中冻结 (detach)，只塑形 hidden states
    
    与 COCONUT 的等价性:
    - COCONUT: last hidden → 过 VLM 全部层 → 新 hidden (不解码) → 循环
    - NLD: last hidden → text_model() (全部 36 层 + norm) → 新 hidden (不解码) → 循环
    - 两者都: 把 Transformer 当 RNN cell 用，hidden state 循环反馈
    - 关键: thought token 只有 1 个，历史信息全在 KV cache 中
    - KV 累积: 每步 thought KV 追加到 cache，后续步可 attend 所有之前的 thought + 历史 token
    
    Args:
        hidden_size: VLM 的隐藏维度 (如 4096)
        max_think_steps: 最大隐空间演化步数 (硬上限, 无论任何信号都强制退出, 如 4~8)
        saturation_exit_threshold: 推理时饱和信号的退出阈值
        latent_end_token_id: <|/latent|> token id，用于退出预测
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        max_think_steps: int = 4,
        saturation_exit_threshold: float = 0.99,
        latent_end_token_id: Optional[int] = None,
        # 以下参数保留接口兼容性，但不再使用
        adaptive_exit: bool = True,
        exit_threshold: float = 0.5,
        ponder_cost_weight: float = 0.01,
        saturation_aux_weight: float = 0.005,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_think_steps = max_think_steps
        self.saturation_exit_threshold = saturation_exit_threshold
        self.latent_end_token_id = latent_end_token_id  # <|/latent|> token id，用于退出预测
        
        # 保留接口兼容性 (不再实际使用)
        self.adaptive_exit = adaptive_exit
        
        # ====== 1. 步骤编码 (多步演化: 让模型知道当前是第几步) ======
        # 使用较大的 embedding 表大小，避免 max_think_steps 限制训练时的实际步数
        # 训练时 target_steps 可能超过 max_think_steps，需要足够的 step embedding
        self._step_embed_size = max(max_think_steps, 32)  # 至少 32，覆盖绝大多数场景
        if self._step_embed_size > 1:
            self.step_embedding = nn.Embedding(self._step_embed_size, hidden_size)
            nn.init.normal_(self.step_embedding.weight, std=0.02)
        else:
            self.step_embedding = None
        
        # ====== 2. 输出 Norm ======
        # thought_output 直接作为 prefix embedding 注入 VLM
        # text_model() 返回的 last_hidden_state 已过 final norm，
        # 这里的 output_norm 是额外的 norm，用于将 thought hidden 转换为 prefix embedding
        self.output_norm = nn.RMSNorm(hidden_size)
        
        # ====== 统计 ======
        self._init_param_count()
    
    def _init_param_count(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._total_params = total
        self._trainable_params = trainable
    
    @staticmethod
    def _clone_cache_detached(cache) -> DynamicCache:
        """Clone KV cache 并 detach 所有 tensor，用于创建局部 cache。
        
        thinker 内部多步演化时使用局部 cache，不修改传入的 global_cache。
        局部 cache 包含 global_cache 的 detached 副本（提供历史 context），
        thinker 的 thought KV 追加到局部 cache 中。
        
        Args:
            cache: 需要 clone 的 DynamicCache (global_cache)
        
        Returns:
            新的 DynamicCache，所有 KV tensor 已 detach (不影响原 cache)
        """
        from transformers.cache_utils import DynamicLayer as _DL
        cloned = DynamicCache()
        for layer in cache.layers:
            new_layer = _DL()
            if hasattr(layer, 'keys') and layer.keys is not None:
                new_layer.keys = layer.keys.detach()
                new_layer.values = layer.values.detach()
                new_layer.is_initialized = True
            cloned.layers.append(new_layer)
        return cloned
    
    def _run_think_step(
        self,
        thought_hidden: torch.Tensor,
        text_model: nn.Module,
        step_idx: int,
        past_key_values,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        执行单步隐空间演化 (COCONUT 等价):
        thought_hidden [B, 1, H] → text_model() 整体调用 → updated [B, 1, H]
        
        使用 text_model() 整体调用替代手动逐层 forward:
        - KV cache 由 DynamicCache 自动管理 (每步 thought KV 自动追加)
        - gradient checkpointing 由 text_model 内部正确处理
        - attention mask 由 text_model 内部自动构建
        
        关键: 通过 past_key_values 看到所有历史 token，
        这等价于 COCONUT 中 hidden state 能看到所有之前的 context。
        thought token 只是一个 "查询探针"，通过 attention 访问所有历史信息。
        
        Args:
            thought_hidden: [B, 1, H] 当前 thought token 的 hidden state
            text_model: VLM 的 language_model (整体调用)
            step_idx: 当前步骤索引 (用于步骤编码)
            past_key_values: 历史 KV cache (DynamicCache, 所有历史 token 的 context)
            attention_mask: attention mask (通常为 None, 让 text_model 自动处理)
            cache_position: cache 中的位置
            position_ids: position ids (4D: [text_pos + mrope_3d])
        
        Returns:
            updated_hidden: [B, 1, H] 更新后的 thought token hidden state (已过 norm)
        """
        device = thought_hidden.device
        hidden = thought_hidden
        
        # 加入步骤编码 (让模型知道当前是第几步演化)
        if self.step_embedding is not None:
            # 如果 step_idx 超出 embedding 表大小，使用最后一个 embedding (安全兜底)
            clamped_idx = min(step_idx, self._step_embed_size - 1)
            step_emb = self.step_embedding(
                torch.tensor(clamped_idx, device=device)
            )  # [H]
            hidden = hidden + step_emb.unsqueeze(0).unsqueeze(0)  # [B, 1, H] + [1, 1, H]
        
        # 使用 text_model() 整体调用 (全部 36 层 + norm)
        # KV cache 自动管理: thought token 的 KV 自动追加到 past_key_values
        # gradient checkpointing: text_model 内部正确处理 (不会导致 KV 重复追加)
        outputs = text_model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        
        # last_hidden_state 已过 final norm
        updated_hidden = outputs.last_hidden_state  # [B, 1, H]
        
        return updated_hidden
    
    def forward(
        self,
        last_hidden: torch.Tensor,
        text_model: nn.Module,
        past_key_values,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb_fn=None,
        mrope_position_ids=None,
        base_thought_pos: Optional[int] = None,
        B: Optional[int] = None,
        target_steps: Optional[int] = None,
        lm_head: Optional[nn.Module] = None,
        embed_tokens: Optional[nn.Module] = None,
        stage_key_token_ids: Optional[List[List[List[int]]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        COCONUT 等价的隐空间推理 (单 token recurrence, text_model() 整体调用)
        
        统一 forward 架构:
        - 使用 text_model() 整体调用替代手动逐层 forward
        - KV cache 由 DynamicCache 自动管理
        - 训推一致: 训练和推理使用完全相同的 forward 路径
        
        退出策略 (简化版, 无 ExitGate):
        - 训练时: 固定执行 target_steps 步 (由数据中 latent_key_tokens 的 stage 数决定)
        - 推理时: 退出 token 预测 OR 饱和信号 OR max_think_steps 硬上限
        
    Hidden States 监督 (Soft Concept Anchoring + Exit Token):
        - Key Token Loss (SCA): 每步 hidden 过 lm_head (frozen) 后的 logit distribution
          与 key tokens 的 embedding 语义邻域构成的 soft target 做 KL 散度对齐 (Top-K 近似)
        - Exit Token Loss: 最后一步 hidden 过 lm_head (frozen) 后应预测 <|/latent|>
        - lm_head 在这两个 loss 中冻结 (detach)，只塑形 hidden states
        
        Args:
            last_hidden: [B, 1, H] 最后一个 token 的 hidden state (已过 norm)
            text_model: VLM 的 language_model (整体调用)
            past_key_values: 历史 KV cache (DynamicCache)
            attention_mask: attention mask
            cache_position: 第一步 thought token 在 cache 中的位置
            position_ids: thought token 的 position ids (4D)
            rotary_emb_fn: VLM 的 rotary_emb 函数
            mrope_position_ids: [3, B, 1] 第一步的 mrope position ids
            base_thought_pos: int, 第一步 thought 的起始位置
            B: batch size
            target_steps: 目标迭代步数 (训练时由数据决定, 推理时为 None)
            lm_head: VLM 的 lm_head (用于 key token loss 和退出 token 预测)
            embed_tokens: VLM 的 input embedding 层 (用于 Soft Concept Anchoring 构建 soft target)
            stage_key_token_ids: [B][stage][token_ids] 每个 batch 样本每个 stage 的关键词 token ids
        
        Returns:
            dict 包含:
            - thought_output: [B, num_steps, H] 全部步的隐空间思考结果
            - exit_token_loss: scalar, 退出 token 预测 loss
            - key_token_loss: scalar, Key Token Loss (Soft Concept Anchoring via KL divergence)
            - exit_stats: dict, 退出统计信息
            - num_thought_steps: int, 实际执行的思考步数
        """
        if B is None:
            B = last_hidden.shape[0]
        device = last_hidden.device
        dtype = last_hidden.dtype
        
        # ---- 单个 thought token: 最后一个 token 的 hidden state ----
        thought_hidden = last_hidden  # [B, 1, H]
        
        # ---- 方案 B: 始终使用局部 cache，不修改 global cache ----
        local_cache = self._clone_cache_detached(past_key_values)
        think_cache = local_cache
        
        # ---- 确定实际迭代步数 ----
        # 训练时: 完全由 target_steps 决定 (数据中几个 stage 就训练几步，不受 max_think_steps 截断)
        # 推理时: max_think_steps 作为安全上限 (配合自适应退出)
        if self.training and target_steps is not None and target_steps > 0:
            effective_steps = target_steps  # 训练时不截断，完全由数据决定
        elif target_steps is not None and target_steps > 0:
            effective_steps = target_steps  # 推理时也优先使用指定步数
        else:
            effective_steps = self.max_think_steps  # 推理时无指定步数，使用 max_think_steps
        
        # ---- 当前步的 cache_position 和 position_ids (每步递增) ----
        current_cache_pos = cache_position
        current_position_ids = position_ids
        
        # ---- 辅助函数: 构建某一步的 position_ids (4D: text_pos + mrope_3d) ----
        def _build_step_position_ids(step_idx):
            """为第 step_idx 步构建 position_ids [4, B, 1]"""
            step_pos = base_thought_pos + step_idx
            step_positions = torch.arange(step_pos, step_pos + 1, device=device)
            step_cache_pos = step_positions
            
            # text_pos: [B, 1]
            text_pos = torch.full((B, 1), step_pos, device=device, dtype=torch.long)
            
            # mrope: [3, B, 1]
            if mrope_position_ids is not None:
                step_mrope = mrope_position_ids[:, :, :1] + step_idx  # [3, B, 1]
            else:
                step_mrope = step_positions.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
            
            # 拼接为 [4, B, 1]: [text_pos, mrope_3d]
            pos_ids = torch.cat([text_pos.unsqueeze(0), step_mrope], dim=0)  # [4, B, 1]
            return step_cache_pos, pos_ids
        
        # ---- 多步隐空间演化 (Coconut recurrence, KV 累积) ----
        all_step_outputs = []
        all_saturations = []
        prev_hidden_for_saturation = thought_hidden.detach()
        
        for step_idx in range(effective_steps):
            # 为当前步生成 position_ids 和 cache_position (每步递增位置)
            if step_idx > 0:
                current_cache_pos, current_position_ids = _build_step_position_ids(step_idx)
            
            # 构建 attention_mask
            step_pos = base_thought_pos + step_idx
            step_attn_mask = torch.ones(B, step_pos + 1, device=device, dtype=torch.long)
            
            # 单步演化: thought_hidden → text_model() → new_hidden
            thought_hidden = self._run_think_step(
                thought_hidden, text_model, step_idx,
                past_key_values=think_cache,
                attention_mask=step_attn_mask,
                cache_position=current_cache_pos,
                position_ids=current_position_ids,
            )
            all_step_outputs.append(thought_hidden)
            
            # 计算饱和度
            saturation = compute_saturation(
                prev_hidden_for_saturation, thought_hidden
            )  # [B]
            all_saturations.append(saturation)
            prev_hidden_for_saturation = thought_hidden.detach()
            
            # ---- 推理时: 退出 token 预测 OR 饱和信号 (双保险) ----
            if not self.training and step_idx < effective_steps - 1:
                saturation_exit = (saturation > self.saturation_exit_threshold).all()
                
                # 退出 token 检测: 如果 lm_head 预测出 <|/latent|>，退出
                token_exit = False
                if (lm_head is not None and self.latent_end_token_id is not None):
                    with torch.no_grad():
                        step_logits = lm_head(thought_hidden[:, -1:, :])  # [B, 1, V]
                        predicted_token = step_logits.argmax(dim=-1).squeeze(-1)  # [B]
                        token_exit = (predicted_token == self.latent_end_token_id).all().item()
                
                if saturation_exit or token_exit:
                    break
        
        actual_steps = len(all_step_outputs)
        
        # ---- 输出全部步的 hidden states ----
        normed_outputs = [self.output_norm(out) for out in all_step_outputs]
        thought_output = torch.cat(normed_outputs, dim=1)  # [B, actual_steps, H]
        
        # ---- 统一获取 lm_head 冻结权重 (Exit Token Loss + Key Token Loss 共用) ----
        # lm_head 冻结: 用 detached 权重计算 logits，保留 hidden states 的梯度
        # 梯度只回传到 hidden states，不修改 lm_head 权重
        exit_token_loss = torch.tensor(0.0, device=device, dtype=dtype)
        key_token_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        if (self.training and lm_head is not None and actual_steps > 0):
            all_hiddens = torch.cat(all_step_outputs, dim=1)  # [B, actual_steps, H]
            
            # 获取 lm_head 的 detached 权重 (只需获取一次)
            lm_weight = lm_head.weight.detach()
            lm_bias = lm_head.bias.detach() if lm_head.bias is not None else None
            
            # 计算所有步的 logits (只需计算一次)
            step_logits = F.linear(all_hiddens, lm_weight, lm_bias)  # [B, actual_steps, V]
            
            # ---- Exit Token Loss: 最后一步应预测 <|/latent|> ----
            if self.latent_end_token_id is not None:
                exit_targets = torch.full(
                    (B, actual_steps), -100, device=device, dtype=torch.long
                )
                exit_targets[:, -1] = self.latent_end_token_id  # 最后一步 → <|/latent|>
                
                exit_token_loss = F.cross_entropy(
                    step_logits.view(-1, step_logits.size(-1)),
                    exit_targets.view(-1),
                    ignore_index=-100,
                )
            
            # ---- Key Token Loss (Soft Concept Anchoring) ----
            if stage_key_token_ids is not None and embed_tokens is not None:
                key_token_loss = self._compute_sca_loss(
                    all_hiddens, step_logits, embed_tokens, lm_weight,
                    stage_key_token_ids, B, actual_steps, device, dtype
                )
        
        # ---- 退出统计 ----
        with torch.no_grad():
            exit_reason = "training" if self.training else "max_steps"
            if not self.training and actual_steps < effective_steps:
                # 检查退出原因
                _token_predicted = False
                if (lm_head is not None and self.latent_end_token_id is not None):
                    _last_logits = lm_head(all_step_outputs[-1])
                    _pred = _last_logits.argmax(dim=-1).squeeze(-1)
                    _token_predicted = (_pred == self.latent_end_token_id).all().item()
                
                if _token_predicted:
                    exit_reason = "exit_token"
                elif all_saturations and (all_saturations[-1] > self.saturation_exit_threshold).all():
                    exit_reason = "saturation"
                else:
                    exit_reason = "mixed_signal"
        
        exit_stats = {
            'mean_steps': float(actual_steps),
            'saturations': [sat.detach() for sat in all_saturations],
            'actual_steps': actual_steps,
            'exit_reason': exit_reason,
        }
        
        return {
            'thought_output': thought_output,  # [B, actual_steps, H]
            'exit_token_loss': exit_token_loss,
            'key_token_loss': key_token_loss,
            'exit_stats': exit_stats,
            'num_thought_steps': actual_steps,
        }
    
    def _compute_sca_loss(
        self,
        all_hiddens: torch.Tensor,
        step_logits: torch.Tensor,
        embed_tokens: nn.Module,
        lm_weight: torch.Tensor,
        stage_key_token_ids: List[List[List[int]]],
        B: int,
        actual_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        tau: float = 2.0,
        topk: int = 128,
    ) -> torch.Tensor:
        """
        Soft Concept Anchoring (SCA) via Self-Distillation:
        
        核心思想:
          利用模型自身的 input embedding 层构建 soft target distribution，
          然后用 KL 散度引导每步 hidden state 的 logit 分布向对应 stage 的
          概念语义邻域 "倾斜"。
        
        设计理由:
          - Self-Distillation: 教师信号来自模型自身的 embedding 层，不引入外部知识
          - Soft Target: 不是 one-hot 的，而是以 key tokens 为中心、向语义相近词扩散
          - Top-K 近似: 只在 key tokens + 它们的 Top-K 语义近邻上计算 KL，
            避免 152K 词表的全量 softmax 计算
          - 温度控制: τ 越大分布越平滑 (更宽松)，τ 越小分布越尖锐 (更严格)
          - 梯度质量: KL 散度始终有梯度 (不像 Margin Ranking 在满足条件后梯度为 0)
        
        数学公式:
          1. Soft Target (教师): q_i = softmax( mean(E[k_j]) · E^T / τ )
             其中 E 是 embedding matrix, k_j 是 key token ids
          2. Student Distribution: p_i = softmax( h_i · W_lm^T / τ )
             其中 h_i 是第 i 步 hidden state, W_lm 是冻结的 lm_head 权重
          3. Loss: KL(q || p)  (在 Top-K 子集上计算)
        
        Args:
            all_hiddens: [B, actual_steps, H] 所有步的 hidden states (有梯度)
            step_logits: [B, actual_steps, V] 已计算好的 logits (lm_head frozen, 有梯度)
            embed_tokens: VLM 的 input embedding 层
            lm_weight: [V, H] lm_head 的 detached 权重
            stage_key_token_ids: [B][stage][token_ids] 每个 batch 样本每个 stage 的关键词 token ids
            B: batch size
            actual_steps: 实际步数
            device: 设备
            dtype: 数据类型
            tau: 温度参数 (控制 soft target 的平滑度)
            topk: Top-K 近似的 K 值 (只在 K 个 token 上计算 KL)
        
        Returns:
            sca_loss: scalar (始终有 grad_fn)
        """
        # 获取 embedding matrix (冻结，不参与梯度)
        embed_weight = embed_tokens.weight.detach()  # [V, H]
        V = embed_weight.shape[0]
        
        kl_losses = []
        
        for b_idx in range(B):
            b_concepts = stage_key_token_ids[b_idx] if b_idx < len(stage_key_token_ids) else []
            
            for step_idx in range(actual_steps):
                if step_idx >= len(b_concepts) or len(b_concepts[step_idx]) == 0:
                    continue
                
                key_ids = b_concepts[step_idx]  # List[int]
                key_ids_tensor = torch.tensor(key_ids, device=device, dtype=torch.long)
                
                # ---- Step 1: 构建 Soft Concept Target (教师信号) ----
                # 获取 key tokens 的 embedding 并取平均，作为 "概念中心"
                key_embeds = embed_weight[key_ids_tensor]  # [m, H]
                concept_center = key_embeds.mean(dim=0)  # [H]
                
                # 计算概念中心与整个词表 embedding 的相似度
                # similarity: [V]
                concept_center_norm = F.normalize(concept_center.unsqueeze(0), dim=-1)  # [1, H]
                embed_weight_norm = F.normalize(embed_weight, dim=-1)  # [V, H]
                similarity = (concept_center_norm @ embed_weight_norm.T).squeeze(0)  # [V]
                
                # ---- Top-K 近似: 选取语义最近的 K 个 token ----
                # 确保 key tokens 本身一定在 Top-K 集合中
                topk_vals, topk_indices = similarity.topk(topk, dim=0)  # [K]
                
                # 合并 key_ids 到 topk_indices (确保 key tokens 不被遗漏)
                combined_indices = torch.cat([topk_indices, key_ids_tensor], dim=0)
                combined_indices = combined_indices.unique()  # 去重
                K_actual = combined_indices.shape[0]
                
                # ---- 在 Top-K 子集上构建 soft target ----
                # 用余弦相似度 / τ 作为 logits，然后 softmax
                subset_similarity = similarity[combined_indices]  # [K_actual]
                q_target = F.softmax(subset_similarity / tau, dim=0)  # [K_actual]
                
                # ---- Step 2: 获取 Student Distribution (在 Top-K 子集上) ----
                # 当前步的 logits (已有梯度，通过 all_hiddens → F.linear)
                student_logits = step_logits[b_idx, step_idx, combined_indices]  # [K_actual]
                p_student = F.log_softmax(student_logits / tau, dim=0)  # [K_actual]
                
                # ---- Step 3: KL(q || p) ----
                # KL(q || p) = sum(q * (log(q) - log(p)))
                kl = F.kl_div(p_student, q_target, reduction='sum')  # scalar
                
                # 温度缩放: KL 散度乘以 τ² (标准 distillation 做法)
                kl = kl * (tau ** 2)
                
                kl_losses.append(kl)
        
        if kl_losses:
            sca_loss = torch.stack(kl_losses).mean()
        else:
            # 无有效 key tokens 时，返回 0 但保持梯度链路
            sca_loss = (step_logits.sum() * 0.0).to(dtype)
        
        return sca_loss
    
    def print_summary(self):
        """打印模块摘要"""
        print(f"\n{'='*60}")
        print(f"NativeLatentThinker 模块摘要 (Coconut 等价, 单 token recurrence)")
        print(f"{'='*60}")
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  thought_tokens: 1 (单个查询探针, 历史信息全在 KV cache 中)")
        print(f"  forward 方式: text_model() 整体调用 (全部层 + norm)")
        print(f"  max_think_steps: {self.max_think_steps} (推理时安全上限, 训练时由数据 target_steps 决定)")
        print(f"  step_embed_size: {self._step_embed_size} (step embedding 表大小)")
        print(f"  saturation_exit_threshold: {self.saturation_exit_threshold}")
        print(f"  推理退出策略: 退出 token 预测 OR 饱和信号 (双保险)")
        print(f"  训练监督: Exit Token Loss + Key Token Loss (Soft Concept Anchoring, Top-K={128})")
        print(f"  设计: last_hidden [B,1,H] → text_model() (with KV cache) × steps → prefix")
        print(f"  COCONUT 等价: hidden state 循环反馈 (Transformer as RNN cell, 全层)")
        print(f"  KV 累积: 每步 thought KV 自动追加到 cache, 后续步可 attend 所有历史 + 之前 thought")
        print(f"  总参数: {self._total_params:,} ({self._total_params/1e6:.2f}M)")
        print(f"  可训练参数: {self._trainable_params:,} ({self._trainable_params/1e6:.2f}M)")
        print(f"{'='*60}\n")


class StageConceptAligner(nn.Module):
    """
    [已废弃] 阶段概念对齐 (Stage-Aligned Concept Supervision, SACS)
    
    已被 Key Token Decoding Loss (Margin Ranking) 替代。
    Key Token Decoding Loss 的实际逻辑在 NativeLatentThinker._compute_key_token_loss 中。
    
    保留此类仅为 import 兼容性。
    """
    
    def __init__(self, hidden_size: int, proj_dim=None, temperature: float = 0.1):
        super().__init__()
        # 空壳，不创建任何参数
    
    def forward(self, thought_hiddens, stage_concept_embeds, valid_mask=None):
        return torch.tensor(0.0, device=thought_hiddens.device, dtype=thought_hiddens.dtype)


class VisualProbe(nn.Module):
    """
    视觉证据探针: 检测 thought hidden 是否编码了视觉信息
    
    核心思想:
      不要求 thought hidden 在表示空间上靠近 visual embedding，
      而是要求 thought hidden 中**编码了足够的视觉信息**，
      使得一个简单的线性 probe 能从中恢复出视觉特征的方向。
      
      这允许 thought 在自己的表示空间中自由组织信息（与 CoT 语义连贯），
      同时确保它确实"看到了"图像。
    
    设计:
      - 轻量级 MLP probe: thought_hidden → predicted_visual_direction
      - 用余弦相似度作为 loss (不要求完全匹配，只要求方向一致)
      - probe 的梯度同时回传到 thinker (鼓励 thought 编码视觉信息)
      - 不参与推理，只在训练时使用
    
    Args:
        hidden_size: VLM 的隐藏维度
        probe_dim: probe 中间层维度 (默认 hidden_size // 8)
    """
    
    def __init__(self, hidden_size: int, probe_dim: Optional[int] = None):
        super().__init__()
        if probe_dim is None:
            probe_dim = max(hidden_size // 8, 64)
        
        self.probe = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, probe_dim),
            nn.SiLU(),
            nn.Linear(probe_dim, hidden_size),
        )
        
        # 初始化: 小权重，避免训练初期 probe loss 过大干扰主 loss
        nn.init.normal_(self.probe[-1].weight, std=0.01)
        nn.init.zeros_(self.probe[-1].bias)
    
    def forward(
        self,
        thought_hidden: torch.Tensor,
        visual_embeds_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算视觉证据探针 loss
        
        Args:
            thought_hidden: [B, H] thought 的平均表示 (所有 thought steps 的均值)
            visual_embeds_pooled: [B, H] 视觉特征的池化表示 (所有视觉 token 的均值)
        
        Returns:
            probe_loss: scalar, 1 - cosine_similarity (越低越好，表示 thought 编码了视觉信息)
        """
        predicted_visual = self.probe(thought_hidden)  # [B, H]
        # 用余弦相似度: 不要求完全匹配，只要求方向一致
        probe_loss = 1.0 - F.cosine_similarity(
            predicted_visual, visual_embeds_pooled.detach(), dim=-1
        ).mean()
        return probe_loss


def compute_complementarity_loss(
    thought_output: torch.Tensor,
    last_cot_hidden: torch.Tensor,
    redundancy_threshold: float = 0.7,
    redundancy_weight: float = 0.01,
    novelty_weight: float = 0.005,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    信息互补正则化: 确保 thought 与 CoT 既继承又互补
    
    核心思想:
      - thought 不应该只是 CoT 的复制品 (冗余惩罚)
      - thought 应该包含 CoT 子空间之外的新信息 (新信息鼓励)
      - 但 thought 也不能与 CoT 完全正交 (需要继承上下文)
    
    设计:
      1. 冗余惩罚: cos_sim(thought, cot)² > threshold 时惩罚
         → 防止 thought 退化为 CoT 的复制品
         → 允许一定程度的相似 (继承上下文)
      2. 新信息鼓励: thought 在 CoT 子空间的投影残差
         → 残差越大 → thought 包含的新信息越多
         → 鼓励 thought 提供 CoT 无法解释的信息
    
    Args:
        thought_output: [B, N_t, H] thinker 输出的 hidden states
        last_cot_hidden: [B, 1, H] 当前 segment 最后一个 CoT token 的 hidden state
        redundancy_threshold: 冗余惩罚的阈值 (cos_sim² > threshold 时惩罚)
        redundancy_weight: 冗余惩罚的权重
        novelty_weight: 新信息鼓励的权重
    
    Returns:
        complementarity_loss: scalar, 信息互补正则化 loss
        stats: dict, 统计信息 (冗余度、新信息量)
    """
    # thought 的平均表示
    thought_mean = thought_output.mean(dim=1)  # [B, H]
    cot_hidden = last_cot_hidden.squeeze(1).detach()  # [B, H]
    
    # 1. 冗余惩罚: thought 不应该只是 CoT 的复制品
    cos_sim = F.cosine_similarity(thought_mean, cot_hidden, dim=-1)  # [B]
    redundancy = cos_sim ** 2  # [B], ∈ [0, 1]
    # 只在相似度² > threshold 时惩罚 (允许一定程度的继承)
    redundancy_loss = F.relu(redundancy - redundancy_threshold).mean()
    
    # 2. 新信息鼓励: thought 在 CoT 子空间之外的信息
    cot_dir = F.normalize(cot_hidden, dim=-1)  # [B, H]
    # thought 在 CoT 方向上的投影
    projection = (thought_mean * cot_dir).sum(-1, keepdim=True) * cot_dir  # [B, H]
    # 残差: thought 中 CoT 无法解释的部分
    residual = thought_mean - projection  # [B, H]
    novelty = residual.norm(dim=-1)  # [B], 新信息量
    # Hinge 形式: 只在 novelty 不够大时才惩罚 (避免 loss 为负)
    # novelty_threshold: 期望 thought 至少有这么多新信息 (L2 norm)
    novelty_threshold = 1.0
    novelty_loss = F.relu(novelty_threshold - novelty).mean()
    
    # 组合 (两项都 >= 0，complementarity_loss 始终非负)
    complementarity_loss = redundancy_weight * redundancy_loss + novelty_weight * novelty_loss
    
    stats = {
        'redundancy': redundancy.mean().item(),
        'novelty': novelty.mean().item(),
        'cos_sim': cos_sim.mean().item(),
    }
    
    return complementarity_loss, stats


def compute_residual_value_loss(
    logits_at_boundary: torch.Tensor,
    labels_at_boundary: torch.Tensor,
    thought_output: torch.Tensor,
    last_cot_hidden: torch.Tensor,
    lm_head: nn.Module,
    K: int = 5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    残差价值 Loss (轻量近似版): 衡量 thought 对后续预测的增益
    
    核心思想:
      不需要额外跑一遍 segment forward，而是在 logits 层面做对比:
      - 有 thought 时: 正常的 logits (已经算好了)
      - 无 thought 时的近似: 用 boundary 处的 last_cot_hidden 直接过 lm_head
      
      如果两者的 KL 散度很小 → thought 没有改变预测分布 → thought 没用
      如果 KL 散度大且 loss 下降 → thought 提供了有益的新信息
    
    注意: 这是一个轻量近似，不需要额外的 forward pass。
    真正的残差价值需要完整的 bypass forward，计算开销较大。
    
    Args:
        logits_at_boundary: [B, K, V] 有 thought 时，boundary 后 K 个 token 的 logits
        labels_at_boundary: [B, K] 对应的 labels
        thought_output: [B, N_t, H] thinker 输出
        last_cot_hidden: [B, 1, H] boundary 处的 CoT hidden
        lm_head: VLM 的 lm_head
        K: 对比的 token 数量
    
    Returns:
        residual_loss: scalar, 残差价值 loss (鼓励 thought 提供正向增益)
        stats: dict, 统计信息
    """
    with torch.no_grad():
        # 无 thought 时的近似 logits: 用 last_cot_hidden 直接过 lm_head
        # 这假设如果没有 thought，后续 token 只能依赖 boundary 处的信息
        bypass_logits = lm_head(last_cot_hidden.expand(-1, K, -1)).float()  # [B, K, V]
        
        # 有 thought 时的 logits
        with_thought_logits = logits_at_boundary.float()  # [B, K, V]
        
        # 计算两者的 KL 散度 (信息增量)
        p_with = F.softmax(with_thought_logits, dim=-1)
        p_without = F.softmax(bypass_logits, dim=-1)
        
        # KL(p_with || p_without): thought 改变了多少预测分布
        kl_div = F.kl_div(
            p_without.log(), p_with, reduction='none'
        ).sum(-1).mean()  # scalar
        
        # 计算 CE loss 差异 (thought 带来的 loss 变化)
        valid_mask = (labels_at_boundary != -100)
        if valid_mask.any():
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            ce_with = loss_fct(
                with_thought_logits.view(-1, with_thought_logits.size(-1)),
                labels_at_boundary.view(-1)
            ).view(labels_at_boundary.shape)
            ce_without = loss_fct(
                bypass_logits.view(-1, bypass_logits.size(-1)),
                labels_at_boundary.view(-1)
            ).view(labels_at_boundary.shape)
            
            # advantage: 正值表示 thought 有帮助 (降低了 loss)
            advantage = (ce_without - ce_with).mean()
        else:
            advantage = torch.tensor(0.0, device=thought_output.device)
    
    # 残差价值 loss: 鼓励 thought 提供正向增益
    # 如果 advantage > 0 (thought 有帮助): 不惩罚
    # 如果 advantage <= 0 (thought 无帮助或有害): 惩罚
    # 用 hinge loss: max(0, -advantage + margin)
    margin = 0.01  # 要求 thought 至少带来 0.01 的 loss 下降
    residual_loss = F.relu(-advantage + margin)
    
    stats = {
        'kl_divergence': kl_div.item(),
        'advantage': advantage.item(),
    }
    
    return residual_loss, stats


class ThoughtPrefixInjector(nn.Module):
    """
    将 NativeLatentThinker 的输出转换为 prefix，注入到 VLM 的 attention 层
    
    方式 A (Embedding Prefix): 直接拼接到下一个 chunk 的 input embeddings 前面
      → 最简单，完全兼容 flash_attention_2
      → thought_output [B, 1, H] 直接作为 prefix embedding
    """
    
    def __init__(self, mode: str = "embedding"):
        super().__init__()
        self.mode = mode
    
    def forward(
        self,
        thought_output: torch.Tensor,
        chunk_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        将思考结果注入到下一个 chunk 的输入中
        
        Args:
            thought_output: [B, N_t, H] 隐空间思考结果 (N_t 个 thought token，
                           即隐式推理进行了几步就有几个)
            chunk_embeds: [B, chunk_len, H] 下一个 chunk 的 token embeddings
        
        Returns:
            injected_embeds: [B, N_t + chunk_len, H] 注入后的 embeddings
        """
        if self.mode == "embedding":
            return torch.cat([thought_output, chunk_embeds], dim=1)
        else:
            raise ValueError(f"不支持的注入模式: {self.mode}")
