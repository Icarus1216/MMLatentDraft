"""RLD Model: 包装 Qwen3-VL VLM 的 Reflective Latent Draft 模型

基座模型: Qwen3-VL-8B-Instruct (标准 full_attention + DeepStack 视觉注入)
- 在 <think> 块内部使用 </step> 做细粒度分步控制 RLD 更新
- Loss 来源采用分段监督: 高质量推理链 (free_reasoning/corrected) 的 think 块参与 loss,
  伪造/低质量的 think 块 labels=-100; final answer 始终参与 loss
- 梯度通过 KV cache 从 final answer 回传穿过所有段的 RLD Controller

核心设计:
1. 冻结 Qwen3-VL 全部参数
2. 只训练 RLD Controller 的外挂模块 (< 25M 参数)
3. 通过 Embedding 注入影响后续自回归生成，梯度自然连通
4. 训练时按 step 段展开，每个 token 只 forward 一次
5. 每段只做一次 text_model.forward (prefix + seg 合并), 总计 N 段 = N 次 forward

关键约束:
- 不对 Qwen3-VL 重复 forward
- 兼容原生推理过程 (flash attention / mRoPE)
- Qwen3-VL 有 DeepStack，视觉编码器返回 (hidden_states, deepstack_features)
- prefix embedding 参与完整 Transformer forward，CE Loss 梯度直达所有 controller 组件
- 训推一致: 训练和推理都使用 <think>...</think> + </step> 格式

与 Qwen3.5-4B 版本的主要差异:
- 标准 full_attention (无 linear_attention 混合层), KV cache 为标准 DynamicCache
- 有 DeepStack 视觉特征注入 (在 decoder 前几层注入多层视觉特征)
- hidden_size: 4096 (而非 2560)
- num_hidden_layers: 36 (而非 32)
- 无需特殊处理 conv_states / recurrent_states
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

# Qwen3-VL 版本: 使用 Qwen3VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .controller import RLDController


def _extract_visual_output(vision_output):
    """从视觉编码器输出中提取 merged_hidden_states 和 deepstack_features。
    
    Qwen3-VL 的 VisionModel 返回 (hidden_states, deepstack_feature_lists):
    - hidden_states: 经过 PatchMerger 后的视觉 token [total_visual_tokens, out_hidden_size]
    - deepstack_feature_lists: list of Tensor, 各层 DeepStack 特征
    
    兼容两种返回格式:
    - tuple 模式 (正常): vision_output = (hidden_states, deepstack_features)
    - 单个 tensor 模式 (fallback): 直接返回
    """
    if isinstance(vision_output, (tuple, list)) and len(vision_output) == 2:
        hidden_states = vision_output[0]
        deepstack_features = vision_output[1]
    elif isinstance(vision_output, (tuple, list)):
        hidden_states = vision_output[0]
        deepstack_features = None
    else:
        hidden_states = vision_output
        deepstack_features = None
    return hidden_states, deepstack_features


def _is_main_process():
    """判断当前进程是否为主进程"""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank() == 0
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', -1)))
    return local_rank in (-1, 0)


class RLDModel(nn.Module):
    """
    RLD Model: Qwen3-VL + Reflective Latent Draft Controller
    
    架构:
    ┌──────────────────────────────────────────────────────┐
    │  Qwen3-VL (冻结)                                          │
    │  ├── Vision Encoder → visual tokens V + DeepStack feats  │
    │  └── Decoder (自回归生成, 标准 full attn + DeepStack) │
    │                                                      │
    │  RLD Controller (可训练)                               │
    │  ├── Evidence Resampler: V → Z^e (冻结证据)           │
    │  ├── Step Resampler: H^step → S_c                    │
    │  ├── Trace Updater: [T_{c-1}; S_c] → T_c (2层CA)    │
    │  ├── Reflection: (T_c, Z^e) → G_c (2层CA)           │
    │  ├── Draft Updater: KV=G_c + T_c bias → Z^d_{c+1}   │
    │  ├── Evidence Gate: β(Z_d,step) ⊙ Z_e → Z_e_gated  │
    │  └── Embedding Projector: [Z_e'; Z^d] → prefix embs│
    └──────────────────────────────────────────────────────┘
    
    Args:
        model_path: Qwen3-VL 模型路径
        hidden_size: 隐藏维度 (4096 for Qwen3-VL-8B)
        d_z: controller 空间维度 (512)
        num_evidence_slots: 证据槽数量 (16)
        num_draft_slots: 草稿槽数量 (16)
        num_trace_slots: 轨迹槽数量 (16)
        total_layers: 总层数 (36)
        torch_dtype: 数据类型
        attn_implementation: 注意力实现
        lambda_div: 多样性正则化权重
    """

    # 步骤分隔符 token 文本
    STEP_DELIMITER = "</step>"

    def __init__(
        self,
        model_path: str,
        hidden_size: int = 4096,
        d_z: int = 512,
        num_evidence_slots: int = 16,
        num_draft_slots: int = 16,
        num_trace_slots: int = 16,
        total_layers: int = 36,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        lambda_div: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.total_layers = total_layers
        self._verbose = _is_main_process()

        # ====== 注意力实现: 支持通过环境变量 RLD_ATTN_IMPL 动态切换 ======
        # 优先级: 环境变量 > 调用方传入的参数 > 默认值 (flash_attention_2)
        # 可选值: flash_attention_2 / sdpa / eager
        env_attn = os.environ.get('RLD_ATTN_IMPL', None)
        if env_attn is not None and env_attn != attn_implementation:
            if self._verbose:
                print(f"[RLD] ⚠️  注意力实现: {attn_implementation} → {env_attn} (通过 RLD_ATTN_IMPL 环境变量覆盖)")
            attn_implementation = env_attn
        self.attn_implementation = attn_implementation

        # ====== 1. 加载并冻结 Qwen3-VL ======
        if self._verbose:
            print(f"[RLD] 加载 Qwen3-VL: {model_path}")
            print(f"[RLD] 注意力实现: {attn_implementation}")

        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        # 冻结全部基座参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        if self._verbose:
            frozen_params = sum(p.numel() for p in self.base_model.parameters())
            print(f"[RLD] Qwen3-VL 已冻结: {frozen_params / 1e9:.2f}B 参数")

        # ====== 2. 创建 RLD Controller ======
        self.controller = RLDController(
            hidden_size=hidden_size,
            d_z=d_z,
            num_evidence_slots=num_evidence_slots,
            num_draft_slots=num_draft_slots,
            num_trace_slots=num_trace_slots,
            total_layers=total_layers,
            num_heads=8,
            evidence_layers=2,
            use_gate=True,
            lambda_div=lambda_div,
        )

        # 确保 controller 使用与基座相同的 dtype
        self.controller = self.controller.to(dtype=torch_dtype)

        if self._verbose:
            self.controller.print_param_summary()

        # ====== 3. Step delimiter token ids (支持多 token 序列) ======
        self.step_delimiter_ids = None  # 在 set_processor 中设置
        self.step_delimiter_id = None   # 兼容单 token 场景

        # ====== 4. Prefix 长度 ======
        self.K_p = num_evidence_slots + num_draft_slots  # 32

    def set_processor(self, processor):
        """设置 processor 并获取 delimiter token id 序列
        
        P1-2: 不再强制将 </step> 注册为单 token special token。
        而是保留其原始分词结果 (可能是多个 token)，
        通过后缀匹配来检测 delimiter。
        """
        self.processor = processor
        tokenizer = processor.tokenizer

        # 获取 </step> 的分词结果
        delimiter_ids = tokenizer.encode(self.STEP_DELIMITER, add_special_tokens=False)

        self.step_delimiter_ids = delimiter_ids  # 可能是多个 token
        # 兼容: 如果恰好是单 token，也保留 step_delimiter_id
        if len(delimiter_ids) == 1:
            self.step_delimiter_id = delimiter_ids[0]
        else:
            self.step_delimiter_id = None

        if self._verbose:
            decoded = tokenizer.decode(delimiter_ids)
            print(f"[RLD] Step delimiter: '{self.STEP_DELIMITER}' → token_ids={delimiter_ids} (decoded='{decoded}')")
            if len(delimiter_ids) > 1:
                print(f"[RLD]   使用多 token 后缀匹配模式 (共 {len(delimiter_ids)} 个 token)")

    def _get_full_attn_cache_len(self, past_key_values) -> int:
        """安全获取 KV cache 序列长度
        
        Qwen3-VL 使用标准 full_attention，所有层的 KV cache 格式一致: [B, H, seq_len, D]
        """
        if past_key_values is None:
            return 0
        
        # 直接从第一个层的 KV cache 获取长度
        if hasattr(past_key_values, 'key_cache') and len(past_key_values.key_cache) > 0:
            kc = past_key_values.key_cache[0]
            if kc is not None and kc.dim() == 4:
                return kc.shape[2]
        
        # 回退: 使用默认方法
        return past_key_values.get_seq_length() if past_key_values is not None else 0

    def _find_delimiter_positions(self, token_ids: torch.Tensor) -> List[int]:
        """在 token 序列中查找所有 </step> delimiter 的末尾位置
        
        支持多 token delimiter 的后缀匹配。
        返回每个 delimiter 的最后一个 token 的位置索引。
        
        Args:
            token_ids: [seq_len] 1D token id 序列
        
        Returns:
            positions: list of int, delimiter 末尾位置
        """
        if self.step_delimiter_ids is None:
            return []
        
        delim = self.step_delimiter_ids
        delim_len = len(delim)
        positions = []
        seq = token_ids.tolist()
        
        for i in range(delim_len - 1, len(seq)):
            if seq[i - delim_len + 1 : i + 1] == delim:
                positions.append(i)
        
        return positions

    def get_visual_tokens(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, list]:
        """
        从 Qwen3-VL 的视觉编码器获取视觉 token 序列
        
        Qwen3-VL 视觉编码器返回 (hidden_states, deepstack_feature_lists)。
        
        Args:
            pixel_values: [total_patches, channels]
            image_grid_thw: [num_images, 3]
        
        Returns:
            visual_tokens: [B, L_v, hidden_size]
            deepstack_features: list of deepstack visual embeddings
        """
        inner_model = self.base_model.model  # Qwen3VLModel

        # 调用视觉编码器
        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(
            _pixel_values, grid_thw=image_grid_thw
        )
        merged_hidden_states, deepstack_features = _extract_visual_output(vision_output)

        # merged_hidden_states: [total_visual_tokens, out_hidden_size=4096]
        # 按图像分组
        split_sizes = (
            image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
        ).tolist()

        visual_token_list = torch.split(merged_hidden_states, split_sizes)

        if len(visual_token_list) == 1:
            return visual_token_list[0].unsqueeze(0), deepstack_features
        else:
            max_len = max(t.shape[0] for t in visual_token_list)
            B = len(visual_token_list)
            D = visual_token_list[0].shape[-1]
            device = visual_token_list[0].device
            dtype = visual_token_list[0].dtype

            padded = torch.zeros(B, max_len, D, device=device, dtype=dtype)
            for i, t in enumerate(visual_token_list):
                padded[i, :t.shape[0]] = t

            return padded, deepstack_features

    def _build_prefix_mrope_ids(self, mrope_anchor: torch.Tensor, K_p: int) -> torch.Tensor:
        """
        基于 mRoPE anchor 构建 prefix 的 4D position_ids
        
        关键修复: prefix 的 RoPE 位置必须基于 mRoPE 时间轴 (而非 cache token count),
        因为 Qwen3-VL 的 mRoPE 推进速度与 token 数不一致 (图像 token 在 mRoPE 中
        只对应 max(H,W)/spatial_merge_size 个位置，但在 cache 中占了几百个 token)。
        
        prefix 的 mRoPE 位置 = [anchor - K_p, anchor - K_p + 1, ..., anchor - 1]
        这样 prefix 始终紧贴"真实 mRoPE 时间轴"上的当前位置，而不是 cache token 轴。
        
        Args:
            mrope_anchor: [3, B, 1] 或 [B] — 下一个 token 的 mRoPE 起始位置
                          如果是 [B], 三维使用相同值
            K_p: prefix 长度
        
        Returns:
            position_ids: [4, B, K_p] — 第 0 行是 text_position_ids (用于 causal mask),
                          第 1-3 行是 mRoPE position_ids (用于 rotary)
        """
        device = mrope_anchor.device
        
        if mrope_anchor.dim() == 1:
            # [B] → [3, B, 1]
            mrope_anchor = mrope_anchor.unsqueeze(0).unsqueeze(-1).expand(3, -1, 1)
        
        # 建议 4: 强制 text-like mRoPE (t=h=w)，避免多模态后三维不一致导致频段错位
        # 取三维的最大值作为统一 anchor (对文本段三维本就相同，对图像后取 max 最保守)
        if mrope_anchor.shape[0] == 3:
            unified_anchor = mrope_anchor.max(dim=0, keepdim=True).values  # [1, B, 1]
            mrope_anchor = unified_anchor.expand(3, -1, -1)  # [3, B, 1]
        
        B = mrope_anchor.shape[1]
        
        # mRoPE: anchor 向前偏移 K_p
        start = (mrope_anchor - K_p).clamp(min=0)   # [3, B, 1]
        offsets = torch.arange(K_p, device=device).view(1, 1, K_p)  # [1, 1, K_p]
        mrope_pos = start + offsets                  # [3, B, K_p]
        
        # text_position_ids: 简单递增 0..K_p-1 (仅用于 prefix 自身的 causal mask)
        text_pos = torch.arange(K_p, device=device).unsqueeze(0).expand(B, -1)  # [B, K_p]
        
        # 组合为 [4, B, K_p]
        position_ids = torch.cat([text_pos.unsqueeze(0), mrope_pos], dim=0)
        return position_ids

    def forward_with_prefix_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        position_ids: Optional[torch.Tensor] = None,
            global_position_offset: int = 0,  # 已弃用，保留接口兼容性
            skip_prefix: bool = False,
            mrope_position_offset: Optional[torch.Tensor] = None,
            cached_visual_embeds: Optional[torch.Tensor] = None,
            cached_deepstack_features: Optional[list] = None,
            cached_visual_pos_masks: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        带 prefix embedding 注入的前向传播 (方案 A - 梯度连通)
        
        位置编码策略 (三大原则):
        1. 真实 token 的 mRoPE 位置保持 get_rope_index() 原始值，不做偏移
           → 图像/query 的空间编码完全不受 prefix 影响
        2. Prefix 的 mRoPE 位置滑动贴靠真实 token 起始位置之前
           → 无论推理链多长，draft 与最近 token 的距离始终 <= K_p
        3. cache_position 和 text_position_ids (用于 causal mask) 保持全局连续递增
           → 分段 forward 之间位置不冲突
        
        位置分配示意 (以 K_p=32, global_position_offset=100 为例):
          causal mask 用的 text_position_ids: [100, 101, ..., 131, 132, 133, ...]
                                               ← prefix (32) → ← real tokens →
          mRoPE position_ids (3维):
            prefix: 使用原始 token 的起始 mRoPE 位置向前偏移 K_p
            real tokens: 保持 get_rope_index() 原始值不变
          cache_position: [100, 101, ..., 100+K_p+seq_len-1] (全局连续)
        
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            labels: [B, seq_len] (训练时使用)
            prefix_embeds: [B, K_p, hidden_size] 由 controller 生成的 prefix embeddings
            pixel_values, image_grid_thw: 视觉输入
            past_key_values: KV cache
            position_ids: 位置 id
            global_position_offset: 全局位置偏移 (训练分段时使用，确保段间位置连续)
            skip_prefix: 如果为 True，不拼接 prefix_embeds 到 inputs_embeds
                         (用于段 1+ 的训练，prefix KV 已通过可微覆盖写入 cache)
        
        Returns:
            CausalLMOutputWithPast
        """
        if prefix_embeds is None:
            # 没有 prefix: 直接调用基座模型
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'past_key_values': outputs.past_key_values,
                'hidden_states': None,
                'mrope_last_pos': None,
            }

        # ====== 有 prefix embedding 注入 ======
        B = input_ids.shape[0]
        K_p = prefix_embeds.shape[1]  # prefix 长度
        seq_len = input_ids.shape[1]

        # skip_prefix 模式: prefix KV 已通过可微覆盖写入 cache (旧版训练路径),
        # 或用于不拼 prefix 的纯推理场景。
        # 4B Dense 训练路径已不再使用 skip_prefix=True (统一走 skip_prefix=False)
        if skip_prefix:
            extended_attention_mask = attention_mask
        else:
            # 正常模式: 在左侧加 K_p 个 1 (prefix token 始终可见)
            prefix_attn = torch.ones(B, K_p, device=attention_mask.device, dtype=attention_mask.dtype)
            extended_attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)

        inner_model = self.base_model.model  # Qwen3VLModel
        lm_head = self.base_model.lm_head

        # ---- Step 1: Embedding ----
        inputs_embeds = inner_model.get_input_embeddings()(input_ids)

        # ---- Step 2: 视觉融合 (P1-1: 优先使用缓存的视觉特征) ----
        if cached_visual_embeds is not None:
            # 使用缓存的视觉特征 (视觉 encoder 已在 forward() 中跑过一次)
            # 注意: 分段 forward 时 input_ids 是子序列，image placeholder token 数量
            # 可能少于 cached_visual_embeds 中的 feature 数量。
            # 因此不能使用 get_placeholder_mask (它要求严格数量匹配)，
            # 而是手动构建 mask 并只替换当前段中存在的 image placeholder tokens。
            image_token_id = inner_model.config.image_token_id
            image_mask = (input_ids == image_token_id)  # [B, seg_len]
            num_image_tokens_in_seg = image_mask.sum().item()
            if num_image_tokens_in_seg > 0:
                image_embeds = cached_visual_embeds[:num_image_tokens_in_seg].to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                # masked_scatter 要求 mask 扩展到 embed 维度
                image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask_3d, image_embeds)
        elif pixel_values is not None:
            # 回退: 如果没有缓存，运行视觉 encoder (Qwen3-VL 返回 (hidden_states, deepstack_features))
            _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
            vision_output = inner_model.visual(
                _pixel_values, grid_thw=image_grid_thw
            )
            merged_hidden_states, _deepstack_feats = _extract_visual_output(vision_output)
            split_sizes = (
                image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
            ).tolist()
            image_embeds = list(torch.split(merged_hidden_states, split_sizes))
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = inner_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # ---- Step 2.5: 提前初始化 cache 并计算 past_seen ----
        # 关键修复: 必须在 Step 3 构造 position_ids 之前确定 past_seen，
        # 用它覆盖 global_position_offset，确保 text_position_ids 和 cache_position 严格一致。
        text_model = inner_model.language_model  # Qwen3VLTextModel

        if past_key_values is None:
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache()
        else:
            cache = past_key_values

        past_seen = self._get_full_attn_cache_len(cache)
        # 覆盖 global_position_offset 为 past_seen，确保 text_position_ids 与 cache_position 对齐
        global_position_offset = past_seen

        # ---- Step 3: 位置编码 (核心修改: 真实 token 位置不偏移) ----
        #
        # Qwen3-VL 的 position_ids 结构: [3, B, seq_len] (mRoPE: temporal, height, width)
        # 或 [4, B, seq_len] (第 0 维是 text_position_ids 用于 causal mask)
        #
        # 关键设计:
        # (A) mRoPE position_ids (用于旋转位置编码):
        #     - 真实 token: 保持 get_rope_index() 的原始值，不偏移
        #     - prefix: 紧贴真实 token 起始位置之前 (滑动贴靠)
        # (B) text_position_ids (用于 causal mask):
        #     - 全局连续递增，确保因果关系正确
        # (C) cache_position (用于 KV cache 索引):
        #     - 全局连续递增
        #
        # 注意: get_rope_index() 只需要当前段 token 的 attention_mask (不含 cache 历史),
        # 因此构造一个只覆盖当前段的 seg_token_mask。

        # 为 get_rope_index 构造只覆盖当前段的 mask
        seg_token_mask = torch.ones(B, seq_len, device=input_ids.device, dtype=torch.long)

        # 构造 mm_token_type_ids 和 get_rope_index 参数
        # Qwen3-VL 的 get_rope_index 不需要 mm_token_type_ids 参数，
        # 它自动从 input_ids 中检测 image/video token
        has_image = pixel_values is not None and image_grid_thw is not None

        inner_model.rope_deltas = None
        position_ids_computed, rope_deltas = inner_model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw if has_image else None,
            video_grid_thw=None,
            attention_mask=seg_token_mask,
        )
        inner_model.rope_deltas = rope_deltas
        # position_ids_computed: [3, B, seq_len] — 真实 token 的 mRoPE 位置

        if position_ids_computed is not None and position_ids_computed.dim() == 3:
            device = position_ids_computed.device

            if skip_prefix:
                # skip_prefix 模式: 只有 seg_tokens, 不拼 prefix
                # mRoPE 位置必须从全局偏移继续 (不能从 0 开始!)
                # seg1+ 都是纯文本段，三维 mRoPE 相同
                # text_position_ids: 全局连续, 从 global_position_offset 开始
                text_pos = torch.arange(
                    global_position_offset,
                    global_position_offset + seq_len,
                    device=device,
                )  # [seq_len]
                text_pos = text_pos.unsqueeze(0).expand(B, -1)  # [B, seq_len]
                
                # mRoPE: 从 mrope_position_offset 继续递增
                if mrope_position_offset is not None:
                    # mrope_position_offset: [3, B, 1] — 每个维度上一段的最后位置
                    mrope_start = mrope_position_offset + 1  # [3, B, 1]
                    mrope_arange = torch.arange(seq_len, device=device).view(1, 1, seq_len)  # [1, 1, seq_len]
                    mrope_pos = mrope_start + mrope_arange  # [3, B, seq_len]
                else:
                    # 兜底: 用 get_rope_index 的原始值 (不推荐，仅用于无 offset 的退化场景)
                    mrope_pos = position_ids_computed
                
                position_ids_computed = torch.cat(
                    [text_pos.unsqueeze(0), mrope_pos], dim=0
                )  # [4, B, seq_len]
            else:
                # 正常模式: prefix + seg_tokens
                # 段 0: 使用 get_rope_index 的原始 mRoPE 位置
                # 段 1+: 真实 token 的 mRoPE 需要从 mrope_position_offset+1 连续递增
                if mrope_position_offset is not None:
                    # 段 1+: 纯文本段，三维 mRoPE 相同，从 offset+1 连续递增
                    mrope_start = mrope_position_offset + 1  # [3, B, 1]
                    mrope_arange = torch.arange(seq_len, device=device).view(1, 1, seq_len)
                    real_mrope_pos = mrope_start + mrope_arange  # [3, B, seq_len]
                    
                    # prefix mRoPE: 紧贴 real_mrope_pos 的起始位置之前
                    real_start = real_mrope_pos[:, :, 0:1]  # [3, B, 1]
                    prefix_offsets = torch.arange(-K_p, 0, device=device).view(1, 1, K_p)
                    prefix_mrope_pos = (real_start + prefix_offsets).clamp(min=0)  # [3, B, K_p]
                    
                    combined_mrope_pos = torch.cat(
                        [prefix_mrope_pos, real_mrope_pos], dim=2
                    )  # [3, B, K_p + seq_len]
                else:
                    # 段 0: 使用 get_rope_index 的原始值 (可能含图像 mRoPE)
                    # (A) 为 prefix 构造 mRoPE 位置: 紧贴真实 token 的起始位置之前
                    real_start_pos = position_ids_computed[:, :, 0:1]  # [3, B, 1]
                    prefix_offsets = torch.arange(-K_p, 0, device=device)  # [-K_p, ..., -1]
                    prefix_offsets = prefix_offsets.view(1, 1, K_p)  # [1, 1, K_p]
                    prefix_mrope_pos = real_start_pos + prefix_offsets  # [3, B, K_p]
                    prefix_mrope_pos = prefix_mrope_pos.clamp(min=0)

                    # 真实 token 的 mRoPE 位置保持不变 (不加 K_p!)
                    combined_mrope_pos = torch.cat(
                        [prefix_mrope_pos, position_ids_computed], dim=2
                    )  # [3, B, K_p + seq_len]

                # (B) 构造 text_position_ids: 全局连续递增
                text_pos = torch.arange(
                    global_position_offset,
                    global_position_offset + K_p + seq_len,
                    device=device,
                )  # [K_p + seq_len]
                text_pos = text_pos.unsqueeze(0).expand(B, -1)  # [B, K_p + seq_len]

                # 组合为 [4, B, total_len]
                position_ids_computed = torch.cat(
                    [text_pos.unsqueeze(0), combined_mrope_pos], dim=0
                )  # [4, B, K_p + seq_len]

        # ---- Step 4: 在 inputs_embeds 前面拼接 prefix embedding (方案 A 核心) ----
        if not skip_prefix:
            # prefix_embeds 由 controller 生成，带有梯度，直接参与 Transformer forward
            prefix_embeds = prefix_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        # ---- Step 5: 调用 Qwen3VLTextModel.forward ----
        # prefix embedding 作为 inputs_embeds 的一部分，自然参与整个 forward
        # 无需 hook、无需手动展开 decoder 层
        lm_head = self.base_model.lm_head

        # 计算 cache_position (全局连续递增，与 text_position_ids 对齐)
        total_len = inputs_embeds.shape[1]
        cache_position = torch.arange(
            past_seen, past_seen + total_len, device=inputs_embeds.device
        )

        # Qwen3-VL 有 DeepStack，当段包含 visual token 时需要传入
        # 对于纯文本段 (没有 image placeholder)，传 None 即可
        _visual_pos_masks = cached_visual_pos_masks
        _deepstack_visual_embeds = cached_deepstack_features
        
        # 如果有 prefix 拼接，需要在 visual_pos_masks 前面补 K_p 个 False
        if _visual_pos_masks is not None and not skip_prefix and K_p > 0:
            prefix_mask = torch.zeros(B, K_p, device=device, dtype=torch.bool)
            _visual_pos_masks = torch.cat([prefix_mask, _visual_pos_masks], dim=1)
        
        text_outputs = text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            position_ids=position_ids_computed,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=_visual_pos_masks,
            deepstack_visual_embeds=_deepstack_visual_embeds,
        )

        hidden_states = text_outputs.last_hidden_state

        # 提取 mRoPE 最后位置 (用于段间传递，确保 mRoPE 连续)
        # position_ids_computed: [4, B, total_len]，后 3 行是 mRoPE
        mrope_last_pos = None
        if position_ids_computed is not None and position_ids_computed.dim() == 3 and position_ids_computed.shape[0] == 4:
            mrope_last_pos = position_ids_computed[1:, :, -1:]  # [3, B, 1]

        # 提取 last_hidden_state (去掉 prefix 部分)，供 step_resampler 使用
        if not skip_prefix:
            real_hidden_states = hidden_states[:, K_p:, :]
        else:
            real_hidden_states = hidden_states

        # LM Head
        logits = lm_head(real_hidden_states)

        # 建议 3: 不在段内计算 loss，只返回 logits
        # loss 由外层 forward() 用全局 shift 一次性计算，避免跨段边界监督信号丢失

        return {
            'loss': None,
            'logits': logits,
            'past_key_values': text_outputs.past_key_values,
            'hidden_states': real_hidden_states,  # [B, seg_len, hidden_size] 不含 prefix
            'mrope_last_pos': mrope_last_pos,     # [3, B, 1] mRoPE 最后位置
        }

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, seq_len] 完整序列 (含 </step> delimiter)
        attention_mask: torch.Tensor,       # [B, seq_len]
        labels: Optional[torch.Tensor] = None,   # [B, seq_len]
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        step_boundaries: Optional[List[List[int]]] = None,  # 每个样本的 step 边界位置
        prompt_lens: Optional[List[int]] = None,  # 每个样本的 prompt 长度 (padding 后)
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播 (按 step 段展开，训推一致的可微 KV 覆盖)
        
        核心设计: prompt (含图像) 独立 prefill → 分段只覆盖生成部分 (think + answer)
        
        新流程:
          prefill: [prompt(含 image)]  ← 不算段，只做 KV 缓存
          段 0: [prefix | think_step1 + </step>]
          段 1: [prefix | think_step2 + </step>]
          ...
          段 N: [prefix | think_lastN + answer]
        
        好处:
        - 所有分段都是纯文本段，不含 image token
        - 消除 </step> token 碎片意外落入 prompt 区域的风险
        - 与推理流程完全一致 (推理也是先 prefill prompt 再逐步生成)
        - 视觉信息只在 prefill 阶段处理一次
        
        KV cache 结构:
          prefill 后: cache = [prompt_kv]
          段 0 后:    cache = [prompt_kv | prefix_0_kv | seg_0_kv]
          段 1 前:    剥离前 K_p 位置 → cache = [prompt_kv | seg_0_kv]
          段 1 后:    cache = [prompt_kv | prefix_1_kv | seg_1_kv]
          ...
        
        Args:
            input_ids: [B, seq_len] 完整输入 (包含 </step> 分隔符)
            attention_mask: [B, seq_len]
            labels: [B, seq_len] 训练标签
            pixel_values: 视觉输入
            image_grid_thw: 图像网格信息
            step_boundaries: 每个样本中 </step> 的位置列表
                             如果为 None，自动检测
            prompt_lens: 每个样本的 prompt 长度列表 (padding 后的绝对位置)
                         如果为 None，退化为旧逻辑 (段0含prompt)
        
        Returns:
            dict with 'loss', 'logits', 'div_loss'
        """
        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device
        ctrl_dtype = next(self.controller.parameters()).dtype

        # ====== 1. 获取视觉 token 并初始化 RLD 状态 ======
        # P1-1: 视觉 encoder 只跑一次，缓存特征供后续复用
        cached_visual_embeds = None     # 缓存的视觉 embedding (用于 placeholder 替换)
        deepstack_features = None       # DeepStack 视觉特征 (用于 decoder 前几层注入)

        if pixel_values is not None:
            inner_model = self.base_model.model
            _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
            vision_output = inner_model.visual(
                _pixel_values, grid_thw=image_grid_thw
            )
            merged_hidden_states, deepstack_features = _extract_visual_output(vision_output)
            split_sizes = (
                image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
            ).tolist()
            image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))

            # 构造 visual_tokens (给 controller prefill 用)
            if len(image_embeds_list) == 1:
                visual_tokens = image_embeds_list[0].unsqueeze(0)
            else:
                max_len = max(t.shape[0] for t in image_embeds_list)
                B_imgs = len(image_embeds_list)
                D = image_embeds_list[0].shape[-1]
                padded = torch.zeros(B_imgs, max_len, D, device=device, dtype=image_embeds_list[0].dtype)
                for i, t in enumerate(image_embeds_list):
                    padded[i, :t.shape[0]] = t
                visual_tokens = padded

            # 缓存视觉特征 (给 forward_with_prefix_embeds 用)
            cached_visual_embeds = torch.cat(image_embeds_list, dim=0)
        else:
            visual_tokens = torch.zeros(
                B, 1, self.hidden_size, device=device, dtype=ctrl_dtype
            )

        rld_state = self.controller.prefill(visual_tokens)

        # ====== 2. Prompt 独立 prefill ======
        # prompt 包含系统提示 + 图像 + 用户问题 + generation prompt
        # 将 prompt 独立 prefill，后续分段只覆盖生成部分 (think + answer)
        # 这样所有分段都是纯文本，不含 image token
        
        # 确定 prompt 边界: 取 batch 中最大的 prompt_len
        # (left padding 后，所有样本的 prompt 部分右对齐)
        _prompt_hidden = None
        if prompt_lens is not None:
            prompt_end = max(prompt_lens)  # batch 中最大的 prompt 结束位置
        else:
            # 退化: 无 prompt_lens 时用 0，等价于旧行为（不 prefill prompt）
            prompt_end = 0
        
        if prompt_end > 0:
            # Prefill prompt 部分 (含图像，不带 prefix embedding，无 loss)
            prompt_ids = input_ids[:, :prompt_end]
            prompt_mask = attention_mask[:, :prompt_end]
            
            # 构造 visual_pos_masks (标记 prompt 中哪些位置是 image token)
            # DeepStack 需要此 mask 来在 decoder 前几层注入视觉特征
            prompt_visual_pos_masks = None
            prompt_deepstack_features = None
            if pixel_values is not None and image_grid_thw is not None:
                inner_model = self.base_model.model
                image_token_id = inner_model.config.image_token_id
                prompt_visual_pos_masks = (prompt_ids == image_token_id)  # [B, prompt_end]
                prompt_deepstack_features = deepstack_features
            
            prefill_result = self.forward_with_prefix_embeds(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                labels=None,  # prompt 无 loss
                prefix_embeds=None,  # prompt prefill 不带 prefix
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                global_position_offset=0,
                past_key_values=None,
                cached_visual_embeds=cached_visual_embeds,
                cached_deepstack_features=prompt_deepstack_features,
                cached_visual_pos_masks=prompt_visual_pos_masks,
            )
            
            past_key_values = prefill_result['past_key_values']
            mrope_position_offset = prefill_result['mrope_last_pos']  # [3, B, 1]
            
            # KV cache detach: prompt 部分不需要梯度回传
            # Qwen3-VL 使用标准 DynamicCache，只需 detach key/value cache
            if past_key_values is not None:
                for _li in range(len(past_key_values.key_cache)):
                    if past_key_values.key_cache[_li] is not None:
                        past_key_values.key_cache[_li] = past_key_values.key_cache[_li].detach()
                    if past_key_values.value_cache[_li] is not None:
                        past_key_values.value_cache[_li] = past_key_values.value_cache[_li].detach()
            
            global_position_offset = prompt_end
            
            # 保存 prompt hidden states，待 per_sample_hidden_buffers 初始化后写入
            _prompt_hidden = prefill_result['hidden_states']  # [B, prompt_end, H]
            del prefill_result
        
        # 生成部分的起止范围
        gen_start = prompt_end
        gen_len = seq_len - gen_start

        # ====== 3. 自动检测 step 边界 (per-sample, 只在生成部分搜索) ======
        if step_boundaries is None and self.step_delimiter_ids is not None:
            step_boundaries = []
            for b in range(B):
                positions = self._find_delimiter_positions(input_ids[b])
                # 过滤: 只保留 prompt 之后的 delimiter
                positions = [p for p in positions if p >= gen_start]
                step_boundaries.append(positions)

        # ====== 3.5 统一 step boundaries (batch 级) + per-sample update mask ======
        all_boundaries = set()
        if step_boundaries:
            for b_list in step_boundaries:
                all_boundaries.update(b_list)
        # 只保留生成部分内的 split points
        split_points = sorted([p for p in all_boundaries if gen_start < p < seq_len])

        per_sample_boundaries = step_boundaries if step_boundaries else [[] for _ in range(B)]
        per_sample_sets = [set(bl) for bl in per_sample_boundaries]

        # ====== 4. 构建段: 只覆盖生成部分 [gen_start, ...] ======
        segments = []  # list of (start, end) 左闭右开
        prev = gen_start
        for sp in split_points:
            segments.append((prev, sp + 1))
            prev = sp + 1
        if prev < seq_len:
            segments.append((prev, seq_len))

        # ====== 4.5 段数限制 (简化版, ZeRO-2 无需段数对齐) ======
        # ZeRO-2 不需要 all-gather 参数，无需 dummy 段对齐
        # 但仍需限制最大段数避免极端样本导致 NCCL 超时
        MAX_SEGMENTS = 8
        if len(segments) > MAX_SEGMENTS:
            last_start = segments[MAX_SEGMENTS - 1][0]
            last_end = segments[-1][1]
            segments = segments[:MAX_SEGMENTS - 1] + [(last_start, last_end)]
            # split_points 也需要对应截断 (保留前 MAX_SEGMENTS-1 个)
            split_points = split_points[:MAX_SEGMENTS - 1]

        # ====== 5. 分段 forward + RLD 更新 (4B Dense 优化版) ======
        #
        # 新流程 (prompt 独立 prefill 后):
        #   - 所有段都是纯文本段 (think + answer)，不含 image token
        #   - 每段统一走 skip_prefix=False 的单次 forward [prefix | seg_tokens]
        #   - prefix 在 prompt_kv 的后面追加到 cache
        #
        # KV cache 策略:
        #   prompt prefill 后: cache = [prompt_kv]
        #   段 0 后:  cache = [prompt_kv | prefix_0_kv | seg_0_kv]
        #   段 1 前:  剥离 prefix_0_kv → cache = [prompt_kv | seg_0_kv]
        #   段 1 后:  cache = [prompt_kv | seg_0_kv | prefix_1_kv | seg_1_kv]
        #   ...
        #   注意: prompt_kv 始终保留在 cache 最前面，只剥离 prefix 部分
        #
        # 梯度路径:
        #   路径 1 (rld_state 链): loss → prefix_embeds → controller → rld_state
        #   路径 2 (hidden 直接): loss → prefix_embeds → step_update ← hidden_states
        #
        div_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        seg_loss_sum = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        seg_token_count = 0
        num_updates = 0
        K_p = self.K_p
        # 如果有 prompt prefill，这些变量已在上面初始化
        if prompt_end == 0:
            global_position_offset = 0
            past_key_values = None
            mrope_position_offset = None

        # per-sample hidden buffer (不 detach, 保留梯度供 step_update 使用)
        per_sample_hidden_buffers = [[] for _ in range(B)]

        # 将 prompt hidden states 加入 buffer (与推理时保持一致)
        if prompt_end > 0 and _prompt_hidden is not None:
            for b in range(B):
                per_sample_hidden_buffers[b].append(_prompt_hidden[b:b+1])
            del _prompt_hidden

        # prompt_kv_len: prompt 部分占用的 cache 长度 (剥离 prefix 时需要跳过)
        prompt_kv_len = prompt_end  # prompt 的 KV 在 cache 最前面

        # 调试日志 (仅在 RLD_DEBUG=1 时启用)
        _rld_debug = os.environ.get('RLD_DEBUG', '0') == '1'
        _rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        if _rld_debug and _rank == 0:
            _cache_len = self._get_full_attn_cache_len(past_key_values) if past_key_values is not None else 0
            print(f"\n[RLD DEBUG] ========== 段循环开始 ==========")
            print(f"  B={B}, seq_len={seq_len}, prompt_end={prompt_end}, gen_len={gen_len}")
            print(f"  总段数: {len(segments)}, segments: {segments}")
            print(f"  split_points: {split_points}")
            print(f"  prompt_kv_len={prompt_kv_len}, K_p={K_p}")
            print(f"  初始 cache 长度 (full_attn): {_cache_len}")

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_len = seg_end - seg_start

            # 当前段的 input_ids, labels (纯文本，不含 image token)
            seg_ids = input_ids[:, seg_start:seg_end]
            seg_labels = labels[:, seg_start:seg_end] if labels is not None else None

            # 生成当前 prefix embedding (方案 A: 梯度连通)
            prefix_embeds = self.controller.get_prefix_embeds(rld_state)

            # 调试日志: 每段 forward 前的状态
            if _rld_debug and _rank == 0:
                _cache_len = self._get_full_attn_cache_len(past_key_values) if past_key_values is not None else 0
                print(f"\n  [段 {seg_idx}] range=[{seg_start}, {seg_end}), seg_len={seg_len}")
                print(f"    cache 长度 (full_attn, forward前): {_cache_len}")
                print(f"    prefix_embeds shape: {prefix_embeds.shape}")

            if seg_idx == 0 and prompt_end > 0:
                # ---- 段 0 (prompt 已 prefill): forward [prefix | seg_tokens] 接在 prompt_kv 后面 ----
                # cache 中已有 prompt_kv，构造 attention_mask 覆盖 [prompt_kv + seg_tokens]
                # forward_with_prefix_embeds 会在左侧加 K_p 个 1
                # 使用 _get_full_attn_cache_len 安全获取 KV cache 长度
                actual_cache_len = self._get_full_attn_cache_len(past_key_values)
                current_total_attn_len = actual_cache_len + seg_len
                seg_mask = torch.ones(B, current_total_attn_len, device=device, dtype=attention_mask.dtype)

                if _rld_debug and _rank == 0:
                    print(f"    [段0+prompt] actual_cache_len={actual_cache_len}, seg_mask len={current_total_attn_len}")
                    print(f"    extended_mask len 将为: {K_p + current_total_attn_len}")
                    print(f"    cache_position 将从 {actual_cache_len} 到 {actual_cache_len + K_p + seg_len - 1}")
                    print(f"    text_pos 将从 {actual_cache_len} 到 {actual_cache_len + K_p + seg_len - 1}")

                seg_result = self.forward_with_prefix_embeds(
                    input_ids=seg_ids,
                    attention_mask=seg_mask,
                    labels=seg_labels,
                    prefix_embeds=prefix_embeds,
                    pixel_values=None,      # 图像已在 prefill 处理
                    image_grid_thw=None,
                    global_position_offset=global_position_offset,
                    past_key_values=past_key_values,
                    skip_prefix=False,
                    mrope_position_offset=mrope_position_offset,
                    cached_visual_embeds=None,
                )

            elif seg_idx == 0 and prompt_end == 0:
                # ---- 退化: 无 prompt_lens，段 0 包含 prompt (旧行为) ----
                seg_mask = attention_mask[:, seg_start:seg_end]

                seg_result = self.forward_with_prefix_embeds(
                    input_ids=seg_ids,
                    attention_mask=seg_mask,
                    labels=seg_labels,
                    prefix_embeds=prefix_embeds,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    global_position_offset=0,
                    past_key_values=None,
                    skip_prefix=False,
                    cached_visual_embeds=cached_visual_embeds,
                )

            else:
                # ---- 段 1+: 剥离旧 prefix → 一次 forward [new_prefix | seg_tokens] ----
                # Step 1: 从 cache 中剥离 prefix KV (位于 prompt_kv 之后)
                # cache 结构: [prompt_kv(prompt_kv_len) | prefix_kv(K_p) | history_seg_kv(...)]
                # 需要剥离 prefix_kv，保留 prompt_kv + history_seg_kv
                _cache_before_strip = self._get_full_attn_cache_len(past_key_values) if past_key_values is not None else 0
                if past_key_values is not None:
                    for _li in range(len(past_key_values.key_cache)):
                        # Qwen3-VL: 所有层都是标准 full_attention，直接剥离
                        kc = past_key_values.key_cache[_li]
                        if kc is not None and kc.dim() == 4 and kc.shape[2] > prompt_kv_len + K_p:
                            past_key_values.key_cache[_li] = torch.cat([
                                kc[:, :, :prompt_kv_len, :],
                                kc[:, :, prompt_kv_len + K_p:, :],
                            ], dim=2)
                            vc = past_key_values.value_cache[_li]
                            past_key_values.value_cache[_li] = torch.cat([
                                vc[:, :, :prompt_kv_len, :],
                                vc[:, :, prompt_kv_len + K_p:, :],
                            ], dim=2)
                # Step 2: 构造 attention_mask
                # 使用 full_attention 层的实际 KV cache 长度 (剥离后)
                actual_cache_len = self._get_full_attn_cache_len(past_key_values)
                current_total_attn_len = actual_cache_len + seg_len
                seg_mask = torch.ones(B, current_total_attn_len, device=device, dtype=attention_mask.dtype)

                if _rld_debug and _rank == 0:
                    print(f"    [段{seg_idx} 剥离] cache 剥离前={_cache_before_strip}, 剥离后={actual_cache_len}, 差={_cache_before_strip - actual_cache_len} (应为 K_p={K_p})")
                    print(f"    seg_mask len={current_total_attn_len}, extended_mask len 将为: {K_p + current_total_attn_len}")
                    print(f"    cache_position 将从 {actual_cache_len} 到 {actual_cache_len + K_p + seg_len - 1}")
                    print(f"    text_pos 将从 {actual_cache_len} 到 {actual_cache_len + K_p + seg_len - 1}")

                # Step 3: 一次 forward [prefix | seg_tokens]
                seg_result = self.forward_with_prefix_embeds(
                    input_ids=seg_ids,
                    attention_mask=seg_mask,
                    labels=seg_labels,
                    prefix_embeds=prefix_embeds,
                    pixel_values=None,
                    image_grid_thw=None,
                    global_position_offset=global_position_offset,
                    past_key_values=past_key_values,
                    skip_prefix=False,
                    mrope_position_offset=mrope_position_offset,
                    cached_visual_embeds=None,
                )

            seg_hidden = seg_result['hidden_states']
            mrope_position_offset = seg_result['mrope_last_pos']

            past_key_values = seg_result['past_key_values']

            # 调试日志: 每段 forward 后的状态
            if _rld_debug and _rank == 0:
                _cache_after = self._get_full_attn_cache_len(past_key_values) if past_key_values is not None else 0
                _hidden_shape = seg_hidden.shape if seg_hidden is not None else None
                _has_nan = torch.isnan(seg_hidden).any().item() if seg_hidden is not None else False
                print(f"    [段{seg_idx} forward后] cache 长度={_cache_after}, hidden shape={_hidden_shape}, has_nan={_has_nan}")
                if seg_result.get('logits') is not None:
                    _logits = seg_result['logits']
                    _logits_nan = torch.isnan(_logits).any().item()
                    _logits_inf = torch.isinf(_logits).any().item()
                    print(f"    logits shape={_logits.shape}, has_nan={_logits_nan}, has_inf={_logits_inf}")

            # KV cache detach: 阻止跨段 attention 梯度，控制内存
            # Qwen3-VL 使用标准 DynamicCache，只需 detach key/value cache
            if past_key_values is not None:
                for _li in range(len(past_key_values.key_cache)):
                    if past_key_values.key_cache[_li] is not None:
                        past_key_values.key_cache[_li] = past_key_values.key_cache[_li].detach()
                    if past_key_values.value_cache[_li] is not None:
                        past_key_values.value_cache[_li] = past_key_values.value_cache[_li].detach()
            # 关键修复: 不再手动累积 global_position_offset
            # forward_with_prefix_embeds 内部用 past_seen (cache 实际长度) 覆盖了它，
            # 这样 text_position_ids 和 cache_position 始终一致。
            # 旧代码 `global_position_offset += K_p + seg_len` 是错误的:
            # 段1+ 剥离了旧 prefix (K_p) 后 cache 净增只有 seg_len，
            # 但这里多加了 K_p，导致 text_position_ids 每段多偏移 K_p。

            # 不 detach hidden_states, controller 获得双重梯度路径
            for b in range(B):
                per_sample_hidden_buffers[b].append(seg_hidden[b:b+1])  # [1, seg_len, H] 保留梯度

            # 逐段计算 loss，立即释放 logits
            if labels is not None:
                seg_logits = seg_result['logits']  # [B, seg_len, vocab_size]
                seg_lbl = labels[:, seg_start:seg_end]
                if seg_logits.shape[1] > 1 and seg_lbl.shape[1] > 1:
                    s_logits = seg_logits[:, :-1, :].contiguous()
                    s_labels = seg_lbl[:, 1:].contiguous()
                    valid_mask = (s_labels != -100)
                    num_valid = valid_mask.sum().item()
                    if num_valid > 0:
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
                        seg_ce = loss_fct(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1))
                        seg_loss_sum = seg_loss_sum + seg_ce
                        seg_token_count += num_valid
                del seg_logits

            seg_result = None

            # per-sample update mask — 只有真正遇到 delimiter 的样本触发更新
            is_delimiter_segment = seg_idx < len(segments) - 1
            if is_delimiter_segment:
                sp = split_points[seg_idx]  # 当前 split point
                update_mask = torch.tensor(
                    [sp in per_sample_sets[b] for b in range(B)],
                    dtype=torch.bool, device=device,
                )  # [B]

                # 建议 2: 用 per-sample hidden buffer 构造 step_hiddens
                # 拼接每个样本从上一次触发以来累积的所有 hidden states
                # 注意: 必须遍历所有样本计算 max_buf_len，否则 update_mask=False 的样本
                # buffer 长度可能与 max_buf_len 不一致，导致 torch.cat 维度不匹配
                max_buf_len = 0
                for b in range(B):
                    if len(per_sample_hidden_buffers[b]) > 0:
                        buf_len = sum(h.shape[1] for h in per_sample_hidden_buffers[b])
                        max_buf_len = max(max_buf_len, buf_len)
                max_buf_len = max(max_buf_len, 1)  # 至少为 1 防止空 tensor

                padded_hiddens = []
                for b in range(B):
                    if len(per_sample_hidden_buffers[b]) > 0:
                        h = torch.cat(per_sample_hidden_buffers[b], dim=1)  # [1, L_b, H]
                    else:
                        h = torch.zeros(1, 1, self.hidden_size, device=device, dtype=ctrl_dtype)
                    # pad 到 max_buf_len
                    if h.shape[1] < max_buf_len:
                        pad = torch.zeros(1, max_buf_len - h.shape[1], self.hidden_size,
                                          device=device, dtype=h.dtype)
                        h = torch.cat([h, pad], dim=1)
                    padded_hiddens.append(h)
                step_hiddens = torch.cat(padded_hiddens, dim=0)  # [B, max_buf_len, H]

                rld_state = self.controller.step_update(rld_state, step_hiddens, update_mask=update_mask)
                div_loss = div_loss + self.controller.compute_diversity_loss(rld_state)
                num_updates += 1

                # 清空已触发样本的 buffer
                for b in range(B):
                    if update_mask[b]:
                        per_sample_hidden_buffers[b] = []

        # OOM 优化: 使用逐段累积的 loss (避免保留全序列 logits)
        total_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        if labels is not None and seg_token_count > 0:
            total_loss = seg_loss_sum / seg_token_count  # 平均 loss

        if num_updates > 0:
            div_loss = div_loss / num_updates

        main_loss = total_loss.detach()
        total_loss = total_loss + div_loss

        # 调试日志: 段循环总结
        if _rld_debug and _rank == 0:
            _is_nan = torch.isnan(total_loss).item()
            _is_inf = torch.isinf(total_loss).item()
            print(f"\n  [段循环完成] total_loss={total_loss.item():.6f}, main_loss={main_loss.item():.6f}, "
                  f"div_loss={div_loss.item():.6f}")
            print(f"  seg_token_count={seg_token_count}, num_updates={num_updates}")
            print(f"  loss has_nan={_is_nan}, has_inf={_is_inf}")
            if _is_nan or _is_inf:
                print(f"  ❌ 警告: loss 为 NaN/Inf! backward 可能导致某些 rank hang 住!")
            print(f"  ========== 段循环结束 ==========\n")

        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'div_loss': div_loss.detach(),
            'logits': None,  # OOM 优化: 不再返回完整 logits，已逐段计算 loss
            'rld_state': rld_state,
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        推理时的生成 (带 step 级 RLD 更新)
        
        修复版本:
        - P0-2: decode 走 text_model + 显式 position_ids，确保 RoPE 训推一致
        - P0-3: per-sample delimiter 检测和 update_mask
        - P1-1: 视觉 encoder 只跑一次
        - P1-2: 多 token 后缀匹配检测 delimiter
        - P1-3: 不再使用 hook，直接获取 hidden_states
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. P1-1: 视觉 encoder 只跑一次，构造 visual_tokens 和缓存特征
        inner_model = self.base_model.model
        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(
            _pixel_values, grid_thw=image_grid_thw
        )
        merged_hidden_states = _extract_visual_output(vision_output)
        split_sizes = (
            image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
        ).tolist()
        image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))

        # visual_tokens: 给 controller prefill 用
        if len(image_embeds_list) == 1:
            visual_tokens = image_embeds_list[0].unsqueeze(0)
        else:
            max_len = max(t.shape[0] for t in image_embeds_list)
            B_imgs = len(image_embeds_list)
            D = image_embeds_list[0].shape[-1]
            padded = torch.zeros(B_imgs, max_len, D, device=device, dtype=image_embeds_list[0].dtype)
            for i, t in enumerate(image_embeds_list):
                padded[i, :t.shape[0]] = t
            visual_tokens = padded

        cached_visual_embeds = torch.cat(image_embeds_list, dim=0)

        # 2. 初始化 RLD 状态
        rld_state = self.controller.prefill(visual_tokens)

        # 3. 生成初始 prefix embedding
        prefix_embeds = self.controller.get_prefix_embeds(rld_state)

        # 4. Prefill: 用 forward_with_prefix_embeds 处理 prompt
        K_p = self.K_p
        prefill_result = self.forward_with_prefix_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=prefix_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cached_visual_embeds=cached_visual_embeds,
        )

        past_key_values = prefill_result['past_key_values']
        next_token_logits = prefill_result['logits'][:, -1, :]
        # P0-2: 保存 mRoPE 最后位置，用于 decode 时连续递增
        mrope_last_pos = prefill_result['mrope_last_pos']  # [3, B, 1]

        # 5. 自回归生成
        generated_ids = input_ids.clone()
        # P1-3: per-sample hidden state buffer
        step_hidden_buffers = [[] for _ in range(B)]

        # 训推一致修复: 将 prefill 阶段的 hidden states 加入 step buffer
        # 训练时段 0 的 buffer 包含 prompt 的全部 real hidden states,
        # 推理时也必须将 prefill 的 hidden states 累积进来,
        # 否则第一次 step_update 的输入分布与训练时不一致。
        prefill_hidden = prefill_result['hidden_states']  # [B, prompt_len, H] 或 None
        if prefill_hidden is not None:
            for b in range(B):
                step_hidden_buffers[b].append(prefill_hidden[b:b+1])  # [1, prompt_len, H]

        eos_token_id = self.processor.tokenizer.eos_token_id if hasattr(self, 'processor') else None

        # P1-2: per-sample 最近生成的 token 滑窗（用于多 token 后缀匹配）
        delim_ids = self.step_delimiter_ids  # 可能是多 token
        delim_len = len(delim_ids) if delim_ids is not None else 0
        recent_tokens = [[] for _ in range(B)]  # 每个样本维护最近 delim_len 个 token

        text_model = inner_model.language_model
        lm_head = self.base_model.lm_head
        prefill_total_len = K_p + input_ids.shape[1]
        current_pos = prefill_total_len  # 当前位置

        for gen_step in range(max_new_tokens):
            # 采样下一个 token
            if do_sample and temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            next_token = next_token.squeeze(-1)  # [B]
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

            # 检查 EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # P0-2 修法 A: decode 走 text_model + 显式 position_ids
            # 构造新 token 的 mRoPE position_ids: 从 mrope_last_pos + 1 开始
            new_mrope_pos = mrope_last_pos + 1  # [3, B, 1]
            # text_position_ids: 用 cache_position 对齐
            text_pos = torch.full((B, 1), current_pos, device=device, dtype=torch.long)
            position_ids_4d = torch.cat([text_pos.unsqueeze(0), new_mrope_pos], dim=0)  # [4, B, 1]

            # 构造 inputs_embeds
            token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))  # [B, 1, hidden_size]

            # attention mask
            current_total_len = current_pos + 1
            new_attention_mask = torch.ones(B, current_total_len, device=device, dtype=attention_mask.dtype)

            # cache_position
            cache_position = torch.tensor([current_pos], device=device, dtype=torch.long)

            # 调用 text_model 做 decode（不再通过 base_model 避免 rope_deltas 问题）
            text_outputs = text_model(
                inputs_embeds=token_embeds,
                attention_mask=new_attention_mask,
                position_ids=position_ids_4d,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
            )

            past_key_values = text_outputs.past_key_values
            hidden_states = text_outputs.last_hidden_state  # [B, 1, hidden_size]
            next_token_logits = lm_head(hidden_states[:, -1, :])  # [B, vocab_size]

            # 更新 mRoPE 位置
            mrope_last_pos = new_mrope_pos

            # P1-3: 收集 hidden states (per-sample)
            for b in range(B):
                step_hidden_buffers[b].append(hidden_states[b:b+1])  # [1, 1, H]

            # P1-2: 更新每个样本的最近 token 滑窗
            for b in range(B):
                recent_tokens[b].append(next_token[b].item())
                if len(recent_tokens[b]) > delim_len:
                    recent_tokens[b] = recent_tokens[b][-delim_len:]

            # P0-3 + P1-2: per-sample delimiter 检测 (多 token 后缀匹配)
            if delim_ids is not None and delim_len > 0:
                trigger_mask = torch.zeros(B, dtype=torch.bool, device=device)
                for b in range(B):
                    if len(recent_tokens[b]) >= delim_len and recent_tokens[b][-delim_len:] == delim_ids:
                        trigger_mask[b] = True

                if trigger_mask.any():
                    # 为触发的样本执行 step update
                    # 拼接所有样本的 step hidden buffer (使用所有样本的 buffer 长度的最大值做 pad)
                    max_buf_len = max(len(step_hidden_buffers[b]) for b in range(B))
                    if max_buf_len > 0:
                        padded_hiddens = []
                        for b in range(B):
                            if len(step_hidden_buffers[b]) > 0:
                                h = torch.cat(step_hidden_buffers[b], dim=1)  # [1, L_b, H]
                            else:
                                h = torch.zeros(1, 1, self.hidden_size, device=device, dtype=hidden_states.dtype)
                            # pad 到 max_buf_len
                            if h.shape[1] < max_buf_len:
                                pad = torch.zeros(1, max_buf_len - h.shape[1], self.hidden_size,
                                                  device=device, dtype=hidden_states.dtype)
                                h = torch.cat([h, pad], dim=1)
                            padded_hiddens.append(h)
                        step_hiddens = torch.cat(padded_hiddens, dim=0)  # [B, max_buf_len, H]

                        rld_state = self.controller.step_update(
                            rld_state, step_hiddens, update_mask=trigger_mask
                        )

                        # 生成更新后的 prefix embedding
                        new_prefix_embeds = self.controller.get_prefix_embeds(rld_state)
                        new_prefix_embeds = new_prefix_embeds.to(
                            device=device, dtype=past_key_values.key_cache[0].dtype,
                        )

                        # 覆盖 KV cache 中 prefix 位置
                        # 修复: 使用 mRoPE anchor 而非 token count
                        mrope_anchor_gen = mrope_last_pos + 1  # [3, B, 1]
                        self._update_prefix_kv_cache(
                            prefix_embeds=new_prefix_embeds,
                            past_key_values=past_key_values,
                            K_p=K_p,
                            mrope_anchor=mrope_anchor_gen,
                        )

                    # 只清空触发了的样本的 buffer
                    for b in range(B):
                        if trigger_mask[b]:
                            step_hidden_buffers[b] = []
                            recent_tokens[b] = []

            current_pos += 1

        return generated_ids

    def _differentiable_update_prefix_kv(
        self,
        prefix_embeds: torch.Tensor,   # [B, K_p, hidden_size]
        past_key_values,                # DynamicCache
        K_p: int,
        mrope_anchor: torch.Tensor,     # [3, B, 1] 或 [B] — 下一 token 的 mRoPE 位置
    ):
        """
        [DEPRECATED] 训练时: 用 text_model.forward 可微地覆盖 KV cache 中前 K_p 个位置的 prefix KV
        
        4B Dense 优化后已不再使用此方法:
        训练时段1+ 改为剥离旧 prefix → 一次 forward [new_prefix | seg_tokens]，
        不再需要单独的 prefix KV 覆盖步骤。保留此方法仅供参考。
        
        关键修复 (相比旧版):
        1. prefix RoPE 位置基于 mRoPE anchor (而非 token count)，确保多模态场景正确
        2. 使用 text_model.forward 而非手写逐层 forward，保证与 HF 实现完全一致
        
        可微性:
        - prefix_embeds → text_model.forward → 得到的 KV cache 有梯度
        - torch.cat([new_prefix_kv, old_seg_kv]) 可微，保持梯度图
        - loss → attend 到 new_prefix_kv → prefix_embeds → controller ✅
        
        Args:
            prefix_embeds: [B, K_p, hidden_size] 由 controller 生成的新 prefix embeddings
            past_key_values: DynamicCache，包含历史 KV
            K_p: prefix 长度
            mrope_anchor: 下一个 token 的 mRoPE 位置 (用于计算 prefix 的滑动 RoPE)
        
        Returns:
            new_past_key_values: 替换后的 DynamicCache (新对象，避免 inplace 修改)
        """
        text_model = self.base_model.model.language_model
        B_size = prefix_embeds.shape[0]
        device = prefix_embeds.device

        # 1. 构建 prefix 的 position_ids (基于 mRoPE anchor, 而非 token count)
        position_ids = self._build_prefix_mrope_ids(mrope_anchor, K_p)  # [4, B, K_p]
        position_ids = position_ids.to(device)

        # 2. 用 text_model.forward 生成 prefix 的 KV cache
        from transformers.cache_utils import DynamicCache
        prefix_cache = DynamicCache()
        cache_position = torch.arange(K_p, device=device)
        prefix_attn_mask = torch.ones(B_size, K_p, device=device, dtype=torch.long)

        # 直接调 text_model, 让 HF 内部处理所有 attention/RoPE/FFN 细节
        _prefix_outputs = text_model(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_attn_mask,
            position_ids=position_ids,
            past_key_values=prefix_cache,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        )
        new_prefix_cache = _prefix_outputs.past_key_values

        # 3. 可微替换: 把旧 cache 的前 K_p 个 KV 换成新计算的
        # Qwen3-VL: 所有层都是标准 full_attention
        for layer_idx in range(len(text_model.layers)):
            new_key = new_prefix_cache.key_cache[layer_idx]     # [B, H, K_p, D]
            new_value = new_prefix_cache.value_cache[layer_idx]  # [B, H, K_p, D]
            
            old_seg_key = past_key_values.key_cache[layer_idx][:, :, K_p:, :]
            old_seg_value = past_key_values.value_cache[layer_idx][:, :, K_p:, :]
            
            past_key_values.key_cache[layer_idx] = torch.cat([new_key, old_seg_key], dim=2)
            past_key_values.value_cache[layer_idx] = torch.cat([new_value, old_seg_value], dim=2)

        return past_key_values

    def _update_prefix_kv_cache(
        self,
        prefix_embeds: torch.Tensor,   # [B, K_p, hidden_size]
        past_key_values,                # DynamicCache
        K_p: int,
        mrope_anchor: torch.Tensor,     # [3, B, 1] 或 [B] — 下一 token 的 mRoPE 位置
    ):
        """
        推理时: 用 text_model.forward 重新计算 prefix 的 KV 并 inplace 覆盖 cache
        
        关键修复 (相比旧版):
        1. prefix RoPE 位置基于 mRoPE anchor (而非 token count)
        2. 使用 text_model.forward 而非手写逐层 forward
        """
        text_model = self.base_model.model.language_model
        B_size = prefix_embeds.shape[0]
        device = prefix_embeds.device

        # 1. 构建 prefix 的 position_ids (基于 mRoPE anchor)
        position_ids = self._build_prefix_mrope_ids(mrope_anchor, K_p)  # [4, B, K_p]
        position_ids = position_ids.to(device)

        # 2. 用 text_model.forward 生成 prefix 的 KV cache
        from transformers.cache_utils import DynamicCache
        prefix_cache = DynamicCache()
        cache_position = torch.arange(K_p, device=device)
        prefix_attn_mask = torch.ones(B_size, K_p, device=device, dtype=torch.long)

        _prefix_outputs = text_model(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_attn_mask,
            position_ids=position_ids,
            past_key_values=prefix_cache,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        )
        new_prefix_cache = _prefix_outputs.past_key_values

        # 3. inplace 覆盖 cache 中前 K_p 个位置
        # Qwen3-VL: 所有层都是标准 full_attention
        for layer_idx in range(len(text_model.layers)):
            past_key_values.key_cache[layer_idx][:, :, :K_p, :] = new_prefix_cache.key_cache[layer_idx]
            past_key_values.value_cache[layer_idx][:, :, :K_p, :] = new_prefix_cache.value_cache[layer_idx]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """RoPE 辅助函数"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def save_pretrained(self, save_dir: str):
        """保存 RLD Controller 参数 (兼容 ZeRO-3 参数分片)"""
        os.makedirs(save_dir, exist_ok=True)
        controller_path = os.path.join(save_dir, "rld_controller.pt")

        # ZeRO-3 下参数被分片到多卡，需要 gather 后才能保存
        try:
            import deepspeed
            # 检查参数是否真的被 ZeRO-3 分片了
            first_param = next(self.controller.parameters())
            if hasattr(first_param, 'ds_id'):
                # ZeRO-3 模式: 所有进程都必须参与 gather，否则会死锁
                state_dict = {}
                for name, param in self.controller.named_parameters():
                    # GatheredParameters 是集合通信操作，所有进程都必须执行
                    with deepspeed.zero.GatheredParameters(param):
                        # 但只有主进程需要克隆数据用于保存
                        if self._verbose:
                            state_dict[name] = param.data.cpu().clone()
                if self._verbose:
                    torch.save(state_dict, controller_path)
                    print(f"[RLD] Controller 已保存到 {controller_path} (ZeRO-3 gathered)")
                return

        except ImportError:
            pass

        # 非 ZeRO-3 模式 (包括 ZeRO-2 和非 DeepSpeed)，直接保存
        if self._verbose:
            torch.save(self.controller.state_dict(), controller_path)
            print(f"[RLD] Controller 已保存到 {controller_path}")

    def load_pretrained(self, save_dir: str):
        """加载 RLD Controller 参数"""
        controller_path = os.path.join(save_dir, "rld_controller.pt")
        state_dict = torch.load(controller_path, map_location="cpu")
        self.controller.load_state_dict(state_dict)
        if self._verbose:
            print(f"[RLD] Controller 已从 {controller_path} 加载")
