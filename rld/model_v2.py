"""NLD Model v2: Native Latent Draft — 基于 VLM 原生隐空间的双流推理解码

核心思想 (COCONUT 式隐空间推理 + CoT 统一序列架构):
  主线: VLM 生成自然语言 CoT (可解释)
  辅助: 在 step boundary 处插入隐空间思考步骤 (信息增益)
  统一: 隐式推理的 hidden states 作为特殊 token 永久存在于历史序列中，从不丢弃
       隐式推理 + 自然语言 CoT 构成完整的、连续的推理过程

与 RLD v1 的根本区别:
  NLD v2: VLM 原生空间思考 → thought KV 直接追加到全局 cache (无瓶颈, 信息增益)

统一 forward 架构:
  训练和推理都使用 text_model() 整体调用 (而非手动逐层 forward)
  - KV cache 由 DynamicCache 自动管理
  - gradient checkpointing 由 text_model 内部正确处理
  - attention mask 由 text_model 内部自动构建
  - thinker 内部也使用 text_model() 调用，thought KV 直接追加到全局 cache
  - 训推一致: 训练和推理使用完全相同的 forward 路径

训练流程 (Segment-wise + Latent Thinking, 统一序列):
  1. 视觉 encoder → visual_embeddings (no_grad)
  2. 按 step boundary 切分序列为 segments
  3. 对每个 segment:
     a. text_model() 整体调用 (全量微调, 所有层有梯度)
        KV cache 自动管理，如果之前有 thought tokens，它们的 KV 已在 cache 中
        当前 segment 的 attention 自然能看到 thought tokens
     b. 如果是 step boundary:
        - NativeLatentThinker: last_hidden → text_model() × steps → thought_output
        - 方案 B: thinker 内部使用局部 cache (训练时不修改 global_cache)
        - thought_output 作为 prefix embedding 注入下一个 segment 的 inputs_embeds
        - 下一个 segment 的 text_model() 把 thought prefix + segment tokens 的 KV 一起写入 global_cache
        - 梯度路径: CE loss → Segment hidden → inputs_embeds → thought_output → thinker → VLM
  4. concat 所有 segment hidden (含 thought prefix) → logits → CE loss
  5. 梯度路径: CE loss → logits → segment hidden → inputs_embeds → thought_output → thinker → VLM 全部层

监督信号:
  不需要对隐空间步骤本身标注！
  CE loss 的梯度通过 thought hidden states 自动回传到 NativeLatentThinker。

参数量:
NLD v2: NativeLatentThinker ~2M + 全量微调 VLM 8.3B (FSDP 分布式)
"""
import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
from transformers.cache_utils import DynamicCache as _RawDynamicCache

# Qwen3-VL
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .latent_thinker import NativeLatentThinker, StageConceptAligner, VisualProbe, compute_complementarity_loss, compute_residual_value_loss




# ============================================================
# DynamicCache 兼容层 (复用 model.py 的实现)
# ============================================================

class _CacheProxy:
    """代理 DynamicCache.layers 中每个 layer 的 keys 或 values 属性"""
    def __init__(self, cache: "_RawDynamicCache", attr: str):
        self._cache = cache
        self._attr = attr

    def __len__(self):
        return len(self._cache.layers)

    def __getitem__(self, idx):
        layer = self._cache.layers[idx]
        return getattr(layer, self._attr, None)

    def __setitem__(self, idx, value):
        while len(self._cache.layers) <= idx:
            from transformers.cache_utils import DynamicLayer
            self._cache.layers.append(DynamicLayer())
        setattr(self._cache.layers[idx], self._attr, value)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, value):
        from transformers.cache_utils import DynamicLayer
        idx = len(self._cache.layers)
        self._cache.layers.append(DynamicLayer())
        setattr(self._cache.layers[idx], self._attr, value)


def _patch_dynamic_cache():
    """为 DynamicCache 添加 key_cache / value_cache 兼容属性 (幂等)"""
    if hasattr(_RawDynamicCache, '_compat_patched'):
        return

    @property
    def _key_cache_compat(self):
        return _CacheProxy(self, "keys")

    @property
    def _value_cache_compat(self):
        return _CacheProxy(self, "values")

    if not hasattr(_RawDynamicCache, 'key_cache') or isinstance(
        getattr(_RawDynamicCache, 'key_cache', None), property
    ):
        _RawDynamicCache.key_cache = _key_cache_compat

    if not hasattr(_RawDynamicCache, 'value_cache') or isinstance(
        getattr(_RawDynamicCache, 'value_cache', None), property
    ):
        _RawDynamicCache.value_cache = _value_cache_compat

    _RawDynamicCache._compat_patched = True

_patch_dynamic_cache()

from transformers.cache_utils import DynamicCache


def _extract_visual_output(vision_output):
    """从视觉编码器输出中提取 merged_hidden_states 和 deepstack_features"""
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


class NLDModel(nn.Module):
    """
    NLD Model v2: Native Latent Draft
    
    Qwen3-VL + NativeLatentThinker 双流推理解码
    
    统一 forward 架构:
    ┌──────────────────────────────────────────────────────────────┐
    │  训练流程 (Segment-wise + Latent Thinking):                   │
    │  1. 视觉 encoder → visual_embeddings (no_grad)               │
    │  2. 按 step boundary 切分序列为 segments                      │
    │  3. 对每个 segment:                                           │
    │     a. text_model() 整体调用 (全量微调, 所有层有梯度)         │
    │        KV cache 自动管理, thought KV 已在 cache 中            │
    │     b. 如果是 step boundary:                                  │
    │        NativeLatentThinker → text_model() × steps             │
    │        thought KV 直接追加到全局 cache                        │
    │        thought hidden states 收集到输出序列                   │
    │  4. concat 所有 hidden (含 thought 占位) → logits             │
    │     thought 位置 labels=-100 (不计算 loss)                    │
    │  5. CE loss → 梯度回传到 NativeLatentThinker + VLM 全部层     │
    └──────────────────────────────────────────────────────────────┘
    """

    # Step boundary 检测: 使用 <|latent|> token id
    # <|latent|> 是注册的特殊 token，boundary 检测通过 token id 比较完成 (O(1))
    LATENT_TOKEN = "<|latent|>"

    def __init__(
        self,
        model_path: str,
        hidden_size: int = 4096,
        total_layers: int = 36,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        # NativeLatentThinker 参数
        max_think_steps: int = 5,
        think_layer_start: int = 0,
        think_layer_end: int = 36,
        saturation_exit_threshold: float = 0.99,
        # 以下参数保留接口兼容性，但不再使用
        adaptive_exit: bool = True,
        exit_threshold: float = 0.5,
        ponder_cost_weight: float = 0.01,
        saturation_aux_weight: float = 0.005,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_layers = total_layers
        self._verbose = _is_main_process()
        self._debug = self._verbose and os.environ.get('RLD_DEBUG', '0') == '1'

        # 注意力实现
        env_attn = os.environ.get('RLD_ATTN_IMPL', None)
        if env_attn is not None and env_attn != attn_implementation:
            if self._verbose:
                print(f"[NLD] ⚠️ 注意力实现: {attn_implementation} → {env_attn} (通过 RLD_ATTN_IMPL 覆盖)")
            attn_implementation = env_attn
        self.attn_implementation = attn_implementation

        # ====== 1. 加载并冻结 Qwen3-VL ======
        if self._verbose:
            print(f"[NLD] 加载 Qwen3-VL: {model_path}")
            print(f"[NLD] 注意力实现: {attn_implementation}")

        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        # 全量微调: 基座参数全部可训练 (使用 FSDP 分布式)
        # 不再冻结基座参数
        for param in self.base_model.parameters():
            param.requires_grad = True

        if self._verbose:
            total_params = sum(p.numel() for p in self.base_model.parameters())
            print(f"[NLD] Qwen3-VL 全量微调: {total_params / 1e9:.2f}B 参数 (全部可训练)")

        # ====== 2. 创建 NativeLatentThinker ======
        # 从基座 config 获取实际 hidden_size
        text_config = self.base_model.config.text_config if hasattr(self.base_model.config, 'text_config') else self.base_model.config
        actual_hidden_size = getattr(text_config, 'hidden_size', hidden_size)
        num_attention_heads = getattr(text_config, 'num_attention_heads', 28)
        
        # think_layers: 隐空间思考使用全部 VLM 层 (与 COCONUT 一致)
        # COCONUT: 每步 recurrence 过完整 Transformer 全部层
        self.think_layer_start = think_layer_start
        self.think_layer_end = min(think_layer_end, total_layers)
        self.think_layer_indices = list(range(total_layers))  # 全部层
        
        self.latent_thinker = NativeLatentThinker(
            hidden_size=actual_hidden_size,
            max_think_steps=max_think_steps,
            saturation_exit_threshold=saturation_exit_threshold,
            latent_end_token_id=None,  # 在 set_processor 中设置
        )
        self.latent_thinker = self.latent_thinker.to(dtype=torch_dtype)
        
        # ====== 2.5 功能分化监督模块 ======
        # VisualProbe: 检测 thought hidden 是否编码了视觉信息
        self.visual_probe = VisualProbe(hidden_size=actual_hidden_size)
        self.visual_probe = self.visual_probe.to(dtype=torch_dtype)
        
        # 功能分化监督权重 (可通过环境变量覆盖)
        self.complementarity_weight = float(os.environ.get('NLD_COMPLEMENTARITY_WEIGHT', '0.01'))
        self.visual_probe_weight = float(os.environ.get('NLD_VISUAL_PROBE_WEIGHT', '0.005'))
        self.residual_value_weight = float(os.environ.get('NLD_RESIDUAL_VALUE_WEIGHT', '0.01'))
        # Key Token Decoding Loss 权重 (直接塑形 hidden states，不缩放)
        self.key_token_weight = float(os.environ.get('NLD_KEY_TOKEN_WEIGHT', '1.0'))
        
        if self._verbose:
            probe_params = sum(p.numel() for p in self.visual_probe.parameters())
            print(f"[NLD] VisualProbe 参数: {probe_params:,} ({probe_params/1e6:.2f}M)")
            print(f"[NLD] 功能分化监督权重: complementarity={self.complementarity_weight}, "
                  f"visual_probe={self.visual_probe_weight}, residual_value={self.residual_value_weight}, "
                  f"key_token={self.key_token_weight})")

        # GQA 参数        
        # GQA 参数 (推理时需要)
        self.num_kv_heads = getattr(text_config, 'num_key_value_heads', 8)
        self.head_dim = getattr(text_config, 'head_dim', actual_hidden_size // num_attention_heads)

        if self._verbose:
            self.latent_thinker.print_summary()
            print(f"[NLD] Think layers: 全部 {len(self.think_layer_indices)} 层 (L0~L{total_layers-1}, 与 COCONUT 一致)")
            print(f"[NLD] 全量微调模式: 所有层都有梯度 (FSDP 分布式)")
            print(f"[NLD] GQA: num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}")

        # ====== 3. Step delimiter token ids ======
        self.step_delimiter_ids = None
        self.step_delimiter_id = None

    @property
    def _inner_model(self):
        """获取内部 Qwen3VL 模型"""
        return self.base_model.model

    @property
    def _lm_head(self):
        """获取 lm_head"""
        return self.base_model.lm_head

    def set_processor(self, processor):
        """设置 processor (用于推理时的 tokenizer)"""
        self.processor = processor
        tokenizer = processor.tokenizer
        # 获取 <|latent|> token id (必须已注册为特殊 token)
        self.latent_token_id = tokenizer.convert_tokens_to_ids(self.LATENT_TOKEN)
        if self.latent_token_id == tokenizer.unk_token_id:
            if self._verbose:
                print(f"[NLD] ⚠️ {self.LATENT_TOKEN} 未在 tokenizer 中注册, boundary 检测将使用 fallback")
            self.latent_token_id = None
        else:
            if self._verbose:
                print(f"[NLD] ✅ <|latent|> token id = {self.latent_token_id}")
        
        # 获取 <|/latent|> token id (退出 token)
        LATENT_END_TOKEN = "<|/latent|>"
        self.latent_end_token_id = tokenizer.convert_tokens_to_ids(LATENT_END_TOKEN)
        if self.latent_end_token_id == tokenizer.unk_token_id:
            if self._verbose:
                print(f"[NLD] ⚠️ {LATENT_END_TOKEN} 未在 tokenizer 中注册, 退出 token 预测将禁用")
            self.latent_end_token_id = None
        else:
            if self._verbose:
                print(f"[NLD] ✅ <|/latent|> token id = {self.latent_end_token_id}")
            # 将 latent_end_token_id 传递给 NativeLatentThinker
            self.latent_thinker.latent_end_token_id = self.latent_end_token_id
        
        # 兼容旧接口
        self.step_delimiter_ids = None
        self.step_delimiter_id = self.latent_token_id

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点 (代理到 base_model, DeepSpeed/Trainer 会调用此方法)"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            if self._verbose:
                print("[NLD] ✅ 梯度检查点已启用 (节省显存, 以计算换内存)")

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        """检查是否启用了梯度检查点"""
        if hasattr(self.base_model, 'is_gradient_checkpointing'):
            return self.base_model.is_gradient_checkpointing
        return False

    @staticmethod
    def _detach_cache(cache: DynamicCache) -> DynamicCache:
        """Detach KV cache 中所有 tensor 的梯度，截断跨 segment 的梯度传播。
        
        Args:
            cache: 需要 detach 的 DynamicCache
        
        Returns:
            新的 DynamicCache，所有 KV tensor 已 detach
        """
        from transformers.cache_utils import DynamicLayer as _DL
        detached = DynamicCache()
        for layer in cache.layers:
            new_layer = _DL()
            if hasattr(layer, 'keys') and layer.keys is not None:
                new_layer.keys = layer.keys.detach()
                new_layer.values = layer.values.detach()
                new_layer.is_initialized = True
            detached.layers.append(new_layer)
        return detached

    def _find_delimiter_positions(self, input_ids_1d: torch.Tensor) -> List[int]:
        """在 token 序列中查找 step boundary 位置
        
        仅使用 <|latent|> token id 直接定位 (O(n) 扫描, 无需文本解码)
        注意: Final Answer 后不触发隐空间推理，因此不作为 boundary
        """
        positions = []
        
        # 通过 token id 查找 <|latent|> 位置 (精确、高效)
        if hasattr(self, 'latent_token_id') and self.latent_token_id is not None:
            latent_positions = (input_ids_1d == self.latent_token_id).nonzero(as_tuple=True)[0].tolist()
            positions.extend(latent_positions)
        
        return sorted(set(positions))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        step_boundaries: Optional[List[List[int]]] = None,
        prompt_lens: Optional[List[int]] = None,
        loss_weight_mask: Optional[torch.Tensor] = None,
        latent_think_steps: Optional[List[List[int]]] = None,
        stage_concept_token_ids: Optional[List[List[List[int]]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播 (Chunkwise + Latent Thinking)
        
        核心流程 (统一序列: thought tokens 永久保留):
          1. 视觉 encoder → visual_embeddings (no_grad)
          2. 按 step boundary 切分序列为 chunks
          3. 对每个 chunk:
             a. VLM 36 层 forward (有梯度)
             b. 如果是 step boundary:
                NativeLatentThinker → thought_output
                thought_output 作为下一个 chunk 的 prefix embeddings
                thought KV 永久保留在历史 cache 中
          4. concat 所有 chunk hidden (含 thought 占位, labels=-100) → logits → CE loss
        
        梯度路径:
          CE loss → logits → chunk_{c+1} hidden → thought_prefix → NativeLatentThinker
          → VLM 全部层 (全量微调) → chunk_c hidden
        
        统一序列设计:
          thought tokens 的 KV 不从 cache 中剥离，后续所有 chunk/token 都能通过
          attention 看到之前所有的 thought tokens。隐式推理 + 自然语言 CoT 构成
          完整的、连续的推理序列。
        """
        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device

        # 重置 accumulated losses 累积器
        self._accumulated_complementarity_loss = None
        self._accumulated_visual_probe_loss = None
        self._accumulated_residual_value_loss = None
        self._accumulated_exit_token_loss = None
        self._accumulated_key_token_loss = None
        self._last_exit_stats = None
        self._last_complementarity_stats = None
        self._last_residual_stats = None

        import time as _time
        _fwd_start = _time.time()
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        _rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        _is_rank0 = (_rank == 0)

        # ====== dtype 对齐 ======
        if not hasattr(self, '_base_model_dtype'):
            self._base_model_dtype = torch.bfloat16
            for _n, _p in self.base_model.named_parameters():
                self._base_model_dtype = _p.dtype
                break
            if self._verbose:
                thinker_dtype = next(self.latent_thinker.parameters()).dtype
                print(f"[NLD] dtype 检测: 基座模型={self._base_model_dtype}, "
                      f"latent_thinker={thinker_dtype}")

        _base_dtype = self._base_model_dtype

        # ====== 1. 视觉 encoder (no_grad) ======
        _t_vision_start = _time.time()
        with torch.no_grad():
            cached_visual_embeds = None
            deepstack_features = None

            if pixel_values is not None:
                inner_model = self._inner_model
                _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
                vision_output = inner_model.visual(
                    _pixel_values, grid_thw=image_grid_thw
                )
                merged_hidden_states, deepstack_features = _extract_visual_output(vision_output)
                split_sizes = (
                    image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
                ).tolist()
                image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))

                cached_visual_embeds = torch.cat(image_embeds_list, dim=0)

        _t_vision_end = _time.time()

        # ====== 2. Embedding + 视觉融合 ======
        inner_model = self._inner_model
        text_model = inner_model.language_model
        lm_head = self._lm_head

        with torch.no_grad():
            inputs_embeds = inner_model.get_input_embeddings()(input_ids)

            if cached_visual_embeds is not None:
                image_token_id = inner_model.config.image_token_id
                image_mask = (input_ids == image_token_id)
                num_image_tokens = image_mask.sum().item()
                if num_image_tokens > 0:
                    image_embeds = cached_visual_embeds[:num_image_tokens].to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask_3d, image_embeds)

        # 确保 inputs_embeds 有梯度 (gradient checkpointing 需要至少一个输入有 requires_grad)
        if not inputs_embeds.requires_grad:
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        # visual_pos_masks (DeepStack 需要)
        visual_pos_masks = None
        deepstack_for_fwd = None
        if pixel_values is not None and image_grid_thw is not None:
            image_token_id = inner_model.config.image_token_id
            visual_pos_masks = (input_ids == image_token_id)
            deepstack_for_fwd = deepstack_features

        # ====== 3. mRoPE position_ids ======
        with torch.no_grad():
            seg_token_mask = torch.ones(B, seq_len, device=device, dtype=torch.long)
            inner_model.rope_deltas = None
            position_ids, rope_deltas = inner_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw if pixel_values is not None else None,
                video_grid_thw=None,
                attention_mask=seg_token_mask,
            )
            inner_model.rope_deltas = rope_deltas

            if position_ids is not None and position_ids.dim() == 3:
                text_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
                position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

        # position_ids 处理
        _position_ids = position_ids
        if _position_ids is not None and _position_ids.ndim == 3 and _position_ids.shape[0] == 4:
            _text_position_ids = _position_ids[0]
            _mrope_position_ids = _position_ids[1:]
        elif _position_ids is not None:
            _text_position_ids = _position_ids[0]
            _mrope_position_ids = _position_ids
        else:
            cache_position = torch.arange(seq_len, device=device)
            _text_position_ids = cache_position.view(1, 1, -1).expand(3, B, -1)[0]
            _mrope_position_ids = cache_position.view(1, 1, -1).expand(3, B, -1)

        cache_position = torch.arange(seq_len, device=device)

        # ====== 4. 检测 step 边界 ======
        if prompt_lens is not None:
            gen_start = max(prompt_lens)
        else:
            gen_start = 0

        if step_boundaries is None and hasattr(self, 'processor') and self.processor is not None:
            step_boundaries = []
            for b in range(B):
                positions = self._find_delimiter_positions(input_ids[b])
                positions = [p for p in positions if p >= gen_start]
                step_boundaries.append(positions)

        # 收集所有 step 边界
        all_boundaries = set()
        if step_boundaries:
            for b_list in step_boundaries:
                all_boundaries.update(b_list)
        split_points = sorted([p for p in all_boundaries if gen_start < p < seq_len])

        per_sample_sets = [set(bl) for bl in step_boundaries] if step_boundaries else [set() for _ in range(B)]

        MAX_STEPS = 14
        if len(split_points) > MAX_STEPS:
            split_points = split_points[:MAX_STEPS]

        # ====== 构建 chunks ======
        chunk_ranges = []
        if len(split_points) == 0:
            chunk_ranges.append((0, seq_len))
        else:
            chunk_ranges.append((0, split_points[0] + 1))
            for i in range(1, len(split_points)):
                chunk_ranges.append((split_points[i - 1] + 1, split_points[i] + 1))
            if split_points[-1] + 1 < seq_len:
                chunk_ranges.append((split_points[-1] + 1, seq_len))

        # ====== FSDP 预同步: 在循环之前同步所有 rank 的 text_model() 调用次数 ======
        # FSDP full_shard 模式下, 每次 text_model() 调用都触发 AllGather 集合通信,
        # 要求所有 rank 同步参与。不同样本的 chunks 数和 thinker steps 数不同,
        # 导致各 rank 的 text_model() 调用次数不同 → 集合通信不匹配 → NCCL 死锁。
        # 解决方案: 在循环之前预计算本 rank 的调用次数, 通过 all_reduce 获取全局最大值,
        # 循环结束后不足的 rank 用 dummy forward 补齐。
        # 注意: all_reduce 必须在所有 text_model() 调用之前执行, 否则会与 FSDP 的
        # AllGather 交叉导致死锁 (不同 rank 在同一 SeqNum 执行不同类型的集合操作)。
        import torch.distributed as dist
        _fsdp_sync_needed = dist.is_initialized() and self.training
        _fsdp_max_calls = 0
        _fsdp_my_calls = 0
        if _fsdp_sync_needed:
            # 预计算本 rank 的 text_model() 调用次数:
            # - 每个 chunk 调用 1 次 text_model()
            # - 每个 boundary 处的 thinker 调用 target_steps 次 text_model()
            # 训练时不再受 max_think_steps 硬上限约束，使用实际的 latent_think_steps
            _num_boundaries = len(split_points)
            # 计算每个 boundary 的实际 target_steps (取 batch 中最大值)
            _predicted_think_steps_per_boundary = self.latent_thinker.max_think_steps  # 默认回退值
            if latent_think_steps is not None:
                _all_steps = []
                for b_idx in range(B):
                    if b_idx < len(latent_think_steps):
                        for s in latent_think_steps[b_idx]:
                            if s > 0:
                                _all_steps.append(s)
                if _all_steps:
                    _predicted_think_steps_per_boundary = max(_all_steps)
            _my_predicted_calls = len(chunk_ranges) + _num_boundaries * _predicted_think_steps_per_boundary
            _calls_tensor = torch.tensor([_my_predicted_calls], device=device, dtype=torch.long)
            dist.all_reduce(_calls_tensor, op=dist.ReduceOp.MAX)
            _fsdp_max_calls = _calls_tensor.item()
            _fsdp_my_calls = 0  # 实际调用计数器, 在循环中递增
            
            if self._verbose and self._fwd_count <= 2:
                print(f"[rank {_rank}] FSDP 预同步: 预计 {_my_predicted_calls} 次 text_model() 调用, "
                      f"全局最大 {_fsdp_max_calls} 次", flush=True)

        # ====== 5. Segment-wise Forward + Latent Thinking ======
        # 统一 forward 架构: 使用 text_model() 整体调用替代手动逐层 forward
        # - KV cache 由 DynamicCache 自动管理
        # - attention mask 由 text_model 内部自动构建
        # - 方案 B (梯度可回传): thinker 内部使用局部 cache (不修改 global_cache)
        #   thinker 完成后, thought_output 作为 prefix embedding 注入下一个 segment 的 inputs_embeds
        #   梯度路径: CE loss → Segment 2 hidden → inputs_embeds → thought_output → thinker → VLM
        #   下一个 segment 的 text_model() 会把 thought prefix + segment tokens 的 KV 一起写入 global_cache
        #   这样后续所有 segment 都能通过 KV cache 看到 thought tokens

        # ---- 临时禁用 gradient checkpointing ----
        # activation checkpointing 与有状态 KV cache 不兼容:
        # backward 重计算 layer forward 时, DynamicCache 已被后续 segment/thinker 步骤修改,
        # 导致 KV shape 不一致 (forward 时 410, recompute 时 420)。
        # 解决方案: 在 segment-wise forward 循环中禁用 gradient checkpointing,
        # 仍保留 FSDP 参数分片来节省显存。循环结束后恢复。
        _gc_was_enabled = getattr(text_model, 'gradient_checkpointing', False)
        if _gc_was_enabled:
            text_model.gradient_checkpointing = False
            if self._verbose and self._fwd_count == 0:
                print("[NLD] ⚠️ 临时禁用 gradient checkpointing (segment-wise forward 与 KV cache 不兼容)")

        all_chunk_hidden = []  # 收集每个 segment 的 hidden states (含 thought prefix)
        thought_count = 0  # 实际触发的隐空间思考次数
        kv_offset = 0  # thought tokens 在 KV cache 中占据的额外位置数
        thought_insertions = []  # [(insert_pos_in_output_seq, num_thought_tokens), ...]
        cumulative_thought_tokens = 0  # 已插入的 thought token 总数
        
        # 暂存上一个 boundary 产生的 thought_output, 注入到下一个 segment
        pending_thought_prefix = None  # (thought_output, num_thought_steps, thinker_result)

        # 全局 KV cache (由 DynamicCache 自动管理)
        global_cache = DynamicCache()

        _t_base_fwd_start = _time.time()

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
            chunk_len = chunk_end - chunk_start

            # ---- 当前 segment 的输入切片 ----
            chunk_embeds = inputs_embeds[:, chunk_start:chunk_end, :]  # [B, chunk_len, H]
            # cache_position 需要加上 kv_offset (之前所有 thought tokens 占据的位置)
            chunk_cache_position = cache_position[chunk_start:chunk_end] + kv_offset
            chunk_mrope_pos = _mrope_position_ids[:, :, chunk_start:chunk_end]

            # ---- position_ids: [4, B, chunk_len] ----
            chunk_text_pos = (_text_position_ids[:, chunk_start:chunk_end] + kv_offset
                              if _text_position_ids.dim() == 2
                              else _text_position_ids[:, chunk_start:chunk_end])
            actual_position_ids = torch.cat([chunk_text_pos.unsqueeze(0), chunk_mrope_pos], dim=0)

            # ---- 方案 B: 如果有 pending thought prefix, 注入到当前 segment 的 inputs_embeds 前面 ----
            if pending_thought_prefix is not None:
                thought_output_prefix, n_thought_prefix, _ = pending_thought_prefix
                
                # thought prefix 的 position_ids: [4, B, n_thought]
                # thought tokens 的位置紧跟在上一个 segment 末尾之后
                thought_base_pos = chunk_cache_position[0].item() - n_thought_prefix
                thought_cache_positions = torch.arange(
                    thought_base_pos, thought_base_pos + n_thought_prefix, device=device
                )
                thought_mrope_positions = thought_cache_positions.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                thought_text_positions = torch.full(
                    (B, n_thought_prefix), thought_base_pos, device=device, dtype=torch.long
                )
                # 多步 thought 的 text_pos 递增
                for t_idx in range(n_thought_prefix):
                    thought_text_positions[:, t_idx] = thought_base_pos + t_idx
                thought_pos_ids = torch.cat([
                    thought_text_positions.unsqueeze(0),  # [1, B, n_thought]
                    thought_mrope_positions,  # [3, B, n_thought]
                ], dim=0)  # [4, B, n_thought]
                
                # 拼接: thought prefix + segment embeds
                chunk_embeds = torch.cat([
                    thought_output_prefix.to(dtype=chunk_embeds.dtype),
                    chunk_embeds,
                ], dim=1)  # [B, n_thought + chunk_len, H]
                
                # 拼接 cache_position
                chunk_cache_position = torch.cat([thought_cache_positions, chunk_cache_position])
                
                # 拼接 position_ids
                actual_position_ids = torch.cat([thought_pos_ids, actual_position_ids], dim=2)
                
                pending_thought_prefix = None  # 已消费

            # ---- attention_mask ----
            total_kv_len = chunk_cache_position[-1].item() + 1
            actual_attn_mask = torch.ones(B, total_kv_len, device=device, dtype=attention_mask.dtype)

            # ---- visual_pos_masks 处理 ----
            # 如果有 thought prefix, visual_pos_masks 需要在前面补 False
            seg_visual_pos_masks = (visual_pos_masks[:, chunk_start:chunk_end] if visual_pos_masks is not None else None)
            if seg_visual_pos_masks is not None and chunk_embeds.shape[1] > chunk_len:
                # thought prefix 不是视觉 token, 补 False
                n_prefix = chunk_embeds.shape[1] - chunk_len
                prefix_mask = torch.zeros(B, n_prefix, device=device, dtype=seg_visual_pos_masks.dtype)
                seg_visual_pos_masks = torch.cat([prefix_mask, seg_visual_pos_masks], dim=1)

            # ---- DeepStack: 只在包含视觉 token 的 segment 传递 ----
            # 视觉 token 全部在 prompt 中 (第一个 chunk), 后续 chunk 只有纯文本 CoT 步骤
            # _deepstack_process 要求 visual_embeds 数量 == visual_pos_masks 中 True 的数量
            # 因此: 无视觉 token 的 segment 必须不传 deepstack, 否则尺寸不匹配
            seg_deepstack = None
            if seg_visual_pos_masks is not None and deepstack_for_fwd is not None:
                num_vis_in_seg = seg_visual_pos_masks.sum().item()
                if num_vis_in_seg > 0:
                    # 第一个 chunk 包含所有视觉 token, deepstack_for_fwd 完整传入即可
                    seg_deepstack = deepstack_for_fwd
                else:
                    # 后续 chunk 无视觉 token, 不传 deepstack
                    seg_visual_pos_masks = None

            # ---- text_model() 整体调用 ----
            # KV cache 自动管理: 当前 segment (含 thought prefix) 的 KV 自动追加到 global_cache
            # thought prefix 的 KV 通过这里写入 global_cache (而非 thinker 内部)
            # 梯度路径: CE loss → hidden → inputs_embeds → thought_output → thinker ✅
            seg_outputs = text_model(
                inputs_embeds=chunk_embeds,
                attention_mask=actual_attn_mask,
                position_ids=actual_position_ids,
                past_key_values=global_cache,
                cache_position=chunk_cache_position,
                use_cache=True,
                visual_pos_masks=seg_visual_pos_masks,
                deepstack_visual_embeds=seg_deepstack,
            )
            global_cache = seg_outputs.past_key_values
            if _fsdp_sync_needed:
                _fsdp_my_calls += 1
            
            # last_hidden_state 已过 final norm
            chunk_normed = seg_outputs.last_hidden_state  # [B, actual_input_len, H]

            # 收集 segment 的 hidden states (含 thought prefix 部分)
            all_chunk_hidden.append(chunk_normed)

            # ---- 跨 segment 梯度截断: detach KV cache ----
            # 每个 segment 的计算图独立，通过 detach KV cache 截断跨 segment 梯度
            # 梯度通过 thought prefix → inputs_embeds → thinker 回传
            global_cache = self._detach_cache(global_cache)

            # ---- 如果是 step boundary, 触发 NativeLatentThinker ----
            chunk_end_token = chunk_end - 1
            is_at_boundary = chunk_end_token in set(split_points)

            if is_at_boundary and chunk_idx < len(chunk_ranges) - 1:
                # 取当前 segment 最后 1 个 token 的 hidden state 作为 thought token
                # 注意: 如果当前 segment 有 thought prefix, 最后一个 token 仍然是原始 segment 的最后一个
                last_hidden = chunk_normed[:, -1:, :]  # [B, 1, H]
                
                # 第一步 thought 的位置 (放在当前 segment 末尾之后)
                base_thought_pos = chunk_end + kv_offset
                thought_positions = torch.arange(
                    base_thought_pos, base_thought_pos + 1, device=device
                )
                thought_mrope_pos = thought_positions.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                
                # 第一步的 position_ids: [4, B, 1]
                thought_text_pos_t = torch.full((B, 1), base_thought_pos, device=device, dtype=torch.long)
                thought_position_ids = torch.cat([thought_text_pos_t.unsqueeze(0), thought_mrope_pos], dim=0)
                
                # 第一步的 attention_mask
                think_attn_mask = torch.ones(B, base_thought_pos + 1, device=device, dtype=torch.long)
                
                # NativeLatentThinker: Coconut 等价的隐空间推理
                # 方案 B: thinker 内部使用局部 cache (训练时不修改 global_cache)
                # thought_output 将作为 prefix 注入下一个 segment
                
                # 确定当前 boundary 的 target_steps (per-sample 的隐空间迭代步数)
                # latent_think_steps: List[List[int]], 每个样本的每个 <|latent|> 对应的步数
                # thought_count 是当前已触发的 boundary 索引
                _target_steps = None
                if latent_think_steps is not None:
                    # 取 batch 中所有样本在当前 boundary 的 target_steps 的最大值
                    # (因为 thinker 对整个 batch 统一执行，步数取 max)
                    _batch_steps = []
                    for b_idx in range(B):
                        if b_idx < len(latent_think_steps) and thought_count < len(latent_think_steps[b_idx]):
                            _s = latent_think_steps[b_idx][thought_count]
                            if _s > 0:
                                _batch_steps.append(_s)
                    if _batch_steps:
                        _target_steps = max(_batch_steps)
                
                thinker_result = self.latent_thinker(
                    last_hidden=last_hidden,
                    text_model=text_model,
                    past_key_values=global_cache,
                    attention_mask=think_attn_mask,
                    cache_position=thought_positions,
                    position_ids=thought_position_ids,
                    rotary_emb_fn=text_model.rotary_emb,
                    mrope_position_ids=thought_mrope_pos,
                    base_thought_pos=base_thought_pos,
                    B=B,
                    target_steps=_target_steps,
                    lm_head=lm_head,
                    embed_tokens=inner_model.get_input_embeddings(),
                    stage_key_token_ids=stage_concept_token_ids,
                )
                
                thought_output = thinker_result['thought_output']  # [B, num_steps, H]
                num_thought_steps = thinker_result.get('num_thought_steps', 1)
                if _fsdp_sync_needed:
                    # 实际 thinker 内部执行了 num_thought_steps 次 text_model() 调用
                    # 预同步时按实际 target_steps 预计，dummy forward 会补齐差额
                    _fsdp_my_calls += num_thought_steps
                
                # 记录 thought token 在输出序列中的插入位置
                insert_pos = chunk_end + cumulative_thought_tokens
                thought_insertions.append((insert_pos, num_thought_steps))
                cumulative_thought_tokens += num_thought_steps
                
                # 方案 B: 暂存 thought_output, 作为下一个 segment 的 prefix
                pending_thought_prefix = (thought_output, num_thought_steps, thinker_result)
                
                # 累积 exit_token_loss (退出 token 预测 loss)
                if 'exit_token_loss' in thinker_result:
                    if self._accumulated_exit_token_loss is None:
                        self._accumulated_exit_token_loss = thinker_result['exit_token_loss']
                    else:
                        self._accumulated_exit_token_loss = self._accumulated_exit_token_loss + thinker_result['exit_token_loss']
                
                # 累积 key_token_loss (Key Token Decoding Loss)
                if 'key_token_loss' in thinker_result:
                    _kt_loss = self.key_token_weight * thinker_result['key_token_loss']
                    if self._accumulated_key_token_loss is None:
                        self._accumulated_key_token_loss = _kt_loss
                    else:
                        self._accumulated_key_token_loss = self._accumulated_key_token_loss + _kt_loss
                
                # 收集 exit_stats
                if 'exit_stats' in thinker_result:
                    self._last_exit_stats = thinker_result['exit_stats']
                
                # ====== 功能分化监督: 确保 thought 与 CoT 既继承又互补 ======
                if self.training:
                    # --- P1: 信息互补正则化 ---
                    # thought 不应该只是 CoT 的复制品，应该包含 CoT 子空间之外的新信息
                    if self.complementarity_weight > 0:
                        comp_loss, comp_stats = compute_complementarity_loss(
                            thought_output=thought_output,
                            last_cot_hidden=last_hidden,  # [B, 1, H] boundary 处的 CoT hidden
                            redundancy_threshold=0.7,
                            redundancy_weight=self.complementarity_weight,
                            novelty_weight=self.complementarity_weight * 0.5,
                        )
                        if self._accumulated_complementarity_loss is None:
                            self._accumulated_complementarity_loss = comp_loss
                        else:
                            self._accumulated_complementarity_loss = self._accumulated_complementarity_loss + comp_loss
                        self._last_complementarity_stats = comp_stats
                    
                    # --- P2: 视觉证据探针 ---
                    # thought 应该编码了足够的视觉信息 (通过 probe 检测)
                    if self.visual_probe_weight > 0 and cached_visual_embeds is not None:
                        thought_mean = thought_output.mean(dim=1)  # [B, H]
                        # 池化视觉特征: 取所有视觉 token 的均值
                        visual_pooled = cached_visual_embeds.mean(dim=0, keepdim=True)  # [1, H]
                        visual_pooled = visual_pooled.expand(B, -1).to(thought_mean.dtype)  # [B, H]
                        probe_loss = self.visual_probe(thought_mean, visual_pooled)
                        probe_loss = self.visual_probe_weight * probe_loss
                        if self._accumulated_visual_probe_loss is None:
                            self._accumulated_visual_probe_loss = probe_loss
                        else:
                            self._accumulated_visual_probe_loss = self._accumulated_visual_probe_loss + probe_loss
                    
                thought_count += 1
                kv_offset += num_thought_steps
                
                # 方案 B: 不需要 detach thinker 的 KV cache
                # 因为 thinker 训练时使用局部 cache, 不修改 global_cache
                # thought KV 将在下一个 segment forward 时通过 prefix 注入写入 global_cache

        # ====== FSDP Dummy Forward 对齐 ======
        # 根据循环前预同步的 _fsdp_max_calls, 补齐不足的 text_model() 调用次数。
        # 此时所有 rank 都已完成 segment-wise forward 循环, 不会与 FSDP AllGather 交叉。
        if _fsdp_sync_needed:
            _dummy_needed = _fsdp_max_calls - _fsdp_my_calls
            
            if _dummy_needed > 0:
                # 用 1 个 dummy token 做 forward, 不使用 global_cache (避免污染)
                # dummy 输出不参与 loss 计算
                _dummy_embeds = torch.zeros(B, 1, inputs_embeds.shape[-1], device=device, dtype=inputs_embeds.dtype)
                _dummy_embeds.requires_grad_(True)
                _dummy_cache_pos = torch.zeros(1, device=device, dtype=torch.long)
                _dummy_attn_mask = torch.ones(B, 1, device=device, dtype=torch.long)
                _dummy_pos_ids = torch.zeros(4, B, 1, device=device, dtype=torch.long)
                
                for _di in range(_dummy_needed):
                    _dummy_cache = DynamicCache()
                    _dummy_out = text_model(
                        inputs_embeds=_dummy_embeds,
                        attention_mask=_dummy_attn_mask,
                        position_ids=_dummy_pos_ids,
                        past_key_values=_dummy_cache,
                        cache_position=_dummy_cache_pos,
                        use_cache=True,
                    )
                    del _dummy_cache, _dummy_out
                
                if self._verbose and self._fwd_count <= 2:
                    print(f"[rank {_rank}] FSDP 对齐: 补齐 {_dummy_needed} 次 dummy forward "
                          f"(本 rank {_fsdp_my_calls} 次, 全局最大 {_fsdp_max_calls} 次)", flush=True)

        # 释放 KV cache
        del global_cache

        # ---- 恢复 gradient checkpointing ----
        if _gc_was_enabled:
            text_model.gradient_checkpointing = True

        _t_base_fwd_end = _time.time()

        # ====== 6. Concat 所有 chunk hidden → 完整序列 (含 thought tokens) ======
        if len(all_chunk_hidden) == 1:
            adapted_hidden = all_chunk_hidden[0]
        else:
            adapted_hidden = torch.cat(all_chunk_hidden, dim=1)  # [B, seq_len + total_thought_tokens, H]

        # ====== 7. 计算 logits ======
        logits = lm_head(adapted_hidden)  # [B, extended_seq_len, V]

        # ====== 7.5 对齐 labels 和 loss_weight_mask ======
        # 由于 thought tokens 被保留在序列中，adapted_hidden 的长度 > 原始 seq_len
        # 需要在 labels 和 loss_weight_mask 的 thought 位置插入 -100 / 0.0 占位
        if thought_insertions and labels is not None:
            # 构建扩展后的 labels: 在 thought 位置插入 -100
            extended_len = adapted_hidden.shape[1]
            extended_labels = torch.full((B, extended_len), -100, dtype=labels.dtype, device=device)
            
            # 构建映射: 原始序列中的每个位置 → 扩展序列中的位置
            # thought_insertions: [(insert_pos, num_tokens), ...]
            # insert_pos 是在扩展序列中的位置 (已考虑之前的 thought tokens)
            src_pos = 0  # 原始序列中的位置
            dst_pos = 0  # 扩展序列中的位置
            for insert_pos, n_thought in thought_insertions:
                # 复制 thought 之前的原始 token labels
                copy_len = insert_pos - dst_pos  # 这段是原始 token
                if copy_len > 0:
                    # 从原始 labels 中复制
                    orig_start = src_pos
                    orig_end = src_pos + copy_len
                    extended_labels[:, dst_pos:dst_pos + copy_len] = labels[:, orig_start:orig_end]
                    src_pos += copy_len
                    dst_pos += copy_len
                # thought token 位置: 保持 -100 (已初始化)
                dst_pos += n_thought
            # 复制剩余的原始 token labels
            remaining = seq_len - src_pos
            if remaining > 0:
                extended_labels[:, dst_pos:dst_pos + remaining] = labels[:, src_pos:src_pos + remaining]
            
            labels = extended_labels
            
            # 同样扩展 loss_weight_mask
            if loss_weight_mask is not None:
                extended_weights = torch.zeros((B, extended_len), dtype=loss_weight_mask.dtype, device=device)
                src_pos = 0
                dst_pos = 0
                for insert_pos, n_thought in thought_insertions:
                    copy_len = insert_pos - dst_pos
                    if copy_len > 0:
                        orig_start = src_pos
                        orig_end = src_pos + copy_len
                        extended_weights[:, dst_pos:dst_pos + copy_len] = loss_weight_mask[:, orig_start:orig_end]
                        src_pos += copy_len
                        dst_pos += copy_len
                    # thought token 位置: 权重为 0 (不计算 loss)
                    dst_pos += n_thought
                remaining = seq_len - src_pos
                if remaining > 0:
                    extended_weights[:, dst_pos:dst_pos + remaining] = loss_weight_mask[:, src_pos:src_pos + remaining]
                loss_weight_mask = extended_weights

        # ====== 7.6 残差价值 Loss (P0): 衡量 thought 对后续预测的增益 ======
        # 在 logits 和 labels 对齐后计算，利用 thought_insertions 定位 thought 后面的 token
        if (self.training and self.residual_value_weight > 0 
            and thought_insertions and labels is not None):
            K_residual = 5  # 对比 thought 后面 K 个 token
            for insert_pos, n_thought in thought_insertions:
                # thought 后面的 token 在扩展序列中的位置
                after_thought_start = insert_pos + n_thought
                after_thought_end = min(after_thought_start + K_residual, logits.shape[1] - 1)
                actual_K = after_thought_end - after_thought_start
                
                if actual_K > 0:
                    # 有 thought 时的 logits (已经算好了)
                    logits_after = logits[:, after_thought_start:after_thought_end, :]
                    labels_after = labels[:, after_thought_start + 1:after_thought_end + 1]  # shift by 1
                    
                    # 需要 boundary 处的 CoT hidden (thought 插入位置之前的最后一个 token)
                    boundary_hidden = adapted_hidden[:, insert_pos - 1:insert_pos, :]  # [B, 1, H]
                    
                    if labels_after.shape[1] == logits_after.shape[1]:
                        rv_loss, rv_stats = compute_residual_value_loss(
                            logits_at_boundary=logits_after,
                            labels_at_boundary=labels_after,
                            thought_output=adapted_hidden[:, insert_pos:insert_pos + n_thought, :],
                            last_cot_hidden=boundary_hidden,
                            lm_head=lm_head,
                            K=actual_K,
                        )
                        rv_loss = self.residual_value_weight * rv_loss
                        if self._accumulated_residual_value_loss is None:
                            self._accumulated_residual_value_loss = rv_loss
                        else:
                            self._accumulated_residual_value_loss = self._accumulated_residual_value_loss + rv_loss
                        self._last_residual_stats = rv_stats

        # ====== 8. 计算 loss ======
        ctrl_dtype = next(self.latent_thinker.parameters()).dtype
        total_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        main_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()

            if loss_weight_mask is not None:
                shift_weights = loss_weight_mask[:, 1:].contiguous().to(shift_logits.dtype)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                per_token_loss = per_token_loss.view(shift_labels.shape)
                valid_mask = (shift_labels != -100).float()
                weighted_loss = per_token_loss * shift_weights * valid_mask
                ce_loss = weighted_loss.sum() / (shift_weights * valid_mask).sum().clamp(min=1.0)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            main_loss = ce_loss.detach()
            total_loss = ce_loss
            
            # 加入信息互补正则化 (thought 与 CoT 既继承又互补)
            if hasattr(self, '_accumulated_complementarity_loss') and self._accumulated_complementarity_loss is not None:
                total_loss = total_loss + self._accumulated_complementarity_loss
            
            # 加入视觉证据探针 loss (thought 应该编码视觉信息)
            if hasattr(self, '_accumulated_visual_probe_loss') and self._accumulated_visual_probe_loss is not None:
                total_loss = total_loss + self._accumulated_visual_probe_loss
            
            # 加入残差价值 loss (thought 应该对预测有正向增益)
            if hasattr(self, '_accumulated_residual_value_loss') and self._accumulated_residual_value_loss is not None:
                total_loss = total_loss + self._accumulated_residual_value_loss
            
            # 加入退出 token 预测 loss (模型学习自适应退出 latent 推理)
            if hasattr(self, '_accumulated_exit_token_loss') and self._accumulated_exit_token_loss is not None:
                total_loss = total_loss + self._accumulated_exit_token_loss
            
            # 加入 Key Token Decoding Loss (每步 thought 应能解码出对应 stage 的关键词)
            if hasattr(self, '_accumulated_key_token_loss') and self._accumulated_key_token_loss is not None:
                total_loss = total_loss + self._accumulated_key_token_loss

        # 确保 total_loss 有 grad_fn
        if total_loss.grad_fn is None and self.training:
            anchor = next(self.latent_thinker.parameters())
            total_loss = total_loss + (anchor.sum() * 0.0).to(total_loss.dtype)

        # NaN/Inf 检测
        if torch.isnan(total_loss).item() or torch.isinf(total_loss).item():
            _loss_type = "NaN" if torch.isnan(total_loss).item() else "Inf"
            print(f"  ❌ [rank {_rank}] loss 为 {_loss_type}! total_loss={total_loss.item()}", flush=True)
            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
            anchor = next(self.latent_thinker.parameters())
            total_loss = total_loss + (anchor.sum() * 0.0).to(total_loss.dtype)

        # ====== 监控指标 ======
        _draft_metrics = {}
        with torch.no_grad():
            _draft_metrics['nld/num_steps'] = float(len(split_points))
            _draft_metrics['nld/num_chunks'] = float(len(chunk_ranges))
            _draft_metrics['nld/thought_count'] = float(thought_count)
            _draft_metrics['nld/total_thought_tokens'] = float(cumulative_thought_tokens)
            
            # 退出统计
            if hasattr(self, '_last_exit_stats') and self._last_exit_stats is not None:
                _draft_metrics['nld/mean_think_steps'] = self._last_exit_stats.get('mean_steps', 0)
                _draft_metrics['nld/actual_think_steps'] = self._last_exit_stats.get('actual_steps', 0)
            
            # 功能分化监督指标
            if hasattr(self, '_accumulated_complementarity_loss') and self._accumulated_complementarity_loss is not None:
                _draft_metrics['nld/complementarity_loss'] = self._accumulated_complementarity_loss.item() if isinstance(self._accumulated_complementarity_loss, torch.Tensor) else self._accumulated_complementarity_loss
            if hasattr(self, '_last_complementarity_stats') and self._last_complementarity_stats is not None:
                _draft_metrics['nld/thought_cot_redundancy'] = self._last_complementarity_stats.get('redundancy', 0)
                _draft_metrics['nld/thought_cot_novelty'] = self._last_complementarity_stats.get('novelty', 0)
                _draft_metrics['nld/thought_cot_cos_sim'] = self._last_complementarity_stats.get('cos_sim', 0)
            if hasattr(self, '_accumulated_visual_probe_loss') and self._accumulated_visual_probe_loss is not None:
                _draft_metrics['nld/visual_probe_loss'] = self._accumulated_visual_probe_loss.item() if isinstance(self._accumulated_visual_probe_loss, torch.Tensor) else self._accumulated_visual_probe_loss
            if hasattr(self, '_accumulated_residual_value_loss') and self._accumulated_residual_value_loss is not None:
                _draft_metrics['nld/residual_value_loss'] = self._accumulated_residual_value_loss.item() if isinstance(self._accumulated_residual_value_loss, torch.Tensor) else self._accumulated_residual_value_loss
            if hasattr(self, '_last_residual_stats') and self._last_residual_stats is not None:
                _draft_metrics['nld/thought_kl_divergence'] = self._last_residual_stats.get('kl_divergence', 0)
                _draft_metrics['nld/thought_advantage'] = self._last_residual_stats.get('advantage', 0)
            if hasattr(self, '_accumulated_exit_token_loss') and self._accumulated_exit_token_loss is not None:
                _draft_metrics['nld/exit_token_loss'] = self._accumulated_exit_token_loss.item() if isinstance(self._accumulated_exit_token_loss, torch.Tensor) else self._accumulated_exit_token_loss
            if hasattr(self, '_accumulated_key_token_loss') and self._accumulated_key_token_loss is not None:
                _draft_metrics['nld/key_token_loss'] = self._accumulated_key_token_loss.item() if isinstance(self._accumulated_key_token_loss, torch.Tensor) else self._accumulated_key_token_loss

            # 饱和度和退出原因统计
            if hasattr(self, '_last_exit_stats') and self._last_exit_stats is not None:
                if 'saturations' in self._last_exit_stats and self._last_exit_stats['saturations']:
                    last_sat = self._last_exit_stats['saturations'][-1]
                    _draft_metrics['nld/last_saturation'] = last_sat.mean().item() if isinstance(last_sat, torch.Tensor) else last_sat
                if 'exit_reason' in self._last_exit_stats:
                    # 将退出原因编码为数值: max_steps=0, mlp_gate=1, saturation=2, mlp_or_saturation=3, training=4
                    _reason_map = {'max_steps': 0, 'mlp_gate': 1, 'saturation': 2, 'exit_token': 3, 'mixed_signal': 4, 'training': 5, 'fixed_steps': 6}
                    _draft_metrics['nld/exit_reason'] = float(_reason_map.get(self._last_exit_stats['exit_reason'], -1))

            # Per-token loss 分解
            if loss_weight_mask is not None and labels is not None:
                _shift_labels = labels[:, 1:].contiguous()
                _shift_weights = loss_weight_mask[:, 1:].contiguous().float()
                _valid = (_shift_labels != -100)
                _think_mask = _valid & (_shift_weights <= 1.01)
                _answer_mask = _valid & (_shift_weights > 1.01)

                _loss_fct_none = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                _per_token_ce = _loss_fct_none(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    _shift_labels.view(-1)
                ).view(_shift_labels.shape)

                if _think_mask.any():
                    _draft_metrics['loss/think_ce'] = (_per_token_ce * _think_mask.float()).sum().item() / _think_mask.float().sum().item()
                if _answer_mask.any():
                    _draft_metrics['loss/answer_ce'] = (_per_token_ce * _answer_mask.float()).sum().item() / _answer_mask.float().sum().item()

                # Top-1 准确率
                if _valid.any():
                    _pred_top1 = logits[:, :-1, :].contiguous().argmax(dim=-1)
                    _top1_correct = (_pred_top1 == _shift_labels) & _valid
                    _draft_metrics['acc/top1_overall'] = _top1_correct.float().sum().item() / _valid.float().sum().item()
                    if _answer_mask.any():
                        _draft_metrics['acc/top1_answer'] = (_top1_correct & _answer_mask).float().sum().item() / _answer_mask.float().sum().item()

        # ====== 耗时汇总 ======
        _fwd_end = _time.time()
        _fwd_total = _fwd_end - _fwd_start
        _should_profile = (_is_rank0 and self._fwd_count % 50 == 0)
        if _should_profile or _fwd_total > 30:
            _t_vision = _t_vision_end - _t_vision_start
            _t_base = _t_base_fwd_end - _t_base_fwd_start
            print(f"[rank {_rank}] fwd#{self._fwd_count} 耗时: "
                  f"total={_fwd_total:.1f}s (vision={_t_vision:.1f}s, "
                  f"chunkwise={_t_base:.1f}s) "
                  f"chunks={len(chunk_ranges)} thoughts={thought_count}", flush=True)

        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'div_loss': torch.tensor(0.0, device=device),  # NLD 无 diversity loss
            'commit_loss': torch.tensor(0.0, device=device),
            'grounding_loss': torch.tensor(0.0, device=device),
            'logits': None,
            'draft_metrics': _draft_metrics,
        }

    # ================================================================
    # 推理流程
    # ================================================================

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
        推理时的生成
        
        核心设计 (统一序列, 训推一致):
        - 正常自回归生成 CoT
        - 检测到 step boundary 时触发 NativeLatentThinker
        - 方案 B (训推一致): thinker 使用局部 cache，thought_output 作为 prefix
          再过一遍 text_model() 写入主 cache (与训练时行为完全一致)
        - thought tokens 的 KV 永久保留在历史 cache 中 (不剥离)
        - 后续所有 token 的 attention 自然看到 thought 信息
        - 隐式推理 + 自然语言 CoT 构成完整的推理序列
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. 视觉 encoder
        inner_model = self._inner_model
        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(_pixel_values, grid_thw=image_grid_thw)
        merged_hidden_states, gen_deepstack_features = _extract_visual_output(vision_output)
        split_sizes = (
            image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
        ).tolist()
        image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))
        cached_visual_embeds = torch.cat(image_embeds_list, dim=0)

        # 2. Prefill
        text_model = inner_model.language_model
        lm_head = self._lm_head

        prompt_embeds = inner_model.get_input_embeddings()(input_ids)
        image_token_id = inner_model.config.image_token_id
        image_mask = (input_ids == image_token_id)
        num_image_tokens = image_mask.sum().item()
        if num_image_tokens > 0:
            img_emb = cached_visual_embeds[:num_image_tokens].to(prompt_embeds.device, prompt_embeds.dtype)
            image_mask_3d = image_mask.unsqueeze(-1).expand_as(prompt_embeds)
            prompt_embeds = prompt_embeds.masked_scatter(image_mask_3d, img_emb)

        seg_token_mask = torch.ones(B, input_ids.shape[1], device=device, dtype=torch.long)
        inner_model.rope_deltas = None
        position_ids, rope_deltas = inner_model.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, video_grid_thw=None,
            attention_mask=seg_token_mask,
        )
        inner_model.rope_deltas = rope_deltas

        prompt_len = input_ids.shape[1]
        if position_ids is not None and position_ids.dim() == 3:
            text_pos = torch.arange(prompt_len, device=device).unsqueeze(0).expand(B, -1)
            position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

        visual_pos_masks = (input_ids == image_token_id) if pixel_values is not None else None
        cache_position = torch.arange(prompt_len, device=device)

        cache = DynamicCache()
        prefill_out = text_model(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=gen_deepstack_features,
        )
        past_key_values = prefill_out.past_key_values
        mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None

        prefill_hidden = prefill_out.last_hidden_state
        next_token_logits = lm_head(prefill_hidden[:, -1, :])

        # 3. 自回归生成
        generated_ids = input_ids.clone()
        eos_token_id = self.processor.tokenizer.eos_token_id if hasattr(self, 'processor') else None

        # <|latent|> 触发计数 (每个样本最多触发一定次数)
        latent_trigger_count = [0 for _ in range(B)]

        # 收集最近的 hidden states (用于 NativeLatentThinker)
        # 保留最近的 hidden state (用于 NativeLatentThinker)
        recent_last_hidden = prefill_hidden[:, -1:, :]  # [B, 1, H] 最后一个 token

        current_pos = prompt_len

        for gen_step in range(max_new_tokens):
            # 采样
            if do_sample and temperature > 0:
                scaled_logits = next_token_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            next_token = next_token.squeeze(-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # ---- 标准自回归 forward ----
            # 每个 token 直接过 text_model()，KV cache 自动管理
            # 如果之前有 thought tokens，它们的 KV 已在 past_key_values 中
            # 当前 token 的 attention 自然能看到 thought tokens
            new_mrope_pos = mrope_last_pos + 1
            text_pos_gen = torch.full((B, 1), current_pos, device=device, dtype=torch.long)
            position_ids_4d = torch.cat([text_pos_gen.unsqueeze(0), new_mrope_pos], dim=0)

            token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))

            current_total_len = current_pos + 1
            new_attention_mask = torch.ones(B, current_total_len, device=device, dtype=attention_mask.dtype)
            cache_position_gen = torch.tensor([current_pos], device=device, dtype=torch.long)

            text_outputs = text_model(
                inputs_embeds=token_embeds,
                attention_mask=new_attention_mask,
                position_ids=position_ids_4d,
                past_key_values=past_key_values,
                cache_position=cache_position_gen,
                use_cache=True,
            )
            past_key_values = text_outputs.past_key_values
            new_hidden = text_outputs.last_hidden_state  # [B, 1, H]
            next_token_logits = lm_head(new_hidden[:, -1, :])

            # 更新最近的 hidden state
            recent_last_hidden = new_hidden  # [B, 1, H]

            mrope_last_pos = new_mrope_pos
            current_pos += 1

            # ---- Step boundary 检测 (通过 <|latent|> token id) ----
            trigger_mask = torch.zeros(B, dtype=torch.bool, device=device)
            for b in range(B):
                token_id = next_token[b].item()
                # 检测 <|latent|> token: 直接比较 token id (精确、高效)
                if hasattr(self, 'latent_token_id') and self.latent_token_id is not None:
                    if token_id == self.latent_token_id:
                        trigger_mask[b] = True

            if trigger_mask.any():
                # 触发 NativeLatentThinker (Coconut 等价的隐空间推理)
                # 方案 B (训推一致): thinker 使用局部 cache，thought_output 作为 prefix
                # 再过一遍 text_model() 写入主 cache
                last_hidden = recent_last_hidden  # [B, 1, H]
                
                # 第一步 thought 的位置和 position_ids
                base_thought_pos = current_pos
                thought_pos = torch.arange(current_pos, current_pos + 1, device=device)
                thought_mrope_pos = thought_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                
                # 第一步的 position_ids: [4, B, 1]
                thought_text_pos = torch.full((B, 1), base_thought_pos, device=device, dtype=torch.long)
                thought_position_ids = torch.cat([thought_text_pos.unsqueeze(0), thought_mrope_pos], dim=0)
                
                # 第一步的 attention_mask
                think_attn_mask = torch.ones(B, base_thought_pos + 1, device=device, dtype=torch.long)
                
                thinker_result = self.latent_thinker(
                    last_hidden=last_hidden,
                    text_model=text_model,
                    past_key_values=past_key_values,
                    attention_mask=think_attn_mask,
                    cache_position=thought_pos,
                    position_ids=thought_position_ids,
                    rotary_emb_fn=text_model.rotary_emb,
                    mrope_position_ids=thought_mrope_pos,
                    base_thought_pos=base_thought_pos,
                    B=B,
                    lm_head=lm_head,
                )
                thought_output = thinker_result['thought_output']  # [B, num_steps, H]
                num_thought_steps = thinker_result.get('num_thought_steps', 1)
                
                # 方案 B (训推一致): thought_output 作为 prefix 过一遍 text_model()
                # 写入主 cache，与训练时的行为完全一致
                # 构建 thought prefix 的 position_ids 和 cache_position
                thought_prefix_cache_pos = torch.arange(
                    current_pos, current_pos + num_thought_steps, device=device
                )
                thought_prefix_mrope = thought_prefix_cache_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                thought_prefix_text_pos = torch.arange(
                    current_pos, current_pos + num_thought_steps, device=device
                ).unsqueeze(0).expand(B, -1)
                thought_prefix_pos_ids = torch.cat([
                    thought_prefix_text_pos.unsqueeze(0),
                    thought_prefix_mrope,
                ], dim=0)  # [4, B, num_steps]
                
                thought_prefix_attn_mask = torch.ones(
                    B, current_pos + num_thought_steps, device=device, dtype=attention_mask.dtype
                )
                
                # 把 thought_output 作为 inputs_embeds 过 text_model()
                # 这会把 thought prefix 的 KV 写入主 cache
                prefix_outputs = text_model(
                    inputs_embeds=thought_output,
                    attention_mask=thought_prefix_attn_mask,
                    position_ids=thought_prefix_pos_ids,
                    past_key_values=past_key_values,
                    cache_position=thought_prefix_cache_pos,
                    use_cache=True,
                )
                past_key_values = prefix_outputs.past_key_values
                
                # 更新位置以反映 thought tokens 占据的位置
                current_pos += num_thought_steps
                mrope_last_pos = mrope_last_pos + num_thought_steps

                # 更新触发计数
                for b in range(B):
                    if trigger_mask[b]:
                        latent_trigger_count[b] += 1

        return generated_ids

    # ================================================================
    # 保存/加载
    # ================================================================

    def save_pretrained(self, save_dir: str):
        """保存全量模型: NativeLatentThinker + VLM 全量参数"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存 NativeLatentThinker
        thinker_path = os.path.join(save_dir, "nld_latent_thinker.pt")
        torch.save(self.latent_thinker.state_dict(), thinker_path)
        if self._verbose:
            print(f"[NLD] NativeLatentThinker 已保存到 {thinker_path}")
        
        # 保存全量 VLM 模型 (使用 HuggingFace save_pretrained)
        vlm_dir = os.path.join(save_dir, "vlm_full")
        self.base_model.save_pretrained(vlm_dir)
        if self._verbose:
            print(f"[NLD] VLM 全量模型已保存到 {vlm_dir}")

    def load_pretrained(self, save_dir: str):
        """加载全量模型: NativeLatentThinker + VLM 全量参数"""
        # 加载 NativeLatentThinker
        thinker_path = os.path.join(save_dir, "nld_latent_thinker.pt")
        if os.path.exists(thinker_path):
            state_dict = torch.load(thinker_path, map_location="cpu")
            missing, unexpected = self.latent_thinker.load_state_dict(state_dict, strict=False)
            if self._verbose:
                print(f"[NLD] NativeLatentThinker 已从 {thinker_path} 加载")
                if missing:
                    print(f"[NLD]   新增参数: {missing}")
                if unexpected:
                    print(f"[NLD]   旧参数: {unexpected}")
        else:
            if self._verbose:
                print(f"[NLD] ⚠️ 未找到 NativeLatentThinker 权重: {thinker_path}")

        # 对齐 dtype/device
        target_dtype = next(self.base_model.parameters()).dtype
        target_device = next(self.base_model.parameters()).device
        self.latent_thinker = self.latent_thinker.to(dtype=target_dtype, device=target_device)
        
        # 加载全量 VLM 模型
        vlm_dir = os.path.join(save_dir, "vlm_full")
        if os.path.exists(vlm_dir):
            import safetensors.torch
            import glob
            safetensor_files = glob.glob(os.path.join(vlm_dir, "*.safetensors"))
            if safetensor_files:
                vlm_state = {}
                for sf in safetensor_files:
                    vlm_state.update(safetensors.torch.load_file(sf))
                missing, unexpected = self.base_model.load_state_dict(vlm_state, strict=False)
                if self._verbose:
                    print(f"[NLD] VLM 全量模型已加载: {vlm_dir}")
                    if missing:
                        print(f"[NLD]   缺失参数: {len(missing)} 个")
                    if unexpected:
                        print(f"[NLD]   多余参数: {len(unexpected)} 个")
        

