"""RLD Model: 包装 Qwen3-VL VLM 的 Reflective Latent Draft 模型

重构版架构: 单次全序列前向 + delimiter 门控的 recurrent draft + readout adapter

核心设计:
1. 冻结 Qwen3-VL 全部参数
2. 只训练 RLD Controller + DraftReadoutAdapter 的外挂模块 (< 30M 参数)
3. 训练时:
   a. 对完整序列只跑一次 teacher-forcing forward (no_grad)
   b. 从 hidden states 按 </step> 边界切出 step summaries
   c. recurrent controller 扫描 → Z_d_0, Z_d_1, ..., Z_d_C
   d. DraftReadoutAdapter(H, Z_d_expanded) → adapted_hidden → logits
   e. 全局 CE loss
4. 推理时也用 readout adapter (训推一致)
   a. 每生成一个 token，用 readout_adapter 修正 hidden → logits
   b. 遇到 </step> 时更新 Z_d (controller.step_update)
   c. 无需 prefix KV cache，无需 hook

与旧版段循环方案的区别:
- 不再有分段多次 forward
- 不再有训练时 cache rewrite
- 训推一致: 训练和推理都使用 readout adapter
- 所有 rank 的调用图完全一致 (彻底消除 cross-rank CUDA error)
- 速度提升 ~20x (从 ~100s/step 降到 ~4s/step)
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache as _RawDynamicCache

# Qwen3-VL 版本: 使用 Qwen3VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .controller import RLDController
from .modules import DraftReadoutAdapter, MultiLayerDraftReadout, InSituDraftInjector

# ============================================================
# DynamicCache 兼容层: 仍保留给推理路径使用
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


class RLDModel(nn.Module):
    """
    RLD Model: Qwen3-VL + Reflective Latent Draft Controller
    
    重构版架构 (训练时单次全序列前向 + readout adapter):
    ┌──────────────────────────────────────────────────────────────┐
    │  训练流程:                                                    │
    │  1. 视觉 encoder → visual_tokens → Z_e (证据槽)              │
    │  2. 冻结 base model 对完整序列做一次 forward (no_grad)        │
    │     → H_{1:T} (全序列 hidden states)                        │
    │  3. 按 </step> 边界从 H 切出 step summaries → S_c   │
    │  4. Recurrent controller 扫描 → Z_d_0, Z_d_1, ..., Z_d_C
    │  5. 将 Z_d 按 step 展开到 token 级 → [B, T, K_d, d_z]
    │  6. DraftReadoutAdapter(H, Z_d_expanded) → adapted_hidden → logits
    │  7. 全局 CE loss
    │                                                               │
    │  推理流程 (训推一致):                                          │
    │  - 标准 KV cache 自回归 + 每 token readout adapter 修正        │
    │  - 遇到 </step> 时 controller.step_update 更新 Z_d             │
    │  - 无需 prefix KV, 无需 hook, 无需多次 forward                │
    └──────────────────────────────────────────────────────────────┘
    """

    STEP_DELIMITER = "</step>"

    def __init__(
        self,
        model_path: str,
        hidden_size: int = 4096,
        d_z: int = 768,
        num_evidence_slots: int = 16,
        num_draft_slots: int = 16,
        num_trace_slots: int = 16,
        total_layers: int = 36,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        lambda_div: float = 0.01,
        lambda_kl: float = 0.1,
        max_scale: float = 0.3,
        selective_injection: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.total_layers = total_layers
        self.lambda_kl = lambda_kl
        self.max_scale = max_scale
        self.selective_injection = selective_injection
        self._verbose = _is_main_process()
        # _debug: 仅在 RLD_DEBUG=1 且主进程时启用重量级调试 (含额外 lm_head 计算)
        # ⚠️ 调试模式会导致 rank 0 比其他 rank 慢 20-40s, 可能触发 NCCL timeout
        self._debug = self._verbose and os.environ.get('RLD_DEBUG', '0') == '1'

        # ====== 注意力实现: 支持通过环境变量 RLD_ATTN_IMPL 动态切换 ======
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
            use_gate=True,  # [已废弃] 保留兼容旧配置
            lambda_div=lambda_div,
            max_steps=14,
        )

        # 确保 controller 使用与基座相同的 dtype
        self.controller = self.controller.to(dtype=torch_dtype)

        # ====== 3. 创建 InSituDraftInjector (Top-K Rerun 方案, Phase 3: K=8, 3注入点) ======
        # Phase 3: K=8, 重算最后 8 层 (L28-L35), 3 个注入点 (L28, L31, L35)
        # 修正后的 hidden 在 L28-L35 层间通过 self-attention 传播,
        # 实现 "draft 真正影响后续 token 生成" 的因果一致性。
        self.rerun_k = 8  # Phase 3: 重算最后 8 层
        self.readout_adapter = InSituDraftInjector(
            hidden_size=hidden_size,
            d_z=d_z,
            num_heads=8,
            total_layers=total_layers,
            rerun_k=self.rerun_k,
            injection_offsets=[0, 3, 7],  # Phase 3: L28, L31, L35
            max_scale=max_scale,
        )
        self.readout_adapter = self.readout_adapter.to(dtype=torch_dtype)

        if self._verbose:
            self.controller.print_param_summary()
            adapter_params = sum(p.numel() for p in self.readout_adapter.parameters() if p.requires_grad)
            print(f"[RLD] InSituDraftInjector 参数: {adapter_params:,} ({adapter_params/1e6:.2f}M)")
            print(f"[RLD]   Top-K Rerun: K={self.rerun_k}, rerun_start=L{self.readout_adapter.rerun_start}")
            print(f"[RLD]   注入层索引: {self.readout_adapter.injection_layer_indices}")
            print(f"[RLD]   Scale 上界: {max_scale}")
            print(f"[RLD]   KL 散度正则化: lambda_kl={lambda_kl}")
            print(f"[RLD]   选择性注入: {selective_injection}")
            for idx_str, adapter in self.readout_adapter.adapters.items():
                print(f"[RLD]   Layer {idx_str}: scale={adapter.scale.item():.2f}")

        # ====== 4. Step delimiter token ids ======
        self.step_delimiter_ids = None
        self.step_delimiter_id = None

    def set_processor(self, processor):
        """设置 processor 并获取 delimiter token id 序列"""
        self.processor = processor
        tokenizer = processor.tokenizer

        delimiter_ids = tokenizer.encode(self.STEP_DELIMITER, add_special_tokens=False)
        self.step_delimiter_ids = delimiter_ids
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
        """安全获取 KV cache 序列长度 (推理用)"""
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, 'key_cache') and len(past_key_values.key_cache) > 0:
            kc = past_key_values.key_cache[0]
            if kc is not None and kc.dim() == 4:
                return kc.shape[2]
        return past_key_values.get_seq_length() if past_key_values is not None else 0

    def _find_delimiter_positions(self, token_ids: torch.Tensor) -> List[int]:
        """在 token 序列中查找所有 </step> delimiter 的末尾位置"""
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
        """从 Qwen3-VL 的视觉编码器获取视觉 token 序列"""
        inner_model = self.base_model.model

        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(
            _pixel_values, grid_thw=image_grid_thw
        )
        merged_hidden_states, deepstack_features = _extract_visual_output(vision_output)

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

    # ================================================================
    # 训练流程: 重构版 — 单次全序列前向 + readout adapter
    # ================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        step_boundaries: Optional[List[List[int]]] = None,
        prompt_lens: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播 (重构版: 单次全序列前向 + readout adapter)
        
        核心流程:
          1. 视觉 encoder → visual_tokens → Z_e (no_grad)
          2. 冻结 base model 对完整序列做一次 forward (no_grad) → H_{1:T}
          3. 按 </step> 边界从 H 切出 step summaries → S_c
          4. recurrent controller 扫描 → Z_d_0, Z_d_1, ..., Z_d_C
          5. 将 Z_d 按 step 展开到 token 级 → [B, T, K_d, d_z]
          6. DraftReadoutAdapter(H, Z_d_expanded) → adapted_hidden → logits
          7. 全局 CE loss
        
        优势:
        - 只有 1 次 base forward (no_grad, 不持有计算图)
        - 所有 rank 调用图完全一致
        - 没有 cache 操作
        - 梯度路径: loss → readout_adapter → Z_d → controller
        """
        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device
        ctrl_dtype = next(self.controller.parameters()).dtype

        _rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))

        # ====== forward 阶段耗时监控: 精确定位 hang 在哪个阶段 ======
        import time as _time
        _fwd_start = _time.time()
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        # 每 10 次 forward 且仅 rank 0 打印; 超 30s 异常时所有 rank 打印
        _is_rank0 = (_rank == 0)
        _should_profile = (_is_rank0 and self._fwd_count % 10 == 0)
        _force_profile = False

        # ====== 显存监控: 帮助诊断 cudaErrorContained (OOM → NVLink 越界) ======
        if self._debug and device.type == 'cuda':
            if not hasattr(self, '_mem_monitor_count'):
                self._mem_monitor_count = 0
            self._mem_monitor_count += 1
            if self._mem_monitor_count <= 2 or self._mem_monitor_count % 100 == 0:
                _allocated = torch.cuda.memory_allocated(device) / 1024**3
                _reserved = torch.cuda.memory_reserved(device) / 1024**3
                _max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
                print(f"\n📊 [rank {_rank}] forward 入口显存: "
                      f"seq_len={seq_len}, "
                      f"已分配={_allocated:.2f}GB, "
                      f"已预留={_reserved:.2f}GB, "
                      f"峰值={_max_allocated:.2f}GB",
                      flush=True)

        # ====== 1. 视觉 encoder (no_grad，只跑一次) ======
        _t_vision_start = _time.time()
        with torch.no_grad():
            cached_visual_embeds = None
            deepstack_features = None

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
            else:
                visual_tokens = torch.zeros(
                    B, 1, self.hidden_size, device=device, dtype=ctrl_dtype
                )

        _t_vision_end = _time.time()

        # ====== 2. 初始化 RLD 状态 (controller.prefill 有梯度) ======
        rld_state = self.controller.prefill(visual_tokens)

        # ====== 3. 一次性完整 base forward (no_grad) ======
        _t_base_fwd_start = _time.time()
        with torch.no_grad():
            inner_model = self.base_model.model
            text_model = inner_model.language_model
            lm_head = self.base_model.lm_head

            # 3a. Embedding + 视觉融合
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

            # 3b. 构建 visual_pos_masks (DeepStack 需要)
            visual_pos_masks = None
            deepstack_for_fwd = None
            if pixel_values is not None and image_grid_thw is not None:
                image_token_id = inner_model.config.image_token_id
                visual_pos_masks = (input_ids == image_token_id)
                deepstack_for_fwd = deepstack_features

            # 3c. 获取 mRoPE position_ids
            seg_token_mask = torch.ones(B, seq_len, device=device, dtype=torch.long)
            inner_model.rope_deltas = None
            position_ids, rope_deltas = inner_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw if pixel_values is not None else None,
                video_grid_thw=None,
                attention_mask=seg_token_mask,
            )
            inner_model.rope_deltas = rope_deltas

            # 构造 4D position_ids: [4, B, seq_len]
            if position_ids is not None and position_ids.dim() == 3:
                text_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
                position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

            # 3d. 手动展开 text_model forward, 收集最后层 hidden + 保存 rerun 入口点
            # Top-K Rerun 方案: Pass 1 跑完全 36 层 (no_grad), 保存 rerun_start-1 层的输出
            # Pass 2 将用有梯度的方式重算最后 K 层 + adapter 注入
            cache_position = torch.arange(seq_len, device=device)

            # ---- 复现 Qwen3VLTextModel.forward() 的内部逻辑 ----
            # (与原生实现严格一致, 参见 modeling_qwen3_vl.py Qwen3VLTextModel.forward)

            # position_ids 处理 (与原生一致)
            _position_ids = position_ids
            if _position_ids is not None and _position_ids.ndim == 3 and _position_ids.shape[0] == 4:
                _text_position_ids = _position_ids[0]
                _mrope_position_ids = _position_ids[1:]
            elif _position_ids is not None:
                _text_position_ids = _position_ids[0]
                _mrope_position_ids = _position_ids
            else:
                _text_position_ids = cache_position.view(1, 1, -1).expand(3, B, -1)[0]
                _mrope_position_ids = cache_position.view(1, 1, -1).expand(3, B, -1)

            from transformers.masking_utils import create_causal_mask
            _attention_mask = create_causal_mask(
                config=text_model.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=_text_position_ids,
            )

            hidden_states = inputs_embeds
            position_embeddings = text_model.rotary_emb(hidden_states, _mrope_position_ids)

            # 收集 all_hidden_states: 索引 0 = embedding 输出, 索引 i+1 = layer i 输出
            all_hidden_states_list = [hidden_states]  # 索引 0: embedding 层输出

            # Top-K Rerun: 保存 rerun 起点 (rerun_start 前一层的输出)
            rerun_start = self.readout_adapter.rerun_start  # Phase 3: 28 (= 36 - 8)
            rerun_entry_hidden = None  # 将在 layer rerun_start-1 (L27) 之后保存

            for layer_idx, decoder_layer in enumerate(text_model.layers):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=_attention_mask,
                    position_ids=_text_position_ids,
                    past_key_values=None,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                # DeepStack: 与原生实现一致
                if deepstack_for_fwd is not None and layer_idx in range(len(deepstack_for_fwd)):
                    hidden_states = text_model._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_for_fwd[layer_idx],
                    )

                # Top-K Rerun: 保存 rerun 起点
                # rerun_start-1 层的输出就是 Pass 2 的输入
                if layer_idx == rerun_start - 1:
                    rerun_entry_hidden = hidden_states.clone()

                # 收集 hidden states (保持索引对齐)
                all_hidden_states_list.append(None)  # 占位

            # 最终 norm (完整 36 层的结果, 用于 step summaries)
            hidden_states = text_model.norm(hidden_states)
            all_hidden_states_list.append(hidden_states)  # 索引 L+1: norm 后的输出

            full_hidden = hidden_states  # [B, seq_len, hidden_size], 用于 step summaries
            all_hidden_states = tuple(all_hidden_states_list)

        _t_base_fwd_end = _time.time()

        # ====== 显存监控: base forward 完成后 ======
        if self._debug and device.type == 'cuda' and hasattr(self, '_mem_monitor_count') and (self._mem_monitor_count <= 2 or self._mem_monitor_count % 100 == 0):
            _allocated = torch.cuda.memory_allocated(device) / 1024**3
            _reserved = torch.cuda.memory_reserved(device) / 1024**3
            _max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"📊 [rank {_rank}] base forward 完成后显存: "
                  f"已分配={_allocated:.2f}GB, "
                  f"已预留={_reserved:.2f}GB, "
                  f"峰值={_max_allocated:.2f}GB",
                  flush=True)

        # ====== 调试: 验证 rerun 入口已保存 ======
        if self._debug and (not hasattr(self, '_debug_collect_count') or self._debug_collect_count < 2):
            if not hasattr(self, '_debug_collect_count'):
                self._debug_collect_count = 0
            self._debug_collect_count += 1
            print("\n" + "=" * 70)
            print(f"🔍 [RLDModel.forward] Top-K Rerun 入口验证 (#{self._debug_collect_count})")
            print(f"   总层数: {self.total_layers}, rerun_k={self.rerun_k}, rerun_start=L{rerun_start}")
            print(f"   注入层索引: {self.readout_adapter.injection_layer_indices}")
            if rerun_entry_hidden is not None:
                print(f"   ✅ rerun_entry_hidden (L{rerun_start-1}输出): shape={list(rerun_entry_hidden.shape)}, "
                      f"norm={rerun_entry_hidden.detach().float().norm(dim=-1).mean().item():.4f}")
            else:
                print(f"   ❌ rerun_entry_hidden 为 None! rerun_start={rerun_start}")
            print("=" * 70)

        # ====== 4. 自动检测 step 边界 ======
        if prompt_lens is not None:
            gen_start = max(prompt_lens)
        else:
            gen_start = 0

        if step_boundaries is None and self.step_delimiter_ids is not None:
            step_boundaries = []
            for b in range(B):
                positions = self._find_delimiter_positions(input_ids[b])
                positions = [p for p in positions if p >= gen_start]
                step_boundaries.append(positions)

        # ====== 5. 统一 step boundaries 并按 step 切出 hidden → summaries ======
        # 收集所有 step 边界 (跨 batch 取并集，确保所有 rank 的循环次数一致)
        all_boundaries = set()
        if step_boundaries:
            for b_list in step_boundaries:
                all_boundaries.update(b_list)
        split_points = sorted([p for p in all_boundaries if gen_start < p < seq_len])

        per_sample_sets = [set(bl) for bl in step_boundaries] if step_boundaries else [set() for _ in range(B)]

        # 限制最大 step 数 (保持合理的 controller scan 长度)
        MAX_STEPS = 14
        if len(split_points) > MAX_STEPS:
            split_points = split_points[:MAX_STEPS]

        # 构建段: 每个段的 hidden 用于计算 step summary
        segments = []
        prev = gen_start
        for sp in split_points:
            segments.append((prev, sp + 1))
            prev = sp + 1
        # 最后一段不需要作为 step (它之后没有需要更新的内容)

        # 为每个 step 计算 summary S_c 和 update_mask
        step_summaries = []
        update_masks = []

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            sp = split_points[seg_idx]

            # 累积 hidden: 从序列起始到当前 step 边界
            # 简化版: 只取当前段的 hidden (与旧版 per_sample_hidden_buffers 等价)
            seg_hidden = full_hidden[:, seg_start:seg_end, :]  # [B, seg_len, hidden_size]

            # StepResampler: seg_hidden → S_c
            S_c = self.controller.step_resampler(seg_hidden)  # [B, K_t, d_z]
            step_summaries.append(S_c)

            # Per-sample update mask
            mask = torch.tensor(
                [sp in per_sample_sets[b] for b in range(B)],
                dtype=torch.bool, device=device,
            )
            update_masks.append(mask)

        # ====== 6. Recurrent controller scan ======
        div_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)

        if len(step_summaries) > 0:
            final_state, all_Z_d = self.controller.scan_steps(
                rld_state, step_summaries, update_masks
            )
            # 每个 step 的 diversity loss (同时惩罚 Z_d 和 Z_e 的坍塌)
            Z_e_for_div = rld_state['Z_e']  # Z_e 不随 step 变化
            for Z_d_c in all_Z_d[1:]:  # 跳过 Z_d_0
                tmp_state = {'Z_d': Z_d_c, 'Z_e': Z_e_for_div}
                div_loss = div_loss + self.controller.compute_diversity_loss(tmp_state)
            div_loss = div_loss / max(len(all_Z_d) - 1, 1)
            rld_state = final_state
        else:
            all_Z_d = [rld_state['Z_d']]

        # ====== 7. 将 step-level Z_d 展开到 token 级 ======
        # 每个 token 使用其所属 step 的 draft state
        # Z_d_expanded: [B, seq_len, K_d, d_z]
        K_d = all_Z_d[0].shape[1]

        # 建立 token → step 映射
        # step_indices[t] = 哪个 Z_d (0 = Z_d_0, 1 = Z_d_1, ...)
        step_indices = torch.zeros(seq_len, dtype=torch.long, device=device)
        for seg_idx, sp in enumerate(split_points):
            # sp 之后的 token 使用 Z_d_{seg_idx+1}
            if sp + 1 < seq_len:
                step_indices[sp + 1:] = seg_idx + 1

        # 堆叠所有 Z_d: [C+1, B, K_d, d_z]
        Z_d_stack = torch.stack(all_Z_d, dim=0)  # [C+1, B, K_d, d_z]

        # 按 step_indices 展开: [seq_len] → index into [C+1, ...]
        # Z_d_expanded: [B, seq_len, K_d, d_z]
        Z_d_expanded = Z_d_stack[step_indices]  # [seq_len, B, K_d, d_z]
        Z_d_expanded = Z_d_expanded.permute(1, 0, 2, 3)  # [B, seq_len, K_d, d_z]

        # ====== 8. Pass 2: Top-K Rerun — 有梯度重算最后 K 层 + in-situ adapter 注入 ======
        # 核心思想: 从 rerun_entry_hidden (L{rerun_start-1}的输出) 出发,
        # 重新跑最后 K 层 decoder layer, 在注入点插入 adapter。
        # 修正后的 hidden 继续参与后续层的 self-attention → 真正的因果一致性。
        #
        # Pass 2 的梯度路径: loss → lm_head → norm → rerun layers → adapter → Z_d → controller
        # 冻结层的参数 requires_grad=False, 但 hidden states 有梯度 (从 adapter 注入)。
        _t_pass2_start = _time.time()

        # ====== 源头选择性注入: 在 Pass 2 rerun 之前计算 token 级置信度权重 ======
        # 核心思想: 基座模型已经很确定的 token 不需要 draft 修正,
        # 在 adapter 注入层就直接做 token 级加权, 而非事后衰减。
        # 这样 KV cache 中也不会包含不必要的修正信息。
        _inject_weight = None  # [B, seq_len, 1] 或 None
        if self.selective_injection and labels is not None:
            with torch.no_grad():
                # 基座模型原始 logits (用于计算置信度)
                _base_logits_for_conf = lm_head(full_hidden)  # [B, seq_len, V]
                base_probs = F.softmax(_base_logits_for_conf, dim=-1)
                # 基座 top-1 概率: 越高说明越确定, 越不需要修正
                base_confidence = base_probs.max(dim=-1).values  # [B, seq_len]
                # 注入权重: 基座越不确定 → 权重越大 (1.0), 基座很确定 → 权重趋近 0
                # min=0.05: 即使基座 100% 确定, 也保留 5% 的微弱注入 (防止完全断开梯度)
                _inject_weight = (1.0 - base_confidence).clamp(min=0.05).unsqueeze(-1)  # [B, seq_len, 1]
                del base_probs  # 立即释放显存
                # 注意: _base_logits_for_conf 保留, 供后续 KL 散度复用

        # 从 rerun 入口点出发, detach 切断与 Pass 1 的梯度连接
        rerun_hidden = rerun_entry_hidden.detach()

        # 注意: position_embeddings, _attention_mask, _text_position_ids, cache_position
        # 都是在 no_grad 块中定义的局部变量, 它们不需要梯度 (纯位置信息)
        for layer_idx in range(rerun_start, self.total_layers):
            decoder_layer = text_model.layers[layer_idx]
            # 冻结层 forward: 参数 requires_grad=False, 但输入 hidden 有梯度
            # → 输出 hidden 有梯度 (因为是 hidden 的函数)
            rerun_hidden = decoder_layer(
                rerun_hidden,
                attention_mask=_attention_mask,
                position_ids=_text_position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            # DeepStack: 通常不会触发 (DeepStack 只影响前几层)
            if deepstack_for_fwd is not None and layer_idx in range(len(deepstack_for_fwd)):
                rerun_hidden = text_model._deepstack_process(
                    rerun_hidden,
                    visual_pos_masks,
                    deepstack_for_fwd[layer_idx],
                )

            # ★ In-situ adapter 注入 (源头选择性): inject_weight 在 adapter 内部直接加权修正量
            # 基座确定的 token → 修正量被压缩 → KV 中几乎不含 draft 信息
            # 基座不确定的 token → 修正量充分保留 → draft 在关键位置发力
            rerun_hidden = self.readout_adapter.inject(
                layer_idx, rerun_hidden, Z_d_expanded, inject_weight=_inject_weight
            )

        # 最终 norm (有梯度)
        adapted_hidden = text_model.norm(rerun_hidden)

        # ====== 基座 logits 缓存 (供 KL 散度复用) ======
        _cached_base_logits = None
        if self.lambda_kl > 0 and labels is not None:
            if self.selective_injection and _inject_weight is not None:
                # 复用选择性注入阶段已计算的基座 logits (避免重复 lm_head 计算)
                _cached_base_logits = _base_logits_for_conf
            else:
                with torch.no_grad():
                    _cached_base_logits = lm_head(full_hidden)  # [B, seq_len, V]

        _t_pass2_end = _time.time()

        # ====== 调试: Pass 2 注入前后 hidden 对比 (证明 draft 影响了 CoT) ======
        # ⚠️ 仅在 RLD_DEBUG=1 时执行, 包含额外 lm_head 计算, 会导致 rank 0 显著变慢
        if self._debug and (not hasattr(self, '_debug_inject_count') or self._debug_inject_count < 1):
            if not hasattr(self, '_debug_inject_count'):
                self._debug_inject_count = 0
            self._debug_inject_count += 1
            with torch.no_grad():
                _diff = (adapted_hidden - full_hidden.to(adapted_hidden.dtype)).float()
                _diff_norm = _diff.norm(dim=-1)  # [B, seq_len]
                _orig_norm = full_hidden.float().norm(dim=-1)  # [B, seq_len]

                # 对比注入前后的 logits (取前 50 个 token 采样, 减少开销)
                _sample_len = min(50, seq_len)
                _logits_original = self.base_model.lm_head(full_hidden[:, :_sample_len, :])  # [B, sample, V]
                _logits_adapted = self.base_model.lm_head(adapted_hidden[:, :_sample_len, :])  # [B, sample, V]
                _logit_diff = (_logits_adapted - _logits_original).float().abs().mean().item()

                # top-1 token 变化率
                _top1_orig = _logits_original.argmax(dim=-1)  # [B, sample]
                _top1_adapt = _logits_adapted.argmax(dim=-1)  # [B, sample]
                _top1_changed = (_top1_orig != _top1_adapt).float().mean().item()

                # 立即释放大 tensor
                del _logits_original, _logits_adapted

                print("\n" + "=" * 70)
                print(f"🎯 [RLDModel.forward] Top-K Rerun Draft 注入效果验证 (#{self._debug_inject_count})")
                print(f"   序列长度: {seq_len}, Step 数: {len(split_points)}, Z_d 版本数: {len(all_Z_d)}")
                print(f"   Rerun K={self.rerun_k}, 注入层: {self.readout_adapter.injection_layer_indices}")
                print(f"   Pass 2 耗时: {_t_pass2_end - _t_pass2_start:.2f}s")
                print(f"   ── Hidden 修正量 ──")
                print(f"   修正量 L2 范数 (全序列均值): {_diff_norm.mean().item():.6f}")
                print(f"   修正量 L2 范数 (最大值):     {_diff_norm.max().item():.6f}")
                print(f"   原始 hidden L2 范数均值:      {_orig_norm.mean().item():.4f}")
                print(f"   修正/原始比:                  {_diff_norm.mean().item() / max(_orig_norm.mean().item(), 1e-8):.6f}")
                print(f"   ── Logits 影响 ──")
                print(f"   Logit 绝对差均值 (前{_sample_len}token): {_logit_diff:.6f}")
                print(f"   Top-1 token 变化率:           {_top1_changed*100:.1f}%")
                if _diff_norm.mean().item() > 1e-8:
                    print(f"   ✅ Top-K Rerun Draft 注入正在影响 CoT: hidden 被修正, logits 已改变")
                else:
                    print(f"   ⚠️ Draft 注入量极小, 可能尚未学到有效修正")
                print("=" * 70)

        # ====== 9. 计算 logits ======
        lm_head = self.base_model.lm_head
        logits = lm_head(adapted_hidden)  # [B, seq_len, V]

        # ====== 10. 计算 loss ======
        total_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        main_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        kl_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)

        if labels is not None:
            # 标准 next-token prediction: shift logits 和 labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            main_loss = ce_loss.detach()

            # ====== KL 散度正则化: 防止 adapted logits 偏离基座 logits 太远 ======
            # 核心思想: draft 只应做 "微调" 而非 "重写", KL 散度惩罚过大的分布偏移
            # 只在有效 label 位置计算 KL (忽略 padding 和 prompt)
            kl_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
            if self.lambda_kl > 0:
                with torch.no_grad():
                    # 复用缓存的基座 logits (避免重复计算 lm_head(full_hidden))
                    if _cached_base_logits is not None:
                        base_log_probs = F.log_softmax(_cached_base_logits[:, :-1, :], dim=-1)
                    else:
                        base_logits_for_kl = lm_head(full_hidden)  # fallback: 重新计算
                        base_log_probs = F.log_softmax(base_logits_for_kl[:, :-1, :], dim=-1)
                        del base_logits_for_kl
                    # 构建有效 token mask: 只在有 label 的位置计算 KL
                    valid_mask = (shift_labels != -100)  # [B, seq_len-1]

                # adapted logits 的 log_softmax (有梯度)
                adapted_log_probs = F.log_softmax(shift_logits, dim=-1)

                # KL(base || adapted) = Σ p_base * (log p_base - log p_adapted)
                # 使用 F.kl_div(input=log_adapted, target=log_base, log_target=True)
                # 逐 token 计算, 然后只对有效位置取均值
                kl_per_token = F.kl_div(
                    adapted_log_probs,
                    base_log_probs.detach(),
                    reduction='none',
                    log_target=True,
                ).sum(dim=-1)  # [B, seq_len-1]

                # 只对有效 label 位置取均值
                if valid_mask.any():
                    kl_loss = (kl_per_token * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
                else:
                    kl_loss = kl_per_token.mean()

                del base_log_probs, adapted_log_probs, kl_per_token  # 释放显存

            # 释放缓存的基座 logits (KL 散度已使用完毕)
            if _cached_base_logits is not None:
                del _cached_base_logits

            # ====== Scale L2 软约束: 温和地拉回 scale, 替代硬 clamp 的梯度断裂问题 ======
            # 硬 clamp 在 scale > max_scale 时梯度为 0 (卡死), L2 正则化始终有梯度
            # lambda_scale_l2 * Σ scale² → scale 被温和地拉向 0
            # 配合 adapter 内部的范数比约束 (max_adapt_ratio=15%), 双重保障
            scale_l2_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
            lambda_scale_l2 = 0.1  # Scale L2 正则化权重
            for _idx_str, _adapter in self.readout_adapter.adapters.items():
                scale_l2_loss = scale_l2_loss + _adapter.scale ** 2
            scale_l2_loss = lambda_scale_l2 * scale_l2_loss

            total_loss = ce_loss + div_loss + self.lambda_kl * kl_loss + scale_l2_loss

        # 确保 total_loss 有 grad_fn (DeepSpeed 要求)
        if total_loss.grad_fn is None and self.training:
            anchor = next(self.controller.parameters())
            total_loss = total_loss + (anchor.sum() * 0.0).to(total_loss.dtype)

        # NaN/Inf 检测
        _loss_is_nan = torch.isnan(total_loss).item()
        _loss_is_inf = torch.isinf(total_loss).item()
        if _loss_is_nan or _loss_is_inf:
            _loss_type = "NaN" if _loss_is_nan else "Inf"
            _div_val = div_loss.item() if isinstance(div_loss, torch.Tensor) else div_loss
            print(f"  ❌❌❌ [rank {_rank}] loss 为 {_loss_type}! "
                  f"total_loss={total_loss.item()}, main_loss={main_loss.item()}, "
                  f"div_loss={_div_val}",
                  flush=True)
            total_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype, requires_grad=True)
            anchor = next(self.controller.parameters())
            total_loss = total_loss + (anchor.sum() * 0.0).to(total_loss.dtype)

        # Draft 状态监控
        _draft_metrics_accum = {}
        with torch.no_grad():
            _draft_metrics_accum = self.controller.compute_draft_metrics(
                state=rld_state,
            )
            _draft_metrics_accum['draft/num_steps'] = float(len(split_points))
            _draft_metrics_accum['draft/readout_scale'] = self.readout_adapter.scale.detach().item()
            # Top-K Rerun: 记录每个注入层 adapter 的 scale
            for idx_str, adapter in self.readout_adapter.adapters.items():
                _draft_metrics_accum[f'draft/readout_scale_L{idx_str}'] = adapter.scale.detach().item()

            # 多层注入效果量化指标 (用于 TensorBoard)
            _diff_for_metric = (adapted_hidden - full_hidden.to(adapted_hidden.dtype)).float()
            _draft_metrics_accum['draft/inject_norm_mean'] = _diff_for_metric.norm(dim=-1).mean().item()
            _draft_metrics_accum['draft/inject_ratio'] = (
                _diff_for_metric.norm(dim=-1).mean().item() /
                max(full_hidden.float().norm(dim=-1).mean().item(), 1e-8)
            )

            # 四重防线监控指标
            _draft_metrics_accum['loss/kl_loss'] = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            _draft_metrics_accum['loss/scale_l2_loss'] = scale_l2_loss.item() if isinstance(scale_l2_loss, torch.Tensor) else 0.0
            # 选择性注入权重统计: 反映基座置信度分布
            if _inject_weight is not None:
                _iw = _inject_weight.squeeze(-1)  # [B, seq_len]
                _draft_metrics_accum['draft/inject_weight_mean'] = _iw.mean().item()
                _draft_metrics_accum['draft/inject_weight_min'] = _iw.min().item()
                _draft_metrics_accum['draft/inject_weight_max'] = _iw.max().item()
                _draft_metrics_accum['draft/inject_weight_std'] = _iw.std().item()
                # 分位数统计: 了解 inject_weight 的分布形态
                _draft_metrics_accum['draft/inject_weight_p10'] = _iw.quantile(0.1).item()
                _draft_metrics_accum['draft/inject_weight_p25'] = _iw.quantile(0.25).item()
                _draft_metrics_accum['draft/inject_weight_median'] = _iw.quantile(0.5).item()
                _draft_metrics_accum['draft/inject_weight_p75'] = _iw.quantile(0.75).item()
                _draft_metrics_accum['draft/inject_weight_p90'] = _iw.quantile(0.9).item()
                # 分桶占比: 直观了解基座置信度分布
                # 低注入 (weight < 0.1): 基座非常确定, draft 几乎不参与
                _draft_metrics_accum['draft/iw_bucket_very_low'] = (_iw < 0.1).float().mean().item()
                # 低注入 (0.1 ≤ weight < 0.2): 基座较确定
                _draft_metrics_accum['draft/iw_bucket_low'] = ((_iw >= 0.1) & (_iw < 0.2)).float().mean().item()
                # 中注入 (0.2 ≤ weight < 0.5): 基座中等不确定, draft 有意义的辅助区间
                _draft_metrics_accum['draft/iw_bucket_medium'] = ((_iw >= 0.2) & (_iw < 0.5)).float().mean().item()
                # 高注入 (0.5 ≤ weight < 0.8): 基座较不确定, draft 重要发力区间
                _draft_metrics_accum['draft/iw_bucket_high'] = ((_iw >= 0.5) & (_iw < 0.8)).float().mean().item()
                # 极高注入 (weight ≥ 0.8): 基座犯错, draft 关键修正区间
                _draft_metrics_accum['draft/iw_bucket_very_high'] = (_iw >= 0.8).float().mean().item()
                # 有效训练信号占比: inject_weight ≥ 0.2 的 token 才能提供有意义的梯度
                _draft_metrics_accum['draft/effective_signal_ratio'] = (_iw >= 0.2).float().mean().item()
                # 兼容旧指标名
                _draft_metrics_accum['draft/low_inject_ratio'] = (_iw < 0.2).float().mean().item()

        # ====== forward 阶段耗时汇总 ======
        _fwd_end = _time.time()
        _fwd_total = _fwd_end - _fwd_start
        if _fwd_total > 30:
            _force_profile = True
        if _should_profile or _force_profile:
            _t_vision = _t_vision_end - _t_vision_start
            _t_base = _t_base_fwd_end - _t_base_fwd_start
            _t_pass2 = _t_pass2_end - _t_pass2_start
            _t_rest = _fwd_total - _t_vision - _t_base - _t_pass2
            _warn = " ⚠️ SLOW" if _fwd_total > 30 else ""
            print(f"[rank {_rank}] fwd#{self._fwd_count} 耗时: "
                  f"total={_fwd_total:.1f}s (vision={_t_vision:.1f}s, "
                  f"pass1={_t_base:.1f}s, pass2={_t_pass2:.1f}s, "
                  f"ctrl+loss={_t_rest:.1f}s) "
                  f"steps={len(split_points)} rerun_k={self.rerun_k}{_warn}", flush=True)

        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'div_loss': div_loss.detach(),
            'kl_loss': kl_loss.detach() if isinstance(kl_loss, torch.Tensor) else torch.tensor(0.0),
            'logits': None,  # 不返回完整 logits (太大)
            'rld_state': rld_state,
            'draft_metrics': _draft_metrics_accum,
        }

    # ================================================================
    # 推理流程: 训推一致的 readout adapter 模式
    # 每个 token 生成后，用 readout_adapter 修正 hidden → logits
    # 遇到 </step> 时 controller.step_update 更新 Z_d
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
        推理时的生成 (Top-K Rerun 训推一致: 使用 forward hook in-situ injection)
        
        核心设计:
        - 在注入层 (如 L35) 注册 forward hook, 自动修正 hidden states
        - 修正后的 hidden 写入 KV cache, 后续 token 的 self-attention 自然看到修正值
        - 遇到 </step> delimiter 时触发 controller.step_update 更新 Z_d
        - hook 中引用的 Z_d 会实时更新, 无需手动管理
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. 视觉 encoder
        inner_model = self.base_model.model
        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(
            _pixel_values, grid_thw=image_grid_thw
        )
        merged_hidden_states, gen_deepstack_features = _extract_visual_output(vision_output)
        split_sizes = (
            image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
        ).tolist()
        image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))

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

        # 3. Prompt prefill (标准 KV cache)
        text_model = inner_model.language_model
        lm_head = self.base_model.lm_head

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
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=seg_token_mask,
        )
        inner_model.rope_deltas = rope_deltas

        prompt_len = input_ids.shape[1]
        if position_ids is not None and position_ids.dim() == 3:
            text_pos = torch.arange(prompt_len, device=device).unsqueeze(0).expand(B, -1)
            position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

        visual_pos_masks = (input_ids == image_token_id) if pixel_values is not None else None
        cache = DynamicCache()
        cache_position = torch.arange(prompt_len, device=device)

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

        # ====== 注册 forward hook: 在注入层自动修正 hidden ======
        # 使用可变引用 (list) 来持有当前 Z_d, hook 中通过引用读取最新值
        Z_d_current = rld_state['Z_d']  # [B, K_d, d_z]
        _z_d_ref = [Z_d_current]  # 可变容器, hook 中引用

        # ====== 推理时选择性注入: 保证训推一致 ======
        # 训练时 inject_weight 由基座 logits 计算, 推理时也需要同样的机制
        # 使用可变引用持有当前 inject_weight, 每个 token 生成后更新
        _inject_weight_ref = [None]  # [B, 1, 1] 或 None, hook 中引用

        def _make_injection_hook(layer_idx, injector, z_d_ref, inject_weight_ref):
            """为指定层创建 in-situ injection hook (支持选择性注入)"""
            def hook_fn(module, input, output):
                # output: [B, T, H] (T=1 for autoregressive, T=prompt_len for prefill)
                z_d = z_d_ref[0]  # [B, K_d, d_z]
                T = output.shape[1]
                draft = z_d.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, K_d, d_z]
                iw = inject_weight_ref[0]  # [B, 1, 1] 或 None
                if iw is not None and T > 1:
                    # prefill 阶段: inject_weight 可能需要扩展到 T
                    iw = iw.expand(-1, T, -1)  # [B, T, 1]
                return injector.inject_single_token(layer_idx, output, draft, inject_weight=iw)
            return hook_fn

        _hooks = []
        for inj_layer_idx in self.readout_adapter.injection_layer_indices:
            h = text_model.layers[inj_layer_idx].register_forward_hook(
                _make_injection_hook(inj_layer_idx, self.readout_adapter, _z_d_ref, _inject_weight_ref)
            )
            _hooks.append(h)

        # ★ prefill 最后一个 token 的 logits (hook 已注册，但 prefill 已经跑完了)
        # 需要用 adapter 手动修正最后一个 token
        prefill_hidden = prefill_out.last_hidden_state  # [B, prompt_len, H]
        last_hidden = prefill_hidden[:, -1:, :]  # [B, 1, H]
        last_draft = Z_d_current.unsqueeze(1)  # [B, 1, K_d, d_z]

        # ====== 推理时选择性注入: 计算 prefill 最后 token 的 inject_weight ======
        if self.selective_injection:
            _prefill_logits = lm_head(last_hidden)  # [B, 1, V]
            _prefill_probs = F.softmax(_prefill_logits, dim=-1)
            _prefill_conf = _prefill_probs.max(dim=-1).values  # [B, 1]
            _prefill_iw = (1.0 - _prefill_conf).clamp(min=0.05).unsqueeze(-1)  # [B, 1, 1]
            _inject_weight_ref[0] = _prefill_iw
            del _prefill_logits, _prefill_probs
        
        adapted_last = self.readout_adapter.forward_single_token(
            last_hidden, last_draft, inject_weight=_inject_weight_ref[0]
        )  # [B, 1, H]
        next_token_logits = lm_head(adapted_last).squeeze(1)  # [B, V]

        # 收集 step hidden 用于 delimiter 触发时的 step_update
        step_hidden_buffers = [[] for _ in range(B)]
        for b in range(B):
            step_hidden_buffers[b].append(prefill_hidden[b:b+1])

        # 4. 追加 <think>\n token
        think_start_text = "<think>\n"
        think_start_ids = self.processor.tokenizer.encode(
            think_start_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        think_start_ids = think_start_ids.expand(B, -1)
        think_start_len = think_start_ids.shape[1]

        actual_cache_len = self._get_full_attn_cache_len(past_key_values)
        think_start_attn = torch.ones(B, actual_cache_len + think_start_len, device=device, dtype=attention_mask.dtype)

        think_mrope_pos = mrope_last_pos + 1
        if think_start_len > 1:
            think_mrope_offsets = torch.arange(think_start_len, device=device).view(1, 1, -1)
            think_mrope_pos_full = think_mrope_pos + think_mrope_offsets
        else:
            think_mrope_pos_full = think_mrope_pos
        think_text_pos = torch.arange(actual_cache_len, actual_cache_len + think_start_len, device=device).unsqueeze(0).expand(B, -1)
        think_position_ids = torch.cat([think_text_pos.unsqueeze(0), think_mrope_pos_full], dim=0)

        think_start_embeds = inner_model.get_input_embeddings()(think_start_ids)
        think_cache_pos = torch.arange(actual_cache_len, actual_cache_len + think_start_len, device=device)

        think_outputs = text_model(
            inputs_embeds=think_start_embeds,
            attention_mask=think_start_attn,
            position_ids=think_position_ids,
            past_key_values=past_key_values,
            cache_position=think_cache_pos,
            use_cache=True,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        )
        past_key_values = think_outputs.past_key_values

        # ★ Hook 已注册, think forward 中最后一层的 hidden 已被 hook 修正
        # 但 text_model.forward 内部在 decoder layers 之后还有 norm，
        # last_hidden_state 是 norm(修正后的hidden), 可以直接用
        # 注意: hook 修正的是 decoder layer 的输出, norm 在最后
        next_token_logits = lm_head(think_outputs.last_hidden_state[:, -1:, :]).squeeze(1)  # [B, V]

        mrope_last_pos = think_mrope_pos_full[:, :, -1:]

        generated_ids = torch.cat([input_ids, think_start_ids], dim=-1)

        # 5. 自回归生成
        eos_token_id = self.processor.tokenizer.eos_token_id if hasattr(self, 'processor') else None
        delim_ids = self.step_delimiter_ids
        delim_len = len(delim_ids) if delim_ids is not None else 0
        recent_tokens = [[] for _ in range(B)]

        current_pos = actual_cache_len + think_start_len

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

            # 标准自回归 forward (单 token)
            new_mrope_pos = mrope_last_pos + 1
            text_pos = torch.full((B, 1), current_pos, device=device, dtype=torch.long)
            position_ids_4d = torch.cat([text_pos.unsqueeze(0), new_mrope_pos], dim=0)

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
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
            )

            past_key_values = text_outputs.past_key_values
            hidden_states = text_outputs.last_hidden_state  # [B, 1, H]

            # ★ Hook 已在注入层自动修正 hidden, last_hidden_state 是 norm(修正后) 的结果
            # 直接过 lm_head 即可
            next_token_logits = lm_head(hidden_states[:, -1, :])  # [B, V]

            # ====== 推理时选择性注入: 更新 inject_weight 供下一个 token 的 hook 使用 ======
            # 基于当前 token 的 logits 计算置信度, 用于下一个 token 的注入控制
            # 开销: 仅一次 softmax + max, 相比 lm_head 的矩阵乘法可忽略
            if self.selective_injection:
                with torch.no_grad():
                    _gen_probs = F.softmax(next_token_logits, dim=-1)  # [B, V]
                    _gen_conf = _gen_probs.max(dim=-1).values  # [B]
                    _gen_iw = (1.0 - _gen_conf).clamp(min=0.05).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                    _inject_weight_ref[0] = _gen_iw

            mrope_last_pos = new_mrope_pos

            # 收集 step hidden
            for b in range(B):
                step_hidden_buffers[b].append(hidden_states[b:b+1])

            # 检测 delimiter
            for b in range(B):
                recent_tokens[b].append(next_token[b].item())
                if len(recent_tokens[b]) > delim_len:
                    recent_tokens[b] = recent_tokens[b][-delim_len:]

            if delim_ids is not None and delim_len > 0:
                trigger_mask = torch.zeros(B, dtype=torch.bool, device=device)
                for b in range(B):
                    if len(recent_tokens[b]) >= delim_len and recent_tokens[b][-delim_len:] == delim_ids:
                        trigger_mask[b] = True

                if trigger_mask.any():
                    # 拼接 step hidden 用于 step_update
                    max_buf_len = max(len(step_hidden_buffers[b]) for b in range(B))
                    if max_buf_len > 0:
                        padded_hiddens = []
                        for b in range(B):
                            if len(step_hidden_buffers[b]) > 0:
                                h = torch.cat(step_hidden_buffers[b], dim=1)
                            else:
                                h = torch.zeros(1, 1, self.hidden_size, device=device, dtype=hidden_states.dtype)
                            if h.shape[1] < max_buf_len:
                                pad = torch.zeros(1, max_buf_len - h.shape[1], self.hidden_size,
                                                  device=device, dtype=hidden_states.dtype)
                                h = torch.cat([h, pad], dim=1)
                            padded_hiddens.append(h)
                        step_hiddens_gen = torch.cat(padded_hiddens, dim=0)

                        # 更新 Z_d (controller.step_update，与训练一致)
                        rld_state = self.controller.step_update(
                            rld_state, step_hiddens_gen, update_mask=trigger_mask
                        )
                        # 更新 hook 中的 Z_d 引用 (可变容器, hook 实时读取)
                        _z_d_ref[0] = rld_state['Z_d']

                    for b in range(B):
                        if trigger_mask[b]:
                            step_hidden_buffers[b] = []
                            recent_tokens[b] = []

            current_pos += 1

        # 清理 forward hooks
        for h in _hooks:
            h.remove()

        return generated_ids

    # ================================================================
    # 保存/加载
    # ================================================================

    def save_pretrained(self, save_dir: str):
        """保存 RLD Controller + ReadoutAdapter 参数 (兼容 ZeRO-3)"""
        os.makedirs(save_dir, exist_ok=True)
        controller_path = os.path.join(save_dir, "rld_controller.pt")
        adapter_path = os.path.join(save_dir, "rld_readout_adapter.pt")

        try:
            import deepspeed
            first_param = next(self.controller.parameters())
            if hasattr(first_param, 'ds_id'):
                # ZeRO-3: gather 参数
                controller_state = {}
                for name, param in self.controller.named_parameters():
                    with deepspeed.zero.GatheredParameters(param):
                        if self._verbose:
                            controller_state[name] = param.data.cpu().clone()
                adapter_state = {}
                for name, param in self.readout_adapter.named_parameters():
                    with deepspeed.zero.GatheredParameters(param):
                        if self._verbose:
                            adapter_state[name] = param.data.cpu().clone()
                if self._verbose:
                    torch.save(controller_state, controller_path)
                    torch.save(adapter_state, adapter_path)
                    print(f"[RLD] Controller 已保存到 {controller_path} (ZeRO-3 gathered)")
                    print(f"[RLD] ReadoutAdapter 已保存到 {adapter_path} (ZeRO-3 gathered)")
                return
        except ImportError:
            pass

        if self._verbose:
            torch.save(self.controller.state_dict(), controller_path)
            torch.save(self.readout_adapter.state_dict(), adapter_path)
            print(f"[RLD] Controller 已保存到 {controller_path}")
            print(f"[RLD] ReadoutAdapter 已保存到 {adapter_path}")

    def load_pretrained(self, save_dir: str):
        """加载 RLD Controller + ReadoutAdapter 参数"""
        controller_path = os.path.join(save_dir, "rld_controller.pt")
        state_dict = torch.load(controller_path, map_location="cpu")
        self.controller.load_state_dict(state_dict)
        if self._verbose:
            print(f"[RLD] Controller 已从 {controller_path} 加载")

        adapter_path = os.path.join(save_dir, "rld_readout_adapter.pt")
        if os.path.exists(adapter_path):
            adapter_state = torch.load(adapter_path, map_location="cpu")
            self.readout_adapter.load_state_dict(adapter_state)
            if self._verbose:
                print(f"[RLD] ReadoutAdapter 已从 {adapter_path} 加载")
        else:
            if self._verbose:
                print(f"[RLD] ⚠️ 未找到 ReadoutAdapter 权重: {adapter_path}")
