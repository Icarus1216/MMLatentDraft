"""RLD Model: 包装 Qwen3-VL VLM 的 Reflective Latent Draft 模型

Full KV Cache Chunkwise Training 方案: 按 step boundary 切 chunks, 每个 chunk 使用对应 Z_d 的 prefix KV + 完整历史 KV cache

核心设计:
1. 基座 L0~L27 冻结, L28~L35 通过 LoRA 协同训练 (Draft Evolver + VLM 协同)
2. 训练 RLD Controller + PrefixKVProjector + LoRA (~29M 参数)
3. 训练时 (Full KV Cache Chunkwise Training, 训推完全一致):
   a. 按 step boundary 将序列切分为 chunks
   b. 维护完整的 per-layer KV cache (detach, 只提供历史上下文)
   c. 对每个 chunk:
      - Z_d_c → PrefixKVProjector → prefix K/V (有梯度)
      - 注入层: [prefix_KV + 历史 KV cache (detach)] → 当前 chunk token 能看到所有历史
      - 非注入层: [历史 KV cache (detach)] → 当前 chunk token 能看到所有历史
      - 36 层 forward (有梯度, LoRA 层有可训练增量)
      - 从 chunk hidden 提取 step summary S_c
      - controller update: S_c → T_c → G_c → Z_d_{c+1}
      - 将当前 chunk 的 KV detach 后追加到历史 KV cache
   d. concat 所有 chunk hidden → logits → CE loss
   e. 每个 Z_d_c 都有来自 CE loss 的直接梯度 (训推一致)
4. 推理时:
   a. prefix KV 写入 DynamicCache 头部, 后续 token 自然看到
   b. 遇到 step boundary 时更新 Z_d → 覆写 cache 中 prefix 位置的 KV
   c. 完全兼容 flash_attention_2 (KV_len > Q_len 是标准 KV cache 用法)

LoRA 协同训练:
- LoRA 让注入层 (L28~L35) 的 Q/V 投影学会更好地"查询"和"协调" prefix KV
- LoRA 参数量极小 (~4M), 不会显著增加显存
- 差异化学习率: Controller/PrefixKVProjector lr=1e-4, LoRA lr=3e-5
"""
import os
import re
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
from .modules import PrefixKVProjector

# LoRA 支持 (Stage 2)
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

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
    
    Chunkwise Prefix KV Training 架构:
    ┌──────────────────────────────────────────────────────────────┐
    │  训练流程 (Chunkwise Prefix KV Training):                     │
    │  1. 视觉 encoder → visual_tokens → Z_e (证据槽)              │
    │  2. controller.prefill → Z_d_0 (有梯度)                      │
    │  3. 按 step boundary 切分序列为 chunks                        │
    │  4. 对每个 chunk_c:                                           │
    │     a. Z_d_c → PrefixKVProjector → prefix K/V_c              │
    │     b. 36 层 forward (注入层使用 prefix K/V_c)                │
    │     c. 从 chunk hidden 提取 step summary S_c                  │
    │     d. controller: S_c → T_c → G_c → Z_d_{c+1}              │
    │  5. concat 所有 chunk hidden → logits → CE loss               │
    │  6. 每个 Z_d_c 都有来自 CE loss 的直接梯度                    │
    │                                                               │
    │  推理流程 (训推一致):                                          │
    │  - prefix KV 写入 DynamicCache 头部, 后续 token 自然看到      │
    │  - 遇到 step boundary 时更新 Z_d → 覆写 cache 中 prefix KV   │
    │  - 完全兼容 flash_attention_2                                 │
    └──────────────────────────────────────────────────────────────┘
    """

    # Step boundary 检测: 新格式使用 "Step N:" (N>=2) 和 "Final Answer:" 作为 boundary
    # 保留旧常量用于兼容
    STEP_DELIMITER = "</step>"  # 旧格式兼容
    # 新格式 boundary 标记
    STEP_PATTERN = re.compile(r'Step\s+(\d+)\s*[:.\s]', re.IGNORECASE)
    FINAL_ANSWER_PATTERN = re.compile(r'Final\s+Answer\s*:', re.IGNORECASE)

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
        max_scale: float = 0.3,
        selective_injection: bool = True,
        use_trace_updater: bool = True,
        use_bidirectional_reflection: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_z = d_z
        self.total_layers = total_layers
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
            lambda_div=lambda_div,
            max_steps=14,
            use_trace_updater=use_trace_updater,
            use_bidirectional_reflection=use_bidirectional_reflection,
        )

        # 确保 controller 使用与基座相同的 dtype
        self.controller = self.controller.to(dtype=torch_dtype)

        # ====== 3. 创建 PrefixKVProjector (Persistent Prefix Memory 方案) ======
        # 核心思想: 将 Z_d 投影为中高层 attention 的 prefix K/V slots
        # 注入层: L18, L26, L35 (均匀覆盖中层语义组合→高层推理→输出决策)
        # prefix KV 通过 DynamicCache 注入, 完全兼容 flash_attention_2
        # 训练时: 单 Pass 全序列有梯度, 注入层通过 per-layer prefix cache 注入
        # 推理时: prefix KV 写入 DynamicCache 头部, 覆写更新
        self.injection_layer_indices = [18, 26, 35]  # 中层+高层注入点
        self.num_prefix_slots = num_draft_slots  # prefix slots 数 = draft slots 数 (16)

        # 从基座 config 获取 GQA 参数
        text_config = self.base_model.config.text_config if hasattr(self.base_model.config, 'text_config') else self.base_model.config
        self.num_kv_heads = getattr(text_config, 'num_key_value_heads', 8)
        self.head_dim = getattr(text_config, 'head_dim', text_config.hidden_size // text_config.num_attention_heads)

        self.prefix_kv_projector = PrefixKVProjector(
            d_z=d_z,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            injection_layers=self.injection_layer_indices,
        )
        self.prefix_kv_projector = self.prefix_kv_projector.to(dtype=torch_dtype)



        if self._verbose:
            self.controller.print_param_summary()
            projector_params = sum(p.numel() for p in self.prefix_kv_projector.parameters() if p.requires_grad)
            print(f"[RLD] PrefixKVProjector 参数: {projector_params:,} ({projector_params/1e6:.2f}M)")
            print(f"[RLD]   注入层索引: {self.injection_layer_indices}")
            print(f"[RLD]   Prefix slots: {self.num_prefix_slots}")
            print(f"[RLD]   GQA: num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}")
            print(f"[RLD]   Prefix KV: 虚拟 token 模式 (无 scale 门控, V 零初始化)")

        # ====== 4. LoRA 状态标记 ======
        self._lora_enabled = False
        self._lora_layers = None  # 记录 LoRA 应用的层范围

        # ====== 5. Step delimiter token ids ======
        self.step_delimiter_ids = None
        self.step_delimiter_id = None

    def setup_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        lora_layers: list = None,
    ):
        """
        Stage 2: 在基座模型的指定层添加 LoRA adapter
        
        设计要点:
        - 只在注入层范围内的层 (L18~L35) 添加 LoRA
        - 与 PrefixKVProjector 的注入层对齐
        - LoRA 参数量较小 (~9M), 不会显著增加显存
        - 梯度路径: loss → lm_head → norm → L35(LoRA) → ... → L18(LoRA) → prefix KV → Z_d → controller
        
        Args:
            lora_r: LoRA 秩 (默认 16)
            lora_alpha: LoRA 缩放因子 (默认 32)
            lora_dropout: LoRA dropout (默认 0.05)
            target_modules: 目标模块名 (默认 ["q_proj", "v_proj"])
            lora_layers: 要应用 LoRA 的层索引列表 (默认注入层范围内的层)
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "[RLD] Stage 2 需要 peft 库。请运行: pip install peft"
            )
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        if lora_layers is None:
            # 默认: 注入层范围内的所有层 (L18~L35)
            lora_layers = list(range(min(self.injection_layer_indices), self.total_layers))
        
        self._lora_layers = lora_layers
        
        if self._verbose:
            print(f"\n[RLD Stage 2] 配置 LoRA...")
            print(f"  LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            print(f"  目标模块: {target_modules}")
            print(f"  应用层: L{min(lora_layers)}~L{max(lora_layers)} ({len(lora_layers)} 层)")
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            layers_to_transform=lora_layers,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 应用 LoRA 到基座模型
        # ★ autocast_adapter_dtype=False: 阻止 PEFT 将 bfloat16 LoRA 权重 upcast 到 float32
        # PEFT 默认会将 bf16/fp16 的 adapter 权重转为 fp32 (用于标准 HF forward 的稳定性),
        # 但我们手动展开 decoder layer 循环, fp32 LoRA 权重会导致 backward 时
        # "Expected BFloat16, got Float" 错误 (ToCopyBackward 将梯度从 bf16 转为 fp32)
        self.base_model = get_peft_model(self.base_model, lora_config, autocast_adapter_dtype=False)
        self._lora_enabled = True
        
        # ★ 强制确保 LoRA 参数与基座模型 dtype 一致 (双重保险)
        # autocast_adapter_dtype=False 应该已经阻止了 upcast, 但某些 PEFT 版本可能仍然 upcast
        # 这里显式将所有 LoRA 参数转回基座模型的 dtype
        # 从冻结的基座参数中获取 dtype (跳过 LoRA 参数)
        _base_dtype = torch.bfloat16  # 默认值
        for name, param in self.base_model.named_parameters():
            if 'lora_' not in name:
                _base_dtype = param.dtype
                break
        _lora_fp32_count = 0
        for name, param in self.base_model.named_parameters():
            if 'lora_' in name and param.dtype != _base_dtype:
                param.data = param.data.to(_base_dtype)
                _lora_fp32_count += 1
        if _lora_fp32_count > 0 and self._verbose:
            print(f"  ⚠️ 已将 {_lora_fp32_count} 个 LoRA 参数从 float32 转为 {_base_dtype}")
        
        # 统计 LoRA 参数
        lora_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        if self._verbose:
            print(f"  LoRA 可训练参数: {lora_params:,} ({lora_params/1e6:.2f}M)")
            self.base_model.print_trainable_parameters()
            # 诊断: 打印 LoRA 参数 dtype 确认
            for name, param in self.base_model.named_parameters():
                if 'lora_A' in name and param.requires_grad:
                    print(f"  LoRA 参数 dtype 确认: {name} → {param.dtype}")
                    break
            print(f"[RLD Stage 2] ✅ LoRA 已启用\n")

    @property
    def _inner_model(self):
        """透明获取内部 Qwen3VL 模型 (兼容 LoRA 包装)
        
        peft.get_peft_model() 会在 base_model 外面包一层 PeftModel:
        - 无 LoRA: self.base_model.model → Qwen3VLModel
        - 有 LoRA: self.base_model.base_model.model.model → Qwen3VLModel
        
        此属性统一返回 Qwen3VLModel (包含 .visual, .language_model 等)
        """
        if self._lora_enabled:
            # PeftModel 包装: base_model → PeftModel → base_model → LoraModel → model → Qwen3VLForConditionalGeneration → model
            peft_base = self.base_model.base_model  # LoraModel
            original_model = peft_base.model  # Qwen3VLForConditionalGeneration
            return original_model.model  # Qwen3VLModel
        return self.base_model.model

    @property
    def _lm_head(self):
        """透明获取 lm_head (兼容 LoRA 包装)"""
        if self._lora_enabled:
            peft_base = self.base_model.base_model  # LoraModel
            original_model = peft_base.model  # Qwen3VLForConditionalGeneration
            return original_model.lm_head
        return self.base_model.lm_head
    def set_processor(self, processor):
        """设置 processor 并获取 delimiter token id 序列"""
        self.processor = processor
        tokenizer = processor.tokenizer

        # 旧格式 delimiter (保留用于兼容)
        delimiter_ids = tokenizer.encode(self.STEP_DELIMITER, add_special_tokens=False)
        self.step_delimiter_ids = delimiter_ids
        if len(delimiter_ids) == 1:
            self.step_delimiter_id = delimiter_ids[0]
        else:
            self.step_delimiter_id = None

        # 新格式: 预编码常用 boundary token 序列
        # "Step " 的 token ids (用于推理时检测 "Step N:")
        self._step_prefix_ids = tokenizer.encode("Step ", add_special_tokens=False)
        # "Final Answer:" 的 token ids
        self._final_answer_ids = tokenizer.encode("Final Answer:", add_special_tokens=False)
        # "\nStep " 的 token ids (带换行前缀, 更精确匹配)
        self._newline_step_ids = tokenizer.encode("\nStep ", add_special_tokens=False)

        if self._verbose:
            decoded = tokenizer.decode(delimiter_ids)
            print(f"[RLD] Step delimiter (旧): '{self.STEP_DELIMITER}' → token_ids={delimiter_ids}")
            print(f"[RLD] Step prefix (新): 'Step ' → token_ids={self._step_prefix_ids}")
            print(f"[RLD] Final Answer (新): 'Final Answer:' → token_ids={self._final_answer_ids}")
            print(f"[RLD] 新格式: 使用 'Step N:' (N>=2) 和 'Final Answer:' 作为 boundary")

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
        """
        在 token 序列中查找 step boundary 位置 (训练时 fallback 用)
        
        新格式: 查找 "Step N:" (N>=2) 和 "Final Answer:" 的起始 token 位置
        旧格式 (兼容): 查找 </step> 的末尾 token 位置
        
        注意: 训练时通常由 Collator 预计算 step_boundaries 传入,
        此方法仅在 step_boundaries 为 None 时作为 fallback。
        """
        if not hasattr(self, 'processor') or self.processor is None:
            # 无 processor, 回退到旧格式 </step> 匹配
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
        
        # 新格式: decode 整个序列, 用正则匹配 boundary
        tokenizer = self.processor.tokenizer
        try:
            # 使用 offset_mapping 精确定位
            text = tokenizer.decode(token_ids.tolist())
            result = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = result['offset_mapping']
            positions = []
            
            # 查找 "Step N:" (N>=2)
            for m in re.finditer(r'Step\s+(\d+)\s*[:.\s]', text, re.IGNORECASE):
                step_num = int(m.group(1))
                if step_num >= 2:
                    start_char = m.start()
                    for tok_idx, (cs, ce) in enumerate(offsets):
                        if cs <= start_char < ce:
                            positions.append(tok_idx)
                            break
            
            # 查找 "Final Answer:"
            for m in re.finditer(r'Final\s+Answer\s*:', text, re.IGNORECASE):
                start_char = m.start()
                for tok_idx, (cs, ce) in enumerate(offsets):
                    if cs <= start_char < ce:
                        positions.append(tok_idx)
                        break
            
            return sorted(set(positions))
        except Exception:
            # Fallback: 旧格式 </step> 匹配
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
        inner_model = self._inner_model

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
        loss_weight_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播 (Chunkwise Prefix KV Training)
        
        核心流程 (训推一致):
          1. 视觉 encoder → visual_tokens → Z_e (no_grad)
          2. controller.prefill → Z_d_0 (有梯度)
          3. 按 step boundary 切分序列为 chunks
          4. 对每个 chunk:
             a. Z_d_c → PrefixKVProjector → prefix K/V (有梯度)
             b. 36 层 forward (有梯度, 注入层使用当前 prefix KV)
             c. 从 chunk hidden 提取 step summary S_c
             d. controller update: S_c → T_c → G_c → Z_d_{c+1}
             e. 下一个 chunk 使用 Z_d_{c+1} 的 prefix KV
          5. concat 所有 chunk hidden → logits → CE loss
        
        梯度路径 (每个 Z_d_c 都有直接梯度):
          CE loss → logits → chunk_c hidden → prefix_KV_c → Z_d_c → controller
          同时 Z_d_c 依赖 Z_d_{c-1} (残差连接), 梯度可回传到所有历史 draft
        
        优势:
        - 训推完全一致: 训练和推理都是 "生成一段 → 更新 Z_d → 影响后续生成"
        - 每个 Z_d_c 都有来自 CE loss 的直接梯度 (不再只有最后一个 Z_d 有梯度)
        - 完全兼容 flash_attention_2
        """
        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device

        # ====== dtype 对齐: Accelerate 混合精度会将可训练参数 upcast 到 float32 ======
        # 问题: TrainingArguments(bf16=True) 时, Accelerate 的 prepare_model 会将所有
        # requires_grad=True 的参数 upcast 到 float32 (master weights), 但基座模型冻结参数
        # 仍然是 bfloat16。controller/prefix_kv_projector 的参数被 upcast 后, 其 forward
        # 输出也是 float32, 与基座模型的 bf16 KV cache 拼接后, backward 时 dtype 冲突:
        #   RuntimeError: Expected BFloat16, got Float
        #
        # 解决方案: 不修改参数 dtype (Accelerate 需要 float32 master weights 保证精度),
        # 而是在 controller/prefix_kv_projector 的输出传入基座模型之前, 显式 cast 到基座 dtype。
        # 这样 forward 计算在 float32 (精度更高), 但与基座模型交互时用 bfloat16 (兼容)。
        if not hasattr(self, '_base_model_dtype'):
            # 从冻结的基座参数获取 dtype (跳过 LoRA 参数)
            self._base_model_dtype = torch.bfloat16  # 默认值
            for _n, _p in self.base_model.named_parameters():
                if 'lora_' not in _n:
                    self._base_model_dtype = _p.dtype
                    break
            ctrl_dtype_now = next(self.controller.parameters()).dtype
            if self._verbose:
                print(f"[RLD] dtype 检测: 基座模型={self._base_model_dtype}, "
                      f"controller={ctrl_dtype_now}, "
                      f"prefix_kv_projector={next(self.prefix_kv_projector.parameters()).dtype}")
                if ctrl_dtype_now != self._base_model_dtype:
                    print(f"[RLD] ⚠️ controller dtype ({ctrl_dtype_now}) != 基座 dtype ({self._base_model_dtype})")
                    print(f"[RLD]   这是 Accelerate bf16 混合精度的正常行为 (master weights 为 float32)")
                    print(f"[RLD]   将在 controller/prefix_kv 输出处显式 cast 到 {self._base_model_dtype}")

        _base_dtype = self._base_model_dtype  # 基座模型 dtype (bfloat16)
        ctrl_dtype = next(self.controller.parameters()).dtype  # controller 参数 dtype (可能是 float32)

        _rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))

        # ====== forward 阶段耗时监控: 精确定位 hang 在哪个阶段 ======
        import time as _time
        _fwd_start = _time.time()
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        # 每 50 次 forward 且仅 rank 0 打印; 超 30s 异常时所有 rank 打印
        _is_rank0 = (_rank == 0)
        _should_profile = (_is_rank0 and self._fwd_count % 50 == 0)
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

        # ====== 3. 一次性完整 base forward (全序列有梯度 + prefix KV 注入) ======
        # 核心改造: 不再有 Pass 1 (no_grad) + Pass 2 (rerun)
        # 而是单 Pass 全 36 层有梯度, 注入层通过 per-layer prefix KV cache 注入
        # 梯度路径: loss → lm_head → norm → L35 → ... → L28(prefix KV) → ... → L0
        #           ↑ prefix KV 有梯度 → Z_d → controller
        _t_base_fwd_start = _time.time()

        inner_model = self._inner_model
        text_model = inner_model.language_model
        lm_head = self._lm_head

        # 3a. Embedding + 视觉融合 (no_grad: embedding 层冻结)
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

        # 3b. 构建 visual_pos_masks (DeepStack 需要)
        visual_pos_masks = None
        deepstack_for_fwd = None
        if pixel_values is not None and image_grid_thw is not None:
            image_token_id = inner_model.config.image_token_id
            visual_pos_masks = (input_ids == image_token_id)
            deepstack_for_fwd = deepstack_features

        # 3c. 获取 mRoPE position_ids (no_grad: 纯位置信息)
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

            # 构造 4D position_ids: [4, B, seq_len]
            if position_ids is not None and position_ids.dim() == 3:
                text_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
                position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

        # 3d. 手动展开 text_model forward + prefix KV 注入
        # Persistent Prefix Memory 方案:
        # - 非注入层: 正常 forward (past_key_values=None)
        # - 注入层: 构造 per-layer DynamicCache 预填充 prefix KV, 传入 decoder_layer
        #   → self_attn 内部 past_key_values.update() 自动拼接 prefix KV 到 K/V 前面
        #   → flash_attention_2 原生支持 KV_len > Q_len (标准 KV cache 用法)
        #   → create_causal_mask 根据 cache 长度自动扩展 mask
        N_p = self.num_prefix_slots  # prefix slots 数 (= K_d = 16)

        cache_position = torch.arange(seq_len, device=device)

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
        from transformers.cache_utils import DynamicLayer as _DynamicLayer

        # ====== 4. 自动检测 step 边界 (移到 forward 之前, chunkwise 需要) ======
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

        # 收集所有 step 边界 (跨 batch 取并集，确保所有 rank 的循环次数一致)
        all_boundaries = set()
        if step_boundaries:
            for b_list in step_boundaries:
                all_boundaries.update(b_list)
        split_points = sorted([p for p in all_boundaries if gen_start < p < seq_len])

        per_sample_sets = [set(bl) for bl in step_boundaries] if step_boundaries else [set() for _ in range(B)]

        MAX_STEPS = 14
        if len(split_points) > MAX_STEPS:
            split_points = split_points[:MAX_STEPS]

        # ====== 构建 chunks: 按 step boundary 切分序列 ======
        # chunk_ranges: [(start, end), ...], 每个 chunk 的 token 范围 [start, end)
        # 第一个 chunk: [0, first_boundary+1)
        # 中间 chunks: [prev_boundary+1, boundary+1)
        # 最后一个 chunk: [last_boundary+1, seq_len)
        chunk_ranges = []
        if len(split_points) == 0:
            # 无 step boundary: 整个序列作为一个 chunk
            chunk_ranges.append((0, seq_len))
        else:
            # 第一个 chunk: 从序列开头到第一个 boundary (含)
            chunk_ranges.append((0, split_points[0] + 1))
            # 中间 chunks
            for i in range(1, len(split_points)):
                chunk_ranges.append((split_points[i - 1] + 1, split_points[i] + 1))
            # 最后一个 chunk: 从最后一个 boundary 之后到序列末尾
            if split_points[-1] + 1 < seq_len:
                chunk_ranges.append((split_points[-1] + 1, seq_len))

        injection_set = set(self.injection_layer_indices)

        # ====== Full KV Cache Chunkwise Training (训推一致) ======
        # 核心改进: 训练时维护完整 KV cache, 让每个 chunk 的 token 能看到所有历史 token
        # 与推理时完全一致: prefix KV + 完整历史 KV cache
        #
        # KV cache 管理策略:
        # - KV cache 本身 detach (不参与梯度计算, 只提供历史上下文)
        # - 只有 prefix KV 有梯度 (通过 PrefixKVProjector → Z_d → controller)
        # - 每个 chunk 的 36 层 forward 有梯度 (当前 chunk 的 token 有梯度)
        #
        # KV cache 布局 (注入层):
        #   [prefix_KV(0~N_p-1)] [历史 token KV (detach)] [当前 chunk KV (有梯度)]
        # KV cache 布局 (非注入层):
        #   [历史 token KV (detach)] [当前 chunk KV (有梯度)]
        #
        # 梯度路径 (方案 B):
        #   CE loss → logits → chunk_c hidden → L35(prefix_KV_c) → prefix_source → G_c → controller
        #   prefix_source = gate * G_c + (1 - gate) * Z_d_init

        # ====== 将 prefill 输出 cast 到基座 dtype ======
        # controller.prefill 内部使用 controller 参数 dtype (可能是 float32),
        # 但 Z_d 需要传入 prefix_kv_projector → 基座模型, 必须是 bfloat16
        Z_d_current = rld_state['Z_d'].to(_base_dtype)  # [B, K_d, d_z], cast 到基座 dtype
        Z_d_init = rld_state['Z_d_init'].to(_base_dtype)  # [B, K_d, d_z], 固定视觉基线
        # 同步更新 rld_state 中的引用 (后续代码可能直接读取)
        rld_state['Z_d'] = Z_d_current
        rld_state['Z_d_init'] = Z_d_init
        if rld_state.get('visual_hidden_proj') is not None:
            rld_state['visual_hidden_proj'] = rld_state['visual_hidden_proj'].to(_base_dtype)
        all_Z_d = [Z_d_current]  # 收集所有 step 的 prefix_source (含初始)
        all_T_c = []  # 收集每步的 T_c (备用, 对比学习已弃用)
        all_commit_scores = []  # 收集每步的 commit_score (用于 commit loss, 正样本)
        all_neg_commit_scores = []  # 收集非 boundary 位置的 commit_score (负样本)
        _per_step_ranks = []  # 收集每步 Z_d 的有效秩 (秩演化追踪)
        step_summaries = []
        update_masks = []
        all_chunk_hidden = []  # 收集每个 chunk 的 normed hidden (用于最终 logits)
        div_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        grounding_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)  # Visual Grounding Loss

        position_embeddings = text_model.rotary_emb(inputs_embeds, _mrope_position_ids)

        # ====== 初始化 per-layer KV cache (detach, 只提供历史上下文) ======
        # 使用 list of (K, V) per layer, 比 DynamicCache 更灵活
        # 注入层: 额外包含 prefix KV 在头部
        # 非注入层: 只有历史 token KV
        # 初始状态: 所有层的 KV cache 为空 (None)
        _layer_kv_cache = [None] * self.total_layers  # list of (K, V) or None

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
            chunk_len = chunk_end - chunk_start

            # ---- 当前 chunk 的输入切片 ----
            chunk_embeds = inputs_embeds[:, chunk_start:chunk_end, :]  # [B, chunk_len, H]
            chunk_attn_mask_raw = attention_mask[:, :chunk_end] if attention_mask.dim() == 2 else attention_mask
            chunk_cache_position = cache_position[chunk_start:chunk_end]  # [chunk_len]
            chunk_text_pos = _text_position_ids[:, chunk_start:chunk_end] if _text_position_ids.dim() == 2 else _text_position_ids
            chunk_mrope_pos = _mrope_position_ids[:, :, chunk_start:chunk_end]

            # 重新计算 chunk 的 position embeddings (RoPE 依赖位置)
            chunk_pos_emb = text_model.rotary_emb(chunk_embeds, chunk_mrope_pos)

            # ---- 计算当前 Z_d 的 prefix KV (有梯度) ----
            # 确保 Z_d_current 是基座 dtype (bfloat16), 防止 Accelerate upcast 导致的 dtype 传播
            if Z_d_current.dtype != _base_dtype:
                Z_d_current = Z_d_current.to(_base_dtype)
            prefix_kvs = self.prefix_kv_projector.get_all_prefix_kvs(Z_d_current)

            # ====== DEBUG: prefix KV dtype 检查 ======
            if _rank == 0 and self._fwd_count <= 2 and chunk_idx == 0:
                print(f"\n🔍 [DEBUG prefix KV] chunk_idx={chunk_idx}:", flush=True)
                print(f"  Z_d_current: dtype={Z_d_current.dtype}", flush=True)
                print(f"  chunk_embeds: dtype={chunk_embeds.dtype}", flush=True)
                for _li, (_pk, _pv) in prefix_kvs.items():
                    print(f"  prefix_kvs[{_li}]: K.dtype={_pk.dtype}, V.dtype={_pv.dtype}", flush=True)
                    break  # 只打印第一个注入层

            # ---- 36 层 forward (有梯度, 使用完整 KV cache + prefix KV) ----
            hidden_states = chunk_embeds
            for layer_idx, decoder_layer in enumerate(text_model.layers):
                # 构造当前层的 past_key_values (DynamicCache 格式)
                layer_cache = DynamicCache()
                _history_kv_len = 0  # 历史 KV 长度 (不含 prefix)

                if layer_idx in injection_set:
                    # 注入层: [prefix_KV] + [历史 KV (detach)]
                    pfx_k, pfx_v = prefix_kvs[layer_idx]  # 有梯度
                    # 构造 DynamicCache: 先填充前面的空层, 再填充当前层
                    for i in range(layer_idx + 1):
                        dl = _DynamicLayer()
                        if i == layer_idx:
                            if _layer_kv_cache[layer_idx] is not None:
                                # 拼接: [prefix_KV; 历史 KV (detach)]
                                hist_k, hist_v = _layer_kv_cache[layer_idx]
                                _history_kv_len = hist_k.shape[2]
                                dl.keys = torch.cat([pfx_k, hist_k], dim=2)    # [B, H, N_p+hist_len, D]
                                dl.values = torch.cat([pfx_v, hist_v], dim=2)  # [B, H, N_p+hist_len, D]
                            else:
                                # 第一个 chunk: 只有 prefix KV
                                dl.keys = pfx_k
                                dl.values = pfx_v
                            dl.is_initialized = True
                        layer_cache.layers.append(dl)

                    # cache_position: 当前 chunk 的位置从 N_p + history_len 开始
                    _layer_cache_pos = chunk_cache_position + N_p
                    # attention_mask: 需要覆盖 [prefix + 历史 + 当前 chunk] 的完整范围
                    _total_past_len = N_p + _history_kv_len
                    _layer_attn_mask_raw = torch.ones(B, _total_past_len + chunk_len, device=device, dtype=attention_mask.dtype)
                    # 将历史部分的 attention mask 正确填充
                    if _history_kv_len > 0:
                        _layer_attn_mask_raw[:, N_p:N_p + _history_kv_len] = attention_mask[:, :_history_kv_len] if attention_mask.dim() == 2 else 1
                    # 当前 chunk 部分
                    if attention_mask.dim() == 2:
                        _layer_attn_mask_raw[:, _total_past_len:] = attention_mask[:, chunk_start:chunk_end]

                    _layer_attn_mask = create_causal_mask(
                        config=text_model.config,
                        input_embeds=chunk_embeds,
                        attention_mask=_layer_attn_mask_raw,
                        cache_position=_layer_cache_pos,
                        past_key_values=layer_cache,
                        position_ids=chunk_text_pos,
                    )

                    # ====== DEBUG: decoder layer 输入 dtype ======
                    if _rank == 0 and self._fwd_count <= 2 and chunk_idx == 0 and layer_idx == self.injection_layer_indices[0]:
                        print(f"\n🔍 [DEBUG decoder_layer] layer_idx={layer_idx}, chunk_idx={chunk_idx}:", flush=True)
                        print(f"  hidden_states: dtype={hidden_states.dtype}", flush=True)
                        print(f"  _layer_attn_mask: dtype={_layer_attn_mask.dtype if _layer_attn_mask is not None else 'None'}", flush=True)
                        _dl_k = layer_cache.layers[layer_idx].keys
                        _dl_v = layer_cache.layers[layer_idx].values
                        print(f"  layer_cache K: dtype={_dl_k.dtype}, V: dtype={_dl_v.dtype}", flush=True)

                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=_layer_attn_mask,
                        position_ids=chunk_text_pos,
                        past_key_values=layer_cache,
                        cache_position=_layer_cache_pos,
                        position_embeddings=chunk_pos_emb,
                    )
                else:
                    # 非注入层: [历史 KV (detach)] (无 prefix)
                    if _layer_kv_cache[layer_idx] is not None:
                        hist_k, hist_v = _layer_kv_cache[layer_idx]
                        _history_kv_len = hist_k.shape[2]
                        for i in range(layer_idx + 1):
                            dl = _DynamicLayer()
                            if i == layer_idx:
                                dl.keys = hist_k
                                dl.values = hist_v
                                dl.is_initialized = True
                            layer_cache.layers.append(dl)

                        _layer_attn_mask_raw = torch.ones(B, _history_kv_len + chunk_len, device=device, dtype=attention_mask.dtype)
                        if attention_mask.dim() == 2:
                            _layer_attn_mask_raw[:, :_history_kv_len] = attention_mask[:, :_history_kv_len]
                            _layer_attn_mask_raw[:, _history_kv_len:] = attention_mask[:, chunk_start:chunk_end]

                        _layer_attn_mask = create_causal_mask(
                            config=text_model.config,
                            input_embeds=chunk_embeds,
                            attention_mask=_layer_attn_mask_raw,
                            cache_position=chunk_cache_position,
                            past_key_values=layer_cache,
                            position_ids=chunk_text_pos,
                        )

                        hidden_states = decoder_layer(
                            hidden_states,
                            attention_mask=_layer_attn_mask,
                            position_ids=chunk_text_pos,
                            past_key_values=layer_cache,
                            cache_position=chunk_cache_position,
                            position_embeddings=chunk_pos_emb,
                        )
                    else:
                        # 第一个 chunk 的非注入层: 无历史 KV
                        # 传入空 DynamicCache 以收集当前 chunk 的 KV (供后续 chunk 使用)
                        for i in range(layer_idx + 1):
                            dl = _DynamicLayer()
                            layer_cache.layers.append(dl)

                        _layer_attn_mask = create_causal_mask(
                            config=text_model.config,
                            input_embeds=chunk_embeds,
                            attention_mask=attention_mask[:, chunk_start:chunk_end] if attention_mask.dim() == 2 else attention_mask,
                            cache_position=chunk_cache_position,
                            past_key_values=None,  # 第一个 chunk 无历史, 但 layer_cache 用于收集新 KV
                            position_ids=chunk_text_pos,
                        )
                        hidden_states = decoder_layer(
                            hidden_states,
                            attention_mask=_layer_attn_mask,
                            position_ids=chunk_text_pos,
                            past_key_values=layer_cache,  # 传入空 cache, decoder_layer 会写入新 KV
                            cache_position=chunk_cache_position,
                            position_embeddings=chunk_pos_emb,
                        )

                # ---- 更新 KV cache: 从 layer_cache 中提取当前 chunk 产生的 KV, 追加到历史 ----
                # decoder_layer 内部的 self_attn 会调用 past_key_values.update() 将新 KV 追加
                # 我们需要提取追加后的完整 KV, 然后 detach 保存
                if layer_cache.layers and len(layer_cache.layers) > layer_idx:
                    _updated_layer = layer_cache.layers[layer_idx]
                    if hasattr(_updated_layer, 'keys') and _updated_layer.keys is not None:
                        _full_k = _updated_layer.keys.detach()   # [B, H, total_len, D]
                        _full_v = _updated_layer.values.detach()  # [B, H, total_len, D]
                        if layer_idx in injection_set:
                            # 注入层: 去掉 prefix 部分, 只保留真实 token 的 KV
                            _layer_kv_cache[layer_idx] = (_full_k[:, :, N_p:, :], _full_v[:, :, N_p:, :])
                        else:
                            _layer_kv_cache[layer_idx] = (_full_k, _full_v)

                # DeepStack: 与原生实现一致
                if deepstack_for_fwd is not None and layer_idx in range(len(deepstack_for_fwd)):
                    if visual_pos_masks is not None:
                        chunk_visual_pos = visual_pos_masks[:, chunk_start:chunk_end]
                        if chunk_visual_pos.any():
                            full_visual_pos = visual_pos_masks
                            pre_chunk_visual_count = full_visual_pos[:, :chunk_start].sum().item()
                            chunk_visual_count = chunk_visual_pos.sum().item()
                            if chunk_visual_count > 0:
                                chunk_visual_embeds = deepstack_for_fwd[layer_idx][pre_chunk_visual_count:pre_chunk_visual_count + chunk_visual_count]
                                hidden_states = text_model._deepstack_process(
                                    hidden_states,
                                    chunk_visual_pos,
                                    chunk_visual_embeds,
                                )

            # 最终 norm (有梯度)
            chunk_normed = text_model.norm(hidden_states)
            all_chunk_hidden.append(chunk_normed)

            # ---- 如果不是最后一个 chunk, 提取 step summary 并更新 Z_d ----
            is_boundary_chunk = (chunk_idx < len(chunk_ranges) - 1) or (
                len(split_points) > 0 and chunk_end - 1 == split_points[-1]
            )
            chunk_end_token = chunk_end - 1
            is_at_boundary = chunk_end_token in set(split_points)

            if is_at_boundary:
                # 提取当前 chunk 中 gen_start 之后的 hidden 作为 step summary 输入
                seg_start_in_chunk = max(0, gen_start - chunk_start)
                seg_hidden = chunk_normed[:, seg_start_in_chunk:, :]  # [B, seg_len, H]

                # StepResampler: seg_hidden → S_c
                S_c = self.controller.step_resampler(seg_hidden)  # [B, K_t, d_z]
                step_summaries.append(S_c)

                # Per-sample update mask
                mask = torch.tensor(
                    [chunk_end_token in per_sample_sets[b] for b in range(B)],
                    dtype=torch.bool, device=device,
                )
                update_masks.append(mask)

                # ---- Controller update: S_c → T_c → LatentDraftFlow → Z_d_new ----
                T_prev = rld_state['T']
                visual_hidden_proj = rld_state.get('visual_hidden_proj')

                if self.controller.use_trace_updater:
                    T_c = self.controller.trace_ema(T_prev, S_c)
                    flow_query = T_c
                else:
                    T_c = S_c
                    flow_query = S_c

                # 收集 T_c (用于 contrastive learning)
                all_T_c.append(T_c)

                # 计算 commit_score (训练时用于 commit loss 监督)
                S_prev = rld_state.get('_last_S_c', torch.zeros_like(S_c))
                commit_score_val = self.controller.commit_gate(S_running=S_c, S_prev=S_prev, Z_d=Z_d_current)
                all_commit_scores.append(commit_score_val)

                # ---- 负样本: 从当前 boundary chunk 的前半段采样 ----
                _neg_seg_len = seg_hidden.shape[1]
                if _neg_seg_len >= 4:
                    _neg_half = max(2, _neg_seg_len // 2)
                    seg_hidden_neg_intra = chunk_normed[:, seg_start_in_chunk:seg_start_in_chunk + _neg_half, :]
                    with torch.no_grad():
                        S_neg_intra = self.controller.step_resampler(seg_hidden_neg_intra)
                    neg_commit_score_intra = self.controller.commit_gate(
                        S_running=S_neg_intra.detach(), S_prev=S_prev.detach(), Z_d=Z_d_current.detach()
                    )
                    all_neg_commit_scores.append(neg_commit_score_intra)

                # 方案 D: Latent Draft Flow (隐空间流)
                if visual_hidden_proj is not None:
                    prefix_source_new = self.controller.compute_prefix_source(
                        S_c=flow_query,
                        Z_d_prev=Z_d_current,
                        visual_hidden=visual_hidden_proj,
                    )
                else:
                    prefix_source_new = self.controller.compute_prefix_source(
                        S_c=flow_query,
                        Z_d_prev=Z_d_current,
                        visual_hidden=rld_state['Z_e'],
                    )

                # Per-sample 选择性更新
                if mask is not None:
                    float_mask = mask.to(dtype=Z_d_current.dtype).unsqueeze(-1).unsqueeze(-1)
                    prefix_source_new = float_mask * prefix_source_new + (1.0 - float_mask) * Z_d_current
                    T_c = float_mask * T_c + (1.0 - float_mask) * T_prev

                # 更新 rld_state
                rld_state = {
                    'Z_e': rld_state['Z_e'],  # 透传
                    'visual_hidden_proj': visual_hidden_proj,  # 透传
                    'Z_d_init': Z_d_init,                      # 透传
                    'Z_d': prefix_source_new,                  # 方案 D: latent draft flow 输出
                    'T': T_c,
                    'step_count': rld_state['step_count'] + 1,
                    '_last_S_c': S_c.detach(),
                }

                # 收集 per-step 秩指标 (轻量, 只在 detach 后计算)
                with torch.no_grad():
                    _per_step_ranks.append({
                        'Zd_rank': self.controller._effective_rank(prefix_source_new.detach()),
                        'Zd_cosim': self.controller._slot_cosine_similarity(prefix_source_new.detach()),
                        'T_rank': self.controller._effective_rank(T_c.detach()),
                        'Zd_norm': prefix_source_new.detach().norm(dim=-1).mean().item(),
                    })

                # Diversity loss (Z_d + S_c)
                tmp_state = {'Z_d': prefix_source_new, 'Z_e': rld_state['Z_e']}
                div_loss = div_loss + self.controller.compute_diversity_loss(tmp_state)
                # S_c diversity loss: 鼓励 step_queries 的 16 个 slot 输出分化
                # S_c 有效秩 ~1.4 说明 slot 严重坍塌, 需要直接正则化
                div_loss = div_loss + self.controller.diversity_reg(S_c) * self.controller.lambda_div

                # 更新 Z_d_current 为新的 prefix_source (下一个 chunk 使用)
                # 显式 cast 到基座 dtype: controller 输出可能是 float32 (Accelerate upcast),
                # 但 Z_d_current 需要传入 prefix_kv_projector → 基座模型, 必须是 bfloat16
                Z_d_current = prefix_source_new.to(_base_dtype)
                all_Z_d.append(Z_d_current)

                # ---- 更新注入层 KV cache 中的 prefix KV (覆写, 与推理时一致) ----
                # Z_d 更新后, 重新投影 prefix KV, 后续 chunk 的 attention 自然看到新 prefix
                # 注意: 这里不需要显式覆写, 因为每个 chunk 开头都会重新计算 prefix_kvs
                # 但 _layer_kv_cache 中存储的是不含 prefix 的历史 KV, prefix 在每次循环开头拼接

            else:
                # ---- 非 boundary chunk (Answer 区域): 计算 commit_score 作为负样本 (来源 2) ----
                # 负样本来源 1: 每个 boundary chunk 的前半段 (step 内部中间位置, 在上面 is_at_boundary 分支中采样)
                # 负样本来源 2: Answer 区域 (推理已结束, 不该再 commit)
                if len(step_summaries) > 0 and chunk_end > gen_start:
                    seg_start_in_chunk = max(0, gen_start - chunk_start)
                    seg_hidden_neg = chunk_normed[:, seg_start_in_chunk:, :]  # [B, seg_len, H]
                    if seg_hidden_neg.shape[1] > 0:
                        with torch.no_grad():
                            S_neg = self.controller.step_resampler(seg_hidden_neg)  # [B, K_t, d_z]
                        S_prev_neg = rld_state.get('_last_S_c', torch.zeros_like(S_neg))
                        neg_commit_score = self.controller.commit_gate(
                            S_running=S_neg.detach(), S_prev=S_prev_neg, Z_d=Z_d_current.detach()  # 方案 B: Z_d_current 是 prefix_source
                        )
                        all_neg_commit_scores.append(neg_commit_score)

        # 释放 KV cache 显存
        del _layer_kv_cache

        # 归一化 diversity loss 和 grounding loss
        if len(all_Z_d) > 1:
            div_loss = div_loss / max(len(all_Z_d) - 1, 1)
            grounding_loss = grounding_loss / max(len(all_Z_d) - 1, 1)

        # ====== Concat 所有 chunk 的 hidden → 完整序列 ======
        if len(all_chunk_hidden) == 1:
            adapted_hidden = all_chunk_hidden[0]
        else:
            adapted_hidden = torch.cat(all_chunk_hidden, dim=1)  # [B, seq_len, H]
        full_hidden = adapted_hidden

        # 保存 chunkwise 循环中收集的中间结果到 rld_state
        rld_state['_all_T_c'] = all_T_c
        rld_state['_all_commit_scores'] = all_commit_scores
        rld_state['_all_neg_commit_scores'] = all_neg_commit_scores

        _t_base_fwd_end = _time.time()

        # ====== 显存监控: chunkwise forward 完成后 ======
        if self._debug and device.type == 'cuda' and hasattr(self, '_mem_monitor_count') and (self._mem_monitor_count <= 2 or self._mem_monitor_count % 100 == 0):
            _allocated = torch.cuda.memory_allocated(device) / 1024**3
            _reserved = torch.cuda.memory_reserved(device) / 1024**3
            _max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"📊 [rank {_rank}] chunkwise forward 完成后显存: "
                  f"已分配={_allocated:.2f}GB, "
                  f"已预留={_reserved:.2f}GB, "
                  f"峰值={_max_allocated:.2f}GB, "
                  f"chunks={len(chunk_ranges)}, steps={len(split_points)}",
                  flush=True)

        # ====== 6b. Contrastive Draft Learning: 已弃用 (当前数据无 wrong_cot + correct_cot 配对) ======

        # ====== 7. [Chunkwise Prefix KV] 训推一致: 每个 chunk 使用对应 Z_d 的 prefix KV ======
        # 已在上面的 chunkwise forward 循环中实现:
        # - 每个 chunk 使用当前 Z_d_c 的 prefix KV 做 36 层 forward
        # - chunk 边界处: step summary → controller update → Z_d_{c+1} → 新 prefix KV
        # - 下一个 chunk 使用新的 prefix KV
        # 梯度路径: CE loss → logits → chunk_c hidden → prefix_KV_c → Z_d_c → controller
        # 每个 Z_d_c 都有来自 CE loss 的直接梯度 (训推一致)

        # ====== 调试: Prefix KV 注入效果验证 ======
        if self._debug and (not hasattr(self, '_debug_inject_count') or self._debug_inject_count < 1):
            if not hasattr(self, '_debug_inject_count'):
                self._debug_inject_count = 0
            self._debug_inject_count += 1
            with torch.no_grad():
                print("\n" + "=" * 70)
                print(f"🎯 [RLDModel.forward] Prefix KV 注入效果验证 (#{self._debug_inject_count})")
                print(f"   序列长度: {seq_len}, Step 数: {len(split_points)}, Z_d 版本数: {len(all_Z_d)}")
                print(f"   注入层: {self.injection_layer_indices}")
                print(f"   Prefix slots: {N_p}")
                print("=" * 70)

        # ====== 9. 计算 logits ======
        lm_head = self._lm_head
        logits = lm_head(adapted_hidden)  # [B, seq_len, V]

        # ====== 10. 计算 loss ======
        total_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        main_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)

        if labels is not None:
            # 标准 next-token prediction: shift logits 和 labels
            shift_logits = logits[:, :-1, :].contiguous().float()  # 显式 upcast 到 float32, 确保 backward 时 ToCopyBackward 将梯度转回 bfloat16
            shift_labels = labels[:, 1:].contiguous()

            # ====== P2: 加权 CE loss — answer token 获得更高权重 ======
            # 当 answer 很短 (如单字符 "B") 时, 有效监督 token 极少 (~2 个),
            # 而 think 块可能有 ~300 个有效 token, 导致 answer 的梯度信号被淹没。
            # loss_weight_mask 对 answer token 赋予更高权重, 使其梯度贡献与 think 块等量。
            if loss_weight_mask is not None:
                # shift weight mask 与 shift_labels 对齐 (去掉第一个 token)
                shift_weights = loss_weight_mask[:, 1:].contiguous().to(shift_logits.dtype)  # [B, seq_len-1]
                # 逐 token CE loss (reduction='none')
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  # [B*(seq_len-1)]
                per_token_loss = per_token_loss.view(shift_labels.shape)  # [B, seq_len-1]
                # 有效 token mask (labels != -100)
                valid_mask = (shift_labels != -100).float()  # [B, seq_len-1]
                # 加权 loss: 按 weight_mask 加权, 然后对有效 token 取加权均值
                weighted_loss = per_token_loss * shift_weights * valid_mask
                ce_loss = weighted_loss.sum() / (shift_weights * valid_mask).sum().clamp(min=1.0)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            main_loss = ce_loss.detach()

            # ====== Contrastive Draft Learning loss: 已弃用 ======

            # ====== Visual Grounding Loss: 方案 D 中不再使用 (无 G_c) ======
            # grounding_loss 保持为 0, 但保留变量以兼容下游 loss 汇总
            lambda_grounding = 0.0  # 方案 D: 禁用 grounding loss

            # ====== Commit Gate Loss: 正负样本联合监督 ======
            # 正样本: boundary 位置 (chunk 末尾) 的 commit_score → target=1.0 (应该 commit)
            # 负样本来源 (两类):
            #   1. 每个 boundary chunk 的前半段 (step 内部中间位置) → target=0.0 (推理还在进行, 不该 commit)
            #   2. 最后一个非 boundary chunk (Answer 区域) → target=0.0 (已经结束推理, 不该 commit)
            # 核心: 负样本主要来自 step 内部, 让 CommitGate 学会区分 "step 边界" vs "step 中间"
            commit_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
            lambda_commit = 0.1  # commit loss 权重 (不宜过大, 避免干扰主 loss)
            all_commit_scores = rld_state.get('_all_commit_scores', [])
            all_neg_commit_scores = rld_state.get('_all_neg_commit_scores', [])
            _n_pos = len(all_commit_scores)
            _n_neg = len(all_neg_commit_scores)
            if _n_pos + _n_neg > 0:
                # 注意: F.binary_cross_entropy 在 bf16 autocast 下被硬性禁止,
                # 必须在 autocast(enabled=False) 上下文中用 float32 计算
                with torch.amp.autocast('cuda', enabled=False):
                    # 正样本 loss: boundary 位置 → target=1.0
                    pos_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                    if _n_pos > 0:
                        commit_targets_pos = torch.ones(1, device=device, dtype=torch.float32)
                        for cs in all_commit_scores:
                            pos_loss = pos_loss + F.binary_cross_entropy(
                                cs.float(), commit_targets_pos.expand_as(cs),
                                reduction='mean'
                            )
                        pos_loss = pos_loss / _n_pos

                    # 负样本 loss: step 内部中间位置 + Answer 区域 → target=0.0
                    neg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                    if _n_neg > 0:
                        commit_targets_neg = torch.zeros(1, device=device, dtype=torch.float32)
                        for cs in all_neg_commit_scores:
                            neg_loss = neg_loss + F.binary_cross_entropy(
                                cs.float(), commit_targets_neg.expand_as(cs),
                                reduction='mean'
                            )
                        neg_loss = neg_loss / _n_neg

                    # 正负样本均衡加权: 现在每个 boundary chunk 同时贡献 1 正 + 1 负,
                    # 正负样本数量基本平衡, 使用 1:1 权重
                    commit_loss = (pos_loss + neg_loss) / 2.0

            total_loss = ce_loss + div_loss.float() + lambda_commit * commit_loss + lambda_grounding * grounding_loss.float()

        # ====== DEBUG: dtype 追踪 (定位 Expected BFloat16 got Float) ======
        if _rank == 0 and self._fwd_count <= 2:
            print(f"\n🔍 [DEBUG dtype] forward #{self._fwd_count}:", flush=True)
            print(f"  total_loss: dtype={total_loss.dtype}, grad_fn={total_loss.grad_fn}", flush=True)
            if labels is not None:
                print(f"  ce_loss: dtype={ce_loss.dtype}", flush=True)
                print(f"  div_loss: dtype={div_loss.dtype}", flush=True)
                print(f"  commit_loss: dtype={commit_loss.dtype}", flush=True)
                print(f"  logits: dtype={logits.dtype}", flush=True)
                print(f"  adapted_hidden: dtype={adapted_hidden.dtype}", flush=True)
                if len(all_Z_d) > 0:
                    print(f"  Z_d_current: dtype={Z_d_current.dtype}", flush=True)
                if len(all_chunk_hidden) > 0:
                    print(f"  chunk_hidden[0]: dtype={all_chunk_hidden[0].dtype}", flush=True)
                # 检查 prefix_kv_projector 参数 dtype
                _pkv_dtype = next(self.prefix_kv_projector.parameters()).dtype
                print(f"  prefix_kv_projector param dtype: {_pkv_dtype}", flush=True)
                # 检查 LoRA 参数 dtype
                if self._lora_enabled:
                    for name, param in self.base_model.named_parameters():
                        if 'lora_A' in name:
                            print(f"  LoRA param '{name}': dtype={param.dtype}", flush=True)
                            break

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
            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
            anchor = next(self.controller.parameters())
            total_loss = total_loss + (anchor.sum() * 0.0).to(total_loss.dtype)

        # Draft 状态监控
        _draft_metrics_accum = {}
        with torch.no_grad():
            _draft_metrics_accum = self.controller.compute_draft_metrics(
                state=rld_state,
            )
            _draft_metrics_accum['draft/num_steps'] = float(len(split_points))
            _draft_metrics_accum['draft/num_chunks'] = float(len(chunk_ranges))
            # Prefix KV 注入效果量化指标 (用于 TensorBoard)
            # 注意: 在 prefix KV 方案中, full_hidden 就是 adapted_hidden
            # 需要与无 prefix 的基座输出对比才有意义
            # 这里暂时记录 adapted_hidden 的范数作为基线
            _draft_metrics_accum['draft/adapted_hidden_norm'] = adapted_hidden.float().detach().norm(dim=-1).mean().item()

            # ====== 秩演化追踪 (per-step) ======
            if _per_step_ranks:
                _zd_ranks = [r['Zd_rank'] for r in _per_step_ranks]
                _t_ranks = [r['T_rank'] for r in _per_step_ranks]
                _zd_cosims = [r['Zd_cosim'] for r in _per_step_ranks]
                # 秩演化统计
                _draft_metrics_accum['rank/Zd_rank_first'] = _zd_ranks[0]
                _draft_metrics_accum['rank/Zd_rank_last'] = _zd_ranks[-1]
                _draft_metrics_accum['rank/Zd_rank_min'] = min(_zd_ranks)
                _draft_metrics_accum['rank/Zd_rank_max'] = max(_zd_ranks)
                _draft_metrics_accum['rank/Zd_rank_trend'] = _zd_ranks[-1] - _zd_ranks[0]  # 正=秩增长, 负=秩坍塌
                _draft_metrics_accum['rank/T_rank_last'] = _t_ranks[-1]
                _draft_metrics_accum['rank/T_rank_min'] = min(_t_ranks)
                # cosim 演化
                _draft_metrics_accum['rank/Zd_cosim_first'] = _zd_cosims[0]
                _draft_metrics_accum['rank/Zd_cosim_last'] = _zd_cosims[-1]
                _draft_metrics_accum['rank/Zd_cosim_trend'] = _zd_cosims[-1] - _zd_cosims[0]  # 正=趋向坍塌
                # 完整轨迹 (用于详细监控打印, 不写入 TensorBoard)
                _draft_metrics_accum['rank/_per_step_ranks'] = _per_step_ranks  # list of dict

            # Z_d_init 基线秩 (对比用)
            _draft_metrics_accum['rank/Zd_init_rank'] = self.controller._effective_rank(Z_d_init.detach())
            _draft_metrics_accum['rank/Zd_init_cosim'] = self.controller._slot_cosine_similarity(Z_d_init.detach())

            # ====== 方案 D: 隐空间流监控指标 ======
            # LayerScale: per-slot 可学习缩放因子 γ_i 的统计
            if hasattr(self.controller.latent_draft_flow, 'slot_scale'):
                _ss = self.controller.latent_draft_flow.slot_scale.detach().squeeze()  # [K_d]
                _draft_metrics_accum['draft/slot_scale_mean'] = _ss.mean().item()
                _draft_metrics_accum['draft/slot_scale_max'] = _ss.max().item()
                _draft_metrics_accum['draft/slot_scale_min'] = _ss.min().item()
                _draft_metrics_accum['draft/slot_scale_std'] = _ss.std().item()

            # TraceEMA: per-slot 可学习记忆保留率 α_i 的统计
            if hasattr(self.controller, 'trace_ema') and hasattr(self.controller.trace_ema, 'alpha_logit'):
                _alpha = torch.sigmoid(self.controller.trace_ema.alpha_logit.detach()).squeeze()  # [K_t]
                _draft_metrics_accum['draft/trace_ema_alpha_mean'] = _alpha.mean().item()
                _draft_metrics_accum['draft/trace_ema_alpha_max'] = _alpha.max().item()
                _draft_metrics_accum['draft/trace_ema_alpha_min'] = _alpha.min().item()
                _draft_metrics_accum['draft/trace_ema_alpha_std'] = _alpha.std().item()

            if rld_state.get('visual_hidden_proj') is not None:
                _vhp = rld_state['visual_hidden_proj'].detach()
                _draft_metrics_accum['draft/visual_proj_num_tokens'] = float(_vhp.shape[1])
                _draft_metrics_accum['draft/visual_proj_norm'] = _vhp.norm(dim=-1).mean().item()
                # Z_d 与 S_c 的余弦相似度 (验证 draft flow 的信息融合效果)
                if len(step_summaries) > 0:
                    _last_Zd = rld_state['Z_d'].detach().mean(dim=1)  # [B, d_z]
                    _last_S_c = step_summaries[-1].detach().mean(dim=1)  # [B, d_z]
                    _zd_sc_cosim = F.cosine_similarity(_last_Zd, _last_S_c, dim=-1).mean().item()
                    _draft_metrics_accum['draft/Zd_Sc_cosim'] = _zd_sc_cosim
                    _draft_metrics_accum['draft/Sc_norm'] = step_summaries[-1].detach().norm(dim=-1).mean().item()
                    _draft_metrics_accum['draft/Sc_effective_rank'] = self.controller._effective_rank(step_summaries[-1].detach())
                    _draft_metrics_accum['draft/Sc_slot_cosim'] = self.controller._slot_cosine_similarity(step_summaries[-1].detach())

            # ====== 新增: Per-token loss 分解 (think vs answer) + Top-K 准确率 ======
            # 帮助判断模型在推理步骤和最终答案上的学习进度
            if loss_weight_mask is not None and labels is not None:
                _shift_labels = labels[:, 1:].contiguous()
                _shift_weights = loss_weight_mask[:, 1:].contiguous().float()
                _valid = (_shift_labels != -100)

                # 区分 think token (weight ≈ 1.0) 和 answer token (weight > 1.0)
                _think_mask = _valid & (_shift_weights <= 1.01)  # think: weight=1.0
                _answer_mask = _valid & (_shift_weights > 1.01)  # answer: weight>1.0

                # Per-region loss (未加权的原始 CE loss)
                _loss_fct_none = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                _per_token_ce = _loss_fct_none(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    _shift_labels.view(-1)
                ).view(_shift_labels.shape)

                if _think_mask.any():
                    _draft_metrics_accum['loss/think_ce'] = (_per_token_ce * _think_mask.float()).sum().item() / _think_mask.float().sum().item()
                    _draft_metrics_accum['loss/think_token_count'] = _think_mask.float().sum().item()
                if _answer_mask.any():
                    _draft_metrics_accum['loss/answer_ce'] = (_per_token_ce * _answer_mask.float()).sum().item() / _answer_mask.float().sum().item()
                    _draft_metrics_accum['loss/answer_token_count'] = _answer_mask.float().sum().item()

                # Top-1 / Top-5 准确率 (在有效 token 上)
                if _valid.any():
                    _pred_top1 = logits[:, :-1, :].contiguous().argmax(dim=-1)  # [B, seq_len-1]
                    _top1_correct = (_pred_top1 == _shift_labels) & _valid
                    _draft_metrics_accum['acc/top1_overall'] = _top1_correct.float().sum().item() / _valid.float().sum().item()

                    if _think_mask.any():
                        _draft_metrics_accum['acc/top1_think'] = (_top1_correct & _think_mask).float().sum().item() / _think_mask.float().sum().item()
                    if _answer_mask.any():
                        _draft_metrics_accum['acc/top1_answer'] = (_top1_correct & _answer_mask).float().sum().item() / _answer_mask.float().sum().item()

                    # Top-5 准确率
                    _pred_top5 = logits[:, :-1, :].contiguous().topk(5, dim=-1).indices  # [B, seq_len-1, 5]
                    _top5_correct = (_pred_top5 == _shift_labels.unsqueeze(-1)).any(dim=-1) & _valid
                    _draft_metrics_accum['acc/top5_overall'] = _top5_correct.float().sum().item() / _valid.float().sum().item()

                    if _answer_mask.any():
                        _draft_metrics_accum['acc/top5_answer'] = (_top5_correct & _answer_mask).float().sum().item() / _answer_mask.float().sum().item()

            # ====== Contrastive Draft Learning 监控指标: 已弃用 ======
            _draft_metrics_accum['loss/grounding_loss'] = grounding_loss.item() if isinstance(grounding_loss, torch.Tensor) else 0.0

        # ====== forward 阶段耗时汇总 ======
        _fwd_end = _time.time()
        _fwd_total = _fwd_end - _fwd_start
        if _fwd_total > 30:
            _force_profile = True
        if _should_profile or _force_profile:
            _t_vision = _t_vision_end - _t_vision_start
            _t_base = _t_base_fwd_end - _t_base_fwd_start
            _t_rest = _fwd_total - _t_vision - _t_base
            _warn = " ⚠️ SLOW" if _fwd_total > 30 else ""
            print(f"[rank {_rank}] fwd#{self._fwd_count} 耗时: "
                  f"total={_fwd_total:.1f}s (vision={_t_vision:.1f}s, "
                  f"chunkwise_fwd={_t_base:.1f}s, "
                  f"ctrl+loss={_t_rest:.1f}s) "
                  f"chunks={len(chunk_ranges)} steps={len(split_points)} prefix_slots={N_p}{_warn}", flush=True)

        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'div_loss': div_loss.detach(),
            'commit_loss': commit_loss.detach() if isinstance(commit_loss, torch.Tensor) else torch.tensor(0.0),
            'grounding_loss': grounding_loss.detach() if isinstance(grounding_loss, torch.Tensor) else torch.tensor(0.0),
            'logits': None,  # 不返回完整 logits (太大)
            'rld_state': rld_state,
            'draft_metrics': _draft_metrics_accum,
        }

    # ================================================================
    # 推理流程: 训推一致的 prefix KV 注入模式
    # prefix KV 写入 DynamicCache 头部, 后续 token 自然看到
    # 遇到 step boundary 时 controller.step_update 更新 Z_d
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
        推理时的生成 (Persistent Prefix KV Memory 方案)
        
        核心设计:
        - 在 prefill 阶段, 将 Z_d 投影的 prefix KV 写入 DynamicCache 头部
        - 后续每个 token 的 self-attention 自然看到 prefix KV (标准 KV cache 行为)
        - 遇到 step boundary 时触发 controller.step_update 更新 Z_d
        - 更新后覆写 cache 中 prefix 位置的 KV (原地更新, 无需重建 cache)
        - 完全兼容 flash_attention_2
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. 视觉 encoder
        inner_model = self._inner_model
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

        # 3. 计算初始 prefix KV 并写入 DynamicCache
        N_p = self.num_prefix_slots
        Z_d_current = rld_state['Z_d']  # [B, K_d, d_z]
        prefix_kvs = self.prefix_kv_projector.get_all_prefix_kvs(Z_d_current)
        # prefix_kvs: {layer_idx: (prefix_k, prefix_v)}

        # 构建 DynamicCache, 预填充 prefix KV
        # 注入层: 写入真实 prefix KV
        # 非注入层: 写入零向量 (保持 cache 结构一致)
        text_model = inner_model.language_model
        lm_head = self._lm_head
        injection_set = set(self.injection_layer_indices)

        cache = DynamicCache()
        for layer_idx in range(self.total_layers):
            if layer_idx in injection_set:
                pfx_k, pfx_v = prefix_kvs[layer_idx]
            else:
                # 非注入层: 零向量 prefix (对 attention 影响极小)
                pfx_k = torch.zeros(B, self.num_kv_heads, N_p, self.head_dim,
                                    device=device, dtype=Z_d_current.dtype)
                pfx_v = torch.zeros(B, self.num_kv_heads, N_p, self.head_dim,
                                    device=device, dtype=Z_d_current.dtype)
            # 写入 cache (DynamicCache.update 会 cat 到已有 KV 后面)
            cache.update(pfx_k, pfx_v, layer_idx)

        # 4. Prompt prefill (prefix KV 已在 cache 中, cache_position 从 N_p 开始)
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

        # cache_position 从 N_p 开始 (prefix 占据 0~N_p-1)
        cache_position = torch.arange(prompt_len, device=device) + N_p

        # attention_mask 需要扩展: 前面加 N_p 个 1 (prefix 对所有 token 可见)
        prefix_attn = torch.ones(B, N_p, device=device, dtype=attention_mask.dtype)
        extended_attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)

        prefill_out = text_model(
            inputs_embeds=prompt_embeds,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=gen_deepstack_features,
        )
        past_key_values = prefill_out.past_key_values
        mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None

        # prefill 最后一个 token 的 logits
        prefill_hidden = prefill_out.last_hidden_state  # [B, prompt_len, H]
        next_token_logits = lm_head(prefill_hidden[:, -1, :])  # [B, V]

        # ====== Streaming Trace Accumulator: 用 prefill hidden 初始化 S_running ======
        # 用 prefill 最后一个 token 的 hidden 做一次初始累积
        rld_state = self.controller.streaming_accumulate(rld_state, prefill_hidden[:, -1:, :])

        # 句末标点 token ids (用于 CommitGate 的句末检测)
        _tokenizer = self.processor.tokenizer if hasattr(self, 'processor') else None
        _sentence_end_ids = set()
        if _tokenizer is not None:
            for punct in ['.', '。', '\n', ':', '：', ';', '；', '?', '？', '!', '！']:
                _ids = _tokenizer.encode(punct, add_special_tokens=False)
                _sentence_end_ids.update(_ids)

        # 5. 自回归生成
        generated_ids = input_ids.clone()
        eos_token_id = self.processor.tokenizer.eos_token_id if hasattr(self, 'processor') else None
        
        # 保留正则 boundary 检测作为 fallback (双保险)
        generated_text_buffers = ["" for _ in range(B)]
        triggered_steps = [set() for _ in range(B)]

        actual_cache_len = self._get_full_attn_cache_len(past_key_values)
        current_pos = actual_cache_len

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

            # prefix KV 已在 cache 中, self-attention 自然看到 → hidden 已受 prefix 影响
            next_token_logits = lm_head(hidden_states[:, -1, :])  # [B, V]

            mrope_last_pos = new_mrope_pos

            # ====== Streaming Trace Accumulator: 每 token 增量更新 S_running ======
            rld_state = self.controller.streaming_accumulate(rld_state, hidden_states)

            # ====== CommitGate: 只在句末计算 commit score ======
            # 检测当前 token 是否为句末标记
            is_sentence_end = torch.zeros(B, dtype=torch.bool, device=device)
            for b in range(B):
                if next_token[b].item() in _sentence_end_ids:
                    is_sentence_end[b] = True

            # 同时维护正则 boundary 检测作为 fallback
            for b in range(B):
                token_text = self.processor.tokenizer.decode([next_token[b].item()])
                generated_text_buffers[b] += token_text

            # 正则 fallback: 检测 "Step N:" 和 "Final Answer:"
            regex_trigger_mask = torch.zeros(B, dtype=torch.bool, device=device)
            for b in range(B):
                buf = generated_text_buffers[b]
                for m in re.finditer(r'Step\s+(\d+)\s*[:.]\s*', buf, re.IGNORECASE):
                    step_num = int(m.group(1))
                    if step_num >= 2 and step_num not in triggered_steps[b]:
                        triggered_steps[b].add(step_num)
                        regex_trigger_mask[b] = True
                if re.search(r'Final\s+Answer\s*:', buf, re.IGNORECASE) and 'final_answer' not in triggered_steps[b]:
                    triggered_steps[b].add('final_answer')
                    regex_trigger_mask[b] = True

            # CommitGate 触发判断 (只在句末计算)
            trigger_mask = torch.zeros(B, dtype=torch.bool, device=device)
            tokens_since_commit = rld_state.get('tokens_since_commit', 0)

            if is_sentence_end.any() or tokens_since_commit >= self.controller.commit_gate.max_tokens:
                # 计算 commit score
                commit_score = self.controller.compute_commit_score(rld_state)
                # 综合判断
                trigger_mask = self.controller.commit_gate.should_commit(
                    commit_score, tokens_since_commit, is_sentence_end
                )

            # 合并: CommitGate 触发 OR 正则 fallback 触发
            trigger_mask = trigger_mask | regex_trigger_mask

            if trigger_mask.any():
                # 使用 commit_update (基于 S_running) 替代 step_update (基于 raw hidden)
                rld_state = self.controller.commit_update(
                    rld_state, update_mask=trigger_mask
                )

                # ★ 核心: 覆写 cache 中 prefix 位置的 KV (原地更新)
                # 这是 Persistent Prefix Memory 的关键操作:
                # Z_d 更新后, 重新投影 prefix KV, 覆写 cache 的前 N_p 个位置
                # 后续 token 的 attention 自然看到新的 prefix KV
                new_prefix_kvs = self.prefix_kv_projector.get_all_prefix_kvs(rld_state['Z_d'])
                for layer_idx in self.injection_layer_indices:
                    new_pfx_k, new_pfx_v = new_prefix_kvs[layer_idx]
                    # 覆写 cache 中位置 0~N_p-1 的 KV
                    # 使用 .layers[layer_idx] 直接访问 DynamicLayer 对象
                    past_key_values.layers[layer_idx].keys[:, :, :N_p, :] = new_pfx_k
                    past_key_values.layers[layer_idx].values[:, :, :N_p, :] = new_pfx_v

                # 清理正则 fallback 的文本缓冲区
                for b in range(B):
                    if trigger_mask[b]:
                        generated_text_buffers[b] = generated_text_buffers[b][-20:]

            current_pos += 1

        return generated_ids

    # ================================================================
    # 保存/加载
    # ================================================================

    def save_pretrained(self, save_dir: str):
        """保存 RLD Controller + PrefixKVProjector + LoRA 参数 (兼容 ZeRO-3)"""
        os.makedirs(save_dir, exist_ok=True)
        controller_path = os.path.join(save_dir, "rld_controller.pt")
        adapter_path = os.path.join(save_dir, "rld_readout_adapter.pt")  # 文件名保持兼容旧 checkpoint

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
                for name, param in self.prefix_kv_projector.named_parameters():
                    with deepspeed.zero.GatheredParameters(param):
                        if self._verbose:
                            adapter_state[name] = param.data.cpu().clone()
                if self._verbose:
                    torch.save(controller_state, controller_path)
                    torch.save(adapter_state, adapter_path)
                    print(f"[RLD] Controller 已保存到 {controller_path} (ZeRO-3 gathered)")
                    print(f"[RLD] PrefixKVProjector 已保存到 {adapter_path} (ZeRO-3 gathered)")
                # LoRA 保存 (ZeRO-3)
                if self._lora_enabled and self._verbose:
                    lora_dir = os.path.join(save_dir, "lora_adapter")
                    self.base_model.save_pretrained(lora_dir)
                    print(f"[RLD] LoRA adapter 已保存到 {lora_dir} (ZeRO-3)")
                return
        except ImportError:
            pass

        if self._verbose:
            torch.save(self.controller.state_dict(), controller_path)
            torch.save(self.prefix_kv_projector.state_dict(), adapter_path)
            print(f"[RLD] Controller 已保存到 {controller_path}")
            print(f"[RLD] PrefixKVProjector 已保存到 {adapter_path}")
            
            # 保存 LoRA adapter
            if self._lora_enabled:
                lora_dir = os.path.join(save_dir, "lora_adapter")
                self.base_model.save_pretrained(lora_dir)
                print(f"[RLD] LoRA adapter 已保存到 {lora_dir}")

    def load_pretrained(self, save_dir: str, skip_lora: bool = False):
        """加载 RLD Controller + PrefixKVProjector + LoRA 参数
        
        Args:
            save_dir: checkpoint 目录路径
            skip_lora: 是否跳过 LoRA 权重加载 (Stage 2 LoRA 结构不同时设为 True)
        """
        controller_path = os.path.join(save_dir, "rld_controller.pt")
        if os.path.exists(controller_path):
            state_dict = torch.load(controller_path, map_location="cpu")
            # strict=False: 兼容旧 checkpoint (可能缺少 _gc_inject_gate_logit 等新参数)
            missing, unexpected = self.controller.load_state_dict(state_dict, strict=False)
            if self._verbose:
                print(f"[RLD] Controller 已从 {controller_path} 加载")
                if missing:
                    print(f"[RLD]   新增参数 (使用默认值): {missing}")
                if unexpected:
                    print(f"[RLD]   旧参数 (已忽略): {unexpected}")
        else:
            if self._verbose:
                print(f"[RLD] ⚠️ 未找到 Controller 权重: {controller_path}")

        adapter_path = os.path.join(save_dir, "rld_readout_adapter.pt")  # 文件名保持兼容旧 checkpoint
        if os.path.exists(adapter_path):
            adapter_state = torch.load(adapter_path, map_location="cpu")
            self.prefix_kv_projector.load_state_dict(adapter_state)
            if self._verbose:
                print(f"[RLD] PrefixKVProjector 已从 {adapter_path} 加载")
        else:
            if self._verbose:
                print(f"[RLD] ⚠️ 未找到 PrefixKVProjector 权重: {adapter_path}")

        # 将 controller 和 prefix_kv_projector 转换到与 base_model 相同的 dtype/device
        # (load_state_dict 从 CPU float32 加载, 需要重新对齐)
        target_dtype = next(self.base_model.parameters()).dtype
        target_device = next(self.base_model.parameters()).device
        self.controller = self.controller.to(dtype=target_dtype, device=target_device)
        self.prefix_kv_projector = self.prefix_kv_projector.to(dtype=target_dtype, device=target_device)
        if self._verbose:
            print(f"[RLD] Controller & PrefixKVProjector 已转换到 dtype={target_dtype}, device={target_device}")
        
        # 加载 LoRA adapter (如果存在且未要求跳过)
        lora_dir = os.path.join(save_dir, "lora_adapter")
        if skip_lora:
            if self._verbose:
                print(f"[RLD] ⏭️ 跳过 LoRA 权重加载 (skip_lora=True, Stage 2 LoRA 结构不同)")
        elif os.path.exists(lora_dir):
            if PEFT_AVAILABLE:
                from peft import PeftModel as _PeftModel
                if self._lora_enabled:
                    # setup_lora() 已调用, base_model 已是 PeftModel
                    # 直接加载权重到已有的 LoRA 结构中, 避免双重包装
                    import safetensors.torch
                    adapter_weights_path = os.path.join(lora_dir, "adapter_model.safetensors")
                    if os.path.exists(adapter_weights_path):
                        lora_state = safetensors.torch.load_file(adapter_weights_path)
                        missing, unexpected = self.base_model.load_state_dict(lora_state, strict=False)
                        if self._verbose:
                            print(f"[RLD] LoRA 权重已加载到已有 PeftModel (from {adapter_weights_path})")
                            if missing:
                                # 过滤掉非 lora 的 missing keys (正常情况)
                                lora_missing = [k for k in missing if 'lora_' in k]
                                if lora_missing:
                                    print(f"[RLD]   LoRA missing keys: {lora_missing}")
                    else:
                        # 尝试 .bin 格式
                        adapter_bin_path = os.path.join(lora_dir, "adapter_model.bin")
                        if os.path.exists(adapter_bin_path):
                            lora_state = torch.load(adapter_bin_path, map_location="cpu")
                            self.base_model.load_state_dict(lora_state, strict=False)
                            if self._verbose:
                                print(f"[RLD] LoRA 权重已加载到已有 PeftModel (from {adapter_bin_path})")
                else:
                    # setup_lora() 未调用, 用 PeftModel.from_pretrained 加载
                    self.base_model = _PeftModel.from_pretrained(
                        self.base_model, lora_dir
                    )
                    self._lora_enabled = True
                    if self._verbose:
                        print(f"[RLD] LoRA adapter 已从 {lora_dir} 加载 (PeftModel.from_pretrained)")
            else:
                if self._verbose:
                    print(f"[RLD] ⚠️ 发现 LoRA 权重但 peft 未安装: {lora_dir}")
