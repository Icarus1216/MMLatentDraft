#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_embeddings_with_native.py
=================================
方法学升级: 对同一批输入, 抽取 *两次前向* 的 hidden states:
  (A) Latent 通路   — NativeLatentThinker 参与, 记录 thinker 每步 thought_output
  (B) Native 通路   — 跳过 Thinker, 遇到 <|latent|> 当普通 token, 记录相同位置
      和相同 token 预算下的 LLM 原生 hidden states

这样得到的配对数据, 允许做 *paired* statistical tests:
    - ΔCone   = cone(H_latent) - cone(H_native)
    - ΔID     = intrinsic_dim(H_latent) - intrinsic_dim(H_native)
    - ΔPR     = PR(H_latent) - PR(H_native)
    - Δθ_V    = angle(H_latent, V_center) - angle(H_native, V_center)

也同时抽取 V (vision pooled) 和 L (prompt text pooled), 保证以后的几何分析
(modality gap / cone half-angle) 不需要重跑前向.

可选 layer-wise: 用 output_hidden_states=True 抽取中间层 (layer 0/8/16/24/last)
这样 Stage-2 "layer-wise 几何演化" 图可以直接从这份数据出, 不用二次抽取.

输出 schema (.pt):
{
  "V":                list[N] of Tensor (D,)                # vision pooled per sample
  "L":                list[N] of Tensor (D,)                # prompt text pooled per sample
  "H_latent":         list[N] of Tensor (T_i, D)            # T_i 个 latent step (通常 1~2)
  "H_native_latent_pos": list[N] of Tensor (T_i, D)         # 同一 token 位置的 native hidden
  "H_native_answer":  list[N] of Tensor (D,)                # native answer-start hidden
  "H_latent_layers":  list[N] of Tensor (T_i, L, D)         # 可选; layer-wise (L 层)
  "H_native_layers":  list[N] of Tensor (T_i, L, D)         # 可选
  "alpha":            list[N] of Tensor (T_i,)              # latent 每步饱和度 (exit_stats)
  "meta": {
      "num_samples":  N,
      "dim":          D,
      "layers":       [0, 8, 16, 24, -1],        # -1 表示 last layer
      "model_path":   str,
      "checkpoint":   str,
      "data_file":    str,
      "ts":           str,
      "per_sample":   list[{qid, question_type, num_latent_steps, img_path}]
  }
}

用法:
  bash modality_analysis/scripts/run_extract_embeddings.sh
  bash modality_analysis/scripts/run_extract_embeddings.sh --num_samples 500
  bash modality_analysis/scripts/run_extract_embeddings.sh --layers 0,8,16,24,-1
"""
import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


# ==================================================================
# 0. 默认路径
# ==================================================================
DEFAULT_MODEL_PATH = "/mnt/wfs/mmchongqingssdwfssz/project_luban_infra/luban_infra/model_factory/for_lucas_Qwen3-VL-8B-Instruct_export"
DEFAULT_CHECKPOINT = "./outputs/rld_stage2/model"
DEFAULT_DATA_FILE = "./data/erqa/erqa_test.jsonl"
DEFAULT_DATA_ROOT = "./data/erqa"
DEFAULT_OUTPUT = "./modality_analysis/results/embeddings_native_vs_latent.pt"
DEFAULT_LAYERS = "0,8,16,24,-1"
DEFAULT_NUM_SAMPLES = 500
DEFAULT_MAX_NEW_TOKENS = 256  # 抽取用, 不需要很长
DTYPE = torch.bfloat16
DEVICE = "cuda"


# ==================================================================
# 1. Prefill & cache 辅助
#    (基于 diagnose_latent_ab.py 的 _prepare_inputs / _clone_cache, 但精简)
# ==================================================================

def prepare_inputs_for_extraction(model, inputs):
    """
    做一次完整 prefill, 返回:
      - shared (共享的 prefill 状态, 供 latent 和 native 两个通路各自复用)
      - V (vision pooled from prefill)
      - L (text pooled from prefill, 排除 vision token)
    """
    from transformers.cache_utils import DynamicCache
    from rld.model_v2 import _extract_visual_output

    B = inputs['input_ids'].shape[0]
    device = inputs['input_ids'].device
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    pixel_values = inputs['pixel_values']
    image_grid_thw = inputs['image_grid_thw']

    inner_model = model._inner_model
    text_model = inner_model.language_model
    lm_head = model._lm_head

    # 1) Vision encoder
    _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
    vision_output = inner_model.visual(_pixel_values, grid_thw=image_grid_thw)
    merged_hidden_states, gen_deepstack_features = _extract_visual_output(vision_output)
    split_sizes = (
        image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
    ).tolist()
    image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))
    cached_visual_embeds = torch.cat(image_embeds_list, dim=0)

    # 2) 构建 prompt embeds (把 image token 位置替换为视觉特征)
    prompt_embeds = inner_model.get_input_embeddings()(input_ids)
    image_token_id = inner_model.config.image_token_id
    image_mask = (input_ids == image_token_id)
    num_image_tokens = image_mask.sum().item()
    if num_image_tokens > 0:
        img_emb = cached_visual_embeds[:num_image_tokens].to(prompt_embeds.device, prompt_embeds.dtype)
        image_mask_3d = image_mask.unsqueeze(-1).expand_as(prompt_embeds)
        prompt_embeds = prompt_embeds.masked_scatter(image_mask_3d, img_emb)

    # 3) Position ids
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
    # 开 output_hidden_states, 这样 prefill 也给出 layer-wise hidden (用于 V/L pool)
    prefill_out = text_model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=gen_deepstack_features,
        output_hidden_states=True,
    )

    shared = {
        'B': B,
        'device': device,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'prompt_len': prompt_len,
        'position_ids': position_ids,
        'prefill_out': prefill_out,
        'cached_visual_embeds': cached_visual_embeds,
        'gen_deepstack_features': gen_deepstack_features,
        'inner_model': inner_model,
        'text_model': text_model,
        'lm_head': lm_head,
        'image_token_id': image_token_id,
        'image_mask_prompt': image_mask,
    }
    return shared


def pool_V_and_L(shared):
    """
    从 prefill 的 last_hidden_state 做 mean pool:
      V = mean of hidden at vision token positions
      L = mean of hidden at non-vision non-pad token positions
    返回 (V_vec, L_vec), 各自 shape [H]
    """
    prefill_hidden = shared['prefill_out'].last_hidden_state  # [B, T, H]
    image_mask = shared['image_mask_prompt']  # [B, T] bool
    attn_mask = shared['attention_mask']      # [B, T]

    B = prefill_hidden.shape[0]
    assert B == 1, "当前 extraction 默认 B=1"

    h = prefill_hidden[0]        # [T, H]
    img_m = image_mask[0].bool() # [T]
    txt_m = (~img_m) & (attn_mask[0].bool())  # 非 image 且有效 token
    # 进一步排除前面几个 chat_template 的控制 token (保守做法: 排除 BOS)
    # 这里不做精细排除, 用全部非 image 有效 token 即可, pooling 稳健

    if img_m.any():
        V = h[img_m].mean(dim=0)
    else:
        V = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)

    if txt_m.any():
        L = h[txt_m].mean(dim=0)
    else:
        L = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)

    return V.detach().cpu(), L.detach().cpu()


def pool_V_and_L_layerwise(shared, layers: List[int]):
    """
    对 prefill.hidden_states (tuple of [B, T, H]) 做 layer-wise V/L 池化.
    返回 V_layers (len(layers), H), L_layers (len(layers), H).
    """
    hidden_tuple = shared['prefill_out'].hidden_states  # tuple of L+1 tensors
    image_mask = shared['image_mask_prompt']
    attn_mask = shared['attention_mask']

    img_m = image_mask[0].bool()
    txt_m = (~img_m) & (attn_mask[0].bool())

    Vs, Ls = [], []
    n_layers_total = len(hidden_tuple)
    for li in layers:
        idx = li if li >= 0 else n_layers_total + li  # -1 -> last
        idx = max(0, min(idx, n_layers_total - 1))
        h = hidden_tuple[idx][0]  # [T, H]
        V = h[img_m].mean(dim=0) if img_m.any() else torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
        L = h[txt_m].mean(dim=0) if txt_m.any() else torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
        Vs.append(V.detach().cpu())
        Ls.append(L.detach().cpu())
    V_layers = torch.stack(Vs, dim=0)  # [layers, H]
    L_layers = torch.stack(Ls, dim=0)
    return V_layers, L_layers


def clone_cache(cache):
    """深拷贝 DynamicCache (来自 diagnose_latent_ab.py)."""
    from transformers.cache_utils import DynamicCache
    try:
        from transformers.cache_utils import DynamicLayer as _DL
    except ImportError:
        _DL = None
    cloned = DynamicCache()
    for layer in cache.layers:
        if _DL is not None:
            new_layer = _DL()
            if hasattr(layer, 'keys') and layer.keys is not None:
                new_layer.keys = layer.keys.clone()
                new_layer.values = layer.values.clone()
                new_layer.is_initialized = True
            cloned.layers.append(new_layer)
        else:
            cloned.layers.append(layer)
    return cloned


# ==================================================================
# 2. Latent 通路 — 基于 generate_with_latent, 但只记录 thinker 每步 thought_output
# ==================================================================

def forward_latent_path(
    model, processor, shared,
    max_new_tokens: int,
    layers: Optional[List[int]] = None,
):
    """
    运行 Latent 通路, 返回:
      {
        'H_latent':   Tensor (T, D)    thinker 每步 thought_output (last layer)
        'H_latent_layers': Tensor (T, L, D)   layer-wise (若 layers 非空)
        'alpha':      Tensor (T,)      每步 saturation (作为 alpha 的代理)
        'num_latent_steps': int T
        'latent_trigger_pos': int | None   第一次 <|latent|> 触发时的 cache_position
      }
    说明:
      thinker 内部的逐步 hidden (thought_output) 已是 last-layer 输出,
      若需要 layer-wise 则重新用 text_model(output_hidden_states=True) 回跑
      thought_output 一次, 读取中间层.
    """
    B = shared['B']
    device = shared['device']
    input_ids = shared['input_ids']
    attention_mask = shared['attention_mask']
    prompt_len = shared['prompt_len']
    position_ids = shared['position_ids']
    prefill_out = shared['prefill_out']
    cached_visual_embeds = shared['cached_visual_embeds']
    gen_deepstack_features = shared['gen_deepstack_features']
    inner_model = shared['inner_model']
    text_model = shared['text_model']
    lm_head = shared['lm_head']

    past_key_values = clone_cache(prefill_out.past_key_values)
    mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None
    prefill_hidden = prefill_out.last_hidden_state
    next_token_logits = lm_head(prefill_hidden[:, -1, :])
    current_pos = prompt_len
    latent_token_id = model.latent_token_id

    generated_token_list = []
    H_latent_all = None          # (T, D)
    H_latent_layers_all = None   # (T, L, D) or None
    alphas_all = None            # (T,)
    trigger_pos = None
    eos_token_id = processor.tokenizer.eos_token_id

    for gen_step in range(max_new_tokens):
        next_token = next_token_logits.argmax(dim=-1, keepdim=True).squeeze(-1)  # [B]
        generated_token_list.append(next_token[0].item())

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        new_mrope_pos = mrope_last_pos + 1
        text_pos_gen = torch.full((B, 1), current_pos, device=device, dtype=torch.long)
        position_ids_4d = torch.cat([text_pos_gen.unsqueeze(0), new_mrope_pos], dim=0)

        token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))
        total_len = current_pos + 1
        new_attn_mask = torch.ones(B, total_len, device=device, dtype=attention_mask.dtype)
        cache_position_gen = torch.tensor([current_pos], device=device, dtype=torch.long)

        text_outputs = text_model(
            inputs_embeds=token_embeds,
            attention_mask=new_attn_mask,
            position_ids=position_ids_4d,
            past_key_values=past_key_values,
            cache_position=cache_position_gen,
            use_cache=True,
        )
        past_key_values = text_outputs.past_key_values
        new_hidden = text_outputs.last_hidden_state
        next_token_logits = lm_head(new_hidden[:, -1, :])
        mrope_last_pos = new_mrope_pos
        current_pos += 1

        # 触发 thinker
        token_id = next_token[0].item()
        if latent_token_id is not None and token_id == latent_token_id:
            trigger_pos = current_pos  # 记录触发点 cache_position

            last_hidden = new_hidden  # [B, 1, H]
            base_thought_pos = current_pos
            thought_pos = torch.arange(current_pos, current_pos + 1, device=device)
            thought_mrope_pos = thought_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
            thought_text_pos = torch.full((B, 1), base_thought_pos, device=device, dtype=torch.long)
            thought_position_ids = torch.cat([thought_text_pos.unsqueeze(0), thought_mrope_pos], dim=0)
            think_attn_mask = torch.ones(B, base_thought_pos + 1, device=device, dtype=torch.long)

            thinker_result = model.latent_thinker(
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
            thought_output = thinker_result['thought_output']  # [B, T, H]
            num_steps = thinker_result.get('num_thought_steps', thought_output.shape[1])
            exit_stats = thinker_result.get('exit_stats', {})
            sats = exit_stats.get('saturations', [])

            # 每步 alpha (saturation)
            alpha_list = []
            for s in sats[:num_steps]:
                v = s[0].item() if hasattr(s, 'dim') and s.dim() > 0 else float(s)
                alpha_list.append(v)
            # 填齐到 num_steps (若 saturations 少于 num_steps)
            while len(alpha_list) < num_steps:
                alpha_list.append(float('nan'))
            alphas_all = torch.tensor(alpha_list[:num_steps], dtype=torch.float32)

            H_latent_all = thought_output[0].detach().float().cpu()  # [T, H]

            # layer-wise: 把 thought_output 作为 prefix 再过一次 text_model 拿层状态
            if layers is not None:
                prefix_cache_pos = torch.arange(
                    current_pos, current_pos + num_steps, device=device
                )
                prefix_mrope = prefix_cache_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                prefix_text_pos = torch.arange(
                    current_pos, current_pos + num_steps, device=device
                ).unsqueeze(0).expand(B, -1)
                prefix_pos_ids = torch.cat([prefix_text_pos.unsqueeze(0), prefix_mrope], dim=0)
                prefix_attn_mask = torch.ones(
                    B, current_pos + num_steps, device=device, dtype=attention_mask.dtype
                )
                # 独立 cache, 不污染主流
                layer_cache = clone_cache(past_key_values)
                prefix_out = text_model(
                    inputs_embeds=thought_output,
                    attention_mask=prefix_attn_mask,
                    position_ids=prefix_pos_ids,
                    past_key_values=layer_cache,
                    cache_position=prefix_cache_pos,
                    use_cache=True,
                    output_hidden_states=True,
                )
                hidden_tuple = prefix_out.hidden_states  # tuple of L+1 tensors
                n_layers_total = len(hidden_tuple)
                selected = []
                for li in layers:
                    idx = li if li >= 0 else n_layers_total + li
                    idx = max(0, min(idx, n_layers_total - 1))
                    selected.append(hidden_tuple[idx][0])  # [T, H]
                # stack -> [T, L, H]
                H_latent_layers_all = torch.stack(selected, dim=1).detach().float().cpu()

            # 抽完 thinker 即可终止生成 (只要 latent 部分)
            break

    return {
        'H_latent': H_latent_all,                   # [T, H] or None
        'H_latent_layers': H_latent_layers_all,     # [T, L, H] or None
        'alpha': alphas_all,                        # [T] or None
        'num_latent_steps': int(alphas_all.shape[0]) if alphas_all is not None else 0,
        'latent_trigger_pos': trigger_pos,
    }


# ==================================================================
# 3. Native 通路 — 跳过 thinker, 记录同一 token 位置的 LLM 原生 hidden
# ==================================================================

def forward_native_path(
    model, processor, shared,
    max_new_tokens: int,
    num_latent_steps_target: int,
    layers: Optional[List[int]] = None,
    latent_trigger_pos: Optional[int] = None,
):
    """
    复用同一 prefill cache, 但遇到 <|latent|> 当普通 token 继续自回归.
    抽取:
      (a) H_native_latent_pos: 在 <|latent|> 触发处(若存在)及其后 num_latent_steps_target 个 token 位置
          的 last-layer hidden state. 若本通路根本不生成 <|latent|>, 则退化为 answer-start hidden.
      (b) H_native_layers:     同一批位置的 layer-wise hidden.
      (c) H_native_answer:     final answer 的第一个 token 位置的 last-layer hidden.
    """
    B = shared['B']
    device = shared['device']
    input_ids = shared['input_ids']
    attention_mask = shared['attention_mask']
    prompt_len = shared['prompt_len']
    position_ids = shared['position_ids']
    prefill_out = shared['prefill_out']
    inner_model = shared['inner_model']
    text_model = shared['text_model']
    lm_head = shared['lm_head']

    past_key_values = clone_cache(prefill_out.past_key_values)
    mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None
    prefill_hidden = prefill_out.last_hidden_state
    next_token_logits = lm_head(prefill_hidden[:, -1, :])
    current_pos = prompt_len
    latent_token_id = model.latent_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    # 存储目标
    native_latent_pos_hids = []      # list of [H] tensors (last layer)
    native_latent_pos_hids_layers = []  # list of [L, H] tensors
    H_native_answer = None
    H_native_answer_layers = None
    capture_n = num_latent_steps_target  # 要捕获的 native hidden 数量
    capturing = False
    captured = 0
    # 如果根本没遇到 <|latent|>, 退化方案: 在第一个 answer token 开始捕获 capture_n 个
    saw_latent_trigger = False
    answer_start_seen = False

    for gen_step in range(max_new_tokens):
        next_token = next_token_logits.argmax(dim=-1, keepdim=True).squeeze(-1)  # [B]

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        new_mrope_pos = mrope_last_pos + 1
        text_pos_gen = torch.full((B, 1), current_pos, device=device, dtype=torch.long)
        position_ids_4d = torch.cat([text_pos_gen.unsqueeze(0), new_mrope_pos], dim=0)

        token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))
        total_len = current_pos + 1
        new_attn_mask = torch.ones(B, total_len, device=device, dtype=attention_mask.dtype)
        cache_position_gen = torch.tensor([current_pos], device=device, dtype=torch.long)

        # 按需开 layer-wise
        need_layers = (layers is not None) and (
            capturing or
            (latent_token_id is not None and next_token[0].item() == latent_token_id) or
            (H_native_answer is None)
        )

        text_outputs = text_model(
            inputs_embeds=token_embeds,
            attention_mask=new_attn_mask,
            position_ids=position_ids_4d,
            past_key_values=past_key_values,
            cache_position=cache_position_gen,
            use_cache=True,
            output_hidden_states=need_layers,
        )
        past_key_values = text_outputs.past_key_values
        new_hidden = text_outputs.last_hidden_state  # [B, 1, H]

        # --- 捕获逻辑 ---
        # (1) 如果看到 <|latent|>, 从这个位置开始捕获 num_latent_steps_target 个 hidden
        token_id = next_token[0].item()
        if (not saw_latent_trigger) and latent_token_id is not None and token_id == latent_token_id:
            saw_latent_trigger = True
            capturing = True
            captured = 0

        if capturing and captured < capture_n:
            native_latent_pos_hids.append(new_hidden[0, 0].detach().float().cpu())
            if need_layers and text_outputs.hidden_states is not None:
                htuple = text_outputs.hidden_states
                nL = len(htuple)
                sel = []
                for li in layers:
                    idx = li if li >= 0 else nL + li
                    idx = max(0, min(idx, nL - 1))
                    sel.append(htuple[idx][0, 0])
                native_latent_pos_hids_layers.append(
                    torch.stack(sel, dim=0).detach().float().cpu()
                )
            captured += 1
            if captured >= capture_n:
                capturing = False

        # (2) 第一个 non-special non-latent token 作为 answer_start 记录 H_native_answer
        if H_native_answer is None and token_id != latent_token_id and token_id != eos_token_id:
            # 只在此时记录一次
            if current_pos >= prompt_len:  # sanity
                H_native_answer = new_hidden[0, 0].detach().float().cpu()
                if need_layers and text_outputs.hidden_states is not None:
                    htuple = text_outputs.hidden_states
                    nL = len(htuple)
                    sel = []
                    for li in layers:
                        idx = li if li >= 0 else nL + li
                        idx = max(0, min(idx, nL - 1))
                        sel.append(htuple[idx][0, 0])
                    H_native_answer_layers = torch.stack(sel, dim=0).detach().float().cpu()

        next_token_logits = lm_head(new_hidden[:, -1, :])
        mrope_last_pos = new_mrope_pos
        current_pos += 1

        # 早退: 已捕获完 latent_pos 且 answer_start 已记录
        if (not capturing) and saw_latent_trigger and H_native_answer is not None \
                and len(native_latent_pos_hids) >= capture_n:
            break

    # 如果整条生成都没见到 <|latent|>, 退化: native_latent_pos_hids 用 answer-start 重复
    if not saw_latent_trigger:
        if H_native_answer is not None:
            native_latent_pos_hids = [H_native_answer.clone() for _ in range(capture_n)]
            if H_native_answer_layers is not None:
                native_latent_pos_hids_layers = [
                    H_native_answer_layers.clone() for _ in range(capture_n)
                ]

    # pad / crop 到 capture_n
    if len(native_latent_pos_hids) < capture_n and len(native_latent_pos_hids) > 0:
        while len(native_latent_pos_hids) < capture_n:
            native_latent_pos_hids.append(native_latent_pos_hids[-1].clone())
            if native_latent_pos_hids_layers:
                native_latent_pos_hids_layers.append(native_latent_pos_hids_layers[-1].clone())
    native_latent_pos_hids = native_latent_pos_hids[:capture_n] if native_latent_pos_hids else None
    native_latent_pos_hids_layers = native_latent_pos_hids_layers[:capture_n] \
        if native_latent_pos_hids_layers else None

    result = {
        'H_native_latent_pos': torch.stack(native_latent_pos_hids, dim=0) if native_latent_pos_hids else None,
        'H_native_layers': torch.stack(native_latent_pos_hids_layers, dim=0) if native_latent_pos_hids_layers else None,
        'H_native_answer': H_native_answer,
        'H_native_answer_layers': H_native_answer_layers,
        'saw_latent_trigger': saw_latent_trigger,
    }
    return result


# ==================================================================
# 4. 主循环
# ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--data_file', default=DEFAULT_DATA_FILE)
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument('--layers', default=DEFAULT_LAYERS,
                        help='逗号分隔的 layer index (支持 -1 = last), 空串 = 不抽 layer-wise')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save_every', type=int, default=50, help='每 N 条保存一次')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    if args.layers.strip():
        layers = [int(x) for x in args.layers.split(',') if x.strip()]
    else:
        layers = None

    print("=" * 72)
    print("  extract_embeddings_with_native.py")
    print("=" * 72)
    print(f"  model_path  : {args.model_path}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  data_file   : {args.data_file}")
    print(f"  output      : {args.output}")
    print(f"  num_samples : {args.num_samples}")
    print(f"  layers      : {layers}")
    print("=" * 72)

    # 加载模型 (与 diagnose_latent_ab.py 一致)
    from rld.data import NLD_SYSTEM_PROMPT, LATENT_TOKEN, LATENT_END_TOKEN
    from rld.model_v2 import NLDModel
    from transformers import AutoProcessor
    from PIL import Image

    print(f"\n[1/3] 加载模型 ...")
    model = NLDModel(model_path=args.model_path, torch_dtype=DTYPE,
                     attn_implementation="flash_attention_2")
    processor = AutoProcessor.from_pretrained(args.model_path)
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [LATENT_TOKEN, LATENT_END_TOKEN]}
    )
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    model.set_processor(processor)
    model.load_pretrained(args.checkpoint)
    model = model.to(DEVICE)
    model.eval()

    # 禁用饱和度退出 (只用 exit token), 保证 latent 至少走到 target_steps
    original_threshold = model.latent_thinker.saturation_exit_threshold
    model.latent_thinker.saturation_exit_threshold = 2.0
    print(f"[1/3] ✅ saturation_exit_threshold: {original_threshold} -> 2.0")

    # 加载数据
    samples = []
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
                if len(samples) >= args.num_samples:
                    break
    print(f"[2/3] 加载 {len(samples)} 条样本")

    # 存储
    store = {
        'V': [], 'L': [],
        'V_layers': [], 'L_layers': [],
        'H_latent': [], 'H_latent_layers': [],
        'H_native_latent_pos': [], 'H_native_layers': [],
        'H_native_answer': [], 'H_native_answer_layers': [],
        'alpha': [],
        'meta': {
            'num_samples': 0,
            'dim': None,
            'layers': layers,
            'model_path': args.model_path,
            'checkpoint': args.checkpoint,
            'data_file': args.data_file,
            'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            'per_sample': [],
        },
    }

    def _save(final=False):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        out_path = args.output if final else args.output.replace('.pt', '.partial.pt')
        torch.save(store, out_path)
        if final:
            tmp = args.output.replace('.pt', '.partial.pt')
            if os.path.exists(tmp) and tmp != out_path:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        print(f"  💾 saved: {out_path}  ({store['meta']['num_samples']} samples)")

    print(f"[3/3] 开始抽取 ...")
    t_start = time.time()
    n_done = 0
    n_skipped = 0

    for idx, sample in enumerate(samples):
        qid = sample.get('question_id', f'Q_{idx}')
        question = sample['question']
        q_type = sample.get('question_type', 'Unknown')

        image_paths = []
        if 'image_paths' in sample:
            paths_list = sample['image_paths'] if isinstance(sample['image_paths'], list) else [sample['image_paths']]
            for img_rel in paths_list:
                img_path = img_rel if os.path.isabs(img_rel) else os.path.join(args.data_root, img_rel)
                image_paths.append(img_path)
        elif 'image' in sample:
            img_rel = sample['image']
            img_path = img_rel if os.path.isabs(img_rel) else os.path.join(args.data_root, img_rel)
            image_paths.append(img_path)

        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing or not image_paths:
            n_skipped += 1
            continue

        try:
            messages = [{'role': 'system',
                         'content': [{'type': 'text', 'text': NLD_SYSTEM_PROMPT}]}]
            user_content = [{'type': 'image', 'image': p} for p in image_paths]
            user_content.append({'type': 'text', 'text': question})
            messages.append({'role': 'user', 'content': user_content})

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if process_vision_info:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [Image.open(p).convert('RGB') for p in image_paths]
                video_inputs = None
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors='pt', padding=True,
            ).to(DEVICE)

            with torch.no_grad():
                shared = prepare_inputs_for_extraction(model, inputs)

                # V/L
                V_vec, L_vec = pool_V_and_L(shared)
                if layers is not None:
                    V_lay, L_lay = pool_V_and_L_layerwise(shared, layers)
                else:
                    V_lay = L_lay = None

                # Latent 通路
                lat = forward_latent_path(model, processor, shared,
                                          max_new_tokens=args.max_new_tokens,
                                          layers=layers)
                num_lat = lat['num_latent_steps']
                if num_lat == 0:
                    # 该样本没触发 latent, 视为 skip
                    n_skipped += 1
                    continue

                # Native 通路
                nat = forward_native_path(model, processor, shared,
                                          max_new_tokens=args.max_new_tokens,
                                          num_latent_steps_target=num_lat,
                                          layers=layers,
                                          latent_trigger_pos=lat['latent_trigger_pos'])

            store['V'].append(V_vec)
            store['L'].append(L_vec)
            if V_lay is not None:
                store['V_layers'].append(V_lay)
                store['L_layers'].append(L_lay)
            store['H_latent'].append(lat['H_latent'])
            if lat['H_latent_layers'] is not None:
                store['H_latent_layers'].append(lat['H_latent_layers'])
            store['H_native_latent_pos'].append(nat['H_native_latent_pos'])
            if nat['H_native_layers'] is not None:
                store['H_native_layers'].append(nat['H_native_layers'])
            store['H_native_answer'].append(nat['H_native_answer'])
            if nat['H_native_answer_layers'] is not None:
                store['H_native_answer_layers'].append(nat['H_native_answer_layers'])
            store['alpha'].append(lat['alpha'])
            store['meta']['per_sample'].append({
                'qid': qid, 'question_type': q_type,
                'num_latent_steps': int(num_lat),
                'saw_native_latent_trigger': bool(nat['saw_latent_trigger']),
                'img_path': image_paths[0] if image_paths else None,
            })
            if store['meta']['dim'] is None:
                store['meta']['dim'] = int(V_vec.shape[-1])
            n_done += 1
            store['meta']['num_samples'] = n_done

            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{len(samples)}] qid={qid} T={num_lat} "
                  f"native_trig={bool(nat['saw_latent_trigger'])} "
                  f"({n_done} done, {n_skipped} skip, {elapsed:.1f}s, "
                  f"{elapsed/max(n_done,1):.2f}s/sample)")

            if args.save_every > 0 and n_done % args.save_every == 0:
                _save(final=False)

        except Exception as e:
            print(f"  ❌ sample {idx} 出错: {e}")
            import traceback
            traceback.print_exc()
            n_skipped += 1
            continue

    _save(final=True)
    print(f"\n✅ Done.  {n_done} samples extracted, {n_skipped} skipped.")
    print(f"    output: {args.output}")


if __name__ == '__main__':
    main()
