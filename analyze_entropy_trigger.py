#!/usr/bin/env python3
"""
实验: Latent 触发位置与 Token 熵的关系分析

目标: 验证模型从数据中学到了"在自然语言推理困惑处触发 latent"的机制.

方法:
  1. 对 ERQA 测试集每条样本, 用 LatentDraft 模型做 greedy 推理,
     在自回归生成过程中记录每个 token 位置的:
     - next-token 熵 H(p(y|x_{<t}))  (从 logits 计算)
     - 该 token 是否为 <|latent|> 触发位置
  2. 统计分析:
     - latent 触发位置的平均熵 vs 非触发位置的平均熵
     - latent 触发位置的熵百分位排名
     - 局部熵峰值对齐率
  3. 可视化:
     - 熵分布对比 (触发 vs 非触发)
     - 典型样本的 token-level 熵轨迹 + latent 位置标注
     - 熵百分位分布
     - 局部峰值对齐率

输出:
  - outputs/entropy_trigger_analysis/
    ├── per_sample_entropy.json
    ├── summary_stats.json
    ├── fig_entropy_distribution.png
    ├── fig_entropy_trajectory.png
    ├── fig_entropy_percentile.png
    └── fig_local_peak_alignment.png

用法:
  python analyze_entropy_trigger.py \\
    --checkpoint ./outputs/rld_stage2_erqa_latent_cot_v2_ckpt49_balanced_v2/checkpoint-26 \\
    --max_samples 100

  # 全量
  python analyze_entropy_trigger.py \\
    --checkpoint ./outputs/rld_stage2_erqa_latent_cot_v2_ckpt49_balanced_v2/checkpoint-26
"""

import os, sys, json, argparse, re, time
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ====== 分布式辅助 (沿用 eval_erqa.py 方案 B: DDP-free, shard 落盘合并) ======
def _ddp_env():
    """读取 RANK / WORLD_SIZE / LOCAL_RANK"""
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    return rank, world_size, local_rank


def _is_main():
    return int(os.environ.get('RANK', '0')) == 0


def _maybe_init_dist():
    rank, world_size, local_rank = _ddp_env()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[Entropy] ⚠️ qwen_vl_utils 未安装")
    process_vision_info = None

from rld.data import LATENT_TOKEN, LATENT_END_TOKEN, NLD_SYSTEM_PROMPT
from rld.model_v2 import NLDModel, _extract_visual_output
from transformers import AutoProcessor
from PIL import Image


# ============================================================
# 核心: no-latent 推理 (跳过 NativeLatentThinker, 同样记录 entropy)
# ============================================================
@torch.no_grad()
def generate_without_latent_record(
    model,
    processor,
    messages,
    max_new_tokens=2048,
    device="cuda",
):
    """跳过 latent thinking, 遇到 <|latent|> 当普通 token, 记录每个位置的 next-token 熵.

    Returns:
        dict with keys (与 generate_with_entropy_record 对称):
          - generated_text, generated_text_clean, generated_token_ids,
            token_entropies, latent_trigger_positions, latent_end_positions,
            prompt_len, num_latent_triggers,
            total_flops, prefill_flops, decode_flops_per_token,
            prefill_latency_s, decode_latency_s, total_latency_s,
            num_decode_steps
    """
    # ---- 构造 inputs ----
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages) if process_vision_info else (None, None)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    prompt_len = input_ids.shape[1]
    B = input_ids.shape[0]
    device_ = input_ids.device

    # ---- 获取模型内部组件 ----
    inner_model = model._inner_model
    text_model = inner_model.language_model
    lm_head = model._lm_head
    latent_token_id = model.latent_token_id
    latent_end_token_id = processor.tokenizer.convert_tokens_to_ids(LATENT_END_TOKEN)
    eos_token_id = processor.tokenizer.eos_token_id

    # ---- 视觉编码 (只做一次) ----
    if pixel_values is not None and image_grid_thw is not None:
        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(_pixel_values, grid_thw=image_grid_thw)
        merged_hidden_states, gen_deepstack_features = _extract_visual_output(vision_output)
        split_sizes = (
            image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
        ).tolist()
        image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))
        cached_visual_embeds = torch.cat(image_embeds_list, dim=0)
    else:
        cached_visual_embeds = None
        gen_deepstack_features = None

    # ---- Prefill ----
    prompt_embeds = inner_model.get_input_embeddings()(input_ids)
    image_token_id = inner_model.config.image_token_id
    image_mask = (input_ids == image_token_id)
    num_image_tokens = image_mask.sum().item()
    if num_image_tokens > 0 and cached_visual_embeds is not None:
        img_emb = cached_visual_embeds[:num_image_tokens].to(prompt_embeds.device, prompt_embeds.dtype)
        image_mask_3d = image_mask.unsqueeze(-1).expand_as(prompt_embeds)
        prompt_embeds = prompt_embeds.masked_scatter(image_mask_3d, img_emb)

    seg_token_mask = torch.ones(B, input_ids.shape[1], device=device_, dtype=torch.long)
    inner_model.rope_deltas = None
    position_ids, rope_deltas = inner_model.get_rope_index(
        input_ids, image_grid_thw=image_grid_thw, video_grid_thw=None,
        attention_mask=seg_token_mask,
    )
    inner_model.rope_deltas = rope_deltas

    if position_ids is not None and position_ids.dim() == 3:
        text_pos = torch.arange(input_ids.shape[1], device=device_).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

    visual_pos_masks = (input_ids == inner_model.config.image_token_id) if pixel_values is not None and num_image_tokens > 0 else None

    prefill_start = time.perf_counter()
    prefill_out = text_model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=gen_deepstack_features,
    )
    prefill_end = time.perf_counter()
    prefill_latency_s = prefill_end - prefill_start
    past_key_values = prefill_out.past_key_values
    mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None
    prefill_hidden = prefill_out.last_hidden_state
    next_token_logits = lm_head(prefill_hidden[:, -1, :])

    # ---- 计时: prefill 阶段 ----
    # 注意: 这里 prefill 已经执行完了, 我们记录它耗时 (在推理前后包裹即可)
    # 但由于 prefill 在函数开头已执行, 我们在外层计时

    # ---- 记录变量 ----
    generated_ids = input_ids.clone()
    token_entropies = []
    top1_probs = []       # 每个位置的 next-token 最大概率
    topk_probs_list = []  # 每个位置的 top-k 概率分布 (稀疏存储)
    generated_token_ids = []
    latent_trigger_positions = []  # 即使 no-latent, 仍记录模型"想要"触发的位置
    latent_end_positions = []

    recent_last_hidden = prefill_hidden[:, -1:, :]
    current_pos = prompt_len

    # ---- Decode 计时 ----
    decode_start = time.perf_counter()
    num_decode_steps = 0

    # ---- 自回归循环 ----
    for gen_step in range(max_new_tokens):
        # 计算 next-token 熵
        probs = F.softmax(next_token_logits, dim=-1)
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        p = probs[0]
        log_p = log_probs[0]
        mask = p > 0
        entropy = -(p[mask] * log_p[mask]).sum().item()
        token_entropies.append(entropy)
        # 记录 top-1 概率 (最大概率值)
        top1_prob = p.max().item()
        top1_probs.append(top1_prob)
        # 记录 top-k 概率分布用于后续信息论分析 (存 top-50, 节省空间)
        topk_vals, _ = torch.topk(p, min(50, p.numel()))
        topk_probs_list.append(topk_vals.cpu().tolist())

        # Greedy 解码
        next_token = next_token_logits.argmax(dim=-1, keepdim=True).squeeze(-1)
        token_id = next_token[0].item()

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
        generated_token_ids.append(token_id)
        num_decode_steps += 1

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # ---- 检测 <|latent|> (no-latent 模式下, 跳过 thinking, 当普通 token 处理) ----
        is_latent_trigger = (latent_token_id is not None) and (token_id == latent_token_id)
        is_latent_end = (latent_end_token_id is not None) and (token_id == latent_end_token_id)

        if is_latent_trigger:
            latent_trigger_positions.append(len(generated_ids) - 1)
            # ---- NO LATENT: 直接跳过 thinking, 当普通 token 继续 ----
            # 不调用 model.latent_thinker(), 不注入 thought prefix
            # 当前 token 已计入 generated_ids, 直接继续下一个 forward
            pass

        elif is_latent_end:
            latent_end_positions.append(len(generated_ids) - 1)

        # ---- 继续自回归 forward ----
        new_mrope_pos = (mrope_last_pos + 1) if mrope_last_pos is not None else None
        text_pos_gen = torch.full((B, 1), current_pos, device=device_, dtype=torch.long)
        if new_mrope_pos is not None:
            position_ids_4d = torch.cat([text_pos_gen.unsqueeze(0), new_mrope_pos], dim=0)
        else:
            position_ids_4d = text_pos_gen.unsqueeze(0)

        token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))
        current_total_len = current_pos + 1
        new_attention_mask = torch.ones(B, current_total_len, device=device_, dtype=attention_mask.dtype)
        cache_position_gen = torch.tensor([current_pos], device=device_, dtype=torch.long)

        text_outputs = text_model(
            inputs_embeds=token_embeds,
            attention_mask=new_attention_mask,
            position_ids=position_ids_4d,
            past_key_values=past_key_values,
            cache_position=cache_position_gen,
            use_cache=True,
        )
        past_key_values = text_outputs.past_key_values
        new_hidden = text_outputs.last_hidden_state
        next_token_logits = lm_head(new_hidden[:, -1, :])
        recent_last_hidden = new_hidden
        if mrope_last_pos is not None:
            mrope_last_pos = new_mrope_pos

        current_pos += 1

    decode_end = time.perf_counter()

    # 解码
    generated_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    generated_text_clean = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    # ---- FLOPs 估算 (含 latent thinker 额外开销) ----
    total_flops, prefill_flops, decode_flops_per_token = _estimate_flops(
        model, processor, prompt_len, num_decode_steps,
        num_latent_events=len(latent_trigger_positions),
        num_total_thought_steps=sum(
            # 近似: 每个 latent event 平均 num_thought_steps ≈ 默认模型配置
            1  # 粗略估计, 实际步数在 thinker 运行时才知道
            for _ in latent_trigger_positions
        ),
    )

    # ---- Latency 统计 ----
    decode_latency = decode_end - decode_start
    total_latency = prefill_latency_s + decode_latency  # prefill_latency_s 由外层填充

    return {
        "generated_text": generated_text,
        "generated_text_clean": generated_text_clean,
        "generated_token_ids": generated_token_ids,
        "token_entropies": token_entropies,
        "top1_probs": top1_probs,
        "topk_probs_list": topk_probs_list,
        "latent_trigger_positions": latent_trigger_positions,
        "latent_end_positions": latent_end_positions,
        "prompt_len": prompt_len,
        "num_latent_triggers": len(latent_trigger_positions),
        # ---- 效率指标 ----
        "total_flops": total_flops,
        "prefill_flops": prefill_flops,
        "decode_flops_per_token": decode_flops_per_token,
        "prefill_latency_s": prefill_latency_s,  # 由外层填充
        "decode_latency_s": decode_latency,
        "total_latency_s": total_latency,
        "num_decode_steps": num_decode_steps,
    }


# ============================================================
# FLOPs 估算
# ============================================================
def _estimate_flops(model, processor, prompt_len, num_decode_steps,
                    num_latent_events=0, num_total_thought_steps=0):
    """粗略估算 with-latent / no-latent 两种推理模式的 FLOPs.

    Args:
        model: NLDModel 实例
        processor: AutoProcessor 实例
        prompt_len: prefill 序列长度
        num_decode_steps: decode 步数 (不含 latent thinker 内部步)
        num_latent_events: latent 触发次数 (with-latent 模式下 > 0)
        num_total_thought_steps: 总共的 latent thinker 内部步数

    Returns:
        (total_flops, prefill_flops, decode_flops_per_token)
    """
    try:
        inner = model._inner_model
        text_model = inner.language_model
        config = text_model.config
        n_layers = config.num_hidden_layers
        d_model = config.hidden_size
        seq_len = max(prompt_len, 4096)

        # Prefill: 每个 token 过一遍所有层
        prefill_flops = 2 * prompt_len * n_layers * d_model * (1 + 8 * d_model / seq_len)

        # Decode: 每个 token 同样过所有层 (attention 用 KV-cache, 但仍有 MLP 开销)
        decode_flops_per_token = 2 * n_layers * d_model * (1 + 8 * d_model / seq_len)

        # ---- Latent Thinker 额外 FLOPs ----
        # 每次 latent 触发时, thinker 会做 num_thought_steps 步自回归
        # thinker 与 text_model 共享架构但规模更小
        # 粗略估计: thinker 的 hidden_size 是 text_model 的一个比例
        latent_flops = 0
        if num_latent_events > 0 and num_total_thought_steps > 0:
            try:
                thinker = model.latent_thinker
                # thinker 内部循环: 每步过一次 text_model forward
                # 但 thinker 的 hidden_size 可能不同
                thinker_d_model = getattr(thinker, 'hidden_size', d_model) if hasattr(thinker, 'hidden_size') else d_model
                thinker_n_layers = getattr(thinker, 'num_hidden_layers', n_layers) if hasattr(thinker, 'num_hidden_layers') else n_layers
                # 每个 thought step: 一次 forward 过所有 thinker layers
                # 简化: 与 text_model 的 decode 类似
                thought_step_flops = 2 * thinker_n_layers * thinker_d_model * (1 + 8 * thinker_d_model / seq_len)
                latent_flops = num_total_thought_steps * thought_step_flops
            except Exception as _e:
                print(f"[Entropy] ⚠️ Latent thinker FLOPs 估算失败: {_e}, 使用粗略估算")
                # 粗略估算: 每个 thought step ≈ 0.5 * decode_flops_per_token (thinker 通常更小)
                latent_flops = num_total_thought_steps * decode_flops_per_token * 0.5

        total_flops = prefill_flops + num_decode_steps * decode_flops_per_token + latent_flops
        return total_flops, prefill_flops, decode_flops_per_token
    except Exception as e:
        print(f"[Entropy] ⚠️ FLOPs 估算失败: {e}, 返回 0")
        return 0, 0, 0


# ============================================================
# 核心: 带 entropy 记录的自回归推理 (with latent)
# ============================================================
@torch.no_grad()
def generate_with_entropy_record(
    model,
    processor,
    messages,
    max_new_tokens=2048,
    device="cuda",
):
    """对一条样本做 greedy 推理, 记录每个 token 位置的 next-token 熵.

    沿用 diagnose_latent_ab.py / model_v2.py::_run_one_pass 的自回归循环模式,
    额外在每个 step 计算 softmax logits 的熵.

    Returns:
        dict with keys:
          - generated_text: str, 最终生成的文本 (含特殊 token)
          - generated_text_clean: str, 去除特殊 token 的文本
          - generated_token_ids: list[int], 生成的 token id 序列
          - token_entropies: list[float], 每个 token 位置的熵 (thinker 步用 NaN)
          - latent_trigger_positions: list[int], latent 触发在 generated_token_ids 中的下标
          - latent_end_positions: list[int], <|/latent|> 在 generated_token_ids 中的下标
          - prompt_len: int
          - num_latent_triggers: int
    """
    # ---- 构造 inputs (沿用 eval_erqa.py 的方式) ----
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages) if process_vision_info else (None, None)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    prompt_len = input_ids.shape[1]
    B = input_ids.shape[0]
    device_ = input_ids.device

    # ---- 获取模型内部组件 ----
    inner_model = model._inner_model
    text_model = inner_model.language_model
    lm_head = model._lm_head
    latent_token_id = model.latent_token_id
    latent_end_token_id = processor.tokenizer.convert_tokens_to_ids(LATENT_END_TOKEN)
    eos_token_id = processor.tokenizer.eos_token_id

    # ---- 视觉编码 (只做一次) ----
    if pixel_values is not None and image_grid_thw is not None:
        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(_pixel_values, grid_thw=image_grid_thw)
        merged_hidden_states, gen_deepstack_features = _extract_visual_output(vision_output)
        split_sizes = (
            image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
        ).tolist()
        image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))
        cached_visual_embeds = torch.cat(image_embeds_list, dim=0)
    else:
        cached_visual_embeds = None
        gen_deepstack_features = None

    # ---- Prefill ----
    prompt_embeds = inner_model.get_input_embeddings()(input_ids)
    image_token_id = inner_model.config.image_token_id
    image_mask = (input_ids == image_token_id)
    num_image_tokens = image_mask.sum().item()
    if num_image_tokens > 0 and cached_visual_embeds is not None:
        img_emb = cached_visual_embeds[:num_image_tokens].to(prompt_embeds.device, prompt_embeds.dtype)
        image_mask_3d = image_mask.unsqueeze(-1).expand_as(prompt_embeds)
        prompt_embeds = prompt_embeds.masked_scatter(image_mask_3d, img_emb)

    seg_token_mask = torch.ones(B, input_ids.shape[1], device=device_, dtype=torch.long)
    inner_model.rope_deltas = None
    position_ids, rope_deltas = inner_model.get_rope_index(
        input_ids, image_grid_thw=image_grid_thw, video_grid_thw=None,
        attention_mask=seg_token_mask,
    )
    inner_model.rope_deltas = rope_deltas

    if position_ids is not None and position_ids.dim() == 3:
        text_pos = torch.arange(input_ids.shape[1], device=device_).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

    visual_pos_masks = (input_ids == inner_model.config.image_token_id) if pixel_values is not None and num_image_tokens > 0 else None

    prefill_start = time.perf_counter()
    prefill_out = text_model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=gen_deepstack_features,
    )
    prefill_end = time.perf_counter()
    prefill_latency_s = prefill_end - prefill_start
    past_key_values = prefill_out.past_key_values
    mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None
    prefill_hidden = prefill_out.last_hidden_state
    next_token_logits = lm_head(prefill_hidden[:, -1, :])

    # ---- 记录变量 ----
    generated_ids = input_ids.clone()
    token_entropies = []
    top1_probs = []       # 每个位置的 next-token 最大概率
    topk_probs_list = []  # 每个位置的 top-k 概率分布 (稀疏存储)
    generated_token_ids = []
    latent_trigger_positions = []
    latent_end_positions = []
    recent_last_hidden = prefill_hidden[:, -1:, :]
    current_pos = prompt_len
    num_decode_steps = 0

    decode_start = time.perf_counter()
    # ---- 自回归循环 ----
    for gen_step in range(max_new_tokens):
        # 计算 next-token 熵: H = -sum p(y) log p(y)
        probs = F.softmax(next_token_logits, dim=-1)
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        p = probs[0]      # [V]
        log_p = log_probs[0]
        # 只计算 p > 0 的项, 避免 NaN
        mask = p > 0
        entropy = -(p[mask] * log_p[mask]).sum().item()
        token_entropies.append(entropy)
        # 记录 top-1 概率 (最大概率值)
        top1_prob = p.max().item()
        top1_probs.append(top1_prob)
        # 记录 top-k 概率分布用于后续信息论分析 (存 top-50, 节省空间)
        topk_vals, _ = torch.topk(p, min(50, p.numel()))
        topk_probs_list.append(topk_vals.cpu().tolist())
        # 记录完整的概率分布百分比位数

        # Greedy 解码
        next_token = next_token_logits.argmax(dim=-1, keepdim=True).squeeze(-1)
        token_id = next_token[0].item()

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
        generated_token_ids.append(token_id)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # ---- 检测 <|latent|> 触发 ----
        is_latent_trigger = (latent_token_id is not None) and (token_id == latent_token_id)
        is_latent_end = (latent_end_token_id is not None) and (token_id == latent_end_token_id)

        if is_latent_trigger:
            latent_trigger_positions.append(len(generated_token_ids) - 1)

            # 触发 NativeLatentThinker
            last_hidden = recent_last_hidden
            base_thought_pos = current_pos
            thought_pos = torch.arange(current_pos, current_pos + 1, device=device_)
            thought_mrope_pos = thought_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
            thought_text_pos = torch.full((B, 1), base_thought_pos, device=device_, dtype=torch.long)
            thought_position_ids = torch.cat([thought_text_pos.unsqueeze(0), thought_mrope_pos], dim=0)
            think_attn_mask = torch.ones(B, base_thought_pos + 1, device=device_, dtype=torch.long)

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
            thought_output = thinker_result['thought_output']
            num_thought_steps = thinker_result.get('num_thought_steps', 1)

            # thinker 期间没有 next-token 熵 (隐空间无 token),
            # 用 NaN 占位 (num_thought_steps - 1 个), 方便对齐
            for _ in range(num_thought_steps - 1):
                token_entropies.append(float('nan'))
                top1_probs.append(float('nan'))
                topk_probs_list.append(None)

            # 写回 thought prefix 到 KV cache
            thought_prefix_cache_pos = torch.arange(
                current_pos, current_pos + num_thought_steps, device=device_
            )
            thought_prefix_mrope = thought_prefix_cache_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
            thought_prefix_text_pos = torch.arange(
                current_pos, current_pos + num_thought_steps, device=device_
            ).unsqueeze(0).expand(B, -1)
            thought_prefix_pos_ids = torch.cat([
                thought_prefix_text_pos.unsqueeze(0),
                thought_prefix_mrope,
            ], dim=0)
            thought_prefix_attn_mask = torch.ones(
                B, current_pos + num_thought_steps, device=device_, dtype=attention_mask.dtype
            )
            prefix_outputs = text_model(
                inputs_embeds=thought_output,
                attention_mask=thought_prefix_attn_mask,
                position_ids=thought_prefix_pos_ids,
                past_key_values=past_key_values,
                cache_position=thought_prefix_cache_pos,
                use_cache=True,
            )
            past_key_values = prefix_outputs.past_key_values
            current_pos += num_thought_steps
            mrope_last_pos = mrope_last_pos + num_thought_steps

        elif is_latent_end:
            latent_end_positions.append(len(generated_token_ids) - 1)

        # ---- 继续自回归 forward ----
        new_mrope_pos = (mrope_last_pos + 1) if mrope_last_pos is not None else None
        text_pos_gen = torch.full((B, 1), current_pos, device=device_, dtype=torch.long)
        if new_mrope_pos is not None:
            position_ids_4d = torch.cat([text_pos_gen.unsqueeze(0), new_mrope_pos], dim=0)
        else:
            position_ids_4d = text_pos_gen.unsqueeze(0)

        token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))
        current_total_len = current_pos + 1
        new_attention_mask = torch.ones(B, current_total_len, device=device_, dtype=attention_mask.dtype)
        cache_position_gen = torch.tensor([current_pos], device=device_, dtype=torch.long)

        text_outputs = text_model(
            inputs_embeds=token_embeds,
            attention_mask=new_attention_mask,
            position_ids=position_ids_4d,
            past_key_values=past_key_values,
            cache_position=cache_position_gen,
            use_cache=True,
        )
        past_key_values = text_outputs.past_key_values
        new_hidden = text_outputs.last_hidden_state
        next_token_logits = lm_head(new_hidden[:, -1, :])
        recent_last_hidden = new_hidden
        if mrope_last_pos is not None:
            mrope_last_pos = new_mrope_pos
        current_pos += 1
        num_decode_steps += 1

    decode_end = time.perf_counter()

    # 解码
    generated_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    generated_text_clean = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    # ---- FLOPs 估算 (含 latent thinker 额外开销) ----
    total_flops, prefill_flops, decode_flops_per_token = _estimate_flops(
        model, processor, prompt_len, num_decode_steps,
        num_latent_events=len(latent_trigger_positions),
        num_total_thought_steps=sum(
            # 近似: 每个 latent event 平均 num_thought_steps ≈ 默认模型配置
            1  # 粗略估计, 实际步数在 thinker 运行时才知道
            for _ in latent_trigger_positions
        ),
    )

    # ---- Latency 统计 ----
    decode_latency = decode_end - decode_start
    total_latency = prefill_latency_s + decode_latency

    return {
        "generated_text": generated_text,
        "generated_text_clean": generated_text_clean,
        "generated_token_ids": generated_token_ids,
        "token_entropies": token_entropies,
        "top1_probs": top1_probs,
        "topk_probs_list": topk_probs_list,
        "latent_trigger_positions": latent_trigger_positions,
        "latent_end_positions": latent_end_positions,
        "prompt_len": prompt_len,
        "num_latent_triggers": len(latent_trigger_positions),
        "num_decode_steps": num_decode_steps,
        # ---- 效率指标 ----
        "total_flops": total_flops,
        "prefill_flops": prefill_flops,
        "decode_flops_per_token": decode_flops_per_token,
        "prefill_latency_s": prefill_latency_s,
        "decode_latency_s": decode_latency,
        "total_latency_s": total_latency,
    }


# ============================================================
# 数据加载
# ============================================================
def load_erqa_data(data_file, data_root, max_samples=0):
    samples = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append(item)
    if max_samples > 0:
        samples = samples[:max_samples]
    print(f"[Entropy] 加载 {len(samples)} 条 ERQA 样本")
    return samples


# ============================================================
# 可视化
# ============================================================
def plot_results(output_dir, per_sample_data, summary_stats):
    """生成所有可视化图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    trigger_entropies = summary_stats["trigger_entropies"]
    non_trigger_entropies = summary_stats["non_trigger_entropies"]

    # ---- 图1: 熵分布对比 (直方图 + CDF + Box) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a: 直方图
    ax = axes[0]
    if non_trigger_entropies:
        plot_non = non_trigger_entropies if len(non_trigger_entropies) <= 50000 else \
            np.random.choice(non_trigger_entropies, 50000, replace=False).tolist()
        ax.hist(plot_non, bins=80, alpha=0.6, label='Non-trigger', color='#4ECDC4', density=True)
    if trigger_entropies:
        ax.hist(trigger_entropies, bins=40, alpha=0.7, label='Latent trigger', color='#FF6B6B', density=True)
    ax.set_xlabel('Token Entropy (nats)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Entropy Distribution', fontsize=13)
    ax.legend(fontsize=11)

    # 1b: CDF
    ax = axes[1]
    if non_trigger_entropies:
        sorted_non = np.sort(non_trigger_entropies)
        cdf_non = np.arange(1, len(sorted_non) + 1) / len(sorted_non)
        ax.plot(sorted_non, cdf_non, label='Non-trigger', color='#4ECDC4', linewidth=2)
    if trigger_entropies:
        sorted_trig = np.sort(trigger_entropies)
        cdf_trig = np.arange(1, len(sorted_trig) + 1) / len(sorted_trig)
        ax.plot(sorted_trig, cdf_trig, label='Latent trigger', color='#FF6B6B', linewidth=2)
    ax.set_xlabel('Token Entropy (nats)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Entropy CDF', fontsize=13)
    ax.legend(fontsize=11)

    # 1c: Box plot
    ax = axes[2]
    data_for_box = []
    labels_for_box = []
    if trigger_entropies:
        data_for_box.append(trigger_entropies)
        labels_for_box.append('Latent\ntrigger')
    if non_trigger_entropies:
        sample_non = non_trigger_entropies if len(non_trigger_entropies) <= 10000 else \
            np.random.choice(non_trigger_entropies, 10000, replace=False).tolist()
        data_for_box.append(sample_non)
        labels_for_box.append('Non-\ntrigger')
    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                        widths=0.5, showfliers=False)
        colors = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(bp['boxes'], colors[:len(data_for_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel('Token Entropy (nats)', fontsize=12)
    ax.set_title('Entropy Comparison', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_entropy_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Entropy] 已保存: fig_entropy_distribution.png")

    # ---- 图2: 典型样本的熵轨迹 ----
    samples_with_latent = [(i, s) for i, s in enumerate(per_sample_data) if s.get('with_latent', s)['latent_trigger_positions']]
    samples_with_latent.sort(key=lambda x: len(x[1].get('with_latent', x[1])['token_entropies']), reverse=True)
    n_plot = min(4, len(samples_with_latent))
    if n_plot > 0:
        indices = np.linspace(0, len(samples_with_latent) - 1, n_plot, dtype=int)
        plot_samples = [samples_with_latent[idx] for idx in indices]

        fig, axes = plt.subplots(n_plot, 1, figsize=(16, 4 * n_plot))
        if n_plot == 1:
            axes = [axes]

        for ax, (sample_idx, sample) in zip(axes, plot_samples):
            wl = sample.get('with_latent', sample)
            entropies = np.array(wl['token_entropies'], dtype=float)
            # NaN 插值
            valid_mask = ~np.isnan(entropies)
            if valid_mask.sum() > 1:
                entropies_interp = np.copy(entropies)
                entropies_interp[~valid_mask] = np.interp(
                    np.where(~valid_mask)[0],
                    np.where(valid_mask)[0],
                    entropies[valid_mask]
                )
            else:
                entropies_interp = entropies

            ax.plot(entropies_interp, color='#4ECDC4', linewidth=0.8, alpha=0.8)

            # 标注 latent 触发位置
            for tp in wl['latent_trigger_positions']:
                if tp < len(entropies_interp):
                    ax.axvline(x=tp, color='#FF6B6B', linewidth=1.5, alpha=0.7, linestyle='--')
                    ax.plot(tp, entropies_interp[tp], 'v', color='#FF6B6B', markersize=8)

            # 标注 latent end 位置
            for ep in wl['latent_end_positions']:
                if ep < len(entropies_interp):
                    ax.axvline(x=ep, color='#FFD93D', linewidth=1.0, alpha=0.5, linestyle=':')

            ax.set_xlabel('Token Position', fontsize=10)
            ax.set_ylabel('Entropy (nats)', fontsize=10)
            q_type = sample.get('question_type', '')
            n_trig = len(wl['latent_trigger_positions'])
            ax.set_title(f'Sample {sample_idx} | type={q_type} | {n_trig} latent trigger(s)', fontsize=11)
            ax.legend(['Entropy', 'Latent trigger', 'Latent end'], fontsize=8, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_entropy_trajectory.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_entropy_trajectory.png")

    # ---- 图3: 熵百分位排名分布 ----
    percentile_ranks = summary_stats.get("trigger_percentile_ranks", [])
    if percentile_ranks:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(percentile_ranks, bins=40, color='#FF6B6B', alpha=0.7, edgecolor='white')
        ax.axvline(x=50, color='gray', linestyle='--', linewidth=1, label='Median (50th pct)')
        ax.axvline(x=np.mean(percentile_ranks), color='#4ECDC4', linewidth=2,
                   label=f'Mean = {np.mean(percentile_ranks):.1f}th pct')
        ax.set_xlabel('Entropy Percentile Rank of Trigger Position', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Where Do Latent Triggers Land in Entropy Ranking?', fontsize=13)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_entropy_percentile.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_entropy_percentile.png")

    # ---- 图4: 局部熵峰值对齐率 ----
    peak_alignment_rates = summary_stats.get("local_peak_alignment", {})
    if peak_alignment_rates:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        windows = sorted(peak_alignment_rates.keys())
        rates = [peak_alignment_rates[w] for w in windows]
        ax.bar([str(w) for w in windows], rates, color='#FF6B6B', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Local Window Size (±W tokens)', fontsize=12)
        ax.set_ylabel('Peak Alignment Rate', fontsize=12)
        ax.set_title('Latent Trigger at Local Entropy Peak?', fontsize=13)
        ax.set_ylim(0, 1)
        for i, (w, r) in enumerate(zip(windows, rates)):
            ax.text(i, r + 0.02, f'{r:.1%}', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_local_peak_alignment.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_local_peak_alignment.png")

    # ---- 图5: With-Latent vs No-Latent FLOPs 对比 ----
    with_latent_flops_list = summary_stats.get("with_latent_flops_list", [])
    no_latent_flops_list = summary_stats.get("no_latent_flops_list", [])
    flops_ratios = summary_stats.get("flops_ratios", [])
    if with_latent_flops_list or no_latent_flops_list:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 5a: FLOPs 散点图 (每样本 with vs no)
        ax = axes[0]
        n_compare = min(len(with_latent_flops_list), len(no_latent_flops_list))
        if n_compare > 0:
            wf = with_latent_flops_list[:n_compare]
            nf = no_latent_flops_list[:n_compare]
            ax.scatter(nf, wf, alpha=0.5, s=20, color='#4ECDC4', label='Per-sample')
            # 对角线
            max_val = max(max(wf), max(nf)) * 1.1 if wf and nf else 1
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal FLOPs')
            ax.set_xlabel('No-Latent FLOPs', fontsize=12)
            ax.set_ylabel('With-Latent FLOPs', fontsize=12)
            ax.set_title('FLOPs: With-Latent vs No-Latent', fontsize=13)
            ax.legend(fontsize=10)
            ax.set_xscale('log')
            ax.set_yscale('log')

        # 5b: FLOPs 比值分布
        ax = axes[1]
        if flops_ratios:
            ax.hist(flops_ratios, bins=40, color='#FF6B6B', alpha=0.7, edgecolor='white')
            ax.axvline(x=np.mean(flops_ratios), color='#4ECDC4', linewidth=2,
                       label=f'Mean = {np.mean(flops_ratios):.2f}x')
            ax.axvline(x=np.median(flops_ratios), color='#FFD93D', linewidth=2, linestyle='--',
                       label=f'Median = {np.median(flops_ratios):.2f}x')
            ax.set_xlabel('FLOPs Ratio (with/no)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('FLOPs Ratio Distribution', fontsize=13)
            ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_flops_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_flops_comparison.png")

    # ---- 图6: With-Latent vs No-Latent Latency 对比 ----
    with_latent_latency_list = summary_stats.get("with_latent_latency_list", [])
    no_latent_latency_list = summary_stats.get("no_latent_latency_list", [])
    latency_ratios = summary_stats.get("latency_ratios", [])
    if with_latent_latency_list or no_latent_latency_list:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 6a: Latency 散点图 (每样本 with vs no)
        ax = axes[0]
        n_compare = min(len(with_latent_latency_list), len(no_latent_latency_list))
        if n_compare > 0:
            wl_lat = with_latent_latency_list[:n_compare]
            nl_lat = no_latent_latency_list[:n_compare]
            ax.scatter(nl_lat, wl_lat, alpha=0.5, s=20, color='#4ECDC4', label='Per-sample')
            max_val = max(max(wl_lat), max(nl_lat)) * 1.1 if wl_lat and nl_lat else 1
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal latency')
            ax.set_xlabel('No-Latent Latency (s)', fontsize=12)
            ax.set_ylabel('With-Latent Latency (s)', fontsize=12)
            ax.set_title('Latency: With-Latent vs No-Latent', fontsize=13)
            ax.legend(fontsize=10)

        # 6b: Latency 比值分布
        ax = axes[1]
        if latency_ratios:
            ax.hist(latency_ratios, bins=40, color='#FF6B6B', alpha=0.7, edgecolor='white')
            ax.axvline(x=np.mean(latency_ratios), color='#4ECDC4', linewidth=2,
                       label=f'Mean = {np.mean(latency_ratios):.2f}x')
            ax.axvline(x=np.median(latency_ratios), color='#FFD93D', linewidth=2, linestyle='--',
                       label=f'Median = {np.median(latency_ratios):.2f}x')
            ax.set_xlabel('Latency Ratio (with/no)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Latency Ratio Distribution', fontsize=13)
            ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_latency_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_latency_comparison.png")

    # ---- 图7: Decode 步数对比 ----
    with_latent_decode_steps_list = summary_stats.get("with_latent_decode_steps_list", [])
    no_latent_decode_steps_list = summary_stats.get("no_latent_decode_steps_list", [])
    if with_latent_decode_steps_list or no_latent_decode_steps_list:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        data_for_box = []
        labels_for_box = []
        if with_latent_decode_steps_list:
            data_for_box.append(with_latent_decode_steps_list)
            labels_for_box.append('With-Latent')
        if no_latent_decode_steps_list:
            data_for_box.append(no_latent_decode_steps_list)
            labels_for_box.append('No-Latent')
        if data_for_box:
            bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                            widths=0.5, showfliers=False)
            colors = ['#FF6B6B', '#4ECDC4']
            for patch, color in zip(bp['boxes'], colors[:len(data_for_box)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_ylabel('Number of Decode Steps', fontsize=12)
        ax.set_title('Decode Steps: With-Latent vs No-Latent', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_decode_steps_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_decode_steps_comparison.png")

    # ---- 图8: Trigger 前后上下文熵对比 ----
    trigger_pre = summary_stats.get("trigger_pre_context_entropies", [])
    trigger_post = summary_stats.get("trigger_post_context_entropies", [])
    if trigger_pre or trigger_post:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        data_box = []
        labels_box = []
        colors_box = []
        if trigger_pre:
            data_box.append(trigger_pre)
            labels_box.append('Trigger\\n(pre-context)')
            colors_box.append('#4ECDC4')
        if trigger_post:
            data_box.append(trigger_post)
            labels_box.append('Trigger\\n(post-context)')
            colors_box.append('#FFD93D')
        bp = ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True,
                        widths=0.5, showfliers=False)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.set_title('Context Entropy Around Latent Triggers (With-Latent)', fontsize=13)
        if trigger_pre and trigger_post:
            ax.axhline(y=np.median(trigger_pre), color='#4ECDC4', linestyle='--', alpha=0.3)
            ax.axhline(y=np.median(trigger_post), color='#FFD93D', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_trigger_context_entropy.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_trigger_context_entropy.png")

    # ---- 图9: No-Latent 模式下 "想触发" 位置的熵分析 ----
    # 意义: no-latent 中 trigger 位置没有 thinking 帮助,
    #       如果此处熵很高, 说明 latent 确实触发在推理难点
    nl_trig = summary_stats.get("no_latent_trigger_entropies", [])
    nl_pre = summary_stats.get("no_latent_trigger_pre_context_entropies", [])
    nl_post = summary_stats.get("no_latent_post_trigger_entropies", [])
    if nl_trig or nl_pre or nl_post:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        data_box = []
        labels_box = []
        colors_box = []
        if nl_pre:
            data_box.append(nl_pre)
            labels_box.append('Pre\\n(context)')
            colors_box.append('#4ECDC4')
        if nl_trig:
            data_box.append(nl_trig)
            labels_box.append('At Trigger\\n(token)')
            colors_box.append('#FF6B6B')
        if nl_post:
            data_box.append(nl_post)
            labels_box.append('Post\\n(context)')
            colors_box.append('#FFD93D')
        bp = ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True,
                        widths=0.5, showfliers=False)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.set_title('No-Latent Mode: Entropy at "Would-Be Trigger" Positions\n(High entropy here = trigger lands at reasoning bottleneck)', fontsize=13)
        # 添加参考线: 触发位置熵 vs 前后上下文中位数
        if nl_trig:
            ax.axhline(y=np.median([x for x in nl_trig if not np.isnan(x)]), color='red', linestyle=':', alpha=0.5, label='Median trigger entropy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_no_latent_trigger_entropy.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_no_latent_trigger_entropy.png")

    # ---- 图10: Top1-概率对比 (触发 vs 非触发) ----
    t_top1 = summary_stats.get("trigger_top1_probs", [])
    nt_top1 = summary_stats.get("non_trigger_top1_probs", [])
    if t_top1 and nt_top1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # 左: 箱线图对比
        bp = axes[0].boxplot([t_top1, nt_top1[:len(t_top1)*5]], tick_labels=['Trigger','Non-Trigger'],
                              patch_artist=True, widths=0.5, showfliers=False)
        for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0].set_ylabel('Top-1 Probability', fontsize=12)
        axes[0].set_title('Top-1 Probability: Trigger vs Non-Trigger', fontsize=13)
        # 右: 困难度(1-top1)分布直方图
        t_unc = [1 - x for x in t_top1]
        nt_unc = [1 - x for x in nt_top1[:len(t_top1)*5]]
        axes[1].hist(t_unc, bins=30, alpha=0.6, label='Trigger', color='#FF6B6B', density=True)
        axes[1].hist(nt_unc, bins=30, alpha=0.6, label='Non-Trigger', color='#4ECDC4', density=True)
        axes[1].axvline(x=np.median(t_unc), color='red', linestyle='--', label=f'Trigger median={np.median(t_unc):.3f}')
        axes[1].axvline(x=np.median(nt_unc), color='teal', linestyle='--', label=f'Non-Trigger median={np.median(nt_unc):.3f}')
        axes[1].set_xlabel('Uncertainty (1 - Top1 Prob)', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title('Uncertainty Distribution', fontsize=13)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_top1_prob_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_top1_prob_comparison.png")

    # ---- 图11: 触发前后 Top1-概率变化轨迹 ----
    t_pre = summary_stats.get("trigger_top1_pre", [])
    t_trig = summary_stats.get("trigger_top1_probs", [])
    t_post = summary_stats.get("trigger_top1_post", [])
    if t_pre and t_trig and t_post:
        fig, ax = plt.subplots(figsize=(10, 6))
        n = min(len(t_pre), len(t_trig), len(t_post))
        x = np.arange(n)
        ax.plot(x, t_pre[:n], 'o-', color='#4ECDC4', alpha=0.4, label='Pre-trigger (t-1)')
        ax.plot(x, t_trig[:n], 's-', color='#FF6B6B', alpha=0.6, label='At Trigger (t)')
        ax.plot(x, t_post[:n], '^-', color='#FFD93D', alpha=0.4, label='Post-trigger (t+1)')
        ax.axhline(y=np.mean(t_pre[:n]), color='#4ECDC4', linestyle='--', alpha=0.3)
        ax.axhline(y=np.mean(t_trig[:n]), color='#FF6B6B', linestyle='--', alpha=0.3)
        ax.axhline(y=np.mean(t_post[:n]), color='#FFD93D', linestyle='--', alpha=0.3)
        ax.set_xlabel('Trigger Event Index', fontsize=12)
        ax.set_ylabel('Top-1 Probability', fontsize=12)
        ax.set_title('Top-1 Probability Trajectory Around Triggers\n(Drop at trigger = bottleneck; recovery after = latent helps)', fontsize=13)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_top1_trajectory.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_top1_trajectory.png")

    # ---- 图12: 概率分布集中度 (Gini) 对比 ----
    t_gini = summary_stats.get("trigger_gini", [])
    nt_gini = summary_stats.get("non_trigger_gini", [])
    if t_gini and nt_gini:
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([t_gini, nt_gini[:len(t_gini)*5]], tick_labels=['Trigger','Non-Trigger'],
                         patch_artist=True, widths=0.5, showfliers=False)
        for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Gini Coefficient (concentration)', fontsize=12)
        ax.set_title('Probability Distribution Concentration\n(Higher Gini = more concentrated = more confident)', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_gini_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Entropy] 已保存: fig_gini_comparison.png")
def compute_statistics(per_sample_data):
    """计算熵分布统计 + with/no-latent FLOPs & latency 对比."""
    trigger_entropies = []
    non_trigger_entropies = []
    trigger_percentile_ranks = []
    local_peak_counts = {3: 0, 5: 0, 10: 0, 20: 0}
    local_peak_totals = {3: 0, 5: 0, 10: 0, 20: 0}

    # ---- 对比指标 ----
    flops_ratios = []
    latency_ratios = []
    flops_overheads = []
    latency_overheads = []
    with_latent_flops_list = []
    no_latent_flops_list = []
    with_latent_latency_list = []
    no_latent_latency_list = []
    with_latent_decode_steps_list = []
    no_latent_decode_steps_list = []

    for sample in per_sample_data:
        # ---- 熵分析 (取 with-latent 模式的数据) ----
        wl = sample.get("with_latent", sample)
        entropies = np.array(wl['token_entropies'], dtype=float)
        valid_mask = ~np.isnan(entropies)
        trigger_pos = wl['latent_trigger_positions']
        n_tokens = len(entropies)

        if n_tokens == 0:
            continue
        valid_entropies = entropies[valid_mask]
        if len(valid_entropies) == 0:
            continue

        for t_pos in trigger_pos:
            if t_pos < n_tokens and not np.isnan(entropies[t_pos]):
                trigger_entropies.append(float(entropies[t_pos]))
                rank = (valid_entropies <= entropies[t_pos]).sum() / len(valid_entropies) * 100
                trigger_percentile_ranks.append(float(rank))

                for W in local_peak_counts:
                    left = max(0, t_pos - W)
                    right = min(n_tokens, t_pos + W + 1)
                    local_window = entropies[left:right]
                    valid_local = local_window[~np.isnan(local_window)]
                    if len(valid_local) > 0:
                        local_peak_totals[W] += 1
                        if entropies[t_pos] >= valid_local.max() - 1e-6:
                            local_peak_counts[W] += 1

        trigger_set = set(trigger_pos)
        for i in range(n_tokens):
            if i not in trigger_set and not np.isnan(entropies[i]):
                non_trigger_entropies.append(float(entropies[i]))

        # ---- FLOPs & latency 对比 ----
        nl = sample.get("no_latent", {})
        if wl and nl:
            wf = wl.get("total_flops", 0) or 0
            nf = nl.get("total_flops", 0) or 0
            wl_lat = wl.get("total_latency_s", 0) or 0
            nl_lat = nl.get("total_latency_s", 0) or 0

            if nf > 0:
                flops_ratios.append(wf / nf)
            if nl_lat > 0:
                latency_ratios.append(wl_lat / nl_lat)
            flops_overheads.append(wf - nf)
            latency_overheads.append(wl_lat - nl_lat)

            if wf > 0:
                with_latent_flops_list.append(wf)
            if nf > 0:
                no_latent_flops_list.append(nf)
            if wl_lat > 0:
                with_latent_latency_list.append(wl_lat)
            if nl_lat > 0:
                no_latent_latency_list.append(nl_lat)

            wd = wl.get("num_decode_steps", 0) or 0
            nd = nl.get("num_decode_steps", 0) or 0
            if wd > 0:
                with_latent_decode_steps_list.append(wd)
            if nd > 0:
                no_latent_decode_steps_list.append(nd)

    local_peak_alignment = {}
    for W in local_peak_counts:
        if local_peak_totals[W] > 0:
            local_peak_alignment[W] = local_peak_counts[W] / local_peak_totals[W]

    # ---- No-Latent 模式下 (不 thinking) 的 trigger 位置熵分析 ----
    # 意义: 如果 no-latent 中 trigger 位置熵很高,
    #       说明那里确实是推理难点, latent 触发在正确的地方
    no_latent_trigger_entropies = []
    no_latent_post_trigger_entropies = []   # trigger 后第一个正常 token 的熵
    no_latent_trigger_pre_context_entropies = []  # trigger 前1个正常 token 的熵
    trigger_pre_context_entropies = []  # with-latent 中 trigger 前1个正常 token 的熵
    trigger_post_context_entropies = []  # with-latent 中 trigger 后第一个正常 token 的熵 (跳过 NaN)

    for sample in per_sample_data:
        wl = sample.get("with_latent", {})
        nl = sample.get("no_latent", {})

        # ---- with-latent: 触发前/后上下文熵 ----
        if wl:
            wl_ent = np.array(wl.get('token_entropies', []), dtype=float)
            wl_trig = wl.get('latent_trigger_positions', [])
            for t_pos in wl_trig:
                # 触发前1个正常 token 的熵
                for pre_idx in range(t_pos - 1, -1, -1):
                    if pre_idx >= 0 and not np.isnan(wl_ent[pre_idx]):
                        trigger_pre_context_entropies.append(float(wl_ent[pre_idx]))
                        break
                # 触发后第一个正常 token 的熵 (跳过 NaN)
                for post_idx in range(t_pos + 1, len(wl_ent)):
                    if post_idx < len(wl_ent) and not np.isnan(wl_ent[post_idx]):
                        trigger_post_context_entropies.append(float(wl_ent[post_idx]))
                        break

        # ---- no-latent: 同位置 trigger 熵 (模型想触发但没 thinking) ----
        if nl:
            nl_ent = np.array(nl.get('token_entropies', []), dtype=float)
            nl_trig = nl.get('latent_trigger_positions', [])
            for t_pos in nl_trig:
                if t_pos < len(nl_ent) and not np.isnan(nl_ent[t_pos]):
                    no_latent_trigger_entropies.append(float(nl_ent[t_pos]))
                    # trigger 后第一个正常 token 的熵
                    for post_idx in range(t_pos + 1, len(nl_ent)):
                        if post_idx < len(nl_ent) and not np.isnan(nl_ent[post_idx]):
                            no_latent_post_trigger_entropies.append(float(nl_ent[post_idx]))
                            break
                    # trigger 前1个正常 token 的熵
                    for pre_idx in range(t_pos - 1, -1, -1):
                        if pre_idx >= 0 and not np.isnan(nl_ent[pre_idx]):
                            no_latent_trigger_pre_context_entropies.append(float(nl_ent[pre_idx]))
                            break

    # ============================================================
    # 新增: 基于 top1_prob 的信息论分析
    # ============================================================
    # top1_prob = 模型对 next-token 最优选择的置信度
    # uncertainty = 1 - top1_prob, 即"困难度指数"
    trigger_top1_probs = []       # 触发位置的 top1_prob
    non_trigger_top1_probs = []   # 非触发位置的 top1_prob
    trigger_uncertainties = []    # 触发位置的 1 - top1_prob (困难度)
    non_trigger_uncertainties = [] # 非触发位置的困难度
    trigger_top1_pre = []         # 触发前1个位置的 top1_prob
    trigger_top1_post = []        # 触发后第1个正常位置的 top1_prob (with-latent)
    no_latent_trigger_top1 = []   # no-latent 中"假触发"位置的 top1_prob

    # 概率分布集中度指标 (基于 top-k probs 计算 Gini)
    trigger_gini = []             # 触发位置的 Gini 系数 (低=均匀分布/高不确定性)
    non_trigger_gini = []         # 非触发位置的 Gini 系数

    for sample in per_sample_data:
        wl = sample.get("with_latent", {})
        nl = sample.get("no_latent", {})

        # ---- with-latent: Top1 prob 分析 ----
        if wl:
            wl_top1 = np.array(wl.get('top1_probs', []), dtype=float)
            wl_topk = wl.get('topk_probs_list', [])
            wl_trig = wl.get('latent_trigger_positions', [])
            wl_ent = np.array(wl.get('token_entropies', []), dtype=float)
            n = len(wl_top1)

            # 触发位置 top1_prob
            for t_pos in wl_trig:
                if t_pos < n and not np.isnan(wl_top1[t_pos]):
                    trigger_top1_probs.append(float(wl_top1[t_pos]))
                    trigger_uncertainties.append(1.0 - float(wl_top1[t_pos]))
                    # Gini 系数 (基于 top-k 概率分布)
                    if t_pos < len(wl_topk) and wl_topk[t_pos] is not None:
                        pk = np.array(wl_topk[t_pos])
                        pk = pk / pk.sum()  # 归一化
                        g = 1 - np.sum(pk ** 2) * len(pk) / (len(pk) - 1) if len(pk) > 1 else 0
                        trigger_gini.append(float(g))
                    # 触发前/后 top1_prob
                    for pre_idx in range(t_pos - 1, -1, -1):
                        if pre_idx >= 0 and not np.isnan(wl_top1[pre_idx]):
                            trigger_top1_pre.append(float(wl_top1[pre_idx]))
                            break
                    for post_idx in range(t_pos + 1, n):
                        if post_idx < n and not np.isnan(wl_top1[post_idx]):
                            trigger_top1_post.append(float(wl_top1[post_idx]))
                            break

            # 非触发位置 top1_prob
            trigger_set = set(wl_trig)
            for i in range(n):
                if i not in trigger_set and not np.isnan(wl_top1[i]):
                    non_trigger_top1_probs.append(float(wl_top1[i]))
                    non_trigger_uncertainties.append(1.0 - float(wl_top1[i]))
                    if i < len(wl_topk) and wl_topk[i] is not None:
                        pk = np.array(wl_topk[i])
                        pk = pk / pk.sum()
                        g = 1 - np.sum(pk ** 2) * len(pk) / (len(pk) - 1) if len(pk) > 1 else 0
                        non_trigger_gini.append(float(g))

        # ---- no-latent:"假触发"位置 top1_prob ----
        if nl:
            nl_top1 = np.array(nl.get('top1_probs', []), dtype=float)
            nl_trig = nl.get('latent_trigger_positions', [])
            for t_pos in nl_trig:
                if t_pos < len(nl_top1) and not np.isnan(nl_top1[t_pos]):
                    no_latent_trigger_top1.append(float(nl_top1[t_pos]))

    summary = {
        "n_samples_with_latent": sum(1 for s in per_sample_data if s.get('with_latent', s).get('latent_trigger_positions')),
        "n_samples_without_latent": sum(1 for s in per_sample_data if not s.get('with_latent', s).get('latent_trigger_positions')),
        "total_trigger_positions": len(trigger_entropies),
        "total_non_trigger_positions": len(non_trigger_entropies),
        "trigger_entropy_mean": float(np.mean(trigger_entropies)) if trigger_entropies else None,
        "trigger_entropy_median": float(np.median(trigger_entropies)) if trigger_entropies else None,
        "trigger_entropy_std": float(np.std(trigger_entropies)) if trigger_entropies else None,
        "non_trigger_entropy_mean": float(np.mean(non_trigger_entropies)) if non_trigger_entropies else None,
        "non_trigger_entropy_median": float(np.median(non_trigger_entropies)) if non_trigger_entropies else None,
        "non_trigger_entropy_std": float(np.std(non_trigger_entropies)) if non_trigger_entropies else None,
        "trigger_entropies": trigger_entropies,
        "non_trigger_entropies": non_trigger_entropies[:50000],
        "trigger_percentile_ranks": trigger_percentile_ranks,
        "trigger_percentile_mean": float(np.mean(trigger_percentile_ranks)) if trigger_percentile_ranks else None,
        "local_peak_alignment": local_peak_alignment,
        # ---- 新增: 上下文熵分析 ----
        "trigger_pre_context_entropy_mean": float(np.mean(trigger_pre_context_entropies)) if trigger_pre_context_entropies else None,
        "trigger_pre_context_entropy_median": float(np.median(trigger_pre_context_entropies)) if trigger_pre_context_entropies else None,
        "trigger_post_context_entropy_mean": float(np.mean(trigger_post_context_entropies)) if trigger_post_context_entropies else None,
        "trigger_post_context_entropy_median": float(np.median(trigger_post_context_entropies)) if trigger_post_context_entropies else None,
        "no_latent_trigger_entropy_mean": float(np.mean(no_latent_trigger_entropies)) if no_latent_trigger_entropies else None,
        "no_latent_trigger_entropy_median": float(np.median(no_latent_trigger_entropies)) if no_latent_trigger_entropies else None,
        "no_latent_post_trigger_entropy_mean": float(np.mean(no_latent_post_trigger_entropies)) if no_latent_post_trigger_entropies else None,
        "no_latent_post_trigger_entropy_median": float(np.median(no_latent_post_trigger_entropies)) if no_latent_post_trigger_entropies else None,
        "no_latent_trigger_pre_context_entropy_mean": float(np.mean(no_latent_trigger_pre_context_entropies)) if no_latent_trigger_pre_context_entropies else None,
        "no_latent_trigger_pre_context_entropy_median": float(np.median(no_latent_trigger_pre_context_entropies)) if no_latent_trigger_pre_context_entropies else None,
        # 原始列表 (用于绘图)
        "trigger_pre_context_entropies": trigger_pre_context_entropies,
        "trigger_post_context_entropies": trigger_post_context_entropies,
        "no_latent_trigger_entropies": no_latent_trigger_entropies,
        "no_latent_post_trigger_entropies": no_latent_post_trigger_entropies,
        "no_latent_trigger_pre_context_entropies": no_latent_trigger_pre_context_entropies,
        # ---- FLOPs & latency 对比 ----
        "flops_ratio_mean": float(np.mean(flops_ratios)) if flops_ratios else None,
        "flops_ratio_median": float(np.median(flops_ratios)) if flops_ratios else None,
        "latency_ratio_mean": float(np.mean(latency_ratios)) if latency_ratios else None,
        "latency_ratio_median": float(np.median(latency_ratios)) if latency_ratios else None,
        "flops_overhead_mean": float(np.mean(flops_overheads)) if flops_overheads else None,
        "flops_overhead_median": float(np.median(flops_overheads)) if flops_overheads else None,
        "latency_overhead_mean": float(np.mean(latency_overheads)) if latency_overheads else None,
        "latency_overhead_median": float(np.median(latency_overheads)) if latency_overheads else None,
        "with_latent_flops_mean": float(np.mean(with_latent_flops_list)) if with_latent_flops_list else None,
        "no_latent_flops_mean": float(np.mean(no_latent_flops_list)) if no_latent_flops_list else None,
        "with_latent_latency_mean": float(np.mean(with_latent_latency_list)) if with_latent_latency_list else None,
        "no_latent_latency_mean": float(np.mean(no_latent_latency_list)) if no_latent_latency_list else None,
        "with_latent_decode_steps_mean": float(np.mean(with_latent_decode_steps_list)) if with_latent_decode_steps_list else None,
        "no_latent_decode_steps_mean": float(np.mean(no_latent_decode_steps_list)) if no_latent_decode_steps_list else None,
        # ---- 原始列表数据 (用于绘图) ----
        "flops_ratios": flops_ratios,
        "latency_ratios": latency_ratios,
        "with_latent_flops_list": with_latent_flops_list,
        "no_latent_flops_list": no_latent_flops_list,
        "with_latent_latency_list": with_latent_latency_list,
        "no_latent_latency_list": no_latent_latency_list,
        "with_latent_decode_steps_list": with_latent_decode_steps_list,
        "no_latent_decode_steps_list": no_latent_decode_steps_list,
        # ---- 信息论指标: Top1 Prob & Uncertainty ----
        "trigger_top1_prob_mean": float(np.mean(trigger_top1_probs)) if trigger_top1_probs else None,
        "trigger_top1_prob_median": float(np.median(trigger_top1_probs)) if trigger_top1_probs else None,
        "non_trigger_top1_prob_mean": float(np.mean(non_trigger_top1_probs)) if non_trigger_top1_probs else None,
        "non_trigger_top1_prob_median": float(np.median(non_trigger_top1_probs)) if non_trigger_top1_probs else None,
        "trigger_uncertainty_mean": float(np.mean(trigger_uncertainties)) if trigger_uncertainties else None,
        "trigger_uncertainty_median": float(np.median(trigger_uncertainties)) if trigger_uncertainties else None,
        "non_trigger_uncertainty_mean": float(np.mean(non_trigger_uncertainties)) if non_trigger_uncertainties else None,
        "non_trigger_uncertainty_median": float(np.median(non_trigger_uncertainties)) if non_trigger_uncertainties else None,
        "uncertainty_ratio": float(np.mean(trigger_uncertainties)) / float(np.mean(non_trigger_uncertainties)) if trigger_uncertainties and non_trigger_uncertainties and np.mean(non_trigger_uncertainties) > 0 else None,
        "trigger_gini_mean": float(np.mean(trigger_gini)) if trigger_gini else None,
        "non_trigger_gini_mean": float(np.mean(non_trigger_gini)) if non_trigger_gini else None,
        "trigger_top1_pre_mean": float(np.mean(trigger_top1_pre)) if trigger_top1_pre else None,
        "trigger_top1_pre_median": float(np.median(trigger_top1_pre)) if trigger_top1_pre else None,
        "trigger_top1_post_mean": float(np.mean(trigger_top1_post)) if trigger_top1_post else None,
        "trigger_top1_post_median": float(np.median(trigger_top1_post)) if trigger_top1_post else None,
        "no_latent_trigger_top1_mean": float(np.mean(no_latent_trigger_top1)) if no_latent_trigger_top1 else None,
        # 原始列表 (用于绘图)
        "trigger_top1_probs": trigger_top1_probs,
        "non_trigger_top1_probs": non_trigger_top1_probs,
        "trigger_uncertainties": trigger_uncertainties,
        "non_trigger_uncertainties": non_trigger_uncertainties,
        "trigger_gini": trigger_gini,
        "non_trigger_gini": non_trigger_gini,
        "trigger_top1_pre": trigger_top1_pre,
        "trigger_top1_post": trigger_top1_post,
        "no_latent_trigger_top1": no_latent_trigger_top1,
    }
    return summary


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Latent 触发位置与 Token 熵的关系分析")
    parser.add_argument("--model_path", type=str,
                        default="<PATH_TO_QWEN3_VL_8B_INSTRUCT>")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LatentDraft 模型 checkpoint 路径")
    parser.add_argument("--data_file", type=str, default="./data/erqa/erqa_test.jsonl")
    parser.add_argument("--data_root", type=str, default="./data/erqa")
    parser.add_argument("--output_dir", type=str, default="./outputs/entropy_trigger_analysis")
    parser.add_argument("--max_samples", type=int, default=200, help="0=全量")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    # ---- DDP 初始化 ----
    rank, world_size, local_rank = _maybe_init_dist()
    is_main = (rank == 0)
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 加载模型 (沿用 eval_erqa.py 的方式) ----
    if is_main:
        print(f"[Entropy] 加载基座模型: {args.model_path}")
    model = NLDModel(model_path=args.model_path, torch_dtype=dtype, attn_implementation="flash_attention_2")
    processor = AutoProcessor.from_pretrained(args.model_path)

    existing_specials = set(processor.tokenizer.additional_special_tokens or [])
    to_add = [t for t in [LATENT_TOKEN, LATENT_END_TOKEN] if t not in existing_specials]
    if to_add:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    _V_old = model.base_model.get_input_embeddings().weight.shape[0]
    _target_vocab = max(len(processor.tokenizer), _V_old)
    if _target_vocab != _V_old:
        model.base_model.resize_token_embeddings(_target_vocab)
    _V_new = model.base_model.get_input_embeddings().weight.shape[0]
    model.set_processor(processor)
    if is_main:
        print(f"[Entropy] embedding: {_V_old} -> {_V_new}, tokenizer={len(processor.tokenizer)}")

    if args.checkpoint:
        if is_main:
            print(f"[Entropy] 加载 checkpoint: {args.checkpoint}")
        model.load_pretrained(args.checkpoint)

    model = model.to(args.device).eval()
    if is_main:
        print(f"[Entropy] <|latent|> id = {model.latent_token_id}")
        print(f"[Entropy] world_size={world_size}")

    # ---- 加载数据 ----
    samples_all = load_erqa_data(args.data_file, args.data_root, args.max_samples)
    n_global = len(samples_all)

    # ---- 多卡分片: rank i 处理 i, i+world_size, i+2*world_size... ----
    samples = samples_all[rank::world_size] if world_size > 1 else samples_all
    n_local = len(samples)
    print(f"[Entropy] [r{rank}] 本 rank 负责: {n_local} 条 (全局 {n_global})")

    # ---- 推理 ----
    per_sample_data = []
    for idx, item in enumerate(samples):
        question = item.get("question", "")
        answer_gt = item.get("answer", "")
        q_type = item.get("question_type", "unknown")
        qid = item.get("question_id", f"q_{idx}")

        # 构造图片路径
        image_paths = []
        for key in ["image_path", "image"]:
            val = item.get(key, "")
            if val:
                if not os.path.isabs(val):
                    val = os.path.join(args.data_root, val)
                if os.path.exists(val):
                    image_paths.append(val)
        for i in range(10):
            key = f"image_path_{i}"
            val = item.get(key, "")
            if val:
                if not os.path.isabs(val):
                    val = os.path.join(args.data_root, val)
                if os.path.exists(val):
                    image_paths.append(val)

        # 构造 messages
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": f"file://{img_path}"})
        content.append({"type": "text", "text": question})

        messages = [
            {"role": "system", "content": NLD_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        # ---- 计时: 全流程 ----
        sample_total_start = time.perf_counter()

        try:
            # ===== With Latent 推理 =====
            with_latent_start = time.perf_counter()
            result_with = generate_with_entropy_record(
                model, processor, messages,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            with_latent_end = time.perf_counter()

            # Prefill latency 近似: with_latent 总时间 - decode 时间
            # 但更精确的做法是在 generate_with_entropy_record 内部计时 prefill
            # 这里先用 generate_with_entropy_record 返回的 prefill_latency_s
            prefill_latency_with = result_with.get("prefill_latency_s", None)
            decode_latency_with = result_with.get("decode_latency_s", 0)
            total_latency_with = with_latent_end - with_latent_start
            if prefill_latency_with is None:
                # 近似: 总时间 - decode 时间 ≈ prefill
                prefill_latency_with = total_latency_with - decode_latency_with

            # 两次独立 forward，确保 KV cache / rope_deltas 等不互相污染
            if hasattr(model, '_inner_model'):
                inner_model = model._inner_model
                if hasattr(inner_model, 'rope_deltas'):
                    inner_model.rope_deltas = None
            torch.cuda.synchronize(args.device)
            torch.cuda.empty_cache()

            # ===== No Latent 推理 =====
            no_latent_start = time.perf_counter()
            result_no = generate_without_latent_record(
                model, processor, messages,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            no_latent_end = time.perf_counter()

            prefill_latency_no = result_no.get("prefill_latency_s", None)
            decode_latency_no = result_no.get("decode_latency_s", 0)
            total_latency_no = no_latent_end - no_latent_start
            if prefill_latency_no is None:
                prefill_latency_no = total_latency_no - decode_latency_no

            # ===== 合并结果 =====
            result = {
                "question_id": qid,
                "question_type": q_type,
                "answer_gt": answer_gt,
                # ---- with-latent 结果 ----
                "with_latent": {
                    "generated_text": result_with["generated_text"],
                    "generated_text_clean": result_with["generated_text_clean"],
                    "generated_token_ids": result_with["generated_token_ids"],
                    "token_entropies": result_with["token_entropies"],
                    "latent_trigger_positions": result_with["latent_trigger_positions"],
                    "latent_end_positions": result_with["latent_end_positions"],
                    "num_latent_triggers": result_with["num_latent_triggers"],
                    "prompt_len": result_with["prompt_len"],
                    "num_decode_steps": result_with["num_decode_steps"],
                    # ---- 效率指标 ----
                    "total_flops": result_with["total_flops"],
                    "prefill_flops": result_with["prefill_flops"],
                    "decode_flops_per_token": result_with["decode_flops_per_token"],
                    "prefill_latency_s": prefill_latency_with,
                    "decode_latency_s": decode_latency_with,
                    "total_latency_s": total_latency_with,
                },
                # ---- no-latent 结果 ----
                "no_latent": {
                    "generated_text": result_no["generated_text"],
                    "generated_text_clean": result_no["generated_text_clean"],
                    "generated_token_ids": result_no["generated_token_ids"],
                    "token_entropies": result_no["token_entropies"],
                    "latent_trigger_positions": result_no["latent_trigger_positions"],
                    "latent_end_positions": result_no["latent_end_positions"],
                    "num_latent_triggers": result_no["num_latent_triggers"],
                    "prompt_len": result_no["prompt_len"],
                    "num_decode_steps": result_no["num_decode_steps"],
                    # ---- 效率指标 ----
                    "total_flops": result_no["total_flops"],
                    "prefill_flops": result_no["prefill_flops"],
                    "decode_flops_per_token": result_no["decode_flops_per_token"],
                    "prefill_latency_s": prefill_latency_no,
                    "decode_latency_s": decode_latency_no,
                    "total_latency_s": total_latency_no,
                },
                # ---- 对比指标 ----
                "flops_ratio": (
                    result_with["total_flops"] / result_no["total_flops"]
                    if result_no["total_flops"] > 0 else None
                ),
                "latency_ratio": (
                    total_latency_with / total_latency_no
                    if total_latency_no > 0 else None
                ),
                "flops_overhead": (
                    result_with["total_flops"] - result_no["total_flops"]
                ),
                "latency_overhead": (
                    total_latency_with - total_latency_no
                ),
            }
            per_sample_data.append(result)

            n_trig = result_with["num_latent_triggers"]
            mean_ent = np.nanmean(result_with["token_entropies"]) if result_with["token_entropies"] else 0
            flops_r = result["flops_ratio"]
            lat_r = result["latency_ratio"]
            tag = f"r{rank} " if world_size > 1 else ""
            if (idx + 1) % 10 == 0 or n_trig > 0:
                print(f"  [{tag}{idx+1}/{n_local}] {qid} | latent×{n_trig} | "
                      f"mean_entropy={mean_ent:.3f} | "
                      f"FLOPs×{flops_r:.2f} | latency×{lat_r:.2f}")

        except Exception as e:
            tag = f"r{rank} " if world_size > 1 else ""
            print(f"  [{tag}{idx+1}/{n_local}] {qid} | ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # ===== 多卡聚合 (方案 B: 每 rank 落盘 + 主 rank 轮询合并, 无任何 NCCL collective) =====
    if world_size > 1:
        # 打包本 rank 产出
        local_payload = {
            'per_sample_data': per_sample_data,
            'rank': rank,
        }

        # ---- 每 rank 各自把 shard 写到 output_dir (atomic write) ----
        os.makedirs(args.output_dir, exist_ok=True)
        shard_path = os.path.join(
            args.output_dir, f".shard_rank{rank}_of{world_size}.json"
        )
        shard_tmp = shard_path + ".tmp"
        with open(shard_tmp, "w", encoding="utf-8") as f:
            json.dump(local_payload, f, ensure_ascii=False)
        os.replace(shard_tmp, shard_path)
        print(f"[Entropy] [r{rank}] shard 已落盘: {shard_path} "
              f"({len(per_sample_data)} 条)", flush=True)

        if not is_main:
            # 非主 rank 写完分片就退出, 不参与合并
            return

        # ---- 主 rank: 轮询等待所有分片落盘, 然后合并 ----
        expected = [
            os.path.join(args.output_dir, f".shard_rank{r}_of{world_size}.json")
            for r in range(world_size)
        ]
        WAIT_TIMEOUT_S = float(os.environ.get('SHARD_WAIT_TIMEOUT_S', '14400'))
        WAIT_INTERVAL_S = 5.0
        REPORT_INTERVAL_S = 30.0
        wait_start = time.time()
        last_report = wait_start
        while True:
            missing = [p for p in expected if not os.path.exists(p)]
            if not missing:
                print(f"[Entropy] [r0] 所有 {world_size} 个 shard 已就绪, 开始合并", flush=True)
                break
            now = time.time()
            if now - wait_start > WAIT_TIMEOUT_S:
                print(f"[Entropy] [r0] ⚠️ 等待 shard 超时 ({WAIT_TIMEOUT_S}s), "
                      f"仍缺 {len(missing)} 个分片, 用现有分片合并", flush=True)
                break
            if now - last_report > REPORT_INTERVAL_S:
                ready = world_size - len(missing)
                print(f"[Entropy] [r0] 等待 shard 中... {ready}/{world_size} 就绪 "
                      f"(已等 {now-wait_start:.0f}s)", flush=True)
                last_report = now
            time.sleep(WAIT_INTERVAL_S)

        # ---- 读盘 + 合并 ----
        gathered = []
        for p in expected:
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    gathered.append(json.load(f))
            except Exception as _err:
                print(f"[Entropy] [r0] ⚠️ 读取 shard 失败 {p}: {_err}", flush=True)

        # 合并所有 shard 的 per_sample_data
        per_sample_data = []
        for payload in gathered:
            per_sample_data.extend(payload.get('per_sample_data', []))

        # 清理临时 shard 文件
        for p in expected:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        print(f"[Entropy] [r0] 合并完成, 总计 {len(per_sample_data)} 条样本")

    # ---- 统计分析 (仅主 rank 执行) ----
    if not is_main:
        return

    print("\n" + "=" * 60)
    print("  统计分析")
    print("=" * 60)
    summary = compute_statistics(per_sample_data)

    print(f"  有 latent 触发的样本: {summary['n_samples_with_latent']}")
    print(f"  无 latent 触发的样本: {summary['n_samples_without_latent']}")
    print(f"  总触发位置数: {summary['total_trigger_positions']}")
    print(f"  总非触发位置数: {summary['total_non_trigger_positions']}")

    if summary['trigger_entropy_mean'] is not None:
        print(f"\n  触发位置平均熵: {summary['trigger_entropy_mean']:.4f} nats")
        print(f"  触发位置中位熵: {summary['trigger_entropy_median']:.4f} nats")
    if summary['non_trigger_entropy_mean'] is not None:
        print(f"  非触发位置平均熵: {summary['non_trigger_entropy_mean']:.4f} nats")
        print(f"  非触发位置中位熵: {summary['non_trigger_entropy_median']:.4f} nats")
    if summary['trigger_entropy_mean'] is not None and summary['non_trigger_entropy_mean'] is not None:
        ratio = summary['trigger_entropy_mean'] / max(summary['non_trigger_entropy_mean'], 1e-10)
        print(f"  触发/非触发熵比: {ratio:.3f}")

    if summary['trigger_percentile_mean'] is not None:
        print(f"\n  触发位置熵百分位均值: {summary['trigger_percentile_mean']:.1f}th pct")
        print(f"  (如果 > 50, 说明 latent 倾向于在高熵位置触发)")

    if summary['local_peak_alignment']:
        print(f"\n  局部熵峰值对齐率:")
        for W, rate in sorted(summary['local_peak_alignment'].items()):
            print(f"    窗口±{W}: {rate:.1%}")

    # ---- 新增: 上下文熵分析 (判断 latent 是否触发在推理难点) ----
    print("\n" + "=" * 60)
    print("  Latent 触发是否落在推理难点? (上下文熵分析)")
    print("=" * 60)

    # With-latent: 触发前后上下文
    print("\n  [With-Latent 模式]")
    if summary.get('trigger_pre_context_entropy_mean') is not None:
        print(f"    触发前1个正常 token 平均熵: {summary['trigger_pre_context_entropy_mean']:.4f} nats")
    if summary.get('trigger_post_context_entropy_mean') is not None:
        print(f"    触发后第1个正常 token 平均熵: {summary['trigger_post_context_entropy_mean']:.4f} nats")
    if summary.get('trigger_pre_context_entropy_mean') is not None and summary.get('trigger_post_context_entropy_mean') is not None:
        pre_post_ratio = summary['trigger_post_context_entropy_mean'] / max(summary['trigger_pre_context_entropy_mean'], 1e-10)
        print(f"    后/前 熵比: {pre_post_ratio:.3f} (>1 说明 latent 后不确定性增加)")

    # No-latent: 模型"想触发"但无 thinking 时的熵
    print("\n  [No-Latent 模式: 模型'想触发'但未 thinking]")
    if summary.get('no_latent_trigger_entropy_mean') is not None:
        print(f"    '想触发'位置平均熵: {summary['no_latent_trigger_entropy_mean']:.4f} nats")
    if summary.get('no_latent_trigger_pre_context_entropy_mean') is not None:
        print(f"    触发前1个正常 token 平均熵: {summary['no_latent_trigger_pre_context_entropy_mean']:.4f} nats")
    if summary.get('no_latent_post_trigger_entropy_mean') is not None:
        print(f"    触发后第1个正常 token 平均熵: {summary['no_latent_post_trigger_entropy_mean']:.4f} nats")

    # 关键对比: no-latent trigger 熵 vs 正常非触发熵
    if summary.get('no_latent_trigger_entropy_mean') is not None and summary.get('trigger_pre_context_entropy_mean') is not None:
        print(f"\n  >> 关键判断指标: No-latent trigger 位置熵 vs 触发前上下文熵:")
        diff = summary['no_latent_trigger_entropy_mean'] - summary['trigger_pre_context_entropy_mean']
        print(f"     差值: {diff:+.4f} nats")
        if diff > 0.1:
            print(f"     ✅ 结论: 触发位置熵 > 触发前上下文熵, latent 倾向于在推理瓶颈处触发")
        elif diff > -0.1:
            print(f"     ⚠️ 结论: 触发位置熵 ≈ 上下文熵, latent 触发与难点关联不明显")
        else:
            print(f"     ❌ 结论: 触发位置熵 < 上下文熵, latent 可能未触发在真正的推理难点")

    # ---- FLOPs & latency 对比 ----
    print("\n" + "=" * 60)
    print("  With-Latent vs No-Latent 效率对比")
    print("=" * 60)
    if summary['flops_ratio_mean'] is not None:
        print(f"  FLOPs 比值 (with/no): {summary['flops_ratio_mean']:.3f}x "
              f"(中位: {summary['flops_ratio_median']:.3f}x)")
    if summary['latency_ratio_mean'] is not None:
        print(f"  Latency 比值 (with/no): {summary['latency_ratio_mean']:.3f}x "
              f"(中位: {summary['latency_ratio_median']:.3f}x)")
    if summary['flops_overhead_mean'] is not None:
        print(f"  FLOPs 额外开销: {summary['flops_overhead_mean']:.3e} "
              f"(中位: {summary['flops_overhead_median']:.3e})")
    if summary['latency_overhead_mean'] is not None:
        print(f"  Latency 额外开销: {summary['latency_overhead_mean']:.3f}s "
              f"(中位: {summary['latency_overhead_median']:.3f}s)")
    if summary['with_latent_flops_mean'] is not None and summary['no_latent_flops_mean'] is not None:
        print(f"\n  With-latent 平均 FLOPs: {summary['with_latent_flops_mean']:.3e}")
        print(f"  No-latent   平均 FLOPs: {summary['no_latent_flops_mean']:.3e}")
    if summary['with_latent_latency_mean'] is not None and summary['no_latent_latency_mean'] is not None:
        print(f"  With-latent 平均 latency: {summary['with_latent_latency_mean']:.3f}s")
        print(f"  No-latent   平均 latency: {summary['no_latent_latency_mean']:.3f}s")
    if summary['with_latent_decode_steps_mean'] is not None and summary['no_latent_decode_steps_mean'] is not None:
        print(f"  With-latent 平均 decode 步数: {summary['with_latent_decode_steps_mean']:.1f}")
        print(f"  No-latent   平均 decode 步数: {summary['no_latent_decode_steps_mean']:.1f}")

    # ---- 信息论指标对比 (Top1 Prob & Uncertainty) ----
    print("\n" + "=" * 60)
    print("  信息论指标对比: Top1-概率 vs 不确定性 (困难度指数)")
    print("=" * 60)
    print("  【核心指标: 触发位置 vs 非触发位置的 Top1-概率/不确定性】")
    if summary['trigger_top1_prob_mean'] is not None:
        print(f"    触发位置平均 top-1 概率: {summary['trigger_top1_prob_mean']:.4f}")
    if summary['non_trigger_top1_prob_mean'] is not None:
        print(f"    非触发位置平均 top-1 概率: {summary['non_trigger_top1_prob_mean']:.4f}")
    if summary['trigger_uncertainty_mean'] is not None:
        print(f"    触发位置平均困难度 (1-top1): {summary['trigger_uncertainty_mean']:.4f}")
    if summary['non_trigger_uncertainty_mean'] is not None:
        print(f"    非触发位置平均困难度: {summary['non_trigger_uncertainty_mean']:.4f}")
    if summary['uncertainty_ratio'] is not None:
        print(f"    困难度比值 (触发/非触发): {summary['uncertainty_ratio']:.3f}x")
        if summary['uncertainty_ratio'] > 1.2:
            print(f"    ✅ 结论: 触发位置比非触发位置更困难 (不确定性高 {summary['uncertainty_ratio']:.2f}x)")
        elif summary['uncertainty_ratio'] < 0.8:
            print(f"    ❌ 结论: 触发位置比非触发位置更容易 (不确定性低 {1/summary['uncertainty_ratio']:.2f}x)")
        else:
            print(f"    ⚠️ 结论: 触发位置与非触发位置难度相近")

    print("\n  【触发前后对比 (with-latent)】")
    if summary['trigger_top1_pre_mean'] is not None:
        print(f"    触发前1个位置 top-1 概率: {summary['trigger_top1_pre_mean']:.4f}")
    if summary['trigger_top1_prob_mean'] is not None:
        print(f"    触发位置 top-1 概率: {summary['trigger_top1_prob_mean']:.4f}")
    if summary['trigger_top1_post_mean'] is not None:
        print(f"    触发后第1个正常位置 top-1 概率: {summary['trigger_top1_post_mean']:.4f}")
    if summary['trigger_top1_pre_mean'] and summary['trigger_top1_prob_mean'] and summary['trigger_top1_post_mean']:
        diff_pre = summary['trigger_top1_prob_mean'] - summary['trigger_top1_pre_mean']
        diff_post = summary['trigger_top1_post_mean'] - summary['trigger_top1_prob_mean']
        print(f"    触发时骤降: {diff_pre:+.4f} (负值=难度骤增)")
        print(f"    触发后回升: {diff_post:+.4f} (正值=latent有帮助)")

    print("\n  【No-Latent 模式下 '假触发'位置分析】")
    if summary['no_latent_trigger_top1_mean'] is not None:
        print(f"    '假触发'位置平均 top-1 概率: {summary['no_latent_trigger_top1_mean']:.4f}")
        print(f"    '假触发'位置困难度: {1 - summary['no_latent_trigger_top1_mean']:.4f}")
    if summary['trigger_top1_prob_mean'] is not None and summary['no_latent_trigger_top1_mean'] is not None:
        print(f"    对比: With-latent触发={summary['trigger_top1_prob_mean']:.4f} vs No-latent假触发={summary['no_latent_trigger_top1_mean']:.4f}")

    print("\n  【概率分布集中度 (Gini系数)】")
    if summary['trigger_gini_mean'] is not None:
        print(f"    触发位置平均 Gini: {summary['trigger_gini_mean']:.4f}")
    if summary['non_trigger_gini_mean'] is not None:
        print(f"    非触发位置平均 Gini: {summary['non_trigger_gini_mean']:.4f}")
    if summary['trigger_gini_mean'] and summary['non_trigger_gini_mean']:
        print(f"    (Gini越高=分布越不均匀=模型越'确信'自己的判断)")

    # ---- 保存结果 ----
    per_sample_save = []
    for s in per_sample_data:
        wl = s.get("with_latent", s)
        trigger_ents = [wl["token_entropies"][p] for p in wl["latent_trigger_positions"]
                        if p < len(wl["token_entropies"])]
        nl = s.get("no_latent", {})
        d = {
            "question_id": s.get("question_id"),
            "question_type": s.get("question_type"),
            "num_latent_triggers": wl.get("num_latent_triggers", 0),
            "latent_trigger_positions": wl.get("latent_trigger_positions", []),
            "latent_end_positions": wl.get("latent_end_positions", []),
            "n_generated_tokens_with_latent": len(wl.get("generated_token_ids", [])),
            "n_generated_tokens_no_latent": len(nl.get("generated_token_ids", [])),
            "mean_entropy_with_latent": float(np.nanmean(wl["token_entropies"])) if wl.get("token_entropies") else None,
            "trigger_entropy_mean_with_latent": float(np.nanmean(trigger_ents)) if trigger_ents else None,
            # ---- FLOPs & latency 对比 ----
            "flops_ratio": s.get("flops_ratio"),
            "latency_ratio": s.get("latency_ratio"),
            "flops_overhead": s.get("flops_overhead"),
            "latency_overhead": s.get("latency_overhead"),
            "with_latent_flops": wl.get("total_flops"),
            "no_latent_flops": nl.get("total_flops"),
            "with_latent_latency": wl.get("total_latency_s"),
            "no_latent_latency": nl.get("total_latency_s"),
            "with_latent_decode_steps": wl.get("num_decode_steps"),
            "no_latent_decode_steps": nl.get("num_decode_steps"),
        }
        per_sample_save.append(d)

    with open(os.path.join(args.output_dir, "per_sample_entropy.json"), 'w') as f:
        json.dump(per_sample_save, f, indent=2, ensure_ascii=False)

    summary_save = {k: v for k, v in summary.items()
                    if k not in ("trigger_entropies", "non_trigger_entropies")}
    with open(os.path.join(args.output_dir, "summary_stats.json"), 'w') as f:
        json.dump(summary_save, f, indent=2, ensure_ascii=False)

    # ---- 保存 with-latent/no-latent 对比汇总 ----
    comparison_save = {
        "description": "With-latent vs No-latent 推理模式对比",
        "n_samples": len(per_sample_data),
        "with_latent_avg_flops": summary.get("with_latent_flops_mean"),
        "no_latent_avg_flops": summary.get("no_latent_flops_mean"),
        "with_latent_avg_latency": summary.get("with_latent_latency_mean"),
        "no_latent_avg_latency": summary.get("no_latent_latency_mean"),
        "with_latent_avg_decode_steps": summary.get("with_latent_decode_steps_mean"),
        "no_latent_avg_decode_steps": summary.get("no_latent_decode_steps_mean"),
        "flops_ratio_mean": summary.get("flops_ratio_mean"),
        "latency_ratio_mean": summary.get("latency_ratio_mean"),
        "flops_overhead_mean": summary.get("flops_overhead_mean"),
        "latency_overhead_mean": summary.get("latency_overhead_mean"),
    }
    with open(os.path.join(args.output_dir, "comparison_with_no_latent.json"), 'w') as f:
        json.dump(comparison_save, f, indent=2, ensure_ascii=False)
    # ---- 可视化 ----
    plot_results(args.output_dir, per_sample_data, summary)

    print(f"\n✅ 分析完成, 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
