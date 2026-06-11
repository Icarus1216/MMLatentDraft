#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_manifold_with_generation.py
====================================
模态流形分析脚本 (Latent Reasoning 版)：

核心设计：
1. 使用 NLDModel 加载训练后的 checkpoint，触发完整的 Latent Reasoning 流程
2. 使用真实训练数据 (erqa_vsibench_merged) 作为 query，样本量 200+
3. 正确定义三种模态：
   - V (Vision Embedding): 视觉 token 进入 LLM 第一层之前的表征
     即 hidden_states[0] 中视觉区域（经过 vision encoder + projection 后的 embedding）
   - T (Text Embedding): 文本 token 的 word embedding
     即 model.get_input_embeddings()(text_token_ids) 的输出
   - H (Latent Hidden): 触发 Latent Reasoning 后，NativeLatentThinker 输出的
     thought hidden states（隐空间推理后的表征）
     对于 Baseline 模型（无 latent 能力），H 取普通 generate 的 last_hidden_state

设计原理：
  V 和 T 代表"纯粹的视觉/语言模态"的原始表征（输入层）
  H 代表经过 Latent Reasoning 后的隐空间表征（NativeLatentThinker 输出）
  训练目标：Latent Reasoning 创造了"过渡模态"——H 同时编码视觉和语言信息
  即：cos(H,V) 训练后显著增大，同时 cos(H,T) 不显著下降
  这证明 NativeLatentThinker 的隐空间迭代确实在融合视觉-语言信息

用法:
  python3 extract_manifold_with_generation.py \\
      --ckpt_path /path/to/checkpoint-1200 \\
      --baseline_path /path/to/Qwen3-VL-8B-Instruct \\
      --data_path /path/to/erqa_vsibench_merged_1562.json \\
      --output_dir ./results \\
      --n_samples 200 \\
      --max_new_tokens 64
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


# ============================================================
# 1. 模型加载
# ============================================================
def load_baseline_model(model_path: str, device: str = 'cuda'):
    """加载 Baseline Qwen3-VL 模型 (原生 HuggingFace，无 Latent 能力)"""
    from transformers import AutoTokenizer, AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration

    print(f"  Loading baseline model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"  ✅ Baseline 模型加载完成, device={next(model.parameters()).device}")
    return model, tokenizer, processor


def load_nld_model(model_path: str, checkpoint_path: str, device: str = 'cuda'):
    """
    加载 NLD 模型 (含 NativeLatentThinker)，支持触发 Latent Reasoning。
    
    流程与 eval_erqa.py 中 load_nld_model 一致：
    1. 用 base model path 初始化 NLDModel
    2. 注册特殊 token (<|latent|>, <|/latent|>)
    3. 调整 embedding 大小
    4. set_processor
    5. load_pretrained(checkpoint_path)
    """
    from transformers import AutoProcessor
    from rld.model_v2 import NLDModel
    from rld.data import LATENT_TOKEN, LATENT_END_TOKEN

    print(f"  Loading NLD model from base: {model_path}")
    print(f"  Loading checkpoint: {checkpoint_path}")

    model = NLDModel(
        model_path=model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # 注册特殊 token
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [LATENT_TOKEN, LATENT_END_TOKEN]}
    )
    # embedding 大小适配 (与 eval_erqa.py 一致)
    _V_old = model.base_model.get_input_embeddings().weight.shape[0]
    _target_vocab = max(len(processor.tokenizer), _V_old)
    if _target_vocab != _V_old:
        model.base_model.resize_token_embeddings(_target_vocab)
    print(f"  注册特殊 token: {LATENT_TOKEN}, {LATENT_END_TOKEN} (新增 {num_added} 个)")
    print(f"  embedding: {_V_old} -> {model.base_model.get_input_embeddings().weight.shape[0]}")

    model.set_processor(processor)
    model.load_pretrained(checkpoint_path)

    model = model.to(device)
    model.eval()
    print(f"  ✅ NLD 模型加载完成 (含 NativeLatentThinker), device={device}")
    return model, processor.tokenizer, processor


# ============================================================
# 2. 数据加载
# ============================================================
def load_queries(data_path: str, n_samples: int, image_root: str = None) -> List[Dict]:
    """
    从训练数据中加载 query 样本。
    每个样本包含: image_paths, question
    """
    print(f"  加载数据: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  总样本数: {len(data)}")

    # 确定图片根目录
    if image_root is None:
        image_root = PROJ_ROOT

    # 筛选有效样本（图片存在）
    valid_samples = []
    for item in data:
        img_paths = item.get('image_paths', [])
        if not img_paths:
            continue

        # 检查第一张图片是否存在
        first_img = os.path.join(image_root, img_paths[0])
        if os.path.isfile(first_img):
            valid_samples.append({
                'image_paths': [os.path.join(image_root, p) for p in img_paths],
                'question': item.get('question', 'Describe this image.'),
                'task_type': item.get('task_type', 'unknown'),
            })

        if len(valid_samples) >= n_samples:
            break

    print(f"  有效样本: {len(valid_samples)} / {min(n_samples, len(data))}")
    return valid_samples


# ============================================================
# 3. 核心：提取三种模态的 embeddings
# ============================================================
def _prepare_inputs(query, processor, device):
    """构造单个样本的模型输入"""
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    img_path = query['image_paths'][0]
    img = Image.open(img_path).convert('RGB')

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": query['question']},
        ]},
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors='pt',
        padding=True,
    ).to(device)

    return inputs


def extract_baseline_embeddings(
    model, tokenizer, processor,
    queries: List[Dict],
    max_new_tokens: int = 64,
    device: str = 'cuda',
) -> Dict[str, np.ndarray]:
    """
    Baseline 模型 (无 Latent 能力) 的 embedding 提取。
    
    对每个 query:
    1. forward (output_hidden_states=True) 提取 V 和 T
    2. model.generate() 提取 H (普通生成 token 的 last_hidden_state)

    返回:
      {
        'vision': [N, hidden_dim]   — 视觉 embedding 均值（输入层）
        'text': [N, hidden_dim]     — 文本 embedding 均值（输入层）
        'generated': [N, hidden_dim] — 普通生成 token 的 hidden state 均值（输出层）
      }
    """
    embeddings = defaultdict(list)
    n_success = 0
    n_fail = 0

    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    print(f"\n  🔬 [Baseline] 开始提取 {len(queries)} 个样本的模态 embeddings...")
    print(f"     max_new_tokens={max_new_tokens}")
    print(f"     vision_start_id={vision_start_id}, vision_end_id={vision_end_id}")

    for idx, query in enumerate(queries):
        try:
            inputs = _prepare_inputs(query, processor, device)
            input_ids = inputs['input_ids']

            # Step A: Prefill forward — 提取 V 和 T
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            input_embeddings = outputs.hidden_states[0]  # [1, seq_len, hidden_dim]

            # 定位视觉 token 区域
            ids = input_ids[0].tolist()
            vis_start_pos = None
            vis_end_pos = None
            for pos, tid in enumerate(ids):
                if tid == vision_start_id and vis_start_pos is None:
                    vis_start_pos = pos + 1
                if tid == vision_end_id and vis_start_pos is not None:
                    vis_end_pos = pos
                    break

            if vis_start_pos is None or vis_end_pos is None or vis_end_pos <= vis_start_pos:
                n_fail += 1
                continue

            # V: 视觉区域的 input embedding 均值
            vis_emb = input_embeddings[0, vis_start_pos:vis_end_pos, :]
            embeddings['vision'].append(vis_emb.mean(dim=0).cpu().float().numpy())

            # T: 文本区域的 word embedding
            text_token_ids = input_ids[0, vis_end_pos+1:]
            if text_token_ids.shape[0] > 0:
                embed_layer = model.get_input_embeddings()
                with torch.no_grad():
                    txt_emb = embed_layer(text_token_ids)
                embeddings['text'].append(txt_emb.mean(dim=0).cpu().float().numpy())
            else:
                txt_emb = input_embeddings[0, vis_end_pos+1:, :]
                embeddings['text'].append(txt_emb.mean(dim=0).cpu().float().numpy())

            del outputs, input_embeddings
            torch.cuda.empty_cache()

            # Step B: Generate — 提取普通生成 token 的 hidden states
            with torch.no_grad():
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            gen_hidden_states = []
            for step_idx in range(1, len(gen_outputs.hidden_states)):
                step_hs = gen_outputs.hidden_states[step_idx]
                last_layer_hs = step_hs[-1]  # [1, 1, hidden_dim]
                gen_hidden_states.append(last_layer_hs[0, 0, :])

            if gen_hidden_states:
                gen_tensor = torch.stack(gen_hidden_states, dim=0)
                embeddings['generated'].append(gen_tensor.mean(dim=0).cpu().float().numpy())
            else:
                n_fail += 1
                continue

            del gen_outputs, gen_hidden_states
            torch.cuda.empty_cache()
            n_success += 1

        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"    ⚠️ 样本 {idx} 失败: {type(e).__name__}: {e}")
            elif n_fail == 6:
                print(f"    ⚠️ 后续失败不再逐条打印...")

        if (idx + 1) % 20 == 0:
            print(f"    [{idx+1}/{len(queries)}] success={n_success}, fail={n_fail}")

    print(f"\n  ✅ [Baseline] 提取完成: success={n_success}, fail={n_fail}")

    result = {}
    for k, v_list in embeddings.items():
        if v_list:
            result[k] = np.stack(v_list, axis=0)
            print(f"    {k}: shape={result[k].shape}")
    return result


def extract_nld_embeddings(
    model, tokenizer, processor,
    queries: List[Dict],
    max_new_tokens: int = 256,
    device: str = 'cuda',
) -> Dict[str, np.ndarray]:
    """
    NLD 模型 (含 NativeLatentThinker) 的 embedding 提取。
    
    对每个 query:
    1. forward (output_hidden_states=True) 提取 V 和 T (与 baseline 相同)
    2. 使用 NLD 的自定义 generate() 方法，触发 Latent Reasoning
       H = NativeLatentThinker 输出的 thought hidden states
       (即隐空间推理后的表征，而非普通文本生成的 hidden state)

    NLD generate 的 return_diagnostics=True 会返回每次 latent 触发的诊断信息，
    其中包含 thought hidden states 的统计。但 thought_output 本身不直接暴露。
    
    因此我们采用替代方案：
    - 在 generate 前 hook NativeLatentThinker 的 forward，
      拦截 thought_output [B, num_steps, H] 并收集。

    返回:
      {
        'vision': [N, hidden_dim]   — 视觉 embedding 均值（输入层）
        'text': [N, hidden_dim]     — 文本 embedding 均值（输入层）
        'generated': [N, hidden_dim] — latent thought hidden states 均值
      }
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from rld.data import NLD_SYSTEM_PROMPT

    embeddings = defaultdict(list)
    n_success = 0
    n_fail = 0
    n_no_latent = 0  # 未触发 latent 的样本数

    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    print(f"\n  🔬 [NLD] 开始提取 {len(queries)} 个样本的模态 embeddings...")
    print(f"     max_new_tokens={max_new_tokens}")
    print(f"     使用 NLD generate() 触发 Latent Reasoning")
    print(f"     H = NativeLatentThinker thought_output (隐空间推理后的表征)")

    for idx, query in enumerate(queries):
        try:
            # 构造消息 (使用 NLD system prompt 引导模型触发 latent)
            img_path = query['image_paths'][0]
            img = Image.open(img_path).convert('RGB')

            messages = [
                {"role": "system", "content": [{"type": "text", "text": NLD_SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": query['question']},
                ]},
            ]

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                return_tensors='pt',
                padding=True,
            ).to(device)

            input_ids = inputs['input_ids']

            # ============================================================
            # Step A: 提取 V 和 T (与 baseline 相同逻辑)
            # 使用 base_model 的 forward (Qwen3VLForConditionalGeneration)
            # ============================================================
            with torch.no_grad():
                outputs = model.base_model(**inputs, output_hidden_states=True)

            input_embeddings = outputs.hidden_states[0]  # [1, seq_len, hidden_dim]

            ids = input_ids[0].tolist()
            vis_start_pos = None
            vis_end_pos = None
            for pos, tid in enumerate(ids):
                if tid == vision_start_id and vis_start_pos is None:
                    vis_start_pos = pos + 1
                if tid == vision_end_id and vis_start_pos is not None:
                    vis_end_pos = pos
                    break

            if vis_start_pos is None or vis_end_pos is None or vis_end_pos <= vis_start_pos:
                n_fail += 1
                continue

            # V: 视觉区域的 input embedding 均值
            vis_emb = input_embeddings[0, vis_start_pos:vis_end_pos, :]
            embeddings['vision'].append(vis_emb.mean(dim=0).cpu().float().numpy())

            # T: 文本区域的 word embedding
            text_token_ids = input_ids[0, vis_end_pos+1:]
            if text_token_ids.shape[0] > 0:
                embed_layer = model.base_model.get_input_embeddings()
                with torch.no_grad():
                    txt_emb = embed_layer(text_token_ids)
                embeddings['text'].append(txt_emb.mean(dim=0).cpu().float().numpy())
            else:
                txt_emb = input_embeddings[0, vis_end_pos+1:, :]
                embeddings['text'].append(txt_emb.mean(dim=0).cpu().float().numpy())

            del outputs, input_embeddings
            torch.cuda.empty_cache()

            # ============================================================
            # Step B: NLD Generate — 触发 Latent Reasoning，提取 thought hidden states
            # ============================================================
            # 使用 hook 拦截 NativeLatentThinker 的 forward 输出
            thought_outputs_collector = []

            def _hook_thinker_forward(module, args, kwargs, output):
                """Hook NativeLatentThinker.forward() 的输出，收集 thought_output"""
                if isinstance(output, dict) and 'thought_output' in output:
                    # thought_output: [B, num_steps, H]
                    thought_outputs_collector.append(
                        output['thought_output'].detach().clone()
                    )
                return output

            # 注册 hook
            hook_handle = model.latent_thinker.register_forward_hook(
                _hook_thinker_forward, with_kwargs=True
            )

            try:
                with torch.no_grad():
                    # 使用 NLD 的自定义 generate 方法
                    generated_ids = model.generate(
                        pixel_values=inputs.get('pixel_values', inputs.get('pixel_values_videos', None)),
                        image_grid_thw=inputs.get('image_grid_thw', None),
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        enable_fallback=False,  # 禁用 fallback，确保 latent 触发
                    )
            finally:
                hook_handle.remove()

            # 收集 thought hidden states
            if thought_outputs_collector:
                # 合并所有 latent 触发事件的 thought_output
                # 每个 thought_output: [B=1, num_steps, H]
                all_thoughts = torch.cat(thought_outputs_collector, dim=1)  # [1, total_steps, H]
                # 取均值作为该样本的 H
                embeddings['generated'].append(
                    all_thoughts[0].mean(dim=0).cpu().float().numpy()
                )
                n_success += 1
            else:
                # 未触发 latent reasoning (模型没有生成 <|latent|> token)
                n_no_latent += 1
                n_fail += 1
                if n_no_latent <= 5:
                    print(f"    ⚠️ 样本 {idx}: 未触发 Latent Reasoning (无 <|latent|> 生成)")
                continue

            del thought_outputs_collector
            torch.cuda.empty_cache()

        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"    ⚠️ 样本 {idx} 失败: {type(e).__name__}: {e}")
            elif n_fail == 6:
                print(f"    ⚠️ 后续失败不再逐条打印...")

        if (idx + 1) % 20 == 0:
            print(f"    [{idx+1}/{len(queries)}] success={n_success}, fail={n_fail}, no_latent={n_no_latent}")

    print(f"\n  ✅ [NLD] 提取完成: success={n_success}, fail={n_fail}, no_latent={n_no_latent}")
    if n_no_latent > 0:
        print(f"  ⚠️ {n_no_latent} 个样本未触发 Latent Reasoning")

    result = {}
    for k, v_list in embeddings.items():
        if v_list:
            result[k] = np.stack(v_list, axis=0)
            print(f"    {k}: shape={result[k].shape}")
    return result


# ============================================================
# 4. 几何度量计算
# ============================================================
def compute_geometry_metrics(embeddings: Dict[str, np.ndarray]) -> Dict:
    """计算模态间的几何度量"""
    results = {}
    mod_names = sorted(embeddings.keys())

    # 4.1 模态中心间的余弦相似度和角度
    for i, name1 in enumerate(mod_names):
        for j, name2 in enumerate(mod_names):
            if i >= j:
                continue
            center1 = embeddings[name1].mean(axis=0)
            center2 = embeddings[name2].mean(axis=0)
            c1_norm = center1 / (np.linalg.norm(center1) + 1e-8)
            c2_norm = center2 / (np.linalg.norm(center2) + 1e-8)
            cos_center = float(np.dot(c1_norm, c2_norm))
            angle_center = float(np.degrees(np.arccos(np.clip(cos_center, -1, 1))))
            results[f'center_cos_{name1}_{name2}'] = cos_center
            results[f'center_angle_{name1}_{name2}'] = angle_center

    # 4.2 逐样本余弦相似度统计
    for i, name1 in enumerate(mod_names):
        for j, name2 in enumerate(mod_names):
            if i >= j:
                continue
            E1 = embeddings[name1]
            E2 = embeddings[name2]
            n = min(len(E1), len(E2))
            cos_sims = []
            for k in range(n):
                e1 = E1[k] / (np.linalg.norm(E1[k]) + 1e-8)
                e2 = E2[k] / (np.linalg.norm(E2[k]) + 1e-8)
                cos_sims.append(float(np.dot(e1, e2)))
            results[f'pairwise_cos_{name1}_{name2}_mean'] = float(np.mean(cos_sims))
            results[f'pairwise_cos_{name1}_{name2}_std'] = float(np.std(cos_sims))
            results[f'pairwise_cos_{name1}_{name2}_p10'] = float(np.percentile(cos_sims, 10))
            results[f'pairwise_cos_{name1}_{name2}_p90'] = float(np.percentile(cos_sims, 90))

    # 4.3 模态内聚性 (intra-modality cosine similarity)
    for name in mod_names:
        E = embeddings[name]
        if len(E) < 2:
            continue
        # 随机采样 pairs 计算
        n = len(E)
        n_pairs = min(1000, n * (n - 1) // 2)
        intra_cos = []
        rng = np.random.default_rng(42)
        for _ in range(n_pairs):
            i_idx, j_idx = rng.choice(n, size=2, replace=False)
            e_i = E[i_idx] / (np.linalg.norm(E[i_idx]) + 1e-8)
            e_j = E[j_idx] / (np.linalg.norm(E[j_idx]) + 1e-8)
            intra_cos.append(float(np.dot(e_i, e_j)))
        results[f'intra_cos_{name}_mean'] = float(np.mean(intra_cos))
        results[f'intra_cos_{name}_std'] = float(np.std(intra_cos))

    # 4.4 锥体半角 (cone half-angle)
    for name in mod_names:
        E = embeddings[name]
        center = E.mean(axis=0)
        center_norm = center / (np.linalg.norm(center) + 1e-8)
        cos_to_center = []
        for k in range(len(E)):
            e_norm = E[k] / (np.linalg.norm(E[k]) + 1e-8)
            cos_to_center.append(float(np.dot(e_norm, center_norm)))
        mean_cos = float(np.mean(cos_to_center))
        results[f'cone_{name}_half_angle'] = float(np.degrees(np.arccos(np.clip(mean_cos, -1, 1))))
        results[f'cone_{name}_cos_mean'] = mean_cos

    # 4.5 有效维度 (participation ratio)
    for name in mod_names:
        E = embeddings[name]
        if len(E) < 5:
            continue
        # PCA 特征值
        E_centered = E - E.mean(axis=0)
        cov = np.cov(E_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        eigenvalues = eigenvalues / eigenvalues.sum()
        # Participation ratio
        pr = 1.0 / (np.sum(eigenvalues ** 2) + 1e-8)
        results[f'effective_dim_{name}'] = float(pr)

    # 4.6 Norm 统计
    for name in mod_names:
        E = embeddings[name]
        norms = np.linalg.norm(E, axis=1)
        results[f'norm_{name}_mean'] = float(np.mean(norms))
        results[f'norm_{name}_std'] = float(np.std(norms))

    return results


# ============================================================
# 5. CKA 计算
# ============================================================
def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """计算线性 CKA (Centered Kernel Alignment)"""
    n = min(len(X), len(Y))
    X = X[:n]
    Y = Y[:n]
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # HSIC
    XtX = X @ X.T
    YtY = Y @ Y.T
    hsic_xy = np.trace(XtX @ YtY) / ((n - 1) ** 2)
    hsic_xx = np.trace(XtX @ XtX) / ((n - 1) ** 2)
    hsic_yy = np.trace(YtY @ YtY) / ((n - 1) ** 2)
    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-8)
    return float(cka)


def compute_cross_model_cka(emb_a: Dict[str, np.ndarray],
                            emb_b: Dict[str, np.ndarray],
                            label_a: str, label_b: str) -> Dict:
    """计算两个模型之间各模态的 CKA"""
    results = {}
    for mod_a in emb_a:
        for mod_b in emb_b:
            n = min(len(emb_a[mod_a]), len(emb_b[mod_b]))
            if n < 5:
                continue
            cka_val = linear_CKA(emb_a[mod_a][:n], emb_b[mod_b][:n])
            results[f'CKA_{label_a}_{mod_a}_vs_{label_b}_{mod_b}'] = cka_val
    return results


# ============================================================
# 6. 可视化
# ============================================================
def visualize_manifold(emb_baseline: Dict[str, np.ndarray],
                       emb_ckpt: Dict[str, np.ndarray],
                       metrics_baseline: Dict,
                       metrics_ckpt: Dict,
                       cka_results: Dict,
                       output_dir: str):
    """生成论文级别的模态流形可视化"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from sklearn.manifold import TSNE

    BG = '#0D1117'
    FG = '#C9D1D9'
    COLORS = {
        'vision': '#FF6B6B',
        'text': '#4ECDC4',
        'generated': '#A78BFA',  # H = Latent Hidden
    }

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Fig 1: 训练前后模态中心角度对比 (核心图)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

    for ax_idx, (metrics, label) in enumerate([
        (metrics_baseline, 'Baseline'),
        (metrics_ckpt, 'ckpt-1200 (Post-training)')
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor(BG)

        # 提取关键角度
        angles = {}
        for k, v in metrics.items():
            if k.startswith('center_angle_'):
                pair = k.replace('center_angle_', '')
                angles[pair] = v

        # 画极坐标图
        mod_names = ['vision', 'text', 'generated']
        mod_labels = ['V (Vision)', 'T (Text)', 'H (Latent)']
        mod_colors = [COLORS.get(m, '#888') for m in mod_names]

        # 用角度信息画三角形
        if 'generated_vision' in angles and 'generated_text' in angles and 'text_vision' in angles:
            angle_hv = angles.get('generated_vision', 90)
            angle_ht = angles.get('generated_text', 90)
            angle_vt = angles.get('text_vision', 90)

            # 简单布局：V 在左，T 在右，H 在上
            positions = {
                'vision': (-0.8, -0.5),
                'text': (0.8, -0.5),
                'generated': (0, 0.7),
            }

            for m, pos in positions.items():
                color = COLORS.get(m, '#888')
                ax.scatter(*pos, s=300, c=color, zorder=5, edgecolors='white', linewidths=2)
                label_text = {'vision': 'V', 'text': 'T', 'generated': 'H(L)'}[m]
                ax.annotate(label_text, pos, fontsize=16, fontweight='bold',
                           color=color, ha='center', va='bottom',
                           xytext=(0, 15), textcoords='offset points')

            # 画连线并标注角度
            pairs = [
                ('generated', 'vision', angle_hv),
                ('generated', 'text', angle_ht),
                ('text', 'vision', angle_vt),
            ]
            for m1, m2, angle in pairs:
                p1 = positions[m1]
                p2 = positions[m2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                       color=FG, alpha=0.6, linewidth=2, linestyle='--')
                mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                ax.annotate(f'{angle:.1f}°', mid, fontsize=12,
                           color='#FFD700', ha='center', va='center',
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor=BG, edgecolor='#FFD700', alpha=0.8))

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(label, color=FG, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle('Modality Gap Angles: Baseline vs Trained',
                color=FG, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_modality_angles_comparison.png'),
               facecolor=BG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig1_modality_angles_comparison.png")

    # ============================================================
    # Fig 2: 逐样本余弦相似度分布 (violin plot)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)

    pairs = [
        ('generated', 'vision', 'cos(H, V)'),
        ('generated', 'text', 'cos(H, T)'),
        ('text', 'vision', 'cos(T, V)'),
    ]

    for ax_idx, (m1, m2, title) in enumerate(pairs):
        ax = axes[ax_idx]
        ax.set_facecolor(BG)

        # 计算逐样本余弦
        data_list = []
        labels = []
        for emb, lbl in [(emb_baseline, 'Baseline'), (emb_ckpt, 'Trained')]:
            if m1 in emb and m2 in emb:
                E1 = emb[m1]
                E2 = emb[m2]
                n = min(len(E1), len(E2))
                cos_vals = []
                for k in range(n):
                    e1 = E1[k] / (np.linalg.norm(E1[k]) + 1e-8)
                    e2 = E2[k] / (np.linalg.norm(E2[k]) + 1e-8)
                    cos_vals.append(float(np.dot(e1, e2)))
                data_list.append(cos_vals)
                labels.append(lbl)

        if data_list:
            parts = ax.violinplot(data_list, positions=range(len(data_list)),
                                 showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('#A78BFA')
                pc.set_alpha(0.6)
            for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
                if key in parts:
                    parts[key].set_color(FG)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, color=FG, fontsize=11)
            ax.set_ylabel('Cosine Similarity', color=FG)
            ax.set_title(title, color=FG, fontsize=13, fontweight='bold')
            ax.tick_params(colors=FG)
            ax.grid(True, alpha=0.2, color='#21262D')
            ax.axhline(y=0, color='#FF6B6B', linestyle=':', alpha=0.5)

            # 标注均值
            for i, vals in enumerate(data_list):
                mean_val = np.mean(vals)
                ax.annotate(f'μ={mean_val:.3f}', (i, mean_val),
                           fontsize=10, color='#FFD700', ha='center',
                           xytext=(0, 10), textcoords='offset points')

    plt.suptitle('Per-Sample Cosine Similarity Distribution',
                color=FG, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_cosine_distribution.png'),
               facecolor=BG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig2_cosine_distribution.png")

    # ============================================================
    # Fig 3: t-SNE 联合可视化
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

    for ax_idx, (emb, label) in enumerate([
        (emb_baseline, 'Baseline'),
        (emb_ckpt, 'Trained (ckpt-1200)')
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor(BG)

        # 合并所有模态做联合 t-SNE
        all_data = []
        all_labels = []
        all_colors = []
        for mod_name in ['vision', 'text', 'generated']:
            if mod_name in emb:
                E = emb[mod_name]
                all_data.append(E)
                all_labels.extend([mod_name] * len(E))
                all_colors.extend([COLORS[mod_name]] * len(E))

        if all_data:
            combined = np.vstack(all_data)
            perp = min(30, len(combined) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
            emb_2d = tsne.fit_transform(combined)

            for mod_name in ['vision', 'text', 'generated']:
                mask = [l == mod_name for l in all_labels]
                if any(mask):
                    pts = emb_2d[mask]
                    display_name = {'vision': 'V (Vision)', 'text': 'T (Text)',
                                   'generated': 'H (Latent)'}[mod_name]
                    ax.scatter(pts[:, 0], pts[:, 1],
                             c=COLORS[mod_name], alpha=0.6, s=30,
                             label=display_name, edgecolors='white', linewidths=0.3)

        ax.set_title(label, color=FG, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, facecolor=BG, edgecolor='#30363D', labelcolor=FG)
        ax.grid(True, alpha=0.15, color='#21262D')
        ax.tick_params(colors=FG)

    plt.suptitle('t-SNE: Modality Token Manifold',
                color=FG, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_tsne_manifold.png'),
               facecolor=BG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig3_tsne_manifold.png")

    # ============================================================
    # Fig 4: CKA 热力图
    # ============================================================
    if cka_results:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
        ax.set_facecolor(BG)

        # 构建 CKA 矩阵
        all_keys = sorted(set(
            k.split('_vs_')[0].replace('CKA_', '') for k in cka_results.keys()
        ) | set(
            k.split('_vs_')[1] for k in cka_results.keys()
        ))

        n = len(all_keys)
        cka_matrix = np.zeros((n, n))
        for i, k1 in enumerate(all_keys):
            for j, k2 in enumerate(all_keys):
                if i == j:
                    cka_matrix[i, j] = 1.0
                else:
                    key = f'CKA_{k1}_vs_{k2}'
                    if key in cka_results:
                        cka_matrix[i, j] = cka_results[key]
                    else:
                        key2 = f'CKA_{k2}_vs_{k1}'
                        if key2 in cka_results:
                            cka_matrix[i, j] = cka_results[key2]

        im = ax.imshow(cka_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_labels = [k.replace('baseline_', 'B/').replace('ckpt_', 'T/') for k in all_keys]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9, color=FG)
        ax.set_yticklabels(short_labels, fontsize=9, color=FG)
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(n):
            for j in range(n):
                if cka_matrix[i, j] > 0:
                    ax.text(j, i, f'{cka_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8,
                           color='black' if cka_matrix[i, j] > 0.5 else FG)

        ax.set_title('Cross-Model CKA', color=FG, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig4_cka_heatmap.png'),
                   facecolor=BG, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ fig4_cka_heatmap.png")

    # ============================================================
    # Fig 5: 综合对比表格图
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    # 构建对比数据
    table_data = [
        ['Metric', 'Baseline', 'Trained (ckpt-1200)', 'Δ Change'],
    ]

    key_metrics = [
        ('center_cos_generated_vision', 'cos(H, V) center'),
        ('center_cos_generated_text', 'cos(H, T) center'),
        ('center_cos_text_vision', 'cos(T, V) center'),
        ('center_angle_generated_vision', '∠(H, V)'),
        ('center_angle_generated_text', '∠(H, T)'),
        ('pairwise_cos_generated_vision_mean', 'cos(H, V) pairwise mean'),
        ('pairwise_cos_generated_text_mean', 'cos(H, T) pairwise mean'),
        ('intra_cos_generated_mean', 'H intra-cos'),
        ('cone_generated_half_angle', 'H cone half-angle'),
    ]

    for key, display_name in key_metrics:
        val_b = metrics_baseline.get(key, None)
        val_t = metrics_ckpt.get(key, None)
        if val_b is not None and val_t is not None:
            delta = val_t - val_b
            sign = '+' if delta > 0 else ''
            table_data.append([
                display_name,
                f'{val_b:.4f}',
                f'{val_t:.4f}',
                f'{sign}{delta:.4f}'
            ])

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置表格样式
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#30363D')
        if i == 0:
            cell.set_facecolor('#21262D')
            cell.set_text_props(color=FG, fontweight='bold')
        else:
            cell.set_facecolor(BG)
            cell.set_text_props(color=FG)

    ax.set_title('Modality Geometry Metrics: Baseline vs Trained',
                color=FG, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_metrics_table.png'),
               facecolor=BG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig5_metrics_table.png")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='模态流形分析 (带 generation): 提取 V/T/H embeddings 并分析训练前后变化'
    )
    parser.add_argument('--ckpt_path', required=True,
                       help='训练后 checkpoint 路径')
    parser.add_argument('--baseline_path', required=True,
                       help='Baseline 预训练模型路径')
    parser.add_argument('--base_model_path', default=None,
                       help='基座模型路径 (FSDP checkpoint 需要)')
    parser.add_argument('--data_path', required=True,
                       help='训练数据 JSON 路径 (erqa_vsibench_merged)')
    parser.add_argument('--image_root', default=None,
                       help='图片根目录 (默认为项目根目录)')
    parser.add_argument('--output_dir', default=None,
                       help='输出目录')
    parser.add_argument('--n_samples', type=int, default=200,
                       help='样本数量 (建议 200+)')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='每个样本生成的最大 token 数 (NLD 需要足够长度触发 latent)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJ_ROOT, 'modality_manifold_analysis', 'results_v2')
    if args.base_model_path is None:
        args.base_model_path = args.baseline_path
    if args.image_root is None:
        args.image_root = PROJ_ROOT

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("  Modality Manifold Analysis (Latent Reasoning Version)")
    print("=" * 70)
    print(f"  Checkpoint:    {args.ckpt_path}")
    print(f"  Baseline:      {args.baseline_path}")
    print(f"  Data:          {args.data_path}")
    print(f"  n_samples:     {args.n_samples}")
    print(f"  max_new_tokens:{args.max_new_tokens}")
    print(f"  Output:        {args.output_dir}")
    print("=" * 70)

    # 1. 加载数据
    print("\n📂 Step 1: 加载 query 数据...")
    queries = load_queries(args.data_path, args.n_samples, args.image_root)
    if not queries:
        print("❌ 无有效样本, 退出")
        return

    # 2. 提取 Baseline embeddings (原生 Qwen3-VL，无 Latent 能力)
    print("\n" + "=" * 70)
    print("  📦 Step 2: 提取 Baseline embeddings (普通 generate)")
    print("=" * 70)
    model, tokenizer, processor = load_baseline_model(args.baseline_path, 'cuda')
    emb_baseline = extract_baseline_embeddings(
        model, tokenizer, processor, queries,
        max_new_tokens=args.max_new_tokens, device='cuda'
    )
    # 保存
    torch.save(emb_baseline, os.path.join(args.output_dir, 'embeddings_baseline.pt'))

    # 释放模型
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # 3. 提取 NLD ckpt embeddings (含 NativeLatentThinker，触发 Latent Reasoning)
    print("\n" + "=" * 70)
    print("  📦 Step 3: 提取 NLD ckpt embeddings (Latent Reasoning)")
    print("=" * 70)
    model, tokenizer, processor = load_nld_model(
        args.base_model_path, args.ckpt_path, 'cuda'
    )
    emb_ckpt = extract_nld_embeddings(
        model, tokenizer, processor, queries,
        max_new_tokens=args.max_new_tokens, device='cuda'
    )
    # 保存
    torch.save(emb_ckpt, os.path.join(args.output_dir, 'embeddings_ckpt.pt'))

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # 4. 计算几何度量
    print("\n" + "=" * 70)
    print("  📐 Step 4: 计算几何度量")
    print("=" * 70)

    metrics_baseline = compute_geometry_metrics(emb_baseline)
    metrics_ckpt = compute_geometry_metrics(emb_ckpt)

    with open(os.path.join(args.output_dir, 'metrics_baseline.json'), 'w') as f:
        json.dump(metrics_baseline, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'metrics_ckpt.json'), 'w') as f:
        json.dump(metrics_ckpt, f, indent=2, ensure_ascii=False)

    print("\n  Baseline 关键指标:")
    for k, v in sorted(metrics_baseline.items()):
        if 'generated' in k:
            print(f"    {k}: {v:.4f}")

    print("\n  ckpt-1200 关键指标:")
    for k, v in sorted(metrics_ckpt.items()):
        if 'generated' in k:
            print(f"    {k}: {v:.4f}")

    # 5. CKA
    print("\n" + "=" * 70)
    print("  📊 Step 5: 跨模型 CKA")
    print("=" * 70)
    cka_results = compute_cross_model_cka(emb_baseline, emb_ckpt, 'baseline', 'ckpt')
    with open(os.path.join(args.output_dir, 'cka_cross_model.json'), 'w') as f:
        json.dump(cka_results, f, indent=2, ensure_ascii=False)
    for k, v in sorted(cka_results.items()):
        print(f"    {k}: {v:.4f}")

    # 6. 可视化
    print("\n" + "=" * 70)
    print("  🎨 Step 6: 可视化")
    print("=" * 70)
    visualize_manifold(emb_baseline, emb_ckpt, metrics_baseline, metrics_ckpt,
                      cka_results, figures_dir)

    # 7. 输出总结
    print("\n" + "=" * 70)
    print("  📋 总结")
    print("=" * 70)
    print(f"\n  核心结论:")
    cos_hv_base = metrics_baseline.get('center_cos_generated_vision', None)
    cos_hv_ckpt = metrics_ckpt.get('center_cos_generated_vision', None)
    cos_ht_base = metrics_baseline.get('center_cos_generated_text', None)
    cos_ht_ckpt = metrics_ckpt.get('center_cos_generated_text', None)

    if cos_hv_base is not None and cos_hv_ckpt is not None:
        print(f"    cos(H, V): {cos_hv_base:.4f} → {cos_hv_ckpt:.4f} (Δ={cos_hv_ckpt-cos_hv_base:+.4f})")
    if cos_ht_base is not None and cos_ht_ckpt is not None:
        print(f"    cos(H, T): {cos_ht_base:.4f} → {cos_ht_ckpt:.4f} (Δ={cos_ht_ckpt-cos_ht_base:+.4f})")

    angle_hv_base = metrics_baseline.get('center_angle_generated_vision', None)
    angle_hv_ckpt = metrics_ckpt.get('center_angle_generated_vision', None)
    if angle_hv_base is not None and angle_hv_ckpt is not None:
        print(f"    ∠(H, V):   {angle_hv_base:.1f}° → {angle_hv_ckpt:.1f}° (Δ={angle_hv_ckpt-angle_hv_base:+.1f}°)")

    print(f"\n  ✅ 所有结果已保存到: {args.output_dir}")
    print(f"  ✅ 可视化已保存到: {figures_dir}")


if __name__ == '__main__':
    main()
