#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_dual_model_embeddings.py
==================================
分别对 ckpt-1200 和 baseline 的视觉 token、文本 token 和 hidden states
进行 embedding 提取和流形分析。

需要在有 GPU 的 Docker 容器中运行。

产出:
  - embeddings_ckpt1200.pt  (包含 vision / language / hidden_last / hidden_intermediate)
  - embeddings_baseline.pt  (同上)
  - geometry_metrics_ckpt1200.json
  - geometry_metrics_baseline.json
  - cka_between_models.json  (两个模型之间的 CKA)

用法 (Docker GPU):
  python3 extract_dual_model_embeddings.py \
      --ckpt1200_path /path/to/checkpoint-1200 \
      --baseline_path <PATH_TO_QWEN3_VL_8B_INSTRUCT> \
      --output_dir ./modality_manifold_analysis/results \
      --n_samples 100
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 将项目根目录加入 sys.path, 以便 import rld 模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. 模型加载与 embedding 提取
# ============================================================
def load_model_and_processor(model_path: str, device: str = 'cuda',
                             base_model_path: str = None):
    """加载 Qwen3-VL 模型和处理器
    
    支持:
      1. 预训练模型目录 (包含 config.json) — 直接用 AutoModelForCausalLM 加载
      2. 训练后 FSDP checkpoint (无 config.json, 有 model.safetensors) —
         从 base_model_path 加载 config/tokenizer/processor,
         再从 model.safetensors 中提取 base_model.* 前缀的权重加载到
         Qwen3VLForConditionalGeneration (支持 output_hidden_states)
    
    注意:
      - FSDP checkpoint 的 model.safetensors 是 NLDModel 的完整 state_dict,
        key 格式为 'base_model.*' / 'latent_thinker.*'
      - 对于流形分析, 我们只需要 base_model (Qwen3VL) 部分, 不需要 latent_thinker
      - 不使用 NLDModel 加载, 因为 NLDModel.forward() 是训练专用的 segment-wise
        流程, 不支持 output_hidden_states=True
    """
    from transformers import AutoTokenizer, AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration
    import safetensors.torch
    
    # 检测 model_path 下是否有 config.json
    has_config = os.path.isfile(os.path.join(model_path, 'config.json'))
    
    # 确定 tokenizer/processor 的加载路径
    if has_config:
        proc_path = model_path
    else:
        # 训练后 FSDP checkpoint (无 config.json), 用 base_model_path
        if base_model_path is None:
            raise ValueError(
                f"Checkpoint {model_path} 缺少 config.json, "
                "需要指定 --base_model_path 来加载 tokenizer/processor"
            )
        proc_path = base_model_path
        print(f"  ⚠️  FSDP checkpoint 检测: {model_path} 无 config.json")
        print(f"      使用 base_model_path 加载 config/tokenizer: {proc_path}")
    
    print(f"  Loading model from: {model_path}")
    print(f"  Loading processor from: {proc_path}")
    
    # 加载 tokenizer 和 processor
    tokenizer = AutoTokenizer.from_pretrained(
        proc_path, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        proc_path, trust_remote_code=True
    )
    
    if has_config:
        # ============================================================
        # 路径 A: 预训练模型目录 — 直接加载
        # ============================================================
        print(f"  [路径A] 预训练模型目录, 直接 Qwen3VLForConditionalGeneration 加载")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        )
    else:
        # ============================================================
        # 路径 B: FSDP checkpoint — 从 model.safetensors 提取 base_model 权重
        # ============================================================
        print(f"  [路径B] FSDP checkpoint, 从 model.safetensors 提取 base_model 权重")
        
        # Step 1: 用 base_model_path 初始化模型结构
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            proc_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map='cpu',  # 先加载到 CPU, 后面再移到 GPU
        )
        
        # Step 2: 从 FSDP checkpoint 加载权重
        safetensor_path = os.path.join(model_path, 'model.safetensors')
        sharded_index_path = os.path.join(model_path, 'model.safetensors.index.json')
        
        if os.path.isfile(sharded_index_path):
            # Sharded checkpoint
            with open(sharded_index_path, 'r') as f:
                idx = json.load(f)
            shard_names = sorted(set(idx.get('weight_map', {}).values()))
            shard_paths = [os.path.join(model_path, n) for n in shard_names]
            print(f"  发现 sharded checkpoint: {len(shard_paths)} 个 shard")
        elif os.path.isfile(safetensor_path):
            # 单文件 checkpoint
            shard_paths = [safetensor_path]
            sz_gb = os.path.getsize(safetensor_path) / (1024**3)
            print(f"  发现单文件 checkpoint: model.safetensors ({sz_gb:.2f} GB)")
        else:
            raise FileNotFoundError(
                f"FSDP checkpoint 中未找到 model.safetensors: {model_path}"
            )
        
        # Step 3: 加载并提取 base_model.* 前缀的权重
        base_state = {}
        thinker_count = 0
        other_count = 0
        
        for sp in shard_paths:
            print(f"    加载 shard: {os.path.basename(sp)}...")
            shard = safetensors.torch.load_file(sp, device='cpu')
            for k, v in shard.items():
                if k.startswith('base_model.'):
                    # 去掉 'base_model.' 前缀, 与 AutoModelForCausalLM 的 key 对齐
                    base_state[k[len('base_model.'):]] = v
                elif k.startswith('latent_thinker.'):
                    thinker_count += 1
                else:
                    other_count += 1
            del shard  # 释放内存
        
        print(f"  FSDP ckpt 拆分: base_model={len(base_state)} 个参数, "
              f"latent_thinker={thinker_count} 个 (跳过), other={other_count} 个 (跳过)")
        
        # Step 4: 形状适配 (处理 vocab size 差异)
        cur_state = model.state_dict()
        adapted_state = {}
        n_match = 0
        n_pad = 0
        n_skip = 0
        
        for k, v in base_state.items():
            if k not in cur_state:
                n_skip += 1
                continue
            cur_v = cur_state[k]
            if v.shape == cur_v.shape:
                adapted_state[k] = v
                n_match += 1
            elif (
                v.dim() == cur_v.dim()
                and v.shape[1:] == cur_v.shape[1:]
                and v.shape[0] < cur_v.shape[0]
            ):
                # vocab-like 维度: ckpt 比 current 小, 前 V_ckpt 行用 ckpt
                new_v = cur_v.clone()
                new_v[:v.shape[0]] = v.to(cur_v.dtype)
                adapted_state[k] = new_v
                n_pad += 1
                print(f"    形状适配: {k} {tuple(v.shape)} → {tuple(cur_v.shape)}")
            elif (
                v.dim() == cur_v.dim()
                and v.shape[1:] == cur_v.shape[1:]
                and v.shape[0] > cur_v.shape[0]
            ):
                # ckpt 比 current 大, 截断
                adapted_state[k] = v[:cur_v.shape[0]].to(cur_v.dtype)
                n_pad += 1
                print(f"    形状适配(截断): {k} {tuple(v.shape)} → {tuple(cur_v.shape)}")
            else:
                n_skip += 1
                print(f"    ⚠️ 形状不兼容, 跳过: {k} ckpt={tuple(v.shape)} cur={tuple(cur_v.shape)}")
        
        print(f"  权重加载统计: match={n_match}, pad={n_pad}, skip={n_skip}")
        
        # Step 5: 加载权重
        missing, unexpected = model.load_state_dict(adapted_state, strict=False)
        if missing:
            print(f"  ⚠️ 缺失参数: {len(missing)} 个 (使用预训练初始化)")
            if len(missing) <= 10:
                for m in missing:
                    print(f"      {m}")
        if unexpected:
            print(f"  ⚠️ 多余参数: {len(unexpected)} 个")
        
        del base_state, adapted_state  # 释放内存
        
        # Step 6: 移到 GPU
        if device != 'cpu':
            print(f"  移动模型到 {device}...")
            model = model.to(device)
    
    model.eval()
    print(f"  ✅ 模型加载完成, device={next(model.parameters()).device}")
    
    return model, tokenizer, processor


def extract_embeddings_from_model(
    model_path: str,
    output_dir: str,
    n_samples: int = 100,
    model_label: str = '',
    base_model_path: str = None,
):
    """
    从单个模型提取不同模态的 embeddings。
    
    分别提取:
      1. 视觉 token embeddings (vision) — 图文输入中视觉 token 位置的 hidden states
      2. 文本 token embeddings (language) — 纯文本输入的 hidden states
      3. 最后层 hidden states (hidden_last) — 图文输入最后层全序列均值
      4. 中间层 hidden states (hidden_intermediate) — 1/3 和 2/3 层位置
    
    Args:
        model_path: 模型路径 (预训练目录或训练后 FSDP checkpoint 目录)
        output_dir: 输出目录
        n_samples: 提取 embedding 使用的样本数
        model_label: 模型标签 (用于文件命名)
        base_model_path: 基座模型路径 (当 model_path 是 FSDP checkpoint 时需要)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("  ⚠️  No GPU available, using CPU (will be slow)")
    
    model, tokenizer, processor = load_model_and_processor(
        model_path, device, base_model_path=base_model_path
    )
    
    embeddings = {
        'vision': [],
        'language': [],
        'hidden_last': [],
        'hidden_intermediate': [],
    }
    
    # ============================================================
    # 纯文本输入 — 提取 language embeddings + hidden states
    # ============================================================
    texts = [
        "What is the capital of France?",
        "Describe a beautiful sunset over the mountains.",
        "How does photosynthesis work in plants?",
        "Explain the theory of relativity in simple terms.",
        "What are the physical properties of water?",
        "Write a poem about the deep ocean.",
        "What is machine learning and how does it differ from traditional programming?",
        "Describe a modern city skyline at night.",
        "How does the human heart pump blood through the body?",
        "Explain the basic principles of quantum computing.",
        "What causes earthquakes and how are they measured?",
        "Describe the process of making chocolate from cocoa beans.",
        "How do birds navigate during migration?",
        "What is the greenhouse effect and its impact on climate?",
        "Explain how a computer processor executes instructions.",
        "Describe the life cycle of a butterfly.",
        "What are black holes and how do they form?",
        "How does the immune system fight infections?",
        "Explain the concept of supply and demand in economics.",
        "What is the significance of the Rosetta Stone?",
    ][:n_samples]
    
    print(f"\n  📝 Extracting language & hidden embeddings from {len(texts)} text inputs...")
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            # 使用 chat template 格式化输入 (与训练一致)
            messages = [
                {"role": "user", "content": text},
            ]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text_input, return_tensors='pt').to(device)
            
            # 请求所有层的 hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # 语言 embedding (最后一层 hidden state 的均值)
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
            embeddings['language'].append(last_hidden.mean(dim=1).cpu().float())
            
            # Hidden states — 最后层
            embeddings['hidden_last'].append(last_hidden.mean(dim=1).cpu().float())
            
            # 中间层 (取 1/3 和 2/3 位置)
            n_layers = len(outputs.hidden_states)
            for frac in [0.33, 0.66]:
                layer_idx = int(n_layers * frac)
                hs = outputs.hidden_states[layer_idx]
                embeddings['hidden_intermediate'].append(hs.mean(dim=1).cpu().float())
            
            if (i + 1) % 5 == 0:
                print(f"    Processed {i+1}/{len(texts)} text inputs")
    
    # ============================================================
    # 图文输入 — 提取 vision embeddings + hidden states
    # ============================================================
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    
    # 使用真实训练数据中的图片 (如果存在)
    real_images = []
    erqa_dir = os.path.join(PROJ_ROOT, 'data', 'erqa', 'images')
    if os.path.isdir(erqa_dir):
        img_files = sorted([f for f in os.listdir(erqa_dir) if f.endswith(('.jpg', '.png'))])[:n_samples]
        for f in img_files:
            try:
                img = Image.open(os.path.join(erqa_dir, f)).convert('RGB')
                real_images.append(img)
            except Exception:
                pass
    
    if not real_images:
        # Fallback: 创建 dummy 图像
        print("  ⚠️  未找到真实图片, 使用 dummy 图像")
        real_images = [
            Image.new('RGB', (224, 224), color=(i * 40 % 256, i * 30 % 256, i * 20 % 256))
            for i in range(min(10, n_samples))
        ]
    else:
        real_images = real_images[:min(20, n_samples)]
    
    print(f"\n  🖼️  Extracting vision embeddings from {len(real_images)} image inputs...")
    
    with torch.no_grad():
        for i, img in enumerate(real_images):
            try:
                # Qwen3-VL 的标准图文输入方式
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image briefly."},
                    ]},
                ]
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # 处理视觉信息
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors='pt',
                    padding=True,
                ).to(device)
                
                outputs = model(**inputs, output_hidden_states=True)
                
                # 获取所有 hidden states
                all_hidden = outputs.hidden_states  # tuple of [1, seq_len, hidden_size]
                last_hidden = all_hidden[-1]
                seq_len = last_hidden.shape[1]
                
                # 视觉 token 位置检测:
                # Qwen3-VL 中视觉 token 通常在序列前部 (image token 区域)
                # 通过 input_ids 中的特殊 token 来定位
                input_ids = inputs.get('input_ids', None)
                if input_ids is not None:
                    # 找到 image token 的范围
                    # Qwen3-VL 使用 <|vision_start|> 和 <|vision_end|> 标记视觉区域
                    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
                    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
                    
                    ids = input_ids[0].tolist()
                    vis_start_pos = None
                    vis_end_pos = None
                    for pos, tid in enumerate(ids):
                        if tid == vision_start_id and vis_start_pos is None:
                            vis_start_pos = pos
                        if tid == vision_end_id:
                            vis_end_pos = pos
                            break
                    
                    if vis_start_pos is not None and vis_end_pos is not None:
                        # 视觉 token 区域的 hidden states
                        vis_hidden = last_hidden[0, vis_start_pos:vis_end_pos+1, :]  # [vis_len, hidden]
                        embeddings['vision'].append(vis_hidden.mean(dim=0, keepdim=True).cpu().float())
                        
                        # 文本 token 区域 (视觉区域之后)
                        txt_hidden = last_hidden[0, vis_end_pos+1:, :]  # [txt_len, hidden]
                        if txt_hidden.shape[0] > 0:
                            embeddings['hidden_last'].append(txt_hidden.mean(dim=0, keepdim=True).cpu().float())
                    else:
                        # 无法定位视觉区域, 用全序列均值
                        embeddings['vision'].append(last_hidden.mean(dim=1).cpu().float())
                else:
                    embeddings['vision'].append(last_hidden.mean(dim=1).cpu().float())
                
                # 中间层视觉 embedding (浅层保留更多视觉信息)
                for frac in [0.1, 0.25]:
                    layer_idx = int(len(all_hidden) * frac)
                    hs = all_hidden[layer_idx]
                    if vis_start_pos is not None and vis_end_pos is not None:
                        vis_hs = hs[0, vis_start_pos:vis_end_pos+1, :]
                        embeddings['hidden_intermediate'].append(
                            vis_hs.mean(dim=0, keepdim=True).cpu().float()
                        )
                    else:
                        embeddings['hidden_intermediate'].append(
                            hs.mean(dim=1).cpu().float()
                        )
                
            except Exception as e:
                print(f"    ⚠️  Failed on image {i}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            
            if (i + 1) % 5 == 0:
                print(f"    Processed {i+1}/{len(real_images)} image inputs")
    
    # ============================================================
    # 合并和保存
    # ============================================================
    # 将列表转为 numpy 数组
    for k in list(embeddings.keys()):
        if embeddings[k]:
            embeddings[k] = torch.cat(embeddings[k], dim=0).numpy()
        else:
            del embeddings[k]
    
    # 保存 embeddings
    model_name = model_label or Path(model_path).name
    out_path = os.path.join(output_dir, f'embeddings_{model_name}.pt')
    torch.save(embeddings, out_path)
    print(f"\n  💾 Saved embeddings: {out_path}")
    for k, v in embeddings.items():
        print(f"    {k}: shape={v.shape}")
    
    return embeddings


# ============================================================
# 2. 几何度量计算
# ============================================================
def compute_geometry_metrics(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    从 embeddings 字典计算模态几何度量
    """
    results = {}
    
    # 余弦相似度 (两两模态之间)
    for name1 in embeddings:
        for name2 in embeddings:
            if name1 >= name2:
                continue
            E1 = embeddings[name1]
            E2 = embeddings[name2]
            cos_sims = []
            for i in range(min(len(E1), len(E2))):
                e1 = E1[i] / (np.linalg.norm(E1[i]) + 1e-8)
                e2 = E2[i] / (np.linalg.norm(E2[i]) + 1e-8)
                cos_sims.append(np.dot(e1, e2))
            results[f'cos_{name1}_{name2}_mean'] = float(np.mean(cos_sims))
            results[f'cos_{name1}_{name2}_std'] = float(np.std(cos_sims))
    
    # 模态间隙角度
    for name1 in embeddings:
        for name2 in embeddings:
            if name1 >= name2:
                continue
            E1_mean = embeddings[name1].mean(axis=0)
            E2_mean = embeddings[name2].mean(axis=0)
            E1_mean_norm = E1_mean / (np.linalg.norm(E1_mean) + 1e-8)
            E2_mean_norm = E2_mean / (np.linalg.norm(E2_mean) + 1e-8)
            cos_mean = np.dot(E1_mean_norm, E2_mean_norm)
            results[f'modality_gap_angle_{name1}_{name2}'] = float(
                np.degrees(np.arccos(np.clip(cos_mean, -1, 1)))
            )
    
    # 锥体半角 (每个模态内部)
    for name, E in embeddings.items():
        E_mean = E.mean(axis=0)
        E_mean_norm = E_mean / (np.linalg.norm(E_mean) + 1e-8)
        E_norms = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        cos_to_mean = (E_norms @ E_mean_norm).mean()
        half_angle = np.degrees(np.arccos(np.clip(cos_to_mean, -1, 1)))
        results[f'cone_{name}_half_angle'] = float(half_angle)
        results[f'cone_{name}_cos_mean'] = float(cos_to_mean)
    
    # 内部相似度
    for name, E in embeddings.items():
        n = min(len(E), 100)
        idx = np.random.choice(len(E), n, replace=False)
        sub_E = E[idx]
        norms = np.linalg.norm(sub_E, axis=1, keepdims=True)
        sub_E_normed = sub_E / (norms + 1e-8)
        sim_matrix = sub_E_normed @ sub_E_normed.T
        np.fill_diagonal(sim_matrix, 0)
        results[f'intra_{name}_cos_mean'] = float(sim_matrix.sum() / (n * (n - 1)))
    
    # 表示能力指标
    for name, E in embeddings.items():
        results[f'{name}_norm_mean'] = float(np.linalg.norm(E, axis=1).mean())
        results[f'{name}_dim_eff'] = float(
            np.linalg.matrix_rank(E[:, :min(E.shape[1], 500)])
        )
    
    return results


# ============================================================
# 3. 两模型间 CKA 计算
# ============================================================
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA"""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    kx = X @ X.T
    ky = Y @ Y.T
    
    n = X.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    hsic_xy = np.trace(kx @ h @ ky @ h) / (n - 1) ** 2
    hsic_xx = np.trace(kx @ h @ kx @ h) / (n - 1) ** 2
    hsic_yy = np.trace(ky @ h @ ky @ h) / (n - 1) ** 2
    
    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def compute_cka_between_models(emb1: Dict, emb2: Dict, label1: str = 'A', label2: str = 'B') -> Dict:
    """计算两个模型各模态之间的 CKA"""
    results = {}
    
    common_keys = sorted(set(emb1.keys()) & set(emb2.keys()))
    
    for k1 in common_keys:
        for k2 in common_keys:
            if k1 >= k2:
                continue
            E1 = emb1[k1]
            E2 = emb2[k2]
            n = min(len(E1), len(E2), 200)
            idx = np.random.choice(min(len(E1), len(E2)), n, replace=False)
            cka_val = linear_cka(E1[idx], E2[idx])
            results[f'cka_{label1}_{k1}__{label2}_{k2}'] = float(cka_val)
    
    # 同模态 CKA
    for k in common_keys:
        E1 = emb1[k]
        E2 = emb2[k]
        n = min(len(E1), len(E2), 200)
        idx = np.random.choice(min(len(E1), len(E2)), n, replace=False)
        cka_val = linear_cka(E1[idx], E2[idx])
        results[f'cka_same_modality_{k}'] = float(cka_val)
    
    return results


# ============================================================
# 4. Token 语义流形可视化分析
# ============================================================
def visualize_token_manifold(embeddings: Dict, label: str, output_dir: str):
    """
    Token 语义流形可视化分析
    
    生成:
      - 2D t-SNE / UMAP 投影
      - 模态锥体 3D 图
      - 角度关系图
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 配色
    BG = '#0D1117'
    FG = '#C9D1D9'
    C_VISION = '#00D2FF'
    C_LANG = '#FF6B6B'
    C_HIDDEN = '#A78BFA'
    
    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans', 'SimHei'],
        'axes.unicode_minus': False,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })
    
    modality_colors = {
        'vision': C_VISION,
        'language': C_LANG,
        'hidden_last': C_HIDDEN,
        'hidden_intermediate': '#4ADE80',
    }
    
    # ============================================================
    # 图 1: t-SNE 2D 投影
    # ============================================================
    try:
        from sklearn.manifold import TSNE
        
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
        ax.set_facecolor(BG)
        
        for mod_name, emb in embeddings.items():
            if len(emb) < 2:
                continue
            # t-SNE 降维
            n_components = 2 if len(emb) >= 3 else 1
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(emb)-1))
            emb_2d = tsne.fit_transform(emb)
            
            color = modality_colors.get(mod_name, '#888888')
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                      c=color, alpha=0.6, s=20, label=f'{mod_name} (n={len(emb)})')
        
        ax.set_title(f'Token Semantic Manifold — {label}', color=FG, fontweight='bold', fontsize=14)
        ax.set_xlabel('t-SNE Dim 1', color=FG)
        ax.set_ylabel('t-SNE Dim 2', color=FG)
        ax.legend(fontsize=9, facecolor=BG, edgecolor='#30363D', labelcolor=FG)
        ax.grid(True, alpha=0.2, color='#21262D')
        ax.tick_params(colors=FG)
        
        out_path = os.path.join(output_dir, f'tsne_{label}.png')
        plt.savefig(out_path, facecolor=BG)
        plt.close()
        print(f"  ✅ Saved t-SNE: {out_path}")
    except ImportError:
        print("  ⚠️  sklearn not available, skipping t-SNE")
    except Exception as e:
        print(f"  ⚠️  t-SNE failed: {e}")
    
    # ============================================================
    # 图 2: 模态锥体 3D 图 (PCA)
    # ============================================================
    try:
        from sklearn.decomposition import PCA
        
        fig = plt.figure(figsize=(14, 10), facecolor=BG)
        ax = fig.add_subplot(111, projection='3d', facecolor=BG)
        
        all_embs = []
        all_labels = []
        all_colors = []
        
        for mod_name, emb in embeddings.items():
            if len(emb) < 3:
                continue
            pca = PCA(n_components=3, random_state=42)
            emb_3d = pca.fit_transform(emb)
            all_embs.append(emb_3d)
            all_labels.append(mod_name)
            all_colors.append(modality_colors.get(mod_name, '#888888'))
        
        if all_embs:
            # 合并所有数据做全局 PCA
            combined = np.vstack(all_embs)
            global_pca = PCA(n_components=3, random_state=42)
            combined_3d = global_pca.fit_transform(combined)
            
            # 拆分回来
            start = 0
            for emb_3d, mod_name, color in zip(all_embs, all_labels, all_colors):
                n = len(emb_3d)
                sub = combined_3d[start:start+n]
                ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
                          c=color, alpha=0.5, s=15, label=mod_name)
                start += n
            
            # 画模态中心
            centers = []
            for emb_3d, mod_name, color in zip(all_embs, all_labels, all_colors):
                center = emb_3d.mean(axis=0)
                centers.append(center)
            
            # 从原点到各模态中心画向量
            for center, mod_name, color in zip(centers, all_labels, all_colors):
                ax.quiver(0, 0, 0, center[0], center[1], center[2],
                         color=color, arrow_length_ratio=0.15, lw=2.5, label=f'{mod_name} center')
            
            # 画模态间的角度
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    c1 = centers[i] / (np.linalg.norm(centers[i]) + 1e-8)
                    c2 = centers[j] / (np.linalg.norm(centers[j]) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(np.dot(c1, c2), -1, 1)))
                    mid = (centers[i] + centers[j]) / 2
                    ax.text(mid[0], mid[1], mid[2], f'{angle:.1f}°',
                           color=FG, fontsize=9, fontweight='bold')
            
            ax.set_title(f'Modality Cone 3D — {label}', color=FG, fontweight='bold', fontsize=14)
            ax.legend(fontsize=8, facecolor=BG, edgecolor='#30363D', labelcolor=FG)
            ax.tick_params(colors=FG, labelsize=8)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            out_path = os.path.join(output_dir, f'cone3d_{label}.png')
            plt.savefig(out_path, facecolor=BG)
            plt.close()
            print(f"  ✅ Saved 3D cone: {out_path}")
    except ImportError:
        print("  ⚠️  sklearn not available, skipping 3D PCA")
    except Exception as e:
        print(f"  ⚠️  3D PCA failed: {e}")
    
    # ============================================================
    # 图 3: 角度关系热力图
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    
    mod_names = sorted(embeddings.keys())
    n = len(mod_names)
    
    # 余弦相似度矩阵
    cos_matrix = np.zeros((n, n))
    for i, name1 in enumerate(mod_names):
        for j, name2 in enumerate(mod_names):
            if i == j:
                cos_matrix[i, j] = 1.0
            elif i < j:
                E1 = embeddings[name1]
                E2 = embeddings[name2]
                cos_sims = []
                for k in range(min(len(E1), len(E2))):
                    e1 = E1[k] / (np.linalg.norm(E1[k]) + 1e-8)
                    e2 = E2[k] / (np.linalg.norm(E2[k]) + 1e-8)
                    cos_sims.append(np.dot(e1, e2))
                val = float(np.mean(cos_sims))
                cos_matrix[i, j] = val
                cos_matrix[j, i] = val
    
    # Panel A: 余弦相似度热力图
    ax = axes[0]
    ax.set_facecolor(BG)
    im = ax.imshow(cos_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mod_names, rotation=45, ha='right', fontsize=9, color=FG)
    ax.set_yticklabels(mod_names, fontsize=9, color=FG)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'Cosine Similarity — {label}', color=FG, fontweight='bold')
    
    # 标注数值
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cos_matrix[i, j]:.3f}',
                   ha='center', va='center', fontsize=8, color=FG if cos_matrix[i, j] < 0.5 else 'black')
    
    # Panel B: 角度矩阵
    ax = axes[1]
    ax.set_facecolor(BG)
    angle_matrix = np.degrees(np.arccos(np.clip(cos_matrix, -1, 1)))
    im = ax.imshow(angle_matrix, cmap='YlOrRd', vmin=0, vmax=180, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mod_names, rotation=45, ha='right', fontsize=9, color=FG)
    ax.set_yticklabels(mod_names, fontsize=9, color=FG)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'Modality Gap Angle (°) — {label}', color=FG, fontweight='bold')
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{angle_matrix[i, j]:.1f}°',
                   ha='center', va='center', fontsize=8, color=FG if angle_matrix[i, j] > 90 else 'black')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'angle_heatmap_{label}.png')
    plt.savefig(out_path, facecolor=BG)
    plt.close()
    print(f"  ✅ Saved angle heatmap: {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='分别对 ckpt-1200 和 baseline 提取 embeddings 并做流形分析 (需要 GPU)'
    )
    parser.add_argument('--ckpt1200_path', required=True,
                       help='ckpt-1200 模型路径 (包含 model.safetensors)')
    parser.add_argument('--baseline_path', required=True,
                       help='Baseline (预训练) 模型路径')
    parser.add_argument('--base_model_path', default=None,
                       help='基座模型路径 (当 ckpt1200_path 缺少 config.json 时需要, '
                            '默认使用 --baseline_path)')
    parser.add_argument('--output_dir', default=RESULTS_DIR)
    parser.add_argument('--n_samples', type=int, default=100,
                       help='提取 embedding 使用的样本数')
    parser.add_argument('--skip_vision', action='store_true',
                       help='跳过视觉 embedding 提取 (图文输入较慢)')
    parser.add_argument('--skip_cka', action='store_true',
                       help='跳过两模型间 CKA 计算')
    args = parser.parse_args()
    
    # base_model_path 默认使用 baseline_path
    base_model_path = args.base_model_path or args.baseline_path
    
    print("=" * 70)
    print("  Dual-Model Modality Manifold Analysis")
    print("  (Vision / Language / Hidden States)")
    print("=" * 70)
    print(f"  ckpt-1200:  {args.ckpt1200_path}")
    print(f"  Baseline:   {args.baseline_path}")
    print(f"  Base model: {base_model_path}")
    print(f"  Output:     {args.output_dir}")
    print(f"  n_samples:  {args.n_samples}")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 提取 ckpt-1200 embeddings
    print("\n" + "=" * 70)
    print("  📦 Step 1: 提取 ckpt-1200 embeddings")
    print("=" * 70)
    emb_ckpt1200 = extract_embeddings_from_model(
        args.ckpt1200_path, args.output_dir, args.n_samples,
        model_label='ckpt1200', base_model_path=base_model_path,
    )
    
    # 2. 提取 baseline embeddings (baseline 自身有 config.json, 无需 base_model_path)
    print("\n" + "=" * 70)
    print("  📦 Step 2: 提取 Baseline embeddings")
    print("=" * 70)
    emb_baseline = extract_embeddings_from_model(
        args.baseline_path, args.output_dir, args.n_samples,
        model_label='baseline',
    )
    
    # 3. 计算几何度量
    print("\n" + "=" * 70)
    print("  📐 Step 3: 计算几何度量")
    print("=" * 70)
    
    if emb_ckpt1200:
        geo_ckpt = compute_geometry_metrics(emb_ckpt1200)
        out_path = os.path.join(args.output_dir, 'geometry_metrics_ckpt1200.json')
        with open(out_path, 'w') as f:
            json.dump(geo_ckpt, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved: {out_path}")
        for k, v in geo_ckpt.items():
            print(f"    {k}: {v:.4f}")
    
    if emb_baseline:
        geo_base = compute_geometry_metrics(emb_baseline)
        out_path = os.path.join(args.output_dir, 'geometry_metrics_baseline.json')
        with open(out_path, 'w') as f:
            json.dump(geo_base, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved: {out_path}")
        for k, v in geo_base.items():
            print(f"    {k}: {v:.4f}")
    
    # 4. 两模型间 CKA
    if not args.skip_cka and emb_ckpt1200 and emb_baseline:
        print("\n" + "=" * 70)
        print("  📊 Step 4: 两模型间 CKA")
        print("=" * 70)
        cka_results = compute_cka_between_models(emb_ckpt1200, emb_baseline, 'ckpt1200', 'baseline')
        out_path = os.path.join(args.output_dir, 'cka_between_models.json')
        with open(out_path, 'w') as f:
            json.dump(cka_results, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved: {out_path}")
        for k, v in cka_results.items():
            print(f"    {k}: {v:.4f}")
    
    # 5. 可视化
    print("\n" + "=" * 70)
    print("  🎨 Step 5: Token 语义流形可视化")
    print("=" * 70)
    
    if emb_ckpt1200:
        visualize_token_manifold(emb_ckpt1200, 'ckpt1200', args.output_dir)
    
    if emb_baseline:
        visualize_token_manifold(emb_baseline, 'baseline', args.output_dir)
    
    # 6. 两模型对比可视化
    if emb_ckpt1200 and emb_baseline:
        print("\n  📊 两模型对比可视化...")
        # 将两者合并做联合 t-SNE
        try:
            from sklearn.manifold import TSNE
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            BG = '#0D1117'
            FG = '#C9D1D9'
            
            fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
            ax.set_facecolor(BG)
            
            common_keys = sorted(set(emb_ckpt1200.keys()) & set(emb_baseline.keys()))
            for mod_name in common_keys:
                E_ckpt = emb_ckpt1200[mod_name]
                E_base = emb_baseline[mod_name]
                
                # 合并
                combined = np.vstack([E_ckpt, E_base])
                n_ckpt = len(E_ckpt)
                
                if len(combined) < 3:
                    continue
                
                tsne = TSNE(n_components=2, random_state=42,
                           perplexity=min(30, len(combined)-1))
                emb_2d = tsne.fit_transform(combined)
                
                # 分别画
                ax.scatter(emb_2d[:n_ckpt, 0], emb_2d[:n_ckpt, 1],
                          c='#A78BFA', alpha=0.5, s=20, marker='o',
                          label=f'ckpt1200/{mod_name}')
                ax.scatter(emb_2d[n_ckpt:, 0], emb_2d[n_ckpt:, 1],
                          c='#FF6B6B', alpha=0.5, s=20, marker='^',
                          label=f'baseline/{mod_name}')
            
            ax.set_title('Dual-Model Token Manifold Comparison', color=FG, fontweight='bold', fontsize=14)
            ax.legend(fontsize=8, facecolor=BG, edgecolor='#30363D', labelcolor=FG)
            ax.grid(True, alpha=0.2, color='#21262D')
            ax.tick_params(colors=FG)
            
            out_path = os.path.join(args.output_dir, 'dual_model_comparison.png')
            plt.savefig(out_path, facecolor=BG)
            plt.close()
            print(f"  ✅ Saved: {out_path}")
        except Exception as e:
            print(f"  ⚠️  Comparison visualization failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Dual-Model Modality Manifold Analysis 完成!")
    print(f"   结果目录: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
