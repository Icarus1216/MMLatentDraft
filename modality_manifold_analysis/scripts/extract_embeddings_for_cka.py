#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_embeddings_for_cka.py
=============================
从模型 checkpoint 中提取不同模态的 embeddings，
用于精确计算 CKA (Centered Kernel Alignment) 和其他几何度量。

需要 GPU 环境

产出:
  - modality_manifold_analysis/results/embeddings_baseline.pt
  - modality_manifold_analysis/results/embeddings_ckpt1200.pt
  - modality_manifold_analysis/results/cka_metrics.json
  - modality_manifold_analysis/results/geometry_metrics_ckpt1200.json

用法 (Docker):
  python3 extract_embeddings_for_cka.py --model_path /path/to/model
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 0. 路径
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. CKA 计算
# ============================================================
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA (Centered Kernel Alignment)
    参考: Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # 线性核
    kx = X @ X.T
    ky = Y @ Y.T

    # HSIC
    n = X.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    hsic_xy = np.trace(kx @ h @ ky @ h) / (n - 1) ** 2
    hsic_xx = np.trace(kx @ h @ kx @ h) / (n - 1) ** 2
    hsic_yy = np.trace(ky @ h @ ky @ h) / (n - 1) ** 2

    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """RBF Kernel CKA"""
    from scipy.spatial.distance import pdist, squareform

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    if sigma is None:
        sigma = np.median(pdist(X))

    def rbf_kernel(M, s):
        dists = squareform(pdist(M))
        return np.exp(-dists ** 2 / (2 * s ** 2))

    kx = rbf_kernel(X, sigma)
    ky = rbf_kernel(Y, sigma)

    n = X.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    hsic_xy = np.trace(kx @ h @ ky @ h) / (n - 1) ** 2
    hsic_xx = np.trace(kx @ h @ kx @ h) / (n - 1) ** 2
    hsic_yy = np.trace(ky @ h @ ky @ h) / (n - 1) ** 2

    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


# ============================================================
# 2. 模态几何度量
# ============================================================
def compute_geometry_metrics(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    从 embeddings 字典计算模态几何度量
    embeddings: {'vision': V, 'language': L, 'hidden_last': H, ...}
    """
    results = {}

    # 余弦相似度
    for name1 in embeddings:
        for name2 in embeddings:
            if name1 >= name2:
                continue
            E1 = embeddings[name1]
            E2 = embeddings[name2]
            # 平均余弦相似度
            cos_sims = []
            for i in range(min(len(E1), len(E2))):
                c = np.dot(E1[i], E2[i]) / (np.linalg.norm(E1[i]) * np.linalg.norm(E2[i]) + 1e-8)
                cos_sims.append(c)
            results[f'cos_{name1}_{name2}_mean'] = float(np.mean(cos_sims))
            results[f'cos_{name1}_{name2}_std'] = float(np.std(cos_sims))

    # CKA
    for name1 in embeddings:
        for name2 in embeddings:
            if name1 >= name2:
                continue
            E1 = embeddings[name1]
            E2 = embeddings[name2]
            n = min(len(E1), len(E2), 200)  # 限制样本数
            idx = np.random.choice(min(len(E1), len(E2)), n, replace=False)
            cka_val = linear_cka(E1[idx], E2[idx])
            results[f'cka_linear_{name1}_{name2}'] = float(cka_val)

    # 模态间隙角度
    for name1 in embeddings:
        for name2 in embeddings:
            if name1 >= name2:
                continue
            E1_mean = embeddings[name1].mean(axis=0)
            E2_mean = embeddings[name2].mean(axis=0)
            cos_mean = np.dot(E1_mean, E2_mean) / (np.linalg.norm(E1_mean) * np.linalg.norm(E2_mean) + 1e-8)
            results[f'modality_gap_angle_{name1}_{name2}'] = float(np.degrees(np.arccos(np.clip(cos_mean, -1, 1))))

    # 锥体半角
    for name, E in embeddings.items():
        E_mean = E.mean(axis=0)
        E_mean = E_mean / (np.linalg.norm(E_mean) + 1e-8)
        cos_to_mean = E @ E_mean / (np.linalg.norm(E, axis=1) + 1e-8)
        half_angle = np.degrees(np.arccos(np.clip(cos_to_mean.mean(), -1, 1)))
        results[f'cone_{name}_half_angle'] = float(half_angle)
        results[f'cone_{name}_cos_mean'] = float(cos_to_mean.mean())

    # 内部相似度
    for name, E in embeddings.items():
        # 对采样计算
        n = min(len(E), 100)
        idx = np.random.choice(len(E), n, replace=False)
        sub_E = E[idx]
        norms = np.linalg.norm(sub_E, axis=1, keepdims=True)
        sub_E_normed = sub_E / (norms + 1e-8)
        sim_matrix = sub_E_normed @ sub_E_normed.T
        np.fill_diagonal(sim_matrix, 0)
        results[f'intra_{name}_cos_mean'] = float(sim_matrix.sum() / (n * (n - 1)))

    return results


# ============================================================
# 3. 从已有 embeddings 文件计算 (不需要 GPU)
# ============================================================
def compute_from_existing_embeddings(embeddings_dir: str, output_dir: str):
    """
    从已有的 embeddings_*.pt 文件计算 CKA 和几何度量
    """
    print("\n📊 Computing CKA and geometry from existing embeddings...")

    all_results = {}

    # 查找所有 embeddings 文件
    emb_files = list(Path(embeddings_dir).glob('embeddings_*.pt'))
    if not emb_files:
        # 也搜索其他位置
        for subdir in ['modality_geometry_v2', 'modality_geometry_v2_ckpt1200']:
            alt_dir = os.path.join(PROJ_ROOT, 'outputs', subdir)
            if os.path.exists(alt_dir):
                emb_files.extend(Path(alt_dir).glob('embeddings*.pt'))
                emb_files.extend(Path(alt_dir).glob('*.pt'))

    for emb_file in emb_files:
        print(f"  Loading: {emb_file}")
        try:
            data = torch.load(emb_file, map_location='cpu')
        except Exception as e:
            print(f"  ⚠️ Failed to load {emb_file}: {e}")
            continue

        if isinstance(data, dict):
            embeddings = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    embeddings[k] = v.numpy()
                elif isinstance(v, np.ndarray):
                    embeddings[k] = v

            if embeddings:
                metrics = compute_geometry_metrics(embeddings)
                key = emb_file.stem
                all_results[key] = metrics

                # 保存单独的结果
                out_path = os.path.join(output_dir, f'geometry_{key}.json')
                with open(out_path, 'w') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                print(f"  💾 Saved: {out_path}")

                # CKA 矩阵
                if len(embeddings) >= 2:
                    names = sorted(embeddings.keys())
                    n = len(names)
                    cka_matrix = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                cka_matrix[i, j] = 1.0
                            elif i < j:
                                ns = min(len(embeddings[names[i]]), len(embeddings[names[j]]), 200)
                                idx = np.random.choice(min(len(embeddings[names[i]]), len(embeddings[names[j]])), ns, replace=False)
                                cka_matrix[i, j] = linear_cka(embeddings[names[i]][idx], embeddings[names[j]][idx])
                                cka_matrix[j, i] = cka_matrix[i, j]

                    all_results[f'{key}_cka_matrix'] = {
                        'names': names,
                        'matrix': cka_matrix.tolist(),
                    }

    # 汇总保存
    out_path = os.path.join(output_dir, 'cka_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved: {out_path}")

    return all_results


# ============================================================
# 4. 从模型提取 embeddings (需要 GPU)
# ============================================================
def extract_from_model(model_path: str, output_dir: str, n_samples: int = 100):
    """
    从模型 checkpoint 提取不同模态的 embeddings
    需要 GPU 环境和模型依赖
    """
    print(f"\n🔧 Extracting embeddings from: {model_path}")

    try:
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        import torch.nn.functional as F
    except ImportError:
        print("  ⚠️ transformers not available, skipping model extraction")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    print("  Loading model...")
    try:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        model = model.to(device).eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠️ Failed to load model: {e}")
        return None

    # 准备输入数据
    # 纯文本输入
    texts = [
        "What is the capital of France?",
        "Describe a beautiful sunset.",
        "How does photosynthesis work?",
        "Explain the theory of relativity.",
        "What are the properties of water?",
    ][:n_samples]

    # 图文对 (使用 dummy 图像)
    from PIL import Image
    dummy_images = [Image.new('RGB', (224, 224), color=(i*40, i*30, i*20)) for i in range(min(5, n_samples))]

    embeddings = {
        'vision': [],
        'language': [],
        'hidden_last': [],
        'hidden_intermediate': [],
    }

    with torch.no_grad():
        # 文本输入
        for text in texts:
            inputs = processor(text=text, return_tensors='pt').to(device)
            outputs = model(**inputs, output_hidden_states=True)

            # 语言 embedding
            last_hidden = outputs.last_hidden_state
            embeddings['language'].append(last_hidden.mean(dim=1).cpu().float().numpy())

            # Hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                for layer_idx in [-1, -4, -8]:
                    hs = outputs.hidden_states[layer_idx]
                    key = 'hidden_last' if layer_idx == -1 else f'hidden_l{len(outputs.hidden_states)+layer_idx}'
                    embeddings.setdefault(key, []).append(hs.mean(dim=1).cpu().float().numpy())

        # 图文输入
        for img in dummy_images:
            inputs = processor(images=img, text="Describe this image.", return_tensors='pt').to(device)
            outputs = model(**inputs, output_hidden_states=True)

            # 视觉 embedding
            if hasattr(outputs, 'vision_outputs') and outputs.vision_outputs is not None:
                vis_emb = outputs.vision_outputs.last_hidden_state
                embeddings['vision'].append(vis_emb.mean(dim=1).cpu().float().numpy())

            # 图文输入的 hidden states
            last_hidden = outputs.last_hidden_state
            embeddings['hidden_last'].append(last_hidden.mean(dim=1).cpu().float().numpy())

    # 转换为 numpy
    for k in list(embeddings.keys()):
        if embeddings[k]:
            embeddings[k] = np.concatenate(embeddings[k], axis=0)
        else:
            del embeddings[k]

    # 保存
    model_name = Path(model_path).name
    out_path = os.path.join(output_dir, f'embeddings_{model_name}.pt')
    torch.save(embeddings, out_path)
    print(f"  💾 Saved: {out_path}")

    # 计算几何度量
    metrics = compute_geometry_metrics(embeddings)
    out_metrics = os.path.join(output_dir, f'geometry_{model_name}.json')
    with open(out_metrics, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved: {out_metrics}")

    return embeddings


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None,
                       help='模型路径 (需要 GPU)')
    parser.add_argument('--embeddings_dir', default=None,
                       help='已有 embeddings 目录')
    parser.add_argument('--output_dir', default=RESULTS_DIR)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--skip_model', action='store_true',
                       help='跳过模型提取, 仅从已有 embeddings 计算')
    args = parser.parse_args()

    print("=" * 70)
    print("  Modality Manifold Analysis — Embeddings & CKA")
    print("=" * 70)

    # 1. 从已有 embeddings 计算
    if args.embeddings_dir:
        compute_from_existing_embeddings(args.embeddings_dir, args.output_dir)
    else:
        # 默认搜索路径
        for emb_dir in [
            os.path.join(PROJ_ROOT, 'outputs/modality_geometry_v2'),
            os.path.join(PROJ_ROOT, 'outputs/modality_geometry_v2_ckpt1200'),
        ]:
            if os.path.exists(emb_dir):
                compute_from_existing_embeddings(emb_dir, args.output_dir)

    # 2. 从模型提取 (需要 GPU)
    if not args.skip_model and args.model_path:
        extract_from_model(args.model_path, args.output_dir, args.n_samples)

    print("\n✅ Done.")


if __name__ == '__main__':
    main()
