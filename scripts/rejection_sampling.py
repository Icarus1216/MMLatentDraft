#!/usr/bin/env python3
"""
RLD 拒绝采样脚本 — 收集错误 CoT + 正确答案配对数据

核心目标:
  对每个训练样本，用 Qwen3-VL-8B-Instruct 采样 N 次 CoT，
  根据最终答案的正确性将 CoT 分为 "正确" 和 "错误" 两类，
  构建 (wrong_cot, correct_answer) 配对数据，用于 RLD 纠错训练。

输出数据格式 (与 data.py 完全兼容):
  {
    "image": "/path/to/image.jpg",
    "question": "...",
    "answer": "D",                          # GT 正确答案 (始终参与 loss)
    "think_chain": "<think>\\n...错误推理...\\n</think>\\n",  # 错误的 CoT
    "think_chain_status": "valid",
    "think_chain_source": "wrong_cot",      # 不在 SUPERVISED_THINK_SOURCES 中
                                            # → think 块 labels=-100, 只监督 answer
    "correct_cot": "<think>\\n...正确推理...\\n</think>\\n",  # 可选: 配对的正确 CoT
    "rejection_sampling_meta": {
      "num_samples": 8,
      "num_correct": 3,
      "num_wrong": 5,
      "pass_rate": 0.375,
      "selected_wrong_idx": 2
    }
  }

训练时的效果:
  - Pass 1: 基座处理错误 CoT → full_hidden 包含错误推理的语义
  - Controller: 从错误 hidden 中提取 step summaries → 生成 Z_d
  - Pass 2: Adapter 用 Z_d 修正 hidden → 目标是让 logits 输出正确答案
  - Loss: 只在 final answer 部分计算 (think 块 labels=-100)

数据筛选策略:
  - 只保留 "至少有 1 个正确 + 至少有 1 个错误" 的样本 (有对照)
  - 优先选择 "差一点就对" 的错误 CoT (步骤数接近正确 CoT)
  - 按 pass_rate 分层: 低 pass_rate 的题目更有训练价值 (模型容易犯错)

使用方法:
    # 基本用法 (单 GPU)
    python scripts/rejection_sampling.py \\
        --model_path /path/to/Qwen3-VL-8B-Instruct \\
        --input_json data/rld_50k_selected.json \\
        --output_json data/rld_50k_rejection_sampled.json

    # 多 GPU + 自定义采样数
    python scripts/rejection_sampling.py \\
        --model_path /path/to/Qwen3-VL-8B-Instruct \\
        --input_json data/rld_50k_selected.json \\
        --output_json data/rld_50k_rejection_sampled.json \\
        --tensor_parallel_size 4 \\
        --num_samples 16 \\
        --batch_size 64

    # 只采样子集 (调试)
    python scripts/rejection_sampling.py \\
        --model_path /path/to/Qwen3-VL-8B-Instruct \\
        --input_json data/rld_50k_selected.json \\
        --output_json data/rld_50k_rejection_sampled.json \\
        --max_samples 1000 \\
        --num_samples 4
"""

import os

# vLLM V1 引擎的 multiproc executor 在容器环境中使用 fork 创建子进程时,
# 子进程可能无法正确继承 CUDA 上下文, 导致 "CUDA driver initialization failed"。
# 设置为 spawn 让子进程重新初始化 CUDA, 解决此问题。
# 参考: https://github.com/vllm-project/vllm (Qwen3-VL 官方示例)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import re
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("[拒绝采样] ❌ vLLM 未安装，请运行: pip install vllm")
    sys.exit(1)

from PIL import Image


# ============================================================
# System Prompt (与训练时完全一致)
# ============================================================

RLD_SYSTEM_PROMPT = """You are a visual reasoning assistant. Think step by step. Be concise.

Rules:
1. Use numbered steps: "Step 1:", "Step 2:", etc.
2. Each step should focus on one reasoning action (observe, calculate, deduce, etc.).
3. Do NOT add introductions, greetings, summaries, or filler words.
4. End with "Final Answer:" on its own line. This is MANDATORY.

Example:
Step 1: The triangle has base 6 and height 8.
Step 2: Area = 0.5 × 6 × 8 = 24.
Final Answer: 24 square units."""


# ============================================================
# 解析与验证工具函数 (复用 pregenerate_think.py 的逻辑)
# ============================================================

def parse_think_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从模型输出中解析推理链和最终答案

    支持两种格式:
      1. <think>...</think> 格式 (Thinking 模型)
      2. Step N: ... + Final Answer: ... 格式 (Instruct 模型)
    """
    # ---- 策略 1: 尝试 <think>...</think> 格式 (兼容 Thinking 模型) ----
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        think_content = match.group(1).strip()
        answer = text[match.end():].strip()
        return think_content, answer

    # <think> 开头但未闭合 (被截断)
    fallback = re.search(r'<think>(.*)', text, re.DOTALL)
    if fallback:
        content = fallback.group(1).strip()
        if content and '</step>' in content:
            return content, None

    # ---- 策略 2: Step N: + Final Answer: 格式 (Instruct 模型) ----
    # 提取 Final Answer
    # 匹配各种变体: "Final Answer:", "### Final Answer:", "**Final Answer:**", "✅ Final Answer:"
    final_answer_pattern = r'(?:#+\s*)?(?:\*\*)?(?:✅\s*)?Final\s*Answer\s*:?\s*(?:\*\*)?\s*(.+?)(?:\n|$)'
    fa_match = re.search(final_answer_pattern, text, re.IGNORECASE)

    # 如果没有 Final Answer，尝试其他常见答案模式
    gen_answer = None
    answer_pos = len(text)  # 答案在文本中的位置

    if fa_match:
        gen_answer = fa_match.group(1).strip()
        answer_pos = fa_match.start()
    else:
        # 尝试匹配: "The correct answer is **B**", "The answer is B", "Answer: B"
        alt_patterns = [
            (r'(?:The\s+)?(?:correct\s+)?(?:answer|choice)\s+is\s*:?\s*(?:\*\*)?([^\n*]+)', None),
            (r'(?:^|\n)\s*(?:\*\*)?Answer\s*:?\s*(?:\*\*)?\s*(.+?)(?:\n|$)', None),
        ]
        for pat, _ in alt_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                gen_answer = m.group(1).strip()
                answer_pos = m.start()
                break

    # 清理答案中的 markdown 格式符号 (**, *, ---, ✅ 等)
    if gen_answer:
        gen_answer = re.sub(r'\*+$', '', gen_answer).strip()  # 去尾部 **
        gen_answer = re.sub(r'^\*+', '', gen_answer).strip()  # 去头部 **
        gen_answer = re.sub(r'^-+\s*', '', gen_answer).strip()  # 去头部 ---
        gen_answer = re.sub(r'\s*-+$', '', gen_answer).strip()  # 去尾部 ---

    # 提取推理部分 (Final Answer 之前的所有文本)
    reasoning_text = text[:answer_pos].strip()

    if not reasoning_text:
        return None, gen_answer

    # 按 Step N: 分割推理步骤
    # 匹配: "Step 1:", "### Step 1:", "**Step 1:**", "step 1."
    step_pattern = r'(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?(?:Step\s*\d+)\s*[:.]\s*(?:\*\*)?'
    step_splits = re.split(step_pattern, reasoning_text, flags=re.IGNORECASE)

    # 过滤空段
    steps = [s.strip() for s in step_splits if s and s.strip()]

    if len(steps) >= 2:
        # 成功按 Step N: 分割，保留原始 Step N: 格式
        # 重新编号确保连续
        cot_parts = [f"Step {i}: {s}" for i, s in enumerate(steps, 1)]
        think_content = "\n".join(cot_parts)
        return think_content, gen_answer

    # 如果没有 Step N: 格式，尝试按段落分割 (双换行 或 --- 分隔)
    para_splits = re.split(r'\n\s*(?:---+|\n)\s*\n', reasoning_text)
    paras = [p.strip() for p in para_splits if p and p.strip() and len(p.strip()) > 15]

    if len(paras) >= 2:
        cot_parts = [f"Step {i}: {p}" for i, p in enumerate(paras, 1)]
        think_content = "\n".join(cot_parts)
        return think_content, gen_answer

    # 最后兜底: 按单换行分割，合并短段
    lines = [l.strip() for l in reasoning_text.split('\n') if l.strip()]
    if len(lines) >= 3:
        # 每 3~5 行合并为一个 step
        chunk_size = max(2, len(lines) // 4)  # 目标 ~4 个 steps
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        if len(chunks) >= 2:
            cot_parts = [f"Step {i}: {c}" for i, c in enumerate(chunks, 1)]
            think_content = "\n".join(cot_parts)
            return think_content, gen_answer

    # 实在无法分步，返回 None
    return None, gen_answer


def validate_think_chain(
    think_content: str,
    min_steps: int = 4,
    max_steps: int = 10,
    min_chars_per_step: int = 10,
    max_cot_chars: int = 1500,
) -> Tuple[bool, str]:
    """
    验证推理链格式合规性

    支持两种格式:
      1. 新格式: "Step 1: ...\nStep 2: ..." (Instruct 模型原生输出)
      2. 旧格式: "...\n</step>\n...\n</step>" (兼容已有数据)
    """
    if not think_content or not think_content.strip():
        return False, "think 内容为空"

    # 总长度检查 — 防止超长 CoT 导致训练 OOM
    if len(think_content) > max_cot_chars:
        return False, f"CoT 总长度过长: {len(think_content)} > {max_cot_chars}"

    # 检测格式并计算步骤数
    step_n_matches = re.findall(r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]', think_content, re.IGNORECASE)
    if step_n_matches:
        # 新格式: 按 "Step N:" 计数
        num_steps = len(step_n_matches)
        # 按 Step N: 分割提取步骤内容
        step_pattern = r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]'
        step_splits = re.split(step_pattern, think_content, flags=re.IGNORECASE)
        non_empty_steps = [s.strip() for s in step_splits if s.strip()]
    else:
        # 旧格式: 按 </step> 计数
        num_steps = think_content.count("</step>")
        non_empty_steps = [s.strip() for s in think_content.split("</step>") if s.strip()]

    if num_steps < min_steps:
        return False, f"step 数量不足: {num_steps} < {min_steps}"
    if num_steps > max_steps:
        return False, f"step 数量过多: {num_steps} > {max_steps}"

    for i, step in enumerate(non_empty_steps):
        if len(step) < min_chars_per_step:
            return False, f"step {i+1} 内容过短: {len(step)} < {min_chars_per_step}"

    return True, "合规"


def normalize_think_block(think_content: str) -> str:
    """
    规范化推理链，确保与 RLDDataset 新格式一致
    
    新格式: "Step 1: ...\nStep 2: ...\n" (不再包裹 <think>...</think>)
    
    支持输入:
      1. 已有 Step N: 格式 → 直接规范化编号
      2. </step> 分隔格式 → 转换为 Step N: 格式
      3. 纯文本 → 作为单步处理
    """
    if not think_content or not think_content.strip():
        return "Step 1: Analyzing the image.\n"
    
    # 检测是否已经是 Step N: 格式
    step_n_matches = list(re.finditer(r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]', think_content, re.IGNORECASE))
    if step_n_matches:
        # 已经是 Step N: 格式, 提取步骤内容并重新编号
        step_pattern = r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]'
        step_splits = re.split(step_pattern, think_content, flags=re.IGNORECASE)
        steps = [s.strip() for s in step_splits if s.strip()]
    elif "</step>" in think_content:
        # 旧格式: 按 </step> 分割
        parts = think_content.split("</step>")
        steps = [p.strip() for p in parts if p.strip()]
    else:
        # 纯文本: 按行分割
        lines = [l.strip() for l in think_content.split('\n') if l.strip()]
        steps = lines if lines else [think_content.strip()]
    
    if not steps:
        return "Step 1: Analyzing the image.\n"
    
    # 重新编号为 Step N: 格式
    cot_parts = [f"Step {i}: {step}" for i, step in enumerate(steps, 1)]
    return "\n".join(cot_parts) + "\n"


def count_steps(think_content: str) -> int:
    """统计推理链中的步骤数 (兼容新旧格式)"""
    if not think_content:
        return 0
    # 新格式: 统计 "Step N:" 的数量
    step_n_count = len(re.findall(r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]', think_content, re.IGNORECASE))
    if step_n_count > 0:
        return step_n_count
    # 旧格式: 统计 </step> 的数量
    return think_content.count("</step>")


def check_answer_match(generated_answer: str, gt_answer: str) -> bool:
    """
    宽松检查模型生成的 answer 是否与 GT answer 一致

    策略: 归一化后做子串匹配 + 数值匹配 + 选择题字母匹配
    """
    if not generated_answer or not gt_answer:
        return False

    def normalize(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    gen_norm = normalize(generated_answer)
    gt_norm = normalize(gt_answer)

    if not gen_norm or not gt_norm:
        return False

    # 完全匹配
    if gen_norm == gt_norm:
        return True

    # 子串匹配
    if gt_norm in gen_norm or gen_norm in gt_norm:
        return True

    # 选择题字母匹配: GT 是单个字母 (A/B/C/D/E)
    gt_stripped = gt_answer.strip().upper()
    if len(gt_stripped) == 1 and gt_stripped in 'ABCDE':
        # 从生成答案中提取选择题字母
        # 匹配模式: "Answer: D", "The answer is D", "D", "(D)", "D."
        choice_patterns = [
            r'(?:answer|choice|option)\s*(?:is|:)\s*([A-E])\b',
            r'\b([A-E])\s*$',           # 末尾单独字母
            r'\(([A-E])\)',              # 括号包裹
            r'^([A-E])\b',              # 开头单独字母
            r'\b([A-E])\.',             # 字母后跟句号
        ]
        for pat in choice_patterns:
            m = re.search(pat, generated_answer.strip(), re.IGNORECASE)
            if m and m.group(1).upper() == gt_stripped:
                return True

    # 数值匹配
    gen_nums = re.findall(r'[\d.]+', gen_norm)
    gt_nums = re.findall(r'[\d.]+', gt_norm)
    if gen_nums and gt_nums:
        try:
            gen_val = float(gen_nums[-1])
            gt_val = float(gt_nums[-1])
            if abs(gen_val - gt_val) < 1e-6:
                return True
            # 相对误差 < 0.1%
            if gt_val != 0 and abs(gen_val - gt_val) / abs(gt_val) < 0.001:
                return True
        except ValueError:
            pass

    return False


# ============================================================
# vLLM 引擎封装
# ============================================================

class VLLMEngine:
    """vLLM 引擎封装，支持多模态推理和多次采样"""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        max_pixels: int = 602112,
        trust_remote_code: bool = True,
    ):
        print(f"[vLLM] 初始化引擎...")
        print(f"  - 模型: {model_path}")
        print(f"  - Tensor Parallel: {tensor_parallel_size} GPU(s)")
        print(f"  - 最大上下文: {max_model_len}")
        print(f"  - 图片最大像素: {max_pixels}")

        self.max_pixels = max_pixels
        self.max_model_len = max_model_len

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt={"image": 1},
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[vLLM] ✅ 引擎初始化完成")

    def build_chat_prompt(
        self,
        system_prompt: str,
        image_path: str,
        user_text: str,
    ) -> Tuple[str, Optional[Image.Image]]:
        """构造 chat 格式的 prompt"""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        pil_image = None
        if image_path and os.path.exists(image_path):
            try:
                pil_image = Image.open(image_path).convert("RGB")
                # 限制图片分辨率以控制 image tokens 数量
                # Qwen3-VL 的 image tokens 与像素数成正比
                w, h = pil_image.size
                total_pixels = w * h
                if total_pixels > self.max_pixels:
                    scale = (self.max_pixels / total_pixels) ** 0.5
                    new_w = max(28, int(w * scale))
                    new_h = max(28, int(h * scale))
                    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            except Exception as e:
                print(f"[vLLM] ⚠️ 无法加载图像 {image_path}: {e}")

        return prompt, pil_image

    def generate_multi_sample(
        self,
        prompts: List[str],
        images: List[Optional[Image.Image]],
        sampling_params: SamplingParams,
    ) -> Tuple[List[List[str]], List[int]]:
        """
        批量生成，每个 prompt 生成 n 个采样结果
        对超长 prompt 进行容错处理: 逐个添加请求, 跳过超长的

        Returns:
            (results, valid_mask):
              results: List[List[str]], 外层是有效 prompt 维度，内层是 n 个采样结果
              valid_mask: List[int], 有效 prompt 在原始列表中的索引
        """
        vllm_inputs = []
        valid_indices = []  # 记录哪些 prompt 被成功添加

        for idx, (prompt, image) in enumerate(zip(prompts, images)):
            item = {"prompt": prompt}
            if image is not None:
                item["multi_modal_data"] = {"image": image}
            vllm_inputs.append(item)
            valid_indices.append(idx)

        if not vllm_inputs:
            return [], []

        try:
            outputs = self.llm.generate(vllm_inputs, sampling_params)
        except ValueError as e:
            err_msg = str(e)
            if "longer than the maximum model length" in err_msg:
                # 有超长 prompt, 回退到逐个生成模式
                print(f"[vLLM] ⚠️ 批量生成遇到超长 prompt, 回退到逐个生成模式")
                results = []
                valid_indices = []
                for idx, inp in enumerate(vllm_inputs):
                    try:
                        out = self.llm.generate([inp], sampling_params)
                        sample_texts = [o.text for o in out[0].outputs]
                        results.append(sample_texts)
                        valid_indices.append(idx)
                    except ValueError as e2:
                        if "longer than the maximum model length" in str(e2):
                            print(f"[vLLM] ⚠️ 跳过超长 prompt (idx={idx}): {str(e2)[:100]}")
                        else:
                            raise
                return results, valid_indices
            else:
                raise

        results = []
        for output in outputs:
            sample_texts = [o.text for o in output.outputs]
            results.append(sample_texts)
        return results, valid_indices


# ============================================================
# 错误 CoT 选择策略
# ============================================================

def select_best_wrong_cot(
    wrong_cots: List[Dict],
    correct_cots: List[Dict],
    max_cot_chars: int = 1500,
) -> Dict:
    """
    从多个错误 CoT 中选择最有训练价值的一个

    策略 (优先级从高到低):
      1. 优先选择长度在 max_cot_chars 以内的 CoT
      2. "差一点就对" — 步骤数与正确 CoT 最接近的错误 CoT
      3. 步骤数适中 (4~10 步) — 太短信息量不足，太长训练慢
      4. 长度适中 — 优先选简洁的 CoT

    Args:
        wrong_cots: 错误 CoT 列表
        correct_cots: 正确 CoT 列表 (用于计算步骤数差异)
        max_cot_chars: CoT 最大字符数

    Returns:
        选中的错误 CoT dict
    """
    if not wrong_cots:
        return None

    # 优先过滤掉超长 CoT
    short_cots = [c for c in wrong_cots if len(c["think_content"]) <= max_cot_chars]
    candidates = short_cots if short_cots else wrong_cots  # 全部超长时退化为全量

    # 计算正确 CoT 的平均步骤数
    if correct_cots:
        avg_correct_steps = sum(
            count_steps(c["think_content"]) for c in correct_cots
        ) / len(correct_cots)
    else:
        avg_correct_steps = 7.0  # 默认值 (配合 prompt 的 4-10 步约束)

    def score_wrong_cot(cot: Dict) -> float:
        """为错误 CoT 打分 (越高越好)"""
        steps = count_steps(cot["think_content"])
        content_len = len(cot["think_content"])

        score = 0.0

        # 1. 步骤数与正确 CoT 的接近度 (最重要, 权重 0.4)
        step_diff = abs(steps - avg_correct_steps)
        step_proximity = max(0, 1.0 - step_diff / 5.0)
        score += 0.4 * step_proximity

        # 2. 步骤数适中性 (权重 0.3) — 偏好 4-10 步
        if 4 <= steps <= 10:
            score += 0.3
        elif steps <= 15:
            score += 0.15
        else:
            score += 0.0

        # 3. 长度简洁性 (权重 0.3) — 越短越好，但不能太短
        if 100 <= content_len <= 500:
            score += 0.3
        elif content_len <= 1000:
            score += 0.2
        elif content_len <= max_cot_chars:
            score += 0.1
        else:
            score -= 0.2  # 超长惩罚

        return score

    # 按分数排序，选最高分
    scored = [(score_wrong_cot(c), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def select_best_correct_cot(correct_cots: List[Dict], max_cot_chars: int = 1500) -> Optional[Dict]:
    """从多个正确 CoT 中选择最佳的一个 (用于配对)，优先选简洁的"""
    if not correct_cots:
        return None

    # 优先过滤掉超长 CoT
    short_cots = [c for c in correct_cots if len(c["think_content"]) <= max_cot_chars]
    candidates = short_cots if short_cots else correct_cots

    def score_correct_cot(cot: Dict) -> float:
        steps = count_steps(cot["think_content"])
        content_len = len(cot["think_content"])
        score = 0.0
        # 步骤数适中 — 偏好 4-10 步
        if 4 <= steps <= 10:
            score += 0.5
        elif steps <= 15:
            score += 0.25
        else:
            score += 0.0
        # 长度简洁性 — 越短越好
        if 100 <= content_len <= 500:
            score += 0.5
        elif content_len <= 1000:
            score += 0.3
        elif content_len <= max_cot_chars:
            score += 0.1
        else:
            score -= 0.2  # 超长惩罚
        return score

    scored = [(score_correct_cot(c), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


# ============================================================
# 主采样流程
# ============================================================

def run_rejection_sampling(
    engine: VLLMEngine,
    samples: List[Dict],
    num_samples: int = 8,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    min_steps: int = 4,
    max_steps: int = 10,
    max_cot_chars: int = 1500,
    batch_size: int = 32,
    save_raw_outputs: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    对所有样本执行拒绝采样

    对每个样本:
      1. 用 vLLM 采样 num_samples 次 CoT
      2. 按答案正确性分为 correct / wrong
      3. 选择最佳的 wrong_cot + correct_answer 配对

    Args:
        engine: vLLM 引擎
        samples: 输入样本列表
        num_samples: 每个样本的采样次数
        max_tokens: 最大生成 token 数
        temperature: 采样温度 (越高越多样)
        top_p: nucleus sampling 参数
        min_steps: 最少步骤数
        max_steps: 最多步骤数
        max_cot_chars: CoT 最大字符数 (超过则判为无效, 防止训练 OOM)
        batch_size: 批处理大小 (每批处理的样本数, 每个样本生成 num_samples 个)

    Returns:
        (paired_data, stats): 配对数据列表和统计信息
    """
    print(f"\n{'='*70}")
    print(f"[拒绝采样] 开始采样")
    print(f"  - 总样本数: {len(samples)}")
    print(f"  - 每样本采样次数: {num_samples}")
    print(f"  - 总生成次数: {len(samples) * num_samples}")
    print(f"  - 温度: {temperature}, top_p: {top_p}")
    print(f"  - 批大小: {batch_size}")
    print(f"  - 最大 token: {max_tokens}")
    print(f"{'='*70}")

    sampling_params = SamplingParams(
        n=num_samples,           # 每个 prompt 生成 n 个采样
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=None,               # 不固定种子，保证多样性
    )

    paired_data = []
    raw_outputs_log = []  # 保存模型原始输出 (用于诊断)
    total = len(samples)

    # 统计
    stats = {
        "total_samples": total,
        "total_generations": 0,
        "image_errors": 0,
        "all_correct": 0,         # 所有采样都正确 (模型太强, 无错误 CoT)
        "all_wrong": 0,           # 所有采样都错误 (模型太弱, 无正确 CoT 参照)
        "has_both": 0,            # 有正确也有错误 (理想情况)
        "all_invalid": 0,         # 所有采样格式都无效
        "paired_output": 0,       # 最终输出的配对数
        "pass_rate_dist": [],     # 每个样本的 pass rate
    }

    for batch_start in tqdm(range(0, total, batch_size), desc="[拒绝采样] 批次"):
        batch_end = min(batch_start + batch_size, total)
        batch = samples[batch_start:batch_end]

        # 构造 prompts
        prompts, images_list, valid_indices = [], [], []

        for i, sample in enumerate(batch):
            global_idx = batch_start + i
            image_path = sample.get("image", "")
            question = sample.get("question", "")

            prompt, pil_image = engine.build_chat_prompt(
                RLD_SYSTEM_PROMPT, image_path, question
            )

            if pil_image is None and image_path:
                stats["image_errors"] += 1
                continue

            prompts.append(prompt)
            images_list.append(pil_image)
            valid_indices.append(global_idx)

        if not prompts:
            continue

        # 批量生成 (每个 prompt 生成 num_samples 个)
        t0 = time.time()
        all_outputs, gen_valid_indices = engine.generate_multi_sample(prompts, images_list, sampling_params)
        dt = time.time() - t0
        skipped = len(prompts) - len(all_outputs)
        stats["total_generations"] += len(all_outputs) * num_samples
        if skipped > 0:
            stats.setdefault("prompt_too_long", 0)
            stats["prompt_too_long"] += skipped

        if batch_start % (batch_size * 5) == 0:
            print(f"\n[拒绝采样] 批次 {batch_start//batch_size + 1}/"
                  f"{(total + batch_size - 1)//batch_size}, "
                  f"耗时 {dt:.1f}s, "
                  f"已配对 {stats['paired_output']}"
                  f"{f', 跳过超长 {skipped}' if skipped else ''}")

        # 处理每个样本的 N 个采样结果
        for j, sample_outputs in enumerate(all_outputs):
            global_idx = valid_indices[gen_valid_indices[j]]
            sample = samples[global_idx]
            gt_answer = sample.get("answer", "")

            # 保存原始输出用于诊断
            if save_raw_outputs:
                raw_entry = {
                    "global_idx": global_idx,
                    "question": sample.get("question", ""),
                    "gt_answer": gt_answer,
                    "image": sample.get("image", ""),
                    "num_outputs": len(sample_outputs),
                    "outputs": [],
                }
                for k, text in enumerate(sample_outputs):
                    think_content, gen_answer = parse_think_and_answer(text)
                    is_valid, reason = validate_think_chain(
                        think_content, min_steps=min_steps, max_steps=max_steps,
                        max_cot_chars=max_cot_chars
                    ) if think_content else (False, "no think block")

                    # ---- 格式遵循诊断 ----
                    # 检测 Step N: 格式
                    step_matches = re.findall(
                        r'(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?Step\s*(\d+)\s*[:.]\s*(?:\*\*)?',
                        text, re.IGNORECASE
                    )
                    has_step_format = len(step_matches) >= 2
                    num_steps_found = len(step_matches)

                    # 检测 Final Answer: 格式
                    has_final_answer = bool(re.search(
                        r'(?:#+\s*)?(?:\*\*)?Final\s*Answer\s*:', text, re.IGNORECASE
                    ))

                    # 判断使用了哪种解析策略
                    if think_content is None:
                        parse_method = "failed"
                    elif re.search(r'<think>(.*?)</think>', text, re.DOTALL):
                        parse_method = "think_tag"
                    elif has_step_format:
                        parse_method = "step_format"
                    elif re.search(r'\n\s*(?:---+|\n)\s*\n', text):
                        parse_method = "paragraph_split"
                    else:
                        parse_method = "line_merge"

                    # 严格遵循: 同时有 Step N: 分步 + Final Answer: 答案
                    strict_format_compliance = has_step_format and has_final_answer

                    raw_entry["outputs"].append({
                        "sample_idx": k,
                        "raw_text": text[:3000],  # 截断避免文件过大
                        "has_think_block": think_content is not None,
                        "think_valid": is_valid,
                        "think_valid_reason": reason,
                        "parsed_answer": (gen_answer or "")[:200] if gen_answer else None,
                        "answer_match": check_answer_match(gen_answer or "", gt_answer) if gen_answer else False,
                        # 新增: 格式遵循诊断
                        "has_step_format": has_step_format,
                        "num_steps_found": num_steps_found,
                        "has_final_answer": has_final_answer,
                        "parse_method": parse_method,
                        "strict_format_compliance": strict_format_compliance,
                    })
                raw_outputs_log.append(raw_entry)

            correct_cots = []
            wrong_cots = []
            invalid_count = 0

            for k, text in enumerate(sample_outputs):
                # 超长输出直接丢弃，不尝试修复
                if len(text) > max_cot_chars:
                    invalid_count += 1
                    continue

                think_content, gen_answer = parse_think_and_answer(text)

                if think_content is None:
                    invalid_count += 1
                    continue

                is_valid, reason = validate_think_chain(
                    think_content, min_steps=min_steps, max_steps=max_steps,
                    max_cot_chars=max_cot_chars
                )

                if not is_valid:
                    invalid_count += 1
                    continue

                cot_info = {
                    "think_content": think_content,
                    "gen_answer": gen_answer or "",
                    "text": text,
                    "sample_idx": k,
                }

                is_correct = check_answer_match(gen_answer or "", gt_answer)

                if is_correct:
                    correct_cots.append(cot_info)
                else:
                    wrong_cots.append(cot_info)

            # 分类统计
            valid_count = len(correct_cots) + len(wrong_cots)
            if valid_count == 0:
                stats["all_invalid"] += 1
                continue

            pass_rate = len(correct_cots) / valid_count
            stats["pass_rate_dist"].append(pass_rate)

            if len(correct_cots) > 0 and len(wrong_cots) > 0:
                # ✅ 理想情况: 有正确也有错误
                stats["has_both"] += 1

                # 选择最佳的错误 CoT 和正确 CoT
                best_wrong = select_best_wrong_cot(wrong_cots, correct_cots, max_cot_chars=max_cot_chars)
                best_correct = select_best_correct_cot(correct_cots, max_cot_chars=max_cot_chars)

                wrong_think_block = normalize_think_block(best_wrong["think_content"])

                paired_item = {
                    "image": sample["image"],
                    "question": sample["question"],
                    "answer": gt_answer,                    # GT 正确答案
                    "think_chain": wrong_think_block,       # 错误的 CoT
                    "think_chain_status": "valid",
                    "think_chain_source": "wrong_cot",      # 标记为错误 CoT
                    "rejection_sampling_meta": {
                        "num_samples": num_samples,
                        "num_valid": valid_count,
                        "num_correct": len(correct_cots),
                        "num_wrong": len(wrong_cots),
                        "num_invalid": invalid_count,
                        "pass_rate": round(pass_rate, 4),
                        "wrong_gen_answer": best_wrong["gen_answer"][:100],
                        "wrong_num_steps": count_steps(best_wrong["think_content"]),
                    },
                }

                # 可选: 附带正确 CoT (用于后续分析或 DPO 训练)
                if best_correct:
                    correct_think_block = normalize_think_block(best_correct["think_content"])
                    paired_item["correct_cot"] = correct_think_block

                paired_data.append(paired_item)
                stats["paired_output"] += 1

            elif len(correct_cots) == valid_count:
                stats["all_correct"] += 1
            elif len(wrong_cots) == valid_count:
                # 全部错误: 没有正确 CoT 参照，但仍然可以用 GT answer 做监督
                stats["all_wrong"] += 1

                # 也输出这些样本 (没有 correct_cot 字段)
                best_wrong = select_best_wrong_cot(wrong_cots, [], max_cot_chars=max_cot_chars)
                wrong_think_block = normalize_think_block(best_wrong["think_content"])

                paired_item = {
                    "image": sample["image"],
                    "question": sample["question"],
                    "answer": gt_answer,
                    "think_chain": wrong_think_block,
                    "think_chain_status": "valid",
                    "think_chain_source": "wrong_cot",
                    "rejection_sampling_meta": {
                        "num_samples": num_samples,
                        "num_valid": valid_count,
                        "num_correct": 0,
                        "num_wrong": len(wrong_cots),
                        "num_invalid": invalid_count,
                        "pass_rate": 0.0,
                        "wrong_gen_answer": best_wrong["gen_answer"][:100],
                        "wrong_num_steps": count_steps(best_wrong["think_content"]),
                    },
                }
                paired_data.append(paired_item)
                stats["paired_output"] += 1

    return paired_data, stats, raw_outputs_log


# ============================================================
# 数据后处理与分析
# ============================================================

def analyze_and_split(
    paired_data: List[Dict],
    stats: Dict,
    difficulty_bins: int = 5,
) -> Dict:
    """
    分析采样结果，按难度分层

    难度定义: pass_rate 越低 → 题目越难 → 模型越容易犯错 → 训练价值越高

    Returns:
        analysis: 分析结果字典
    """
    analysis = {}

    if not paired_data:
        print("[分析] ⚠️ 无配对数据")
        return analysis

    # 1. Pass rate 分布
    pass_rates = [
        d["rejection_sampling_meta"]["pass_rate"]
        for d in paired_data
    ]

    # 分桶
    bins = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    bin_counts = Counter()
    for pr in pass_rates:
        for i in range(len(bins) - 1):
            if bins[i] <= pr < bins[i + 1] or (i == len(bins) - 2 and pr == bins[i + 1]):
                bin_label = f"{bins[i]:.3f}-{bins[i+1]:.3f}"
                bin_counts[bin_label] += 1
                break

    analysis["pass_rate_distribution"] = dict(bin_counts)

    # 2. 有/无正确 CoT 参照
    has_correct_ref = sum(1 for d in paired_data if "correct_cot" in d)
    no_correct_ref = len(paired_data) - has_correct_ref

    analysis["with_correct_reference"] = has_correct_ref
    analysis["without_correct_reference"] = no_correct_ref

    # 3. 步骤数分布
    wrong_steps = [
        d["rejection_sampling_meta"]["wrong_num_steps"]
        for d in paired_data
    ]
    step_dist = Counter(wrong_steps)
    analysis["wrong_cot_step_distribution"] = dict(sorted(step_dist.items()))

    # 4. 难度分层建议
    # 低 pass_rate (0~0.25): 困难题，模型经常犯错 → Stage 3 训练
    # 中 pass_rate (0.25~0.5): 中等题 → Stage 2 训练
    # 高 pass_rate (0.5~0.875): 简单题，偶尔犯错 → Stage 2 训练
    easy = sum(1 for pr in pass_rates if pr >= 0.5)
    medium = sum(1 for pr in pass_rates if 0.25 <= pr < 0.5)
    hard = sum(1 for pr in pass_rates if pr < 0.25)

    analysis["difficulty_split"] = {
        "easy (pass_rate >= 0.5)": easy,
        "medium (0.25 <= pass_rate < 0.5)": medium,
        "hard (pass_rate < 0.25)": hard,
    }

    return analysis


def build_mixed_training_data(
    paired_data: List[Dict],
    original_data: List[Dict],
    wrong_cot_ratio: float = 0.5,
    seed: int = 42,
) -> List[Dict]:
    """
    构建混合训练数据: 正确 CoT + 错误 CoT

    Args:
        paired_data: 拒绝采样得到的 (wrong_cot, correct_answer) 配对
        original_data: 原始训练数据 (含正确 CoT)
        wrong_cot_ratio: 错误 CoT 在最终数据中的占比 (默认 50%)
        seed: 随机种子

    Returns:
        mixed_data: 混合后的训练数据
    """
    random.seed(seed)

    # 从原始数据中采样正确 CoT 部分
    num_wrong = len(paired_data)
    num_correct = int(num_wrong * (1 - wrong_cot_ratio) / wrong_cot_ratio)
    num_correct = min(num_correct, len(original_data))

    correct_samples = random.sample(original_data, num_correct)

    # 合并
    mixed = correct_samples + paired_data
    random.shuffle(mixed)

    print(f"\n[混合数据] 构建完成:")
    print(f"  - 正确 CoT: {num_correct} ({num_correct / len(mixed) * 100:.1f}%)")
    print(f"  - 错误 CoT: {num_wrong} ({num_wrong / len(mixed) * 100:.1f}%)")
    print(f"  - 总计: {len(mixed)}")

    return mixed


# ============================================================
# 断点续传支持
# ============================================================

def load_checkpoint(checkpoint_path: str) -> Tuple[List[Dict], int]:
    """加载断点续传的中间结果"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
        print(f"[断点续传] 从 {checkpoint_path} 恢复, 已处理 {ckpt['processed']} 个样本")
        return ckpt["paired_data"], ckpt["processed"]
    return [], 0


def save_checkpoint(checkpoint_path: str, paired_data: List[Dict], processed: int):
    """保存断点续传的中间结果"""
    ckpt = {
        "paired_data": paired_data,
        "processed": processed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(ckpt, f, ensure_ascii=False)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RLD 拒绝采样 — 收集错误 CoT + 正确答案配对数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python scripts/rejection_sampling.py \\
      --model_path /path/to/Qwen3-VL-8B-Instruct \\
      --input_json data/rld_50k_selected.json \\
      --output_json data/rld_50k_rejection_sampled.json

  # 多 GPU + 更多采样
  python scripts/rejection_sampling.py \\
      --model_path /path/to/Qwen3-VL-8B-Instruct \\
      --input_json data/rld_50k_selected.json \\
      --output_json data/rld_50k_rejection_sampled.json \\
      --tensor_parallel_size 4 \\
      --num_samples 16 \\
      --batch_size 64

  # 调试模式 (少量样本)
  python scripts/rejection_sampling.py \\
      --model_path /path/to/Qwen3-VL-8B-Instruct \\
      --input_json data/rld_50k_selected.json \\
      --output_json data/rld_debug_rejection.json \\
      --max_samples 100 \\
      --num_samples 4

  # 构建混合训练数据 (50% 正确 + 50% 错误)
  python scripts/rejection_sampling.py \\
      --model_path /path/to/Qwen3-VL-8B-Instruct \\
      --input_json data/rld_50k_selected.json \\
      --output_json data/rld_50k_rejection_sampled.json \\
      --build_mixed \\
      --wrong_cot_ratio 0.5 \\
      --mixed_output_json data/rld_mixed_training.json
        """,
    )

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Qwen3-VL-8B-Instruct 模型路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="tensor parallel GPU 数量")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="vLLM 最大上下文长度 (需要容纳 image tokens + text + generation)")
    parser.add_argument("--max_pixels", type=int, default=602112,
                        help="图片最大像素数, 限制图片分辨率以控制 image tokens (768*28*28=602112)")

    # 数据参数
    parser.add_argument("--input_json", type=str, required=True,
                        help="输入训练数据 JSON 路径")
    parser.add_argument("--output_json", type=str, required=True,
                        help="输出拒绝采样结果 JSON 路径")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多处理的样本数 (调试用, None=全部)")

    # 采样参数
    parser.add_argument("--num_samples", type=int, default=8,
                        help="每个样本的采样次数 (推荐 8~16)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="最大生成 token 数 (控制 CoT 长度, 1024 约够 4-10 步简洁推理)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度 (1.0 保证多样性, 超长输出直接丢弃)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="nucleus sampling 参数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批处理大小")

    # 验证参数
    parser.add_argument("--min_steps", type=int, default=4,
                        help="最少步骤数")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="最多步骤数 (配合 prompt 的 4-10 步约束)")
    parser.add_argument("--max_cot_chars", type=int, default=1500,
                        help="CoT 最大字符数, 超过则判为无效 (配合 max_tokens=1024, 留余量)")

    # 混合数据构建
    parser.add_argument("--build_mixed", action="store_true",
                        help="是否构建混合训练数据 (正确 CoT + 错误 CoT)")
    parser.add_argument("--wrong_cot_ratio", type=float, default=0.5,
                        help="错误 CoT 在混合数据中的占比 (默认 0.5)")
    parser.add_argument("--mixed_output_json", type=str, default=None,
                        help="混合训练数据输出路径 (默认在 output_json 旁边)")

    # 断点续传
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="断点续传目录 (默认在 output_json 同目录)")
    parser.add_argument("--checkpoint_interval", type=int, default=5000,
                        help="每处理多少样本保存一次断点")

    # 诊断: 保存模型原始输出
    parser.add_argument("--save_raw_outputs", action="store_true",
                        help="保存模型原始输出到单独文件 (用于诊断模型输出格式)")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # ---- 1. 加载数据 ----
    print(f"\n[Pipeline] 加载数据: {args.input_json}")
    with open(args.input_json, 'r') as f:
        all_samples = json.load(f)
    print(f"[Pipeline] 共 {len(all_samples)} 个样本")

    # 限制样本数 (调试用)
    if args.max_samples is not None:
        all_samples = all_samples[:args.max_samples]
        print(f"[Pipeline] 限制为前 {args.max_samples} 个样本 (调试模式)")

    # ---- 2. 断点续传检查 ----
    ckpt_dir = args.checkpoint_dir or os.path.dirname(os.path.abspath(args.output_json))
    ckpt_path = os.path.join(ckpt_dir, ".rejection_sampling_checkpoint.json")
    existing_paired, start_idx = load_checkpoint(ckpt_path)

    if start_idx > 0:
        print(f"[Pipeline] 从样本 {start_idx} 继续 (已有 {len(existing_paired)} 个配对)")
        remaining_samples = all_samples[start_idx:]
    else:
        remaining_samples = all_samples

    # ---- 3. 初始化 vLLM 引擎 ----
    engine = VLLMEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_pixels=args.max_pixels,
    )

    t_start = time.time()

    # ---- 4. 分批执行拒绝采样 (支持断点续传) ----
    # 将 remaining_samples 分成大块，每块处理完保存断点
    chunk_size = args.checkpoint_interval
    all_paired = list(existing_paired)

    for chunk_start in range(0, len(remaining_samples), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(remaining_samples))
        chunk = remaining_samples[chunk_start:chunk_end]

        print(f"\n[Pipeline] 处理块 {chunk_start}~{chunk_end} "
              f"(总进度: {start_idx + chunk_end}/{len(all_samples)})")

        paired_chunk, chunk_stats, chunk_raw_outputs = run_rejection_sampling(
            engine=engine,
            samples=chunk,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            max_cot_chars=args.max_cot_chars,
            batch_size=args.batch_size,
            save_raw_outputs=args.save_raw_outputs,
        )

        all_paired.extend(paired_chunk)

        # 保存原始输出诊断文件
        if args.save_raw_outputs and chunk_raw_outputs:
            raw_output_path = args.output_json.replace('.json', '_raw_outputs.json')
            # 追加模式: 如果已有之前 chunk 的数据, 合并
            existing_raw = []
            if os.path.exists(raw_output_path):
                try:
                    with open(raw_output_path, 'r') as f:
                        existing_raw = json.load(f)
                except:
                    existing_raw = []
            existing_raw.extend(chunk_raw_outputs)
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_raw, f, indent=2, ensure_ascii=False)
            print(f"[诊断] 原始输出已保存: {raw_output_path} ({len(existing_raw)} 个样本)")

            # 打印前 3 个样本的原始输出摘要
            if chunk_start == 0 and chunk_raw_outputs:
                # 先打印格式遵循统计汇总
                total_outs = sum(len(e['outputs']) for e in chunk_raw_outputs)
                step_fmt_count = sum(1 for e in chunk_raw_outputs for o in e['outputs'] if o['has_step_format'])
                final_ans_count = sum(1 for e in chunk_raw_outputs for o in e['outputs'] if o['has_final_answer'])
                strict_count = sum(1 for e in chunk_raw_outputs for o in e['outputs'] if o['strict_format_compliance'])
                method_dist = Counter(o['parse_method'] for e in chunk_raw_outputs for o in e['outputs'])

                print(f"\n{'='*70}")
                print(f"[诊断] 格式遵循统计 ({total_outs} 个输出):")
                print(f"  有 Step N: 分步:     {step_fmt_count}/{total_outs} ({step_fmt_count/total_outs*100:.1f}%)")
                print(f"  有 Final Answer::    {final_ans_count}/{total_outs} ({final_ans_count/total_outs*100:.1f}%)")
                print(f"  严格遵循 (两者都有): {strict_count}/{total_outs} ({strict_count/total_outs*100:.1f}%)")
                print(f"  解析策略分布:")
                for method, cnt in method_dist.most_common():
                    print(f"    {method}: {cnt} ({cnt/total_outs*100:.1f}%)")

                print(f"\n{'='*70}")
                print(f"[诊断] 模型原始输出预览 (前 3 个样本):")
                print(f"{'='*70}")
                for i, entry in enumerate(chunk_raw_outputs[:3]):
                    print(f"\n--- 样本 {entry['global_idx']} ---")
                    print(f"  问题: {entry['question'][:100]}...")
                    print(f"  GT答案: {entry['gt_answer']}")
                    for out in entry['outputs'][:2]:  # 只显示前 2 个采样
                        print(f"\n  [采样 {out['sample_idx']}]")
                        print(f"    has_step_format: {out['has_step_format']} ({out['num_steps_found']} steps)")
                        print(f"    has_final_answer: {out['has_final_answer']}")
                        print(f"    strict_compliance: {out['strict_format_compliance']}")
                        print(f"    parse_method: {out['parse_method']}")
                        print(f"    think_valid: {out['think_valid']} ({out['think_valid_reason']})")
                        print(f"    parsed_answer: {out['parsed_answer']}")
                        print(f"    answer_match: {out['answer_match']}")
                        # 显示原始文本的前 500 字符
                        raw_preview = out['raw_text'][:500]
                        print(f"    raw_text (前500字符):")
                        for line in raw_preview.split('\n')[:15]:
                            print(f"      | {line}")
                        if len(out['raw_text']) > 500:
                            print(f"      | ... (截断, 共 {len(out['raw_text'])} 字符)")
                    if len(entry['outputs']) > 2:
                        print(f"  ... 还有 {len(entry['outputs']) - 2} 个采样结果")
                print(f"{'='*70}\n")

        # 保存断点
        processed = start_idx + chunk_end
        save_checkpoint(ckpt_path, all_paired, processed)
        print(f"[Pipeline] 断点已保存: {processed} 个样本已处理, "
              f"{len(all_paired)} 个配对")

    t_total = time.time() - t_start

    # ---- 5. 最终统计 ----
    # 重新计算完整统计
    final_stats = {
        "total_input": len(all_samples),
        "total_paired": len(all_paired),
        "yield_rate": len(all_paired) / max(len(all_samples), 1),
        "with_correct_ref": sum(1 for d in all_paired if "correct_cot" in d),
        "without_correct_ref": sum(1 for d in all_paired if "correct_cot" not in d),
        "total_time_seconds": t_total,
    }

    # 分析
    analysis = analyze_and_split(all_paired, final_stats)

    print(f"\n{'='*70}")
    print(f"[Pipeline] 拒绝采样完成!")
    print(f"  总耗时: {t_total:.1f}s ({t_total/60:.1f}min, {t_total/3600:.2f}h)")
    print(f"  输入样本: {final_stats['total_input']}")
    print(f"  输出配对: {final_stats['total_paired']} "
          f"(产出率: {final_stats['yield_rate']*100:.1f}%)")
    print(f"  有正确 CoT 参照: {final_stats['with_correct_ref']}")
    print(f"  无正确 CoT 参照: {final_stats['without_correct_ref']}")

    if analysis.get("pass_rate_distribution"):
        print(f"\n  Pass Rate 分布:")
        for bin_label, count in sorted(analysis["pass_rate_distribution"].items()):
            bar = "█" * (count * 40 // max(analysis["pass_rate_distribution"].values()))
            print(f"    {bin_label}: {count:>6} {bar}")

    if analysis.get("difficulty_split"):
        print(f"\n  难度分层:")
        for label, count in analysis["difficulty_split"].items():
            print(f"    {label}: {count}")

    if analysis.get("wrong_cot_step_distribution"):
        print(f"\n  错误 CoT 步骤数分布:")
        for steps, count in sorted(analysis["wrong_cot_step_distribution"].items()):
            print(f"    {steps} steps: {count}")

    # ---- 6. 保存结果 ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    # 保存拒绝采样结果
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(all_paired, f, indent=2, ensure_ascii=False)
    file_size_mb = os.path.getsize(args.output_json) / 1024 / 1024
    print(f"\n  ✅ 拒绝采样结果已保存: {args.output_json} ({file_size_mb:.1f}MB)")

    # 保存统计信息
    stats_path = args.output_json.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            "stats": final_stats,
            "analysis": analysis,
            "args": vars(args),
        }, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 统计信息已保存: {stats_path}")

    # ---- 7. 可选: 构建混合训练数据 ----
    if args.build_mixed:
        mixed_output = args.mixed_output_json or args.output_json.replace(
            '.json', '_mixed.json'
        )

        mixed_data = build_mixed_training_data(
            paired_data=all_paired,
            original_data=all_samples,
            wrong_cot_ratio=args.wrong_cot_ratio,
            seed=args.seed,
        )

        with open(mixed_output, 'w', encoding='utf-8') as f:
            json.dump(mixed_data, f, indent=2, ensure_ascii=False)
        mixed_size_mb = os.path.getsize(mixed_output) / 1024 / 1024
        print(f"  ✅ 混合训练数据已保存: {mixed_output} ({mixed_size_mb:.1f}MB)")

        # 打印使用方法
        print(f"\n📝 使用方法:")
        print(f"  修改 configs/rld_train.yaml 中的 train_json 为:")
        print(f'    train_json: "{mixed_output}"')
        print(f"\n  确保 data.py 中 'wrong_cot' 不在 SUPERVISED_THINK_SOURCES 中:")
        print(f"    SUPERVISED_THINK_SOURCES = {{'free_reasoning', 'corrected_free_reasoning', 'dataset_converted'}}")
        print(f"    # 'wrong_cot' 不在其中 → think 块 labels=-100, 只监督 final answer ✅")

    # 清理断点文件
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"\n  🗑️ 断点文件已清理: {ckpt_path}")

    # 打印示例
    if all_paired:
        print(f"\n{'='*70}")
        print(f"📌 配对数据示例:")
        for i, ex in enumerate(all_paired[:3]):
            meta = ex["rejection_sampling_meta"]
            print(f"\n  --- 示例 {i+1} ---")
            print(f"  📝 问题: {ex['question'][:100]}...")
            print(f"  ✅ GT 答案: {ex['answer']}")
            print(f"  ❌ 模型错误答案: {meta['wrong_gen_answer'][:50]}")
            print(f"  📊 Pass Rate: {meta['pass_rate']} "
                  f"({meta['num_correct']}/{meta['num_valid']})")
            print(f"  📋 错误 CoT 步骤数: {meta['wrong_num_steps']}")
            chain = ex['think_chain']
            if len(chain) > 200:
                chain = chain[:200] + "..."
            print(f"  💭 错误 CoT:\n{chain}")
            if "correct_cot" in ex:
                cc = ex["correct_cot"]
                if len(cc) > 200:
                    cc = cc[:200] + "..."
                print(f"  ✅ 正确 CoT:\n{cc}")

    print(f"\n{'='*70}")
    print(f"✅ 完成!")


if __name__ == "__main__":
    main()
