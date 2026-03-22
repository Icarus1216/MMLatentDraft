#!/usr/bin/env python3
"""
RLD 推理链预生成脚本 — 三阶段流水线 (vLLM + Gemini)

完全摒弃"给答案的条件式生成"，采用以下三阶段流水线：

  Stage 1: 自由推理 (Qwen3VL-Thinking, vLLM 加速)
    - 只给 image + question，不给答案
    - 模型完全依靠自身能力推理
    - 推理正确的样本 → 直接使用 (最高质量, think_chain_source="free_reasoning")
    - 推理错误的样本 → 进入 Stage 2

  Stage 2: Gemini 逐步错误定位 (通过公司内部 iChat API 网关调用)
    - 输入: image + question + 有误推理链 + GT answer
    - Gemini 逐 step 审核，输出错误报告 (哪步错了、为什么错、应该怎么改)
    - 所有推理错误的样本都会经过 Gemini 甄误
    - API 网关: http://ichat.woa.com/api/chat_completions (需要 iChat 鉴权)

  Stage 3: 纠正式重新生成 (Qwen3VL-Thinking, vLLM 加速)
    - 输入: image + question + Gemini 的错误报告 (不给答案!)
    - 模型参考错误报告，用自己的语言风格重新生成推理链
    - 生成的推理链天然包含"仔细验证、自我检查"的语义模式
    - 答案正确: think_chain_source="corrected_free_reasoning" (think+answer 都参与 loss)
    - 答案仍错: think_chain_source="corrected_incorrect" (仅 answer loss)

核心设计理念:
  - 自由推理正确的样本: seg_hidden 分布与推理时完全一致
  - 纠正式推理的样本: 包含"犯错→识别→纠正"的语义模式
  - 两者混合训练，Controller 既学会处理正确推理，也学会处理纠错推理
  - 完全没有"看着答案写"的条件式生成，杜绝答案泄漏和确认式偏差

使用方法:
    # Stage 1 only (自由推理，不调用 Gemini)
    python scripts/pregenerate_think.py \\
        --model_path /path/to/Qwen3-VL-8B-Thinking \\
        --input_json data/rld_train.json \\
        --output_json data/rld_train_with_think.json

    # 完整三阶段流水线 (通过公司 iChat API 网关调用 Gemini)
    python scripts/pregenerate_think.py \\
        --model_path /path/to/Qwen3-VL-8B-Thinking \\
        --input_json data/rld_train.json \\
        --output_json data/rld_train_with_think.json \\
        --ichat_source YOUR_RTX \\
        --ichat_appid YOUR_APPID \\
        --ichat_appkey YOUR_APPKEY \\
        --gemini_model gemini-2.5-flash

    # 多 GPU + 自定义参数
    python scripts/pregenerate_think.py \\
        --model_path /path/to/Qwen3-VL-8B-Thinking \\
        --input_json data/rld_train.json \\
        --output_json data/rld_train_with_think.json \\
        --tensor_parallel_size 4 \\
        --batch_size 64 \\
        --ichat_source YOUR_RTX \\
        --ichat_appid YOUR_APPID \\
        --ichat_appkey YOUR_APPKEY \\
        --gemini_model gemini-2.5-flash \\
        --gemini_max_concurrent 8
"""

import argparse
import base64
import hashlib
import hmac
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from tqdm import tqdm

# vLLM 导入
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("[预生成] ❌ vLLM 未安装，请运行: pip install vllm")
    sys.exit(1)

from PIL import Image


# ============================================================
# System Prompts
# ============================================================

# Stage 1: 自由推理 (与训练时的 RLD_SYSTEM_PROMPT 完全一致)
# Qwen3-VL 的 thinking 模式是内建的，apply_chat_template 会自动输出 <think>\n，
# 因此 system prompt 不需要教模型 <think>/<\/think> 格式，只需教 </step> 分步规范。
FREE_REASONING_SYSTEM_PROMPT = """You are a visual reasoning assistant. When thinking through problems, break your reasoning into clear steps separated by </step>.

For each distinct observation, calculation, or deduction, end that step with </step> before moving on.

Example:
I can see the triangle has a base of 6 units and a height of 8 units.
</step>
Using the formula: Area = 0.5 × base × height = 0.5 × 6 × 8 = 24.
</step>
"""


# Stage 2: Gemini 错误定位 prompt
GEMINI_ERROR_LOCALIZATION_PROMPT = """You are an expert reasoning auditor. You will be given:
1. An image
2. A question about the image
3. A reasoning chain (with step boundaries marked by </step>) that led to an INCORRECT answer
4. The correct answer

Your task is to audit each reasoning step and produce an error report.

For each step, determine:
- ✅ CORRECT: The observation/deduction is accurate
- ❌ ERROR: The step contains a mistake

For EACH error step, provide:
1. The step number
2. What went wrong (brief description)
3. What should have been observed/deduced instead (brief correction hint)

Output format (strict JSON):
{
  "error_steps": [
    {
      "step_number": 3,
      "error_description": "Misread the height as 10 instead of 8",
      "correction_hint": "The perpendicular height label clearly shows 8 units"
    }
  ],
  "first_error_step": 3,
  "summary": "One sentence summary of what went wrong overall"
}

Be concise and precise. Focus on factual errors grounded in the image."""


# Stage 3: 纠正式重新生成 (不给答案！只给错误报告)
CORRECTION_SYSTEM_PROMPT = """You are a careful visual reasoning assistant. You have been informed that a previous reasoning attempt about this image contained errors.

You will receive:
1. An image
2. A question about the image
3. An error report from a previous attempt (describing what went wrong)

Your task is to generate a NEW, COMPLETE reasoning chain from scratch, being especially careful about the aspects that were previously wrong.

Critical instructions:
- Reason in a FORWARD, EXPLORATORY style — as if solving the problem for the first time
- When you encounter aspects that were previously mistaken, naturally express careful verification
  (e.g., "Let me look closely at...", "I need to verify this carefully...", "Checking the label...")
- Do NOT say "the previous attempt was wrong" or directly reference the error report
- Your reasoning should read as genuine, careful exploration with natural self-checking
- Break your reasoning into clear steps separated by </step>
- After finishing your reasoning, output your final answer directly"""


# ============================================================
# iChat API 网关鉴权 (公司内部访问 Gemini/GPT)
# ============================================================

class IChatAuth:
    """公司内部 iChat API 网关鉴权封装

    通过 http://ichat.woa.com/api 网关访问 Gemini / GPT 等外部闭源大模型。
    鉴权方式: HMAC-SHA256 签名 (x-timestamp + x-source)
    """

    BASE_URL = "http://ichat.woa.com/api"

    def __init__(self, source: str, appid: str, appkey: str):
        """
        Args:
            source: 本人 RTX 账号
            appid:  从文档获取的 app_id
            appkey: 从文档获取的 app_key
        """
        self.source = source
        self.appid = appid
        self.appkey = appkey

    def _calc_authorization(self) -> tuple:
        """计算 HMAC-SHA256 签名"""
        timestamp = int(time.time())
        sign_str = "x-timestamp: %s\nx-source: %s" % (timestamp, self.source)
        sign = hmac.new(
            self.appkey.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return sign.hex(), timestamp

    def get_auth_headers(self) -> dict:
        """获取带鉴权信息的 HTTP headers"""
        auth, timestamp = self._calc_authorization()
        return {
            "X-AppID": self.appid,
            "X-Source": self.source,
            "X-Timestamp": str(timestamp),
            "X-Authorization": auth,
        }


def _image_to_base64_url(image_path: str) -> Optional[str]:
    """将本地图像转换为 base64 data URL (用于 iChat API 多模态请求)"""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        # 限制图像尺寸，避免 base64 过大
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"[iChat] ⚠️ 图像转 base64 失败 {image_path}: {e}")
        return None


# ============================================================
# 解析与验证工具函数
# ============================================================

def parse_think_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从模型自由推理输出中解析 <think>...</think> 块和 final answer

    Returns:
        (think_content, answer): think 块内容和 final answer
    """
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        # 兜底: <think> 后无闭合
        fallback = re.search(r'<think>(.*)', text, re.DOTALL)
        if fallback:
            content = fallback.group(1).strip()
            if content and '</step>' in content:
                return content, None
        return None, None

    think_content = match.group(1).strip()
    answer = text[match.end():].strip()
    return think_content, answer


def parse_think_chain_only(text: str) -> Optional[str]:
    """从模型输出中解析 <think>...</think> 块（不关心 answer）"""
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    fallback = re.search(r'<think>(.*)', text, re.DOTALL)
    if fallback:
        content = fallback.group(1).strip()
        if content and '</step>' in content:
            return content
    return None


def validate_think_chain(
    think_content: str,
    min_steps: int = 2,
    max_steps: int = 10,
    min_chars_per_step: int = 10,
) -> Tuple[bool, str]:
    """验证推理链格式合规性"""
    if not think_content or not think_content.strip():
        return False, "think 内容为空"

    num_step_boundaries = think_content.count("</step>")

    if num_step_boundaries < min_steps:
        return False, f"step 数量不足: {num_step_boundaries} < {min_steps}"
    if num_step_boundaries > max_steps:
        return False, f"step 数量过多: {num_step_boundaries} > {max_steps}"

    non_empty_steps = [s.strip() for s in think_content.split("</step>") if s.strip()]
    for i, step in enumerate(non_empty_steps):
        if len(step) < min_chars_per_step:
            return False, f"step {i+1} 内容过短: {len(step)} < {min_chars_per_step}"

    return True, "合规"


def normalize_think_block(think_content: str) -> str:
    """规范化 think 块，确保与 RLDDataset 格式一致"""
    parts = think_content.split("</step>")
    steps = [p.strip() for p in parts if p.strip()]
    think_content_normalized = "\n</step>\n".join(steps)
    return f"<think>\n{think_content_normalized}\n</step>\n</think>\n"


def check_answer_match(generated_answer: str, gt_answer: str) -> bool:
    """
    宽松检查模型生成的 answer 是否与 GT answer 一致

    策略: 归一化后做子串匹配 (宽松)
    """
    if not generated_answer or not gt_answer:
        return False

    def normalize(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # 去标点
        text = re.sub(r'\s+', ' ', text)      # 合并空白
        return text

    gen_norm = normalize(generated_answer)
    gt_norm = normalize(gt_answer)

    if not gen_norm or not gt_norm:
        return False

    # 完全匹配
    if gen_norm == gt_norm:
        return True

    # 子串匹配 (GT 出现在生成答案中，或反过来)
    if gt_norm in gen_norm or gen_norm in gt_norm:
        return True

    # 数值匹配: 尝试提取数字
    gen_nums = re.findall(r'[\d.]+', gen_norm)
    gt_nums = re.findall(r'[\d.]+', gt_norm)
    if gen_nums and gt_nums:
        try:
            gen_val = float(gen_nums[-1])
            gt_val = float(gt_nums[-1])
            if abs(gen_val - gt_val) < 1e-6:
                return True
        except ValueError:
            pass

    return False


def get_step_list(think_content: str) -> List[str]:
    """将 think_content 按 </step> 分割为 step 列表"""
    parts = think_content.split("</step>")
    return [p.strip() for p in parts if p.strip()]


# ============================================================
# vLLM 引擎封装
# ============================================================

class VLLMEngine:
    """vLLM 引擎封装，支持多模态推理"""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
    ):
        print(f"[vLLM] 初始化引擎...")
        print(f"  - 模型: {model_path}")
        print(f"  - Tensor Parallel: {tensor_parallel_size} GPU(s)")
        print(f"  - 最大上下文: {max_model_len}")

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
            except Exception as e:
                print(f"[vLLM] ⚠️ 无法加载图像 {image_path}: {e}")

        return prompt, pil_image

    def generate(
        self,
        prompts: List[str],
        images: List[Optional[Image.Image]],
        sampling_params: SamplingParams,
    ) -> List[str]:
        """批量生成"""
        vllm_inputs = []
        for prompt, image in zip(prompts, images):
            item = {"prompt": prompt}
            if image is not None:
                item["multi_modal_data"] = {"image": image}
            vllm_inputs.append(item)

        outputs = self.llm.generate(vllm_inputs, sampling_params)
        return [o.outputs[0].text for o in outputs]


# ============================================================
# Stage 1: 自由推理
# ============================================================

def stage1_free_reasoning(
    engine: VLLMEngine,
    samples: List[Dict],
    max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_steps: int = 2,
    max_steps: int = 10,
    batch_size: int = 32,
) -> Tuple[List[Dict], List[int]]:
    """
    Stage 1: 全量自由推理

    对所有样本执行不给答案的自由推理，根据答案一致性分为:
      - correct: 推理正确 → 直接使用
      - incorrect: 推理错误 → 进入 Stage 2

    Returns:
        (results, incorrect_indices): 结果列表和推理错误的样本索引
    """
    print(f"\n{'='*60}")
    print(f"[Stage 1] 自由推理 — 全量 {len(samples)} 个样本")
    print(f"  - 模式: image + question → 模型自由推理 (不给答案)")
    print(f"  - 温度: {temperature}, max_tokens: {max_tokens}")
    print(f"{'='*60}")

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    results = [None] * len(samples)
    incorrect_indices = []
    total = len(samples)

    stats = {"total": total, "correct": 0, "incorrect": 0,
             "invalid_format": 0, "image_error": 0}

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = samples[batch_start:batch_end]

        print(f"\n[Stage 1] 批次 {batch_start//batch_size + 1}/"
              f"{(total + batch_size - 1)//batch_size} "
              f"({batch_start+1}-{batch_end}/{total})")

        prompts, images_list, valid_indices = [], [], []

        for i, sample in enumerate(batch):
            global_idx = batch_start + i
            image_path = sample.get("image", "")
            question = sample.get("question", "")

            prompt, pil_image = engine.build_chat_prompt(
                FREE_REASONING_SYSTEM_PROMPT, image_path, question
            )

            if pil_image is None and image_path:
                stats["image_error"] += 1
                results[global_idx] = {
                    **sample,
                    "think_chain": None,
                    "think_chain_status": "image_error",
                    "think_chain_source": "failed",
                }
                continue

            prompts.append(prompt)
            images_list.append(pil_image)
            valid_indices.append(global_idx)

        if not prompts:
            continue

        t0 = time.time()
        generated_texts = engine.generate(prompts, images_list, sampling_params)
        dt = time.time() - t0
        print(f"[Stage 1] 批次完成: {dt:.1f}s")

        for j, text in enumerate(generated_texts):
            global_idx = valid_indices[j]
            sample = samples[global_idx]
            gt_answer = sample.get("answer", "")

            # 解析
            think_content, gen_answer = parse_think_and_answer(text)

            if think_content is None:
                stats["invalid_format"] += 1
                # 格式异常的也算 incorrect，进入 Stage 3 fallback
                incorrect_indices.append(global_idx)
                results[global_idx] = {
                    **sample,
                    "think_chain": None,
                    "think_chain_status": "stage1_invalid_format",
                    "think_chain_source": "failed",
                    "_stage1_raw_output": text[:500],  # 保留原始输出用于调试
                }
                continue

            # 验证格式
            is_valid, reason = validate_think_chain(
                think_content, min_steps=min_steps, max_steps=max_steps
            )

            if not is_valid:
                stats["invalid_format"] += 1
                incorrect_indices.append(global_idx)
                results[global_idx] = {
                    **sample,
                    "think_chain": None,
                    "think_chain_status": f"stage1_invalid_steps: {reason}",
                    "think_chain_source": "failed",
                    "_stage1_raw_output": text[:500],
                }
                continue

            # 检查答案正确性
            is_correct = check_answer_match(gen_answer or "", gt_answer)

            if is_correct:
                # ✅ 自由推理正确 → 直接使用 (最高质量!)
                think_block = normalize_think_block(think_content)
                stats["correct"] += 1
                results[global_idx] = {
                    **sample,
                    "think_chain": think_block,
                    "think_chain_status": "valid",
                    "think_chain_source": "free_reasoning",
                }
            else:
                # ❌ 推理错误 → 进入 Stage 2
                stats["incorrect"] += 1
                incorrect_indices.append(global_idx)
                results[global_idx] = {
                    **sample,
                    "think_chain": None,
                    "think_chain_status": "stage1_incorrect",
                    "think_chain_source": "pending",
                    "_stage1_think_content": think_content,
                    "_stage1_gen_answer": gen_answer,
                }

    print(f"\n[Stage 1] 统计:")
    print(f"  ✅ 推理正确: {stats['correct']} ({stats['correct']/max(total,1)*100:.1f}%)")
    print(f"  ❌ 推理错误: {stats['incorrect']} → 进入 Stage 2")
    print(f"  ❌ 格式异常: {stats['invalid_format']}")
    print(f"  ❌ 图像错误: {stats['image_error']}")

    return results, incorrect_indices


# ============================================================
# Stage 2: Gemini 逐步错误定位
# ============================================================

def _call_gemini_single(
    image_path: str,
    question: str,
    think_content: str,
    gt_answer: str,
    ichat_auth: IChatAuth,
    model: str = "gemini-2.5-flash",
) -> Optional[Dict]:
    """通过公司 iChat API 网关调用 Gemini 对单个样本进行逐步错误定位

    使用 http://ichat.woa.com/api/chat_completions 接口，
    支持多模态 (image_url + text) 输入。
    """
    # 构造 step 列表
    steps = get_step_list(think_content)
    numbered_steps = "\n".join(
        f"Step {i+1}: {step}\n</step>" for i, step in enumerate(steps)
    )

    user_text = (
        f"Question: {question}\n\n"
        f"Reasoning chain (INCORRECT — led to wrong answer):\n"
        f"<think>\n{numbered_steps}\n</think>\n\n"
        f"Correct answer: {gt_answer}\n\n"
        f"Please audit each step and produce the error report in JSON format."
    )

    # 构造 user message content (多模态: 图像 + 文本)
    user_content = []

    # 附加图像 (转 base64 data URL)
    img_b64_url = _image_to_base64_url(image_path)
    if img_b64_url:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": img_b64_url, "detail": "high"},
        })

    user_content.append({"type": "text", "text": user_text})

    # 构造 iChat API 请求参数
    params = {
        "model": model,
        "cid": ichat_auth.source,  # RTX 账号
        "temperature": 0.2,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "system",
                "content": GEMINI_ERROR_LOCALIZATION_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    }

    try:
        url = f"{IChatAuth.BASE_URL}/chat_completions"
        headers = ichat_auth.get_auth_headers()
        response = requests.post(
            url, json=params, timeout=150, headers=headers, stream=False
        )

        if response.status_code != 200:
            print(f"[Stage 2] iChat API 返回 {response.status_code}: {response.text[:200]}")
            return None

        resp_data = response.json()

        # iChat API 返回格式兼容 OpenAI chat completions
        response_text = ""
        if "choices" in resp_data and resp_data["choices"]:
            msg = resp_data["choices"][0].get("message", {})
            response_text = msg.get("content", "").strip()
        elif "content" in resp_data:
            response_text = resp_data["content"].strip()
        else:
            print(f"[Stage 2] iChat API 返回格式异常: {json.dumps(resp_data)[:300]}")
            return None

        if not response_text:
            print(f"[Stage 2] iChat API 返回空内容")
            return None

        # 解析 JSON (可能被 markdown 代码块包裹)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            error_report = json.loads(json_match.group())
            return error_report
        else:
            return {"summary": response_text, "error_steps": [], "first_error_step": -1}

    except requests.exceptions.Timeout:
        print(f"[Stage 2] iChat API 超时")
        return None
    except json.JSONDecodeError as e:
        print(f"[Stage 2] JSON 解析失败: {e}")
        return None
    except Exception as e:
        print(f"[Stage 2] iChat API 调用失败: {e}")
        return None


def stage2_gemini_error_localization(
    samples: List[Dict],
    results: List[Dict],
    incorrect_indices: List[int],
    ichat_auth: IChatAuth,
    gemini_model: str = "gemini-2.5-flash",
    max_concurrent: int = 4,
) -> Dict[int, Dict]:
    """
    Stage 2: 对推理错误的样本调用 Gemini 进行逐步错误定位

    Returns:
        error_reports: {global_idx: error_report_dict}
    """
    # 筛选出有有效 think_content 的错误样本
    valid_incorrect = [
        idx for idx in incorrect_indices
        if results[idx] and results[idx].get("_stage1_think_content")
    ]

    print(f"\n{'='*60}")
    print(f"[Stage 2] Gemini 逐步错误定位")
    print(f"  - 待审核样本: {len(valid_incorrect)}")
    print(f"  - Gemini 模型: {gemini_model}")
    print(f"  - 并发数: {max_concurrent}")
    print(f"{'='*60}")

    if not valid_incorrect:
        print("[Stage 2] 无需审核的样本，跳过")
        return {}

    error_reports = {}
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for idx in valid_incorrect:
            sample = samples[idx]
            result = results[idx]
            future = executor.submit(
                _call_gemini_single,
                image_path=sample.get("image", ""),
                question=sample.get("question", ""),
                think_content=result["_stage1_think_content"],
                gt_answer=sample.get("answer", ""),
                ichat_auth=ichat_auth,
                model=gemini_model,
            )
            futures[future] = idx

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="[Stage 2] Gemini 审核"):
            idx = futures[future]
            try:
                report = future.result()
                if report:
                    error_reports[idx] = report
                    success_count += 1
            except Exception as e:
                print(f"[Stage 2] 样本 {idx} 异常: {e}")

    print(f"\n[Stage 2] 统计:")
    print(f"  ✅ 成功获取错误报告: {success_count}")
    print(f"  ❌ 失败/跳过: {len(valid_incorrect) - success_count}")

    return error_reports


# ============================================================
# Stage 3: 纠正式重新生成
# ============================================================

def stage3_correction_regeneration(
    engine: VLLMEngine,
    samples: List[Dict],
    results: List[Dict],
    incorrect_indices: List[int],
    error_reports: Dict[int, Dict],
    max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_steps: int = 2,
    max_steps: int = 10,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Stage 3: 纠正式重新生成

    对推理错误的样本，基于 Gemini 的错误报告进行纠正式重新生成。
    不给答案！只给错误报告，让模型用自己的风格重新推理。

    所有推理错误的样本在 Stage 2 中都会经过 Gemini 甄误，
    因此每个样本都应有对应的错误报告。没有报告的样本视为失败。
    """
    print(f"\n{'='*60}")
    print(f"[Stage 3] 纠正式重新生成")
    print(f"  - 待重新生成: {len(incorrect_indices)}")
    print(f"  - 有 Gemini 错误报告: {len(error_reports)}")
    no_report_count = len(incorrect_indices) - len(error_reports)
    if no_report_count > 0:
        print(f"  - ⚠️ 缺少 Gemini 报告 (将标记为失败): {no_report_count}")
    print(f"{'='*60}")

    if not incorrect_indices:
        print("[Stage 3] 无需重新生成的样本，跳过")
        return results

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    stats = {"total": len(incorrect_indices), "valid": 0, "valid_correct": 0,
             "valid_incorrect": 0, "invalid": 0}

    for batch_start in range(0, len(incorrect_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(incorrect_indices))
        batch_indices = incorrect_indices[batch_start:batch_end]

        print(f"\n[Stage 3] 批次 {batch_start//batch_size + 1}/"
              f"{(len(incorrect_indices) + batch_size - 1)//batch_size}")

        prompts, images_list, valid_gen_indices = [], [], []

        for global_idx in batch_indices:
            sample = samples[global_idx]
            image_path = sample.get("image", "")
            question = sample.get("question", "")

            # 构造 user_text: 基于 Gemini 错误报告
            if global_idx not in error_reports:
                # 没有 Gemini 报告 → 直接标记为失败，跳过生成
                results[global_idx] = {
                    **sample,
                    "think_chain": None,
                    "think_chain_status": "stage3_no_error_report",
                    "think_chain_source": "failed",
                }
                stats["invalid"] += 1
                continue

            report = error_reports[global_idx]
            summary = report.get("summary", "Unknown errors in the reasoning")
            error_details = ""
            for err in report.get("error_steps", []):
                step_num = err.get("step_number", "?")
                desc = err.get("error_description", "")
                hint = err.get("correction_hint", "")
                error_details += f"  - Step {step_num}: {desc}"
                if hint:
                    error_details += f" (Hint: {hint})"
                error_details += "\n"

            user_text = (
                f"{question}\n\n"
                f"⚠️ A previous reasoning attempt had the following issues:\n"
                f"Summary: {summary}\n"
                f"{error_details}\n"
                f"Please reason through this problem carefully from scratch, "
                f"paying special attention to the aspects that were previously wrong. "
                f"Be extra careful when observing details in the image."
            )
            system_prompt = CORRECTION_SYSTEM_PROMPT

            prompt, pil_image = engine.build_chat_prompt(
                system_prompt, image_path, user_text
            )

            if pil_image is None and image_path:
                continue

            prompts.append(prompt)
            images_list.append(pil_image)
            valid_gen_indices.append(global_idx)

        if not prompts:
            continue

        t0 = time.time()
        generated_texts = engine.generate(prompts, images_list, sampling_params)
        dt = time.time() - t0
        print(f"[Stage 3] 批次完成: {dt:.1f}s")

        for j, text in enumerate(generated_texts):
            global_idx = valid_gen_indices[j]
            sample = samples[global_idx]

            think_content, gen_answer = parse_think_and_answer(text)

            if think_content is None:
                stats["invalid"] += 1
                results[global_idx] = {
                    **sample,
                    "think_chain": None,
                    "think_chain_status": "stage3_invalid_format",
                    "think_chain_source": "failed",
                }
                continue

            is_valid, reason = validate_think_chain(
                think_content, min_steps=min_steps, max_steps=max_steps
            )

            if not is_valid:
                stats["invalid"] += 1
                results[global_idx] = {
                    **sample,
                    "think_chain": None,
                    "think_chain_status": f"stage3_invalid_steps: {reason}",
                    "think_chain_source": "failed",
                }
                continue

            # ✅ 格式合规 → 进一步校验答案正确性 (分级信任)
            think_block = normalize_think_block(think_content)
            gt_answer = sample.get("answer", "")
            is_answer_correct = check_answer_match(gen_answer or "", gt_answer)

            # 所有走到这里的样本都有 Gemini 错误报告
            source = ("corrected_free_reasoning" if is_answer_correct
                      else "corrected_incorrect")

            stats["valid"] += 1
            if is_answer_correct:
                stats["valid_correct"] += 1
            else:
                stats["valid_incorrect"] += 1

            results[global_idx] = {
                **sample,
                "think_chain": think_block,
                "think_chain_status": "valid",
                "think_chain_source": source,
            }

    print(f"\n[Stage 3] 统计:")
    print(f"  ✅ 有效: {stats['valid']}")
    print(f"    ├── 答案正确: {stats['valid_correct']} → think 块参与 loss")
    print(f"    └── 答案仍错: {stats['valid_incorrect']} → think 块 labels=-100 (仅 answer loss)")
    print(f"  ❌ 无效: {stats['invalid']}")

    # 最终处理：仍然失败的样本标记为 fallback
    for idx in incorrect_indices:
        if results[idx] is None or results[idx].get("think_chain_source") in ("pending", "failed"):
            if results[idx] is None:
                results[idx] = {**samples[idx]}
            results[idx].update({
                "think_chain": None,
                "think_chain_status": "all_stages_failed",
                "think_chain_source": "fabricated",
            })

    return results


# ============================================================
# 清理输出 (移除内部调试字段)
# ============================================================

def clean_output(results: List[Dict]) -> List[Dict]:
    """移除以 _ 开头的内部调试字段"""
    cleaned = []
    for item in results:
        cleaned_item = {k: v for k, v in item.items() if not k.startswith("_")}
        cleaned.append(cleaned_item)
    return cleaned


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RLD 推理链预生成 — 三阶段流水线 (vLLM + Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
三阶段流水线:
  Stage 1: 自由推理 (vLLM) — 不给答案，依靠模型能力
  Stage 2: Gemini 错误定位 — 对推理错误样本逐步审核
  Stage 3: 纠正式重新生成 (vLLM) — 基于错误报告重新推理

示例:
  # Stage 1 only (不调用 Gemini)
  python scripts/pregenerate_think.py \\
      --model_path /path/to/Qwen3-VL-8B-Thinking \\
      --input_json data/rld_train.json \\
      --output_json data/rld_train_with_think.json

  # 完整三阶段 (通过公司 iChat API 网关调用 Gemini)
  python scripts/pregenerate_think.py \\
      --model_path /path/to/Qwen3-VL-8B-Thinking \\
      --input_json data/rld_train.json \\
      --output_json data/rld_train_with_think.json \\
      --ichat_source YOUR_RTX \\
      --ichat_appid YOUR_APPID \\
      --ichat_appkey YOUR_APPKEY \\
      --gemini_model gemini-2.5-flash
        """,
    )

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Qwen3-VL-Thinking 模型路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="tensor parallel GPU 数量")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)

    # 数据参数
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)

    # 生成参数
    parser.add_argument("--max_think_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=32)

    # 验证参数
    parser.add_argument("--min_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=25)

    # iChat API 网关参数 (公司内部调用 Gemini/GPT, 可选，不提供则跳过 Stage 2)
    parser.add_argument("--ichat_source", type=str, default=None,
                        help="iChat 鉴权: 本人 RTX 账号 (不提供则跳过 Stage 2)")
    parser.add_argument("--ichat_appid", type=str, default=None,
                        help="iChat 鉴权: app_id")
    parser.add_argument("--ichat_appkey", type=str, default=None,
                        help="iChat 鉴权: app_key")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.5-flash",
                        help="通过 iChat 调用的模型名 (默认 gemini-2.5-flash)")
    parser.add_argument("--gemini_max_concurrent", type=int, default=4,
                        help="iChat API 最大并发数")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ---- 1. 加载数据 ----
    print(f"[Pipeline] 加载数据: {args.input_json}")
    with open(args.input_json, 'r') as f:
        samples = json.load(f)
    print(f"[Pipeline] 共 {len(samples)} 个样本")

    if samples:
        s0 = samples[0]
        for key in ["question", "answer"]:
            if key not in s0:
                print(f"[Pipeline] ❌ 数据缺少必要字段 '{key}'")
                sys.exit(1)

    # ---- 2. 初始化 vLLM 引擎 ----
    engine = VLLMEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    t_pipeline_start = time.time()

    # ---- 3. Stage 1: 自由推理 ----
    results, incorrect_indices = stage1_free_reasoning(
        engine=engine,
        samples=samples,
        max_tokens=args.max_think_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )

    # ---- 4. Stage 2: Gemini 错误定位 (通过 iChat API 网关, 可选) ----
    error_reports = {}
    ichat_ready = (args.ichat_source and args.ichat_appid and args.ichat_appkey)
    if ichat_ready and incorrect_indices:
        ichat_auth = IChatAuth(
            source=args.ichat_source,
            appid=args.ichat_appid,
            appkey=args.ichat_appkey,
        )
        print(f"\n[Pipeline] iChat 鉴权: RTX={args.ichat_source}, "
              f"AppID={args.ichat_appid[:8]}...")
        error_reports = stage2_gemini_error_localization(
            samples=samples,
            results=results,
            incorrect_indices=incorrect_indices,
            ichat_auth=ichat_auth,
            gemini_model=args.gemini_model,
            max_concurrent=args.gemini_max_concurrent,
        )
    elif incorrect_indices:
        if not ichat_ready:
            print(f"\n[Pipeline] ❌ 未提供完整 iChat 鉴权信息 "
                  f"(--ichat_source / --ichat_appid / --ichat_appkey)")
            print(f"  → {len(incorrect_indices)} 个错误样本无法进行 Gemini 甄误，"
                  f"将在 Stage 3 中标记为失败")
            print(f"  → 建议提供 iChat 鉴权信息以启用完整三阶段流水线")

    # ---- 5. Stage 3: 纠正式重新生成 ----
    if incorrect_indices:
        results = stage3_correction_regeneration(
            engine=engine,
            samples=samples,
            results=results,
            incorrect_indices=incorrect_indices,
            error_reports=error_reports,
            max_tokens=args.max_think_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
        )

    t_total = time.time() - t_pipeline_start

    # ---- 6. 清理并保存 ----
    output = clean_output(results)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ---- 7. 最终统计 ----
    source_counts = {}
    for item in output:
        src = item.get("think_chain_source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    total_valid = sum(1 for item in output if item.get("think_chain_status") == "valid")

    print(f"\n{'='*60}")
    print(f"[Pipeline] 三阶段流水线完成!")
    print(f"  总耗时: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"  总样本: {len(samples)}")
    print(f"  有效推理链: {total_valid} ({total_valid/max(len(samples),1)*100:.1f}%)")
    print(f"\n  按来源统计:")
    for src, cnt in sorted(source_counts.items()):
        pct = cnt / max(len(samples), 1) * 100
        emoji = "✅" if src in ("free_reasoning", "corrected_free_reasoning") else "⚠️" if src == "corrected_incorrect" else "❌"
        print(f"    {emoji} {src}: {cnt} ({pct:.1f}%)")
    print(f"\n  ✅ 结果已保存到: {args.output_json}")
    print(f"{'='*60}")

    # 打印示例
    for src in ["free_reasoning", "corrected_free_reasoning", "corrected_incorrect"]:
        examples = [s for s in output if s.get("think_chain_source") == src and s.get("think_chain")]
        if examples:
            ex = examples[0]
            print(f"\n📌 示例 [{src}]:")
            print(f"   📝 问题: {ex['question'][:80]}...")
            print(f"   📋 GT答案: {ex['answer'][:80]}...")
            chain = ex['think_chain']
            if len(chain) > 300:
                chain = chain[:300] + "..."
            print(f"   💭 推理链:\n{chain}")
            print(f"   📊 步骤数: {ex['think_chain'].count('</step>')}")


if __name__ == "__main__":
    main()
