#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_erqa_latent_cot.py
===========================
为 ERQA 数据集的每道题目，调用 claude-4.7-opus (通过 iChat 网关) 生成
v6 格式的 LatentDraft 训练数据。

v6 训练格式:
  - reasoning_for_training: 连续自然语言推理链，每个推理子步边界用 <|pause|> 分隔
  - latent_key_tokens: List[List[Dict]]，每个 pause 对应一组 key tokens
  - 字段: image_path, question, answer, reasoning_for_training, latent_key_tokens, task_type

用法:
export OPENAI_API_KEY=...    # OPENAI_BASE_URL 可选 (默认 https://api.openai.com/v1)
    python scripts/generate_erqa_latent_cot.py \
        --erqa_root ./data/erqa \
        --erqa_file ./data/erqa/erqa_test.jsonl \
        --output ./data/erqa/erqa_latent_cot_v2.json \
        --model claude-opus-4-7 \
        --workers 8 \
        --verbose
"""

import argparse
import concurrent.futures as cf
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# 让脚本可独立执行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V3_DIR = os.path.join(_SCRIPT_DIR, "v3")
if _V3_DIR not in sys.path:
    sys.path.insert(0, _V3_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from ichat_client import (
    create_openai_client,
    encode_image_to_base64,
    call_chat_with_image,
    parse_gpt_response,
    build_auth_config_from_env_or_args,
    _save_intermediate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_erqa_latent_cot")


# ============================================================
# 任务类型映射 (question_type -> task_type)
# ============================================================
QUESTION_TYPE_TO_TASK_TYPE = {
    "Spatial Reasoning":     "spatial_relation",
    "Action Reasoning":      "physical",
    "Trajectory Reasoning":  "physical",
    "State Estimation":      "detail_observation",
    "Task Reasoning":        "physical",
    "Multi-view Reasoning":  "perspective",
    "Pointing":              "fine_grained",
    "Other":                 "detail_observation",
}


# ============================================================
# System Prompt (v6 格式)
# ============================================================
SYSTEM_PROMPT = """You are a visual reasoning expert that generates training data for a latent-reasoning VLM.

Given an image, a multiple-choice question, and the correct answer, you must produce a CONTINUOUS natural-language chain-of-thought that:
1. Breaks the reasoning into 2-4 sequential sub-steps
2. Each sub-step boundary is marked by the token PAUSE_MARKER
3. The text flows naturally WITHOUT numbered lists, headings, or bullet points

## Output Format (strict JSON):

{
  "n_substeps": <int 2..4>,
  "task_type": "<one of: perspective | geometry | spatial_relation | physical | counterfactual | fine_grained | counting | comparison | detail_observation>",
  "reasoning_for_training": "<continuous CoT with PAUSE_MARKER between sub-steps>",
  "substeps": [
    {
      "substep_id": 1,
      "k_latent": <int 1..3>,
      "role": "<abstract | concrete | unified>",
      "key_tokens": ["<4-6 short visual-state phrases, 1-4 words each>"]
    },
    ...
  ]
}

## CRITICAL RULES for reasoning_for_training:

The reasoning must follow this exact pattern:
  <intent_statement_1> PAUSE_MARKER <anchor_1 + intent_statement_2> PAUSE_MARKER <anchor_2 + intent_statement_3> PAUSE_MARKER <final_conclusion>

Where:
- intent_statement: Names what needs to be computed WITHOUT giving the answer (e.g. "I need to determine which direction the trajectory curves toward")
- PAUSE_MARKER: The exact string that marks a latent thinking boundary
- anchor: States the SUB-CONCLUSION that was "computed" by the latent step, then transitions to the next sub-step

## Sub-step Design Rules:
- Each sub-step should tackle ONE visual reasoning challenge (spatial relation, trajectory tracing, physical prediction, etc.)
- k_latent=1: simple depth boost (one mental computation)
- k_latent=2: one concept transformation (most common, default to this)
- k_latent=3: complex multi-step transformation (rare, only for truly complex spatial/physical reasoning)
- NOT every sub-step needs to be a bottleneck; some can be simple observations

## Key Tokens Rules:
- Each substep produces 4-6 key tokens (short English phrases, 1-4 words)
- Good tokens: "yellow trajectory arc", "gripper orientation", "top stair surface", "endpoint marker"
- Bad tokens: "analyze", "determine", "look at", "check"
- Tokens should describe VISUAL STATES, not actions

## BLANK CONTRACT:
If you remove all PAUSE_MARKER tokens, the remaining text must still read as a coherent paragraph, but with LOGICAL GAPS — the conclusions after each marker must reference something that wasn't explicitly derived in the visible text.

## Quality:
- Reasoning must be CORRECT and arrive at the given answer
- 150-500 words total
- Write in English, flowing prose (no lists, no headings)
- Final sentence must state the answer clearly

## Example:
For a question about which direction a robot should move:
{
  "n_substeps": 3,
  "task_type": "physical",
  "reasoning_for_training": "I observe the robot gripper holding a marker, and I need to determine its current orientation relative to the target cap. PAUSE_MARKER With that spatial offset established, the gripper needs to translate rightward to align horizontally with the cap position. I now need to verify whether the vertical alignment also requires adjustment. PAUSE_MARKER Given that the cap sits at roughly the same height, only the horizontal correction matters. Therefore, the camera (attached to the gripper) should turn Right, which is answer A.",
  "substeps": [
    {"substep_id": 1, "k_latent": 2, "role": "concrete", "key_tokens": ["marker body center", "cap upper-right", "horizontal offset", "rightward displacement"]},
    {"substep_id": 2, "k_latent": 1, "role": "abstract", "key_tokens": ["vertical alignment check", "same height level", "no vertical correction"]},
    {"substep_id": 3, "k_latent": 1, "role": "unified", "key_tokens": ["horizontal-only correction", "rightward turn", "camera pan direction"]}
  ]
}"""


# ============================================================
# User Prompt Template
# ============================================================
USER_PROMPT_TEMPLATE = """Look at this image carefully.

**Question:** {question}

**Correct Answer:** {answer}

Generate a v6-format chain-of-thought reasoning that arrives at answer "{answer}".
- Break into 2-4 sub-steps separated by PAUSE_MARKER
- Each sub-step should handle one visual reasoning challenge
- Provide key_tokens for each sub-step

Return your response as a JSON object with fields: n_substeps, task_type, reasoning_for_training, substeps."""


# ============================================================
# 后处理: 从 Claude 输出构建 v6 训练样本
# ============================================================

PAUSE_TOKEN = "<|pause|>"

def build_v6_training_sample(
    erqa_item: Dict[str, Any],
    claude_response: Dict[str, Any],
    erqa_root: str,
) -> Optional[Dict[str, Any]]:
    """从 Claude 返回的 JSON 构建 v6 格式训练样本。"""

    reasoning = claude_response.get("reasoning_for_training", "")
    substeps = claude_response.get("substeps", [])
    task_type = claude_response.get("task_type", "")

    if not reasoning or not substeps:
        return None

    # 替换 PAUSE_MARKER -> <|pause|>
    reasoning = reasoning.replace("PAUSE_MARKER", PAUSE_TOKEN)

    # 验证 pause 数量
    pause_count = reasoning.count(PAUSE_TOKEN)
    if pause_count < 1:
        return None
    if pause_count > 5:
        # 截断过多的 pause
        parts = reasoning.split(PAUSE_TOKEN)
        reasoning = PAUSE_TOKEN.join(parts[:5]) + parts[5]

    # 清理多余空白
    reasoning = re.sub(r"\s+", " ", reasoning).strip()

    # 构建 latent_key_tokens (v6 格式: List[List[Dict]])
    latent_key_tokens = []
    for i, substep in enumerate(substeps):
        k_latent = substep.get("k_latent", 2)
        role = substep.get("role", "concrete")
        key_tokens = substep.get("key_tokens", [])

        # 确保 key_tokens 是 4-6 个
        if len(key_tokens) < 3:
            key_tokens = key_tokens + ["visual_state"] * (3 - len(key_tokens))
        if len(key_tokens) > 6:
            key_tokens = key_tokens[:6]

        # 根据 k_latent 决定 stage 结构
        if k_latent == 1:
            # 单 stage (unified)
            boundary = [{
                "tokens": key_tokens,
                "stage_id": 1,
                "role": "unified",
                "from_substep": i + 1,
            }]
        elif k_latent >= 2:
            # 双 stage (abstract + concrete)
            mid = len(key_tokens) // 2
            abstract_tokens = key_tokens[:mid] if mid >= 2 else key_tokens[:2]
            concrete_tokens = key_tokens[mid:] if len(key_tokens) - mid >= 2 else key_tokens[2:]
            boundary = [
                {
                    "tokens": abstract_tokens,
                    "stage_id": 1,
                    "role": "abstract",
                    "from_substep": i + 1,
                },
                {
                    "tokens": concrete_tokens,
                    "stage_id": 2,
                    "role": "concrete",
                    "from_substep": i + 1,
                },
            ]
        else:
            boundary = [{
                "tokens": key_tokens,
                "stage_id": 1,
                "role": "unified",
                "from_substep": i + 1,
            }]

        latent_key_tokens.append(boundary)

    # 确保 latent_key_tokens 数量与 pause 数量一致
    while len(latent_key_tokens) < pause_count:
        latent_key_tokens.append([{
            "tokens": ["visual_reasoning", "spatial_analysis", "conclusion"],
            "stage_id": 1,
            "role": "unified",
            "from_substep": len(latent_key_tokens) + 1,
        }])
    if len(latent_key_tokens) > pause_count:
        latent_key_tokens = latent_key_tokens[:pause_count]

    # 构建图片路径
    image_paths = erqa_item.get("image_paths", [])
    if image_paths:
        image_path = os.path.join(erqa_root, image_paths[0])
    else:
        image_path = ""

    # 如果 task_type 不在合法列表中，从 question_type 映射
    valid_task_types = {"perspective", "geometry", "spatial_relation", "physical",
                        "counterfactual", "fine_grained", "counting", "comparison",
                        "detail_observation", "ocr", "multi_hypothesis"}
    if task_type not in valid_task_types:
        question_type = erqa_item.get("question_type", "Other")
        task_type = QUESTION_TYPE_TO_TASK_TYPE.get(question_type, "detail_observation")

    # 最终训练记录 (v6 slim 格式)
    sample = {
        "image_path": image_path,
        "question": erqa_item["question"],
        "answer": erqa_item["answer"],
        "reasoning_for_training": reasoning,
        "latent_key_tokens": latent_key_tokens,
        "task_type": task_type,
    }
    return sample


# ============================================================
# 单样本处理
# ============================================================

def process_one_sample(
    erqa_item: Dict[str, Any],
    client,
    auth_config: Dict[str, str],
    model: str,
    erqa_root: str,
    temperature: float,
    max_tokens: int,
    num_retries: int,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """处理单条 ERQA 样本，调用 Claude 生成 v6 格式训练数据。"""
    qid = erqa_item["question_id"]
    question = erqa_item["question"]
    answer = erqa_item["answer"]

    # 构建图片路径
    image_paths = erqa_item.get("image_paths", [])
    if not image_paths:
        logger.warning(f"[{qid}] 无图片路径，跳过")
        return None

    image_path = os.path.join(erqa_root, image_paths[0])
    if not os.path.exists(image_path):
        logger.warning(f"[{qid}] 图片不存在: {image_path}")
        return None

    # 编码图片
    image_b64 = encode_image_to_base64(image_path)
    if not image_b64:
        logger.warning(f"[{qid}] 图片编码失败: {image_path}")
        return None

    # 构建 user prompt
    user_text = USER_PROMPT_TEMPLATE.format(question=question, answer=answer)

    # 调用 Claude
    for attempt in range(num_retries):
        raw_response = call_chat_with_image(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_text=user_text,
            image_base64=image_b64,
            auth_config=auth_config,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
            max_retries=3,
        )

        if not raw_response:
            logger.warning(f"[{qid}] attempt {attempt+1}/{num_retries}: API 返回空")
            continue

        # 解析 JSON
        parsed_json = parse_gpt_response(raw_response)
        if not parsed_json:
            logger.warning(f"[{qid}] attempt {attempt+1}/{num_retries}: JSON 解析失败")
            continue

        reasoning = parsed_json.get("reasoning_for_training", "")
        if not reasoning:
            logger.warning(f"[{qid}] attempt {attempt+1}/{num_retries}: 无 reasoning_for_training 字段")
            continue

        # 检查是否包含 PAUSE_MARKER
        if "PAUSE_MARKER" not in reasoning:
            preview = reasoning[:200].replace('\n', ' ')
            logger.warning(f"[{qid}] attempt {attempt+1}/{num_retries}: 无 PAUSE_MARKER. preview: {preview}")
            continue

        # 构建 v6 训练样本
        sample = build_v6_training_sample(erqa_item, parsed_json, erqa_root)
        if not sample:
            logger.warning(f"[{qid}] attempt {attempt+1}/{num_retries}: 构建训练样本失败")
            continue

        if verbose:
            pause_count = sample["reasoning_for_training"].count(PAUSE_TOKEN)
            logger.info(f"  ✅ [{qid}] 成功 | pauses={pause_count} "
                       f"stages={len(sample['latent_key_tokens'])} "
                       f"task={sample['task_type']}")
        return sample

    logger.error(f"[{qid}] 所有 {num_retries} 次重试均失败")
    return None


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ERQA Latent CoT 数据合成 - v6 格式 (claude-4.7-opus)")
    parser.add_argument("--erqa_root", type=str, default="./data/erqa",
                        help="ERQA 数据根目录 (含 images/)")
    parser.add_argument("--erqa_file", type=str, default="./data/erqa/erqa_test.jsonl",
                        help="ERQA 测试数据 jsonl")
    parser.add_argument("--output", type=str, default="./data/erqa/erqa_latent_cot_v2.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--rtx", type=str, default=None,
                        help="兼容参数（已废弃，当前版本不使用）")
    parser.add_argument("--model", type=str, default="claude-opus-4-7",
                        help="iChat 模型名")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发 worker 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="最大生成 token 数")
    parser.add_argument("--num_retries_per_sample", type=int, default=3,
                        help="每样本最大重试次数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最大处理样本数 (0=全量)")
    parser.add_argument("--single_image_only", action="store_true",
                        help="仅处理单图样本")
    parser.add_argument("--multi_image_only", action="store_true",
                        help="仅处理多图样本")
    parser.add_argument("--question_types", nargs="+", default=None,
                        help="仅处理指定 question_type")
    parser.add_argument("--force_data_type", type=str, default=None,
                        help="强制 data_type")
    parser.add_argument("--resume", action="store_true",
                        help="从已有输出文件恢复 (跳过已完成的 question_id)")
    parser.add_argument("--verbose", action="store_true",
                        help="详细日志")
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- 鉴权 ----
    auth_config = build_auth_config_from_env_or_args(rtx=args.rtx)
    if not auth_config:
logger.error("❌ 鉴权失败，请设置环境变量 OPENAI_API_KEY (可选 OPENAI_BASE_URL)")
        sys.exit(1)

    # ---- 加载数据 ----
    if not os.path.exists(args.erqa_file):
        logger.error(f"❌ ERQA 数据文件不存在: {args.erqa_file}")
        sys.exit(1)

    with open(args.erqa_file, "r", encoding="utf-8") as f:
        erqa_data = [json.loads(line.strip()) for line in f if line.strip()]

    logger.info(f"📂 加载 ERQA 数据: {len(erqa_data)} 条 from {args.erqa_file}")

    # ---- 过滤 ----
    if args.single_image_only:
        erqa_data = [d for d in erqa_data if d.get("num_images", 1) == 1]
        logger.info(f"  过滤后 (单图): {len(erqa_data)} 条")
    if args.multi_image_only:
        erqa_data = [d for d in erqa_data if d.get("num_images", 1) > 1]
        logger.info(f"  过滤后 (多图): {len(erqa_data)} 条")
    if args.question_types:
        erqa_data = [d for d in erqa_data if d.get("question_type") in args.question_types]
        logger.info(f"  过滤后 (question_types={args.question_types}): {len(erqa_data)} 条")
    if args.max_samples > 0:
        erqa_data = erqa_data[:args.max_samples]
        logger.info(f"  截断为前 {args.max_samples} 条")

    # ---- Resume ----
    done_ids = set()
    results = []
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            # v6 格式没有 question_id 字段，用 question 做去重
            done_questions = {r["question"] for r in results}
            # 过滤已完成的
            erqa_data_filtered = [d for d in erqa_data if d["question"] not in done_questions]
            logger.info(f"🔄 Resume: 已完成 {len(results)} 条，剩余 {len(erqa_data_filtered)} 条")
            erqa_data = erqa_data_filtered
        except Exception as e:
            logger.warning(f"Resume 加载失败: {e}，从头开始")
            results = []

    pending = erqa_data
    logger.info(f"📋 待处理: {len(pending)} 条")

    if not pending:
        logger.info("✅ 所有样本已完成，无需处理")
        return

    # ---- 创建 client ----
    client = create_openai_client()

    # ---- 并发处理 ----
    total = len(pending)
    success_count = 0
    fail_count = 0
    start_time = time.time()

    # 中间保存间隔
    SAVE_INTERVAL = 10

    logger.info(f"🚀 开始合成 (model={args.model}, workers={args.workers}, temp={args.temperature})")
    logger.info("=" * 60)

    def _worker(item):
        return process_one_sample(
            erqa_item=item,
            client=client,
            auth_config=auth_config,
            model=args.model,
            erqa_root=args.erqa_root,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            num_retries=args.num_retries_per_sample,
            verbose=args.verbose,
        )

    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {executor.submit(_worker, item): item for item in pending}

        for i, future in enumerate(cf.as_completed(future_to_item), 1):
            item = future_to_item[future]
            qid = item["question_id"]
            try:
                sample = future.result()
                if sample:
                    results.append(sample)
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"[{qid}] 异常: {e}")
                fail_count += 1

            # 进度
            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / speed if speed > 0 else 0
            logger.info(
                f"  📊 [{i}/{total}] {elapsed:.0f}s | "
                f"speed={speed:.2f}it/s | ETA={eta:.0f}s | "
                f"✅{success_count} ❌{fail_count} | "
                f"rate={success_count/(success_count+fail_count)*100:.1f}%"
            )

            # 中间保存
            if i % SAVE_INTERVAL == 0:
                _do_save(results, args.output)

    # ---- 最终保存 ----
    _do_save(results, args.output)

    # ---- 统计 ----
    elapsed_total = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ✅ ERQA Latent CoT 合成完成 (v6 格式)")
    logger.info("=" * 60)
    logger.info(f"  总样本:     {total}")
    logger.info(f"  成功:       {success_count}")
    logger.info(f"  失败:       {fail_count}")
    logger.info(f"  成功率:     {success_count/(success_count+fail_count)*100:.1f}%")
    logger.info(f"  耗时:       {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    logger.info(f"  输出:       {args.output}")
    logger.info(f"  总记录数:   {len(results)}")

    # 保存 config
    config_path = args.output.replace(".json", "_config.json")
    config = {
        "data_source": "erqa",
        "format": "v6_slim",
        "model": args.model,
        "num_samples": len(results),
        "total_erqa_samples": total,
        "success_rate": round(success_count / max(1, success_count + fail_count) * 100, 1),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_retries_per_sample": args.num_retries_per_sample,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"  配置:       {config_path}")


def _do_save(results: List[Dict], output_path: str):
    """原子保存结果。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)
    logger.info(f"  💾 已保存 {len(results)} 条 -> {output_path}")


if __name__ == "__main__":
    main()
