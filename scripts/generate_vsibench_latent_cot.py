#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_vsibench_latent_cot.py
================================
为 VSI-Bench 的 direction / route_planning 题型生成 v6 格式训练数据。

关键特点:
  - 多图输入: 每条数据有 16 帧室内扫描视频帧，采样 N 帧传给 Claude
  - 输出 image_paths 为列表 (兼容多图训练)
  - v6 格式: reasoning_for_training + latent_key_tokens

用法:
export OPENAI_API_KEY=...    # OPENAI_BASE_URL 可选
    python scripts/generate_vsibench_latent_cot.py \
        --data_root ./data/VSI-Bench_eval \
        --data_file ./data/VSI-Bench_eval/vsibench_test.jsonl \
        --output ./data/vsibench/vsibench_direction_route_latent_cot.json \
        --model claude-opus-4-7 \
        --workers 6 \
        --max_frames 8 \
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
from typing import Any, Dict, List, Optional

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
logger = logging.getLogger("generate_vsibench_latent_cot")


# ============================================================
# 题型过滤: 只保留 direction 和 route_planning
# ============================================================
TARGET_QUESTION_TYPES = {
    "VSI-Bench/object_rel_direction_easy",
    "VSI-Bench/object_rel_direction_medium",
    "VSI-Bench/object_rel_direction_hard",
    "VSI-Bench/route_planning",
}

# VSI-Bench question_type -> v6 task_type 映射
VSIBENCH_TO_TASK_TYPE = {
    "VSI-Bench/object_rel_direction_easy":   "spatial_relation",
    "VSI-Bench/object_rel_direction_medium": "spatial_relation",
    "VSI-Bench/object_rel_direction_hard":   "perspective",
    "VSI-Bench/route_planning":              "perspective",
}


# ============================================================
# System Prompt (v6 格式, 适配多图室内场景)
# ============================================================
SYSTEM_PROMPT = """You are a visual-spatial reasoning expert that generates training data for a latent-reasoning VLM.

You will be given MULTIPLE FRAMES from an indoor 3D scan video (showing different viewpoints of the same room/space), a spatial reasoning question, and the correct answer. You must produce a CONTINUOUS natural-language chain-of-thought that:
1. Breaks the reasoning into 2-4 sequential sub-steps
2. Each sub-step boundary is marked by the token PAUSE_MARKER
3. The text flows naturally WITHOUT numbered lists, headings, or bullet points

## Output Format (strict JSON):

{
  "n_substeps": <int 2..4>,
  "task_type": "<one of: perspective | spatial_relation>",
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
  <intent_statement_1> PAUSE_MARKER <anchor_1 + intent_statement_2> PAUSE_MARKER ... PAUSE_MARKER <final_conclusion>

Where:
- intent_statement: Names what needs to be computed WITHOUT giving the answer
- PAUSE_MARKER: The exact string that marks a latent thinking boundary
- anchor: States the SUB-CONCLUSION that was "computed" by the latent step, then transitions to the next sub-step

## Spatial Reasoning Strategy:
For DIRECTION questions (left/right/back):
- Sub-step 1: Locate the reference object (where I'm standing) and the facing object in the frames
- Sub-step 2: Mentally reconstruct the 3D layout / floor plan from multiple viewpoints
- Sub-step 3: Determine the target object's position relative to the viewer's facing direction

For ROUTE PLANNING questions:
- Sub-step 1: Identify all landmarks mentioned in the route from the frames
- Sub-step 2: Reconstruct the spatial connectivity (which rooms/objects connect to which)
- Sub-step 3: Trace the path and determine which turns are needed at each waypoint
- Sub-step 4 (if needed): Verify the final direction to reach the destination

## Sub-step Design Rules:
- k_latent=1: simple spatial observation (one object location)
- k_latent=2: mental 3D reconstruction or perspective transformation (most common)
- k_latent=3: complex multi-object spatial layout reasoning (rare)

## Key Tokens Rules:
- Each substep produces 4-6 key tokens (short English phrases, 1-4 words)
- Good tokens: "stove left-wall", "sofa facing-north", "tv behind-right", "floor plan layout", "viewer facing direction"
- Bad tokens: "analyze", "determine", "look at", "check"
- Tokens should describe SPATIAL STATES and OBJECT POSITIONS, not actions

## BLANK CONTRACT:
If you remove all PAUSE_MARKER tokens, the remaining text must still read as a coherent paragraph, but with LOGICAL GAPS — the conclusions after each marker must reference spatial knowledge that wasn't explicitly derived in the visible text.

## Quality:
- Reasoning must be CORRECT and arrive at the given answer
- 150-500 words total
- Write in English, flowing prose (no lists, no headings)
- Final sentence must state the answer clearly (e.g., "the answer is C")
- Ground your reasoning in what can be observed across the multiple frames

## Example (direction question):
{
  "n_substeps": 3,
  "task_type": "spatial_relation",
  "reasoning_for_training": "I need to locate the stove and the sofa in these room frames to establish my standing position and facing direction. PAUSE_MARKER From the frames, the stove is along the kitchen counter on the east wall, and the sofa is in the living area to the south. Standing at the stove and facing the sofa means I am facing roughly south. Now I need to determine where the TV is relative to this orientation. PAUSE_MARKER The TV is mounted on the west wall, which from my south-facing perspective at the stove would be to my right. Therefore the answer is B, right.",
  "substeps": [
    {"substep_id": 1, "k_latent": 2, "role": "concrete", "key_tokens": ["stove east-wall", "sofa south-area", "kitchen counter position", "living room layout"]},
    {"substep_id": 2, "k_latent": 2, "role": "abstract", "key_tokens": ["facing south direction", "viewer orientation", "tv west-wall mount"]},
    {"substep_id": 3, "k_latent": 1, "role": "unified", "key_tokens": ["west relative to south-facing", "right-side position", "direction answer"]}
  ]
}"""


# ============================================================
# User Prompt Template (多图)
# ============================================================
USER_PROMPT_TEMPLATE = """These {n_frames} images are frames from an indoor 3D scan video, showing different viewpoints of the same space. Examine them carefully to understand the room layout.

**Question:** {question}

**Correct Answer:** {answer}

Generate a v6-format chain-of-thought reasoning that arrives at answer "{answer}".
- Break into 2-4 sub-steps separated by PAUSE_MARKER
- Each sub-step should handle one spatial reasoning challenge
- Ground your reasoning in what you observe across the frames
- Provide key_tokens for each sub-step

Return your response as a JSON object with fields: n_substeps, task_type, reasoning_for_training, substeps."""


# ============================================================
# 帧采样策略
# ============================================================

def sample_frames(image_paths: List[str], max_frames: int) -> List[str]:
    """从 16 帧中均匀采样 max_frames 帧。
    
    策略: 均匀间隔采样，确保覆盖场景的不同视角。
    """
    n = len(image_paths)
    if n <= max_frames:
        return image_paths
    
    # 均匀采样
    indices = [int(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
    # 去重
    indices = sorted(set(indices))
    # 如果去重后不够，补充
    while len(indices) < max_frames and len(indices) < n:
        for i in range(n):
            if i not in indices:
                indices.append(i)
                indices.sort()
                if len(indices) >= max_frames:
                    break
    
    return [image_paths[i] for i in indices[:max_frames]]


# ============================================================
# 后处理: 从 Claude 输出构建 v6 训练样本 (多图版)
# ============================================================

PAUSE_TOKEN = "<|pause|>"

def build_v6_training_sample(
    vsi_item: Dict[str, Any],
    claude_response: Dict[str, Any],
    data_root: str,
    sampled_paths: List[str],
) -> Optional[Dict[str, Any]]:
    """从 Claude 返回的 JSON 构建 v6 格式训练样本 (多图)。"""

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

        # 确保 key_tokens 是 3-6 个
        if len(key_tokens) < 3:
            key_tokens = key_tokens + ["spatial_state"] * (3 - len(key_tokens))
        if len(key_tokens) > 6:
            key_tokens = key_tokens[:6]

        # 根据 k_latent 决定 stage 结构
        if k_latent == 1:
            boundary = [{
                "tokens": key_tokens,
                "stage_id": 1,
                "role": "unified",
                "from_substep": i + 1,
            }]
        elif k_latent >= 2:
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
            "tokens": ["spatial_reasoning", "3d_layout", "direction_conclusion"],
            "stage_id": 1,
            "role": "unified",
            "from_substep": len(latent_key_tokens) + 1,
        }])
    if len(latent_key_tokens) > pause_count:
        latent_key_tokens = latent_key_tokens[:pause_count]

    # 构建图片路径列表 (多图!)
    # 使用采样后的帧路径
    resolved_paths = []
    for p in sampled_paths:
        if data_root and not os.path.isabs(p):
            resolved_paths.append(os.path.join(data_root, p))
        else:
            resolved_paths.append(p)

    # task_type 验证
    valid_task_types = {"perspective", "geometry", "spatial_relation", "physical",
                        "counterfactual", "fine_grained", "counting", "comparison",
                        "detail_observation"}
    if task_type not in valid_task_types:
        q_type = vsi_item.get("question_type", "")
        task_type = VSIBENCH_TO_TASK_TYPE.get(q_type, "spatial_relation")

    # 最终训练记录 (v6 slim 格式, 多图版)
    # 使用 image_paths (列表) 代替 image_path (字符串)
    sample = {
        "image_paths": resolved_paths,
        "question": vsi_item["question"],
        "answer": vsi_item["answer"],
        "reasoning_for_training": reasoning,
        "latent_key_tokens": latent_key_tokens,
        "task_type": task_type,
    }
    return sample


# ============================================================
# 单样本处理
# ============================================================

def process_one_sample(
    vsi_item: Dict[str, Any],
    client,
    auth_config: Dict[str, str],
    model: str,
    data_root: str,
    max_frames: int,
    temperature: float,
    max_tokens: int,
    num_retries: int,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """处理单条 VSI-Bench 样本，调用 Claude 生成 v6 格式训练数据。"""
    qid = vsi_item["question_id"]
    question = vsi_item["question"]
    answer = vsi_item["answer"]

    # 获取所有帧路径
    image_paths = vsi_item.get("image_paths", [])
    if not image_paths:
        logger.warning(f"[{qid}] 无图片路径，跳过")
        return None

    # 解析绝对路径
    abs_paths = []
    for p in image_paths:
        if data_root and not os.path.isabs(p):
            abs_paths.append(os.path.join(data_root, p))
        else:
            abs_paths.append(p)

    # 检查图片存在性 (只检查第一帧)
    if not os.path.exists(abs_paths[0]):
        logger.warning(f"[{qid}] 首帧不存在: {abs_paths[0]}")
        return None

    # 采样帧
    sampled_abs = sample_frames(abs_paths, max_frames)
    sampled_rel = sample_frames(image_paths, max_frames)

    # 编码第一张图 (作为主图)
    image_b64 = encode_image_to_base64(sampled_abs[0])
    if not image_b64:
        logger.warning(f"[{qid}] 主图编码失败: {sampled_abs[0]}")
        return None

    # 其余图片作为 extra_user_images
    extra_images = sampled_abs[1:] if len(sampled_abs) > 1 else None

    # 构建 user prompt
    user_text = USER_PROMPT_TEMPLATE.format(
        n_frames=len(sampled_abs),
        question=question,
        answer=answer,
    )

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
            extra_user_images=extra_images,
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
            logger.warning(f"[{qid}] attempt {attempt+1}/{num_retries}: 无 reasoning_for_training")
            continue

        # 检查是否包含 PAUSE_MARKER
        if "PAUSE_MARKER" not in reasoning:
            preview = reasoning[:200].replace('\n', ' ')
            logger.warning(f"[{qid}] attempt {attempt+1}/{num_retries}: 无 PAUSE_MARKER. preview: {preview}")
            continue

        # 构建 v6 训练样本
        sample = build_v6_training_sample(vsi_item, parsed_json, data_root, sampled_rel)
        if not sample:
            logger.warning(f"[{qid}] attempt {attempt+1}/{num_retries}: 构建训练样本失败")
            continue

        if verbose:
            pause_count = sample["reasoning_for_training"].count(PAUSE_TOKEN)
            logger.info(f"  ✅ [{qid}] 成功 | pauses={pause_count} "
                       f"stages={len(sample['latent_key_tokens'])} "
                       f"task={sample['task_type']} "
                       f"frames={len(sample['image_paths'])}")
        return sample

    logger.error(f"[{qid}] 所有 {num_retries} 次重试均失败")
    return None


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VSI-Bench Direction/Route Latent CoT 数据合成 - v6 格式 (多图)")
    parser.add_argument("--data_root", type=str, default="./data/VSI-Bench_eval",
                        help="VSI-Bench 数据根目录 (含 frames/)")
    parser.add_argument("--data_file", type=str,
                        default="./data/VSI-Bench_eval/vsibench_test.jsonl",
                        help="VSI-Bench 测试数据 jsonl")
    parser.add_argument("--output", type=str,
                        default="./data/vsibench/vsibench_direction_route_latent_cot.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--rtx", type=str, default=None,
                        help="兼容参数（已废弃，当前版本不使用）")
    parser.add_argument("--model", type=str, default="claude-opus-4-7",
                        help="iChat 模型名")
    parser.add_argument("--workers", type=int, default=6,
                        help="并发 worker 数 (多图请求较大，建议 <=8)")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="每条数据最多传给 Claude 的帧数 (从 16 帧中均匀采样)")
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
    parser.add_argument("--question_types", nargs="+", default=None,
                        help="仅处理指定 question_type (默认: direction + route_planning)")
    parser.add_argument("--resume", action="store_true",
                        help="从已有输出文件恢复")
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
    if not os.path.exists(args.data_file):
        logger.error(f"❌ 数据文件不存在: {args.data_file}")
        sys.exit(1)

    with open(args.data_file, "r", encoding="utf-8") as f:
        all_data = [json.loads(line.strip()) for line in f if line.strip()]

    logger.info(f"📂 加载 VSI-Bench 数据: {len(all_data)} 条 from {args.data_file}")

    # ---- 过滤题型 ----
    if args.question_types:
        target_types = set(args.question_types)
    else:
        target_types = TARGET_QUESTION_TYPES

    filtered_data = [d for d in all_data if d.get("question_type") in target_types]
    logger.info(f"  过滤后 (direction + route_planning): {len(filtered_data)} 条")

    # 统计题型分布
    type_counts = Counter(d["question_type"] for d in filtered_data)
    for qt, cnt in type_counts.most_common():
        logger.info(f"    {qt}: {cnt}")

    if args.max_samples > 0:
        filtered_data = filtered_data[:args.max_samples]
        logger.info(f"  截断为前 {args.max_samples} 条")

    # ---- Resume ----
    results = []
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_questions = {r["question"] for r in results}
            filtered_data = [d for d in filtered_data if d["question"] not in done_questions]
            logger.info(f"🔄 Resume: 已完成 {len(results)} 条，剩余 {len(filtered_data)} 条")
        except Exception as e:
            logger.warning(f"Resume 加载失败: {e}，从头开始")
            results = []

    pending = filtered_data
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

    SAVE_INTERVAL = 20

    logger.info(f"🚀 开始合成 (model={args.model}, workers={args.workers}, "
                f"max_frames={args.max_frames}, temp={args.temperature})")
    logger.info("=" * 60)

    def _worker(item):
        return process_one_sample(
            vsi_item=item,
            client=client,
            auth_config=auth_config,
            model=args.model,
            data_root=args.data_root,
            max_frames=args.max_frames,
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
            if i % 5 == 0 or i == total:
                logger.info(
                    f"  📊 [{i}/{total}] {elapsed:.0f}s | "
                    f"speed={speed:.2f}it/s | ETA={eta:.0f}s | "
                    f"✅{success_count} ❌{fail_count} | "
                    f"rate={success_count/max(1,success_count+fail_count)*100:.1f}%"
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
    logger.info("  ✅ VSI-Bench Latent CoT 合成完成 (v6 格式, 多图)")
    logger.info("=" * 60)
    logger.info(f"  总样本:     {total}")
    logger.info(f"  成功:       {success_count}")
    logger.info(f"  失败:       {fail_count}")
    logger.info(f"  成功率:     {success_count/max(1,success_count+fail_count)*100:.1f}%")
    logger.info(f"  耗时:       {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    logger.info(f"  输出:       {args.output}")
    logger.info(f"  总记录数:   {len(results)}")

    # 保存 config
    config_path = args.output.replace(".json", "_config.json")
    config = {
        "data_source": "vsibench",
        "format": "v6_slim_multi_image",
        "question_types": list(target_types),
        "model": args.model,
        "max_frames": args.max_frames,
        "num_samples": len(results),
        "total_filtered_samples": total,
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
