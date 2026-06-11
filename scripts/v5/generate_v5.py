"""
generate_v5.py
==============
v5 agentic-trajectory 数据合成主脚本。

输入: vcr_stage2v2 风格的种子图像 + task_type
输出: 一次性生成的完整 trajectory（单轮 QA 形式）

设计要点
--------
1. 复用 ichat_client (OpenAI 兼容客户端 / JSON 修复 / 重试)。
2. 复用 v4 的 auto_seed_from_stage2v2 / load_seed_jsonl 抽样逻辑。
3. 调 Claude 让其按 prompts/v5/generator_v5.txt 一次性产出
   {question, answer, task_type, difficulty_hops, trajectory_steps}.
4. schema_v5.validate_v5_sample 做 hard-constraint 校验; 不过的丢弃。
5. 同时把 trajectory 渲染成 <step>…</step> 串行 assistant 文本, 直接落盘
   到训练就绪的 messages 格式 (单轮 QA)。
6. 写 .report.json 汇总统计 + 失败原因分布。

使用
----
export OPENAI_API_KEY=...    # OPENAI_BASE_URL 可选
    python scripts/v5/generate_v5.py \
        --seed_file data/v4/phase0_seed.jsonl \
        --output data/v5/trajectories.raw.json \
        --max_samples 5
"""
import argparse
import concurrent.futures as cf
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# 让脚本可独立执行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V3_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, "v3"))
for p in (_SCRIPT_DIR, _V3_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from ichat_client import (  # type: ignore
    create_openai_client,
    encode_image_to_base64,
    call_chat_with_image,
    parse_gpt_response,
    build_auth_config_from_env_or_args,
    _save_intermediate,
)
from schema_v5 import (  # type: ignore
    validate_v5_sample,
    render_assistant_text,
    stats_summary,
    get_data_version,
    inspect_v5_sample,
    normalize_sample_inplace,
    HIGH_VALUE_TASK_TYPES,
    MEDIUM_VALUE_TASK_TYPES,
    LOW_VALUE_TASK_TYPES,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_v5")


# ============================================================
# Prompt loading
# ============================================================

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# Seed loading (复用 v4 风格)
# ============================================================

def load_seed_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def auto_seed_from_stage2v2(
    stage2v2_path: str,
    n: int,
    seed: int = 42,
    only_vcr_erqa_style: bool = True,
) -> List[Dict]:
    from collections import defaultdict
    with open(stage2v2_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if only_vcr_erqa_style:
        data = [s for s in data if s.get("data_source") == "vcr_erqa_style"]
    if not data:
        raise RuntimeError(f"empty stage2_v2 pool from {stage2v2_path}")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in data:
        groups[s.get("task_type", "spatial_relation")].append(s)
    keys = sorted(groups.keys())
    per = n // max(len(keys), 1)
    rem = n - per * len(keys)
    selected: List[Dict] = []
    for i, k in enumerate(keys):
        pool = groups[k]
        rng.shuffle(pool)
        take = per + (1 if i < rem else 0)
        selected.extend(pool[:take])
    rng.shuffle(selected)
    selected = selected[:n]
    seeds = []
    for s in selected:
        img = s.get("image", "") or s.get("image_path", "")
        seeds.append({
            "sample_id": f"v5_seed_{s.get('image', 'x').replace('/', '_')[-60:]}",
            "source_id": s.get("image", ""),
            "data_source": "vcr_stage2v2",
            "reference_question": s.get("question", ""),
            "reference_answer_long": s.get("answer", ""),
            "task_type": s.get("task_type", "spatial_relation"),
            "image_paths": [img],
            "num_images": 1,
        })
    return seeds


# ============================================================
# Per-seed generation
# ============================================================

def _resolve_image_path(p: str, image_root: Optional[str]) -> str:
    if os.path.isabs(p):
        return p
    if image_root:
        cand = os.path.join(image_root, p)
        if os.path.exists(cand):
            return cand
    return p


# task_type hint: v4 上游 task_type 与 v5 schema 的映射 (尽量保留语义)
# v5.1: 增加 geometry / fine_grained / multi_hypothesis / ocr 映射
TASK_TYPE_REMAP = {
    # passthrough
    "spatial_relation":     "spatial_relation",
    "perspective_taking":   "perspective",
    "physical_intuition":   "physical",
    "dynamic_simulation":   "physical",
    "detail_observation":   "detail_observation",
    "intent_inference":     "temporal",
    "temporal_reasoning":   "temporal",
    "visual_counterfactual":"counterfactual",
    "counterfactual":       "counterfactual",
    "counting":             "counting",
    "comparison":           "comparison",
    "perspective":          "perspective",
    "physical":             "physical",
    "temporal":             "temporal",
    # v5.1 新高价值类型 passthrough
    "geometry":             "geometry",
    "fine_grained":         "fine_grained",
    "multi_hypothesis":     "multi_hypothesis",
    "ocr":                  "ocr",
}

# v5.1: 默认 task_type 采样配额 (高价值为主)
# 上层 generate 脚本可通过 --task_type_dist 覆盖
DEFAULT_TASK_TYPE_DIST = {
    # HIGH-VALUE total = 80%
    "perspective":         0.20,
    "geometry":            0.20,
    "fine_grained":        0.15,
    "counterfactual":      0.15,
    "multi_hypothesis":    0.10,
    # MEDIUM-VALUE total = 12%
    "physical":            0.07,
    "comparison":          0.05,
    # LOW-VALUE total = 8% (仅用于多样性)
    "detail_observation":  0.04,
    "counting":            0.02,
    "temporal":            0.02,
    # "spatial_relation" / "ocr" 不列入默认配额
}


def _map_task_type(t: str) -> str:
    return TASK_TYPE_REMAP.get(t, "perspective")  # v5.1: 默认回落从 spatial 改为 perspective


def _resample_seeds_by_task_dist(
    seeds: List[Dict],
    target_dist: Dict[str, float],
    target_n: int,
    rng: random.Random,
) -> List[Dict]:
    """v5.1: 按目标分布对 seeds 重采样。双向 hint:
      - 源 task_type -> remap target task_type
      - 不足量的 task_type 会从任意源补齐, 同时 override seed.task_type
        以保证 generator 拿到的 hint = target task_type.
    """
    from collections import defaultdict
    pool_by_target: Dict[str, List[Dict]] = defaultdict(list)
    for s in seeds:
        tgt = _map_task_type(s.get("task_type", ""))
        pool_by_target[tgt].append(s)
    spare_pool: List[Dict] = list(seeds)
    rng.shuffle(spare_pool)

    selected: List[Dict] = []
    # 归一化分布
    total_w = sum(max(v, 0.0) for v in target_dist.values()) or 1.0
    for tt, w in target_dist.items():
        quota = int(round(target_n * (max(w, 0.0) / total_w)))
        if quota <= 0:
            continue
        avail = list(pool_by_target.get(tt, []))
        rng.shuffle(avail)
        # 优先从原生同 task_type 取
        take = avail[:quota]
        for s in take:
            s2 = dict(s); s2["task_type"] = tt
            selected.append(s2)
        # 不够从 spare pool 补齐, 并 override task_type hint
        need = quota - len(take)
        if need > 0:
            for _ in range(need):
                if not spare_pool:
                    break
                s = spare_pool.pop()
                s2 = dict(s); s2["task_type"] = tt
                selected.append(s2)
    # 如果化入后不足, 从 spare 随机补齐 (并随机指派高价值 task_type)
    while len(selected) < target_n and spare_pool:
        s = spare_pool.pop()
        s2 = dict(s); s2["task_type"] = rng.choice(sorted(HIGH_VALUE_TASK_TYPES))
        selected.append(s2)
    rng.shuffle(selected)
    return selected[:target_n]


def generate_one(
    seed: Dict,
    *,
    client,
    model: str,
    system_prompt: str,
    auth_config: Dict,
    image_root: Optional[str],
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[Dict], str]:
    image_paths = seed.get("image_paths", [])
    if not image_paths:
        return None, "no_image_paths"
    img_resolved = _resolve_image_path(image_paths[0], image_root)
    b64 = encode_image_to_base64(img_resolved)
    if not b64:
        return None, f"image_encode_fail:{img_resolved}"

    src_tt = seed.get("task_type", "spatial_relation")
    target_tt = _map_task_type(src_tt)
    ref_q = (seed.get("reference_question") or "")[:200]
    ref_a = (seed.get("reference_answer_long") or "")[:200]

    user_text = (
        f"task_type hint: {target_tt}\n"
        f"reference_question (style only, do NOT copy): {ref_q}\n"
        f"reference_answer (style only): {ref_a}\n\n"
        f"Now produce the JSON object as specified."
    )

    raw = call_chat_with_image(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        image_base64=b64,
        auth_config=auth_config,
        max_retries=3,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format_json=True,
        image_detail="high",
    )
    if not raw:
        return None, "api_fail"

    parsed = parse_gpt_response(raw)
    if not parsed:
        return None, "json_parse_fail"

    # 必填字段
    q = (parsed.get("question") or "").strip()
    a = (parsed.get("answer") or "").strip()
    tt_out = (parsed.get("task_type") or target_tt).strip()
    hops = parsed.get("difficulty_hops")
    steps = parsed.get("trajectory_steps")
    if not (q and a and steps):
        return None, "missing_fields"

    # ---- 解析 difficulty_hops ----
    try:
        hops_int = int(hops)
    except Exception:
        hops_int = 4
    if not (3 <= hops_int <= 7):
        hops_int = max(3, min(7, hops_int)) if hops_int else 4

    # ---- v5.1: 解析 latent_necessity / non_verbal_signal ----
    try:
        ln_int = int(parsed.get("latent_necessity", -1))
    except Exception:
        ln_int = -1
    if not (0 <= ln_int <= 3):
        ln_int = -1
    nvs = (parsed.get("non_verbal_signal") or "").strip()

    # ---- 组装最终 sample (顶层 + messages + struct) ----
    sample = {
        "sample_id":       seed["sample_id"],
        "data_source":     "v5_agentic_vcr_seed",
        "data_version":    get_data_version(),
        "image_paths":     [img_resolved],
        "task_type":       tt_out,
        "difficulty_hops": hops_int,
        "question":        q,
        "answer":          a,
        "trajectory_steps": steps,
        "origin":          "v5_agentic",
        "original_sample_ref": seed.get("source_id", ""),
        "src_task_type":   src_tt,
        # v5.1
        "latent_necessity":  ln_int,
        "non_verbal_signal": nvs,
    }

    # 防御性清洗: 剔除 LLM 偶发产出的未声明 step 字段
    # (step_id_check / observation_public / public_hint_redacted / latent_hint_note ...)
    normalize_sample_inplace(sample)

    ok, reason = validate_v5_sample(sample)
    if not ok:
        return None, f"schema_invalid:{reason}"

    # 渲染 assistant 文本 (单轮 QA 形式)
    assistant_text = render_assistant_text(sample)
    sample["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_resolved},
                {"type": "text",  "text": q},
            ],
        },
        {
            "role": "assistant",
            "content": assistant_text,
        },
    ]

    return sample, "ok"


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    # ---- 输入种子 ----
    ap.add_argument("--seed_file", type=str, default=None,
                    help="seed jsonl (与 v4 phase0_seed.jsonl 同格式)")
    ap.add_argument("--auto_seed_from", type=str,
                    default="data/stage2_v2/stage2_hybrid_90k_latent_only.json",
                    help="若不传 --seed_file, 从该 stage2_v2 主文件抽样")
    ap.add_argument("--auto_seed_n", type=int, default=50)
    # ---- 输出 ----
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--save_seed", type=str, default=None)
    # ---- API ----
    ap.add_argument("--model", type=str, default="claude-opus-4.7")
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=6000,
                    help="v5 trajectory 比 v4 长, 默认 6000")
    # ---- 鉴权 ----
    ap.add_argument("--source", type=str, default=None)
    ap.add_argument("--appid", type=str, default=None)
    ap.add_argument("--appkey", type=str, default=None)
    ap.add_argument("--rtx", type=str, default=None)
    # ---- 其他 ----
    ap.add_argument("--image_root", type=str, default=None)
    ap.add_argument("--prompt_file", type=str,
                    default=os.path.join(_SCRIPT_DIR, "prompts", "generator_v5.txt"))
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    # v5.1: task_type 采样分布
    ap.add_argument(
        "--task_type_dist", type=str, default=None,
        help="JSON dict, e.g. '{\"perspective\":0.3,\"geometry\":0.3,...}'. "
             "默认使用 schema_v5.DEFAULT_TASK_TYPE_DIST (高价值为主)。传 'off' 禁用重采样。",
    )
    args = ap.parse_args()

    # ---- 鉴权 ----
    auth_config = build_auth_config_from_env_or_args(
        source=args.source, appid=args.appid, appkey=args.appkey, rtx=args.rtx,
    )
    if not auth_config:
        return 1

    # ---- 加载种子 ----
    if args.seed_file:
        seeds = load_seed_jsonl(args.seed_file)
        logger.info(f"loaded seed file: {len(seeds)} samples from {args.seed_file}")
    else:
        if not os.path.exists(args.auto_seed_from):
            logger.error(f"auto_seed_from not found: {args.auto_seed_from}")
            return 1
        seeds = auto_seed_from_stage2v2(
            args.auto_seed_from, n=args.auto_seed_n, seed=args.seed,
        )
        logger.info(f"auto_seed: {len(seeds)} samples from {args.auto_seed_from}")
        if args.save_seed:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_seed)) or ".", exist_ok=True)
            with open(args.save_seed, "w", encoding="utf-8") as f:
                for s in seeds:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            logger.info(f"saved seed -> {args.save_seed}")

    if args.max_samples is not None:
        seeds = seeds[: args.max_samples]
        logger.info(f"capped to max_samples={args.max_samples}: {len(seeds)} seeds")
    if not seeds:
        logger.error("no seed samples")
        return 1

    # ---- v5.1: 按 task_type_dist 重采样 ----
    if args.task_type_dist != "off":
        if args.task_type_dist:
            try:
                dist = json.loads(args.task_type_dist)
            except Exception as e:
                logger.error(f"bad --task_type_dist json: {e}")
                return 1
        else:
            dist = DEFAULT_TASK_TYPE_DIST
        rng = random.Random(args.seed)
        before = Counter(s.get("task_type", "?") for s in seeds)
        seeds = _resample_seeds_by_task_dist(seeds, dist, len(seeds), rng)
        after = Counter(s.get("task_type", "?") for s in seeds)
        logger.info(f"task_type_dist resampling applied. before={dict(before)}")
        logger.info(f"task_type_dist resampling applied. after ={dict(after)}")

    # ---- prompt + client ----
    system_prompt = load_prompt(args.prompt_file)
    logger.info(f"loaded prompt: {args.prompt_file} ({len(system_prompt)} chars)")
    client = create_openai_client()

    # ---- 并行生成 ----
    results: List[Dict] = []
    fail_reasons: Counter = Counter()
    t0 = time.time()
    total = len(seeds)

    def _task(seed):
        return generate_one(
            seed,
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            auth_config=auth_config,
            image_root=args.image_root,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(_task, s): s for s in seeds}
        done = 0
        for fut in cf.as_completed(futures):
            seed = futures[fut]
            done += 1
            try:
                sample, status = fut.result()
            except Exception as e:
                sample, status = None, f"exception:{type(e).__name__}:{str(e)[:120]}"
            if sample is not None:
                results.append(sample)
                if done % 5 == 0 or done == total:
                    logger.info(
                        f"[{done}/{total}] ok={len(results)} "
                        f"fail={done-len(results)} elapsed={time.time()-t0:.1f}s"
                    )
            else:
                fail_reasons[status] += 1
                logger.warning(
                    f"[{done}/{total}] FAIL seed={seed.get('sample_id','?')[:50]} "
                    f"reason={status}"
                )
            if len(results) and len(results) % args.save_every == 0:
                _save_intermediate(results, args.output)

    # ---- 落盘 ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ saved: {args.output} ({len(results)} samples)")

    # ---- report ----
    summary = stats_summary(results)
    report = {
        "total_seeds":     total,
        "succeeded":       len(results),
        "failed":          total - len(results),
        "fail_reasons":    dict(fail_reasons),
        "elapsed_seconds": round(time.time() - t0, 1),
        "model":           args.model,
        "data_version":    get_data_version(),
        "summary":         summary,
    }
    report_path = args.output.replace(".json", ".report.json")
    if report_path == args.output:
        report_path = args.output + ".report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"📊 report: {report_path}")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
