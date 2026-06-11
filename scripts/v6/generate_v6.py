"""
generate_v6.py
==============
v6 continuous-blank-CoT 数据合成主脚本 (最小 wrapper).

设计:
  - 复用 v3/ichat_client + v5 seed 流水线 (auto_seed_from_stage2v2 / load_seed_jsonl).
  - 调 Claude 让其按 prompts/v6/generator_v6.txt 一次性产出 v6 schema 的 JSON.
  - schema_v6.validate_v6_sample 做硬校验.
  - 同时挂上 messages (assistant content = sample.cot_text) 形成训练就绪样本.
  - 写 .report.json + .static_leak.json (静态留白检查).

使用:
export OPENAI_API_KEY=...    # OPENAI_BASE_URL 可选
    bash scripts/v6/run_phase0_v6_smoke.sh
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
_V5_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, "v5"))
for p in (_SCRIPT_DIR, _V3_DIR, _V5_DIR):
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
# v5 seed loader 直接复用
from generate_v5 import (  # type: ignore
    load_seed_jsonl,
    auto_seed_from_stage2v2,
    _resolve_image_path,
    TASK_TYPE_REMAP,
    _resample_seeds_by_task_dist,
    DEFAULT_TASK_TYPE_DIST,
    _map_task_type,
)
from schema_v6 import (  # type: ignore
    validate_v6_sample,
    render_assistant_text,
    stats_summary,
    get_data_version,
    inspect_v6_sample,
    normalize_sample_inplace,
    static_leak_check,
    HIGH_VALUE_TASK_TYPES,
    to_training_record,
    PREFERRED_KEY_TOKENS_PER_STAGE,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_v6")


# ============================================================
# Resume / checkpoint utilities (v6.5b)
# ============================================================
# partial 文件结构: {"results":[...], "dropped":[...], "done_ids":[...], "meta":{...}}
#   * results   : 通过 schema 校验的 sample (写最终 output 用)
#   * dropped   : 失败带 raw_parsed (写 .dropped.json 用)
#   * done_ids  : 已尝试过的 sample_id (失败也算 done, 不重试)
#   * meta      : data_version / 时间戳 / 上次工作配置摘要
# 原子写: 先写 .partial.json.new, 再 rename -> .partial.json

def _partial_path(output: str) -> str:
    return output.replace(".json", ".partial.json") if output.endswith(".json") else output + ".partial.json"


def _load_partial(output: str) -> Tuple[List[Dict], List[Dict], set]:
    path = _partial_path(output)
    if not os.path.exists(path):
        return [], [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        results = obj.get("results", []) or []
        dropped = obj.get("dropped", []) or []
        done_ids = set(obj.get("done_ids", []) or [])
        # 防御性: results / dropped 里所有 sample_id 也并入 done_ids (旧 partial 兼容)
        for r in results:
            sid = r.get("sample_id")
            if sid:
                done_ids.add(sid)
        for d in dropped:
            sid = d.get("sample_id")
            if sid:
                done_ids.add(sid)
        logger.info(
            f"[resume] loaded partial: {len(results)} ok + {len(dropped)} dropped "
            f"= {len(done_ids)} done_ids from {path}"
        )
        return results, dropped, done_ids
    except Exception as e:
        logger.warning(f"[resume] failed to load partial {path}: {e}; starting fresh")
        return [], [], set()


def _save_partial(output: str, results: List[Dict], dropped: List[Dict], done_ids: set, meta: Optional[Dict] = None):
    path = _partial_path(output)
    tmp = path + ".new"
    payload = {
        "results": results,
        "dropped": dropped,
        "done_ids": sorted(done_ids),
        "meta": meta or {},
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # 原子 rename
        logger.info(f"[partial] {len(results)} ok + {len(dropped)} dropped -> {path}")
    except Exception as e:
        logger.warning(f"[partial] save failed: {e}")


def _cleanup_partial(output: str):
    path = _partial_path(output)
    if os.path.exists(path):
        try:
            os.remove(path)
            logger.info(f"[partial] cleaned up: {path}")
        except Exception as e:
            logger.warning(f"[partial] cleanup failed: {e}")


# ============================================================
# Prompt loading
# ============================================================

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# Per-seed generation
# ============================================================

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
    n_substeps_target: Optional[int] = None,
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

    # n_substeps 强制硬约束 (扩量多样性): 上层按目标分布抽样得到 target, 这里强制写入 user_text
    if n_substeps_target is not None and 2 <= n_substeps_target <= 5:
        n_sub_directive = (
            f"\n\nHARD CONSTRAINT: n_substeps MUST be EXACTLY {n_substeps_target}. "
            f"Generate exactly {n_substeps_target} substep entries and exactly "
            f"{n_substeps_target} <|latent|>...<|/latent|> blocks in cot_text. "
            f"Do NOT use any other value."
        )
    else:
        n_sub_directive = ""

    user_text = (
        f"task_type hint: {target_tt}\n"
        f"reference_question (style only, do NOT copy): {ref_q}\n"
        f"reference_answer (style only): {ref_a}\n\n"
        f"Now produce the JSON object as specified in the system prompt. "
        f"Remember the BLANK CONTRACT (Section C): each anchor_phrase must reuse "
        f"a referent introduced before <|latent|>, NOT restate the latent's value. "
        f"Stay QUALITATIVE-ONLY: never name precise numbers (no '30 degrees', "
        f"no 'thirty degrees', no '2 metres', no pixel coordinates, no percent)."
        f"{n_sub_directive}"
    )

    raw = call_chat_with_image(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        image_base64=b64,
        auth_config=auth_config,
        max_retries=5,
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

    # ---- 必填字段 ----
    q = (parsed.get("question") or "").strip()
    a = (parsed.get("answer") or "").strip()
    tt_out = (parsed.get("task_type") or target_tt).strip()
    hops = parsed.get("difficulty_hops")
    n_sub = parsed.get("n_substeps")
    substeps = parsed.get("substeps")
    cot = parsed.get("cot_text")
    if not (q and a and substeps and cot):
        return None, "missing_fields"

    # ---- 数值字段 ----
    try:
        hops_int = int(hops)
    except Exception:
        hops_int = 3
    if not (2 <= hops_int <= 6):
        hops_int = max(2, min(6, hops_int)) if hops_int else 3
    try:
        n_sub_int = int(n_sub)
    except Exception:
        n_sub_int = -1
    try:
        ln_int = int(parsed.get("latent_necessity", -1))
    except Exception:
        ln_int = -1
    if not (0 <= ln_int <= 3):
        ln_int = -1
    nvs = (parsed.get("non_verbal_signal") or "").strip()

    # ---- 组装 sample ----
    sample = {
        "sample_id":          seed["sample_id"],
        "data_source":        "v6_blank_cot_vcr_seed",
        "data_version":       get_data_version(),
        "image_paths":        [img_resolved],
        "task_type":          tt_out,
        "difficulty_hops":    hops_int,
        "n_substeps":         n_sub_int,
        "latent_necessity":   ln_int,
        "non_verbal_signal":  nvs,
        "question":           q,
        "answer":             a,
        "substeps":           substeps,
        "cot_text":           cot,
        "origin":             "v6_blank_cot",
        "original_sample_ref": seed.get("source_id", ""),
        "src_task_type":      src_tt,
    }

    # 防御性清洗 (剔除 substep 中未声明字段)
    normalize_sample_inplace(sample)

    ok, reason = validate_v6_sample(sample)
    if not ok:
        # 携带 raw parsed dict, 供上层落盘调参
        sample["_drop_reason"] = reason
        sample["_raw_parsed"] = parsed
        return sample, f"schema_invalid:{reason}"

    # 渲染 assistant 文本 (即 cot_text 本身)
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
    ap.add_argument("--seed_file", type=str, default=None)
    ap.add_argument("--auto_seed_from", type=str,
                    default="data/stage2_v2/stage2_hybrid_90k_latent_only.json")
    ap.add_argument("--auto_seed_n", type=int, default=20)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--save_seed", type=str, default=None)
    ap.add_argument("--model", type=str, default="claude-opus-4.7")
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=4500)
    ap.add_argument("--source", type=str, default=None)
    ap.add_argument("--appid", type=str, default=None)
    ap.add_argument("--appkey", type=str, default=None)
    ap.add_argument("--rtx", type=str, default=None)
    ap.add_argument("--image_root", type=str, default=None)
    ap.add_argument("--prompt_file", type=str,
                    default=os.path.join(_SCRIPT_DIR, "prompts", "generator_v6.txt"))
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <output>.partial.json (skip already-attempted sample_ids).")
    ap.add_argument("--task_type_dist", type=str, default=None,
                    help="JSON dict overriding default task_type distribution; "
                         "pass 'off' to disable resampling.")
    ap.add_argument("--n_substeps_dist", type=str, default=None,
                    help='JSON dict for per-call n_substeps target distribution, '
                         'e.g. \'{"2":0.30,"3":0.40,"4":0.20,"5":0.10}\'. '
                         'Pass "off" to let the LLM choose freely. Default = '
                         '30/40/20/10 over {2,3,4,5} to enforce diversity.')
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

    # ---- v6.5b resume: 载入 partial 并过滤已完成的 sample_id ----
    resumed_results: List[Dict] = []
    resumed_dropped: List[Dict] = []
    resumed_done_ids: set = set()
    if args.resume:
        resumed_results, resumed_dropped, resumed_done_ids = _load_partial(args.output)
        if resumed_done_ids:
            before_n = len(seeds)
            seeds = [s for s in seeds if s.get("sample_id") not in resumed_done_ids]
            logger.info(
                f"[resume] filtered seeds: {before_n} -> {len(seeds)} "
                f"(skipped {before_n - len(seeds)} already-attempted)"
            )
        if not seeds:
            logger.info("[resume] all seeds already attempted; will just re-finalize outputs.")

    # ---- 按 task_type_dist 重采样 (复用 v5) ----
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
        logger.info(f"task_type_dist resampling. before={dict(before)}")
        logger.info(f"task_type_dist resampling. after ={dict(after)}")

    # ---- 按 n_substeps_dist 为每条 seed 分配 target (强制多样性, 防止 LLM 全输出 3) ----
    n_sub_targets: List[Optional[int]] = []
    if args.n_substeps_dist == "off":
        n_sub_targets = [None] * len(seeds)
        logger.info("n_substeps_dist: OFF (LLM chooses freely)")
    else:
        if args.n_substeps_dist:
            try:
                ns_dist_raw = json.loads(args.n_substeps_dist)
                ns_dist = {int(k): float(v) for k, v in ns_dist_raw.items()}
            except Exception as e:
                logger.error(f"bad --n_substeps_dist json: {e}")
                return 1
        else:
            # 默认: 强制多样化 (避免 LLM 偷懒全输出 3)
            ns_dist = {2: 0.30, 3: 0.40, 4: 0.20, 5: 0.10}
        # 归一化
        _s = sum(ns_dist.values())
        ns_dist = {k: v / _s for k, v in ns_dist.items()}
        # 按比例切分: 前 30% seed -> n=2, 接下来 40% -> n=3, ...
        rng2 = random.Random(args.seed + 1)
        ks = sorted(ns_dist.keys())
        N = len(seeds)
        counts = {k: int(round(ns_dist[k] * N)) for k in ks}
        # 修正使 sum == N
        diff = N - sum(counts.values())
        if diff != 0:
            biggest = max(ks, key=lambda k: ns_dist[k])
            counts[biggest] += diff
        # 按 count 平铺并 shuffle
        n_sub_targets = []
        for k in ks:
            n_sub_targets.extend([k] * counts[k])
        rng2.shuffle(n_sub_targets)
        n_sub_targets = n_sub_targets[:N]
        logger.info(f"n_substeps_dist enforced: counts={counts}")

    # ---- prompt + client ----
    system_prompt = load_prompt(args.prompt_file)
    logger.info(f"loaded prompt: {args.prompt_file} ({len(system_prompt)} chars)")
    client = create_openai_client()

    # ---- 并行生成 ----
    # v6.5b resume: results / dropped 预填充已恢复内容
    results: List[Dict] = list(resumed_results)
    dropped: List[Dict] = list(resumed_dropped)
    done_ids: set = set(resumed_done_ids)
    fail_reasons: Counter = Counter()
    t0 = time.time()
    total = len(seeds)
    partial_meta = {
        "data_version": get_data_version(),
        "model": args.model,
        "max_workers": args.max_workers,
        "start_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output": args.output,
    }

    def _task(seed_with_idx):
        i, seed = seed_with_idx
        nst = n_sub_targets[i] if i < len(n_sub_targets) else None
        return generate_one(
            seed,
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            auth_config=auth_config,
            image_root=args.image_root,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n_substeps_target=nst,
        )

    try:
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(_task, (i, s)): (i, s) for i, s in enumerate(seeds)}
            done = 0
            since_last_save = 0
            for fut in cf.as_completed(futures):
                i_seed, seed = futures[fut]
                done += 1
                try:
                    sample, status = fut.result()
                except Exception as e:
                    sample, status = None, f"exception:{type(e).__name__}:{str(e)[:120]}"
                # 记录 done_id (成功 / 失败都算 done, 防 resume 重跑同一坏 seed)
                sid = seed.get("sample_id")
                if sid:
                    done_ids.add(sid)
                if sample is not None and status == "ok":
                    results.append(sample)
                    logger.info(
                        f"[{done}/{total}] OK n_substeps={sample.get('n_substeps')} "
                        f"k_total={sum(int(s.get('k_latent') or 0) for s in sample.get('substeps') or [])} "
                        f"task={sample.get('task_type')}"
                    )
                else:
                    fail_reasons[status] += 1
                    if isinstance(sample, dict):
                        # schema_invalid 路径: 带 raw parsed 落到 dropped
                        dropped.append({
                            "sample_id": seed.get("sample_id"),
                            "reason": status,
                            "raw_parsed": sample.get("_raw_parsed"),
                        })
                    else:
                        # 非 schema 失败也记一条简单原因 (便于 resume 后查)
                        dropped.append({
                            "sample_id": seed.get("sample_id"),
                            "reason": status,
                            "raw_parsed": None,
                        })
                    logger.warning(
                        f"[{done}/{total}] FAIL seed={seed.get('sample_id','?')[:50]} "
                        f"reason={status}"
                    )
                # 每 save_every 条落一次 partial 检查点 (包含 results+dropped+done_ids)
                since_last_save += 1
                if since_last_save >= args.save_every:
                    _save_partial(args.output, results, dropped, done_ids, partial_meta)
                    since_last_save = 0
    except KeyboardInterrupt:
        logger.warning("[resume] KeyboardInterrupt; saving partial before exit ...")
        _save_partial(args.output, results, dropped, done_ids, partial_meta)
        raise
    except Exception as e:
        logger.warning(f"[resume] unexpected exception {type(e).__name__}: {e}; saving partial ...")
        _save_partial(args.output, results, dropped, done_ids, partial_meta)
        raise
    # 循环正常结束: 再保存一次 partial 以防 final dump 在中间坏掉
    _save_partial(args.output, results, dropped, done_ids, partial_meta)

    # ---- 落盘 ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ saved: {args.output} ({len(results)} samples)")

    # ---- slim training 落盘 (rld/data.py v6.2 直接消费的 6 字段) ----
    slim_path = args.output.replace(".json", "_training_slim.json")
    if slim_path == args.output:
        slim_path = args.output + ".training_slim.json"
    slim_recs = [to_training_record(s, tokens_per_stage=PREFERRED_KEY_TOKENS_PER_STAGE, keep_meta=False)
                 for s in results]
    with open(slim_path, "w", encoding="utf-8") as f:
        json.dump(slim_recs, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ slim training saved: {slim_path} ({len(slim_recs)} samples)")

    # ---- dropped 落盘 ----
    if dropped:
        dropped_path = args.output.replace(".json", ".dropped.json")
        if dropped_path == args.output:
            dropped_path = args.output + ".dropped.json"
        with open(dropped_path, "w", encoding="utf-8") as f:
            json.dump(dropped, f, ensure_ascii=False, indent=2)
        logger.info(f"🔴 saved dropped: {dropped_path} ({len(dropped)} samples)")

    # ---- 静态 leak check ----
    leaks: List[Dict] = []
    n_leak_total = 0
    for s in results:
        lk = static_leak_check(s)
        leaks.append({"sample_id": s.get("sample_id"),
                       "n_leaks": lk["n_leaks"],
                       "per_substep": lk["per_substep"]})
        n_leak_total += lk["n_leaks"]
    leak_path = args.output.replace(".json", ".static_leak.json")
    if leak_path == args.output:
        leak_path = args.output + ".static_leak.json"
    with open(leak_path, "w", encoding="utf-8") as f:
        json.dump(leaks, f, ensure_ascii=False, indent=2)
    logger.info(f"🔎 static leak check: total_leaks={n_leak_total} -> {leak_path}")

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
        "static_leak_total": n_leak_total,
    }
    report_path = args.output.replace(".json", ".report.json")
    if report_path == args.output:
        report_path = args.output + ".report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"📊 report: {report_path}")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))

    # ---- v6.5b resume: 一切落盘成功, 清理 partial 检查点 ----
    _cleanup_partial(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
