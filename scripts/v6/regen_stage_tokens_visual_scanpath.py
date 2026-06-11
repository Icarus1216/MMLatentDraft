"""regen_stage_tokens_visual_scanpath.py — 用 visual scanpath 重生 batch1 latent_key_tokens.

为何必要:
  当前 build_latent_key_tokens 是纯文本启发式 (从 latent_hint / non_verbal_signal /
  intent / anchor 切词), 产生的 stage_tokens 是文本词袋而不是视觉认知流, 不符合
  LASER 论文 (arxiv 2601.06803, ybb6/laser) 的 "Visual Scanpath: Forest -> Trees"
  设计哲学. v6.5_S1 数据 anchor / cot_text 已经合格, 只需要把 stage_tokens 用
  visual scanpath 重生即可.

策略 (B 方案):
  1) 读 batch1 src json (含 substeps 的完整版): v6_5f_n10000_multisrc.json
  2) 对每条 sample, 对每个 substep, 调 claude-opus-4.7 看图 + substep_intent +
     substep_anchor + k_latent → 产出 visual_scanpath (有序的 atomic visual concepts)
  3) 按 k_latent 把 visual_scanpath 切成 stage_tokens:
       k=1: 整段 → unified stage
       k=2: 前半 → abstract (Global Anchor + Subject Localization)
            后半 → concrete (Visual Evidence + Critical Resolution)
       k=3: 三等分 → abstract / bridge / concrete
  4) 覆写训练 slim 的 latent_key_tokens, 其他字段保持不变
  5) 同时把 visual_scanpath 完整序列写到 src json 的 substep 里 (字段 visual_scanpath),
     便于审计/复用

设计要点:
  - 软合格判据: visual_scanpath 至少 max(2, k_latent) 个 concepts; 否则 fallback
    回退到原文本启发式 (build_latent_key_tokens) 的 stage_tokens, 防止训练侧空 stage.
  - resume: 写一个 .partial.json 保存已完成的 sample_id, 中断可续. 与 generate_v6
    的 partial 风格对齐.
  - 并行: 8 个 worker 同步调 opus, 与 batch1 主流程一致.

调用:
    nohup python3 scripts/v6/regen_stage_tokens_visual_scanpath.py \
        --src    data/v6/v6_5f_n10000_multisrc.json \
        --slim   data/v6/v6_5f_n10000_multisrc_training_slim.json \
        --out_src data/v6/v6_5f_n10000_multisrc_vsp.json \
        --out_slim data/v6/v6_5f_n10000_multisrc_training_slim_vsp.json \
        --max_workers 8 \
        > logs/v6/regen_vsp.log 2>&1 &
    disown
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
)
from schema_v6 import (  # type: ignore
    build_latent_key_tokens,  # fallback 用
    PREFERRED_KEY_TOKENS_PER_STAGE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("regen_vsp")

DEFAULT_PROMPT = os.path.join(_SCRIPT_DIR, "prompts", "visual_scanpath_v1.txt")


# ============================================================
# Stage slicing
# ============================================================

def _slice_scanpath_to_stages(visual_scanpath: List[str],
                              k_latent: int,
                              substep_id: int) -> List[Dict[str, Any]]:
    """把 visual scanpath (有序原子视觉概念) 切成 k_latent 个 stage_tokens.

    切分语义 (与 LASER 4-stage 扫描对齐):
      k=1 (unified):   整段 → 单 stage (Global → Resolution 的全程压缩进 1 个 hidden)
      k=2 (abs/conc):  前 ⌈N/2⌉ → abstract (Global Anchor + Subject Localization)
                       后 ⌊N/2⌋ → concrete (Visual Evidence + Critical Resolution)
      k=3 (abs/br/conc): 前 ⌈N/3⌉  → abstract (Global Anchor)
                          中 ⌈N/3⌉  → bridge   (Subject Localization)
                          后 N-2*⌈N/3⌉ → concrete (Visual Evidence + Critical Resolution)

    保护:
      - 任一 stage 为空时 fallback: 取 scanpath 末尾 1 个概念填充 (永不留空 stage,
        否则训练侧 W_t 会出错).
      - 整 scanpath 太短 (< k_latent) 时, 由调用方 fallback 回旧启发式, 这里不处理.
    """
    n = len(visual_scanpath)
    if n == 0:
        return []
    sp = list(visual_scanpath)

    if k_latent == 1:
        stages = [{
            "tokens": sp,
            "stage_id": 1,
            "role": "unified",
            "from_substep": substep_id,
        }]
    elif k_latent == 2:
        half = max(1, (n + 1) // 2)
        s1 = sp[:half]
        s2 = sp[half:]
        if not s2:
            s2 = sp[-1:]
        stages = [
            {"tokens": s1, "stage_id": 1, "role": "abstract", "from_substep": substep_id},
            {"tokens": s2, "stage_id": 2, "role": "concrete", "from_substep": substep_id},
        ]
    else:  # k=3
        third = max(1, (n + 2) // 3)
        s1 = sp[:third]
        s2 = sp[third: 2 * third]
        s3 = sp[2 * third:]
        if not s2:
            s2 = sp[third-1: third] if third <= n else sp[-1:]
        if not s3:
            s3 = sp[-1:]
        stages = [
            {"tokens": s1, "stage_id": 1, "role": "abstract", "from_substep": substep_id},
            {"tokens": s2, "stage_id": 2, "role": "bridge",   "from_substep": substep_id},
            {"tokens": s3, "stage_id": 3, "role": "concrete", "from_substep": substep_id},
        ]
    return stages


# ============================================================
# Per-substep VSP call
# ============================================================

def _build_user_text(question: str, task_type: str, substep: Dict[str, Any]) -> str:
    intent = (substep.get("intent") or "").strip()
    anchor = (substep.get("anchor_phrase") or "").strip()
    k_latent = int(substep.get("k_latent") or 1)
    return (
        f"question:        {question}\n"
        f"task_type:       {task_type}\n"
        f"substep_intent:  {intent}\n"
        f"substep_anchor:  {anchor}\n"
        f"k_latent:        {k_latent}\n\n"
        f"Now produce the JSON object {{\"visual_scanpath\": [...]}} for THIS substep "
        f"following the 4-stage scanning logic. Length is YOUR decision based on "
        f"the visual demand of the substep; typically 4-10 concepts."
    )


def _call_vsp(
    *,
    client,
    auth_config: Dict,
    model: str,
    system_prompt: str,
    image_b64: str,
    question: str,
    task_type: str,
    substep: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[List[str]], str]:
    """Call opus once for a single substep. Return (visual_scanpath_list, status)."""
    user_text = _build_user_text(question, task_type, substep)
    raw = call_chat_with_image(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        image_base64=image_b64,
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
    sp = parsed.get("visual_scanpath")
    if not isinstance(sp, list) or not sp:
        return None, "missing_visual_scanpath"
    # 清洗: 仅保留非空字符串, lower, strip
    cleaned: List[str] = []
    for t in sp:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        # 限长 (防止模型违反 1-3 词约束)
        if len(s.split()) > 4:
            s = " ".join(s.split()[:4])
        cleaned.append(s)
    if not cleaned:
        return None, "empty_after_clean"
    return cleaned, "ok"


# ============================================================
# Per-sample worker
# ============================================================

def _process_one_sample(
    src_sample: Dict,
    *,
    client,
    auth_config: Dict,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[Dict], str]:
    """对单条 sample: 遍历所有 substep, 调 opus, 装回 src_sample.

    返回 (updated_sample_with_vsp, status). updated_sample 包含:
      - 原所有字段
      - substeps[i].visual_scanpath  : 新字段 (List[str])
      - substeps[i].vsp_stages       : 新字段 (List[Dict], k_latent 切分后的 stages)
                                        与 build_latent_key_tokens 单 boundary 同形.

    任一 substep VSP 失败 -> 该 substep fallback 用 build_latent_key_tokens 启发式;
    全部失败 -> 整 sample fallback (但仍走完所有 substep, 保证训练数据量不丢).
    """
    image_paths = src_sample.get("image_paths") or []
    if not image_paths:
        return None, "no_image_paths"
    img_path = image_paths[0]
    if not os.path.exists(img_path):
        return None, f"image_missing:{img_path}"
    b64 = encode_image_to_base64(img_path)
    if not b64:
        return None, "image_encode_fail"

    question = src_sample.get("question") or ""
    task_type = src_sample.get("task_type") or "spatial_relation"
    substeps = src_sample.get("substeps") or []

    # 串行遍历该 sample 的 substeps (sample 级并行已由 ThreadPoolExecutor 提供;
    # substep 级再并行会导致单 image 多次重复 encode + opus QPS 抖动)
    n_ok = 0
    n_fail = 0
    for st in substeps:
        if not isinstance(st, dict):
            continue
        k_latent = int(st.get("k_latent") or 1)
        sp_list, status = _call_vsp(
            client=client, auth_config=auth_config, model=model,
            system_prompt=system_prompt, image_b64=b64,
            question=question, task_type=task_type, substep=st,
            temperature=temperature, max_tokens=max_tokens,
        )
        if sp_list and len(sp_list) >= max(2, k_latent):
            st["visual_scanpath"] = sp_list
            st["vsp_stages"] = _slice_scanpath_to_stages(
                sp_list, k_latent, int(st.get("substep_id") or 0)
            )
            n_ok += 1
        else:
            # fallback: 单 substep 用文本启发式 (build_latent_key_tokens 处理整 sample,
            # 这里只取对应 substep 的 stage 列)
            st["visual_scanpath"] = []   # 空数组标记"VSP 失败"
            st["_vsp_fail_reason"] = status
            n_fail += 1
    if n_ok == 0:
        return src_sample, "all_substeps_fallback"
    return src_sample, f"ok({n_ok}/{n_ok+n_fail})"


# ============================================================
# Boundaries assembly
# ============================================================

def _assemble_latent_key_tokens(src_sample: Dict) -> List[List[Dict[str, Any]]]:
    """把 src_sample.substeps[*].vsp_stages 拼成 latent_key_tokens (per-boundary nested).

    若某 substep 的 visual_scanpath 为空 (失败) → 用 build_latent_key_tokens 该 substep
    的 stages 兜底 (调一次 build_latent_key_tokens 全样本, 取对应 boundary 索引).
    """
    fallback_full = build_latent_key_tokens(
        src_sample, tokens_per_stage=PREFERRED_KEY_TOKENS_PER_STAGE
    )
    out: List[List[Dict[str, Any]]] = []
    substeps = src_sample.get("substeps") or []
    for i, st in enumerate(substeps):
        if not isinstance(st, dict):
            continue
        vsp_stages = st.get("vsp_stages")
        if vsp_stages and isinstance(vsp_stages, list) and len(vsp_stages) > 0:
            out.append(vsp_stages)
        else:
            # fallback to text-heuristic stages for this substep
            if i < len(fallback_full):
                out.append(fallback_full[i])
            else:
                # ultimate fallback: 1 unified stage with intent words
                intent = (st.get("intent") or "unknown").split()[:4]
                out.append([{
                    "tokens": intent or ["unknown"],
                    "stage_id": 1,
                    "role": "unified",
                    "from_substep": int(st.get("substep_id") or i + 1),
                }])
    return out


# ============================================================
# Resume
# ============================================================

def _partial_path(out_src: str) -> str:
    return out_src.replace(".json", ".vsp_partial.json") \
        if out_src.endswith(".json") else out_src + ".vsp_partial.json"


def _save_partial(out_src: str, processed: Dict[str, Dict]):
    path = _partial_path(out_src)
    tmp = path + ".new"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"processed": processed}, f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_partial(out_src: str) -> Dict[str, Dict]:
    path = _partial_path(out_src)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("processed", {}) or {}
    except Exception as e:
        logger.warning(f"[resume] failed to load partial {path}: {e}")
        return {}


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="batch1 src json (with substeps), e.g. v6_5f_*.json")
    ap.add_argument("--slim", required=True,
                    help="batch1 training slim json (will be rewritten with new latent_key_tokens)")
    ap.add_argument("--out_src", required=True,
                    help="output src json with .substeps[*].visual_scanpath added")
    ap.add_argument("--out_slim", required=True,
                    help="output slim json with new latent_key_tokens")
    ap.add_argument("--prompt_file", default=DEFAULT_PROMPT)
    ap.add_argument("--model", default="claude-opus-4.7")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--source", default=None)
    ap.add_argument("--appid", default=None)
    ap.add_argument("--appkey", default=None)
    ap.add_argument("--rtx", default=None)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="cap (for smoke test); None = full batch")
    args = ap.parse_args()

    # ---- 鉴权 ----
    auth_config = build_auth_config_from_env_or_args(
        source=args.source, appid=args.appid, appkey=args.appkey, rtx=args.rtx,
    )
    if not auth_config:
        return 1

    # ---- 加载 ----
    logger.info(f"[load] src  : {args.src}")
    with open(args.src, "r", encoding="utf-8") as f:
        src_data: List[Dict] = json.load(f)
    logger.info(f"  -> {len(src_data)} samples in src")

    logger.info(f"[load] slim : {args.slim}")
    with open(args.slim, "r", encoding="utf-8") as f:
        slim_data: List[Dict] = json.load(f)
    logger.info(f"  -> {len(slim_data)} samples in slim")

    if args.max_samples is not None:
        src_data = src_data[: args.max_samples]
        logger.info(f"capped src to max_samples={args.max_samples}")

    # ---- prompt + client ----
    if not os.path.exists(args.prompt_file):
        logger.error(f"prompt not found: {args.prompt_file}")
        return 1
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    logger.info(f"[prompt] {args.prompt_file} ({len(system_prompt)} chars)")
    client = create_openai_client()

    # ---- resume ----
    processed: Dict[str, Dict] = _load_partial(args.out_src)
    if processed:
        logger.info(f"[resume] loaded {len(processed)} processed sample_ids")

    # ---- 并行处理 ----
    fail_reasons: Counter = Counter()
    n_total_substeps = 0
    n_ok_substeps = 0
    n_fail_substeps = 0
    t0 = time.time()

    def _task(sample: Dict) -> Tuple[str, Optional[Dict], str]:
        sid = sample.get("sample_id") or ""
        if sid in processed:
            return sid, processed[sid], "resumed"
        updated, status = _process_one_sample(
            sample,
            client=client, auth_config=auth_config,
            model=args.model, system_prompt=system_prompt,
            temperature=args.temperature, max_tokens=args.max_tokens,
        )
        return sid, updated, status

    todo = [s for s in src_data if (s.get("sample_id") or "") not in processed]
    logger.info(f"[plan] total={len(src_data)}  resumed={len(processed)}  todo={len(todo)}")

    since_save = 0
    try:
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(_task, s): s for s in todo}
            done = 0
            for fut in cf.as_completed(futures):
                done += 1
                sample = futures[fut]
                try:
                    sid, updated, status = fut.result()
                except Exception as e:
                    sid = sample.get("sample_id") or ""
                    updated = sample
                    status = f"exception:{type(e).__name__}:{str(e)[:120]}"

                if updated is None:
                    fail_reasons[status] += 1
                    logger.warning(f"[{done}/{len(todo)}] FAIL sid={sid[:50]} reason={status}")
                    continue
                processed[sid] = updated

                # substep 级统计
                n_total = len(updated.get("substeps") or [])
                n_ok_local = sum(1 for st in updated.get("substeps") or []
                                 if isinstance(st, dict) and st.get("visual_scanpath"))
                n_total_substeps += n_total
                n_ok_substeps   += n_ok_local
                n_fail_substeps += (n_total - n_ok_local)

                if not status.startswith("ok"):
                    fail_reasons[status] += 1

                if done % 20 == 0 or done <= 5:
                    elapsed = time.time() - t0
                    rate = done / max(1e-6, elapsed)
                    eta = (len(todo) - done) / max(1e-6, rate)
                    logger.info(
                        f"[{done}/{len(todo)}] sid={sid[:30]} status={status} "
                        f"| rate={rate:.2f}/s ETA={eta/60:.1f}min "
                        f"| substeps OK={n_ok_substeps}/{n_total_substeps}"
                    )

                since_save += 1
                if since_save >= args.save_every:
                    _save_partial(args.out_src, processed)
                    since_save = 0
                    logger.info(f"  [partial] saved {len(processed)} samples")
    except KeyboardInterrupt:
        logger.warning("[interrupt] saving partial before exit")
        _save_partial(args.out_src, processed)
        raise

    _save_partial(args.out_src, processed)
    logger.info(f"\n[partial final] {len(processed)} samples processed; "
                f"substeps OK={n_ok_substeps}/{n_total_substeps} "
                f"({100*n_ok_substeps/max(1,n_total_substeps):.1f}%)")

    # ---- 重写 src + slim ----
    # 1) src: 把 processed 写回 (按原 src_data 顺序)
    new_src: List[Dict] = []
    for s in src_data:
        sid = s.get("sample_id") or ""
        new_src.append(processed.get(sid, s))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_src)) or ".", exist_ok=True)
    with open(args.out_src, "w", encoding="utf-8") as f:
        json.dump(new_src, f, ensure_ascii=False, indent=2)
    logger.info(f"[write] {args.out_src} ({len(new_src)} samples with visual_scanpath)")

    # 2) slim: 用新 src 重新 build latent_key_tokens, 但 slim 的其他字段保留
    # ----------------------------------------------------------
    # ⚠️ Bug 修复 (2026-05-18): 旧版仅用 image_path 当 dict key, 但 src 中常见
    #   "同 image 不同 question" 的样本 (b1+b2+b3 共 4957 对). dict 互相覆盖 →
    #   slim 中 reasoning_for_training/question 与 latent_key_tokens 错配 18%.
    #
    # 修复:
    #   1) 用 (image_path, question) 联合 key 索引 slim, 这两个字段在 slim 中都
    #      一定存在, 且联合后唯一性高很多.
    #   2) generate_v6 阶段允许 LLM 改写 question, 因此联合 key 找不到时用
    #      image_path 作为 fallback (但仅当 image_path 在 slim 唯一时), 避免
    #      retro-compat 场景退化到老 bug.
    #   3) 用 set 跟踪已 append 过的 slim 引用, 防止单条 slim 被多条 src 重复
    #      映射 (老 bug 的另一面: 同一条 slim 被覆盖 N 次 + 重复 append).
    # ----------------------------------------------------------
    slim_by_iq: Dict[tuple, Dict] = {}
    slim_by_img: Dict[str, Dict] = {}
    img_seen_count: Dict[str, int] = {}
    n_iq_dup = 0
    for r in slim_data:
        ip = r.get("image_path", "")
        q = r.get("question", "")
        iq_key = (ip, q)
        if iq_key in slim_by_iq:
            n_iq_dup += 1
        slim_by_iq[iq_key] = r
        # 跟踪每个 image_path 出现次数; 仅唯一时才提供 img-only fallback
        img_seen_count[ip] = img_seen_count.get(ip, 0) + 1
        if img_seen_count[ip] == 1:
            slim_by_img[ip] = r
        else:
            # 重复 image_path: 把 fallback 标 None, 强制走 (img,q) 联合 key
            slim_by_img[ip] = None
    n_img_dup = sum(1 for c in img_seen_count.values() if c > 1)
    if n_iq_dup > 0:
        logger.warning(
            f"[slim-index] {n_iq_dup} duplicate (image_path, question) pairs "
            f"in slim — last write wins (rare)"
        )
    if n_img_dup > 0:
        logger.info(
            f"[slim-index] {n_img_dup} image_paths used by multiple slim records "
            f"— img-only fallback disabled for these"
        )

    new_slim: List[Dict] = []
    n_changed = 0
    n_unchanged = 0
    n_missing_in_slim = 0
    n_matched_by_iq = 0
    n_matched_by_img_fallback = 0
    n_skipped_dup_slim = 0
    seen_slim_ids: set = set()  # id(rec) 去重, 防止单条 slim 被重复 append
    for s in new_src:
        sid_img_paths = s.get("image_paths") or []
        if not sid_img_paths:
            continue
        img_key = sid_img_paths[0]
        q_key = s.get("question", "")
        # 优先 (image, question) 联合 key
        rec = slim_by_iq.get((img_key, q_key))
        if rec is not None:
            n_matched_by_iq += 1
        else:
            # fallback: 仅 image_path (但仅当 slim 中该 image 唯一时)
            rec = slim_by_img.get(img_key)
            if rec is not None:
                n_matched_by_img_fallback += 1
        if rec is None:
            n_missing_in_slim += 1
            continue
        # 防止同一条 slim 被多条 src 重复映射 (老 bug 的另一面)
        if id(rec) in seen_slim_ids:
            n_skipped_dup_slim += 1
            continue
        seen_slim_ids.add(id(rec))
        old_lkt = rec.get("latent_key_tokens")
        new_lkt = _assemble_latent_key_tokens(s)
        if old_lkt != new_lkt:
            n_changed += 1
        else:
            n_unchanged += 1
        rec_new = dict(rec)
        rec_new["latent_key_tokens"] = new_lkt
        new_slim.append(rec_new)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_slim)) or ".", exist_ok=True)
    with open(args.out_slim, "w", encoding="utf-8") as f:
        json.dump(new_slim, f, ensure_ascii=False, indent=2)

    logger.info(f"\n[write] {args.out_slim} ({len(new_slim)} slim samples)")
    logger.info(f"  latent_key_tokens changed   : {n_changed}")
    logger.info(f"  latent_key_tokens unchanged : {n_unchanged}")
    logger.info(f"  matched by (image, question): {n_matched_by_iq}")
    logger.info(f"  matched by image fallback   : {n_matched_by_img_fallback}")
    logger.info(f"  src missing in slim         : {n_missing_in_slim}")
    logger.info(f"  src skipped (dup slim ref)  : {n_skipped_dup_slim}")

    # ---- 自检: 写完后校验 pause 数 == boundary 数 ----
    n_align = 0
    for r in new_slim:
        n_pause = (r.get("reasoning_for_training") or "").count("<|pause|>")
        n_boundary = len(r.get("latent_key_tokens", []))
        if n_pause == n_boundary:
            n_align += 1
    align_rate = 100.0 * n_align / max(1, len(new_slim))
    logger.info(
        f"  pause/boundary alignment    : {n_align}/{len(new_slim)} "
        f"({align_rate:.2f}%)"
    )
    if align_rate < 99.0:
        logger.warning(
            f"⚠️ alignment rate {align_rate:.2f}% < 99% — "
            f"check if generate_v6 produced inconsistent cot_text vs n_substeps"
        )

    # ---- 最终 report ----
    elapsed = time.time() - t0
    report = {
        "src_in":   args.src,
        "slim_in":  args.slim,
        "src_out":  args.out_src,
        "slim_out": args.out_slim,
        "model":    args.model,
        "n_src_samples":      len(new_src),
        "n_slim_samples":     len(new_slim),
        "n_lkt_changed":      n_changed,
        "n_lkt_unchanged":    n_unchanged,
        "n_substeps_ok":      n_ok_substeps,
        "n_substeps_fail":    n_fail_substeps,
        "n_substeps_total":   n_total_substeps,
        "fail_reasons":       dict(fail_reasons),
        # ---- v2 索引修复后的诊断信息 (2026-05-18) ----
        "n_matched_by_iq":           n_matched_by_iq,
        "n_matched_by_img_fallback": n_matched_by_img_fallback,
        "n_skipped_dup_slim":        n_skipped_dup_slim,
        "n_missing_in_slim":         n_missing_in_slim,
        "slim_iq_dup_pairs":         n_iq_dup,
        "slim_img_dup_records":      n_img_dup,
        "pause_boundary_aligned":    f"{n_align}/{len(new_slim)}",
        "pause_boundary_align_rate": round(align_rate, 4),
        # ---- 计时 ----
        "elapsed_seconds":    round(elapsed, 1),
        "elapsed_minutes":    round(elapsed / 60, 1),
    }
    rep_path = args.out_slim.replace(".json", ".vsp_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\n[report] {rep_path}")
    logger.info(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
