"""build_multi_source_seeds.py - v6.5b: 多源真实场景 seed 池构建

解决 v6_5c 样本图像 100% VCR 单一源问题.

支持源:
  1. VCR (电影截图, 备用 + 兜底) — 从 stage2_v2 pool 池抽
  2. RealWorldQA (手机拍摄真实场景, parquet+内嵌 PNG)
  3. BLINK (多任务视觉推理, 13 类: 空间/深度/多视角/计数/几何...)
  4. MUIRBENCH (多图推理, parquet)
  5. MME-RealWorld (高分辨率真实场景: 自动驾驶/监控/图表)

输出: jsonl, 每行一条 seed dict, 可被 generate_v6.py --seed_file 直接消费.

seed dict schema (与 generate_v5 auto_seed_from_stage2v2 一致):
    {
        "sample_id":             str,    # 唯一标识
        "source_id":             str,    # 源数据集原始 id
        "data_source":           str,    # 'realworldqa' / 'blink_<sub>' / 'muirbench' / 'mme_realworld_<subtask>' / 'vcr_stage2v2'
        "reference_question":    str,    # 原 question (只用作风格参考)
        "reference_answer_long": str,    # 原 answer
        "task_type":             str,    # 推例映射到 v6 task_type
        "image_paths":           [str],  # 绝对路径 (如果源是 parquet 则先 dump 为磁盘文件)
        "num_images":            int,    # 总是 1 (只抽单图样本)
    }
"""
import argparse
import io
import json
import os
import random
import sys
import glob
from collections import defaultdict
from typing import Dict, List, Optional

DATA_ROOT = "./data"
EXPORT_ROOT = f"{DATA_ROOT}/v6_seeds_multisource"

# ----------------- task_type 映射 -----------------
# 将各源原始标签 → v6 统一 task_type (与 generator_v6.txt SECTION C 对应)
V6_TASK_TYPES = [
    "spatial_relation", "geometry", "perspective",
    "physical", "comparison", "counting", "fine_grained",
    "ocr", "counterfactual", "multi_hypothesis", "detail_observation",
]

BLINK_SUBTASK_TO_V6 = {
    "Spatial_Relation": "spatial_relation",
    "Relative_Depth":   "perspective",
    "Counting":         "counting",
    "Object_Localization": "spatial_relation",
    "Visual_Similarity": "comparison",
    "Forensic_Detection": "detail_observation",
    "IQ_Test":           "multi_hypothesis",
    "Art_Style":         "fine_grained",
    "Relative_Reflectance": "physical",
    # 跳过必须多图能力的任务: Jigsaw / Multi-view_Reasoning /
    # Visual_Correspondence / Semantic_Correspondence / Functional_Correspondence
    # 这些任务本质上需要对比 2-3 张图, 单图不适用.
}


def _make_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _save_image_from_bytes(img_bytes: bytes, out_path: str) -> bool:
    """从 parquet 内嵌的 image bytes 存为磁盘文件"""
    try:
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        print(f"  [WARN] save image failed: {e}", flush=True)
        return False


def _extract_image_bytes(img_field) -> Optional[bytes]:
    """parquet image 字段可能是 dict({'bytes':..., 'path':...}) 或直接 bytes"""
    if img_field is None:
        return None
    if isinstance(img_field, bytes):
        return img_field
    if isinstance(img_field, dict):
        b = img_field.get("bytes")
        if isinstance(b, bytes):
            return b
    # 如果是 PIL.Image, 转为 PNG bytes
    try:
        buf = io.BytesIO()
        img_field.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ============================================================
# Source 1: RealWorldQA
# ============================================================
def build_realworldqa_seeds(n: int, rng: random.Random) -> List[Dict]:
    """RealWorldQA: 手机拍摄真实场景, 单图 + 单问"""
    import pandas as pd
    parquet_files = sorted(glob.glob(f"{DATA_ROOT}/RealWorldQA/data/test-*.parquet"))
    if not parquet_files:
        print("[realworldqa] no parquet files found", flush=True)
        return []

    out_img_dir = _make_dir(f"{EXPORT_ROOT}/realworldqa_images")
    seeds = []
    print(f"[realworldqa] reading {len(parquet_files)} parquet files...", flush=True)
    df_list = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"[realworldqa] total {len(df)} rows, cols={list(df.columns)}", flush=True)

    # cols expected: image, question, answer (or query, choices, label)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    indices = indices[:min(n * 2, len(df))]  # 多抽一些以防失败
    for idx in indices:
        if len(seeds) >= n:
            break
        row = df.iloc[idx]
        img_field = row.get("image")
        img_bytes = _extract_image_bytes(img_field)
        if img_bytes is None:
            continue
        out_p = f"{out_img_dir}/rw_{idx:05d}.png"
        if not os.path.exists(out_p):
            if not _save_image_from_bytes(img_bytes, out_p):
                continue
        # question 在不同版本方法不同
        q = row.get("question") or row.get("query") or ""
        a = row.get("answer") or row.get("label") or ""
        seeds.append({
            "sample_id":             f"rw_{idx:05d}",
            "source_id":             f"realworldqa_{idx}",
            "data_source":           "realworldqa",
            "reference_question":    str(q)[:500],
            "reference_answer_long": str(a)[:500],
            "task_type":             rng.choice(V6_TASK_TYPES),
            "image_paths":           [out_p],
            "num_images":            1,
        })
    print(f"[realworldqa] built {len(seeds)} seeds", flush=True)
    return seeds


# ============================================================
# Source 2: BLINK (13 sub-tasks, mostly real-world)
# ============================================================
def build_blink_seeds(n: int, rng: random.Random) -> List[Dict]:
    """BLINK: 13 个子任务, 单图抽 image_1 列 (过滤多图样本)"""
    import pandas as pd
    blink_root = f"{DATA_ROOT}/BLINK"
    sub_tasks = [d for d in os.listdir(blink_root)
                 if os.path.isdir(f"{blink_root}/{d}") and d in BLINK_SUBTASK_TO_V6]
    print(f"[blink] sub-tasks: {sub_tasks}", flush=True)
    out_img_dir = _make_dir(f"{EXPORT_ROOT}/blink_images")

    per_task = max(1, n // len(sub_tasks))
    seeds: List[Dict] = []
    for sub in sub_tasks:
        if len(seeds) >= n:
            break
        # 优先 val (有 answer label), 后 fallback test
        for split in ["val", "test"]:
            files = glob.glob(f"{blink_root}/{sub}/{split}-*.parquet")
            if files:
                break
        if not files:
            continue
        try:
            df = pd.read_parquet(files[0])
        except Exception as e:
            print(f"[blink] {sub} read err: {e}", flush=True)
            continue
        # v6.5b: 不强制单图 —— 只取 image_1 入场, 多图任务也能产生有意义 seed
        # 判断只要 image_1 有实际 bytes 即可
        indices = list(range(len(df)))
        rng.shuffle(indices)
        cnt = 0
        for idx in indices[:per_task * 2]:
            if cnt >= per_task:
                break
            row = df.iloc[idx]
            img_field = row.get("image_1")
            img_bytes = _extract_image_bytes(img_field)
            if img_bytes is None:
                continue
            out_p = f"{out_img_dir}/blink_{sub}_{idx:05d}.png"
            if not os.path.exists(out_p):
                if not _save_image_from_bytes(img_bytes, out_p):
                    continue
            q = row.get("question") or ""
            a = row.get("answer") or ""
            choices = row.get("choices")
            if isinstance(choices, (list, tuple)) and len(choices) > 0:
                q = f"{q} Options: {' / '.join(str(c) for c in choices)}"
            seeds.append({
                "sample_id":             f"blink_{sub}_{idx:05d}",
                "source_id":             f"blink/{sub}/{idx}",
                "data_source":           f"blink_{sub.lower()}",
                "reference_question":    str(q)[:500],
                "reference_answer_long": str(a)[:500],
                "task_type":             BLINK_SUBTASK_TO_V6.get(sub, "spatial_relation"),
                "image_paths":           [out_p],
                "num_images":            1,
            })
            cnt += 1
        print(f"[blink] {sub}: +{cnt}", flush=True)
    print(f"[blink] total {len(seeds)} seeds", flush=True)
    return seeds[:n]


# ============================================================
# Source 3: MUIRBENCH (single image samples only)
# ============================================================
def build_muirbench_seeds(n: int, rng: random.Random) -> List[Dict]:
    import pandas as pd
    files = sorted(glob.glob(f"{DATA_ROOT}/MUIRBENCH/data/test-*.parquet"))
    if not files:
        return []
    out_img_dir = _make_dir(f"{EXPORT_ROOT}/muirbench_images")
    seeds = []
    print(f"[muirbench] {len(files)} parquet files", flush=True)
    for fp in files:
        if len(seeds) >= n:
            break
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        # MUIRBENCH 可能为 multi-image, 只抽单图
        # 检查列名
        cols = list(df.columns)
        if "image_list" in cols:
            df = df[df["image_list"].apply(lambda x: x is not None and len(x) == 1)]
        indices = list(range(len(df)))
        rng.shuffle(indices)
        for idx in indices:
            if len(seeds) >= n:
                break
            row = df.iloc[idx]
            img_bytes = None
            if "image_list" in cols:
                imgs = row["image_list"]
                if imgs is not None and len(imgs) >= 1:
                    img_bytes = _extract_image_bytes(imgs[0])
            elif "image" in cols:
                img_bytes = _extract_image_bytes(row["image"])
            if img_bytes is None:
                continue
            out_p = f"{out_img_dir}/mb_{idx:05d}.png"
            if not os.path.exists(out_p):
                if not _save_image_from_bytes(img_bytes, out_p):
                    continue
            q = row.get("question") or ""
            a = row.get("answer") or ""
            seeds.append({
                "sample_id":             f"mb_{idx:05d}",
                "source_id":             f"muirbench_{idx}",
                "data_source":           "muirbench",
                "reference_question":    str(q)[:500],
                "reference_answer_long": str(a)[:500],
                "task_type":             rng.choice(V6_TASK_TYPES),
                "image_paths":           [out_p],
                "num_images":            1,
            })
    print(f"[muirbench] built {len(seeds)} seeds", flush=True)
    return seeds


# ============================================================
# Source 4: MME-RealWorld (替代 MMStar, 真实场景高分辨率)
# ============================================================
MME_RAW_ROOT = f"{DATA_ROOT}/MME_RealWorld_raw"

# Subtask -> v6 task_type 映射
MME_SUBTASK_TO_V6 = {
    "Autonomous_Driving":       "spatial_relation",   # 自动驾驶物体/方向/意图
    "Monitoring":               "detail_observation", # 监控场景细节观察/计数
    "Diagram and Table":        "fine_grained",       # 图表细粒度阅读
    "OCR with Complex Context": "ocr",
    "Remote Sensing":           "counting",           # 遥感主要是 counting/位置
}

# v6.5g (batch2): 启用 Remote Sensing (DOTA-v2 卫星图, 多对象千万复杂场景, 作为视角互补)
# OCR_CC 仍然不启用: 单步 OCR 与 v6 多步推理目标不符 + 选项模板 bias 严重
_MME_ENABLED_SUBTASKS_DEFAULT = {
    "Autonomous_Driving",
    "Monitoring",
    "Diagram and Table",
}

_MME_ENABLED_SUBTASKS_BATCH2 = {
    "Autonomous_Driving",
    "Monitoring",
    "Diagram and Table",
    "Remote Sensing",
}


def build_mme_realworld_seeds(n: int, rng: random.Random,
                              enabled_subtasks: Optional[set] = None,
                              exclude_ids: Optional[set] = None) -> List[Dict]:
    """MME-RealWorld: 高分辨率真实场景, 5 大子任务, 单图单问.

    JSON entry schema:
        Question_id, Question Type, Image (相对路径), Text, Answer choices [list],
        Ground truth, Task, Subtask, Category, Dataset

    Args:
        enabled_subtasks: 指定启用的 subtask 集合 (默认: 3 件套)
        exclude_ids: 需排除的 source_id 集合 (与 batch1 注入的 used_ids 去重)
    """
    if enabled_subtasks is None:
        enabled_subtasks = _MME_ENABLED_SUBTASKS_DEFAULT
    if exclude_ids is None:
        exclude_ids = set()
    json_p = f"{MME_RAW_ROOT}/MME_RealWorld.json"
    if not os.path.exists(json_p):
        print(f"[mme_realworld] {json_p} not found, skip", flush=True)
        return []
    print(f"[mme_realworld] loading {json_p} ... (enabled={sorted(enabled_subtasks)}, exclude={len(exclude_ids)})", flush=True)
    raw = json.load(open(json_p))

    # 按启用 subtask 过滤 + 必须图片实际存在 + 排除已用 ID
    candidates = []
    miss_img = 0
    excluded = 0
    for item in raw:
        sub = item.get("Subtask", "")
        if sub not in enabled_subtasks:
            continue
        qid = item.get("Question_id", "")
        if qid in exclude_ids:
            excluded += 1
            continue
        rel_img = item.get("Image", "")
        if not rel_img:
            continue
        abs_img = f"{MME_RAW_ROOT}/{rel_img}"
        if not os.path.exists(abs_img):
            miss_img += 1
            continue
        candidates.append((abs_img, item))
    print(f"[mme_realworld] excluded by used_ids: {excluded}", flush=True)
    print(f"[mme_realworld] candidates after filter: {len(candidates)} (miss_img={miss_img})", flush=True)
    if not candidates:
        return []

    # 按 subtask 均衡采样
    by_sub: Dict[str, List] = defaultdict(list)
    for ap_, it in candidates:
        by_sub[it["Subtask"]].append((ap_, it))
    print(f"[mme_realworld] by-subtask: { {k: len(v) for k,v in by_sub.items()} }", flush=True)

    per_sub = max(1, n // len(by_sub))
    seeds: List[Dict] = []
    for sub, lst in by_sub.items():
        rng.shuffle(lst)
        take = min(per_sub, len(lst))
        for abs_img, item in lst[:take]:
            qid = item.get("Question_id", "").replace("/", "_")
            q = item.get("Text", "") or ""
            choices = item.get("Answer choices") or []
            if isinstance(choices, list) and choices:
                q = f"{q} Options: {' / '.join(str(c) for c in choices)}"
            gt = item.get("Ground truth", "") or ""
            seeds.append({
                "sample_id":             f"mme_{qid[-60:]}",
                "source_id":             item.get("Question_id", ""),
                "data_source":           f"mme_realworld_{sub.lower().replace(' ', '_')}",
                "reference_question":    str(q)[:500],
                "reference_answer_long": str(gt)[:500],
                "task_type":             MME_SUBTASK_TO_V6.get(sub, "detail_observation"),
                "image_paths":           [abs_img],
                "num_images":            1,
            })
        print(f"[mme_realworld] {sub}: +{take}", flush=True)

    # 不足总数, 再从未取的 candidates 里补足
    if len(seeds) < n:
        existing = {s["sample_id"] for s in seeds}
        leftover = []
        for ap_, it in candidates:
            qid = it.get("Question_id", "").replace("/", "_")
            sid = f"mme_{qid[-60:]}"
            if sid not in existing:
                leftover.append((ap_, it))
        rng.shuffle(leftover)
        for abs_img, item in leftover[: n - len(seeds)]:
            qid = item.get("Question_id", "").replace("/", "_")
            sub = item.get("Subtask", "")
            q = item.get("Text", "") or ""
            choices = item.get("Answer choices") or []
            if isinstance(choices, list) and choices:
                q = f"{q} Options: {' / '.join(str(c) for c in choices)}"
            gt = item.get("Ground truth", "") or ""
            seeds.append({
                "sample_id":             f"mme_{qid[-60:]}",
                "source_id":             item.get("Question_id", ""),
                "data_source":           f"mme_realworld_{sub.lower().replace(' ', '_')}",
                "reference_question":    str(q)[:500],
                "reference_answer_long": str(gt)[:500],
                "task_type":             MME_SUBTASK_TO_V6.get(sub, "detail_observation"),
                "image_paths":           [abs_img],
                "num_images":            1,
            })

    print(f"[mme_realworld] built {len(seeds)} seeds", flush=True)
    return seeds[:n]


# ============================================================
# Source 5: VCR (备用 + 兜底, 从 stage2_v2 pool 抽)
# ============================================================
def build_vcr_seeds(n: int, rng: random.Random,
                    exclude_ids: Optional[set] = None) -> List[Dict]:
    if exclude_ids is None:
        exclude_ids = set()
    pool_p = f"{DATA_ROOT}/stage2_v2/stage2_hybrid_90k_latent_only.json"
    print(f"[vcr] loading {pool_p} ... (exclude {len(exclude_ids)})", flush=True)
    data = json.load(open(pool_p))
    data = [s for s in data if s.get("data_source") == "vcr_erqa_style"]
    print(f"[vcr] pool size={len(data)}", flush=True)
    rng.shuffle(data)
    seeds = []
    excluded = 0
    for s in data:
        if len(seeds) >= n:
            break
        img = s.get("image_path") or s.get("image", "")
        if not img:
            continue
        # 路径如果是相对路径, 补全为绝对
        if not os.path.isabs(img):
            img = f"./{img}" if not img.startswith("/") else img
        sid = s.get("image", "")
        if sid in exclude_ids:
            excluded += 1
            continue
        seeds.append({
            "sample_id":             f"vcr_{sid.replace('/', '_')[-50:]}",
            "source_id":             sid,
            "data_source":           "vcr_stage2v2",
            "reference_question":    s.get("question", "")[:500],
            "reference_answer_long": s.get("answer", "")[:500],
            "task_type":             s.get("task_type", "spatial_relation"),
            "image_paths":           [img],
            "num_images":            1,
        })
    print(f"[vcr] built {len(seeds)} seeds (excluded {excluded})", flush=True)
    return seeds


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_total",      type=int, default=200, help="总 seed 数量")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--ratio_vcr",    type=float, default=0.30)
    ap.add_argument("--ratio_realworldqa", type=float, default=0.10)
    ap.add_argument("--ratio_blink",  type=float, default=0.15)
    ap.add_argument("--ratio_muirbench", type=float, default=0.0)  # 全量多图, 不适合单图范式
    ap.add_argument("--ratio_mme_realworld", type=float, default=0.45)  # 替代 mmstar 的真实场景主源
    ap.add_argument("--vcr_fill", action="store_true", default=True,
                    help="VCR 兜底: 其他源不足时用 VCR 补到 n_total (默认开启)")
    ap.add_argument("--no_vcr_fill", dest="vcr_fill", action="store_false")
    ap.add_argument("--output",       type=str, required=True, help="输出 jsonl 路径")
    ap.add_argument("--sources", type=str, default="vcr,realworldqa,blink,mme_realworld",
                    help="逗号分隔的源名子集")
    ap.add_argument("--exclude_ids_file", type=str, default="",
                    help="一行一个 source_id 的 txt, 该些 ID 不会被采入 (用于 batch2 去重 batch1)")
    ap.add_argument("--enable_remote_sensing", action="store_true", default=False,
                    help="启用 MME-RealWorld Remote Sensing 子任务 (DOTA-v2 卫星图)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    # 加载 exclude_ids
    exclude_ids: set = set()
    if args.exclude_ids_file and os.path.exists(args.exclude_ids_file):
        with open(args.exclude_ids_file) as f:
            exclude_ids = {line.strip() for line in f if line.strip()}
        print(f"[exclude] loaded {len(exclude_ids)} ids from {args.exclude_ids_file}", flush=True)

    mme_subtasks = (
        _MME_ENABLED_SUBTASKS_BATCH2 if args.enable_remote_sensing
        else _MME_ENABLED_SUBTASKS_DEFAULT
    )
    print(f"[mme] enabled subtasks: {sorted(mme_subtasks)}", flush=True)

    n_per = {
        "vcr":           int(args.n_total * args.ratio_vcr),
        "realworldqa":   int(args.n_total * args.ratio_realworldqa),
        "blink":         int(args.n_total * args.ratio_blink),
        "muirbench":     int(args.n_total * args.ratio_muirbench),
        "mme_realworld": int(args.n_total * args.ratio_mme_realworld),
    }
    print(f"target per-source counts: {n_per}", flush=True)

    all_seeds: List[Dict] = []
    # 先跑非 VCR 源 (它们池子小, 实际产出可能小于配额)
    if "realworldqa"   in sources: all_seeds += build_realworldqa_seeds(n_per["realworldqa"], rng)
    if "blink"         in sources: all_seeds += build_blink_seeds(n_per["blink"], rng)
    if "muirbench"     in sources: all_seeds += build_muirbench_seeds(n_per["muirbench"], rng)
    if "mme_realworld" in sources:
        all_seeds += build_mme_realworld_seeds(
            n_per["mme_realworld"], rng,
            enabled_subtasks=mme_subtasks, exclude_ids=exclude_ids,
        )

    # VCR 最后跑: 默认配额 + 兑底剩余缺口 (因为 VCR 池子 30K 充足)
    if "vcr" in sources:
        vcr_quota = n_per["vcr"]
        if args.vcr_fill:
            shortage = args.n_total - len(all_seeds) - vcr_quota
            if shortage > 0:
                print(f"[vcr-fill] other sources produced {len(all_seeds)}; VCR base quota={vcr_quota} + fill={shortage} = {vcr_quota + shortage}", flush=True)
                vcr_quota += shortage
        all_seeds += build_vcr_seeds(vcr_quota, rng, exclude_ids=exclude_ids)
    rng.shuffle(all_seeds)
    print(f"\n[total] {len(all_seeds)} seeds (target {args.n_total})", flush=True)
    src_dist = defaultdict(int)
    for s in all_seeds:
        src_dist[s["data_source"]] += 1
    print(f"[final source dist] {dict(src_dist)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in all_seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
