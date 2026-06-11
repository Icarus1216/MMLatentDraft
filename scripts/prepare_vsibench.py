#!/usr/bin/env python3
"""
prepare_vsibench.py - VSI-Bench 数据预处理

功能:
  1. 解压 arkitscenes.zip / scannet.zip / scannetpp.zip 到 data/VSI-Bench/videos/
  2. 对每个场景视频均匀抽取 N 帧 (默认 16 帧), 保存为 JPEG
  3. 从 test.jsonl 生成标准评测 jsonl:
       - 选择题 (有 options): answer_mode=choice_abcd, answer=字母
       - 数值题 (无 options): answer_mode=numerical, answer=数值字符串
  4. 输出到 data/VSI-Bench_eval/vsibench_test.jsonl

用法:
  python3 scripts/prepare_vsibench.py
  python3 scripts/prepare_vsibench.py --num_frames 16 --config full
  python3 scripts/prepare_vsibench.py --config debiased   # 只用去偏子集
"""
import os
import sys
import json
import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
VSI_SRC = DATA_ROOT / "VSI-Bench"
VSI_OUT = DATA_ROOT / "VSI-Bench_eval"

# 每个场景抽取的帧数 (VSI-Bench 官方用 16 帧)
DEFAULT_NUM_FRAMES = 16

# 数值题的提示后缀
NUMERICAL_SUFFIX = "\nPlease answer with a single number (integer or decimal). Do not include units."
# 选择题的提示后缀
CHOICE_SUFFIX = "\nAnswer with just the option letter (A/B/C/D)."


def _extract_zips(video_dir: Path):
    """解压三个 zip 到 video_dir/{dataset}/ 目录下 (已存在则跳过)."""
    video_dir.mkdir(parents=True, exist_ok=True)
    for ds in ["arkitscenes", "scannet", "scannetpp"]:
        zip_path = VSI_SRC / f"{ds}.zip"
        out_dir = video_dir / ds
        if not zip_path.exists():
            print(f"  [skip] {zip_path} 不存在", flush=True)
            continue
        # 检查是否已解压 (目录存在且非空)
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  [skip] {out_dir} 已存在, 跳过解压", flush=True)
            continue
        print(f"  解压 {zip_path.name} → {out_dir} ...", flush=True)
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(video_dir))
        print(f"  ✅ {ds} 解压完成", flush=True)


def _extract_frames(video_path: Path, out_dir: Path, num_frames: int) -> list:
    """从视频均匀抽取 num_frames 帧, 保存为 JPEG, 返回相对于 VSI_OUT 的路径列表."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 检查是否已抽帧
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= num_frames:
        return [str(p.relative_to(VSI_OUT)) for p in existing[:num_frames]]

    try:
        import cv2
    except ImportError:
        raise ImportError("需要 opencv-python: pip install opencv-python-headless")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"无法读取视频帧数: {video_path}")

    # 均匀采样索引
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        indices = [int(i * step) for i in range(num_frames)]

    saved = []
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = out_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(str(out_path.relative_to(VSI_OUT)))
    cap.release()
    return saved


def prepare_vsibench(num_frames: int = DEFAULT_NUM_FRAMES, config: str = "full"):
    """主预处理函数.

    config: 'full' | 'debiased'
      - full: 使用全部 5130 条
      - debiased: 只使用 pruned=False 的子集 (需要 pruned_ids.txt)
    """
    print(f"\n[VSI-Bench] 输出 → {VSI_OUT}  num_frames={num_frames}  config={config}", flush=True)

    # ---- 1. 解压视频 ----
    video_dir = VSI_SRC / "videos"
    print("\n[1/3] 解压视频 zip...", flush=True)
    _extract_zips(video_dir)

    # ---- 2. 读取 pruned_ids (debiased 模式) ----
    pruned_ids = set()
    if config == "debiased":
        pruned_file = VSI_SRC / "pruned_ids.txt"
        if pruned_file.exists():
            with open(pruned_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pruned_ids.add(int(line))
            print(f"  debiased 模式: 已加载 {len(pruned_ids)} 个 pruned id", flush=True)
        else:
            print(f"  ⚠️ pruned_ids.txt 不存在, 退化为 full 模式", flush=True)
            config = "full"

    # ---- 3. 读取 test.jsonl ----
    src_jsonl = VSI_SRC / "test.jsonl"
    assert src_jsonl.exists(), f"{src_jsonl} 不存在"
    raw_data = []
    with open(src_jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))
    print(f"\n[2/3] 原始数据: {len(raw_data)} 条", flush=True)

    if config == "debiased":
        raw_data = [d for d in raw_data if d['id'] not in pruned_ids]
        print(f"  debiased 过滤后: {len(raw_data)} 条", flush=True)

    # ---- 4. 抽帧 + 生成 jsonl ----
    VSI_OUT.mkdir(parents=True, exist_ok=True)
    frames_dir = VSI_OUT / "frames"
    suffix = "_debiased" if config == "debiased" else ""
    out_jsonl = VSI_OUT / f"vsibench_test{suffix}.jsonl"

    print(f"\n[3/3] 抽帧 + 生成 jsonl → {out_jsonl}", flush=True)

    # 预先收集所有需要处理的 scene_name, 避免重复抽帧
    scene_to_frames = {}  # (dataset, scene_name) -> [frame_rel_paths]

    n_ok = 0
    n_skip = 0
    n_choice = 0
    n_numerical = 0

    with open(out_jsonl, 'w', encoding='utf-8') as fout:
        for i, row in enumerate(raw_data):
            rid = row['id']
            dataset = row['dataset']       # arkitscenes / scannet / scannetpp
            scene_name = row['scene_name']
            question_type = row['question_type']
            question_raw = row['question']
            ground_truth = str(row['ground_truth']).strip()
            options = row.get('options')   # None 或 list

            # 视频路径
            video_path = video_dir / dataset / f"{scene_name}.mp4"
            if not video_path.exists():
                print(f"  [skip] 视频不存在: {video_path}", flush=True)
                n_skip += 1
                continue

            # 抽帧 (同一场景复用)
            scene_key = (dataset, scene_name)
            if scene_key not in scene_to_frames:
                frame_out_dir = frames_dir / dataset / scene_name
                try:
                    frame_paths = _extract_frames(video_path, frame_out_dir, num_frames)
                    scene_to_frames[scene_key] = frame_paths
                except Exception as e:
                    print(f"  [skip] 抽帧失败 {scene_name}: {e}", flush=True)
                    scene_to_frames[scene_key] = []
            frame_paths = scene_to_frames[scene_key]
            if not frame_paths:
                n_skip += 1
                continue

            # 构建问题文本
            if options:
                # 选择题: options 形如 ["A. front-left", "B. back-right", ...]
                opts_str = "\n".join(options)
                full_q = f"{question_raw}\nOptions:\n{opts_str}{CHOICE_SUFFIX}"
                answer_mode = "choice_abcd"
                # ground_truth 已经是字母 "A"/"B"/"C"/"D"
                answer = ground_truth.strip().upper()
                if answer not in "ABCD":
                    # 尝试从选项文本中提取
                    for opt in options:
                        if opt.startswith(ground_truth):
                            answer = opt[0].upper()
                            break
                n_choice += 1
            else:
                # 数值题
                full_q = f"{question_raw}{NUMERICAL_SUFFIX}"
                answer_mode = "numerical"
                answer = ground_truth
                n_numerical += 1

            rec = {
                "question_id": f"vsibench_{rid:05d}",
                "question": full_q,
                "answer": answer,
                "answer_mode": answer_mode,
                "image_paths": frame_paths,
                "question_type": f"VSI-Bench/{question_type}",
                "dataset": dataset,
                "scene_name": scene_name,
                "num_frames": len(frame_paths),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

            if n_ok % 200 == 0:
                print(f"  已处理 {n_ok} 条 (skip={n_skip})...", flush=True)

    print(f"\n[VSI-Bench] ✅ 完成:", flush=True)
    print(f"  总条数: {n_ok}  (跳过: {n_skip})", flush=True)
    print(f"  选择题: {n_choice}  数值题: {n_numerical}", flush=True)
    print(f"  输出: {out_jsonl}", flush=True)
    return out_jsonl, n_ok


def main():
    parser = argparse.ArgumentParser(description="VSI-Bench 数据预处理")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help=f"每个场景抽取的帧数 (默认 {DEFAULT_NUM_FRAMES})")
    parser.add_argument("--config", default="full", choices=["full", "debiased"],
                        help="full=全量5130条, debiased=去偏子集~2363条")
    args = parser.parse_args()
    prepare_vsibench(num_frames=args.num_frames, config=args.config)


if __name__ == "__main__":
    main()
