#!/usr/bin/env python3
"""
批量下载 VLM benchmark 到 /mnt/cephszjt/user_juntianzhang/LatentDraft/data/

已支持:
- MMStar         → Lin-Chen/MMStar
- RealWorldQA    → xai-org/RealworldQA  (fallback: visheratin/realworldqa)
- BLINK          → BLINK-Benchmark/BLINK
- MUIRBENCH      → MUIRBENCH/MUIRBENCH  (fallback: MUIRBENCH/MUIRbench)
- MMBench        → lmms-lab/MMBench  (内含 en/cn 子集; fallback: opencompass/MMBench)
- HallusionBench → lmms-lab/HallusionBench  (fallback: PahaII/HallusionBench)
- SimpleVQA      → lmms-lab/SimpleVQA  (fallback: opencompass/SimpleVQA)

用法：
    python3 scripts/download_benchmarks.py
    python3 scripts/download_benchmarks.py --only MMBench HallusionBench SimpleVQA
    python3 scripts/download_benchmarks.py --endpoint https://hf-mirror.com

每个数据集落到 data/<NAME>/ 下；断点续传开启；失败重试 3 次；所有日志写到 stdout。
"""
import os
import sys
import time
import argparse
import traceback
from pathlib import Path

# ---- 关键：在 import huggingface_hub 之前强制重定向 HF cache ----
# 默认 ~/.cache/huggingface 可能被 root 占用无写权限，必须重定向到用户可写目录。
# 即使下载时用 local_dir，hub/xet 后端仍会尝试写 cache 元数据/日志。
_HF_USER_CACHE = "/mnt/cephszjt/user_juntianzhang/LatentDraft/outputs/.hf_cache"
os.environ.setdefault("HF_HOME", _HF_USER_CACHE)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_USER_CACHE, "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_HF_USER_CACHE, "datasets"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(_HF_USER_CACHE, "xet"))
os.environ.setdefault("XET_LOG_DIR", os.path.join(_HF_USER_CACHE, "xet_logs"))
# 创建所有可能用到的子目录，避免底层代码找不到目录
for _d in [_HF_USER_CACHE,
           os.environ["HF_HUB_CACHE"],
           os.environ["HF_DATASETS_CACHE"],
           os.environ["HF_XET_CACHE"],
           os.environ["XET_LOG_DIR"]]:
    os.makedirs(_d, exist_ok=True)

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

DATA_ROOT = "/mnt/cephszjt/user_juntianzhang/LatentDraft/data"

# (local_dir_name, [candidate_repo_ids])
BENCHMARKS = [
    ("MMStar",         ["Lin-Chen/MMStar"]),
    ("RealWorldQA",    ["xai-org/RealworldQA", "visheratin/realworldqa"]),
    ("BLINK",          ["BLINK-Benchmark/BLINK"]),
    ("MUIRBENCH",      ["MUIRBENCH/MUIRBENCH", "MUIRBENCH/MUIRbench"]),
    # —— 新增 3 个 ——
    # MMBench: 官方推荐 lmms-lab 整合包, 内含 en/cn 多个 split (dev/test); 备选 opencompass 整理版
    ("MMBench",        ["lmms-lab/MMBench", "opencompass/MMBench"]),
    # HallusionBench: lmms-lab 整理版 (含 image / question / gt_answer / category)
    ("HallusionBench", ["lmms-lab/HallusionBench", "PahaII/HallusionBench"]),
    # SimpleVQA: opencompass / lmms-lab 整理版 (开放回答, 主要是单词/短语)
    ("SimpleVQA",      ["lmms-lab/SimpleVQA", "opencompass/SimpleVQA"]),
]

def download_one(name: str, candidates: list, root: str, retries: int = 3) -> dict:
    """尝试从候选仓库列表下载一个数据集到 root/name/。"""
    target = os.path.join(root, name)
    os.makedirs(target, exist_ok=True)
    info = {
        "name": name,
        "target_dir": target,
        "status": "pending",
        "repo_id_used": None,
        "error": None,
        "elapsed_sec": 0.0,
    }
    t_start = time.time()

    last_err = None
    for repo_id in candidates:
        print(f"\n[{name}] 尝试仓库: {repo_id}")
        print(f"[{name}]   → 本地目录: {target}")
        for attempt in range(1, retries + 1):
            try:
                # huggingface_hub >= 1.x: 已移除 local_dir_use_symlinks / resume_download 参数
                # local_dir 本身即为实体文件存放；断点续传由底层自动处理
                path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=target,
                    max_workers=4,
                    tqdm_class=None,                # 关闭 tqdm（后台 nohup 日志更干净）
                )
                info["status"] = "ok"
                info["repo_id_used"] = repo_id
                info["elapsed_sec"] = time.time() - t_start
                print(f"[{name}] ✅ 下载完成 (repo={repo_id}, attempt={attempt}, "
                      f"elapsed={info['elapsed_sec']:.1f}s)")
                return info
            except (RepositoryNotFoundError,) as e:
                # 仓库不存在，立刻切下一个 candidate，不重试
                last_err = f"RepoNotFound: {repo_id} ({e})"
                print(f"[{name}] ❌ 仓库不存在: {repo_id}，尝试下一个候选")
                break
            except (HfHubHTTPError, Exception) as e:
                last_err = f"{type(e).__name__}: {e}"
                print(f"[{name}] ⚠️  attempt {attempt}/{retries} 失败: {last_err}")
                if attempt < retries:
                    sleep_s = 5 * attempt
                    print(f"[{name}]   {sleep_s}s 后重试...")
                    time.sleep(sleep_s)
        # 当前 candidate 重试用尽 → 切下一个
        print(f"[{name}] 当前候选 {repo_id} 已用尽重试次数，切换下一个（如有）")

    info["status"] = "failed"
    info["error"] = last_err
    info["elapsed_sec"] = time.time() - t_start
    print(f"[{name}] ❌ 全部候选仓库都失败: {last_err}")
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None,
                        help="只下载指定的数据集（例如：--only MMStar BLINK）")
    parser.add_argument("--endpoint", default=None,
                        help="自定义 HF endpoint，例如 https://hf-mirror.com")
    parser.add_argument("--data_root", default=DATA_ROOT)
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
        print(f"使用 HF_ENDPOINT = {args.endpoint}")
    else:
        print(f"使用默认 HF_ENDPOINT (huggingface.co)")

    os.makedirs(args.data_root, exist_ok=True)

    pending = BENCHMARKS
    if args.only:
        name_set = set(args.only)
        pending = [b for b in BENCHMARKS if b[0] in name_set]
        missing = name_set - {b[0] for b in BENCHMARKS}
        if missing:
            print(f"⚠️  忽略未知数据集: {missing}")

    print("=" * 70)
    print("  批量下载 VLM Benchmarks")
    print("=" * 70)
    print(f"  data_root: {args.data_root}")
    print(f"  待下载数据集: {[b[0] for b in pending]}")
    print(f"  Python: {sys.version.split()[0]}")
    import huggingface_hub
    print(f"  huggingface_hub: {huggingface_hub.__version__}")
    print(f"  HF_HOME:         {os.environ.get('HF_HOME')}")
    print(f"  HF_HUB_CACHE:    {os.environ.get('HF_HUB_CACHE')}")
    print(f"  XET_LOG_DIR:     {os.environ.get('XET_LOG_DIR')}")
    print(f"  HF_ENDPOINT:     {os.environ.get('HF_ENDPOINT', '<default>')}")
    print("=" * 70)

    summary = []
    for name, candidates in pending:
        print(f"\n{'#' * 70}")
        print(f"# 开始下载: {name}")
        print(f"{'#' * 70}")
        try:
            info = download_one(name, candidates, args.data_root)
        except Exception as e:
            info = {
                "name": name, "target_dir": os.path.join(args.data_root, name),
                "status": "crash", "repo_id_used": None,
                "error": f"unhandled: {e}", "elapsed_sec": 0.0,
            }
            traceback.print_exc()
        summary.append(info)

    print("\n" + "=" * 70)
    print("  📋 下载汇总")
    print("=" * 70)
    ok_cnt = 0
    for s in summary:
        mark = "✅" if s["status"] == "ok" else "❌"
        print(f"  {mark} {s['name']:<15} | status={s['status']:<8} | "
              f"repo={s['repo_id_used']} | elapsed={s['elapsed_sec']:.1f}s")
        if s["status"] != "ok":
            print(f"       error: {s['error']}")
        else:
            ok_cnt += 1
    print(f"\n  成功 {ok_cnt}/{len(summary)}")

    # 磁盘占用
    print("\n  💾 各数据集磁盘占用：")
    for s in summary:
        d = s["target_dir"]
        if os.path.isdir(d):
            total = 0
            for root, _, files in os.walk(d):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            print(f"    {s['name']:<15}  {total / (1024**2):8.1f} MB  ({d})")

    sys.exit(0 if ok_cnt == len(summary) else 1)

if __name__ == "__main__":
    main()
