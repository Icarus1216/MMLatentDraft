#!/bin/bash
# ============================================================
# prepare_vsibench.sh - VSI-Bench 数据预处理启动脚本
#
# 功能:
#   1. 解压 data/VSI-Bench/{arkitscenes,scannet,scannetpp}.zip
#   2. 对每个场景视频均匀抽取 N 帧 (默认 16), 保存为 JPEG
#   3. 生成 data/VSI-Bench_eval/vsibench_test.jsonl
#
# 用法:
#   bash prepare_vsibench.sh                    # 默认 full 集, 16 帧
#   bash prepare_vsibench.sh --num_frames 8     # 每场景 8 帧
#   bash prepare_vsibench.sh --config debiased  # 只用去偏子集 (~2363 条)
#   bash prepare_vsibench.sh --num_frames 16 --config full
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======================== 解析参数 ========================
NUM_FRAMES=16
CONFIG="full"

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_frames)      NUM_FRAMES="$2";      shift 2 ;;
        --config)          CONFIG="$2";          shift 2 ;;
        "") shift ;;  # 跳过空字符串参数（多行命令 \ 换行可能产生）
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

cd "${SCRIPT_DIR}"

echo ""
echo "============================================================"
echo "  VSI-Bench 数据预处理"
echo "============================================================"
echo "  num_frames : ${NUM_FRAMES}"
echo "  config     : ${CONFIG}  (full=5130条 / debiased=~2363条)"
echo "  输出目录   : ./data/VSI-Bench_eval/"
echo "============================================================"
echo ""

if command -v python3 &>/dev/null; then PYTHON=python3; else PYTHON=python; fi

${PYTHON} -u scripts/prepare_vsibench.py \
    --num_frames "${NUM_FRAMES}" \
    --config     "${CONFIG}"

echo ""
echo "============================================================"
echo "  ✅ VSI-Bench 预处理完成"
echo "  📄 ./data/VSI-Bench_eval/vsibench_test.jsonl"
echo "  现在可以启动评测:"
echo "    bash start_dp_eval_one_bench.sh --benchmark vsibench --eval_mode a_only"
echo "============================================================"
