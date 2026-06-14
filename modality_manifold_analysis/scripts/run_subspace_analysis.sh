#!/bin/bash
# ============================================================
# run_subspace_analysis.sh
# 运行正交分解分析 (实验A) 和子空间主角分析 (实验B)
#
# 该脚本直接基于已有的 embeddings 数据进行纯线性代数运算，
# 无需 GPU，无需重新跑模型推理。
#
# 用法:
#   bash run_subspace_analysis.sh [--results_dir DIR] [--output_dir DIR]
# ============================================================

set -e

PROJ_ROOT=""
ANALYSIS_DIR="${PROJ_ROOT}/modality_manifold_analysis"
SCRIPT="${ANALYSIS_DIR}/scripts/orthogonal_decomposition_analysis.py"

# 默认参数
RESULTS_DIR="${ANALYSIS_DIR}/results_v3"
OUTPUT_DIR=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 如果未指定 output_dir，使用默认值
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${RESULTS_DIR}/subspace_analysis"
fi

echo "============================================================"
echo "  正交分解 & 子空间主角分析"
echo "============================================================"
echo ""
echo "  Results dir: ${RESULTS_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo ""

# 检查 embeddings 文件是否存在
if [ ! -f "${RESULTS_DIR}/embeddings_baseline.pt" ]; then
    echo "❌ 错误: 未找到 ${RESULTS_DIR}/embeddings_baseline.pt"
    echo "   请先运行模态流形分析脚本提取 embeddings"
    exit 1
fi

if [ ! -f "${RESULTS_DIR}/embeddings_ckpt.pt" ]; then
    echo "❌ 错误: 未找到 ${RESULTS_DIR}/embeddings_ckpt.pt"
    echo "   请先运行模态流形分析脚本提取 embeddings"
    exit 1
fi

echo "  ✅ Embeddings 文件存在"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/figures"

# 运行分析
echo "🔬 开始分析..."
echo ""

python3 -u "${SCRIPT}" \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "✅ 分析完成!"
echo "   结果: ${OUTPUT_DIR}/"
echo "   图表: ${OUTPUT_DIR}/figures/"
echo "============================================================"
