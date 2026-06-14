#!/bin/bash
# ============================================================
# Latent 触发位置与 Token 熵关系分析 (单卡版)
# 分析 200 条样本，with-latent 和 no-latent 两次独立 forward
# 用法: bash run_entropy_analysis.sh
# ============================================================

set -e

# ======================== 默认参数 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="<PATH_TO_QWEN3_VL_8B_INSTRUCT>"
CHECKPOINT="${SCRIPT_DIR}/outputs/rld_stage2_erqa_latent_cot_v2_ckpt49_balanced_v2/checkpoint-26"
DATA_FILE="${SCRIPT_DIR}/data/erqa/erqa_test.jsonl"
DATA_ROOT="${SCRIPT_DIR}/data/erqa"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/entropy_trigger_analysis"

MAX_NEW_TOKENS=2048
MAX_SAMPLES=200
DTYPE="bfloat16"
DEVICE="cuda"

# ======================== 解析参数 ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)      CHECKPOINT="$2";       shift 2 ;;
        --model_path)      MODEL_PATH="$2";       shift 2 ;;
        --data_file)       DATA_FILE="$2";        shift 2 ;;
        --data_root)       DATA_ROOT="$2";        shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2";       shift 2 ;;
        --max_new_tokens)  MAX_NEW_TOKENS="$2";   shift 2 ;;
        --max_samples)     MAX_SAMPLES="$2";      shift 2 ;;
        --dtype)           DTYPE="$2";            shift 2 ;;
        --device)          DEVICE="$2";           shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

cd "${SCRIPT_DIR}"

# ======================== 数据 / ckpt 校验 ========================
if [ ! -f "${DATA_FILE}" ]; then
    echo "❌ data_file 不存在: ${DATA_FILE}"
    exit 1
fi

if [ -n "${CHECKPOINT}" ] && [ ! -d "${CHECKPOINT}" ]; then
    echo "⚠️  checkpoint 不存在: ${CHECKPOINT}"
    CHECKPOINT=""
fi

mkdir -p "${OUTPUT_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/entropy_analysis_${TS}.log"

# ======================== 环境变量 ========================
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "  Latent 触发位置与 Token 熵 关系分析"
echo "============================================================"
echo "  data_file:    ${DATA_FILE}"
echo "  data_root:    ${DATA_ROOT}"
echo "  model_path:   ${MODEL_PATH}"
echo "  checkpoint:   ${CHECKPOINT:-(none, 纯 base model)}"
echo "  output_dir:   ${OUTPUT_DIR}"
echo "  max_samples:  ${MAX_SAMPLES}"
echo "  max_new:      ${MAX_NEW_TOKENS}"
echo "  dtype:        ${DTYPE}"
echo "  device:       ${DEVICE}"
echo "  log:          ${LOG_FILE}"
echo "============================================================"

# ======================== 选 python ========================
if command -v python3 &> /dev/null; then PYTHON=python3; else PYTHON=python; fi

CKPT_ARG=""
if [ -n "${CHECKPOINT}" ]; then CKPT_ARG="--checkpoint ${CHECKPOINT}"; fi

${PYTHON} -u "${SCRIPT_DIR}/analyze_entropy_trigger.py" \
  --model_path      "${MODEL_PATH}" \
  --data_file       "${DATA_FILE}" \
  --data_root       "${DATA_ROOT}" \
  --output_dir      "${OUTPUT_DIR}" \
  --max_new_tokens  ${MAX_NEW_TOKENS} \
  --max_samples     ${MAX_SAMPLES} \
  --dtype           "${DTYPE}" \
  --device          "${DEVICE}" \
  ${CKPT_ARG} \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================"
echo "  ✅ 熵触发位置分析完成"
echo "============================================================"
echo "  📄 结果目录: ${OUTPUT_DIR}"
echo "  📄 日志:      ${LOG_FILE}"
