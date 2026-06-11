#!/bin/bash
# ============================================================
# Latent 触发位置与 Token 熵的关系分析 (8 卡数据并行)
#
# 用法:
#   bash start_analyze_entropy_trigger.sh
#   bash start_analyze_entropy_trigger.sh --max_samples 50   # 调试
#   bash start_analyze_entropy_trigger.sh --num_gpus 4
# ============================================================

set -e

# ======================== 默认参数 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="/mnt/wfs/mmchongqingssdwfssz/project_luban_infra/luban_infra/model_factory/for_lucas_Qwen3-VL-8B-Instruct_export"
CHECKPOINT="${SCRIPT_DIR}/outputs/rld_stage2_erqa_latent_cot_v2_ckpt49_balanced_v2/checkpoint-26"
DATA_FILE="${SCRIPT_DIR}/data/erqa/erqa_test.jsonl"
DATA_ROOT="${SCRIPT_DIR}/data/erqa"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/entropy_trigger_analysis"

MAX_NEW_TOKENS=2048
MAX_SAMPLES=0
DTYPE="bfloat16"
NUM_GPUS=8
GPU_IDS=""
MASTER_PORT="${MASTER_PORT:-29500}"

# ======================== 解析参数 ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)        CHECKPOINT="$2";       shift 2 ;;
        --model_path)        MODEL_PATH="$2";       shift 2 ;;
        --data_file)         DATA_FILE="$2";        shift 2 ;;
        --data_root)         DATA_ROOT="$2";        shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";       shift 2 ;;
        --max_new_tokens)    MAX_NEW_TOKENS="$2";   shift 2 ;;
        --max_samples)       MAX_SAMPLES="$2";      shift 2 ;;
        --dtype)             DTYPE="$2";            shift 2 ;;
        --num_gpus)          NUM_GPUS="$2";         shift 2 ;;
        --gpu_ids)           GPU_IDS="$2";          shift 2 ;;
        --master_port)       MASTER_PORT="$2";      shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

cd "${SCRIPT_DIR}"

# ======================== GPU 校验 ========================
if [ -n "${GPU_IDS}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
fi

# ======================== 数据 / ckpt 校验 ========================
if [ ! -f "${DATA_FILE}" ]; then
    echo "❌ ERQA jsonl 不存在: ${DATA_FILE}"
    if [ -f "${SCRIPT_DIR}/prepare_erqa.py" ]; then
        python3 -u "${SCRIPT_DIR}/prepare_erqa.py" || { echo "❌ prepare_erqa.py 失败"; exit 1; }
    fi
fi
if [ ! -f "${DATA_FILE}" ]; then
    echo "❌ 即便 prepare 后仍找不到: ${DATA_FILE}"
    exit 1
fi

if [ -n "${CHECKPOINT}" ] && [ ! -d "${CHECKPOINT}" ]; then
    echo "⚠️  checkpoint 不存在: ${CHECKPOINT}, 将走纯 base model 推理 (无 NLD 权重)"
    CHECKPOINT=""
fi
if [ -n "${CHECKPOINT}" ]; then
    if [ ! -d "${CHECKPOINT}/vlm_full" ] \
       && [ ! -f "${CHECKPOINT}/model.safetensors" ] \
       && [ ! -f "${CHECKPOINT}/model.safetensors.index.json" ]; then
        echo "⚠️  checkpoint 既无 vlm_full/ 也无 model.safetensors: ${CHECKPOINT}"
        echo "    将走纯 base model 推理"
        CHECKPOINT=""
    fi
fi

mkdir -p "${OUTPUT_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/entropy_trigger_${TS}.log"

# ======================== 环境变量 ========================
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "============================================================"
echo "  Latent 触发位置 vs Token 熵 分析"
echo "  ${NUM_GPUS} 卡 DP (torchrun + NCCL)"
echo "============================================================"
echo "  data_file:    ${DATA_FILE}  ($(wc -l <"${DATA_FILE}" 2>/dev/null || echo '?') 条)"
echo "  data_root:    ${DATA_ROOT}"
echo "  model_path:   ${MODEL_PATH}"
echo "  checkpoint:   ${CHECKPOINT:-(none, base model only)}"
echo "  output_dir:   ${OUTPUT_DIR}"
echo "  max_samples:  ${MAX_SAMPLES} (0=全量)"
echo "  max_new:      ${MAX_NEW_TOKENS}"
echo "  num_gpus:     ${NUM_GPUS}    gpu_ids: ${GPU_IDS:-(0..$((NUM_GPUS-1)))}"
echo "  master_port:  ${MASTER_PORT}"
echo "  log:          ${LOG_FILE}"
echo "============================================================"

# ======================== 端口冲突预检 ========================
if command -v ss &> /dev/null && ss -tln 2>/dev/null | grep -q ":${MASTER_PORT}\b"; then
    echo "⚠️  端口 ${MASTER_PORT} 被占用, 自动换一个空闲端口"
    for try_port in 29501 29502 29503 29510 29520 29530 29550; do
        if ! ss -tln 2>/dev/null | grep -q ":${try_port}\b"; then
            MASTER_PORT=${try_port}
            echo "   → 使用端口 ${MASTER_PORT}"
            break
        fi
    done
fi

# ======================== 选 python ========================
if command -v python3 &> /dev/null; then PYTHON=python3; else PYTHON=python; fi

CKPT_ARG=""
if [ -n "${CHECKPOINT}" ]; then CKPT_ARG="--checkpoint ${CHECKPOINT}"; fi

echo ""
echo "🚀 torchrun 启动 ${NUM_GPUS} 个 process (NCCL DDP)..."
echo ""

torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    "${SCRIPT_DIR}/analyze_entropy_trigger.py" \
    --model_path     "${MODEL_PATH}" \
    --data_file      "${DATA_FILE}" \
    --data_root      "${DATA_ROOT}" \
    --output_dir     "${OUTPUT_DIR}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --max_samples    ${MAX_SAMPLES} \
    --dtype          "${DTYPE}" \
    ${CKPT_ARG} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================"
echo "  ✅ Latent 触发位置 vs Token 熵 分析完成"
echo "============================================================"
echo "  📄 汇总统计: ${OUTPUT_DIR}/summary_stats.json"
echo "  📄 逐样本:   ${OUTPUT_DIR}/per_sample_entropy.json"
echo "  📊 图表:"
echo "     ${OUTPUT_DIR}/fig_entropy_distribution.png"
echo "     ${OUTPUT_DIR}/fig_entropy_trajectory.png"
echo "     ${OUTPUT_DIR}/fig_entropy_percentile.png"
echo "     ${OUTPUT_DIR}/fig_local_peak_alignment.png"
echo "  📄 日志:     ${LOG_FILE}"
