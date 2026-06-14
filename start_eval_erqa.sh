#!/bin/bash
# ============================================================
# ERQA 全数据集纯推理测试启动脚本（无可视化）
#
# 用法:
#   # 默认: 带 latent 推理，全量测试
#   bash start_eval_erqa.sh
#
#   # 无 latent 基线对比
#   bash start_eval_erqa.sh --skip_latent
#
#   # 无 system prompt 基线对比
#   bash start_eval_erqa.sh --no_system_prompt
#
#   # 只测试前 50 条
#   bash start_eval_erqa.sh --max_samples 50
#
#   # 指定 checkpoint
#   bash start_eval_erqa.sh --checkpoint ./outputs/nld_train_phase2/checkpoint-2000
# ============================================================

set -e

echo "📊 ERQA 全数据集纯推理测试"
echo ""

# ======================== 配置 ========================
MODEL_PATH="<PATH_TO_QWEN3_VL_8B_INSTRUCT>"
NLD_CHECKPOINT="./outputs/rld_stage2/checkpoint-400"
DATA_FILE="./data/erqa/erqa_test.jsonl"
DATA_ROOT="./data/erqa"
OUTPUT_DIR="./outputs/erqa_eval_ckpt400"
MAX_NEW_TOKENS=2048
MAX_SAMPLES=0
DEVICE="cuda"
DTYPE="bfloat16"
SKIP_LATENT=""
NO_SYSTEM_PROMPT=""

# ======================== 解析命令行参数 ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            NLD_CHECKPOINT="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --skip_latent)
            SKIP_LATENT="--skip_latent"
            shift
            ;;
        --no_system_prompt)
            NO_SYSTEM_PROMPT="--no_system_prompt"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

# ======================== 检测 Python ========================
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

# ======================== 运行前检查 ========================
echo ""
echo "============================================================"
echo "  ERQA 评估配置"
echo "============================================================"
echo "  基座模型:       ${MODEL_PATH}"
echo "  NLD 权重:       ${NLD_CHECKPOINT}"
echo "  数据文件:       ${DATA_FILE}"
echo "  数据根目录:     ${DATA_ROOT}"
echo "  输出目录:       ${OUTPUT_DIR}"
echo "  最大生成 tokens: ${MAX_NEW_TOKENS}"
echo "  最大样本数:     ${MAX_SAMPLES} (0=全部)"
echo "  跳过 Latent:    ${SKIP_LATENT:-否}"
echo "  无 System Prompt: ${NO_SYSTEM_PROMPT:-否}"
echo "  设备:           ${DEVICE}"
echo "  精度:           ${DTYPE}"
echo "============================================================"
echo ""

# 检查模型路径
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ 基座模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

# 检查数据文件
if [ ! -f "${DATA_FILE}" ]; then
    echo "❌ 数据文件不存在: ${DATA_FILE}"
    exit 1
fi

# 检查 NLD 权重
if [ -n "${NLD_CHECKPOINT}" ] && [ ! -d "${NLD_CHECKPOINT}" ]; then
    echo "⚠️  NLD 权重路径不存在: ${NLD_CHECKPOINT}"
    echo "  将使用基座模型进行推理 (无 NLD 权重)"
    NLD_CHECKPOINT=""
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "🚀 开始 ERQA 评估..."
echo ""

# ======================== 运行评估 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES=0 $PYTHON "${SCRIPT_DIR}/eval_erqa.py" \
    --model_path "${MODEL_PATH}" \
    --data_file "${DATA_FILE}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --max_samples ${MAX_SAMPLES} \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    ${NLD_CHECKPOINT:+--checkpoint "${NLD_CHECKPOINT}"} \
    ${SKIP_LATENT} \
    ${NO_SYSTEM_PROMPT}

echo ""
echo "✅ ERQA 评估完成！结果保存在: ${OUTPUT_DIR}"
