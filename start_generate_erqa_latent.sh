#!/bin/bash

# ERQA Latent 推理链合成启动脚本
# 使用 OpenAI 兼容接口为 ERQA 数据生成带 latent 标记的推理链

set -e

echo "🧠 ERQA Latent 推理链合成"
echo ""

# ======================== 配置 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ERQA 数据路径
ERQA_ROOT="${SCRIPT_DIR}/data/erqa"
ERQA_FILE="${SCRIPT_DIR}/data/erqa/erqa_test.jsonl"

# 输出路径 (v2, 不覆盖旧数据)
OUTPUT="${SCRIPT_DIR}/data/erqa/erqa_latent_cot_v2.json"

# 模型配置
MODEL="${MODEL:-claude-opus-4-7}"

# 生成参数
WORKERS="${WORKERS:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
NUM_RETRIES="${NUM_RETRIES:-3}"
SEED="${SEED:-42}"

# 数据过滤选项 (取消注释以启用)
# SINGLE_IMAGE_ONLY="--single_image_only"
# MULTI_IMAGE_ONLY="--multi_image_only"
# MAX_SAMPLES="--max_samples 50"
# FORCE_DATA_TYPE="--force_data_type single_latent"
# QUESTION_TYPES='--question_types "Trajectory Reasoning" "Action Reasoning"'

echo "============================================================"
echo "  ERQA Latent CoT 合成配置"
echo "============================================================"
echo "  ERQA 数据:     ${ERQA_FILE}"
echo "  数据根目录:    ${ERQA_ROOT}"
echo "  输出文件:      ${OUTPUT}"
echo "  模型:          ${MODEL}"
echo "  并发数:        ${WORKERS}"
echo "  温度:          ${TEMPERATURE}"
echo "  每样本重试:    ${NUM_RETRIES}"
echo "============================================================"
echo ""

# ======================== 检查环境 ========================
if [ ! -f "${ERQA_FILE}" ]; then
    echo "❌ ERQA 数据文件不存在: ${ERQA_FILE}"
    exit 1
fi

if [ ! -d "${ERQA_ROOT}/images" ]; then
    echo "❌ ERQA 图片目录不存在: ${ERQA_ROOT}/images"
    exit 1
fi

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "❌ 请设置环境变量 OPENAI_API_KEY"
    echo "   export OPENAI_API_KEY='your_api_key'"
    echo "   (可选) export OPENAI_BASE_URL='https://api.openai.com/v1'"
    exit 1
fi

# ======================== 运行 ========================
echo "🚀 开始合成..."
echo ""

python3 "${SCRIPT_DIR}/scripts/generate_erqa_latent_cot.py" \
    --erqa_root "${ERQA_ROOT}" \
    --erqa_file "${ERQA_FILE}" \
    --output "${OUTPUT}" \
    --model "${MODEL}" \
    --workers ${WORKERS} \
    --temperature ${TEMPERATURE} \
    --max_tokens ${MAX_TOKENS} \
    --num_retries_per_sample ${NUM_RETRIES} \
    --seed ${SEED} \
    --resume \
    ${SINGLE_IMAGE_ONLY:-} \
    ${MULTI_IMAGE_ONLY:-} \
    ${MAX_SAMPLES:-} \
    ${FORCE_DATA_TYPE:-} \
    ${QUESTION_TYPES:-} \
    --verbose

echo ""
echo "✅ 合成完成!"
echo "   结果: ${OUTPUT}"
