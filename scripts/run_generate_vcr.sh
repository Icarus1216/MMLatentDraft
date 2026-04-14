#!/bin/bash
# ============================================================
# 运行 VCR Latent CoT 数据生成脚本
# ============================================================

# 内部 API 鉴权信息
export ICHAT_APPID="apprbtwqufqr9mqgedn"
export ICHAT_APPKEY="WlokXJldsHNttISobOuRWTKAVZRCjDXJ"

# RTX 账号 (请修改为你自己的 RTX)
RTX="skyejtzhang"

# 数据路径
VCR_ROOT="data/nld_phase1/raw/vcr"
OUTPUT="data/nld_phase1/vcr_latent_cot_v2_test.json"

# 生成参数
NUM_SAMPLES=10
MODEL="gemini-3-pro-preview"
WORKERS=8
TEMPERATURE=0.7
MAX_TOKENS=8192
SAVE_INTERVAL=100
SEED=42

# ============================================================
# 运行
# ============================================================
echo "=========================================="
echo "  VCR Latent CoT 数据生成"
echo "=========================================="
echo "  RTX:         ${RTX}"
echo "  模型:        ${MODEL}"
echo "  目标样本数:  ${NUM_SAMPLES}"
echo "  并发线程:    ${WORKERS}"
echo "  输出路径:    ${OUTPUT}"
echo "=========================================="

python3 scripts/generate_vcr_latent_cot.py \
    --vcr_root "${VCR_ROOT}" \
    --output "${OUTPUT}" \
    --rtx "${RTX}" \
    --model "${MODEL}" \
    --num_samples ${NUM_SAMPLES} \
    --workers ${WORKERS} \
    --temperature ${TEMPERATURE} \
    --max_tokens ${MAX_TOKENS} \
    --save_interval ${SAVE_INTERVAL} \
    --seed ${SEED} \
    --verbose
