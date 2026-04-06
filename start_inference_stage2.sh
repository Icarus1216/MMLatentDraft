#!/bin/bash
# ============================================================
# RLD Stage 2 推理启动脚本 (带完整 CoT 输出)
#
# Stage 2 模型使用更大的 LoRA (r=32, 7 modules, L9~L35)
# load_pretrained 会自动从 lora_adapter/adapter_config.json 读取配置
#
# 使用方法:
#   bash start_inference_stage2.sh                    # 默认参数推理
#   bash start_inference_stage2.sh --greedy           # 确定性解码
# ============================================================
set -e

echo "🔮 RLD Stage 2 推理启动 (带完整 CoT)"
echo ""

# ==================== 可配置参数 ====================

MODEL_PATH="${MODEL_PATH:-/mnt/wfs/mmchongqingssdwfssz/project_luban_infra/luban_infra/model_factory/for_lucas_Qwen3-VL-8B-Instruct_export}"
RLD_CHECKPOINT="${RLD_CHECKPOINT:-./outputs/rld_stage2_full_lora/model}"
TEST_FILE="${TEST_FILE:-./test_unseen_50.json}"
OUTPUT_FILE="${OUTPUT_FILE:-./test_results_stage2_cot.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.6}"
GPU_ID="${GPU_ID:-0}"
DECODE_MODE="sampling"

# ==================== 解析命令行参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --greedy) DECODE_MODE="greedy"; shift ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --max_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --checkpoint) RLD_CHECKPOINT="$2"; shift 2 ;;
        --test_file) TEST_FILE="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

# ==================== 初始化 conda 环境 ====================
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

if command -v conda &> /dev/null; then
    if conda env list | grep -q "^colamem "; then
        conda activate colamem
        echo "✅ 已激活 conda 环境: colamem"
    fi
fi

echo ""
echo "📋 Stage 2 推理配置:"
echo "  基座模型:     $MODEL_PATH"
echo "  RLD Checkpoint: $RLD_CHECKPOINT"
echo "  测试数据:     $TEST_FILE"
echo "  输出文件:     $OUTPUT_FILE"
echo "  GPU:          $GPU_ID"
echo "  最大生成长度: $MAX_NEW_TOKENS tokens"
echo "  解码模式:     $DECODE_MODE"
echo ""

# 检查文件
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ 测试数据文件不存在: $TEST_FILE"
    exit 1
fi

if [ ! -d "$RLD_CHECKPOINT" ]; then
    echo "❌ RLD checkpoint 目录不存在: $RLD_CHECKPOINT"
    exit 1
fi

# 检查 LoRA adapter
if [ -d "$RLD_CHECKPOINT/lora_adapter" ]; then
    echo "🔧 检测到 LoRA adapter, 将自动加载 (从 adapter_config.json 读取配置)"
else
    echo "⚠️ 未检测到 LoRA adapter 目录"
fi

# ==================== PyTorch 显存配置 ====================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ==================== 构建推理命令 ====================
# Stage 2 的 LoRA 由 load_pretrained 自动从 adapter_config.json 加载
# 不需要手动指定 --use_lora 和 LoRA 参数
INFERENCE_CMD="python3 scripts/inference_with_cot.py \
    --model_path $MODEL_PATH \
    --rld_checkpoint $RLD_CHECKPOINT \
    --test_file $TEST_FILE \
    --output_file $OUTPUT_FILE \
    --max_new_tokens $MAX_NEW_TOKENS"

if [ "$DECODE_MODE" = "greedy" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --no_sample"
    echo "🎯 使用 Greedy 解码"
else
    INFERENCE_CMD="$INFERENCE_CMD --temperature $TEMPERATURE"
    echo "🎲 使用 Sampling 解码 (temperature=$TEMPERATURE)"
fi

echo ""
echo "🚀 启动 Stage 2 推理..."
echo "   命令: $INFERENCE_CMD"
echo ""

START_TIME=$(date +%s)
eval $INFERENCE_CMD 2>&1 | tee "${OUTPUT_FILE%.json}_log.txt"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "✅ Stage 2 推理完成！"
echo "  耗时: $((ELAPSED / 60))分$((ELAPSED % 60))秒"
echo "  结果: $OUTPUT_FILE"
echo "============================================================"
