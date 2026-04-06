#!/bin/bash
# ============================================================
# Qwen3-VL 基座模型推理启动脚本 (无 RLD, 作为对比基线)
#
# 使用与 RLD 推理完全相同的:
#   - System Prompt
#   - 测试数据 (test_unseen_50.json)
#   - 评估逻辑
#   - 解码参数 (temperature, max_tokens)
#
# 使用方法:
#   bash start_inference_baseline.sh                    # 默认参数
#   bash start_inference_baseline.sh --greedy           # 确定性解码
#   bash start_inference_baseline.sh --gpu 2            # 指定 GPU
# ============================================================
set -e

echo "🔮 Qwen3-VL 基座推理启动 (无 RLD, 对比基线)"
echo ""

# ==================== 可配置参数 ====================

# 基座模型路径 (Qwen3-VL-8B-Instruct, 与 RLD 训练使用同一个)
MODEL_PATH="${MODEL_PATH:-/mnt/wfs/mmchongqingssdwfssz/project_luban_infra/luban_infra/model_factory/for_lucas_Qwen3-VL-8B-Instruct_export}"

# 测试数据文件 (与 RLD 推理使用完全相同的 50 条)
TEST_FILE="${TEST_FILE:-./test_unseen_50.json}"

# 输出结果文件
OUTPUT_FILE="${OUTPUT_FILE:-./test_results_baseline.json}"

# 生成参数 (与 RLD 推理保持一致)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.6}"

# GPU 编号
GPU_ID="${GPU_ID:-0}"

# 解码模式
DECODE_MODE="sampling"

# ==================== 解析命令行参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --greedy)
            DECODE_MODE="greedy"
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test_file)
            TEST_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: bash start_inference_baseline.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --greedy              确定性解码 (greedy, 不采样)"
            echo "  --gpu <id>            指定 GPU 编号 (默认: 0)"
            echo "  --max_tokens <n>      最大生成 token 数 (默认: 1024)"
            echo "  --temperature <t>     采样温度 (默认: 0.6)"
            echo "  --model <path>        基座模型路径"
            echo "  --test_file <path>    测试数据文件"
            echo "  --output <path>       输出结果文件"
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1 (使用 --help 查看帮助)"
            exit 1
            ;;
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

# ==================== 环境检查 ====================
echo ""
echo "📋 配置信息:"
echo "  模式:         Qwen3-VL 基座 (无 RLD)"
echo "  基座模型:     $MODEL_PATH"
echo "  测试数据:     $TEST_FILE"
echo "  输出文件:     $OUTPUT_FILE"
echo "  GPU:          $GPU_ID"
echo "  最大生成长度: $MAX_NEW_TOKENS tokens"
echo "  解码模式:     $DECODE_MODE"
if [ "$DECODE_MODE" = "sampling" ]; then
    echo "  采样温度:     $TEMPERATURE"
fi
echo ""

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
    echo "🖥️  检测到 $GPU_COUNT 张 GPU"
else
    echo "⚠️  未检测到 nvidia-smi, 请确认 GPU 环境"
fi

# 检查文件
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ 测试数据文件不存在: $TEST_FILE"
    exit 1
fi

echo ""

# ==================== PyTorch 显存配置 ====================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ==================== 构建推理命令 ====================
INFERENCE_CMD="python scripts/inference_baseline.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --output_file $OUTPUT_FILE \
    --max_new_tokens $MAX_NEW_TOKENS"

if [ "$DECODE_MODE" = "greedy" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --no_sample"
    echo "🎯 使用 Greedy 解码 (确定性输出)"
else
    INFERENCE_CMD="$INFERENCE_CMD --temperature $TEMPERATURE"
    echo "🎲 使用 Sampling 解码 (temperature=$TEMPERATURE)"
fi

echo ""
echo "🚀 启动基座推理..."
echo "   命令: $INFERENCE_CMD"
echo ""

# ==================== 执行推理 ====================
START_TIME=$(date +%s)

eval $INFERENCE_CMD 2>&1 | tee "${OUTPUT_FILE%.json}_log.txt"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "✅ 基座推理完成！"
echo "  耗时: ${MINUTES}分${SECONDS}秒"
echo "  结果: $OUTPUT_FILE"
echo "  日志: ${OUTPUT_FILE%.json}_log.txt"
echo ""
echo "📊 对比方法:"
echo "  RLD 结果:  test_results_cot.json"
echo "  基座结果:  $OUTPUT_FILE"
echo "============================================================"
