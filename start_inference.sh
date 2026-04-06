#!/bin/bash
# ============================================================
# RLD 推理启动脚本 (带完整 CoT 输出)
#
# 使用方法:
#   bash start_inference.sh                    # 默认参数推理
#   bash start_inference.sh --greedy           # 确定性解码 (greedy)
#   bash start_inference.sh --gpu 2            # 指定 GPU
#   bash start_inference.sh --max_tokens 2048  # 增大生成长度
# ============================================================
set -e

echo "🔮 RLD 推理启动 (带完整 CoT)"
echo ""

# ==================== 可配置参数 ====================

# 基座模型路径 (Qwen3-VL-8B-Instruct)
MODEL_PATH="${MODEL_PATH:-/mnt/wfs/mmchongqingssdwfssz/project_luban_infra/luban_infra/model_factory/for_lucas_Qwen3-VL-8B-Instruct_export}"

# RLD 训练好的 checkpoint 路径 (最新 Stage 1: bidirection)
RLD_CHECKPOINT="${RLD_CHECKPOINT:-./outputs/rld_kv_prefix_768_bidirection_stage1/model}"

# 测试数据文件 (训练集中未出现过的 50 条样本)
TEST_FILE="${TEST_FILE:-./test_unseen_50.json}"

# 输出结果文件
OUTPUT_FILE="${OUTPUT_FILE:-./test_results_cot.json}"

# 生成参数
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.6}"

# GPU 编号 (默认使用第 0 号)
GPU_ID="${GPU_ID:-0}"

# 解码模式: sampling (默认) 或 greedy
DECODE_MODE="sampling"

# LoRA 自动检测: load_pretrained 会自动检测 checkpoint 中的 lora_adapter 目录
# 如果需要手动指定 LoRA 结构 (如 Stage 2 推理), 使用 --use_lora 参数
USE_LORA="false"

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
        --checkpoint)
            RLD_CHECKPOINT="$2"
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
        --no_lora)
            USE_LORA="false"
            shift
            ;;
        --help|-h)
            echo "用法: bash start_inference.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --greedy              确定性解码 (greedy, 不采样)"
            echo "  --gpu <id>            指定 GPU 编号 (默认: 0)"
            echo "  --max_tokens <n>      最大生成 token 数 (默认: 1024)"
            echo "  --temperature <t>     采样温度 (默认: 0.6)"
            echo "  --model <path>        基座模型路径"
            echo "  --checkpoint <path>   RLD checkpoint 路径"
            echo "  --test_file <path>    测试数据文件"
            echo "  --output <path>       输出结果文件"
            echo ""
            echo "环境变量:"
            echo "  MODEL_PATH, RLD_CHECKPOINT, TEST_FILE, OUTPUT_FILE"
            echo "  MAX_NEW_TOKENS, TEMPERATURE, GPU_ID"
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
echo "  基座模型:     $MODEL_PATH"
echo "  RLD Checkpoint: $RLD_CHECKPOINT"
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
    if [ "$GPU_ID" -ge "$GPU_COUNT" ] 2>/dev/null; then
        echo "⚠️  警告: 指定的 GPU $GPU_ID 可能不存在 (共 $GPU_COUNT 张)"
    fi
else
    echo "⚠️  未检测到 nvidia-smi, 请确认 GPU 环境"
fi

# 检查文件
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ 测试数据文件不存在: $TEST_FILE"
    echo ""
    echo "请先生成测试数据:"
    echo "  python scripts/prepare_test_data.py \\"
    echo "      --train_json rld_training_mixed.json \\"
    echo "      --source_jsonl data/MMathCoT-1M/train.jsonl \\"
    echo "      --image_root data/MMathCoT-1M \\"
    echo "      --num_samples 50 \\"
    echo "      --output $TEST_FILE"
    exit 1
fi

if [ ! -d "$RLD_CHECKPOINT" ]; then
    echo "❌ RLD checkpoint 目录不存在: $RLD_CHECKPOINT"
    exit 1
fi

if [ ! -f "$RLD_CHECKPOINT/rld_controller.pt" ]; then
    echo "❌ 未找到 controller 权重: $RLD_CHECKPOINT/rld_controller.pt"
    exit 1
fi

echo ""

# ==================== PyTorch 显存配置 ====================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ==================== 构建推理命令 ====================
INFERENCE_CMD="python scripts/inference_with_cot.py \
    --model_path $MODEL_PATH \
    --rld_checkpoint $RLD_CHECKPOINT \
    --test_file $TEST_FILE \
    --output_file $OUTPUT_FILE \
    --max_new_tokens $MAX_NEW_TOKENS"

if [ "$USE_LORA" = "true" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --use_lora"
    echo "🔧 手动启用 LoRA adapter (setup_lora + load)"
else
    # 自动检测: load_pretrained 会自动检测 lora_adapter 目录并加载
    if [ -d "$RLD_CHECKPOINT/lora_adapter" ]; then
        echo "🔧 检测到 LoRA adapter, 将自动加载"
    fi
fi

if [ "$DECODE_MODE" = "greedy" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --no_sample"
    echo "🎯 使用 Greedy 解码 (确定性输出)"
else
    INFERENCE_CMD="$INFERENCE_CMD --temperature $TEMPERATURE"
    echo "🎲 使用 Sampling 解码 (temperature=$TEMPERATURE)"
fi

echo ""
echo "🚀 启动推理..."
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
echo "✅ 推理完成！"
echo "  耗时: ${MINUTES}分${SECONDS}秒"
echo "  结果: $OUTPUT_FILE"
echo "  日志: ${OUTPUT_FILE%.json}_log.txt"
echo "============================================================"
