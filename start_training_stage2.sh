#!/bin/bash
# RLD Stage 2 训练启动脚本
# 方案 B: 全模块 LoRA (7 modules) + 扩展到 L9~L35 (27层)
# 数据: Stage 1 50K + Stage 2 checked 59K 合并去重 ≈ 80K
set -e

echo "🚀 RLD Stage 2 训练启动 (方案 B: 全模块 LoRA + 扩展低层)"
echo ""

# 初始化 conda
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# 激活环境
if command -v conda &> /dev/null; then
    if conda env list | grep -q "^colamem "; then
        conda activate colamem
    fi
fi

# 检测 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "8")
else
    GPU_COUNT=8
fi

echo "GPU 数量: $GPU_COUNT"
echo ""

# 检查 Stage 1 checkpoint 是否存在
STAGE1_CKPT="./outputs/rld_kv_prefix_768_bidirection_stage1/model"
if [ ! -d "$STAGE1_CKPT" ]; then
    echo "❌ Stage 1 checkpoint 不存在: $STAGE1_CKPT"
    echo "   请先完成 Stage 1 训练"
    exit 1
fi
echo "✅ Stage 1 checkpoint: $STAGE1_CKPT"

# 检查合并数据是否存在
MERGED_DATA="./data/rld_stage2_merged.json"
if [ ! -f "$MERGED_DATA" ]; then
    echo "⚠️ 合并数据不存在，正在生成..."
    python3 scripts/merge_stage2_data.py \
        --stage1_data data/rld_50k_newformat.json \
        --stage2_data rld_training_stage2_checked.json \
        --output "$MERGED_DATA"
    echo ""
fi
echo "✅ 训练数据: $MERGED_DATA"
echo ""

# NCCL 配置
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# PyTorch 显存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RLD 调试日志
export RLD_DEBUG=0

# 创建输出目录
mkdir -p outputs/rld_stage2_full_lora

echo "📋 Stage 2 配置:"
echo "  LoRA: r=32, alpha=64, 7 modules (q/k/v/o_proj + gate/up/down_proj)"
echo "  LoRA 层范围: L9~L35 (27层)"
echo "  数据: ~80K 样本, 2 epochs"
echo "  学习率: Controller=2e-5, LoRA=5e-5"
echo ""

# 启动训练
echo "📦 使用 PyTorch DDP (torchrun) 启动 Stage 2 训练..."
torchrun --nproc_per_node=$GPU_COUNT \
    scripts/train.py \
    --config configs/rld_train_stage2.yaml

echo ""
echo "✅ Stage 2 训练完成！"
