#!/bin/bash
# RLD (Reflective Latent Draft) 训练启动脚本
set -e

echo "🚀 RLD 训练启动"
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

# NCCL 配置
export NCCL_DEBUG=WARN             # NCCL 日志级别
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000  # 启用 FlightRecorder 便于调试

# 注意力实现切换 (flash_attention_2 / sdpa / eager)
# 默认使用 flash_attention_2 (最快)，如需切换可设置环境变量:
#   export RLD_ATTN_IMPL=sdpa
#   export RLD_ATTN_IMPL=eager

# 创建输出目录
mkdir -p outputs/rld_train

# ===== 数据预过滤: 检查图片存在性 =====
# 仅在过滤后文件不存在时执行，避免重复检查
DATA_INPUT="data/rld_mmathcot_filtered.json"
DATA_CHECKED="data/rld_mmathcot_filtered_checked.json"

if [ ! -f "$DATA_CHECKED" ]; then
    echo "🔍 首次运行: 执行数据预过滤 (检查图片存在性)..."
    python scripts/prefilter_data.py \
        --input "$DATA_INPUT" \
        --output "$DATA_CHECKED" \
        --num_workers 32
    echo ""
else
    echo "✅ 预过滤数据已存在: $DATA_CHECKED (跳过)"
    echo ""
fi

# 启动训练
deepspeed --num_gpus=$GPU_COUNT \
    scripts/train.py \
    --config configs/rld_train.yaml

echo ""
echo "✅ 训练完成！"
