#!/bin/bash
# 调试模式快速启动脚本
# 超时缩短到 120 秒 (2 分钟)，快速发现 hang 问题
# 使用方法: bash start_debug.sh
set -e

echo "🔧 RLD 调试模式启动 (超时 120 秒)"
echo ""

# 初始化 conda
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
    fi
fi

# 检测 GPU
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "8")
echo "GPU 数量: $GPU_COUNT"

# ====== 调试模式: 超时缩短为 120 秒 (2 分钟) ======
export DEEPSPEED_TIMEOUT=120
export TORCH_NCCL_TIMEOUT_S=120
export NCCL_TIMEOUT=120
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_DEBUG=INFO            # 更详细的 NCCL 日志
export RLD_DEBUG=1                # 开启 RLD 段循环调试日志

# 可选: 开启 CUDA 同步模式 (更慢但错误定位更精确)
# export CUDA_LAUNCH_BLOCKING=1

echo "⏱️  NCCL 超时: 120 秒"
echo "📝 调试日志: 已开启"
echo ""

mkdir -p outputs/rld_train

# 启动训练
deepspeed --num_gpus=$GPU_COUNT \
    scripts/train.py \
    --config configs/rld_train.yaml

echo ""
echo "✅ 训练完成！"
