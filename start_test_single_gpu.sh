#!/bin/bash
# 单卡快速验证脚本
# 无需 DeepSpeed 分布式，2-3 分钟内发现 CUDA 错误
# 使用方法: bash start_test_single_gpu.sh
set -e

echo "🧪 RLD 单卡快速验证 (无需 DeepSpeed)"
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
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
echo "GPU 数量: $GPU_COUNT"

# ====== 单卡测试: 只使用第 0 号 GPU ======
export CUDA_VISIBLE_DEVICES=0
export RLD_DEBUG=1                # 开启 RLD 段循环调试日志

# 可选: 开启 CUDA 同步模式 (更慢但错误定位更精确)
# export CUDA_LAUNCH_BLOCKING=1

echo "🖥️  使用 GPU: 0 (单卡模式)"
echo "📝 调试日志: 已开启"
echo ""

mkdir -p outputs/rld_train

# 启动单卡验证
python scripts/test_single_gpu.py \
    --config configs/rld_train.yaml \
    --num_samples 2 \
    --verbose \
    --check_backward

echo ""
echo "✅ 单卡验证完成！"
