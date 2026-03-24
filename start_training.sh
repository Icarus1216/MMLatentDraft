#!/bin/bash
# RLD (Reflective Latent Draft) 训练启动脚本
# 默认使用 PyTorch DDP (torchrun), 如需使用 DeepSpeed 请设置: USE_DEEPSPEED=1
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
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# NVLink P2P: 已通过 max_seq_len=2048 严格截断解决 OOM/cudaErrorContained
# 如果再次出现 peer memory 越界, 取消注释下行禁用 P2P:
# export NCCL_P2P_DISABLE=1

# CUDA 同步调试: 让 CUDA 操作同步执行，精确定位异步错误
# 调试完成后注释掉此行恢复性能
# export CUDA_LAUNCH_BLOCKING=1

# PyTorch 显存管理: 启用 expandable_segments 减少碎片化
# 方案三每段 forward 中 DynamicLayer.update() 的 torch.cat 会创建大量临时 tensor,
# expandable_segments 让 allocator 以更大块分配/释放, 减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RLD 调试日志 (开启段循环详细日志 + 每段显存监控)
# ⚠️ 调试模式会导致 rank 0 有额外的 lm_head 计算 + 大量打印, 增加 rank 间延迟
# 稳定运行后建议关闭 (注释下行), 只保留 TensorBoard 指标监控
# export RLD_DEBUG=1
export RLD_DEBUG=0

# 注意力实现: 使用 flash_attention_2 (性能最优)
# 如果 flash-attn 有兼容性问题, 可切换到 sdpa:
#   export RLD_ATTN_IMPL=sdpa

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

# 分布式后端选择:
#   默认: torchrun (PyTorch DDP) — 通信简单, 只需 1 次 allreduce 同步 61M 可训练参数
#   可选: DeepSpeed ZeRO-2 — 设置 USE_DEEPSPEED=1 启用 (显存不是瓶颈时不推荐)
# export USE_DEEPSPEED=1

# 启动训练
if [ "${USE_DEEPSPEED:-0}" = "1" ]; then
    echo "📦 使用 DeepSpeed ZeRO-2 启动训练..."
    deepspeed --num_gpus=$GPU_COUNT \
        scripts/train.py \
        --config configs/rld_train.yaml \
        --use_deepspeed
else
    echo "📦 使用 PyTorch DDP (torchrun) 启动训练..."
    torchrun --nproc_per_node=$GPU_COUNT \
        scripts/train.py \
        --config configs/rld_train.yaml
fi

echo ""
echo "✅ 训练完成！"
