#!/bin/bash
# ============================================================
# RLD Stage 2 训练启动脚本
# 
# 在 Stage 1 基础上:
#   - 解冻最后 8 层 LoRA (L28~L35)
#   - 100K correct CoT 全监督
#   - 多学习率分组: Controller/Adapter=3e-5, LoRA=1e-5
#   - 放松约束: lambda_kl=0.03, max_scale=0.5
#
# 用法:
#   bash scripts/run_stage2.sh
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TOKENIZERS_PARALLELISM=false

# 检查并安装 peft (Stage 2 LoRA 依赖)
python3 -c "import peft" 2>/dev/null || {
    echo "📦 安装 peft (LoRA 依赖)..."
    pip install peft -q
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${PROJECT_DIR}/configs/rld_train_stage2.yaml"
OUTPUT_DIR="${PROJECT_DIR}/outputs/rld_stage2"

NUM_GPUS=8

echo "============================================================"
echo "🚀 RLD Stage 2 训练"
echo "============================================================"
echo "  配置文件: ${CONFIG}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  GPU 数量: ${NUM_GPUS}"
echo "  分布式后端: PyTorch DDP (torchrun)"
echo "============================================================"

echo ""
echo "[Step 1/2] 检查 Stage 1 checkpoint..."
STAGE1_CKPT="${PROJECT_DIR}/outputs/rld_top8_rerun_768_stage1/model"
if [ -f "${STAGE1_CKPT}/rld_controller.pt" ] && [ -f "${STAGE1_CKPT}/rld_readout_adapter.pt" ]; then
    echo "  ✅ Stage 1 checkpoint 存在: ${STAGE1_CKPT}"
    echo "  Controller: $(du -h ${STAGE1_CKPT}/rld_controller.pt | cut -f1)"
    echo "  Adapter:    $(du -h ${STAGE1_CKPT}/rld_readout_adapter.pt | cut -f1)"
else
    echo "  ❌ Stage 1 checkpoint 不存在: ${STAGE1_CKPT}"
    echo "  请先完成 Stage 1 训练！"
    exit 1
fi

echo ""
echo "[Step 2/2] 检查训练数据..."
TRAIN_DATA="${PROJECT_DIR}/rld_training_stage2_checked.json"
if [ -f "${TRAIN_DATA}" ]; then
    NUM_SAMPLES=$(python3 -c "import json; print(len(json.load(open('${TRAIN_DATA}'))))")
    echo "  ✅ 训练数据存在: ${TRAIN_DATA}"
    echo "  样本数: ${NUM_SAMPLES}"
else
    echo "  ❌ 训练数据不存在: ${TRAIN_DATA}"
    echo "  请先运行: python scripts/prepare_stage2_data.py"
    exit 1
fi

echo ""
echo "============================================================"
echo "🚀 开始训练..."
echo "============================================================"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    "${SCRIPT_DIR}/train.py" \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "✅ Stage 2 训练完成！"
echo "  模型保存在: ${OUTPUT_DIR}/model"
echo "============================================================"
