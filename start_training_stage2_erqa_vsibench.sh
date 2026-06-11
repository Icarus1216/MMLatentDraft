#!/bin/bash
# ============================================================
#  RLD Stage-2 · ERQA+VSI-Bench · ckpt-1200 继续训练 (1 epoch)
#  ----------------------------------------------------------
#  数据: ERQA(400) + VSI-Bench(1162) = 1562 条
#  起点: outputs/rld_stage2_swsrs_v6b1b2b3_ckpt200/checkpoint-1200
#  输出: outputs/rld_stage2_erqa_vsibench_ckpt1200/
#
#  用法:
#    bash start_training_stage2_erqa_vsibench.sh
# ============================================================

set -e

echo "🚀 RLD Stage-2 · ERQA+VSI-Bench · ckpt-1200 继续训练 (1 epoch)"
echo ""

# ======================== Step 1: 环境变量 ========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TOKENIZERS_PARALLELISM=false

# ======================== Step 4: 路径与参数 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/configs/rld_stage2_erqa_vsibench_ckpt1200.yaml"
NUM_GPUS=8
MASTER_PORT=${MASTER_PORT:-29506}

# ======================== Step 5: 合并数据 ========================
echo "============================================================"
echo "  📦 Step 0: 合并训练数据 (ERQA + VSI-Bench)"
echo "============================================================"

MERGED_DATA="${SCRIPT_DIR}/data/erqa_vsibench_merged/erqa_vsibench_merged_1562.json"
if [ ! -f "${MERGED_DATA}" ]; then
    echo "  合并数据文件不存在, 正在生成..."
    python3 "${SCRIPT_DIR}/scripts/merge_erqa_vsibench.py"
    echo ""
else
    echo "  ✅ 合并数据已存在: ${MERGED_DATA}"
    NUM_MERGED=$(grep -c '"reasoning_for_training"' "${MERGED_DATA}")
    echo "     样本数: ${NUM_MERGED}"
fi
echo ""

# ======================== Step 6: 预检查 ========================
echo "============================================================"
echo "  🚀 RLD Stage-2 · ERQA+VSI-Bench · ckpt-1200"
echo "============================================================"
echo "  配置文件:     ${CONFIG}"
echo "  GPU 数量:     ${NUM_GPUS}"
echo "  Master Port:  ${MASTER_PORT}"
echo "  分布式后端:   FSDP (full_shard auto_wrap)"
echo "============================================================"

if [ ! -f "${CONFIG}" ]; then
    echo "❌ 配置文件不存在: ${CONFIG}"
    exit 1
fi

# Resume checkpoint 检查
RESUME_CKPT=$(python3 -c "
import yaml
c = yaml.safe_load(open('${CONFIG}'))
r = c.get('nld', {}).get('resume_from_model_only', '') or ''
print(r)
" 2>/dev/null)

if [ -z "${RESUME_CKPT}" ]; then
    echo "❌ YAML 中未配置 nld.resume_from_model_only"
    exit 1
fi

if [ ! -d "${RESUME_CKPT}" ]; then
    echo "❌ Resume checkpoint 目录不存在: ${RESUME_CKPT}"
    exit 1
fi

if ! ls "${RESUME_CKPT}"/*.safetensors >/dev/null 2>&1; then
    echo "❌ Resume checkpoint 缺少 .safetensors 文件: ${RESUME_CKPT}"
    exit 1
fi

echo ""
echo "  ✅ Resume checkpoint: ${RESUME_CKPT}"
CKPT_SIZE=$(du -sh "${RESUME_CKPT}" 2>/dev/null | awk '{print $1}')
echo "     大小: ${CKPT_SIZE}"

# 基座模型
BASE_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model']['model_path'])")
if [ ! -d "${BASE_MODEL}" ]; then
    echo "❌ 基座模型不存在: ${BASE_MODEL}"
    exit 1
fi
echo "  ✅ 基座模型: ${BASE_MODEL}"

# 训练数据
TRAIN_JSON=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['train_json'])")
if [ ! -f "${TRAIN_JSON}" ]; then
    echo "❌ 训练数据不存在: ${TRAIN_JSON}"
    exit 1
fi
echo "  ✅ 训练数据: ${TRAIN_JSON}"

# 数据统计
python3 -c "
import json
from collections import Counter
data = json.load(open('${TRAIN_JSON}'))
n = len(data)
src = Counter(d.get('source','?') for d in data)
tt = Counter(d.get('task_type','?') for d in data)
single = sum(1 for d in data if 'image_path' in d and 'image_paths' not in d)
multi = sum(1 for d in data if 'image_paths' in d)
print(f'  样本数: {n}')
print(f'  来源: {dict(src)}')
print(f'  单图: {single}, 多图: {multi}')
print(f'  task_type: {dict(tt)}')
" 2>/dev/null || true

# 训练规模
python3 -c "
import yaml, json
cfg = yaml.safe_load(open('${CONFIG}'))
tc = cfg['training']
n = len(json.load(open('${TRAIN_JSON}')))
bs = tc['per_device_train_batch_size']
ga = tc['gradient_accumulation_steps']
gpus = ${NUM_GPUS}
steps_per_epoch = n // (bs * ga * gpus)
total_steps = steps_per_epoch * tc['num_epochs']
print(f'')
print(f'  [训练规模]')
print(f'    样本数:          {n}')
print(f'    effective bs:    {bs * ga * gpus}')
print(f'    steps/epoch:     {steps_per_epoch}')
print(f'    epochs:          {tc[\"num_epochs\"]}')
print(f'    total steps:     {total_steps}')
print(f'    lr (VLM/Thinker): {tc[\"vlm_lr\"]} / {tc[\"thinker_lr\"]}')
print(f'    warmup:          {tc[\"warmup_steps\"]} steps')
print(f'    save_steps:      {tc[\"save_steps\"]}')
print(f'    output:          {tc[\"output_dir\"]}')
" 2>/dev/null || true

# GPU 信息
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "  GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'
fi

echo ""
echo "============================================================"

# ======================== Step 7: 创建输出目录 ========================
OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['training']['output_dir'])")
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "🚀 开始训练..."
echo "   起点: ${RESUME_CKPT}"
echo "   输出: ${OUTPUT_DIR}"
echo ""

# ======================== Step 8: 启动训练 ========================
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    scripts/train_nld.py \
    --config "${CONFIG}"

echo ""
echo "============================================================"
echo "✅ 训练完成!"
echo "   模型保存在: ${OUTPUT_DIR}/"
echo "============================================================"
