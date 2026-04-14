#!/bin/bash
# ============================================================
# Stage 1: 生成 10000 条 VCR Latent CoT 训练数据
# ============================================================
# 使用 v3 staged prompt（认知阶段分组 + 视觉空间操作）
# 预计耗时：根据 API 速率，约 3-6 小时
# ============================================================

set -e

# 内部 API 鉴权信息
export ICHAT_APPID="apprbtwqufqr9mqgedn"
export ICHAT_APPKEY="WlokXJldsHNttISobOuRWTKAVZRCjDXJ"

# RTX 账号
RTX="skyejtzhang"

# 数据路径
VCR_ROOT="data/nld_phase1/raw/vcr"
OUTPUT="data/nld_phase1/vcr_latent_cot_v3_stage1_10k.json"
LOG_FILE="data/nld_phase1/generate_v3_stage1_10k.log"

# 生成参数
NUM_SAMPLES=10000
MODEL="gemini-3-pro-preview"
WORKERS=16
TEMPERATURE=0.7
MAX_TOKENS=8192
SAVE_INTERVAL=500
SEED=42

# ============================================================
# 运行前检查
# ============================================================
echo "=========================================="
echo "  Stage 1: VCR Latent CoT 数据生成 (10k)"
echo "=========================================="
echo "  RTX:         ${RTX}"
echo "  模型:        ${MODEL}"
echo "  目标样本数:  ${NUM_SAMPLES}"
echo "  并发线程:    ${WORKERS}"
echo "  输出路径:    ${OUTPUT}"
echo "  日志路径:    ${LOG_FILE}"
echo "  保存间隔:    每 ${SAVE_INTERVAL} 条"
echo "=========================================="

# 检查 VCR 数据是否存在
if [ ! -f "${VCR_ROOT}/train.jsonl" ]; then
    echo "错误: VCR 训练数据不存在: ${VCR_ROOT}/train.jsonl"
    exit 1
fi

# 检查输出文件是否已存在
if [ -f "${OUTPUT}" ]; then
    echo "警告: 输出文件已存在: ${OUTPUT}"
    echo "  已有数据条数: $(python3 -c "import json; print(len(json.load(open('${OUTPUT}'))))" 2>/dev/null || echo '未知')"
    echo "  如需覆盖，请先删除或重命名该文件"
    read -p "  是否继续覆盖? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
fi

echo ""
echo "开始生成..."
echo "日志输出到: ${LOG_FILE}"
echo ""

# ============================================================
# 运行生成脚本
# ============================================================
python3 scripts/generate_vcr_latent_cot.py \
    --vcr_root "${VCR_ROOT}" \
    --output "${OUTPUT}" \
    --rtx "${RTX}" \
    --model "${MODEL}" \
    --num_samples ${NUM_SAMPLES} \
    --workers ${WORKERS} \
    --temperature ${TEMPERATURE} \
    --max_tokens ${MAX_TOKENS} \
    --save_interval ${SAVE_INTERVAL} \
    --seed ${SEED} \
    2>&1 | tee "${LOG_FILE}"

# ============================================================
# 生成完成后的质量检查
# ============================================================
echo ""
echo "=========================================="
echo "  生成完成，执行质量检查..."
echo "=========================================="

python3 -c "
import json

data = json.load(open('${OUTPUT}'))
print(f'总样本数: {len(data)}')

# 任务类型分布
from collections import Counter
task_dist = Counter(d['task_type'] for d in data)
print(f'任务类型分布:')
for task, cnt in sorted(task_dist.items()):
    print(f'  {task}: {cnt} ({cnt/len(data)*100:.1f}%)')

# latent 位置分布
positions = [d.get('latent_position', 0) for d in data]
print(f'Latent 位置: mean={sum(positions)/len(positions):.3f}, min={min(positions):.3f}, max={max(positions):.3f}')

# 认知阶段数分布
stages = [d.get('num_stages', 0) for d in data]
stage_dist = Counter(stages)
print(f'认知阶段数分布:')
for n, cnt in sorted(stage_dist.items()):
    print(f'  {n} stages: {cnt} ({cnt/len(data)*100:.1f}%)')

print(f'\\n数据文件: ${OUTPUT}')
print(f'配置文件: ${OUTPUT}'.replace('.json', '_config.json'))
"

echo ""
echo "Stage 1 数据生成完成!"
