#!/bin/bash
# ============================================================
# Stage 1 Batch 2: 生成第二批 10000 条 VCR Latent CoT 训练数据
# ============================================================
# 与 Batch 1 的关键区别：
#   1. 排除 Batch 1 已使用的图片（--exclude_images）
#   2. 调整任务类型权重，补偿 Batch 1 的不均衡
#   3. 温度从 0.7 → 0.8，增加多样性
#   4. seed 从 42 → 123，避免随机选择模式重叠
#   5. 在 prompt 中增加反模式固化指令
# ============================================================

set -e

# 内部 API 鉴权信息
export ICHAT_APPID="apprbtwqufqr9mqgedn"
export ICHAT_APPKEY="WlokXJldsHNttISobOuRWTKAVZRCjDXJ"

# RTX 账号
RTX="skyejtzhang"

# 数据路径
VCR_ROOT="data/nld_phase1/raw/vcr"
BATCH1_DATA="data/nld_phase1/vcr_latent_cot_v3_stage1_10k.json"
OUTPUT="data/nld_phase1/vcr_latent_cot_v3_stage1_batch2_10k.json"
LOG_FILE="data/nld_phase1/generate_v3_stage1_batch2_10k.log"

# 生成参数（与 Batch 1 的区别已标注）
NUM_SAMPLES=10000
MODEL="gemini-3-pro-preview"
WORKERS=16
TEMPERATURE=0.8          # ← Batch 1 是 0.7，提高多样性
MAX_TOKENS=8192
SAVE_INTERVAL=500
SEED=123                 # ← Batch 1 是 42，避免重叠

# ============================================================
# 运行前检查
# ============================================================
echo "=========================================="
echo "  Stage 1 Batch 2: VCR Latent CoT (10k)"
echo "=========================================="
echo "  RTX:         ${RTX}"
echo "  模型:        ${MODEL}"
echo "  目标样本数:  ${NUM_SAMPLES}"
echo "  并发线程:    ${WORKERS}"
echo "  温度:        ${TEMPERATURE} (Batch 1: 0.7)"
echo "  Seed:        ${SEED} (Batch 1: 42)"
echo "  输出路径:    ${OUTPUT}"
echo "  日志路径:    ${LOG_FILE}"
echo "  排除图片源:  ${BATCH1_DATA}"
echo "=========================================="

# 检查 VCR 数据是否存在
if [ ! -f "${VCR_ROOT}/train.jsonl" ]; then
    echo "错误: VCR 训练数据不存在: ${VCR_ROOT}/train.jsonl"
    exit 1
fi

# 检查 Batch 1 数据是否存在（用于排除已用图片）
if [ ! -f "${BATCH1_DATA}" ]; then
    echo "错误: Batch 1 数据不存在: ${BATCH1_DATA}"
    echo "  请先完成 Batch 1 的生成"
    exit 1
fi

# 统计 Batch 1 已用图片数
BATCH1_COUNT=$(python3 -c "import json; print(len(json.load(open('${BATCH1_DATA}'))))" 2>/dev/null || echo '未知')
echo "  Batch 1 已有样本: ${BATCH1_COUNT}"

# 检查输出文件是否已存在
if [ -f "${OUTPUT}" ]; then
    echo "警告: 输出文件已存在: ${OUTPUT}"
    echo "  已有数据条数: $(python3 -c "import json; print(len(json.load(open('${OUTPUT}'))))" 2>/dev/null || echo '未知')"
    read -p "  是否继续覆盖? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
fi

echo ""
echo "开始生成 Batch 2..."
echo "日志输出到: ${LOG_FILE}"
echo ""

# ============================================================
# 运行生成脚本（带 Batch 2 专用参数）
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
    --exclude_images "${BATCH1_DATA}" \
    --batch2_diversity \
    2>&1 | tee "${LOG_FILE}"

# ============================================================
# 生成完成后的质量检查
# ============================================================
echo ""
echo "=========================================="
echo "  Batch 2 生成完成，执行质量检查..."
echo "=========================================="

python3 -c "
import json
from collections import Counter

# 加载两批数据
batch1 = json.load(open('${BATCH1_DATA}'))
batch2 = json.load(open('${OUTPUT}'))

print(f'=== Batch 2 基本统计 ===')
print(f'Batch 2 样本数: {len(batch2)}')

# 检查图片重复
b1_images = set(d['image'] for d in batch1)
b2_images = set(d['image'] for d in batch2)
overlap = b1_images & b2_images
print(f'与 Batch 1 图片重叠: {len(overlap)} (应为 0)')

# 任务类型分布对比
print(f'\n=== 任务类型分布对比 ===')
b1_tasks = Counter(d['task_type'] for d in batch1)
b2_tasks = Counter(d['task_type'] for d in batch2)
all_tasks = sorted(set(list(b1_tasks.keys()) + list(b2_tasks.keys())))
print(f'{\"任务类型\":<25} {\"Batch1\":>8} {\"Batch2\":>8} {\"合计\":>8} {\"合计%\":>8}')
for t in all_tasks:
    b1c = b1_tasks.get(t, 0)
    b2c = b2_tasks.get(t, 0)
    total = b1c + b2c
    pct = total / (len(batch1) + len(batch2)) * 100
    print(f'{t:<25} {b1c:>8} {b2c:>8} {total:>8} {pct:>7.1f}%')

# Latent 内容主题对比
print(f'\n=== Latent 内容主题对比 ===')
def count_themes(data):
    themes = Counter()
    for d in data:
        lt = d.get('latent_text', '').lower()
        if 'rotat' in lt: themes['rotation'] += 1
        if 'light' in lt or 'shadow' in lt: themes['light_shadow'] += 1
        if 'gravity' in lt or 'balance' in lt or 'torque' in lt: themes['physics'] += 1
        if 'occlu' in lt: themes['occlusion'] += 1
        if 'trajectory' in lt or 'path' in lt: themes['trajectory'] += 1
        if 'perspect' in lt or 'viewpoint' in lt: themes['perspective'] += 1
        if 'depth' in lt or '3d' in lt: themes['3d_depth'] += 1
        if 'emotion' in lt or 'feel' in lt or 'express' in lt: themes['emotion'] += 1
        if 'social' in lt or 'interact' in lt or 'gesture' in lt: themes['social'] += 1
        if 'texture' in lt or 'material' in lt or 'surface' in lt: themes['material'] += 1
        if 'causal' in lt or 'cause' in lt or 'effect' in lt: themes['causal'] += 1
    return themes

b1_themes = count_themes(batch1)
b2_themes = count_themes(batch2)
all_themes = sorted(set(list(b1_themes.keys()) + list(b2_themes.keys())))
print(f'{\"主题\":<15} {\"Batch1\":>8} {\"Batch2\":>8} {\"变化\":>8}')
for t in all_themes:
    b1c = b1_themes.get(t, 0)
    b2c = b2_themes.get(t, 0)
    change = '+' if b2c > b1c else ('-' if b2c < b1c else '=')
    print(f'{t:<15} {b1c:>8} {b2c:>8} {change:>8}')

# Key tokens 多样性
all_tokens_b2 = []
for d in batch2:
    for s in d.get('latent_key_tokens', []):
        if isinstance(s, dict):
            all_tokens_b2.extend(s.get('tokens', []))
token_freq = Counter(all_tokens_b2)
print(f'\n=== Batch 2 Key Tokens 多样性 ===')
print(f'  总tokens: {len(all_tokens_b2)}, 唯一: {len(token_freq)}, 唯一率: {len(token_freq)/max(len(all_tokens_b2),1)*100:.1f}%')

print(f'\n=== 合并后总计 ===')
print(f'  总样本: {len(batch1) + len(batch2)}')
print(f'  唯一图片: {len(b1_images | b2_images)}')
"

echo ""
echo "=========================================="
echo "  Batch 2 完成！"
echo "  如需合并两批数据，运行："
echo "  python3 -c \"import json; b1=json.load(open('${BATCH1_DATA}')); b2=json.load(open('${OUTPUT}')); json.dump(b1+b2, open('data/nld_phase1/vcr_latent_cot_v3_stage1_20k.json','w'), ensure_ascii=False)\""
echo "=========================================="
