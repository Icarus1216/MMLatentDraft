#!/bin/bash
# ============================================================
#  RLD Stage-2 SW-SRS · v6 b1+b2+b3 · ckpt-200 起点 训练启动脚本
#  ----------------------------------------------------------
#  目标:
#    从 outputs/rld_stage2/checkpoint-200 (HF Trainer 原生格式) 仅加载模型权重,
#    切到 SW-SRS loss + 3 层 anti-collapse, 在 v6 b1+b2+b3 修复后的 19195 样本
#    训练数据上精修 hidden 几何.
#
#  配置文件: configs/rld_stage2_swsrs_v6b1b2b3_ckpt200.yaml
#  起点:     outputs/rld_stage2/checkpoint-200 (model.safetensors 权重)
#  数据:     data/v6/v6_b1b2b3_merged_training_slim_vsp_fixed.json (19195 samples)
#
#  用法:
#    bash start_training_stage2_v6b1b2b3_ckpt200.sh
#    bash start_training_stage2_v6b1b2b3_ckpt200.sh --gpus 4
#    bash start_training_stage2_v6b1b2b3_ckpt200.sh --config /xxx/yyy.yaml
#
#  注意:
#    - 走 nld.resume_from_model_only: 仅加载 model 权重, optimizer/scheduler/step 全新.
#    - HF Trainer 不会从 ckpt-200 续上 200 step, 本次从 step 0 开始.
# ============================================================

set -e

echo "🚀 RLD Stage-2 SW-SRS · v6 b1+b2+b3 · ckpt-200 起点训练启动"
echo "   起点: outputs/rld_stage2/checkpoint-200"
echo ""

# ======================== Step 1: 环境变量 ========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TOKENIZERS_PARALLELISM=false

# ======================== Step 4: 路径与参数 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/configs/rld_stage2_swsrs_v6b1b2b3_ckpt200.yaml"
NUM_GPUS=8
# 错开端口: cold-start=29503, swsrs_resume=29504, 本次=29505
MASTER_PORT=${MASTER_PORT:-29505}

# ======================== Step 5: 解析命令行参数 ========================
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# ======================== Step 6: 预检查 ========================
echo "============================================================"
echo "  🚀 RLD Stage-2 SW-SRS · v6 b1+b2+b3 · ckpt-200"
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

# ---- Resume sanity check ----
RESUME_CKPT=$(python3 -c "
import yaml
c = yaml.safe_load(open('${CONFIG}'))
r = c.get('nld', {}).get('resume_from_model_only', '') or ''
print(r)
" 2>/dev/null)

if [ -z "${RESUME_CKPT}" ]; then
    echo "❌ YAML 中未配置 nld.resume_from_model_only, 本脚本专门用于 'model-only resume'."
    exit 1
fi

if [[ "${RESUME_CKPT}" != /* ]]; then
    RESUME_CKPT_ABS="${SCRIPT_DIR}/${RESUME_CKPT}"
else
    RESUME_CKPT_ABS="${RESUME_CKPT}"
fi

if [ ! -d "${RESUME_CKPT_ABS}" ]; then
    echo "❌ Resume checkpoint 目录不存在: ${RESUME_CKPT_ABS}"
    exit 1
fi

if ! ls "${RESUME_CKPT_ABS}"/*.safetensors >/dev/null 2>&1; then
    echo "❌ Resume checkpoint 缺少 .safetensors 文件: ${RESUME_CKPT_ABS}"
    exit 1
fi

echo ""
echo "  ✅ Resume checkpoint (model-only): ${RESUME_CKPT_ABS}"
CKPT_SIZE=$(du -sh "${RESUME_CKPT_ABS}" 2>/dev/null | awk '{print $1}')
echo "     大小:                          ${CKPT_SIZE}"
echo "     加载方式:                      仅 model 权重 (strict=False + shape 适配)"
echo "     Trainer:                       step 0 开始 (optimizer / scheduler / RNG 全新)"

if [ -f "${RESUME_CKPT_ABS}/trainer_state.json" ]; then
    CKPT_STEP=$(python3 -c "
import json
s = json.load(open('${RESUME_CKPT_ABS}/trainer_state.json'))
print(s.get('global_step', '?'), s.get('epoch', '?'))
" 2>/dev/null)
    echo "     (参考) 原 ckpt:                global_step=$(echo ${CKPT_STEP} | awk '{print $1}'), epoch=$(echo ${CKPT_STEP} | awk '{print $2}') — 本次不会续上"
fi

# 基座模型
BASE_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model']['model_path'])")
if [ ! -d "${BASE_MODEL}" ]; then
    echo "❌ 基座模型不存在: ${BASE_MODEL}"
    exit 1
fi
echo ""
echo "  ✅ 基座模型 (tokenizer/config 来源): ${BASE_MODEL}"

# 训练数据
TRAIN_JSON=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['train_json'])")
if [ ! -f "${TRAIN_JSON}" ]; then
    echo "❌ 训练数据不存在: ${TRAIN_JSON}"
    exit 1
fi

NUM_SAMPLES=$(python3 -c "import json; print(len(json.load(open('${TRAIN_JSON}'))))")
echo ""
echo "  训练数据:     ${TRAIN_JSON}"
echo "  样本数:       ${NUM_SAMPLES}"

# 分布摘要 (v6 schema: task_type / boundary 数 / pause-boundary 对齐率)
python3 -c "
import json
from collections import Counter
data = json.load(open('${TRAIN_JSON}'))
n = len(data)
tt = Counter(d.get('task_type','?') for d in data)
bd = Counter(len(d.get('latent_key_tokens', [])) for d in data)
align = sum(
    1 for d in data
    if (d.get('reasoning_for_training') or '').count('<|pause|>')
       == len(d.get('latent_key_tokens', []))
)
print('  task_type:')
for k,v in tt.most_common():
    print(f'    {k:20s} {v:6d} ({v/n*100:.1f}%)')
print('  boundary count (per sample):')
for k,v in sorted(bd.items()):
    print(f'    {k} boundary  {v:6d} ({v/n*100:.1f}%)')
print(f'  pause/boundary aligned: {align}/{n} ({align/n*100:.2f}%)')
" 2>/dev/null || true

# 关键超参摘要
python3 -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
mc = cfg['model']
nc = cfg.get('nld', {})
tc = cfg['training']
print()
print('  [Attention 实现]:', mc.get('attn_implementation', 'N/A'))
print('  [NLD Thinker / SW-SRS]:')
print(f'    max_think_steps:            {nc.get(\"max_think_steps\", \"N/A\")}')
print(f'    saturation_exit_threshold:  {nc.get(\"saturation_exit_threshold\", \"N/A\")}')
print(f'    key_token_weight (SW-SRS):  {nc.get(\"key_token_weight\", \"N/A\")}')
print(f'    exit_margin / margin_weight:{nc.get(\"exit_margin\", \"N/A\")} / {nc.get(\"exit_margin_weight\", \"N/A\")}')
print(f'    sw_srs_alpha:               {nc.get(\"sw_srs_alpha\", \"N/A\")}')
print(f'    diversity_threshold:        {nc.get(\"diversity_threshold\", \"N/A\")}')
print('  [学习率 / 调度]:')
print(f'    VLM lr:      {tc.get(\"vlm_lr\", \"N/A\")}')
print(f'    Thinker lr:  {tc.get(\"thinker_lr\", \"N/A\")}')
print(f'    Warmup:      {tc.get(\"warmup_steps\", \"N/A\")} steps')
print(f'    Scheduler:   {tc.get(\"lr_scheduler_type\", \"N/A\")}')
print(f'    Epochs:      {tc.get(\"num_epochs\", \"N/A\")}')
print(f'    save_steps:  {tc.get(\"save_steps\", \"N/A\")}  (limit={tc.get(\"save_total_limit\", \"N/A\")})')
print(f'    Output:      {tc.get(\"output_dir\", \"N/A\")}')
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
mkdir -p "${SCRIPT_DIR}/logs"

if ls "${OUTPUT_DIR}"/checkpoint-* >/dev/null 2>&1; then
    echo ""
    echo "⚠️  output_dir 下已有 checkpoint: ${OUTPUT_DIR}"
    echo "    本次 model-only resume 不会读这些 ckpt, 但 save_total_limit 可能会清理掉它们."
    read -p "继续 (保留已有 ckpt)? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 开始训练..."
echo "   起点目录: ${RESUME_CKPT_ABS}"
echo "   输出目录: ${OUTPUT_DIR}"
echo "   日志:     ${OUTPUT_DIR}/logs/  (tensorboard --logdir ${OUTPUT_DIR}/logs)"
echo ""

# ======================== Step 8: 启动训练 ========================
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    scripts/train_nld.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

echo ""
echo "============================================================"
echo "✅ 训练完成!"
echo "   模型保存在: ${OUTPUT_DIR}/model/"
echo "============================================================"
