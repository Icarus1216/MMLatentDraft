#!/bin/bash
# RLD 拒绝采样启动脚本
# 用 Qwen3-VL-8B-Instruct 对训练数据采样多条 CoT，收集错误推理链 + 正确答案配对
#
# 运行模式:
#   MODE=debug   → 只采样 100 个样本, 快速验证流程 (~10min)
#   MODE=full    → 全量 50K 采样 (~8-12h on 4×A100)
#   MODE=mixed   → 全量采样 + 构建混合训练数据
#
# 使用方法:
#   bash start_rejection_sampling.sh              # 默认 debug 模式
#   MODE=full bash start_rejection_sampling.sh    # 全量模式
#   MODE=mixed bash start_rejection_sampling.sh   # 全量 + 混合数据
set -e

echo "🔬 RLD 拒绝采样启动"
echo ""

# ===== 初始化 conda =====
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

# ===== 检测 GPU =====
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
    echo "GPU 数量: $GPU_COUNT"
    
    # 验证每个 GPU 的 CUDA 可用性
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "GPU 列表:"
        nvidia-smi --list-gpus
    fi
else
    echo "⚠️  nvidia-smi 不可用, 请确保在 GPU 节点上运行"
    GPU_COUNT=0
fi

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "❌ 未检测到可用 GPU, 请在 GPU 计算节点上运行此脚本"
    exit 1
fi
echo ""

# ===== 配置 =====

# 模型路径 (与 rld_train.yaml 中一致)
MODEL_PATH="/mnt/wfs/mmchongqingssdwfssz/project_luban_infra/luban_infra/model_factory/for_lucas_Qwen3-VL-8B-Instruct_export"

# 数据路径
INPUT_JSON="data/rld_50k_selected.json"

# 输出目录
OUTPUT_DIR="data/rejection_sampling"
mkdir -p "$OUTPUT_DIR"

# 运行模式: debug / full / mixed (默认 full)
MODE="${MODE:-full}"

# ===== 采样参数 =====
# 每个样本的采样次数 (越多越能区分正确/错误, 但耗时线性增长)
# 推荐: debug=4, full=4, 精细=16
NUM_SAMPLES="${NUM_SAMPLES:-4}"

# 采样温度 (1.0 保证多样性, 超长输出直接丢弃)
TEMPERATURE="${TEMPERATURE:-1.0}"

# 批处理大小 (每批处理的样本数, 每个样本生成 NUM_SAMPLES 个)
# 根据 GPU 显存调整: 4×A100-80G 推荐 32~64
BATCH_SIZE="${BATCH_SIZE:-32}"

# 最大生成 token 数 (2048 不够, 模型推理链较长, 需要 4096 以确保输出完整)
MAX_TOKENS="${MAX_TOKENS:-4096}"

# CoT 最大字符数 (超过则判为无效, 防止超长 CoT 导致训练 OOM)
MAX_COT_CHARS="${MAX_COT_CHARS:-2048}"

# 最大模型上下文长度 (需要足够容纳 image tokens + text prompt + generation)
# Qwen3-VL 的 image tokens 数量取决于图片分辨率, 高分辨率图片可能产生 10000+ tokens
# H20 96GB 显存充足, 设为 16384 以覆盖大部分情况
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# 图片最大像素数 (限制图片分辨率以控制 image tokens 数量)
# Qwen3-VL 默认 min_pixels=256*28*28, max_pixels=1280*28*28
# 设为 602112 (768*28*28) 可将 image tokens 控制在合理范围
MAX_PIXELS="${MAX_PIXELS:-602112}"

# vLLM tensor parallel (使用全部 GPU)
TP_SIZE="${TP_SIZE:-$GPU_COUNT}"

# 错误 CoT 在混合数据中的占比 (仅 mixed 模式)
WRONG_COT_RATIO="${WRONG_COT_RATIO:-0.5}"

# vLLM multiproc 方法: 解决 V1 引擎在容器环境中 CUDA init 失败的问题
# fork 模式下子进程无法正确继承 CUDA 上下文, 必须使用 spawn 创建全新子进程
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# vLLM 引擎版本: 默认使用 V1 引擎
# 如果 V1 引擎有兼容性问题 (如 CUDA init failed), 设置 USE_V0=1 回退到 V0
# 用法: USE_V0=1 bash start_rejection_sampling.sh
if [ "${USE_V0:-0}" = "1" ]; then
    export VLLM_USE_V1=0
    echo "⚠️  已回退到 vLLM V0 引擎"
fi

# ===== 根据模式设置参数 =====
case "$MODE" in
    debug)
        echo "📋 模式: DEBUG (快速验证, 100 个样本)"
        MAX_SAMPLES=100
        NUM_SAMPLES=4
        OUTPUT_JSON="${OUTPUT_DIR}/rld_debug_rejection.json"
        BUILD_MIXED_FLAG=""
        SAVE_RAW_FLAG="--save_raw_outputs"
        ;;
    full)
        echo "📋 模式: FULL (全量 50K 采样)"
        MAX_SAMPLES=""  # 不限制
        OUTPUT_JSON="${OUTPUT_DIR}/rld_50k_rejection_sampled.json"
        BUILD_MIXED_FLAG=""
        SAVE_RAW_FLAG="--save_raw_outputs"
        ;;
    mixed)
        echo "📋 模式: MIXED (全量采样 + 构建混合训练数据)"
        MAX_SAMPLES=""  # 不限制
        OUTPUT_JSON="${OUTPUT_DIR}/rld_50k_rejection_sampled.json"
        MIXED_OUTPUT_JSON="${OUTPUT_DIR}/rld_mixed_training.json"
        BUILD_MIXED_FLAG="--build_mixed --wrong_cot_ratio ${WRONG_COT_RATIO} --mixed_output_json ${MIXED_OUTPUT_JSON}"
        SAVE_RAW_FLAG=""
        ;;
    *)
        echo "❌ 未知模式: $MODE (可选: debug / full / mixed)"
        exit 1
        ;;
esac

echo ""
echo "配置信息:"
echo "  模型路径:     $MODEL_PATH"
echo "  输入数据:     $INPUT_JSON"
echo "  输出路径:     $OUTPUT_JSON"
echo "  Tensor Parallel: $TP_SIZE GPU(s)"
echo "  每样本采样数: $NUM_SAMPLES"
echo "  采样温度:     $TEMPERATURE"
echo "  批大小:       $BATCH_SIZE"
echo "  最大 token:   $MAX_TOKENS"
echo "  CoT 字符上限: $MAX_COT_CHARS"
echo "  最大上下文:   $MAX_MODEL_LEN"
echo "  图片最大像素: $MAX_PIXELS"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  最大样本数:   $MAX_SAMPLES"
fi
if [ -n "$BUILD_MIXED_FLAG" ]; then
    echo "  混合数据:     $MIXED_OUTPUT_JSON"
    echo "  错误 CoT 占比: $WRONG_COT_RATIO"
fi
echo ""

# ===== 检查依赖 =====
echo "🔍 检查依赖..."

# 检查 vLLM
python3 -c "import vllm; print(f'  ✅ vLLM {vllm.__version__}')" 2>/dev/null || {
    echo "  ❌ vLLM 未安装, 请运行: pip install vllm"
    exit 1
}

# 检查 CUDA 可用性 (通过 PyTorch)
CUDA_GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "  ✅ PyTorch 可见 GPU: $CUDA_GPU_COUNT"
if [ "$CUDA_GPU_COUNT" -eq 0 ]; then
    echo "  ❌ PyTorch 无法检测到 CUDA GPU, 请检查 CUDA 驱动和环境"
    exit 1
fi

# 自动修正 TP_SIZE: 不能超过实际可用 GPU 数
if [ "$TP_SIZE" -gt "$CUDA_GPU_COUNT" ]; then
    echo "  ⚠️  TP_SIZE=$TP_SIZE > 可用GPU=$CUDA_GPU_COUNT, 自动调整为 $CUDA_GPU_COUNT"
    TP_SIZE=$CUDA_GPU_COUNT
fi

# 检查输入数据
if [ ! -f "$INPUT_JSON" ]; then
    echo "  ❌ 输入数据不存在: $INPUT_JSON"
    exit 1
fi
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$INPUT_JSON'))))")
echo "  ✅ 输入数据: $SAMPLE_COUNT 个样本"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "  ❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi
echo "  ✅ 模型路径存在"

# 检查脚本
if [ ! -f "scripts/rejection_sampling.py" ]; then
    echo "  ❌ 拒绝采样脚本不存在: scripts/rejection_sampling.py"
    exit 1
fi
echo "  ✅ 拒绝采样脚本存在"

echo ""

# ===== 显存预估 =====
echo "📊 资源预估:"
if [ -n "$MAX_SAMPLES" ]; then
    TOTAL_GEN=$((MAX_SAMPLES * NUM_SAMPLES))
else
    TOTAL_GEN=$((SAMPLE_COUNT * NUM_SAMPLES))
fi
echo "  总生成次数: $TOTAL_GEN"
echo "  预估耗时: ~$((TOTAL_GEN / 3000))min (基于 ~3000 gen/min on 4×A100)"
echo ""

# ===== 构建命令 =====
CMD="python3 scripts/rejection_sampling.py \
    --model_path $MODEL_PATH \
    --input_json $INPUT_JSON \
    --output_json $OUTPUT_JSON \
    --tensor_parallel_size $TP_SIZE \
    --num_samples $NUM_SAMPLES \
    --temperature $TEMPERATURE \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    --max_cot_chars $MAX_COT_CHARS \
    --max_model_len $MAX_MODEL_LEN \
    --max_pixels $MAX_PIXELS \
    --checkpoint_interval 5000"

# 添加可选参数
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$BUILD_MIXED_FLAG" ]; then
    CMD="$CMD $BUILD_MIXED_FLAG"
fi

if [ -n "$SAVE_RAW_FLAG" ]; then
    CMD="$CMD $SAVE_RAW_FLAG"
    echo "📋 诊断模式: 将保存模型原始输出到 ${OUTPUT_JSON%.json}_raw_outputs.json"
fi

# ===== 执行 =====
echo "🚀 开始拒绝采样..."
echo "命令: $CMD"
echo ""

START_TIME=$(date +%s)

eval $CMD

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "⏱️  总耗时: ${ELAPSED_MIN}min ${ELAPSED_SEC}s"
echo ""

# ===== 结果汇总 =====
echo "📦 输出文件:"
if [ -f "$OUTPUT_JSON" ]; then
    PAIRED_COUNT=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_JSON'))))")
    FILE_SIZE=$(du -h "$OUTPUT_JSON" | cut -f1)
    echo "  ✅ 拒绝采样结果: $OUTPUT_JSON ($PAIRED_COUNT 条, $FILE_SIZE)"
fi

STATS_JSON="${OUTPUT_JSON%.json}_stats.json"
if [ -f "$STATS_JSON" ]; then
    echo "  ✅ 统计信息: $STATS_JSON"
fi

if [ -n "$MIXED_OUTPUT_JSON" ] && [ -f "$MIXED_OUTPUT_JSON" ]; then
    MIXED_COUNT=$(python3 -c "import json; print(len(json.load(open('$MIXED_OUTPUT_JSON'))))")
    MIXED_SIZE=$(du -h "$MIXED_OUTPUT_JSON" | cut -f1)
    echo "  ✅ 混合训练数据: $MIXED_OUTPUT_JSON ($MIXED_COUNT 条, $MIXED_SIZE)"
    echo ""
    echo "📝 下一步: 修改 configs/rld_train.yaml 中的 train_json 为:"
    echo "    train_json: \"$MIXED_OUTPUT_JSON\""
fi

echo ""
echo "✅ 拒绝采样完成！"
