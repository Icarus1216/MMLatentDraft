#!/bin/bash
# ============================================================
# NLD 训练/推理启动脚本 - 适配 Gemini 平台
#
# 功能:
#   1. 自动探测并初始化 conda 环境
#   2. 激活 colamem conda 环境
#   3. 启动训练或推理任务
#
# 用法:
#   bash start_training.sh                          # 默认启动训练
#   bash start_training.sh train                    # 启动训练
#   bash start_training.sh train --resume /path     # 恢复训练
#   bash start_training.sh inference                # 启动推理
#   bash start_training.sh inference --checkpoint /path/to/ckpt
# ============================================================

set -e

echo "🚀 NLD 训练启动 (Gemini 平台)"
echo ""

# ======================== Step 1: 初始化 conda ========================
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "🔧 初始化 conda (miniconda3)..."
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo "🔧 初始化 conda (/opt/conda)..."
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "🔧 初始化 conda ($HOME/miniconda3)..."
    source $HOME/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "🔧 初始化 conda (anaconda3)..."
    source $HOME/anaconda3/etc/profile.d/conda.sh
elif [ -n "$CONDA_EXE" ]; then
    echo "🔧 使用环境变量中的 conda..."
    eval "$($CONDA_EXE shell.bash hook)"
else
    echo "⚠️  未找到 conda，尝试直接使用 conda 命令..."
fi

# ======================== Step 2: 激活 conda 环境 ========================
if command -v conda &> /dev/null; then
    echo "✅ conda 已就绪"
    
    # 检查 colamem 环境是否存在
    if conda env list | grep -q "^colamem "; then
        echo "🚀 激活 conda 环境: colamem"
        conda activate colamem
        echo "✅ conda 环境已激活: $(conda info --envs | grep '*' | awk '{print $1}')"
    else
        echo "⚠️  未找到 conda 环境 'colamem'，使用当前环境"
    fi
else
    echo "⚠️  conda 命令不可用，尝试激活 Python 虚拟环境..."
    
    # 回退到 Python 虚拟环境
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -d "/root/colamem" ]; then
            echo "检测到虚拟环境 /root/colamem，正在激活..."
            source /root/colamem/bin/activate
        elif [ -d "$HOME/colamem" ]; then
            echo "检测到虚拟环境 $HOME/colamem，正在激活..."
            source $HOME/colamem/bin/activate
        elif [ -d "colamem" ]; then
            echo "检测到虚拟环境 ./colamem，正在激活..."
            source colamem/bin/activate
        elif [ -d "venv" ]; then
            echo "检测到虚拟环境 ./venv，正在激活..."
            source venv/bin/activate
        else
            echo "⚠️  未检测到任何环境，使用系统 Python"
        fi
    else
        echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
    fi
fi
echo ""

# ======================== Step 3: 检测 Python ========================
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "Python: $($PYTHON --version)"
echo ""

# ======================== Step 4: 检测 GPU ========================
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "8")
    echo "============================================================"
    echo "🔍 GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo "============================================================"
else
    GPU_COUNT=8  # Gemini 平台默认 8 卡
fi

echo "GPU 数量: $GPU_COUNT"
echo ""

# ======================== Step 5: 解析参数 ========================
MODE="${1:-train}"  # 默认训练模式
shift 2>/dev/null || true

# 收集剩余参数传递给子脚本
EXTRA_ARGS="$@"

# ======================== Step 6: 执行任务 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${MODE}" in
    train|training)
        echo "============================================================"
        echo "🚀 启动 NLD Phase 1 训练"
        echo "============================================================"
        echo ""
        
        # 创建输出目录
        mkdir -p outputs/nld_train
        
        # 调用训练脚本，传递额外参数（包括 --gpus 覆盖自动检测的值）
        bash "${SCRIPT_DIR}/scripts/run_train_nld.sh" --gpus ${GPU_COUNT} ${EXTRA_ARGS}
        ;;
    
    infer|inference|eval)
        echo "============================================================"
        echo "🚀 启动 NLD 推理"
        echo "============================================================"
        echo ""
        
        $PYTHON "${SCRIPT_DIR}/scripts/inference.py" ${EXTRA_ARGS}
        ;;
    
    *)
        echo "用法: bash start_training.sh [train|inference] [额外参数...]"
        echo ""
        echo "模式:"
        echo "  train       启动 NLD Phase 1 训练 (默认)"
        echo "  inference   启动推理/评估"
        echo ""
        echo "训练额外参数:"
        echo "  --resume /path/to/checkpoint   从检查点恢复训练"
        echo "  --config /path/to/config.yaml  指定配置文件"
        echo "  --gpus N                       指定 GPU 数量"
        echo ""
        echo "推理额外参数:"
        echo "  --checkpoint /path/to/ckpt     指定模型检查点"
        exit 1
        ;;
esac

echo ""
echo "✅ ${MODE} 任务完成！"
