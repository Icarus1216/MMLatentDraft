#!/bin/bash
# ============================================================
# run_modality_manifold_analysis.sh
# 一键运行"模态流形分析" pipeline
#
# 分为两部分:
#   Part 1: 宿主机可运行 — 提取 trainer_state 指标 + 可视化
#   Part 2: Docker 可运行 — 提取 embeddings + CKA 计算
#
# 用法:
#   # 宿主机: 运行 Part 1
#   bash run_modality_manifold_analysis.sh --part 1
#
#   # 宿主机: 运行全部
#   bash run_modality_manifold_analysis.sh
#
#   # Docker: 运行 Part 2
#   bash run_modality_manifold_analysis.sh --part 2
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
FIGURES_DIR="${SCRIPT_DIR}/../figures"

PART="${1:-all}"

# 训练数据路径
CKPT1200_STATE="${PROJ_ROOT}/outputs/<RUN_NAME>/checkpoint-1200/trainer_state.json"
CKPT900_STATE="${PROJ_ROOT}/outputs/<RUN_NAME>/checkpoint-900/trainer_state.json"
CKPT600_STATE="${PROJ_ROOT}/outputs/<RUN_NAME>/checkpoint-600/trainer_state.json"

# Baseline 模型路径 (Docker 内)
BASELINE_MODEL="/workspace/LatentDraft/outputs/modality_geometry_v2"
CKPT1200_EMB="/workspace/LatentDraft/outputs/modality_geometry_v2_ckpt1200"

echo "============================================================"
echo "  Modality Manifold Analysis Pipeline"
echo "============================================================"
echo "  Project: ${PROJ_ROOT}"
echo "  Results: ${RESULTS_DIR}"
echo "  Figures: ${FIGURES_DIR}"
echo "  Part:    ${PART}"
echo "============================================================"

# ============================================================
# Part 1: 宿主机可运行
# ============================================================
run_part1() {
    echo ""
    echo "📊 Part 1: 提取训练指标 & 可视化 (宿主机)"
    echo "------------------------------------------------------------"

    # Step 1: 提取训练曲线指标
    echo ""
    echo "  [1/3] 提取训练曲线指标..."
    python3 -u "${SCRIPT_DIR}/extract_training_metrics.py" \
        --ckpt_path "${CKPT1200_STATE}" \
        --output_dir "${RESULTS_DIR}"

    # 同时提取 ckpt-900 和 ckpt-600 用于对比
    echo ""
    echo "  [1b] 提取 ckpt-900 指标..."
    python3 -u "${SCRIPT_DIR}/extract_training_metrics.py" \
        --ckpt_path "${CKPT900_STATE}" \
        --output_dir "${RESULTS_DIR}/ckpt900"

    echo ""
    echo "  [1c] 提取 ckpt-600 指标..."
    python3 -u "${SCRIPT_DIR}/extract_training_metrics.py" \
        --ckpt_path "${CKPT600_STATE}" \
        --output_dir "${RESULTS_DIR}/ckpt600"

    # Step 2: 生成可视化
    echo ""
    echo "  [2/3] 生成可视化图表..."
    python3 -u "${SCRIPT_DIR}/visualize_modality_manifold.py" \
        --results_dir "${RESULTS_DIR}" \
        --figures_dir "${FIGURES_DIR}"

    # Step 3: 从已有 embeddings 计算 CKA (如果存在)
    echo ""
    echo "  [3/3] 计算 CKA (从已有 embeddings)..."
    EMB_DIRS=""
    for d in "${PROJ_ROOT}/outputs/modality_geometry_v2" \
             "${PROJ_ROOT}/outputs/modality_geometry_v2_ckpt1200"; do
        if [ -d "$d" ]; then
            EMB_DIRS="${d}"
        fi
    done
    if [ -n "${EMB_DIRS}" ]; then
        python3 -u "${SCRIPT_DIR}/extract_embeddings_for_cka.py" \
            --embeddings_dir "${EMB_DIRS}" \
            --output_dir "${RESULTS_DIR}" \
            --skip_model
    else
        echo "    ⚠️ No existing embeddings found, skipping CKA computation"
    fi

    echo ""
    echo "  ✅ Part 1 完成!"
    echo "  📊 图表位于: ${FIGURES_DIR}/"
    echo "  📊 数据位于: ${RESULTS_DIR}/"
}

# ============================================================
# Part 2: Docker 可运行
# ============================================================
run_part2() {
    echo ""
    echo "🔧 Part 2: 提取 embeddings & 精确 CKA (Docker GPU)"
    echo "------------------------------------------------------------"

    # 检查 GPU
    if ! command -v nvidia-smi &>/dev/null; then
        echo "  ⚠️ nvidia-smi not found, GPU may not be available"
    fi

    # 从已有 embeddings 计算
    echo ""
    echo "  [1/2] 从已有 embeddings 计算 CKA..."
    for d in "${BASELINE_MODEL}" "${CKPT1200_EMB}"; do
        if [ -d "$d" ]; then
            python3 -u "${SCRIPT_DIR}/extract_embeddings_for_cka.py" \
                --embeddings_dir "$d" \
                --output_dir "${RESULTS_DIR}" \
                --skip_model
        fi
    done

    # 从模型提取 (需要 GPU)
    echo ""
    echo "  [2/2] 从模型提取 embeddings (可选)..."
    echo "    运行以下命令手动执行:"
    echo "    python3 ${SCRIPT_DIR}/extract_embeddings_for_cka.py --model_path /path/to/model"

    echo ""
    echo "  ✅ Part 2 完成!"
}

# ============================================================
# Run
# ============================================================
mkdir -p "${RESULTS_DIR}" "${FIGURES_DIR}"

case "${PART}" in
    1|--part\ 1)
        run_part1
        ;;
    2|--part\ 2)
        run_part2
        ;;
    all|"")
        run_part1
        echo ""
        echo "💡 Part 2 需要 Docker GPU 环境, 请手动运行:"
        echo "  bash ${SCRIPT_DIR}/run_modality_manifold_analysis.sh --part 2"
        ;;
    *)
        echo "Unknown part: ${PART}"
        echo "Usage: bash $0 [--part 1|2|all]"
        exit 1
        ;;
esac
