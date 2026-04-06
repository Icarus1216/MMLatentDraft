#!/bin/bash
# GPU 硬件诊断脚本 — 排查 cudaErrorContained (error 226)
# 用法: bash scripts/gpu_diag.sh

echo "=========================================="
echo "🔍 GPU 硬件诊断 (cudaErrorContained 排查)"
echo "=========================================="
echo ""

# 1. GPU 基本信息
echo "📋 1. GPU 型号和驱动"
echo "-------------------------------------------"
nvidia-smi --query-gpu=index,name,driver_version,cuda_version --format=csv,noheader
echo ""

# 2. ECC 错误统计 (关键!)
echo "📋 2. ECC 错误统计 (不可纠正错误 = 硬件故障)"
echo "-------------------------------------------"
nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv
echo ""
echo "⚠️ uncorrected.volatile > 0 或 uncorrected.aggregate > 0 表示有不可纠正的ECC错误!"
echo ""

# 3. 已退役的显存页 (Retired Pages)
echo "📋 3. 已退役显存页 (有退役页 = GPU 显存有坏块)"
echo "-------------------------------------------"
nvidia-smi --query-retired-pages=gpu_uuid,address,cause --format=csv 2>/dev/null || echo "(此驱动版本不支持查询退役页)"
echo ""

# 4. Row Remapper 状态
echo "📋 4. Row Remapper 状态"
echo "-------------------------------------------"
nvidia-smi --query-remapped-rows=gpu_uuid,remapped_rows.correctable,remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure --format=csv 2>/dev/null || echo "(此驱动版本不支持查询 Row Remapper)"
echo ""

# 5. NVLink 状态
echo "📋 5. NVLink 链路状态"
echo "-------------------------------------------"
for gpu_id in $(seq 0 7); do
    echo "--- GPU $gpu_id ---"
    nvidia-smi nvlink -s -i $gpu_id 2>/dev/null || echo "  (无法查询 GPU $gpu_id 的 NVLink)"
done
echo ""

# 6. NVLink 错误计数
echo "📋 6. NVLink 错误计数 (非零 = 链路不稳定)"
echo "-------------------------------------------"
for gpu_id in $(seq 0 7); do
    echo "--- GPU $gpu_id ---"
    nvidia-smi nvlink -e -i $gpu_id 2>/dev/null || echo "  (无法查询 GPU $gpu_id 的 NVLink 错误)"
done
echo ""

# 7. 内核日志中的 XID 错误
echo "📋 7. 最近的 NVIDIA XID 错误 (dmesg)"
echo "-------------------------------------------"
dmesg 2>/dev/null | grep -i "xid\|nvrm\|nvidia" | tail -30 || echo "(无法读取 dmesg，可能需要 root 权限)"
echo ""

# 8. GPU 温度和功耗
echo "📋 8. GPU 温度和功耗"
echo "-------------------------------------------"
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit --format=csv,noheader
echo ""

# 9. P2P 矩阵
echo "📋 9. GPU P2P 拓扑"
echo "-------------------------------------------"
nvidia-smi topo -m
echo ""

echo "=========================================="
echo "诊断完成"
echo ""
echo "🔴 如果发现以下任一问题，请更换机器:"
echo "   - uncorrected ECC errors > 0"
echo "   - 有 retired pages"
echo "   - NVLink 错误计数 > 0"
echo "   - dmesg 中有 XID 错误 (特别是 XID 48, 63, 64, 74, 79)"
echo "=========================================="
