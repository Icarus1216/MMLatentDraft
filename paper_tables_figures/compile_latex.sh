#!/bin/bash
set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo -e "${RED}Error: pdflatex not found. Please install TeX Live.${NC}"
    exit 1
fi

# 支持的表格文件
if [ $# -eq 0 ]; then
    echo "Available LaTeX table files:"
    for f in *.tex; do
        echo "  - $f"
    done
    echo ""
    echo "Usage: ./compile_latex.sh <tex_file>"
    echo "Example: ./compile_latex.sh efficiency_table.tex"
    exit 0
fi

INPUT_FILE="$1"
FILE_BASENAME="${INPUT_FILE%.tex}"

# 创建临时 compile 目录
BUILD_DIR="/tmp/latentdraft_compile_$$"
mkdir -p "$BUILD_DIR"

echo -e "${BLUE}Compiling ${INPUT_FILE}...${NC}"

# 复制所需文件到 build 目录
cp "$SCRIPT_DIR/${INPUT_FILE}" "$BUILD_DIR/"

# 如果有 main_results_table.tex 或 其他依赖，一起复制
for dep_tex in main_results_table.tex efficiency_table.tex; do
    if [ -f "$SCRIPT_DIR/$dep_tex" ] && [ "$dep_tex" != "$INPUT_FILE" ]; then
        cp "$SCRIPT_DIR/$dep_tex" "$BUILD_DIR/"
    fi
done

# 创建最小 tex wrapper（standalone 类用于单独编译表格）
cat > "$BUILD_DIR/${FILE_BASENAME}_standalone.tex" << 'EOF'
\documentclass[border=5pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{colortbl}
\usepackage{array}
\usepackage{graphicx}

% 颜色定义（与表格内一致，防止 standalone 环境中缺失）
\definecolor{lightblue}{RGB}{223,234,242}
\definecolor{lightpink}{RGB}{255,230,235}
\definecolor{categorygray}{RGB}{235,235,235}

\begin{document}
\input{EOF
cat >> "$BUILD_DIR/${FILE_BASENAME}_standalone.tex" << EOF
${FILE_BASENAME}}
\\end{document}
EOF

cd "$BUILD_DIR"

# 编译两次以解析交叉引用
RUN_COUNT=0
MAX_RUNS=2

while [ $RUN_COUNT -lt $MAX_RUNS ]; do
    if ! pdflatex -interaction=nonstopmode -halt-on-error "${FILE_BASENAME}_standalone.tex" > compile.log 2>&1; then
        echo -e "${RED}Compilation failed. Log:${NC}"
        cat compile.log | tail -n 30
        rm -rf "$BUILD_DIR"
        exit 1
    fi
    RUN_COUNT=$((RUN_COUNT + 1))
done

# 复制结果回到原目录
cp "$BUILD_DIR/${FILE_BASENAME}_standalone.pdf" "$SCRIPT_DIR/${FILE_BASENAME}.pdf"

# 转换为 PNG（可选，需安装 ImageMagick）
if command -v convert &> /dev/null; then
    convert -density 300 "$BUILD_DIR/${FILE_BASENAME}_standalone.pdf" \
            -background white -flatten -trim \
            +repage "$SCRIPT_DIR/${FILE_BASENAME}.png" 2>/dev/null || echo "PNG conversion skipped (ImageMagick issue)"
fi

# 清理
rm -rf "$BUILD_DIR"

echo -e "${GREEN}Success!${NC}"
echo -e "  PDF: ${SCRIPT_DIR}/${FILE_BASENAME}.pdf"
if [ -f "$SCRIPT_DIR/${FILE_BASENAME}.png" ]; then
    echo -e "  PNG: ${SCRIPT_DIR}/${FILE_BASENAME}.png"
fi
