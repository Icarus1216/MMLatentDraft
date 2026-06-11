#!/bin/bash
# VSI-Bench direction + route_planning 训练数据生成
# 1162 条, 每条 8 帧多图, v6 格式
#
# 需要先设置 OpenAI 兼容服务的环境变量:
#   export OPENAI_API_KEY="your_api_key"
#   export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选

set -e

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "❌ 请先 export OPENAI_API_KEY"
    exit 1
fi

export PYTHONUNBUFFERED=1

python3 -u scripts/generate_vsibench_latent_cot.py \
    --data_root ./data/VSI-Bench_eval \
    --data_file ./data/VSI-Bench_eval/vsibench_test.jsonl \
    --output ./data/vsibench/vsibench_direction_route_latent_cot.json \
    --model claude-opus-4-7 \
    --workers 6 \
    --max_frames 8 \
    --temperature 0.7 \
    --num_retries_per_sample 3 \
    --resume \
    --verbose
