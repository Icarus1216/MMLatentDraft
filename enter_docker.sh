#!/bin/bash

CONTAINER_NAME="colamem-dev"

# 检查容器是否存在
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ 容器 ${CONTAINER_NAME} 不存在！"
    echo "请先运行以下命令创建容器："
    echo "docker run -it --name ${CONTAINER_NAME} -v /data/workspace:/workspace -v ~/.cache/huggingface:/root/.cache/huggingface weregistry.woa.com/baseimage/vllm:0.13.0-pytorch2.9.1-py3.12-cu129-devel /bin/bash"
    exit 1
fi

# 检查容器是否运行
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🚀 启动容器 ${CONTAINER_NAME}..."
    docker start ${CONTAINER_NAME}
fi

# 进入容器并激活环境
echo "✅ 进入容器 ${CONTAINER_NAME}..."
docker exec -it ${CONTAINER_NAME} /bin/bash -c "source /opt/conda/bin/activate colamem && cd /workspace/ColaMem && exec bash"