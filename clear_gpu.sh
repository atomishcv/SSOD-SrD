#!/bin/bash

# ============================================
# 🧹 清理 GPU 显存脚本
# ============================================

echo "========================================"
echo "🔍 当前 GPU 使用情况："
echo "========================================"
nvidia-smi

echo ""
echo "========================================"
echo "🧹 查找并清理 Python 进程..."
echo "========================================"

# 查找所有 Python 进程
PYTHON_PIDS=$(ps aux | grep python | grep -v grep | awk '{print $2}')

if [ -z "$PYTHON_PIDS" ]; then
    echo "✅ 没有发现 Python 进程"
else
    echo "找到以下 Python 进程："
    ps aux | grep python | grep -v grep
    echo ""
    echo "是否要杀掉这些进程？(y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "$PYTHON_PIDS" | xargs kill -9 2>/dev/null
        echo "✅ Python 进程已清理"
    else
        echo "⏭️  跳过清理"
    fi
fi

echo ""
echo "========================================"
echo "🔄 清理后的 GPU 状态："
echo "========================================"
nvidia-smi

echo ""
echo "✅ 完成！"


