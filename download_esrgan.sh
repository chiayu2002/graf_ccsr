#!/bin/bash

# ESRGAN 预训练模型下载脚本

echo "================================================"
echo "  RRDB_ESRGAN_x4 预训练模型下载脚本"
echo "================================================"
echo ""

# 创建目录
echo "创建 pretrained_models 目录..."
mkdir -p pretrained_models

# 模型文件路径
MODEL_PATH="pretrained_models/RRDB_ESRGAN_x4.pth"

# 检查文件是否已存在
if [ -f "$MODEL_PATH" ]; then
    echo "✓ 模型文件已存在: $MODEL_PATH"
    echo "  文件大小: $(du -h $MODEL_PATH | cut -f1)"
    read -p "是否重新下载？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过下载。"
        exit 0
    fi
fi

echo ""
echo "开始下载 RRDB_ESRGAN_x4 模型..."
echo ""

# 尝试方法 1: wget
echo "方法 1: 使用 wget 从 GitHub 下载..."
if command -v wget &> /dev/null; then
    wget https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_ESRGAN_x4.pth \
         -O "$MODEL_PATH"

    if [ $? -eq 0 ]; then
        echo "✓ 下载成功！"
        echo "  文件路径: $MODEL_PATH"
        echo "  文件大小: $(du -h $MODEL_PATH | cut -f1)"
        exit 0
    else
        echo "✗ wget 下载失败"
        rm -f "$MODEL_PATH"
    fi
else
    echo "✗ wget 未安装"
fi

# 尝试方法 2: curl
echo ""
echo "方法 2: 使用 curl 下载..."
if command -v curl &> /dev/null; then
    curl -L https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_ESRGAN_x4.pth \
         -o "$MODEL_PATH"

    if [ $? -eq 0 ]; then
        echo "✓ 下载成功！"
        echo "  文件路径: $MODEL_PATH"
        echo "  文件大小: $(du -h $MODEL_PATH | cut -f1)"
        exit 0
    else
        echo "✗ curl 下载失败"
        rm -f "$MODEL_PATH"
    fi
else
    echo "✗ curl 未安装"
fi

# 尝试方法 3: gdown
echo ""
echo "方法 3: 使用 gdown 下载..."
if command -v gdown &> /dev/null; then
    # Google Drive ID (如果有的话)
    gdown --id 1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene -O "$MODEL_PATH"

    if [ $? -eq 0 ]; then
        echo "✓ 下载成功！"
        echo "  文件路径: $MODEL_PATH"
        echo "  文件大小: $(du -h $MODEL_PATH | cut -f1)"
        exit 0
    else
        echo "✗ gdown 下载失败"
        rm -f "$MODEL_PATH"
    fi
else
    echo "✗ gdown 未安装"
    echo "  提示: 可以使用 'pip install gdown' 安装"
fi

# 所有方法都失败
echo ""
echo "================================================"
echo "  ❌ 自动下载失败"
echo "================================================"
echo ""
echo "请手动下载模型："
echo "1. 访问: https://github.com/xinntao/ESRGAN/releases"
echo "2. 下载: RRDB_ESRGAN_x4.pth"
echo "3. 放置到: $MODEL_PATH"
echo ""
echo "或者尝试:"
echo "  pip install gdown"
echo "  bash download_esrgan.sh"
echo ""

exit 1
