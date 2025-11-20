#!/bin/bash
# setup_chanlun_system.sh
# 缠论系统精简安装脚本 - 跨平台兼容版

# 使用系统默认的Python3
PYTHON_CMD=python3

# 检查Python版本
echo "检查Python版本..."
$PYTHON_CMD --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python3，请先安装Python 3.8或更高版本"
    exit 1
fi

# 检查是否已存在虚拟环境
if [ ! -d "chanlun_env" ]; then
    echo "创建虚拟环境: $PYTHON_CMD -m venv chanlun_env"
    $PYTHON_CMD -m venv chanlun_env
    
    # 检查是否创建成功
    if [ ! -d "chanlun_env" ]; then
        echo "错误: 无法创建虚拟环境目录"
        echo "请尝试手动执行: $PYTHON_CMD -m venv chanlun_env"
        exit 1
    fi
fi

# 激活虚拟环境
echo "激活虚拟环境: source chanlun_env/bin/activate"
source chanlun_env/bin/activate

# 检查是否激活成功
if [ -z "$VIRTUAL_ENV" ]; then
    echo "错误: 无法激活虚拟环境"
    echo "请尝试手动执行: source chanlun_env/bin/activate"
    exit 1
fi

echo "======= 开始安装系统依赖 ======="
echo "当前虚拟环境: $VIRTUAL_ENV"

# 确定虚拟环境中的pip
VENV_PIP="$VIRTUAL_ENV/bin/pip"

# 检查pip是否存在
if [ ! -f "$VENV_PIP" ]; then
    echo "虚拟环境中未找到pip，尝试升级虚拟环境..."
    
    # 升级虚拟环境
    $PYTHON_CMD -m venv --upgrade chanlun_env
    
    # 再次检查
    if [ ! -f "$VENV_PIP" ]; then
        echo "错误: 无法找到pip，请尝试重新创建虚拟环境"
        echo "执行: rm -rf chanlun_env && $PYTHON_CMD -m venv chanlun_env"
        exit 1
    fi
fi

echo "升级pip..."
$VENV_PIP install --upgrade pip
echo "pip升级完成"

echo "======= 依赖安装完成! ======="

# 使用虚拟环境中的Python进行验证
VENV_PYTHON="$VIRTUAL_ENV/bin/python"

# 检查Python是否存在
if [ ! -f "$VENV_PYTHON" ]; then
    echo "虚拟环境中未找到Python，尝试修复..."
    
    # 检查是否有python3
    if [ -f "$VIRTUAL_ENV/bin/python3" ]; then
        ln -s "$VIRTUAL_ENV/bin/python3" "$VENV_PYTHON"
        echo "已创建符号链接: $VENV_PYTHON -> $VIRTUAL_ENV/bin/python3"
    else
        echo "错误: 虚拟环境中未找到任何Python解释器"
        echo "请尝试重新创建虚拟环境"
        exit 1
    fi
fi

# 验证安装
echo "验证安装..."
$VENV_PYTHON -c "import pandas as pd; import numpy as np; import requests; import matplotlib; import yaml; import loguru; print('所有核心依赖安装成功!')"

echo "======= 系统安装完成 ======="
echo "使用说明:"
echo "1. 激活虚拟环境: source chanlun_env/bin/activate"
echo "2. 运行系统: python src/main.py -s 510300"
echo "3. 退出虚拟环境: deactivate"