#!/bin/bash
# setup_chanlun_system.sh
# 缠论系统启动脚本 - 沙盒模式

# 启用沙盒模式标志
export CHANLUN_SANDBOX_MODE=true

# 提示信息
echo "======= 缠论系统启动脚本 ======="
echo "沙盒模式已启用"
echo ""

# 检查是否已存在虚拟环境
if [ ! -d "chanlun_env" ]; then
    echo "提示: 未找到虚拟环境 'chanlun_env'"
    echo "请手动创建虚拟环境并安装依赖:"
    echo "1. python3 -m venv chanlun_env"
    echo "2. source chanlun_env/bin/activate"
    echo "3. pip install -r requirements.txt"
    echo ""
    return 1 2>/dev/null || exit 1
fi

# 检查虚拟环境中的Python是否存在
VENV_PYTHON="chanlun_env/bin/python"
if [ ! -f "$VENV_PYTHON" ] && [ ! -f "chanlun_env/bin/python3" ]; then
    echo "错误: 虚拟环境中未找到Python解释器"
    echo "请重新创建虚拟环境"
    return 1 2>/dev/null || exit 1
fi

# 激活虚拟环境
echo "正在激活虚拟环境..."
if command -v source >/dev/null; then
    source chanlun_env/bin/activate
else
    . chanlun_env/bin/activate
fi

# 检查是否激活成功
if [ -z "$VIRTUAL_ENV" ]; then
    echo "警告: 虚拟环境可能未正确激活"
    echo "请尝试手动执行: source chanlun_env/bin/activate"
    return 1 2>/dev/null || exit 1
fi

echo "虚拟环境激活成功: $VIRTUAL_ENV"
echo "沙盒模式: $CHANLUN_SANDBOX_MODE"
echo ""
echo "使用说明:"
echo "1. 运行系统: python src/main.py [参数]"
echo "2. 退出虚拟环境: deactivate"
echo ""
echo "提示: 所有依赖已移至requirements.txt，请手动安装"
echo "执行: pip install -r requirements.txt"