#!/usr/bin/env bash
# torch2c 一键式环境部署脚本（目标：Linux / macOS）
# 用法: bash scripts/setup_env.sh
set -euo pipefail

PYTHON_VERSION="3.10"
ENV_DIR=".venv"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== torch2c 环境部署 ==="

# ---- 1. 检测/安装 miniconda ----
if command -v conda &>/dev/null; then
    echo "[OK] conda 已安装: $(conda --version)"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    echo "[OK] 发现 miniconda3，初始化 shell..."
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
else
    echo "[INFO] 未检测到 conda，正在安装 Miniconda..."
    ARCH="$(uname -m)"
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
    elif [ "$OS" = "Darwin" ]; then
        INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-${ARCH}.sh"
    else
        echo "[ERROR] 不支持的操作系统: $OS" >&2
        exit 1
    fi
    INSTALLER="/tmp/miniconda_installer.sh"
    echo "  下载: $INSTALLER_URL"
    curl -fsSL "$INSTALLER_URL" -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$HOME/miniconda3"
    rm -f "$INSTALLER"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash --quiet 2>/dev/null || true
    echo "[OK] Miniconda 安装完成"
fi

# ---- 2. 接受 TOS（非交互模式需要） ----
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ---- 3. 创建 conda 环境 ----
if [ -d "$ENV_DIR" ] && "$ENV_DIR/bin/python" --version 2>/dev/null | grep -q "3\.10"; then
    echo "[OK] 环境已存在: $ENV_DIR (Python $PYTHON_VERSION)"
else
    echo "[INFO] 创建 conda 环境: $ENV_DIR (Python $PYTHON_VERSION)..."
    rm -rf "$ENV_DIR"
    conda create -y -p "$ENV_DIR" python="$PYTHON_VERSION"
    echo "[OK] 环境创建完成"
fi

# ---- 4. 激活并安装依赖 ----
eval "$(conda shell.bash hook)"
conda activate "$PROJECT_ROOT/$ENV_DIR"

echo "[INFO] 安装项目依赖..."
pip install --upgrade pip setuptools wheel -q
pip install -e ".[dev]" -q

echo ""
echo "=== 部署完成 ==="
echo "Python: $(python --version)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "激活环境:"
echo "  conda activate $PROJECT_ROOT/$ENV_DIR"
echo ""
echo "运行测试:"
echo "  pytest torch2c/ -v"
