#!/bin/bash
# Install dependencies for GRPO training
# Usage: ./install-dependencies.sh [venv_path]
# Default venv path: /mnt/task_runtime/myenv

set -e  # Exit on error

VENV_PATH="${1:-/mnt/task_runtime/myenv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "GRPO Training Environment Setup"
echo "=============================================="
echo "Virtual environment path: $VENV_PATH"
echo ""

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "[ERROR] Python 3.12 is required but not found."
    echo "Please install Python 3.12 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "[1/5] Creating virtual environment..."
    python3.12 -m venv "$VENV_PATH"
else
    echo "[1/5] Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
echo "[2/5] Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "[3/5] Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.4 first (special index)
echo "[4/5] Installing PyTorch with CUDA 12.4..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn (needs special handling)
echo "[5/5] Installing remaining dependencies..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install main requirements
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To start Ray worker (connect to head node):"
echo "  source $VENV_PATH/bin/activate"
echo "  ray start --address=<HEAD_NODE_IP>:6379"
echo ""

