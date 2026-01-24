#!/bin/bash
# Install dependencies for GRPO training
# Usage: ./install-dependencies.sh [venv_path]
# Default venv path: /mnt/task_runtime/myenv
#
# This script installs packages in a specific order to avoid dependency conflicts
# between vllm (needs opentelemetry-sdk<1.27) and ray[default] (needs opentelemetry-sdk>=1.30)
# Solution: Install verl with --no-deps, then install ray without default extras

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
    echo "[1/9] Creating virtual environment..."
    python3.12 -m venv "$VENV_PATH"
else
    echo "[1/9] Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
echo "[2/9] Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip, setuptools, wheel
echo "[3/9] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 first (special index)
echo "[4/9] Installing PyTorch with CUDA 12.4..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install opentelemetry packages FIRST with exact versions (vllm compatible)
echo "[5/9] Installing opentelemetry packages (pinned to 1.26.0 for vllm compatibility)..."
pip install \
    opentelemetry-api==1.26.0 \
    opentelemetry-sdk==1.26.0 \
    opentelemetry-proto==1.26.0 \
    opentelemetry-exporter-otlp==1.26.0 \
    opentelemetry-exporter-otlp-proto-grpc==1.26.0 \
    opentelemetry-exporter-otlp-proto-http==1.26.0 \
    opentelemetry-exporter-otlp-proto-common==1.26.0 \
    opentelemetry-semantic-conventions==0.47b0

# Install ray WITHOUT extras to avoid pulling in ray[default] which needs opentelemetry-sdk>=1.30.0
# ray[default] adds observability features which we don't need
echo "[6/9] Installing ray (without default extras)..."
pip install ray==2.53.0

# Install verl with --no-deps because it requires ray[default] which conflicts with vllm
# Then we install verl's actual dependencies separately
echo "[7/9] Installing verl (without dependencies) and its deps..."
pip install verl==0.4.1 --no-deps
# Install verl dependencies (except ray[default], we already have ray)
pip install \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    numpy \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    torchdata \
    "tensordict<=0.6.2" \
    transformers \
    wandb \
    "packaging>=20.0"

# Install flash-attn (needs special handling - requires CUDA toolkit)
echo "[8/9] Installing flash-attn..."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
pip install flash-attn==2.7.4.post1 --no-build-isolation 2>/dev/null || {
    echo "[WARNING] flash-attn 2.7.4.post1 installation failed. Trying without version pin..."
    pip install flash-attn --no-build-isolation 2>/dev/null || {
        echo "[WARNING] flash-attn could not be installed. Training may still work without it."
    }
}

# Create constraints file to prevent torch/opentelemetry from being changed
CONSTRAINTS_FILE=$(mktemp)
cat > "$CONSTRAINTS_FILE" << EOF
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
opentelemetry-api==1.26.0
opentelemetry-sdk==1.26.0
opentelemetry-proto==1.26.0
opentelemetry-exporter-otlp==1.26.0
opentelemetry-exporter-otlp-proto-grpc==1.26.0
opentelemetry-exporter-otlp-proto-http==1.26.0
opentelemetry-exporter-otlp-proto-common==1.26.0
opentelemetry-semantic-conventions==0.47b0
ray==2.53.0
EOF

# Install remaining packages with constraints
echo "[9/9] Installing remaining dependencies (vllm, sglang, etc.)..."
pip install -c "$CONSTRAINTS_FILE" -r "$SCRIPT_DIR/requirements-core.txt"

# Cleanup constraints file
rm -f "$CONSTRAINTS_FILE"

# Verify key packages
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import verl; print(f'verl: {verl.__version__}')"
pip show opentelemetry-sdk 2>/dev/null | grep Version || echo "OpenTelemetry SDK: installed"

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
