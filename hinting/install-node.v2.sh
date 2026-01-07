#!/bin/bash
# Setup script using pre-built flash_attn tarball (faster, ~5 mins)
# Use this when you have a compatible tarball from another node with same CUDA/PyTorch
#
# Prerequisites:
#   1. Copy flash_attn tarball from a working node:
#      scp working_node:/mnt/task_runtime/hinting/wheels/flash_attn-2.8.3-py312-cu124.tar.gz ./wheels/
#   2. Create and activate virtualenv:
#      python3 -m venv verl_env && source verl_env/bin/activate && bash install-node.v2.sh

set -e

# Force unbuffered output
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_TARBALL="${SCRIPT_DIR}/wheels/flash_attn-2.8.3-py312-cu124.tar.gz"

echo "========================================="
echo "=== verl Training Node Setup (v2 - tarball) ==="
echo "========================================="
echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo ""

# Check tarball exists
if [ ! -f "$FLASH_ATTN_TARBALL" ]; then
    echo "ERROR: flash_attn tarball not found at: $FLASH_ATTN_TARBALL"
    echo ""
    echo "Copy it from a working node first:"
    echo "  mkdir -p ${SCRIPT_DIR}/wheels"
    echo "  scp working_node:/mnt/task_runtime/hinting/wheels/flash_attn-2.8.3-py312-cu124.tar.gz ${SCRIPT_DIR}/wheels/"
    echo ""
    echo "Or use install-node.sh to build from source instead."
    exit 1
fi

echo "=== Step 1: Installing PyTorch with CUDA 12.4 ==="
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if [ $? -ne 0 ]; then echo "ERROR: PyTorch install failed"; exit 1; fi

echo "=== Step 2: Installing numpy ==="
pip install --no-cache-dir numpy==1.26.4
if [ $? -ne 0 ]; then echo "ERROR: numpy install failed"; exit 1; fi

echo "=== Step 3: Installing Ray ==="
pip install --no-cache-dir ray==2.53.0
if [ $? -ne 0 ]; then echo "ERROR: ray install failed"; exit 1; fi

echo "=== Step 4: Installing vLLM ==="
pip install --no-cache-dir vllm==0.8.5.post1
if [ $? -ne 0 ]; then echo "ERROR: vllm install failed"; exit 1; fi

echo "=== Step 5: Installing verl ==="
pip install --no-cache-dir verl==0.4.1
if [ $? -ne 0 ]; then echo "ERROR: verl install failed"; exit 1; fi

echo "=== Step 6: Installing trl ==="
pip install --no-cache-dir trl==0.26.2
if [ $? -ne 0 ]; then echo "ERROR: trl install failed"; exit 1; fi

echo "=== Step 7: Installing transformers ecosystem ==="
pip install --no-cache-dir transformers==4.57.3 accelerate==1.12.0 peft==0.18.0 datasets==4.4.2 tokenizers==0.22.1 safetensors==0.7.0
if [ $? -ne 0 ]; then echo "ERROR: transformers ecosystem install failed"; exit 1; fi

echo "=== Step 8: Installing flash_attn from tarball ==="
echo "Using pre-built flash_attn from: $FLASH_ATTN_TARBALL"
pip uninstall flash_attn flash-attn -y 2>/dev/null || true
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cd "$SITE_PACKAGES"
tar xzf "$FLASH_ATTN_TARBALL"
if [ $? -ne 0 ]; then echo "ERROR: flash_attn tarball extraction failed"; exit 1; fi
echo "flash_attn installed from tarball"
cd "$SCRIPT_DIR"

echo "=== Step 9: Installing flashinfer ==="
pip install --no-cache-dir flashinfer-python==0.5.2
if [ $? -ne 0 ]; then echo "WARNING: flashinfer install failed (may be OK)"; fi

echo "=== Step 10: Installing config & data processing ==="
pip install --no-cache-dir pandas==2.3.3 scipy omegaconf==2.3.0 hydra-core sentencepiece==0.2.1 protobuf==4.25.8 regex
if [ $? -ne 0 ]; then echo "ERROR: config/data dependencies install failed"; exit 1; fi

echo "=== Step 11: Installing other utilities ==="
pip install --no-cache-dir bitsandbytes==0.49.0 wandb ninja packaging
if [ $? -ne 0 ]; then echo "WARNING: utilities install failed"; fi

echo ""
echo "========================================="
echo "=== Verifying installation ==="
echo "========================================="
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import verl; print(f'verl: {verl.__version__}')"
python -c "import ray; print(f'ray: {ray.__version__}')"

echo ""
echo "========================================="
echo "=== Installation complete! ==="
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "For HEAD node:"
echo "  export GLOO_SOCKET_IFNAME=eth0"
echo "  export NCCL_SOCKET_IFNAME=eth0"
echo "  export RAY_DISABLE_DASHBOARD=1"
echo "  ray start --head --port=6379 --num-gpus=8"
echo ""
echo "For WORKER node:"
echo "  export GLOO_SOCKET_IFNAME=eth0"
echo "  export NCCL_SOCKET_IFNAME=eth0"
echo "  export RAY_DISABLE_DASHBOARD=1"
echo "  ray start --address='<HEAD_IP>:6379' --num-gpus=8"
echo ""

