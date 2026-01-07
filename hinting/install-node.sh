#!/bin/bash
# Install script for worker node - installs packages in correct order
# Run this after creating and activating a new virtualenv:
#   python3 -m venv myenv && source myenv/bin/activate && bash install-node.sh

# Force unbuffered output
export PYTHONUNBUFFERED=1

echo "Starting installation..."
echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo ""

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

echo "=== Step 8: Installing flash attention from pre-built tarball ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_TARBALL="${SCRIPT_DIR}/wheels/flash_attn-2.8.3-py312-cu124.tar.gz"

if [ -f "$FLASH_ATTN_TARBALL" ]; then
    echo "Using pre-built flash_attn from: $FLASH_ATTN_TARBALL"
    pip uninstall flash_attn flash-attn -y 2>/dev/null || true
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    cd "$SITE_PACKAGES"
    tar xzf "$FLASH_ATTN_TARBALL"
    echo "flash_attn installed from tarball"
else
    echo "ERROR: flash_attn tarball not found at: $FLASH_ATTN_TARBALL"
    echo "Copy it from the head node first."
    exit 1
fi

echo "=== Step 9: OpenTelemetry already installed by vllm ==="
echo "Note: vllm and ray have conflicting opentelemetry requirements."
echo "This is OK - we use RAY_DISABLE_DASHBOARD=1 at runtime to avoid the conflict."

echo "=== Step 10: Installing other dependencies ==="
pip install --no-cache-dir pandas==2.3.3 scipy omegaconf==2.3.0 hydra-core sentencepiece==0.2.1 protobuf==4.25.8 bitsandbytes==0.49.0
if [ $? -ne 0 ]; then echo "ERROR: other dependencies install failed"; exit 1; fi

echo ""
echo "========================================="
echo "=== Installation complete! ==="
echo "========================================="
echo ""
echo "To join Ray cluster, run:"
echo "  RAY_DISABLE_DASHBOARD=1 ray start --address='<HEAD_IP>:6379' --num-gpus=8"
