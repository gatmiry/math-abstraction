#!/bin/bash
# Complete setup script for any node (head or worker)
# Run this after creating and activating a new virtualenv:
#   python3 -m venv verl_env && source verl_env/bin/activate && bash install-node.sh

set -e

# Force unbuffered output
export PYTHONUNBUFFERED=1

echo "========================================="
echo "=== verl Training Node Setup ==="
echo "========================================="
echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo ""

echo "=== Step 1: Installing build tools (for flash_attn) ==="
pip install --no-cache-dir ninja packaging wheel setuptools
if [ $? -ne 0 ]; then echo "ERROR: build tools install failed"; exit 1; fi

echo "=== Step 2: Installing PyTorch with CUDA 12.4 ==="
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if [ $? -ne 0 ]; then echo "ERROR: PyTorch install failed"; exit 1; fi

echo "=== Step 3: Installing numpy ==="
pip install --no-cache-dir numpy==1.26.4
if [ $? -ne 0 ]; then echo "ERROR: numpy install failed"; exit 1; fi

echo "=== Step 4: Installing Ray ==="
pip install --no-cache-dir ray==2.53.0
if [ $? -ne 0 ]; then echo "ERROR: ray install failed"; exit 1; fi

echo "=== Step 5: Installing vLLM ==="
pip install --no-cache-dir vllm==0.8.5.post1
if [ $? -ne 0 ]; then echo "ERROR: vllm install failed"; exit 1; fi

echo "=== Step 6: Installing verl ==="
pip install --no-cache-dir verl==0.4.1
if [ $? -ne 0 ]; then echo "ERROR: verl install failed"; exit 1; fi

echo "=== Step 7: Installing trl ==="
pip install --no-cache-dir trl==0.26.2
if [ $? -ne 0 ]; then echo "ERROR: trl install failed"; exit 1; fi

echo "=== Step 8: Installing transformers ecosystem ==="
pip install --no-cache-dir transformers==4.57.3 accelerate==1.12.0 peft==0.18.0 datasets==4.4.2 tokenizers==0.22.1 safetensors==0.7.0
if [ $? -ne 0 ]; then echo "ERROR: transformers ecosystem install failed"; exit 1; fi

echo "=== Step 9: Building flash_attn from source (this takes 10-20 minutes) ==="
echo "Building with: CUDA $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'unknown')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation
if [ $? -ne 0 ]; then 
    echo "ERROR: flash_attn build failed"
    echo "If build fails, you can skip flash_attn and use eager attention instead:"
    echo "  export TRANSFORMERS_ATTN_IMPLEMENTATION=eager"
    exit 1
fi

echo "=== Step 10: Installing flashinfer ==="
pip install --no-cache-dir flashinfer-python==0.5.2
if [ $? -ne 0 ]; then echo "WARNING: flashinfer install failed (may be OK)"; fi

echo "=== Step 11: Installing config & data processing ==="
pip install --no-cache-dir pandas==2.3.3 scipy omegaconf==2.3.0 hydra-core sentencepiece==0.2.1 protobuf==4.25.8 regex
if [ $? -ne 0 ]; then echo "ERROR: config/data dependencies install failed"; exit 1; fi

echo "=== Step 12: Installing other utilities ==="
pip install --no-cache-dir bitsandbytes==0.49.0 wandb
if [ $? -ne 0 ]; then echo "WARNING: utilities install failed"; fi

echo "=== Step 13: Installing apple_bolt ==="
pip install --no-cache-dir apple_bolt
if [ $? -ne 0 ]; then echo "WARNING: apple_bolt install failed"; fi

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
