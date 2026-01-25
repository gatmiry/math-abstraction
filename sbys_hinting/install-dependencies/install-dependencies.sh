#!/bin/bash
# Install dependencies for GRPO training on B200 GPUs
# Usage: ./install-dependencies.sh [venv_path]
# Default venv path: /mnt/task_runtime/myenv
#
# This script installs packages in a specific order to avoid dependency conflicts.
# Supports B200 GPUs (sm_100) with PyTorch 2.9.1+cu128.
#
# IMPORTANT: After running this script, you also need to:
# 1. Install sbys_hinting as a package: cd /mnt/task_runtime && pip install -e .
# 2. Apply runtime patches (done automatically by this script)

set -e  # Exit on error

VENV_PATH="${1:-/mnt/task_runtime/myenv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=============================================="
echo "GRPO Training Environment Setup (B200 GPU)"
echo "=============================================="
echo "Virtual environment path: $VENV_PATH"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "[ERROR] Python 3.12 is required but not found."
    echo "Please install Python 3.12 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "[1/12] Creating virtual environment..."
    python3.12 -m venv "$VENV_PATH"
else
    echo "[1/12] Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
echo "[2/12] Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip, setuptools, wheel
echo "[3/12] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 (supports B200 GPUs with sm_100)
echo "[4/12] Installing PyTorch 2.9.1 with CUDA 12.8 (B200 GPU support)..."
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install triton 3.5.1 (required for B200 sm_100 support in ptxas)
# Note: sglang 0.4.6.post5 expects triton 3.1.0 API but we'll patch it
echo "[5/12] Installing triton 3.5.1 (B200 GPU support)..."
pip install triton==3.5.1

# Install xformers compatible with PyTorch 2.9.1
echo "[6/12] Installing xformers..."
pip install xformers==0.0.33.post2

# Install vllm (latest version compatible with PyTorch 2.9.1)
echo "[7/12] Installing vllm..."
pip install vllm

# Install sglang 0.4.6.post5 (compatible with verl API) and sgl-kernel
echo "[8/12] Installing sglang and dependencies..."
pip install sglang==0.4.6.post5 decord sgl-kernel torch-memory-saver torchao

# Install verl with --no-deps to avoid ray[default] conflicts
echo "[9/12] Installing verl and its dependencies..."
pip install verl==0.4.1 --no-deps
# Install verl dependencies (except ray[default], we already have ray from vllm)
pip install \
    accelerate \
    apple-bolt \
    codetiming \
    datasets \
    dill \
    hydra-core \
    math-verify \
    numpy \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    torchdata \
    tensordict \
    transformers \
    wandb \
    "packaging>=20.0"

# Install flash-attn (needs special handling - requires CUDA toolkit and build dependencies)
echo "[10/12] Installing flash-attn..."
# Install build dependencies first
pip install ninja packaging wheel

# Detect CUDA_HOME if not set
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_HOME=/usr/local/cuda-12.8
    elif [ -d "/usr/local/cuda-12" ]; then
        export CUDA_HOME=/usr/local/cuda-12
    else
        # Try to find CUDA via nvcc
        CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo ""))) 2>/dev/null || true
        if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
            echo "[WARNING] CUDA_HOME not found. Setting to /usr/local/cuda (may need manual adjustment)"
            export CUDA_HOME=/usr/local/cuda
        else
            export CUDA_HOME
        fi
    fi
fi

echo "Using CUDA_HOME: $CUDA_HOME"

# Set up CUDA environment variables for build
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

LOG_FILE="/tmp/flash_attn_install.log"

# Install flash-attn with proper error handling
echo "Attempting to install flash-attn..."
if pip install flash-attn --no-build-isolation --no-cache-dir > "$LOG_FILE" 2>&1; then
    echo "[SUCCESS] flash-attn installed successfully"
else
    # If pre-built wheel fails, try building from source
    echo "[INFO] Pre-built wheel not available, attempting to build from source..."
    echo "[INFO] This may take several minutes..."
    
    # Try building from source
    echo "Building flash-attn from source (this may take 5-10 minutes)..."
    if MAX_JOBS=$(nproc) pip install flash-attn --no-build-isolation --no-cache-dir >> "$LOG_FILE" 2>&1; then
        echo "[SUCCESS] flash-attn built and installed from source"
    else
        echo "[ERROR] flash-attn installation failed. Last 50 lines of log:"
        tail -n 50 "$LOG_FILE"
        echo ""
        echo "[ERROR] Full log available at: $LOG_FILE"
        echo "[ERROR] This is a required dependency for verl. Installation cannot continue."
        exit 1
    fi
fi

# Verify flash-attn installation
if python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')" 2>/dev/null; then
    echo "[VERIFIED] flash-attn is properly installed"
else
    echo "[ERROR] flash-attn installation verification failed"
    echo "[ERROR] Check log at: $LOG_FILE"
    exit 1
fi

# Apply patches for B200 GPU compatibility
echo "[11/12] Applying patches for B200 GPU compatibility..."

# Patch 1: sglang utils.py - fix triton 3.5.x compatibility (default_cache_dir removed)
SGLANG_UTILS="$VENV_PATH/lib/python3.12/site-packages/sglang/srt/utils.py"
if [ -f "$SGLANG_UTILS" ]; then
    python << 'PATCH_UTILS'
filepath = '/mnt/task_runtime/myenv/lib/python3.12/site-packages/sglang/srt/utils.py'
with open(filepath, 'r') as f:
    content = f.read()

old_import = '''from triton.runtime.cache import (
    FileCacheManager,
    default_cache_dir,
    default_dump_dir,
    default_override_dir,
)'''

new_import = '''from triton.runtime.cache import FileCacheManager
try:
    from triton.runtime.cache import (
        default_cache_dir,
        default_dump_dir,
        default_override_dir,
    )
except ImportError:
    # triton 3.5.x compatibility - these functions were removed
    import os
    def default_cache_dir():
        return os.path.expanduser("~/.triton/cache")
    def default_dump_dir():
        return os.path.expanduser("~/.triton/dump")
    def default_override_dir():
        return os.path.expanduser("~/.triton/override")'''

if old_import in content:
    content = content.replace(old_import, new_import)
    with open(filepath, 'w') as f:
        f.write(content)
    print('[PATCHED] sglang/srt/utils.py for triton 3.5.x compatibility')
else:
    print('[SKIP] sglang/srt/utils.py already patched or pattern not found')
PATCH_UTILS
fi

# Patch 2: sglang awq.py - handle missing fused_marlin_moe
SGLANG_AWQ="$VENV_PATH/lib/python3.12/site-packages/sglang/srt/layers/quantization/awq.py"
if [ -f "$SGLANG_AWQ" ]; then
    python << 'PATCH_AWQ'
filepath = '/mnt/task_runtime/myenv/lib/python3.12/site-packages/sglang/srt/layers/quantization/awq.py'
with open(filepath, 'r') as f:
    content = f.read()

old_block = '''if _is_cuda:
    from sgl_kernel import awq_dequantize, fused_marlin_moe'''

new_block = '''if _is_cuda:
    from sgl_kernel import awq_dequantize
    try:
        from sgl_kernel import fused_marlin_moe
    except ImportError:
        fused_marlin_moe = None'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(filepath, 'w') as f:
        f.write(content)
    print('[PATCHED] sglang/srt/layers/quantization/awq.py for missing fused_marlin_moe')
else:
    print('[SKIP] awq.py already patched or pattern not found')
PATCH_AWQ
fi

# Patch 3: sglang gptq.py - handle missing fused_marlin_moe
SGLANG_GPTQ="$VENV_PATH/lib/python3.12/site-packages/sglang/srt/layers/quantization/gptq.py"
if [ -f "$SGLANG_GPTQ" ]; then
    python << 'PATCH_GPTQ'
import re
filepath = '/mnt/task_runtime/myenv/lib/python3.12/site-packages/sglang/srt/layers/quantization/gptq.py'
with open(filepath, 'r') as f:
    content = f.read()

pattern = r'^(\s*)from sgl_kernel import fused_marlin_moe$'
replacement = r'''\1try:
\1    from sgl_kernel import fused_marlin_moe
\1except ImportError:
\1    fused_marlin_moe = None'''

new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

if new_content != content:
    with open(filepath, 'w') as f:
        f.write(new_content)
    print('[PATCHED] sglang/srt/layers/quantization/gptq.py for missing fused_marlin_moe')
else:
    print('[SKIP] gptq.py already patched or pattern not found')
PATCH_GPTQ
fi

# Patch 4: verl fsdp_sglang.py - fix asyncio event loop for Python 3.12+ / uvloop
VERL_FSDP_SGLANG="$VENV_PATH/lib/python3.12/site-packages/verl/workers/sharding_manager/fsdp_sglang.py"
if [ -f "$VERL_FSDP_SGLANG" ]; then
    python << 'PATCH_FSDP_SGLANG'
filepath = '/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/workers/sharding_manager/fsdp_sglang.py'
with open(filepath, 'r') as f:
    content = f.read()

# Fix __enter__ method
old_enter = '''        with simple_timer("reshard", self.timing):
            loop = asyncio.get_event_loop()'''

new_enter = '''        with simple_timer("reshard", self.timing):
            # Python 3.12+ compatibility: always create/use a fresh event loop
            # Can't use get_running_loop() because run_until_complete() fails on running loops
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new loop if the current one is running
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)'''

# Fix __exit__ method
old_exit = '''    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        loop = asyncio.get_event_loop()'''

new_exit = '''    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        # Python 3.12+ compatibility: always create/use a fresh event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)'''

patched = False
if old_enter in content:
    content = content.replace(old_enter, new_enter)
    patched = True
if old_exit in content:
    content = content.replace(old_exit, new_exit)
    patched = True

if patched:
    with open(filepath, 'w') as f:
        f.write(content)
    print('[PATCHED] verl fsdp_sglang.py for Python 3.12+ / uvloop asyncio fix')
else:
    print('[SKIP] fsdp_sglang.py already patched or pattern not found')
PATCH_FSDP_SGLANG
fi

# Patch 5: verl sglang_rollout.py - disable CUDA graphs (system CUDA 12.4 doesn't support sm_100)
VERL_SGLANG="$VENV_PATH/lib/python3.12/site-packages/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
if [ -f "$VERL_SGLANG" ]; then
    python << 'PATCH_VERL'
filepath = '/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/workers/rollout/sglang_rollout/sglang_rollout.py'
with open(filepath, 'r') as f:
    content = f.read()

old_code = '''                mm_attention_backend="fa3",
            )'''

# Also match if attention_backend was already added but not disable_cuda_graph
old_code_alt = '''                mm_attention_backend="fa3",
                attention_backend="triton",  # Use triton instead of flashinfer (system CUDA doesn't support sm_100a)
            )'''

new_code = '''                mm_attention_backend="fa3",
                attention_backend="triton",  # Use triton instead of flashinfer (system CUDA doesn't support sm_100a)
                disable_cuda_graph=True,  # Disable CUDA graphs for B200 compatibility with system CUDA < 12.8
            )'''

if old_code in content and 'disable_cuda_graph=True' not in content:
    content = content.replace(old_code, new_code)
    with open(filepath, 'w') as f:
        f.write(content)
    print('[PATCHED] verl sglang_rollout.py to disable CUDA graphs')
else:
    print('[SKIP] sglang_rollout.py already patched or pattern not found')
PATCH_VERL
fi

# Patch 6: verl dp_actor.py - fix entropy calculation to use chunking config
VERL_DP_ACTOR="$VENV_PATH/lib/python3.12/site-packages/verl/workers/actor/dp_actor.py"
if [ -f "$VERL_DP_ACTOR" ]; then
    python << 'PATCH_DP_ACTOR'
filepath = '/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/workers/actor/dp_actor.py'
with open(filepath, 'r') as f:
    content = f.read()

# Fix _forward_micro_batch to use self.compute_entropy_from_logits (respects chunking config)
# instead of hardcoded verl_F.entropy_from_logits
old_code = '''                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)'''

new_code = '''                    if calculate_entropy:
                        entropy = self.compute_entropy_from_logits(logits)  # (bsz, response_length)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, 'w') as f:
        f.write(content)
    print('[PATCHED] verl dp_actor.py to use chunking config for entropy calculation')
else:
    print('[SKIP] dp_actor.py already patched or pattern not found')
PATCH_DP_ACTOR
fi

# Patch 7: verl torch_functional.py - fix entropy_from_logits_with_chunking for 3D tensors
VERL_TORCH_FUNC="$VENV_PATH/lib/python3.12/site-packages/verl/utils/torch_functional.py"
if [ -f "$VERL_TORCH_FUNC" ]; then
    python << 'PATCH_TORCH_FUNC'
filepath = '/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/utils/torch_functional.py'
with open(filepath, 'r') as f:
    content = f.read()

# Fix entropy_from_logits_with_chunking to handle 3D tensors (batch, seq_len, vocab)
old_code = '''def entropy_from_logits_with_chunking(logits: torch.Tensor, chunk_size: int = 2048):
    """Memory-efficient entropy calculation with chunking."""
    entropy = torch.zeros(logits.shape[0], device=logits.device)
    for i in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[i : i + chunk_size].float()
        pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk, dim=-1)
        entropy[i : i + chunk_size] = entropy_chunk
    return entropy'''

new_code = '''def entropy_from_logits_with_chunking(logits: torch.Tensor, chunk_size: int = 2048):
    """Memory-efficient entropy calculation with chunking.
    
    Supports both 2D (batch, vocab) and 3D (batch, seq_len, vocab) inputs.
    For 3D input, chunks along the flattened (batch * seq_len) dimension.
    """
    original_shape = logits.shape[:-1]  # All dims except vocab
    vocab_size = logits.shape[-1]
    
    # Flatten to 2D: (total_tokens, vocab_size)
    # Use reshape instead of view to handle non-contiguous tensors
    logits_flat = logits.reshape(-1, vocab_size)
    total_tokens = logits_flat.shape[0]
    
    entropy_flat = torch.zeros(total_tokens, device=logits.device, dtype=logits.dtype)
    for i in range(0, total_tokens, chunk_size):
        end_idx = min(i + chunk_size, total_tokens)
        logits_chunk = logits_flat[i:end_idx].float()
        pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk, dim=-1)
        entropy_flat[i:end_idx] = entropy_chunk.to(entropy_flat.dtype)
    
    # Reshape back to original shape (without vocab dim)
    return entropy_flat.reshape(original_shape)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, 'w') as f:
        f.write(content)
    print('[PATCHED] verl torch_functional.py for 3D tensor support in entropy_from_logits_with_chunking')
else:
    print('[SKIP] torch_functional.py already patched or pattern not found')
PATCH_TORCH_FUNC
fi

# Install sbys_hinting as a package so Ray workers can import it
echo "[12/12] Installing sbys_hinting as editable package..."

# Create __init__.py if it doesn't exist
if [ ! -f "$PROJECT_ROOT/sbys_hinting/__init__.py" ]; then
    echo "# sbys_hinting package" > "$PROJECT_ROOT/sbys_hinting/__init__.py"
fi

# Create setup.py if it doesn't exist
if [ ! -f "$PROJECT_ROOT/setup.py" ]; then
    cat > "$PROJECT_ROOT/setup.py" << 'SETUP_PY'
from setuptools import setup, find_packages

setup(
    name="task_runtime",
    version="0.1.0",
    packages=["sbys_hinting"],
    python_requires=">=3.10",
)
SETUP_PY
fi

# Install as editable package
cd "$PROJECT_ROOT"
pip install -e .
cd -

# Add LD_LIBRARY_PATH to the venv's activate script so it's set automatically
ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
if ! grep -q "GRPO CUDA library paths" "$ACTIVATE_SCRIPT"; then
    cat >> "$ACTIVATE_SCRIPT" << 'ACTIVATE_EOF'

# GRPO CUDA library paths (added by install-dependencies.sh)
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
# Disable flashinfer JIT (system CUDA doesn't support sm_100a for B200 GPUs)
export SGLANG_ATTENTION_BACKEND=triton
ACTIVATE_EOF
    echo "[INFO] Added LD_LIBRARY_PATH to $ACTIVATE_SCRIPT"
else
    echo "[SKIP] LD_LIBRARY_PATH already in activate script"
fi

# Verify key packages
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || true
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import verl; print(f'verl: {verl.__version__}')"
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" || {
    echo "[ERROR] flash-attn verification failed - this is required for verl"
    exit 1
}
python -c "import xformers; print(f'xformers: {xformers.__version__}')" 2>/dev/null || true
python -c "import sglang; print(f'sglang: {sglang.__version__}')" 2>/dev/null || true
python -c "import triton; print(f'triton: {triton.__version__}')" 2>/dev/null || true
python -c "import sbys_hinting; print('sbys_hinting: installed')" || {
    echo "[ERROR] sbys_hinting package not found"
    exit 1
}

# Verify verl can import sglang (tests patches)
echo ""
echo "Verifying patches..."
python -c "from verl.workers.rollout.sglang_rollout import SGLangRollout; print('[OK] verl sglang import works')" 2>/dev/null || {
    echo "[WARNING] verl sglang import test failed - patches may need manual review"
}

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Installed versions:"
echo "  - PyTorch 2.9.1+cu128 (supports B200 GPUs)"
echo "  - triton 3.5.1 (B200 sm_100 support)"
echo "  - vllm (latest)"
echo "  - sglang 0.4.6.post5 (patched for triton 3.5.x)"
echo "  - verl 0.4.1 (patched to disable CUDA graphs)"
echo "  - flash-attn (latest)"
echo "  - xformers 0.0.33.post2"
echo "  - sbys_hinting (editable package)"
echo ""
echo "Applied patches:"
echo "  - sglang utils.py: triton 3.5.x default_cache_dir compatibility"
echo "  - sglang awq.py: handle missing fused_marlin_moe"
echo "  - sglang gptq.py: handle missing fused_marlin_moe"
echo "  - verl fsdp_sglang.py: Python 3.12+ / uvloop asyncio fix"
echo "  - verl sglang_rollout.py: disable CUDA graphs for B200"
echo ""
echo "To activate the environment (LD_LIBRARY_PATH is set automatically):"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run sbys_grpo.py:"
echo "  source $VENV_PATH/bin/activate"
echo "  wandb login  # if not already logged in"
echo "  ray start --head --num-gpus=8"
echo "  cd /mnt/task_runtime/sbys_hinting && python sbys_grpo.py"
echo ""
echo "To start Ray worker (connect to head node):"
echo "  source $VENV_PATH/bin/activate"
echo "  ray start --address=<HEAD_NODE_IP>:6379"
echo ""
