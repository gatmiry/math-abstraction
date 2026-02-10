#!/bin/bash
# Install dependencies for GRPO training on B200 GPUs
# Usage: ./install-dependencies-2.sh [venv_path]
# Default venv path: /mnt/task_runtime/nenv
#
# This script installs packages in a specific order to avoid dependency conflicts.
# Supports B200 GPUs (sm_100) with PyTorch 2.9.1+cu128.
#
# This is a reproducible version based on a working environment (nenv).
# Key differences from install-dependencies.sh:
#   - Uses explicit package versions from pip freeze
#   - Installs vllm and sglang first, then verl dependencies
#   - flash-attn installation is non-blocking (continues on failure)
#
# IMPORTANT: After running this script, you also need to create token file:
#   /mnt/task_runtime/sbys_hinting/hf_token.txt with:
#   WANDB_API_KEY=your_key_here
#   HF_TOKEN=your_token_here (optional)

set -e  # Exit on error

VENV_PATH="${1:-/mnt/task_runtime/nenv}"
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
pip install --upgrade pip setuptools==80.10.2 wheel==0.46.3

# Install PyTorch with CUDA 12.8 (supports B200 GPUs with sm_100)
echo "[4/12] Installing PyTorch 2.9.1 with CUDA 12.8 (B200 GPU support)..."
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install triton 3.5.1 (required for B200 sm_100 support in ptxas)
echo "[5/12] Installing triton 3.5.1 (B200 GPU support)..."
pip install triton==3.5.1

# Install xformers compatible with PyTorch 2.9.1
echo "[6/12] Installing xformers..."
pip install xformers==0.0.33.post2

# Install vllm 0.15.1 (this brings in ray, transformers, etc.)
echo "[7/12] Installing vllm 0.15.1..."
pip install vllm==0.15.1

# Install sglang 0.4.6.post5 (compatible with verl API) and sgl-kernel
echo "[8/12] Installing sglang and dependencies..."
pip install sglang==0.4.6.post5 decord==0.6.0 sgl-kernel==0.3.21 torch-memory-saver==0.0.9 torchao==0.15.0

# Install verl 0.4.1 with --no-deps to avoid ray[default] conflicts
echo "[9/12] Installing verl 0.4.1 and its dependencies..."
pip install verl==0.4.1 --no-deps

# Install verl dependencies with explicit versions from working environment
pip install \
    accelerate==1.12.0 \
    apple-bolt==4.39.0 \
    codetiming==1.4.0 \
    datasets==4.5.0 \
    dill==0.4.0 \
    hydra-core==1.3.2 \
    math-verify==0.9.0 \
    numpy==2.2.6 \
    pandas==3.0.0 \
    peft==0.18.1 \
    pyarrow==23.0.0 \
    pybind11==3.0.1 \
    pylatexenc==2.10 \
    torchdata==0.11.0 \
    tensordict==0.11.0 \
    transformers==4.57.6 \
    wandb==0.24.2 \
    packaging==26.0

# Install flash-attn (needs special handling - requires CUDA toolkit and build dependencies)
echo "[10/12] Installing flash-attn..."
# Install build dependencies first
pip install ninja==1.13.0

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
# Set TORCH_CUDA_ARCH_LIST for B200 GPUs (sm_100 / compute capability 10.0)
export TORCH_CUDA_ARCH_LIST="10.0"
echo "Attempting to install flash-attn with TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST..."
if MAX_JOBS=$(nproc) pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir > "$LOG_FILE" 2>&1; then
    echo "[SUCCESS] flash-attn installed successfully"
else
    echo "[WARNING] flash-attn installation failed. Continuing anyway..."
    echo "[WARNING] Check log at: $LOG_FILE"
    echo "[WARNING] Training may still work if flash-attn is not strictly required"
fi

# Verify flash-attn installation (non-blocking)
if python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')" 2>/dev/null; then
    echo "[VERIFIED] flash-attn is properly installed"
else
    echo "[WARNING] flash-attn verification failed - training may continue without it"
fi

# Apply patches for B200 GPU compatibility
echo "[11/12] Applying patches for B200 GPU compatibility..."

# Patch 1: sglang utils.py - fix triton 3.5.x compatibility (default_cache_dir removed)
SGLANG_UTILS="$VENV_PATH/lib/python3.12/site-packages/sglang/srt/utils.py"
if [ -f "$SGLANG_UTILS" ]; then
    python << PATCH_UTILS
filepath = '$VENV_PATH/lib/python3.12/site-packages/sglang/srt/utils.py'
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
    python << PATCH_AWQ
filepath = '$VENV_PATH/lib/python3.12/site-packages/sglang/srt/layers/quantization/awq.py'
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
    python << PATCH_GPTQ
import re
filepath = '$VENV_PATH/lib/python3.12/site-packages/sglang/srt/layers/quantization/gptq.py'
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
    python << PATCH_FSDP_SGLANG
filepath = '$VENV_PATH/lib/python3.12/site-packages/verl/workers/sharding_manager/fsdp_sglang.py'
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
    python << PATCH_VERL
filepath = '$VENV_PATH/lib/python3.12/site-packages/verl/workers/rollout/sglang_rollout/sglang_rollout.py'
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
    python << PATCH_DP_ACTOR
filepath = '$VENV_PATH/lib/python3.12/site-packages/verl/workers/actor/dp_actor.py'
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
    python << PATCH_TORCH_FUNC
filepath = '$VENV_PATH/lib/python3.12/site-packages/verl/utils/torch_functional.py'
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

# GRPO CUDA library paths (added by install-dependencies-2.sh)
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
# Disable flashinfer JIT (system CUDA doesn't support sm_100a for B200 GPUs)
export SGLANG_ATTENTION_BACKEND=triton
ACTIVATE_EOF
    echo "[INFO] Added LD_LIBRARY_PATH to $ACTIVATE_SCRIPT"
else
    echo "[SKIP] LD_LIBRARY_PATH already in activate script"
fi

# Create token file template if it doesn't exist
TOKEN_FILE="$PROJECT_ROOT/sbys_hinting/hf_token.txt"
if [ ! -f "$TOKEN_FILE" ]; then
    cat > "$TOKEN_FILE" << 'TOKEN_TEMPLATE'
# Token configuration for sbys_grpo.py
# Replace the placeholder values with your actual API keys

# Weights & Biases API key (get yours at https://wandb.ai/authorize)
WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE

# HuggingFace token (optional, for private models)
HF_TOKEN=YOUR_HF_TOKEN_HERE
TOKEN_TEMPLATE
    echo "[INFO] Created token file template at $TOKEN_FILE"
    echo "[ACTION REQUIRED] Edit $TOKEN_FILE and add your API keys!"
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
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" 2>/dev/null || {
    echo "[WARNING] flash-attn not available - training may continue without it"
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
echo "Installed versions (from working environment):"
echo "  - PyTorch 2.9.1+cu128 (supports B200 GPUs)"
echo "  - triton 3.5.1 (B200 sm_100 support)"
echo "  - vllm 0.15.1"
echo "  - sglang 0.4.6.post5 (patched for triton 3.5.x)"
echo "  - sgl-kernel 0.3.21"
echo "  - verl 0.4.1 (patched for B200 GPUs)"
echo "  - flash-attn 2.8.3 (if available)"
echo "  - xformers 0.0.33.post2"
echo "  - transformers 4.57.6"
echo "  - ray 2.53.0"
echo "  - wandb 0.24.2"
echo "  - sbys_hinting (editable package)"
echo ""
echo "Applied patches:"
echo "  - sglang utils.py: triton 3.5.x default_cache_dir compatibility"
echo "  - sglang awq.py: handle missing fused_marlin_moe"
echo "  - sglang gptq.py: handle missing fused_marlin_moe"
echo "  - verl fsdp_sglang.py: Python 3.12+ / uvloop asyncio fix"
echo "  - verl sglang_rollout.py: disable CUDA graphs for B200"
echo "  - verl dp_actor.py: entropy calculation chunking fix"
echo "  - verl torch_functional.py: 3D tensor support for entropy"
echo ""
echo "IMPORTANT: Before running training, edit the token file:"
echo "  $TOKEN_FILE"
echo "  Add your WANDB_API_KEY (required for logging)"
echo ""
echo "To activate the environment (LD_LIBRARY_PATH is set automatically):"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run sbys_grpo.py:"
echo "  source $VENV_PATH/bin/activate"
echo "  cd /mnt/task_runtime/sbys_hinting && python sbys_grpo.py"
echo ""
