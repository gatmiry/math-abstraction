#!/bin/bash
# Install dependencies for newverl environment
# Usage: ./newverl-install-dependencies.sh [venv_path]
# Default venv path: /mnt/task_runtime/myenv
#
# Installs: numpy, apple_bolt (from system), math-verify,
#           verl (editable from local/git source),
#           transformers 4.x (pinned, verl needs AutoModelForVision2Seq removed in v5)

set -e

VENV_PATH="${1:-/mnt/task_runtime/myenv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SYS_SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"

echo "=============================================="
echo "newverl Environment Setup"
echo "=============================================="
echo "Virtual environment: $VENV_PATH"
echo ""

# --- Step 1: Create virtualenv ---
if [ ! -d "$VENV_PATH" ]; then
    echo "[1/7] Creating virtual environment (with system site-packages)..."
    python3.12 -m venv --system-site-packages "$VENV_PATH"
else
    echo "[1/7] Virtual environment already exists at $VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel

# --- Step 1b: Add pip constraints to prevent shadowing system torch/triton/nvidia ---
echo "[1b] Setting up pip constraints (block torch/triton/nvidia from being installed)..."
cat > "$VENV_PATH/pip-constraints.txt" << 'CONSTRAINTS'
# Prevent pip from installing packages that must come from the system (docker image).
# These are inherited via --system-site-packages. Installing them in myenv causes
# version conflicts with CUDA/torch ecosystem.
torch==0.0.0
torchvision==0.0.0
torchaudio==0.0.0
triton==0.0.0
nvidia-cublas-cu12==0.0.0
nvidia-cuda-cupti-cu12==0.0.0
nvidia-cuda-nvrtc-cu12==0.0.0
nvidia-cuda-runtime-cu12==0.0.0
nvidia-cudnn-cu12==0.0.0
nvidia-cufft-cu12==0.0.0
nvidia-curand-cu12==0.0.0
nvidia-cusolver-cu12==0.0.0
nvidia-cusparse-cu12==0.0.0
nvidia-nccl-cu12==0.0.0
nvidia-nvjitlink-cu12==0.0.0
nvidia-nvtx-cu12==0.0.0
CONSTRAINTS

cat > "$VENV_PATH/pip.conf" << PIPCONF
[install]
constraint = $VENV_PATH/pip-constraints.txt
PIPCONF

echo "export PIP_CONFIG_FILE=$VENV_PATH/pip.conf" >> "$VENV_PATH/bin/activate"

# --- Step 2: Install numpy ---
echo "[2/7] Installing numpy..."
pip install numpy

# --- Step 3: Install apple_bolt (Apple-internal, copied from system) ---
echo "[3/7] Installing apple_bolt from system packages..."

DEST="$VENV_PATH/lib/python3.12/site-packages"

# apple_bolt and its Apple-internal dependencies are not on PyPI.
# Copy them from the system-wide installation.
INTERNAL_PKGS=(
    apple_bolt
    boltproto
    apple_certifi
    turibolt_py_blobby_client
    turibolt_py_common
    notary_client
    lightning_py_common
    turi_molecule
    molecule
    apple_sync_boto3
    identity_manager
    bolt_http_tunnel
    botocore
    boto3
    s3transfer
)

for pkg in "${INTERNAL_PKGS[@]}"; do
    if [ -d "$SYS_SITE_PACKAGES/$pkg" ] && [ ! -d "$DEST/$pkg" ]; then
        cp -r "$SYS_SITE_PACKAGES/$pkg" "$DEST/"
        echo "  Copied package dir: $pkg"
    fi
    for d in "$SYS_SITE_PACKAGES/${pkg}"*.dist-info; do
        if [ -d "$d" ] && [ ! -d "$DEST/$(basename "$d")" ]; then
            cp -r "$d" "$DEST/"
            echo "  Copied dist-info:   $(basename "$d")"
        fi
    done
done

# Copy CLI scripts
for script in apple-bolt apple-bolt-mcp bolt bolt_mcp bolt_tunnel; do
    if [ -f "/usr/local/bin/$script" ]; then
        cp "/usr/local/bin/$script" "$VENV_PATH/bin/"
        chmod +x "$VENV_PATH/bin/$script"
    fi
done

# Install public dependencies required by apple_bolt and its internal deps
echo "  Installing apple_bolt public dependencies..."
pip install \
    argcomplete \
    "async-generator>=1.10" \
    "pydantic>=2.0.0" \
    "grpcio!=1.52.0,!=1.55.0,!=1.65.0,!=1.65.1,!=1.68.0,!=1.68.1" \
    "cerberus>1.2" \
    "dpath>=1.5.0" \
    gitpython \
    gitignore-parser \
    pandas \
    pyyaml \
    pyotp \
    "python-dateutil>=2.9.0" \
    pytz \
    requests \
    tabulate \
    "tqdm>=4.38.0" \
    jinja2 \
    "urllib3<3.0.0,>=1.25.3" \
    "watchdog>=1.0.1" \
    "websockets>=13.0.0" \
    "jsonschema<5,>=4.23.0" \
    filelock \
    "protobuf<8.0,>=4.0" \
    "cachetools>=5.0.0" \
    "pyhumps>=3.5.3" \
    "pyjwt<3.0.0,>=2.1.0" \
    strictyaml \
    "tenacity!=8.4.0" \
    "dataclass-wizard>=0.22.0" \
    jmespath

# Verify apple_bolt
python -c "import apple_bolt; print(f'  apple_bolt OK')"

# --- Step 4: Install math-verify (used by math_checker.py) ---
echo "[4/7] Installing math-verify..."
pip install math-verify

# --- Step 5: Install verl from local source (editable) ---
echo "[5/7] Installing verl from local source..."
VERL_DIR="$PROJECT_ROOT/verl"
if [ ! -d "$VERL_DIR" ]; then
    echo "  Cloning verl from git..."
    git clone https://github.com/volcengine/verl.git "$VERL_DIR"
fi
# Use --no-deps to avoid pulling in a newer torch that conflicts with the
# system torch+torchvision+nvidia packages from the docker image
pip install -e "$VERL_DIR" --no-deps

# --- Step 6: Pin transformers to 4.x (verl uses AutoModelForVision2Seq, removed in v5) ---
echo "[6/7] Pinning transformers to 4.x..."
pip install "transformers>=4.57,<5"

# --- Step 7: Clean up redundant shadows ---
echo "[7/7] Removing redundant packages from venv (inherited from system)..."
DEST="$VENV_PATH/lib/python3.12/site-packages"
python3 -c "
import os, shutil
myenv_sp = '$DEST'
system_sp = '/usr/local/lib/python3.12/dist-packages'
# Keep only packages that are myenv-specific (not in system)
keep_prefixes = [
    'math_verify', 'latex2sympy2', 'markdown_it', 'mdurl',
    'rich', 'shellingham', 'typer', 'cryptography', 'jwt', 'pyjwt', 'PyJWT',
    'transformers', '__editable__', '__pycache__', 'distutils-precedence',
]
removed = 0
for entry in os.listdir(myenv_sp):
    if any(entry.startswith(p) for p in keep_prefixes):
        continue
    if entry.endswith('.dist-info'):
        continue  # handled with their modules
    full = os.path.join(myenv_sp, entry)
    sys_full = os.path.join(system_sp, entry)
    if os.path.exists(sys_full) or entry.startswith('nvidia') or entry.startswith('_cuda'):
        if os.path.isdir(full):
            shutil.rmtree(full)
        elif os.path.isfile(full):
            os.remove(full)
        removed += 1
# Also remove dist-info for packages that exist in system
for entry in os.listdir(myenv_sp):
    if entry.endswith('.dist-info'):
        if any(entry.startswith(p) for p in keep_prefixes):
            continue
        name = entry[:-len('.dist-info')].rsplit('-', 1)[0].replace('_', '-').lower()
        # Check if system has a dist-info for same package
        has_system = any(
            e.endswith('.dist-info') and
            e[:-len('.dist-info')].rsplit('-', 1)[0].replace('_', '-').lower() == name
            for e in os.listdir(system_sp)
        )
        if has_system:
            shutil.rmtree(os.path.join(myenv_sp, entry))
            removed += 1
print(f'  Cleaned {removed} redundant entries from venv site-packages')
"

# --- Verify ---
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="
python -c "import numpy; print(f'  numpy:      {numpy.__version__}')"
python -c "import apple_bolt; print(f'  apple_bolt: OK')"
python -c "import verl; print(f'  verl:       installed')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "from verl.interactions.base import BaseInteraction; print(f'  verl.interactions: OK')"

echo ""
echo "Done! Activate with:"
echo "  source $VENV_PATH/bin/activate"
