#!/bin/bash
# Install dependencies for newverl environment
# Usage: ./newverl-install-dependencies.sh [venv_path]
# Default venv path: /mnt/task_runtime/myenv
#
# Installs: numpy, apple_bolt (from system), verl (from git)

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
    echo "[1/4] Creating virtual environment..."
    python3.12 -m venv "$VENV_PATH"
else
    echo "[1/4] Virtual environment already exists at $VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel

# --- Step 2: Install numpy ---
echo "[2/4] Installing numpy..."
pip install numpy

# --- Step 3: Install apple_bolt (Apple-internal, copied from system) ---
echo "[3/4] Installing apple_bolt from system packages..."

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

# --- Step 4: Install verl from git ---
echo "[4/4] Installing verl from git (latest)..."
pip install git+https://github.com/volcengine/verl.git

# --- Verify ---
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="
python -c "import numpy; print(f'  numpy:      {numpy.__version__}')"
python -c "import apple_bolt; print(f'  apple_bolt: OK')"
python -c "import verl; print(f'  verl:       installed')"

echo ""
echo "Done! Activate with:"
echo "  source $VENV_PATH/bin/activate"
