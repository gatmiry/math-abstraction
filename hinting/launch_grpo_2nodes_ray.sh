#!/bin/bash
# Launch script for GRPO training with verl on an existing Ray cluster
# 
# USAGE:
#   Ensure Ray cluster is already running (check with: ray status)
#   Then simply run:
#     bash launch_grpo_2nodes_ray.sh
#
# Prerequisites:
#   - Ray cluster must be running with 2 nodes, 16 GPUs total
#   - verl and ray packages must be installed

set -e

# Disable Ray dashboard to avoid opentelemetry conflicts with vLLM 0.8.5
export RAY_DISABLE_DASHBOARD=1

# vLLM 0.8.5 V1 engine works fine, but keep this for compatibility
export VLLM_USE_V1=0

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/hinting_grpo_ray.py"

# Check if verl is installed
if ! python -c "import verl" 2>/dev/null; then
    echo "ERROR: verl is not installed"
    echo "Please install verl: pip install verl"
    exit 1
fi

# Check if ray is installed
if ! python -c "import ray" 2>/dev/null; then
    echo "ERROR: ray is not installed"
    echo "Please install ray: pip install ray"
    exit 1
fi

# Check Ray cluster status
echo "Checking Ray cluster status..."
AVAILABLE_GPUS=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null)

if [ -z "$AVAILABLE_GPUS" ] || [ "$AVAILABLE_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs available in Ray cluster"
    echo "Please ensure Ray cluster is running: ray status"
    exit 1
fi

echo "Ray cluster connected: $AVAILABLE_GPUS GPUs available"

# Run the training script
echo "Starting GRPO training..."
python "$PYTHON_SCRIPT"

