#!/bin/bash
# Launch script for GRPO training on 2 nodes, each with 8 H100 GPUs
# 
# USAGE:
#   On Node 0 (master):
#     export MASTER_ADDR="<node0_ip>"
#     export NODE_RANK=0
#     bash launch_grpo_2nodes.sh
#
#   On Node 1 (worker):
#     export MASTER_ADDR="<node0_ip>"  # Same as Node 0!
#     export NODE_RANK=1
#     bash launch_grpo_2nodes.sh
#
# See LAUNCH_INSTRUCTIONS.md for detailed step-by-step guide

# Configuration
NUM_NODES=2
GPUS_PER_NODE=8
MASTER_ADDR="${MASTER_ADDR:-localhost}"  # MUST be set to Node 0's IP address
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"  # MUST be set: 0 for master, 1 for worker

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/hinting_grpo.py"

# Validation
if [ "$MASTER_ADDR" == "localhost" ] && [ "$NODE_RANK" == "1" ]; then
    echo "ERROR: MASTER_ADDR is set to 'localhost' but NODE_RANK=1 (worker node)"
    echo "Please set MASTER_ADDR to Node 0's IP address"
    echo "Example: export MASTER_ADDR='192.168.1.100'"
    exit 1
fi

if [ -z "$NODE_RANK" ]; then
    echo "ERROR: NODE_RANK is not set"
    echo "Please set NODE_RANK=0 on Node 0 (master) or NODE_RANK=1 on Node 1 (worker)"
    exit 1
fi

# Check if verl is installed (optional, will fallback to trl)
if ! python -c "import verl" 2>/dev/null; then
    if [ "$NODE_RANK" == "0" ]; then
        echo "Warning: verl not found. Will use trl instead (if available)."
        echo "For optimal multi-node performance, install verl: pip install verl"
    fi
fi

# Launch training using torchrun (PyTorch's distributed launcher)
# This will work with both verl and trl
echo "=========================================="
echo "Launching GRPO training"
echo "=========================================="
echo "Nodes: ${NUM_NODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Total GPUs: $((NUM_NODES * GPUS_PER_NODE))"
echo "Master address: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"
echo "Node rank: ${NODE_RANK}"
echo "=========================================="
echo ""

# Check if this is the master node and if other node is ready
if [ "$NODE_RANK" == "0" ]; then
    echo "Starting as MASTER node (Node 0)"
    echo "Waiting for worker node to connect..."
    echo ""
elif [ "$NODE_RANK" == "1" ]; then
    echo "Starting as WORKER node (Node 1)"
    echo "Connecting to master at ${MASTER_ADDR}:${MASTER_PORT}..."
    echo ""
else
    echo "ERROR: Invalid NODE_RANK=${NODE_RANK}. Must be 0 or 1"
    exit 1
fi

# Launch with torchrun
torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank=${NODE_RANK} \
    ${PYTHON_SCRIPT}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code: $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi

