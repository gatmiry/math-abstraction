#!/bin/bash
# Launch script for GRPO training with verl and Ray on 2 nodes, each with 8 H100 GPUs
# 
# USAGE:
#   On Node 0 (head node):
#     export RAY_HEAD_IP="<node0_ip>"
#     export RAY_HEAD_PORT="10001"
#     bash launch_grpo_2nodes_ray.sh
#
#   On Node 1 (worker node):
#     export RAY_HEAD_IP="<node0_ip>"  # Same as Node 0!
#     export RAY_HEAD_PORT="10001"
#     bash launch_grpo_2nodes_ray.sh worker
#
# See verl documentation for Ray cluster setup details

# Configuration
NUM_NODES=2
GPUS_PER_NODE=8
RAY_HEAD_IP="${RAY_HEAD_IP:-localhost}"  # MUST be set to Node 0's IP address
RAY_HEAD_PORT="${RAY_HEAD_PORT:-10001}"

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

# Determine if this is the head node or worker node
NODE_TYPE="${1:-head}"

if [ "$NODE_TYPE" == "head" ]; then
    echo "Starting Ray head node..."
    echo "Ray head IP: $RAY_HEAD_IP"
    echo "Ray head port: $RAY_HEAD_PORT"
    
    # Start Ray head node
    ray start --head \
        --node-ip-address="$RAY_HEAD_IP" \
        --port="$RAY_HEAD_PORT" \
        --num-gpus="$GPUS_PER_NODE" \
        --dashboard-host=0.0.0.0
    
    echo "Ray head node started. Waiting for workers to connect..."
    echo "Run this script with 'worker' argument on other nodes:"
    echo "  bash launch_grpo_2nodes_ray.sh worker"
    
    # Run the training script
    echo "Starting training..."
    python "$PYTHON_SCRIPT"
    
elif [ "$NODE_TYPE" == "worker" ]; then
    echo "Starting Ray worker node..."
    echo "Connecting to Ray head at: $RAY_HEAD_IP:$RAY_HEAD_PORT"
    
    # Start Ray worker node
    ray start --address="$RAY_HEAD_IP:$RAY_HEAD_PORT" \
        --num-gpus="$GPUS_PER_NODE"
    
    echo "Ray worker node started and connected to head."
    echo "Worker will participate in training automatically."
    
    # Keep the worker running
    echo "Worker node is running. Press Ctrl+C to stop."
    while true; do
        sleep 10
    done
else
    echo "ERROR: Invalid node type: $NODE_TYPE"
    echo "Usage:"
    echo "  Head node: bash launch_grpo_2nodes_ray.sh"
    echo "  Worker node: bash launch_grpo_2nodes_ray.sh worker"
    exit 1
fi

