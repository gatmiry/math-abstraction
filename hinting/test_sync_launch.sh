#!/bin/bash
# Launch script for testing node synchronization
# Usage: Run this script on each node with appropriate NODE_RANK

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_node_sync.py"

# Get node configuration from environment or use defaults
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

# Validate required environment variables
if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: MASTER_ADDR environment variable not set"
    echo "Please set MASTER_ADDR to the IP address of the master node"
    exit 1
fi

if [ -z "$NODE_RANK" ]; then
    echo "ERROR: NODE_RANK environment variable not set"
    echo "Please set NODE_RANK to 0 for master node, 1 for worker node"
    exit 1
fi

echo "=========================================="
echo "Node Synchronization Test Launch"
echo "=========================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "=========================================="

# Set NCCL environment variables
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0  # Adjust if needed for your network interface

# Launch with torchrun
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$NODE_RANK" \
    "$TEST_SCRIPT"


