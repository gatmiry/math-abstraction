#!/bin/bash
# Run this script on each worker node to clean up and connect to Ray cluster

set -e

HEAD_NODE="240.12.51.108:6379"

echo "=== Cleaning up old processes ==="
ray stop --force 2>/dev/null || true
pkill -9 -f "python.*sbys" 2>/dev/null || true
pkill -9 -f sglang 2>/dev/null || true
sleep 2

echo "=== Pulling latest code ==="
cd /mnt/task_runtime
git pull

echo "=== Activating virtual environment ==="
source myenv/bin/activate

echo "=== Connecting to Ray cluster ==="
ray start --address="$HEAD_NODE"

echo "=== Done! Worker connected to $HEAD_NODE ==="

