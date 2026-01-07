# Ray Launch Instructions for GRPO Training

This guide explains how to run GRPO training with verl and Ray on 2 nodes, each with 8 H100 GPUs.

## Prerequisites

1. **verl and ray installed** on both nodes:
   ```bash
   pip install verl ray
   ```

2. **Network connectivity** between nodes (Ray needs to communicate)

3. **Shared filesystem** or synchronized model/dataset paths across nodes

## Step-by-Step Instructions

### Step 1: Prepare Node 0 (Head Node)

On Node 0, set the environment variables and start the Ray head node:

```bash
# Set the IP address of Node 0 (head node)
export RAY_HEAD_IP="<node0_ip_address>"  # Replace with actual IP, e.g., "192.168.1.100"
export RAY_HEAD_PORT="10001"  # Default Ray port

# Navigate to the script directory
cd /mnt/task_runtime/hinting

# Start Ray head node and run training
bash launch_grpo_2nodes_ray.sh
# OR explicitly:
bash launch_grpo_2nodes_ray.sh head
```

**What happens:**
- Ray head node starts on Node 0
- The script will wait for workers to connect
- Once workers connect, training will start automatically

### Step 2: Prepare Node 1 (Worker Node)

**IMPORTANT:** Start the worker node AFTER the head node is running.

On Node 1, set the environment variables and start the Ray worker node:

```bash
# Set the IP address of Node 0 (head node) - MUST be the same as Node 0's IP
export RAY_HEAD_IP="<node0_ip_address>"  # Same IP as Node 0!
export RAY_HEAD_PORT="10001"  # Same port as Node 0

# Navigate to the script directory
cd /mnt/task_runtime/hinting

# Start Ray worker node
bash launch_grpo_2nodes_ray.sh worker
```

**What happens:**
- Ray worker node starts and connects to the head node
- The worker will participate in training automatically
- The worker script will keep running (press Ctrl+C to stop)

## Complete Example

### On Node 0 (Head Node):
```bash
export RAY_HEAD_IP="192.168.1.100"  # Replace with your Node 0 IP
export RAY_HEAD_PORT="10001"
cd /mnt/task_runtime/hinting
bash launch_grpo_2nodes_ray.sh
```

### On Node 1 (Worker Node):
```bash
export RAY_HEAD_IP="192.168.1.100"  # Same as Node 0!
export RAY_HEAD_PORT="10001"
cd /mnt/task_runtime/hinting
bash launch_grpo_2nodes_ray.sh worker
```

## Order of Operations

1. **First:** Start Node 0 (head node) - this will start Ray and wait
2. **Second:** Start Node 1 (worker node) - this will connect to the head
3. **Automatic:** Once both nodes are connected, training starts

## Verifying Ray Cluster

You can check if the Ray cluster is set up correctly:

### On Node 0 (head node):
```bash
# Check Ray status
ray status

# View Ray dashboard (if enabled)
# Open browser to: http://<node0_ip>:8265
```

You should see:
- 1 head node
- 1 worker node (after Node 1 connects)
- 16 GPUs total (8 per node)

## Troubleshooting

### Issue: Worker cannot connect to head
- **Check:** Ensure `RAY_HEAD_IP` is set correctly on the worker node
- **Check:** Ensure firewall allows traffic on port 10001
- **Check:** Ensure head node is running before starting worker

### Issue: Ray cluster shows only 1 node
- **Check:** Ensure worker node script is running
- **Check:** Check Ray logs: `ray status` or `tail -f /tmp/ray/session_latest/logs/*.log`

### Issue: Training doesn't start
- **Check:** Ensure both nodes are connected (use `ray status`)
- **Check:** Ensure dataset and model paths are accessible from both nodes
- **Check:** Check Python script logs for errors

## Stopping the Cluster

### To stop training and Ray cluster:

**On Node 0 (head node):**
- Press Ctrl+C in the terminal running the training script
- Or run: `ray stop`

**On Node 1 (worker node):**
- Press Ctrl+C in the terminal running the worker script
- Or run: `ray stop`

## Notes

- The head node runs the training script automatically
- The worker node just connects and participates - it doesn't run the script
- Ray handles all distributed coordination automatically
- Training progress will be visible on the head node terminal
- Wandb logging (if enabled) will show progress from the head node

