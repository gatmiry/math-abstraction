# Step-by-Step Guide: Running GRPO on 2 Nodes

This guide explains exactly how to run the GRPO training script on 2 nodes, each with 8 H100 GPUs.

## Prerequisites

1. **Network Setup**: Both nodes must be able to communicate with each other
2. **Shared Filesystem** (recommended): Both nodes should have access to:
   - The model directory (`MODEL_PATH`)
   - The dataset (or internet access to download it)
   - The output directory (`OUTPUT_DIR`)

## Step 1: Identify Node IP Addresses

On **Node 0** (master node), find its IP address:
```bash
# Option 1: Using hostname
hostname -I

# Option 2: Using ip command
ip addr show | grep "inet " | grep -v 127.0.0.1

# Option 3: Using ifconfig
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Example output**: `192.168.1.100` (this will be your MASTER_ADDR)

**Write down the IP address of Node 0** - you'll need it for both nodes.

## Step 2: Test Network Connectivity

On **Node 1**, test if you can reach Node 0:
```bash
# Replace with Node 0's actual IP
ping 192.168.1.100

# Or test the port (default 29500)
nc -zv 192.168.1.100 29500
```

If these fail, check:
- Firewall rules (may need to open port 29500)
- Network configuration
- Both nodes on the same network/subnet

## Step 3: Prepare Environment on Both Nodes

On **both nodes**, ensure you're in the correct directory:
```bash
cd /mnt/task_runtime/hinting
```

Verify the script exists:
```bash
ls -la hinting_grpo.py launch_grpo_2nodes.sh
```

## Step 4: Launch Training

### On Node 0 (Master Node)

Open a terminal and run:

```bash
cd /mnt/task_runtime/hinting

# Set environment variables
export MASTER_ADDR="192.168.1.100"  # Replace with Node 0's actual IP
export MASTER_PORT="29500"
export NODE_RANK=0

# Launch training
bash launch_grpo_2nodes.sh
```

**OR** use torchrun directly:

```bash
cd /mnt/task_runtime/hinting

export MASTER_ADDR="192.168.1.100"  # Replace with Node 0's actual IP
export MASTER_PORT="29500"
export NODE_RANK=0

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank=${NODE_RANK} \
    hinting_grpo.py
```

### On Node 1 (Worker Node)

**IMPORTANT**: Start Node 1 **after** Node 0 has started (within a few seconds).

Open a terminal and run:

```bash
cd /mnt/task_runtime/hinting

# Set environment variables
export MASTER_ADDR="192.168.1.100"  # Same as Node 0's IP
export MASTER_PORT="29500"           # Same as Node 0
export NODE_RANK=1                   # Different from Node 0!

# Launch training
bash launch_grpo_2nodes.sh
```

**OR** use torchrun directly:

```bash
cd /mnt/task_runtime/hinting

export MASTER_ADDR="192.168.1.100"  # Same as Node 0's IP
export MASTER_PORT="29500"           # Same as Node 0
export NODE_RANK=1                   # Different from Node 0!

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank=${NODE_RANK} \
    hinting_grpo.py
```

## Step 5: Monitor Training

- **Node 0** will print training logs (rank 0 process)
- **Node 1** will run silently but participate in training
- Check GPU usage on both nodes: `watch -n 1 nvidia-smi`

## Environment Variables Explained

| Variable | Description | Node 0 | Node 1 |
|---------|-------------|--------|--------|
| `MASTER_ADDR` | IP address of the master node | Node 0's IP | Node 0's IP (same!) |
| `MASTER_PORT` | Port for communication | 29500 | 29500 (same!) |
| `NODE_RANK` | Which node this is | 0 | 1 (different!) |
| `WORLD_SIZE` | Total number of processes | Auto-set | Auto-set |
| `RANK` | Global process rank | Auto-set | Auto-set |
| `LOCAL_RANK` | Local GPU rank on this node | Auto-set | Auto-set |

## Common Issues and Solutions

### Issue 1: "Connection refused" or timeout

**Solution**:
- Verify `MASTER_ADDR` is correct on both nodes
- Check firewall: `sudo ufw allow 29500/tcp` (if using ufw)
- Ensure Node 0 starts before Node 1
- Try a different port if 29500 is blocked

### Issue 2: "Address already in use"

**Solution**:
- Port 29500 is already in use
- Change `MASTER_PORT` to something else (e.g., 29501) on both nodes
- Or kill the process using the port: `lsof -ti:29500 | xargs kill -9`

### Issue 3: Nodes can't find each other

**Solution**:
- Verify both nodes are on the same network
- Check routing: `ip route` on both nodes
- Try using hostname instead of IP (if DNS is configured)
- Use `ping` to test connectivity

### Issue 4: Model/dataset not found

**Solution**:
- Ensure paths are accessible from both nodes
- Use absolute paths in the script
- If using NFS/shared storage, verify mounts are correct
- Check file permissions

### Issue 5: CUDA out of memory

**Solution**:
- Reduce `per_device_train_batch_size` in `hinting_grpo.py`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Already using 8-bit optimizer and gradient checkpointing

## Alternative: Using SSH to Launch Remotely

If you want to launch from a single machine:

### Launch Node 0 locally:
```bash
cd /mnt/task_runtime/hinting
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"
export NODE_RANK=0
bash launch_grpo_2nodes.sh
```

### Launch Node 1 remotely (from Node 0):
```bash
ssh user@node1_ip "cd /mnt/task_runtime/hinting && export MASTER_ADDR=192.168.1.100 && export MASTER_PORT=29500 && export NODE_RANK=1 && bash launch_grpo_2nodes.sh"
```

Or use a job scheduler like SLURM if available.

## Verification Checklist

Before starting training, verify:

- [ ] Node 0's IP address is known
- [ ] Both nodes can ping each other
- [ ] Port 29500 is open (or choose another port)
- [ ] Both nodes have access to model and dataset
- [ ] `NODE_RANK=0` on Node 0
- [ ] `NODE_RANK=1` on Node 1
- [ ] `MASTER_ADDR` is the same on both nodes (Node 0's IP)
- [ ] `MASTER_PORT` is the same on both nodes
- [ ] All 16 GPUs are visible (`nvidia-smi` on both nodes)

## Example Complete Session

### Terminal 1 (Node 0):
```bash
# Get IP address
hostname -I
# Output: 192.168.1.100

# Launch training
cd /mnt/task_runtime/hinting
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"
export NODE_RANK=0
bash launch_grpo_2nodes.sh
```

### Terminal 2 (Node 1, started after Node 0):
```bash
# Launch training
cd /mnt/task_runtime/hinting
export MASTER_ADDR="192.168.1.100"  # Node 0's IP
export MASTER_PORT="29500"
export NODE_RANK=1
bash launch_grpo_2nodes.sh
```

## Expected Output

You should see on Node 0:
```
Starting GRPO training with 16 GPUs
Configuration: 2 nodes, 8 GPUs per node
Using verl for GRPO training  # or "Using trl for GRPO training"
Loading model from ...
Loading dataset: KbsdJames/Omni-MATH...
Dataset size: ...
Formatting dataset...
Filtered dataset size: ...
Initializing GRPO trainer...
Starting GRPO training...
```

Training will proceed with logs every 10 steps (as configured).


