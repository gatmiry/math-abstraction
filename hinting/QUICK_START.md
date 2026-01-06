# Quick Start: 2-Node GRPO Training

## TL;DR - Copy and Paste Commands

### Step 1: Find Node 0's IP Address
```bash
# Run on Node 0
hostname -I | awk '{print $1}'
# Copy this IP address (e.g., 192.168.1.100)
```

### Step 2: Launch Node 0 (Master)
```bash
cd /mnt/task_runtime/hinting
export MASTER_ADDR="<PASTE_NODE0_IP_HERE>"  # e.g., 192.168.1.100
export MASTER_PORT="29500"
export NODE_RANK=0
bash launch_grpo_2nodes.sh
```

### Step 3: Launch Node 1 (Worker) - Start AFTER Node 0
```bash
cd /mnt/task_runtime/hinting
export MASTER_ADDR="<PASTE_NODE0_IP_HERE>"  # Same IP as Node 0!
export MASTER_PORT="29500"
export NODE_RANK=1
bash launch_grpo_2nodes.sh
```

## Example with Real IP

If Node 0's IP is `192.168.1.100`:

**Node 0:**
```bash
cd /mnt/task_runtime/hinting
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"
export NODE_RANK=0
bash launch_grpo_2nodes.sh
```

**Node 1:**
```bash
cd /mnt/task_runtime/hinting
export MASTER_ADDR="192.168.1.100"  # Same!
export MASTER_PORT="29500"
export NODE_RANK=1                   # Different!
bash launch_grpo_2nodes.sh
```

## Key Points

- ✅ `MASTER_ADDR` = Node 0's IP (same on both nodes)
- ✅ `MASTER_PORT` = 29500 (same on both nodes)
- ✅ `NODE_RANK` = 0 on Node 0, 1 on Node 1 (different!)
- ✅ Start Node 0 first, then Node 1 within a few seconds
- ✅ Both nodes must be able to communicate (test with `ping`)

## Troubleshooting

**Can't connect?**
```bash
# Test connectivity from Node 1 to Node 0
ping <NODE0_IP>
nc -zv <NODE0_IP> 29500
```

**Port in use?**
```bash
# Change port on both nodes
export MASTER_PORT="29501"
```

For detailed instructions, see `LAUNCH_INSTRUCTIONS.md`


