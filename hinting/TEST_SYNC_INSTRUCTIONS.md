# Node Synchronization Test Instructions

This test script verifies that the two-node synchronization and gather operations required for GRPOTrainer work correctly.

## Purpose

The test simulates what GRPOTrainer does during the gather phase after generation:
1. Sets up distributed training across 2 nodes (16 GPUs total)
2. Tests basic communication (allreduce)
3. Tests gather operations (gather to rank 0)
4. Tests all-gather operations (all ranks get all data)
5. Tests with larger data sizes (simulating longer sequences)
6. Tests multiple sequential gathers (simulating batch processing)

## Running the Test

### On Node 0 (Master Node):

```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export NODE_RANK=0

cd /mnt/task_runtime/hinting
./test_sync_launch.sh
```

### On Node 1 (Worker Node):

```bash
export MASTER_ADDR=<master_node_ip>  # Same as Node 0
export MASTER_PORT=29500
export NODE_RANK=1

cd /mnt/task_runtime/hinting
./test_sync_launch.sh
```

**Important**: Start both nodes at approximately the same time (within a few seconds).

## Alternative: Manual torchrun

If you prefer to run torchrun manually:

### On Node 0:
```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export NODE_RANK=0
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank=0 \
    test_node_sync.py
```

### On Node 1:
```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export NODE_RANK=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank=1 \
    test_node_sync.py
```

## Expected Output

If everything works correctly, you should see:
- All ranks successfully initialize
- Basic communication test passes (allreduce)
- Gather operations complete successfully
- All-gather operations complete successfully
- Large data gather operations complete successfully
- Sequential batch gathers complete successfully
- Final message: "ALL TESTS PASSED!"

## Troubleshooting

### If tests fail:

1. **Check network connectivity**: Ensure nodes can reach each other on the specified port
   ```bash
   # On Node 0, check if port is accessible
   nc -zv <master_node_ip> 29500
   ```

2. **Check NCCL environment**: Ensure NCCL can find the network interface
   ```bash
   export NCCL_SOCKET_IFNAME=eth0  # or your network interface name
   ```

3. **Check firewall**: Ensure port 29500 (or your MASTER_PORT) is open

4. **Check CUDA/NCCL versions**: Ensure all nodes have compatible CUDA and NCCL versions

5. **Check timing**: Ensure both nodes start within a few seconds of each other

### Common Issues:

- **"Connection refused"**: Check MASTER_ADDR and ensure master node is running first
- **"Timeout"**: Increase NCCL_TIMEOUT or check network connectivity
- **"NCCL error"**: Check NCCL_DEBUG output for detailed error messages

## What This Tests

This test verifies:
1. ✅ Basic distributed setup works
2. ✅ Nodes can communicate (allreduce)
3. ✅ Gather operations work (rank 0 collects from all ranks)
4. ✅ All-gather operations work (all ranks get all data)
5. ✅ Large data transfers work (simulating long sequences)
6. ✅ Sequential operations work (simulating multiple batches)

If all tests pass, the issue with GRPOTrainer is likely in its specific implementation, not in the basic communication infrastructure.


