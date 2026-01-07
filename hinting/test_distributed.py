#!/usr/bin/env python3
"""
Simple test script to verify distributed setup is working.
Run this with torchrun to test connectivity before running the full training.
"""

import os
import torch
import torch.distributed as dist
import datetime

def test_distributed():
    """Test basic distributed functionality."""
    print(f"[TEST] Starting distributed test")
    print(f"[TEST] Environment variables:")
    print(f"  RANK: {os.environ.get('RANK', 'NOT SET')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("[ERROR] Not running in distributed mode. Use torchrun to launch.")
        return
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    
    print(f"[TEST] Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    print(f"[TEST] Using device: {device}")
    
    # Initialize process group
    print(f"[TEST] Initializing process group...")
    try:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=300)  # 5 minute timeout for test
        )
        print(f"[TEST] Process group initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize process group: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 1: Simple allreduce
    print(f"[TEST] Test 1: AllReduce operation...")
    try:
        test_tensor = torch.ones(1, device=device) * (rank + 1)
        print(f"[TEST] Rank {rank}: Before allreduce, value = {test_tensor.item()}")
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(1, world_size + 1))
        print(f"[TEST] Rank {rank}: After allreduce, value = {test_tensor.item()}, expected = {expected_sum}")
        if abs(test_tensor.item() - expected_sum) < 0.01:
            print(f"[TEST] ✓ AllReduce test PASSED")
        else:
            print(f"[TEST] ✗ AllReduce test FAILED")
    except Exception as e:
        print(f"[ERROR] AllReduce test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Barrier
    print(f"[TEST] Test 2: Barrier synchronization...")
    try:
        print(f"[TEST] Rank {rank}: Reaching barrier...")
        dist.barrier()
        print(f"[TEST] Rank {rank}: Passed barrier")
        print(f"[TEST] ✓ Barrier test PASSED")
    except Exception as e:
        print(f"[ERROR] Barrier test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Broadcast
    print(f"[TEST] Test 3: Broadcast operation...")
    try:
        if rank == 0:
            broadcast_tensor = torch.ones(1, device=device) * 42.0
        else:
            broadcast_tensor = torch.zeros(1, device=device)
        print(f"[TEST] Rank {rank}: Before broadcast, value = {broadcast_tensor.item()}")
        dist.broadcast(broadcast_tensor, src=0)
        print(f"[TEST] Rank {rank}: After broadcast, value = {broadcast_tensor.item()}")
        if abs(broadcast_tensor.item() - 42.0) < 0.01:
            print(f"[TEST] ✓ Broadcast test PASSED")
        else:
            print(f"[TEST] ✗ Broadcast test FAILED")
    except Exception as e:
        print(f"[ERROR] Broadcast test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"[TEST] All tests completed successfully!")
    print(f"[TEST] Rank {rank}: Distributed setup is working correctly")
    
    # Cleanup
    dist.destroy_process_group()
    print(f"[TEST] Process group destroyed")

if __name__ == "__main__":
    test_distributed()



