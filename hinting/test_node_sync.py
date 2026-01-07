#!/usr/bin/env python3
"""
Minimal test script to verify two-node synchronization and gather operations
required for GRPOTrainer's gather phase.

This simulates what GRPOTrainer does during the gather phase after generation.
"""

import os
import torch
import torch.distributed as dist
import datetime
import time

def setup_distributed():
    """Setup distributed training environment similar to hinting_grpo.py"""
    # Set NCCL timeout environment variable if not already set
    if 'NCCL_TIMEOUT' not in os.environ:
        os.environ['NCCL_TIMEOUT'] = '7200'
    
    # Enable NCCL debugging
    if 'NCCL_DEBUG' not in os.environ:
        os.environ['NCCL_DEBUG'] = 'WARN'
    
    print(f"[TEST] Environment variables:")
    print(f"  RANK: {os.environ.get('RANK', 'NOT SET')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    print(f"  NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT', 'NOT SET')}")
    print(f"  NCCL_DEBUG: {os.environ.get('NCCL_DEBUG', 'NOT SET')}")
    
    # Initialize process group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        print(f"[TEST] Initializing distributed training:")
        print(f"  Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"[TEST] Set device to: {device}")
        
        # Initialize process group
        print(f"[TEST] Calling dist.init_process_group...")
        
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        init_method = f"tcp://{master_addr}:{master_port}"
        print(f"[TEST] Using init_method: {init_method}")
        
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method=init_method,
                    rank=rank,
                    world_size=world_size,
                    timeout=datetime.timedelta(seconds=7200)  # 2 hour timeout
                )
                print(f"[TEST] dist.init_process_group completed successfully")
            else:
                print(f"[TEST] Process group already initialized")
            
            # Test basic communication
            print(f"[TEST] Testing basic communication with allreduce...")
            test_tensor = torch.ones(1, device=device)
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            print(f"[TEST] Communication test successful! Result: {test_tensor.item()} (expected: {world_size})")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize process group: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return rank, world_size, local_rank, device
    else:
        print(f"[ERROR] RANK and WORLD_SIZE environment variables not set")
        print(f"[ERROR] Please run this script with torchrun")
        raise RuntimeError("Not in distributed mode")


def test_gather_operation(rank, world_size, device):
    """Test gather operation similar to what GRPOTrainer does."""
    print(f"\n[TEST] Rank {rank}: Starting gather operation test...")
    
    # Simulate what happens after generation: each rank has generated some data
    # In GRPOTrainer, this would be the generated sequences
    batch_size_per_rank = 2  # Simulate 2 generations per rank
    sequence_length = 100  # Simulate 100 tokens per generation
    
    # Each rank creates its "generated" data
    local_data = torch.randn(batch_size_per_rank, sequence_length, device=device)
    print(f"[TEST] Rank {rank}: Created local data shape: {local_data.shape}")
    
    # Synchronize before gather
    print(f"[TEST] Rank {rank}: Synchronizing before gather...")
    dist.barrier()
    print(f"[TEST] Rank {rank}: Synchronization complete")
    
    # Test 1: Gather all data to rank 0 (similar to what GRPOTrainer does)
    print(f"[TEST] Rank {rank}: Testing gather to rank 0...")
    try:
        if rank == 0:
            # Rank 0 gathers from all ranks
            gathered_list = [torch.zeros_like(local_data) for _ in range(world_size)]
            dist.gather(local_data, gathered_list, dst=0)
            print(f"[TEST] Rank {rank}: Gather successful! Received {len(gathered_list)} tensors")
            for i, tensor in enumerate(gathered_list):
                print(f"[TEST] Rank {rank}: Tensor from rank {i}: shape={tensor.shape}, sum={tensor.sum().item():.2f}")
        else:
            # Other ranks send their data
            dist.gather(local_data, dst=0)
            print(f"[TEST] Rank {rank}: Sent data to rank 0")
        
        # Synchronize after gather
        dist.barrier()
        print(f"[TEST] Rank {rank}: Gather operation completed successfully")
    except Exception as e:
        print(f"[ERROR] Rank {rank}: Gather operation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test 2: All-gather (alternative gather method)
    print(f"\n[TEST] Rank {rank}: Testing all_gather (all ranks get all data)...")
    try:
        # All-gather: all ranks receive data from all ranks
        gathered_tensors = [torch.zeros_like(local_data) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, local_data)
        print(f"[TEST] Rank {rank}: All-gather successful! Received {len(gathered_tensors)} tensors")
        for i, tensor in enumerate(gathered_tensors):
            print(f"[TEST] Rank {rank}: Tensor from rank {i}: shape={tensor.shape}, sum={tensor.sum().item():.2f}")
        
        # Synchronize after all-gather
        dist.barrier()
        print(f"[TEST] Rank {rank}: All-gather operation completed successfully")
    except Exception as e:
        print(f"[ERROR] Rank {rank}: All-gather operation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test 3: Gather with larger data (simulating longer sequences)
    print(f"\n[TEST] Rank {rank}: Testing gather with larger data (simulating longer sequences)...")
    try:
        large_sequence_length = 1536  # Similar to max_completion_length in GRPO
        large_local_data = torch.randn(batch_size_per_rank, large_sequence_length, device=device)
        print(f"[TEST] Rank {rank}: Created large local data shape: {large_local_data.shape}")
        
        dist.barrier()
        
        if rank == 0:
            large_gathered_list = [torch.zeros_like(large_local_data) for _ in range(world_size)]
            dist.gather(large_local_data, large_gathered_list, dst=0)
            print(f"[TEST] Rank {rank}: Large gather successful! Total data size: {sum(t.numel() for t in large_gathered_list)} elements")
        else:
            dist.gather(large_local_data, dst=0)
            print(f"[TEST] Rank {rank}: Sent large data to rank 0")
        
        dist.barrier()
        print(f"[TEST] Rank {rank}: Large gather operation completed successfully")
    except Exception as e:
        print(f"[ERROR] Rank {rank}: Large gather operation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test 4: Multiple sequential gathers (simulating multiple batches)
    print(f"\n[TEST] Rank {rank}: Testing multiple sequential gathers (simulating batch processing)...")
    try:
        num_batches = 3
        for batch_idx in range(num_batches):
            print(f"[TEST] Rank {rank}: Processing batch {batch_idx + 1}/{num_batches}...")
            batch_data = torch.randn(batch_size_per_rank, sequence_length, device=device)
            
            dist.barrier()
            
            if rank == 0:
                batch_gathered = [torch.zeros_like(batch_data) for _ in range(world_size)]
                dist.gather(batch_data, batch_gathered, dst=0)
                print(f"[TEST] Rank {rank}: Batch {batch_idx + 1} gather successful")
            else:
                dist.gather(batch_data, dst=0)
                print(f"[TEST] Rank {rank}: Sent batch {batch_idx + 1} to rank 0")
            
            dist.barrier()
            print(f"[TEST] Rank {rank}: Batch {batch_idx + 1} completed")
        
        print(f"[TEST] Rank {rank}: All sequential gathers completed successfully")
    except Exception as e:
        print(f"[ERROR] Rank {rank}: Sequential gather operation failed at batch: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main test function."""
    print(f"[TEST] Starting node synchronization test at {datetime.datetime.now()}")
    print(f"[TEST] Python version: {os.sys.version}")
    print(f"[TEST] PyTorch version: {torch.__version__}")
    print(f"[TEST] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[TEST] CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[TEST] CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        # Setup distributed training
        rank, world_size, local_rank, device = setup_distributed()
        
        print(f"\n[TEST] Rank {rank}/{world_size}: Starting tests...")
        
        # Test gather operations
        test_gather_operation(rank, world_size, device)
        
        print(f"\n[TEST] Rank {rank}: All tests completed successfully!")
        
        # Final synchronization
        dist.barrier()
        if rank == 0:
            print(f"\n[TEST] ========================================")
            print(f"[TEST] ALL TESTS PASSED!")
            print(f"[TEST] Node synchronization and gather operations work correctly")
            print(f"[TEST] ========================================")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[TEST] Process group destroyed")


if __name__ == "__main__":
    main()


