# GRPO Training on Omni-MATH Dataset

This directory contains scripts for training a model using Group Relative Policy Optimization (GRPO) on the Omni-MATH dataset with distributed training across multiple nodes and GPUs.

## Files

- `hinting_grpo.py`: Main training script implementing GRPO on Omni-MATH dataset
- `launch_grpo_2nodes.sh`: Launch script for 2-node, 8-GPU-per-node setup

## Requirements

### Software Dependencies

```bash
pip install torch transformers datasets accelerate
pip install verl  # For distributed RL training (recommended)
# OR
pip install trl   # Alternative (fallback, less optimized for multi-node)
```

### Hardware Configuration

- **2 nodes**, each with **8 H100 GPUs** (16 GPUs total)
- Sufficient network bandwidth between nodes for distributed training
- Shared filesystem or synchronized model/dataset paths across nodes

## Configuration

### Environment Variables

Before running, set these environment variables:

```bash
# Master node (node 0)
export MASTER_ADDR="<master_node_ip>"  # IP address of the master node
export MASTER_PORT="29500"             # Port for communication
export NODE_RANK=0                     # Rank of this node (0 for master, 1 for worker)

# Worker node (node 1)
export MASTER_ADDR="<master_node_ip>"  # Same as master
export MASTER_PORT="29500"             # Same as master
export NODE_RANK=1                     # Rank of this node
```

### Model and Dataset Paths

Update these paths in `hinting_grpo.py`:

- `MODEL_PATH`: Path to the base/finetuned model
- `DATASET_NAME`: HuggingFace dataset name (default: "KbsdJames/Omni-MATH")
- `OUTPUT_DIR`: Where to save the trained model

## Usage

### Option 1: Using the Launch Script (Recommended)

On each node, run:

```bash
# On master node (node 0)
export NODE_RANK=0
export MASTER_ADDR="<master_node_ip>"
bash launch_grpo_2nodes.sh

# On worker node (node 1)
export NODE_RANK=1
export MASTER_ADDR="<master_node_ip>"
bash launch_grpo_2nodes.sh
```

### Option 2: Using torchrun Directly

On each node:

```bash
# Master node
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr=<master_node_ip> \
    --master_port=29500 \
    --node_rank=0 \
    hinting_grpo.py

# Worker node
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr=<master_node_ip> \
    --master_port=29500 \
    --node_rank=1 \
    hinting_grpo.py
```

### Option 3: Using verl's Distributed Launcher (if verl is installed)

If verl provides its own launcher, use it according to verl's documentation.

## How It Works

1. **Dataset Loading**: Loads the Omni-MATH dataset from HuggingFace
2. **Data Formatting**: Formats problems using chat templates
3. **Reward Function**: Compares generated answers (from `\boxed{}`) with ground truth answers
4. **GRPO Training**: Uses Group Relative Policy Optimization to train the model
5. **Distributed Training**: Automatically handles data parallelism across 16 GPUs

## Reward Function

The reward function:
- Extracts the final answer from `\boxed{...}` in the generated completion
- Compares it with the ground truth answer from the dataset's `answer` field
- Returns 1.0 if answers match (after normalization), 0.0 otherwise

## Monitoring

Training logs will be printed by rank 0 (master node). Check:
- Loss values
- Reward statistics
- Model checkpoints saved to `OUTPUT_DIR`

## Troubleshooting

### Connection Issues

- Ensure nodes can communicate (test with `ping` or `nc`)
- Check firewall settings allow the master port
- Verify `MASTER_ADDR` is set correctly on both nodes

### CUDA/GPU Issues

- Verify all GPUs are visible: `nvidia-smi`
- Check CUDA version compatibility
- Ensure sufficient GPU memory (H100 has 80GB, should be sufficient)

### Dataset Loading Issues

- Ensure internet connection for downloading from HuggingFace
- Or pre-download dataset and use `load_from_disk()`

### Memory Issues

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing` (already enabled)
- Use 8-bit optimizer (already configured)

## Notes

- The script automatically detects if verl is available and uses it, otherwise falls back to trl
- For optimal performance with verl, ensure it's properly installed and configured
- Model checkpoints are saved only by rank 0 to avoid conflicts


