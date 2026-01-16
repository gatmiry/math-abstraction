#!/usr/bin/env python3
"""
GRPO training script for Omni-MATH dataset using verl with Ray for distributed training.
Configured for 2 nodes, each with 8 H100 GPUs (16 GPUs total).
Uses Hydra to properly load verl's default configs and override specific values.
"""

import os
import argparse
import apple_bolt as bolt

# vLLM 0.8.5 V1 engine works fine, but keep this for compatibility with older versions
os.environ.setdefault("VLLM_USE_V1", "0")

# Tell wandb to save code with each run
os.environ["WANDB_SAVE_CODE"] = "true"

# Token file path (not tracked by git)
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "hf_token.txt")

# System prompt file path
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt.txt")

# Which system prompt to use (from system_prompt.txt)
SYSTEM_PROMPT_NAME = "default"

HINT_LEVEL = -1
VAL_SIZE = 512

def load_system_prompt(name: str = None):
    """Load a named system prompt from file.
    
    File format: Multiple prompts separated by ===PROMPT: name=== headers.
    
    Args:
        name: Name of the prompt to load. If None, uses SYSTEM_PROMPT_NAME.
    
    Returns:
        The system prompt string.
    """
    if name is None:
        name = SYSTEM_PROMPT_NAME
    
    with open(SYSTEM_PROMPT_FILE, 'r') as f:
        content = f.read()
    
    # Parse prompts from file
    prompts = {}
    current_name = None
    current_lines = []
    
    for line in content.split('\n'):
        if line.startswith('===PROMPT:') and line.endswith('==='):
            # Save previous prompt if exists
            if current_name is not None:
                prompts[current_name] = '\n'.join(current_lines).strip()
            # Start new prompt
            current_name = line[10:-3].strip()
            current_lines = []
        else:
            current_lines.append(line)
    
    # Save last prompt
    if current_name is not None:
        prompts[current_name] = '\n'.join(current_lines).strip()
    
    if name not in prompts:
        available = list(prompts.keys())
        raise ValueError(f"System prompt '{name}' not found. Available: {available}")
    
    return prompts[name]


def load_tokens():
    """Load tokens from file. Format: KEY=VALUE per line."""
    tokens = {}
    if not os.path.exists(TOKEN_FILE):
        print(f"[WARNING] Token file not found: {TOKEN_FILE}")
        return tokens
    
    with open(TOKEN_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                tokens[key.strip()] = value.strip()
    
    return tokens


def login_huggingface(tokens):
    """Login to HuggingFace using token from dict."""
    token = tokens.get('HF_TOKEN')
    if not token:
        print("[WARNING] HF_TOKEN not found in token file")
        return False
    
    # Set environment variable for transformers/datasets
    os.environ["HF_TOKEN"] = token
    
    # Login via huggingface_hub
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[INFO] Successfully logged in to HuggingFace")
        return True
    except Exception as e:
        print(f"[WARNING] HuggingFace login failed: {e}")
        return False


def login_wandb(tokens):
    """Login to Weights & Biases using API key from dict."""
    api_key = tokens.get('WANDB_API_KEY')
    if not api_key or api_key == 'YOUR_WANDB_API_KEY_HERE':
        print("[WARNING] WANDB_API_KEY not configured in token file")
        return False
    
    # Set environment variable for wandb
    os.environ["WANDB_API_KEY"] = api_key
    
    try:
        import wandb
        wandb.login(key=api_key, relogin=True)
        print("[INFO] Successfully logged in to Wandb")
        return True
    except Exception as e:
        print(f"[WARNING] Wandb login failed: {e}")
        return False

# NCCL P2P workaround for hardware issues with NVLink
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

import re
import sys
import ray
import tempfile
import datetime
import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import load_from_disk
from typing import List, Dict, Optional, Any
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
# Ensure Ray runtime_env carries necessary env vars
try:
    from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
    PPO_RAY_RUNTIME_ENV.setdefault("env_vars", {})
    PPO_RAY_RUNTIME_ENV["env_vars"]["VLLM_USE_V1"] = "0"
    # NCCL P2P workaround for hardware issues
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_IGNORE_DISABLED_P2P"] = "1"
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_P2P_DISABLE"] = "1"
except ImportError:
    pass  # verl 0.4.x may not have this

# Configuration
DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "../newopenaioutputs/hints_dataset"  # Omni-MATH dataset
MAX_NUM = None  # Limit dataset to last MAX_NUM rows (None = use all data). Useful for testing.

# Training hyperparameters
TRAIN_BATCH_SIZE = 256
TOTAL_EPOCHS = 50
TEST_BATCH_SIZE = 256


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GRPO Training for math problem solving")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a local HuggingFace-format model to start training from. "
             "Must contain tokenizer files (vocab.json, etc.). "
             "If not provided, uses the default HuggingFace model."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a verl checkpoint directory to resume training from. "
             "e.g., /path/to/model_dir/global_step_50"
    )
    parser.add_argument(
        "--hint-level",
        type=int,
        default=None,
        help="Which hint level to use (overrides HINT_LEVEL constant)"
    )
    return parser.parse_args()

# Distributed training configuration
# 2 nodes, 8 GPUs per node = 16 GPUs total
NUM_NODES = 4
GPUS_PER_NODE = 8


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} at the end of text."""
    # Find the last \\boxed{ and extract content handling nested braces
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None
    
    start_pos = matches[-1].end()
    depth = 1
    i = start_pos
    
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return text[start_pos:i-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    return answer.strip().lower()


def compute_score(
    data_source: str = None,
    solution_str: str = None,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """Compute reward score for a single solution (verl 0.4.1 format).
    
    This function is called by verl's NaiveRewardManager for each sample.
    
    Args:
        data_source: Data source identifier (not used here)
        solution_str: Generated solution string
        ground_truth: Ground truth answer (from dataset)
        extra_info: Extra info dictionary
        **kwargs: Additional keyword arguments
    
    Returns:
        Reward score (1.0 if answer matches, 0.0 otherwise)
    """
    if ground_truth is None:
        return 0.0
    
    # Extract boxed answer from solution
    boxed_answer = extract_boxed_answer(solution_str)
    boxed_normalized = normalize_answer(boxed_answer)
    answer_normalized = normalize_answer(ground_truth)
    
    # Compare normalized answers
    if boxed_normalized and answer_normalized and boxed_normalized == answer_normalized:
        return 1.0
    else:
        return 0.0


def format_prompt(problem: str, partial_proof: str, hint: str = None, system_prompt: str = None) -> List[Dict[str, str]]:
    if system_prompt is None:
        if hint:
            system_prompt = load_system_prompt('default')
        else:
            system_prompt = load_system_prompt('onlypartialproof')

    if hint:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\n\nPartial proof: {partial_proof}\n\nHint: {hint}"
            }
        ]
        return messages
    
    else:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\n\nPartial proof: {partial_proof}"
            }
        ]
        return messages

def create_rl_dataset(tokenizer, dataset_path: str, max_samples: Optional[int] = None, val_size: int = 128, max_prompt_tokens: int = 2560, hint_level: int = 0):
    """Create RL dataset in verl format with train/val split.
    
    Args:
        tokenizer: Tokenizer instance
        dataset_path: Path to dataset
        max_samples: Maximum number of samples (None = use all)
        val_size: Number of samples for validation (default 64)
        max_prompt_tokens: Maximum prompt length in tokens (default 2560)
    
    Returns:
        Tuple of (train_data, val_data) in verl format
    """
    import json
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Format dataset - store messages as JSON for verl to parse
    def format_dataset(examples, is_train: bool = True, hint_level: int = 0):
        """Format dataset examples for GRPO training."""
        prompts = []
        answers = []
        for problem, partial_proof, final_answer, hints in zip(
            examples['problem'], examples['partial_proof'], examples['final_answer'], examples['hints']
        ):
            # Only include if final answer exists AND hints is non-empty
            if final_answer and hints:
                # Store messages as list of dicts - verl will apply chat template
                if not is_train:
                    # Validation: no hints provided
                    messages = format_prompt(problem, partial_proof)
                    prompts.append(messages)
                    answers.append(final_answer)
                
                elif hint_level >= 0:
                    if len(hints) > hint_level:
                        messages = format_prompt(problem, partial_proof, hints[-1 - hint_level])
                        prompts.append(messages)
                        answers.append(final_answer)
                else:
                    # hint_level < 0: use all hints
                    for hint in hints:
                        messages = format_prompt(problem, partial_proof, hint)
                        prompts.append(messages)
                        answers.append(final_answer)
                

        
        return {"prompt": prompts, "answer": answers}
    
    # Split into train and val
    

    # Filter out None answers
    #formatted_dataset = formatted_dataset.filter(lambda x: x['answer'] is not None)
    
    # Filter out prompts that are too long
    def is_prompt_short_enough(example):
        """Check if prompt fits within max_prompt_tokens."""
        prompt_text = tokenizer.apply_chat_template(example['prompt'], tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer.encode(prompt_text))
        return token_count <= max_prompt_tokens
    
    
    # Limit dataset to last MAX_NUM rows if specified
    if max_samples is not None and max_samples > 0:
        original_size = len(dataset)
        start_idx = max(0, len(dataset) - max_samples)
        dataset = dataset.select(range(start_idx, len(dataset)))
        print(f"[INFO] Limited dataset from {original_size} to {len(dataset)} rows (using last {max_samples} rows)")
    
    
    
    
    
    total_size = len(dataset)
    val_size_actual = min(val_size, total_size)  # Don't exceed dataset size
    train_size = total_size - val_size_actual
    print(f"[INFO] Split dataset: {train_size} train, {val_size_actual} val")
    

    train_dataset = dataset.select(range(0, train_size))
    val_dataset = dataset.select(range(train_size, total_size))
    # Process dataset
    from functools import partial
    train_dataset = train_dataset.map(
        partial(format_dataset, is_train=True, hint_level=hint_level),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        partial(format_dataset, is_train=False),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    original_count = len(train_dataset)
    train_dataset = train_dataset.filter(is_prompt_short_enough)
    filtered_count = original_count - len(train_dataset)
    if filtered_count > 0:
        print(f"[INFO] Filtered out {filtered_count} samples with prompts > {max_prompt_tokens} tokens for train dataset")
    original_count = len(val_dataset)
    val_dataset = val_dataset.filter(is_prompt_short_enough)
    filtered_count = original_count - len(val_dataset)
    if filtered_count > 0:
        print(f"[INFO] Filtered out {filtered_count} samples with prompts > {max_prompt_tokens} tokens for test dataset")

    # Convert to verl format
    # verl expects ground_truth nested under reward_model
    def to_verl_format(dataset_split):
        rl_data = []
        for item in dataset_split:
            rl_data.append({
                "prompt": item["prompt"],  # List of message dicts with 'role' and 'content'
                "reward_model": {
                    "ground_truth": item["answer"],
                },
                "data_source": "omni_math",  # Identifier for reward function
            })
        return rl_data
    
    train_data = to_verl_format(train_dataset)
    val_data = to_verl_format(val_dataset)
    
    return train_data, val_data


def get_verl_config_path():
    """Get path to verl's config directory."""
    import verl
    verl_path = os.path.dirname(verl.__file__)
    config_path = os.path.join(verl_path, "trainer", "config")
    return config_path


def distribute_checkpoint_to_all_nodes(checkpoint_dir: str, target_dir: str):
    """
    Distribute checkpoint shards from head node to all worker nodes.
    
    This is needed because verl's FSDP checkpoint loading expects each node
    to have ALL shards available locally.
    
    Args:
        checkpoint_dir: Path to the gathered checkpoint (with all shards)
        target_dir: Path where checkpoint should be placed on all nodes
    
    Returns:
        True if successful
    """
    actor_dir = os.path.join(checkpoint_dir, "actor")
    target_actor_dir = os.path.join(target_dir, "actor")
    
    # Read all checkpoint files from head node
    print(f"[Distribute] Reading checkpoint from {actor_dir}...")
    files_data = {}
    for filename in os.listdir(actor_dir):
        if filename.endswith(".pt"):
            filepath = os.path.join(actor_dir, filename)
            with open(filepath, 'rb') as f:
                files_data[filename] = f.read()
            print(f"[Distribute] Read {filename} ({len(files_data[filename]) // 1024 // 1024} MB)")
    
    # Also read data.pt if it exists
    data_pt_path = os.path.join(checkpoint_dir, "data.pt")
    data_pt_content = None
    if os.path.exists(data_pt_path):
        with open(data_pt_path, 'rb') as f:
            data_pt_content = f.read()
        print(f"[Distribute] Read data.pt")
    
    print(f"[Distribute] Total files to distribute: {len(files_data)}")
    
    # Define Ray remote function to write files on each node
    @ray.remote
    def write_checkpoint_files(files: dict, target_path: str, data_pt: bytes = None):
        """Write checkpoint files to this node."""
        import os
        os.makedirs(target_path, exist_ok=True)
        
        written = 0
        for filename, data in files.items():
            filepath = os.path.join(target_path, filename)
            with open(filepath, 'wb') as f:
                f.write(data)
            written += 1
        
        # Write data.pt to parent directory
        if data_pt is not None:
            parent_dir = os.path.dirname(target_path)
            data_pt_path = os.path.join(parent_dir, "data.pt")
            with open(data_pt_path, 'wb') as f:
                f.write(data_pt)
        
        import socket
        return f"{socket.gethostname()}: wrote {written} files to {target_path}"
    
    # Get all nodes
    nodes = ray.nodes()
    node_ips = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]
    print(f"[Distribute] Distributing to {len(node_ips)} nodes: {node_ips}")
    
    # Schedule write tasks on each node
    tasks = []
    for node_ip in node_ips:
        task = write_checkpoint_files.options(
            resources={f"node:{node_ip}": 0.001}
        ).remote(files_data, target_actor_dir, data_pt_content)
        tasks.append((node_ip, task))
    
    # Wait for all writes to complete
    for node_ip, task in tasks:
        try:
            result = ray.get(task, timeout=600)  # 10 min timeout for large files
            print(f"[Distribute] {result}")
        except Exception as e:
            print(f"[Distribute] Failed on {node_ip}: {e}")
            return False
    
    print(f"[Distribute] ✓ Checkpoint distributed to all nodes at: {target_dir}")
    return True


def gather_checkpoint_to_head_node(checkpoint_dir: str, output_dir: str, num_nodes: int, gpus_per_node: int):
    """
    Gather all FSDP checkpoint shards from worker nodes to the head node.
    
    This keeps the checkpoint in verl's native format so it can be:
    - Used directly with --resume-from for continued training
    - Loaded with verl's inference utilities
    
    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g., /path/to/global_step_50)
        output_dir: Where to save the gathered checkpoint on head node
        num_nodes: Number of nodes used in training
        gpus_per_node: GPUs per node
    
    Returns:
        True if successful, False otherwise
    """
    import shutil
    
    world_size = num_nodes * gpus_per_node
    actor_dir = os.path.join(checkpoint_dir, "actor")
    output_actor_dir = os.path.join(output_dir, "actor")
    
    print(f"[Gather] Collecting checkpoint shards from {num_nodes} nodes...")
    print(f"[Gather] World size: {world_size}")
    print(f"[Gather] Source: {actor_dir}")
    print(f"[Gather] Destination: {output_dir}")
    
    # Create output directory
    os.makedirs(output_actor_dir, exist_ok=True)
    
    # Define Ray remote function to read checkpoint files from each node
    @ray.remote
    def get_checkpoint_files(checkpoint_path: str):
        """Get list of checkpoint files and their contents from this node."""
        import os
        
        files_data = {}
        if os.path.exists(checkpoint_path):
            for filename in os.listdir(checkpoint_path):
                if filename.endswith(".pt"):
                    filepath = os.path.join(checkpoint_path, filename)
                    # Read file as bytes
                    with open(filepath, 'rb') as f:
                        files_data[filename] = f.read()
                    print(f"[Node] Read {filename} ({len(files_data[filename])} bytes)")
        return files_data
    
    # Get list of all node IPs from Ray
    nodes = ray.nodes()
    node_ips = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]
    print(f"[Gather] Found {len(node_ips)} alive nodes: {node_ips}")
    
    # Schedule gather tasks on each node
    gather_tasks = []
    for node_ip in node_ips:
        # Try to run on specific node using scheduling strategy
        task = get_checkpoint_files.options(
            resources={f"node:{node_ip}": 0.001}
        ).remote(actor_dir)
        gather_tasks.append((node_ip, task))
    
    # Also try without node constraint as fallback (will run on any available node)
    # This helps if node labels aren't set up
    fallback_task = get_checkpoint_files.remote(actor_dir)
    
    # Collect results
    all_files = {}
    
    # First try node-specific tasks
    for node_ip, task in gather_tasks:
        try:
            result = ray.get(task, timeout=300)  # 5 min timeout
            for filename, data in result.items():
                if filename not in all_files:
                    all_files[filename] = data
                    print(f"[Gather] Got {filename} from {node_ip}")
        except Exception as e:
            print(f"[Gather] Failed to get files from {node_ip}: {e}")
    
    # Try fallback if we don't have all files
    expected_model_files = world_size
    model_files_count = sum(1 for f in all_files if f.startswith("model_world_size_"))
    
    if model_files_count < expected_model_files:
        print(f"[Gather] Only got {model_files_count}/{expected_model_files} model shards, trying fallback...")
        try:
            result = ray.get(fallback_task, timeout=300)
            for filename, data in result.items():
                if filename not in all_files:
                    all_files[filename] = data
                    print(f"[Gather] Got {filename} from fallback")
        except Exception as e:
            print(f"[Gather] Fallback also failed: {e}")
    
    # Write all collected files to output directory
    print(f"[Gather] Writing {len(all_files)} files to {output_actor_dir}")
    for filename, data in all_files.items():
        output_path = os.path.join(output_actor_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(data)
        print(f"[Gather] Wrote {filename}")
    
    # Copy the latest_checkpointed_iteration.txt if it exists
    parent_dir = os.path.dirname(checkpoint_dir)
    iteration_file = os.path.join(parent_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(iteration_file):
        output_parent = os.path.dirname(output_dir)
        os.makedirs(output_parent, exist_ok=True)
        shutil.copy2(iteration_file, os.path.join(output_parent, "latest_checkpointed_iteration.txt"))
        print(f"[Gather] Copied latest_checkpointed_iteration.txt")
    
    # Verify we got all shards
    model_files = [f for f in all_files if f.startswith("model_world_size_")]
    optim_files = [f for f in all_files if f.startswith("optim_world_size_")]
    extra_files = [f for f in all_files if f.startswith("extra_state_world_size_")]
    
    print(f"\n[Gather] Summary:")
    print(f"  - Model shards: {len(model_files)}/{world_size}")
    print(f"  - Optim shards: {len(optim_files)}/{world_size}")
    print(f"  - Extra state shards: {len(extra_files)}/{world_size}")
    
    if len(model_files) == world_size:
        print(f"\n✓ Successfully gathered complete checkpoint to: {output_dir}")
        print(f"  Use with: python hinting2_grpo_ray.py --resume-from {output_dir}")
        return True
    else:
        missing_ranks = set(range(world_size)) - {int(f.split("_")[-1].replace(".pt", "")) for f in model_files}
        print(f"\n✗ Incomplete checkpoint - missing ranks: {sorted(missing_ranks)}")
        print(f"  The missing shards may be on nodes that are no longer accessible.")
        return False


def main():
    """Main function to run GRPO training with verl and Ray."""
    # Parse command line arguments
    args = parse_args()
    
    # Determine model path (command line arg or default)
    # --model-path: Use a different base model (must have tokenizer files)
    # --resume-from: Resume from a verl checkpoint (uses default model + loads weights)
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_PATH
    
    # Determine hint level (command line arg or constant)
    hint_level = args.hint_level if args.hint_level is not None else HINT_LEVEL
    
    # Extract base model name for output naming
    if args.resume_from:
        # Resuming from verl checkpoint
        path_parts = args.resume_from.rstrip('/').split('/')
        # Find the main model directory (skip "global_step_X", etc.)
        base_model_name = None
        for part in reversed(path_parts):
            if part and not part.startswith('global_step'):
                base_model_name = part
                break
        if not base_model_name:
            base_model_name = "checkpoint"
    elif args.model_path:
        # Using a different base model
        path_parts = args.model_path.rstrip('/').split('/')
        base_model_name = path_parts[-1].lower() if path_parts else "custom"
    else:
        # Using default HuggingFace model
        base_model_name = model_path.split('/')[-1].lower()
    
    # Generate timestamp and experiment name for this run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    special_name = f"hint_level_{hint_level}"
    
    # Add "resumed" to name if loading from checkpoint
    if args.resume_from:
        special_name = f"{special_name}_resumed"
    
    experiment_name = f"grpo_omni_math_{special_name}_{timestamp}"
    
    # Construct output directory with timestamp, special_name, and base model indicator
    model_name = f"qwen3-4b-instruct-2507_{special_name}_{timestamp}_from_{base_model_name}"
    output_dir = os.path.join(bolt.ARTIFACT_DIR, model_name)
    
    print("=" * 80)
    print("GRPO Training with verl and Ray")
    print("=" * 80)
    print(f"Model: {model_path}")
    if args.resume_from:
        print(f"  Resuming from checkpoint: {args.resume_from}")
    elif args.model_path:
        print(f"  (Using custom model path)")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {output_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Hint Level: {hint_level}")
    print(f"Nodes: {NUM_NODES}, GPUs per node: {GPUS_PER_NODE}")
    print("=" * 80)
    
    # Load tokens and login to services
    tokens = load_tokens()
    login_huggingface(tokens)
    login_wandb(tokens)
    
    # Load tokenizer (use default model for tokenizer if resuming from checkpoint)
    print("Loading tokenizer...")
    tokenizer_path = DEFAULT_MODEL_PATH  # Always use base model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create RL dataset with train/val split
    print("Creating RL dataset...")
    train_data, val_data = create_rl_dataset(tokenizer, DATASET_NAME, max_samples=MAX_NUM, val_size=VAL_SIZE, hint_level=hint_level)
    print(f"Created {len(train_data)} training samples, {len(val_data)} validation samples")
    
    # Calculate total training steps (same formula verl uses)
    num_batches_per_epoch = len(train_data) // TRAIN_BATCH_SIZE
    total_training_steps = num_batches_per_epoch * TOTAL_EPOCHS
    print(f"Calculated total_training_steps: {num_batches_per_epoch} batches/epoch × {TOTAL_EPOCHS} epochs = {total_training_steps}")
    
    # Save train dataset to parquet file
    print("Saving datasets to parquet...")
    df_train = pd.DataFrame(train_data)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='_train.parquet', delete=False, dir='/mnt/tmp') as f:
        df_train.to_parquet(f.name, index=False)
        train_dataset_file = f.name
    print(f"Train dataset saved to {train_dataset_file}")
    
    # Save val dataset to parquet file
    df_val = pd.DataFrame(val_data)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='_val.parquet', delete=False, dir='/mnt/tmp') as f:
        df_val.to_parquet(f.name, index=False)
        val_dataset_file = f.name
    print(f"Val dataset saved to {val_dataset_file}")
    
    # Load verl's default config using Hydra
    print("Loading verl configuration with Hydra...")
    config_path = get_verl_config_path()
    
    # Clear any previous Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with verl's config directory
    initialize_config_dir(config_dir=config_path, version_base=None)
    
    # Use Hydra's compose to load defaults and apply overrides
    # Always use base model for tokenizer (checkpoints don't have tokenizer files)
    tokenizer_path = DEFAULT_MODEL_PATH
    
    overrides = [
        # Model path
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=true",
        
        # Tokenizer path (always use base model, checkpoints don't have tokenizer)
        f"data.tokenizer={tokenizer_path}",
        f"reward_model.model.input_tokenizer={tokenizer_path}",
        f"critic.model.tokenizer_path={tokenizer_path}",
        
        # Rollout config for GRPO - using vLLM
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=4",  # Multiple generations for GRPO
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        "actor_rollout_ref.rollout.prompt_length=2560",
        "actor_rollout_ref.rollout.response_length=4096",
        "actor_rollout_ref.rollout.max_model_len=6656",
        "actor_rollout_ref.rollout.max_num_batched_tokens=6656",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.load_format=auto",
        "actor_rollout_ref.rollout.enforce_eager=true",
        
        # Ray cluster config - connect to existing cluster
        "++ray_kwargs.ray_init.address=auto",
        
        # Actor config
        f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_BATCH_SIZE}",  # Must be <= train_batch_size
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.ppo_epochs=1",
        
        # Algorithm - GRPO
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.norm_adv_by_std_in_grpo=true",
        
        # Data config
        f"data.train_files={train_dataset_file}",
        f"data.val_files={val_dataset_file}",
        "data.prompt_key=prompt",
        "data.max_prompt_length=2560",
        "data.max_response_length=1536",
        f"data.train_batch_size={TRAIN_BATCH_SIZE}",
        f"data.val_batch_size={TEST_BATCH_SIZE}",
        
        # Trainer config
        f"trainer.project_name=grpo-omni-math",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.nnodes={NUM_NODES}",
        f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.total_epochs={TOTAL_EPOCHS}",
        "trainer.save_freq=50",  # Save checkpoint every N steps
        "trainer.val_before_train=true",  # Skip validation before training (too slow)
        "trainer.test_freq=5",  # Validate every 50 training steps
        "trainer.log_val_generations=3",  # Log N validation samples to wandb
        
        # Custom reward function (use ++ to add or override)
        f"++custom_reward_function.path={__file__}",
        "++custom_reward_function.name=compute_score",
        
        # Disable reward model loop (use custom function)
        "++reward_model.enable=false",
        
        # Disable critic (not needed for GRPO)
        "++critic.enable=false",
        
        # Custom script parameters (for wandb logging)
        f"++custom_params.hint_level={hint_level}",
        f"++custom_params.system_prompt_name={SYSTEM_PROMPT_NAME}",
        f"++custom_params.max_samples={MAX_NUM}",
        f"++custom_params.dataset_path={DATASET_NAME}",
        f"++custom_params.val_size={VAL_SIZE}",
        f"++custom_params.model_path={model_path}",
        f"++custom_params.resumed_from_checkpoint={args.resume_from is not None}",
    ]
    
    # Add resume_from_path if resuming from checkpoint
    if args.resume_from:
        # Validate that the path contains global_step_ (required by verl)
        if "global_step_" not in args.resume_from:
            print(f"[ERROR] --resume-from path must contain 'global_step_X' directory")
            print(f"        Given: {args.resume_from}")
            print(f"        Example: {args.resume_from}/global_step_50")
            sys.exit(1)
        
        # Initialize Ray early to distribute checkpoint to all nodes
        print("[INFO] Initializing Ray for checkpoint distribution...")
        if not ray.is_initialized():
            ray.init(address='auto')
        
        # Distribute checkpoint from head node to all worker nodes
        # This is necessary because verl expects ALL shards to be available on each node
        print(f"[INFO] Distributing checkpoint to all nodes...")
        
        success = distribute_checkpoint_to_all_nodes(
            checkpoint_dir=args.resume_from,
            target_dir=args.resume_from  # Same path on all nodes
        )
        
        if not success:
            print("[ERROR] Failed to distribute checkpoint to all nodes")
            sys.exit(1)
        
        # Must set resume_mode to 'resume_path' for verl to use the checkpoint
        overrides.append("trainer.resume_mode=resume_path")
        overrides.append(f"trainer.resume_from_path={args.resume_from}")
        
        # Extract step number from checkpoint path and set total_training_steps
        # This is needed because verl calculates total_training_steps = dataset_size * epochs
        # and training exits when global_steps >= total_training_steps
        step_match = re.search(r'global_step_(\d+)', args.resume_from)
        if step_match:
            resume_step = int(step_match.group(1))
            # When resuming, add another full training cycle to the resume step
            new_total = resume_step + total_training_steps
            print(f"[INFO] Resuming from step {resume_step}, adding {total_training_steps} more steps")
            print(f"[INFO] New total_training_steps: {new_total}")
            overrides.append(f"trainer.total_training_steps={new_total}")
        
        print(f"[INFO] Will resume from checkpoint: {args.resume_from}")
    
    try:
        config = compose(config_name="ppo_trainer", overrides=overrides)
    except Exception as e:
        print(f"Error composing config: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Configuration loaded successfully")
    print(f"Config summary:")
    print(f"  - Model: {config.actor_rollout_ref.model.path}")
    print(f"  - Adv estimator: {config.algorithm.adv_estimator}")
    print(f"  - Train batch size: {config.data.train_batch_size}")
    print(f"  - Nodes: {config.trainer.nnodes}, GPUs/node: {config.trainer.n_gpus_per_node}")
    
    # Note: Ray will be initialized by verl's run_ppo with proper config
    # including address='auto' and VLLM_USE_V1=0 in runtime_env
    
    # Import run_ppo here to avoid import issues
    from verl.trainer.main_ppo import run_ppo
    
    # Run PPO training
    print("Starting GRPO training with verl...")
    try:
        run_ppo(config)
        print("Training completed successfully!")
        
        # After training, gather checkpoint shards from all nodes to head node
        print("\n" + "=" * 80)
        print("Gathering checkpoint shards to head node...")
        print("=" * 80)
        
        # Find the latest checkpoint
        latest_checkpoint_file = os.path.join(output_dir, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_checkpoint_file):
            with open(latest_checkpoint_file, 'r') as f:
                latest_step = f.read().strip()
            checkpoint_dir = os.path.join(output_dir, f"global_step_{latest_step}")
            
            # Gather to a consolidated directory on head node
            gathered_dir = os.path.join(output_dir, "gathered_checkpoint", f"global_step_{latest_step}")
            
            print(f"Found latest checkpoint at step {latest_step}")
            
            # Gather all shards to head node (keeps verl's native format)
            success = gather_checkpoint_to_head_node(
                checkpoint_dir=checkpoint_dir,
                output_dir=gathered_dir,
                num_nodes=NUM_NODES,
                gpus_per_node=GPUS_PER_NODE
            )
            
            if success:
                print(f"\n✓ Complete checkpoint gathered to: {gathered_dir}")
                print(f"  Resume training with: python hinting2_grpo_ray.py --resume-from {gathered_dir}")
            else:
                print("\n✗ Checkpoint gathering incomplete - some shards may be missing")
                print("  The partial checkpoint is still at the gathered location.")
        else:
            print(f"[WARNING] No checkpoint found at {latest_checkpoint_file}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if os.path.exists(train_dataset_file):
            os.unlink(train_dataset_file)
            print(f"Cleaned up temporary train dataset file: {train_dataset_file}")
        if os.path.exists(val_dataset_file):
            os.unlink(val_dataset_file)
            print(f"Cleaned up temporary val dataset file: {val_dataset_file}")


if __name__ == "__main__":
    main()
