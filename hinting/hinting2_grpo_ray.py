#!/usr/bin/env python3
"""
GRPO training script for Omni-MATH dataset using verl with Ray for distributed training.
Configured for 2 nodes, each with 8 H100 GPUs (16 GPUs total).
Uses Hydra to properly load verl's default configs and override specific values.
"""

import os
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

HINT_LEVEL = 0

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
MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
DATASET_NAME = "../newopenaioutputs/hints_dataset"  # Omni-MATH dataset
MAX_NUM = None  # Limit dataset to last MAX_NUM rows (None = use all data). Useful for testing.

# Distributed training configuration
# 2 nodes, 8 GPUs per node = 16 GPUs total
NUM_NODES = 1
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

def create_rl_dataset(tokenizer, dataset_path: str, max_samples: Optional[int] = None, val_size: int = 64, max_prompt_tokens: int = 2560, hint_level: int = 0):
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
        original_size = len(formatted_dataset)
        start_idx = max(0, len(formatted_dataset) - max_samples)
        formatted_dataset = formatted_dataset.select(range(start_idx, len(formatted_dataset)))
        print(f"[INFO] Limited dataset from {original_size} to {len(formatted_dataset)} rows (using last {max_samples} rows)")
    
    
    
    
    
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


def main():
    """Main function to run GRPO training with verl and Ray."""
    # Generate timestamp and experiment name for this run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    special_name = f"hint_level_{HINT_LEVEL}"
    experiment_name = f"grpo_omni_math_{special_name}_{timestamp}"
    
    # Construct output directory with timestamp and special_name
    model_name = f"qwen2-math-7b_{special_name}_{timestamp}"
    output_dir = os.path.join(bolt.ARTIFACT_DIR, model_name)
    
    print("=" * 80)
    print("GRPO Training with verl and Ray")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {output_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Nodes: {NUM_NODES}, GPUs per node: {GPUS_PER_NODE}")
    print("=" * 80)
    
    # Load tokens and login to services
    tokens = load_tokens()
    login_huggingface(tokens)
    login_wandb(tokens)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create RL dataset with train/val split
    print("Creating RL dataset...")
    train_data, val_data = create_rl_dataset(tokenizer, DATASET_NAME, max_samples=MAX_NUM, val_size=64, hint_level=HINT_LEVEL)  # 64 samples for validation
    print(f"Created {len(train_data)} training samples, {len(val_data)} validation samples")
    
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
    overrides = [
        # Model path
        f"actor_rollout_ref.model.path={MODEL_PATH}",
        "actor_rollout_ref.model.trust_remote_code=true",
        
        # Rollout config for GRPO - using vLLM
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=4",  # Multiple generations for GRPO
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.9",
        "actor_rollout_ref.rollout.prompt_length=2560",
        "actor_rollout_ref.rollout.response_length=1536",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.load_format=auto",
        "actor_rollout_ref.rollout.enforce_eager=true",
        
        # Ray cluster config - connect to existing cluster
        "++ray_kwargs.ray_init.address=auto",
        
        # Actor config
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",  # 2 nodes x 8 GPUs
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
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
        "data.train_batch_size=64",  # 2 nodes x 8 GPUs = 16 GPUs
        "data.val_batch_size=32",
        
        # Trainer config
        f"trainer.project_name=grpo-omni-math",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.nnodes={NUM_NODES}",
        f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.total_epochs=5",
        "trainer.save_freq=500",  # Effectively disable checkpointing (no shared filesystem)
        "trainer.val_before_train=false",  # Skip validation before training (too slow)
        "trainer.test_freq=50",  # Validate every 50 training steps
        "trainer.log_val_generations=3",  # Log N validation samples to wandb
        
        # Custom reward function (use ++ to add or override)
        f"++custom_reward_function.path={__file__}",
        "++custom_reward_function.name=compute_score",
        
        # Disable reward model loop (use custom function)
        "++reward_model.enable=false",
        
        # Disable critic (not needed for GRPO)
        "++critic.enable=false",
        
        # Custom script parameters (for wandb logging)
        f"++custom_params.hint_level={HINT_LEVEL}",
        f"++custom_params.system_prompt_name={SYSTEM_PROMPT_NAME}",
        f"++custom_params.max_samples={MAX_NUM}",
        f"++custom_params.dataset_path={DATASET_NAME}",
    ]
    
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
