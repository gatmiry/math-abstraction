#!/usr/bin/env python3
"""
GRPO training script for Omni-MATH dataset using verl with Ray for distributed training.
Configured for 2 nodes, each with 8 H100 GPUs (16 GPUs total).
Uses Hydra to properly load verl's default configs and override specific values.
"""

import os

# vLLM 0.8.5 V1 engine works fine, but keep this for compatibility with older versions
os.environ.setdefault("VLLM_USE_V1", "0")

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
except ImportError:
    pass  # verl 0.4.x may not have this

# Configuration
MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
DATASET_NAME = "../newopenaioutputs/hints_dataset"  # Omni-MATH dataset
OUTPUT_DIR = "../models/qwen2-math-7b-instruct_grpo_hints_dataset_verl"
MAX_NUM = 128  # Limit dataset to last MAX_NUM rows (None = use all data). Useful for testing.

# Distributed training configuration
# 2 nodes, 8 GPUs per node = 16 GPUs total
NUM_NODES = 2
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


def format_prompt(problem: str, partial_proof: str) -> List[Dict[str, str]]:
    """Format problem as a prompt for the model.
    
    Args:
        problem: Problem statement from the dataset
        partial_proof: Partial proof or solution
    
    Returns:
        List of messages in chat format
    """
    messages = [
        {
            "role": "system",
            "content": """You are learning to solve mathematics problems. You will be given a math problem and a partial proof or solution. Your task is to carefully complete the proof or solution, step by step, providing clear reasoning at each stage (do not skip steps). Only after finishing the complete reasoning, write the final answer at the end, clearly enclosed in the \\box{...} environment as is standard in LaTeX. 

- For each step, show the logical process and all intermediate computations or deductions.
- Only after reasoning is finished, put the final answer at the end, in its own line, using \\box{...}
- Use plain text with embedded LaTeX where mathematical symbols or equations are necessary.

## Output Format

Present your solution as a well-formatted, step-by-step proof or solution in plain text (not as a code block). Mathematical expressions and the boxed answer should use proper LaTeX syntax, e.g. \\box{42}. 

## Example

**Example Input:**  
Problem:
Prove that the derivative of \\(f(x) = x^2\\) is \\(2x\\).  

Partial proof:  
The derivative of \\(f(x)\\) is defined as  
\\[
f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}
\\]

**Example Output:**  
Let's substitute \\(f(x) = x^2\\) into the definition:  
\\[
f'(x) = \\lim_{h \\to 0} \\frac{(x+h)^2 - x^2}{h}
\\]  
Expand \\((x+h)^2\\):  
\\[
(x+h)^2 = x^2 + 2xh + h^2
\\]  
Subtract \\(x^2\\):  
\\[
(x^2 + 2xh + h^2) - x^2 = 2xh + h^2
\\]  
So,  
\\[
f'(x) = \\lim_{h \\to 0} \\frac{2xh + h^2}{h}
\\]  
Divide numerator by \\(h\\):  
\\[
= \\lim_{h \\to 0} (2x + h)
\\]  
Take the limit as \\(h \\to 0\\):  
\\[
= 2x
\\]  

\\box{2x}

---

**Reminders:**  
- Complete the proof step by step, showing all logical reasoning before the boxed answer.
- The final answer must always appear at the end, in \\box{...}. 

**IMPORTANT INSTRUCTION SUMMARY:**  
- Show step-by-step reasoning before the conclusion.
- Place final answer boxed (in \\box{...}) at the end."""
        },
        {
            "role": "user",
            "content": f"Problem: {problem}\n\nPartial proof: {partial_proof}"
        }
    ]
    return messages


def create_rl_dataset(tokenizer, dataset_path: str, max_samples: Optional[int] = None):
    """Create RL dataset in verl format.
    
    Args:
        tokenizer: Tokenizer instance
        dataset_path: Path to dataset
        max_samples: Maximum number of samples (None = use all)
    
    Returns:
        List of data items in verl format
    """
    import json
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Format dataset - store messages as JSON for verl to parse
    def format_dataset(examples):
        """Format dataset examples for GRPO training."""
        prompts = []
        answers = []
        for problem, partial_proof, final_answer in zip(
            examples['problem'], examples['partial_proof'], examples['final_answer']
        ):
            if final_answer:  # Only include if final answer exists
                # Store messages as list of dicts - verl will apply chat template
                messages = format_prompt(problem, partial_proof)
                prompts.append(messages)  # Store raw messages, not formatted string
                answers.append(final_answer)
        
        return {"prompt": prompts, "answer": answers}
    
    # Process dataset
    formatted_dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Filter out None answers
    formatted_dataset = formatted_dataset.filter(lambda x: x['answer'] is not None)
    
    # Limit dataset to last MAX_NUM rows if specified
    if max_samples is not None and max_samples > 0:
        original_size = len(formatted_dataset)
        start_idx = max(0, len(formatted_dataset) - max_samples)
        formatted_dataset = formatted_dataset.select(range(start_idx, len(formatted_dataset)))
        print(f"[INFO] Limited dataset from {original_size} to {len(formatted_dataset)} rows (using last {max_samples} rows)")
    
    # Convert to verl format
    # verl expects ground_truth nested under reward_model
    rl_data = []
    for item in formatted_dataset:
        rl_data.append({
            "prompt": item["prompt"],  # List of message dicts with 'role' and 'content'
            "reward_model": {
                "ground_truth": item["answer"],
            },
            "data_source": "omni_math",  # Identifier for reward function
        })
    
    return rl_data


def get_verl_config_path():
    """Get path to verl's config directory."""
    import verl
    verl_path = os.path.dirname(verl.__file__)
    config_path = os.path.join(verl_path, "trainer", "config")
    return config_path


def main():
    """Main function to run GRPO training with verl and Ray."""
    print("=" * 80)
    print("GRPO Training with verl and Ray")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Nodes: {NUM_NODES}, GPUs per node: {GPUS_PER_NODE}")
    print("=" * 80)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create RL dataset
    print("Creating RL dataset...")
    rl_data = create_rl_dataset(tokenizer, DATASET_NAME, max_samples=MAX_NUM)
    print(f"Created {len(rl_data)} training samples")
    
    # Save dataset to parquet file
    print("Saving dataset to parquet...")
    df_data = pd.DataFrame(rl_data)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False, dir='/mnt/tmp') as f:
        df_data.to_parquet(f.name, index=False)
        dataset_file = f.name
    print(f"Dataset saved to {dataset_file}")
    
    # Load verl's default config using Hydra
    print("Loading verl configuration with Hydra...")
    config_path = get_verl_config_path()
    
    # Clear any previous Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with verl's config directory
    initialize_config_dir(config_dir=config_path, version_base=None)
    
    # Compose the config with our overrides
    experiment_name = f"grpo_omni_math_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.prompt_length=2048",
        "actor_rollout_ref.rollout.response_length=1536",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.load_format=auto",
        "actor_rollout_ref.rollout.enforce_eager=true",
        
        # Ray cluster config - connect to existing cluster
        "++ray_kwargs.ray_init.address=auto",
        
        # Actor config
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.ppo_epochs=1",
        
        # Algorithm - GRPO
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.norm_adv_by_std_in_grpo=true",
        
        # Data config
        f"data.train_files={dataset_file}",
        f"data.val_files={dataset_file}",  # Use same file for validation to avoid None error
        "data.prompt_key=prompt",
        "data.max_prompt_length=2048",
        "data.max_response_length=1536",
        "data.train_batch_size=16",
        "data.val_batch_size=16",
        
        # Trainer config
        f"trainer.project_name=grpo-omni-math",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.nnodes={NUM_NODES}",
        f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
        f"trainer.default_local_dir={OUTPUT_DIR}",
        "trainer.total_epochs=1",
        "trainer.save_freq=500",
        "trainer.val_before_train=false",  # Skip validation for now
        
        # Custom reward function (use ++ to add or override)
        f"++custom_reward_function.path={__file__}",
        "++custom_reward_function.name=compute_score",
        
        # Disable reward model loop (use custom function)
        "++reward_model.enable=false",
        
        # Disable critic (not needed for GRPO)
        "++critic.enable=false",
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
        if os.path.exists(dataset_file):
            os.unlink(dataset_file)
            print(f"Cleaned up temporary dataset file: {dataset_file}")


if __name__ == "__main__":
    main()
