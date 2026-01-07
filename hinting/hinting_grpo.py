#!/usr/bin/env python3
"""
GRPO training script for Omni-MATH dataset using verl with Ray for distributed training.
Configured for 2 nodes, each with 8 H100 GPUs (16 GPUs total).
Uses Ray instead of torchrun for distributed training.
"""

import os
import re
import ray
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer
from datasets import load_from_disk
from typing import List, Dict, Optional, Any
import datetime

# Import verl components
from verl.trainer.main_ppo import run_ppo
from verl.trainer.config import AlgoConfig
from verl.trainer.config.algorithm import AdvantageEstimator

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
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Optional[str]],
    extra_infos: List[Dict[str, Any]],
    **kwargs
) -> List[float]:
    """Compute reward scores for a batch of solutions.
    
    This function is used by verl's BatchRewardManager.
    
    Args:
        data_sources: List of data source identifiers (not used here)
        solution_strs: List of generated solution strings
        ground_truths: List of ground truth answers (from dataset)
        extra_infos: List of extra info dictionaries
        **kwargs: Additional keyword arguments
    
    Returns:
        List of reward scores (1.0 if answer matches, 0.0 otherwise)
    """
    rewards = []
    
    for i, (solution_str, ground_truth) in enumerate(zip(solution_strs, ground_truths)):
        if ground_truth is None:
            rewards.append(0.0)
            continue
        
        # Extract boxed answer from solution
        boxed_answer = extract_boxed_answer(solution_str)
        boxed_normalized = normalize_answer(boxed_answer)
        answer_normalized = normalize_answer(ground_truth)
        
        # Compare normalized answers
        if boxed_normalized and answer_normalized and boxed_normalized == answer_normalized:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    print(f"[REWARD] Computed rewards for {len(rewards)} solutions: {sum(rewards)}/{len(rewards)} correct")
    return rewards


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


def create_verl_config() -> DictConfig:
    """Create verl configuration for GRPO training.
    
    Returns:
        OmegaConf DictConfig with verl training configuration
    """
    # Base config structure matching verl's ppo_trainer.yaml
    config_dict = {
        # Actor, rollout, and reference model config
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "nccl_timeout": 600,
            "rollout": {
                "layered_summon": False,
            },
            "actor": {
                "_target_": "verl.trainer.config.actor.dp_actor.DPActorConfig",
                "strategy": "fsdp",
                "model": {
                    "_target_": "verl.trainer.config.model.hf_model.HFModelConfig",
                    "model_path": MODEL_PATH,
                    "trust_remote_code": True,
                },
            },
        },
        # Algorithm config - GRPO
        "algorithm": {
            "_target_": "verl.trainer.config.AlgoConfig",
            "gamma": 1.0,
            "lam": 1.0,
            "adv_estimator": "grpo",  # Use GRPO advantage estimator
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "fixed",
                "kl_coef": 0.001,
                "horizon": 10000,
                "target_kl": 0.1,
            },
            "use_pf_ppo": False,
        },
        # Trainer config
        "trainer": {
            "balance_batch": True,
            "total_epochs": 1,
            "total_training_steps": None,
            "project_name": "grpo-omni-math",
            "experiment_name": f"grpo_omni_math_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "logger": ["console", "wandb"],
            "log_val_generations": 0,
            "rollout_data_dir": None,
            "validation_data_dir": None,
            "nnodes": NUM_NODES,
            "n_gpus_per_node": GPUS_PER_NODE,
            "save_freq": 500,
            "esi_redundant_time": 0,
            "resume_mode": "auto",
            "resume_from_path": None,
            "val_before_train": True,
            "val_only": False,
            "test_freq": -1,
            "critic_warmup": 0,
            "default_hdfs_dir": None,
            "del_local_ckpt_after_load": False,
            "default_local_dir": OUTPUT_DIR,
            "max_actor_ckpt_to_keep": None,
            "max_critic_ckpt_to_keep": None,
            "ray_wait_register_center_timeout": 300,
            "device": "cuda",
            "use_legacy_worker_impl": "auto",
        },
        # Data config
        "data": {
            "_target_": "verl.trainer.config.data.legacy_data.LegacyDataConfig",
            "reward_fn_key": "data_source",
            # Dataset paths will be set dynamically
        },
        # Critic config
        "critic": {
            "_target_": "verl.trainer.config.critic.dp_critic.DPCriticConfig",
            "strategy": "fsdp",
            "model": {
                "_target_": "verl.trainer.config.model.hf_model.HFModelConfig",
                "model_path": MODEL_PATH,
                "trust_remote_code": True,
            },
        },
        # Reward model config (using custom reward function)
        "reward_model": {
            "_target_": "verl.trainer.config.reward_model.dp_reward_loop.DPRewardLoopConfig",
            "enable": True,
            "strategy": "fsdp",
            "enable_resource_pool": False,
            "nnodes": 0,
            "n_gpus_per_node": 0,
        },
        # Custom reward function
        "custom_reward_function": {
            "path": __file__,  # This file
            "name": "compute_score",  # Function name
            "reward_kwargs": {},
        },
        # Ray config
        "ray_kwargs": {
            "ray_init": {
                "num_cpus": None,  # Auto-detect
                "runtime_env": {
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "false",
                        "NCCL_DEBUG": "WARN",
                        "NCCL_TIMEOUT": "7200",
                    }
                }
            },
            "timeline_json_file": None,
        },
        # Profiler config
        "global_profiler": {
            "_target_": "verl.utils.profiler.ProfilerConfig",
            "tool": None,
            "steps": None,
            "profile_continuous_steps": False,
            "save_path": "outputs/profile",
        },
        # Transfer queue config
        "transfer_queue": {
            "enable": False,
        },
    }
    
    return OmegaConf.create(config_dict)


def create_rl_dataset(tokenizer, dataset_path: str, max_samples: Optional[int] = None):
    """Create RL dataset in verl format.
    
    Args:
        tokenizer: Tokenizer instance
        dataset_path: Path to dataset
        max_samples: Maximum number of samples (None = use all)
    
    Returns:
        List of data items in verl format
    """
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Format dataset
    def format_dataset(examples):
        """Format dataset examples for GRPO training."""
        prompts = []
        answers = []
        for problem, partial_proof, final_answer in zip(
            examples['problem'], examples['partial_proof'], examples['final_answer']
        ):
            if final_answer:  # Only include if final answer exists
                messages = format_prompt(problem, partial_proof)
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
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
    # verl expects data in a specific format with prompts and ground_truths
    rl_data = []
    for item in formatted_dataset:
        rl_data.append({
            "prompt": item["prompt"],
            "ground_truth": item["answer"],
            "data_source": "omni_math",  # Identifier for reward function
        })
    
    return rl_data


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
    
    # Create verl config
    print("Creating verl configuration...")
    config = create_verl_config()
    
    # Set dataset in config
    # verl expects parquet files for data
    # We'll need to convert the dataset to parquet format
    import tempfile
    import pandas as pd
    
    # Convert to DataFrame and save as parquet
    df_data = pd.DataFrame(rl_data)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
        df_data.to_parquet(f.name, index=False)
        dataset_file = f.name
    
    config.data.train_files = dataset_file
    config.data.val_files = None  # No validation for now
    config.data.prompt_key = "prompt"
    config.data.max_prompt_length = 2048
    config.data.max_response_length = 1536
    config.data.train_batch_size = 16  # Adjust based on GPU memory
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(
            **OmegaConf.to_container(config.ray_kwargs.ray_init)
        )
        print("Ray initialized successfully")
    
    # Run PPO training
    print("Starting GRPO training with verl...")
    try:
        run_ppo(config)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if os.path.exists(dataset_file):
            os.unlink(dataset_file)


if __name__ == "__main__":
    main()
