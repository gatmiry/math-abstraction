#!/usr/bin/env python3
"""
GRPO training script for Omni-MATH dataset using verl for distributed training.
Configured for 2 nodes, each with 8 H100 GPUs (16 GPUs total).
"""

import os
import re
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from typing import List, Dict, Optional
import json

# Try to import verl, fallback to trl if not available
USE_VERL = False
try:
    # verl might have a different import structure
    # Try common import patterns
    try:
        from verl.trainer import GRPOTrainer
        from verl.config import GRPOConfig
        USE_VERL = True
    except ImportError:
        try:
            from verl import GRPOTrainer, GRPOConfig
            USE_VERL = True
        except ImportError:
            raise ImportError("verl import failed")
except ImportError:
    try:
        from trl import GRPOTrainer, GRPOConfig
        USE_VERL = False
        # Only print warning on rank 0 or if not in distributed mode
        rank = os.environ.get("RANK", "0")
        if rank == "0" or "RANK" not in os.environ:
            print("Warning: verl not found, using trl instead. For multi-node training, please install verl.")
    except ImportError:
        raise ImportError("Neither verl nor trl is installed. Please install one of them: pip install verl or pip install trl")


# Configuration
MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
DATASET_NAME = "newopenaioutputs/hints_dataset"  # Omni-MATH dataset
OUTPUT_DIR = "../models/qwen2-math-7b-instruct_grpo_hints_dataset_verl"

# Distributed training configuration
# 2 nodes, 8 GPUs per node = 16 GPUs total
NUM_NODES = 2
GPUS_PER_NODE = 8
WORLD_SIZE = NUM_NODES * GPUS_PER_NODE


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


def create_reward_func(prompt_to_answer: Dict[str, str]):
    """Create a reward function that checks if boxed answer matches ground truth.
    
    Args:
        prompt_to_answer: Dictionary mapping prompts to ground truth answers
    
    Returns:
        Reward function that takes completions and returns rewards
    """
    def reward_func(completions: List[str], prompts: Optional[List[str]] = None, **kwargs):
        """Compute reward: 1.0 if boxed answer matches ground truth, 0.0 otherwise.
        
        Args:
            completions: List of generated completions
            prompts: Optional list of prompts corresponding to completions
            **kwargs: Additional arguments (may contain 'answers' list)
        
        Returns:
            List of reward values (one per completion)
        """
        rewards = []
        for i, completion in enumerate(completions):
            # Get corresponding answer from prompt
            answer = None
            if prompts and i < len(prompts):
                answer = prompt_to_answer.get(prompts[i])
            
            # Fallback: try to get from kwargs if available
            if answer is None and 'answers' in kwargs:
                answers_list = kwargs['answers']
                if i < len(answers_list):
                    answer = answers_list[i]
            
            if answer is None:
                rewards.append(0.0)
                continue
            
            # Extract boxed answer from completion
            boxed_answer = extract_boxed_answer(completion)
            boxed_normalized = normalize_answer(boxed_answer)
            answer_normalized = normalize_answer(answer)
            
            # Compare normalized answers
            if boxed_normalized and answer_normalized and boxed_normalized == answer_normalized:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    return reward_func


def format_prompt(problem: str, partial_proof: str) -> List[Dict[str, str]]:
    """Format problem as a prompt for the model.
    
    Args:
        problem: Problem statement from the dataset
    
    Returns:
        List of messages in chat format
    """
    messages = [
        {
            "role": "system",
            "content": """You are learning to solve mathematics problems. You will be given a math problem and a partial proof or solution. Your task is to carefully complete the proof or solution, step by step, providing clear reasoning at each stage (do not skip steps). Only after finishing the complete reasoning, write the final answer at the end, clearly enclosed in the \box{...} environment as is standard in LaTeX. 

- For each step, show the logical process and all intermediate computations or deductions.
- Only after reasoning is finished, put the final answer at the end, in its own line, using \box{...}
- Use plain text with embedded LaTeX where mathematical symbols or equations are necessary.

## Output Format

Present your solution as a well-formatted, step-by-step proof or solution in plain text (not as a code block). Mathematical expressions and the boxed answer should use proper LaTeX syntax, e.g. \box{42}. 

## Example

**Example Input:**  
Problem:
Prove that the derivative of \(f(x) = x^2\) is \(2x\).  

Partial proof:  
The derivative of \(f(x)\) is defined as  
\[
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\]

**Example Output:**  
Let's substitute \(f(x) = x^2\) into the definition:  
\[
f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h}
\]  
Expand \((x+h)^2\):  
\[
(x+h)^2 = x^2 + 2xh + h^2
\]  
Subtract \(x^2\):  
\[
(x^2 + 2xh + h^2) - x^2 = 2xh + h^2
\]  
So,  
\[
f'(x) = \lim_{h \to 0} \frac{2xh + h^2}{h}
\]  
Divide numerator by \(h\):  
\[
= \lim_{h \to 0} (2x + h)
\]  
Take the limit as \(h \to 0\):  
\[
= 2x
\]  

\box{2x}

---

**Reminders:**  
- Complete the proof step by step, showing all logical reasoning before the boxed answer.
- The final answer must always appear at the end, in \box{...}. 

**IMPORTANT INSTRUCTION SUMMARY:**  
- Show step-by-step reasoning before the conclusion.
- Place final answer boxed (in \box{...}) at the end."""
        },
        {
            "role": "user",
            "content": f"Problem: {problem}\n\nPartial proof: {partial_proof}"
        }
    ]
    return messages


def setup_distributed():
    """Setup distributed training environment."""
    # Initialize process group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        
        return rank, world_size, local_rank, device
    else:
        # Single GPU or non-distributed mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main training function."""
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    # Only print on rank 0
    if rank == 0:
        print(f"Starting GRPO training with {world_size} GPUs")
        print(f"Configuration: {NUM_NODES} nodes, {GPUS_PER_NODE} GPUs per node")
        print(f"Using {'verl' if USE_VERL else 'trl'} for GRPO training")
    
    # Load model and tokenizer
    if rank == 0:
        print(f"Loading model from {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading configuration
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    
    # For distributed training, don't use device_map
    # Let the distributed framework handle device placement
    if world_size > 1:
        # Distributed mode: load model without device_map
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
        model = model.to(device)
    else:
        # Single GPU mode
        model_kwargs["device_map"] = "cuda:0" if device.type == "cuda" else "cpu"
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    
    # Clear CUDA cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Load dataset
    if rank == 0:
        print(f"Loading dataset: {DATASET_NAME}...")
    
    dataset = load_from_disk(DATASET_NAME)
    
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
    
    # Format dataset for GRPO
    def format_dataset(examples):
        """Format dataset examples for GRPO training."""
        prompts = []
        answers = []
        for problem, partial_proof, final_answer in zip(examples['problem'], examples['partial_proof'], examples['final_answer']):
            if final_answer:  # Only include if final answer exists
                messages = format_prompt(problem, partial_proof, final_answer)
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
                answers.append(final_answer)
        
        return {"prompt": prompts, "answer": answers}
    
    # Process dataset
    if rank == 0:
        print("Formatting dataset...")
    
    formatted_dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Filter out None answers
    formatted_dataset = formatted_dataset.filter(lambda x: x['answer'] is not None)
    
    if rank == 0:
        print(f"Filtered dataset size: {len(formatted_dataset)}")
    
    # Split train/eval
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # Create mapping from prompt to answer for reward function
    prompt_to_answer = {
        prompt: answer
        for prompt, answer in zip(train_dataset['prompt'], train_dataset['answer'])
    }
    reward_func = create_reward_func(prompt_to_answer)
    
    # GRPO training config
    # Optimized for multi-node, multi-GPU training
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        gradient_checkpointing=True,
        max_completion_length=2048,
        num_generations=8,  # Must be divisible by generation_batch_size
        optim="adamw_bnb_8bit",  # 8-bit optimizer to save memory
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        # Distributed training settings
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        # For verl: additional distributed settings
        deepspeed=None,  # Can be configured for ZeRO if needed
    )
    
    # Initialize GRPO trainer
    if rank == 0:
        print("Initializing GRPO trainer...")
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
    )
    
    # Train
    if rank == 0:
        print("Starting GRPO training...")
    
    trainer.train()
    
    # Save model (only on rank 0)
    if rank == 0:
        print(f"Saving model to {OUTPUT_DIR}...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Training completed!")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()

