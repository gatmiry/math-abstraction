#!/usr/bin/env python3
"""
Simple GRPO training script for fine-tuned Qwen2-Math-7B-Instruct on Omni-MATH dataset.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

# Configuration
MODEL_PATH = "./models/qwen2-math-7b-instruct_finetuned_on_first_3542_transformed_omni_math_solutions_filtered_lr:2e-06_warmup_steps:300_num_epochs:3/checkpoint-500"
DATASET_NAME = "qwedsacf/competition_math"  # Omni-MATH dataset
OUTPUT_DIR = "./models/qwen2-math-7b-instruct_grpo_omni_math"

def extract_boxed_answer(text):
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

def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None
    return answer.strip().lower()

def create_reward_func(prompt_to_answer):
    """Create a reward function that checks if boxed answer matches ground truth."""
    def reward_func(completions, prompts=None, **kwargs):
        """Compute reward: 1.0 if boxed answer matches ground truth, 0.0 otherwise."""
        rewards = []
        for i, completion in enumerate(completions):
            # Get corresponding answer from prompt
            if prompts and i < len(prompts):
                answer = prompt_to_answer.get(prompts[i])
            else:
                # Fallback: try to get from kwargs if available
                answer = kwargs.get('answers', [None])[i] if 'answers' in kwargs and i < len(kwargs['answers']) else None
            
            if answer is None:
                rewards.append(0.0)
                continue
            
            boxed_answer = extract_boxed_answer(completion)
            boxed_normalized = normalize_answer(boxed_answer)
            answer_normalized = normalize_answer(answer)
            
            if boxed_normalized and answer_normalized and boxed_normalized == answer_normalized:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    return reward_func

def format_prompt(problem):
    """Format problem as a prompt."""
    messages = [
        {"role": "system", "content": "You are a math tutor. Give a complete solution and put the final answer in the format \\boxed{...}."},
        {"role": "user", "content": problem}
    ]
    return messages

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Format dataset for GRPO
    def format_dataset(examples):
        prompts = []
        answers = []
        for problem, answer in zip(examples['problem'], examples['answer']):
            if answer:  # Only include if answer exists
                messages = format_prompt(problem)
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
                answers.append(answer)
        
        return {"prompt": prompts, "answer": answers}
    
    # Process dataset
    print("Formatting dataset...")
    formatted_dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Filter out None answers
    formatted_dataset = formatted_dataset.filter(lambda x: x['answer'] is not None)
    print(f"Filtered dataset size: {len(formatted_dataset)}")
    
    # Split train/eval
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # Create mapping from prompt to answer for reward function
    prompt_to_answer = {prompt: answer for prompt, answer in zip(train_dataset['prompt'], train_dataset['answer'])}
    reward_func = create_reward_func(prompt_to_answer)
    
    # GRPO training config
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
        max_length=2048,
        max_prompt_length=1024,
    )
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
    )
    
    # Train
    print("Starting GRPO training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completed!")

if __name__ == "__main__":
    main()

