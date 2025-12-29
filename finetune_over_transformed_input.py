#!/usr/bin/env python3
"""
Minimal fine-tuning script for Qwen 7B on geometry problems.
Uses TRL SFTTrainer to only train on assistant tokens.

Requirements:
    pip install transformers datasets torch accelerate trl
"""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
import torch

# Configuration
lr = 2e-5
warmup_steps = 300
num_epochs = 5
MODEL_NAME = "Qwen/Qwen2-Math-7B-Instruct"
DATASET_PATH = "newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct_filtered"
OUTPUT_DIR = "./models/qwen2-math-7b-instruct_finetuned_on_first_3542_transformed_omni_math_solutions_filtered_lr:{lr}_warmup_steps:{warmup_steps}_num_epochs:{num_epochs}"
# MAX_LENGTH: 2048 covers ~95% of examples (median: 934, 95th percentile: 1934)
# Alternative: 2560 covers ~99% of examples (99th percentile: 2466, max: 2752)
# Note: 4096 doubles activation memory compared to 2048 - reduce if you hit OOM errors
MAX_LENGTH = 4096

def format_chat_messages(problem, solution):
    """Format problem and solution in Qwen chat format."""
    messages = [
        {"role": "system", "content": """You are a math tutor. Give a complete solution using the environments \\begin{intermediatederivation}...\\end{intermediatederivation} and \\begin{lemmatheorembox}...\\end{lemmatheorembox}, and put the final answer in the format \\boxed{...} at the end."""}, 
        {"role": "user", "content": f"""{problem}"""},
        {"role": "assistant", "content": solution}
    ]
    return messages

def format_dataset(examples):
    """Format dataset for SFTTrainer - returns messages."""
    messages_list = []
    for problem, solution in zip(examples['problem'], examples['new_proof']):
        messages = format_chat_messages(problem, solution)
        messages_list.append(messages)
    return {"messages": messages_list}

def main():
    # Use GPU 1 (GPU 0 is occupied by Jupyter kernel)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Determine device (will be cuda:0 from the perspective of this script, but actually GPU 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (actual GPU 1)")
    
    # Load model with low_cpu_mem_usage to save memory during loading
    # Use bfloat16 for better training stability (will be used via bf16=True in TrainingArguments)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    print(f"Dataset size: {len(dataset)}")
    
    # Format dataset for SFTTrainer (converts to messages format)
    print("Formatting dataset...")
    formatted_dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split into train/validation (90/10)
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Verify dataset format (SFTTrainer requires "messages" column for assistant-only training)
    if "messages" not in train_dataset.column_names:
        raise ValueError("Dataset must have 'messages' column for SFTTrainer to mask non-assistant tokens")
    print("Dataset has 'messages' format - SFTTrainer will automatically mask non-assistant tokens")
    
    import wandb
    wandb.init(project="omni_math_solutions_filtered", name=f"lr:{lr}_warmup_steps:{warmup_steps}")
    # Training arguments using SFTConfig (which extends TrainingArguments)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 4
        learning_rate=lr,
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_steps=100,
        save_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,  # Use bf16 for training (better than fp16 on H100, avoids gradient scaling issues)
        gradient_checkpointing=True,  # Save memory (trades compute for memory)
        # Use 8-bit optimizer if bitsandbytes is available (reduces optimizer memory from ~28GB to ~7GB)
        # Install: pip install bitsandbytes
        # Then uncomment: optim="adamw_bnb_8bit"
        optim="adamw_torch",  # Default, uses ~28GB for optimizer states
        dataloader_pin_memory=False,  # Save memory
        dataloader_num_workers=0,  # Save memory
        remove_unused_columns=False,  # Keep all columns
        report_to="wandb",  # Disable wandb/tensorboard
        max_length=MAX_LENGTH,  # Maximum sequence length for tokenization
    )
    
    # Initialize SFTTrainer
    # SFTTrainer automatically masks non-assistant tokens when dataset has "messages" format
    # It sets labels to -100 for system/user tokens, only computing loss on assistant tokens
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Changed from 'tokenizer' to 'processing_class'
        dataset_text_field=None,  # Not needed when using messages format
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completed!")

if __name__ == "__main__":
    main()

