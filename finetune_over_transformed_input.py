#!/usr/bin/env python3
"""
Minimal fine-tuning script for Qwen 7B on geometry problems.
Uses Qwen's chat template format for training.

Requirements:
    pip install transformers datasets torch accelerate

Optional for QLoRA (memory-efficient fine-tuning):
    pip install peft bitsandbytes
"""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import torch

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B"
DATASET_PATH = "newopenaioutputs/transformed_solutions_qwen2.57b_dataset"
OUTPUT_DIR = "./models/qwen2.57b_finetuned_on_first_5000_transformed_math_solutions"
# MAX_LENGTH: 2048 covers ~95% of examples (median: 934, 95th percentile: 1934)
# Alternative: 2560 covers ~99% of examples (99th percentile: 2466, max: 2752)
# Note: 4096 doubles activation memory compared to 2048 - reduce if you hit OOM errors
MAX_LENGTH = 4096

def format_chat_messages(problem, solution):
    """Format problem and solution in Qwen chat format."""
    messages = [
        {"role": "user", "content": f"Solve this geometry problem:\n\n{problem}"},
        {"role": "assistant", "content": solution}
    ]
    return messages

def preprocess_function(examples, tokenizer):
    """Preprocess examples using Qwen's chat template."""
    inputs = []
    
    for problem, solution in zip(examples['problem'], examples['new_proof']):
        messages = format_chat_messages(problem, solution)
        # Use apply_chat_template to format according to Qwen's format
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        inputs.append(text)
    
    # Tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False
    )
    
    # Labels will be set by DataCollatorForLanguageModeling (same as input_ids)
    # No need to set labels here for causal LM - the collator handles it
    
    return model_inputs

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
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split into train/validation (90/10)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 4
        learning_rate=2e-5,
        warmup_steps=100,
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
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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

