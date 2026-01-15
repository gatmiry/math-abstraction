#!/usr/bin/env python3
"""
Simple SFT finetuning using HuggingFace Trainer with FSDP.
Uses hints_dataset with problem and ground_truth_solution.

Run with: accelerate launch --config_file fsdp_config.yaml --num_processes=8 sft_hf.py
Or: torchrun --nproc_per_node=8 sft_hf.py
"""

import os
import torch
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from accelerate import Accelerator

# Config
MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
DATASET_PATH = "/mnt/task_runtime/newopenaioutputs/hints_dataset"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/mnt/task_wrapper/user_output/artifacts/sft-hf-checkpoints_{TIMESTAMP}"

SYSTEM_PROMPT = "You are a mathematics expert. Solve the problem step by step. Put your final answer in \\boxed{...} format."


def load_tokens():
    """Load HF token."""
    token_file = "/mnt/task_runtime/hinting/hf_token.txt"
    if os.path.exists(token_file):
        with open(token_file) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.strip().split("=", 1)[1]
    return None


def format_example(example, tokenizer):
    """Format problem + solution into chat format. Returns (full_text, prompt_text) for masking."""
    problem = example.get("problem", "")
    solution = example.get("ground_truth_solution", "")
    answer = example.get("final_answer", "")
    
    if not problem or not solution:
        return None, None
    
    # Add boxed answer if missing
    if answer and "\\box" not in solution:
        solution = solution.strip() + f"\n\n\\boxed{{{answer}}}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}"},
        {"role": "assistant", "content": solution}
    ]
    
    # Full text with assistant response
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Prompt only (system + user) to find where assistant starts
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}"},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    return full_text, prompt_text


def main():
    # Setup
    hf_token = load_tokens()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with FSDP-compatible settings
    print(f"Loading model: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=False,  # Required for gradient checkpointing
    )
    model.gradient_checkpointing_enable()
    
    # Load and format dataset
    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    
    # Split FIRST before any processing - matches eval_checkpoint_vllm.py split
    # This ensures the test set is held out and never seen during training
    print("Splitting dataset (test_size=0.1, seed=42) to match eval script...")
    raw_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_raw = raw_split["train"]
    holdout_raw = raw_split["test"]
    print(f"Raw split - Train: {len(train_raw)}, Holdout (not used): {len(holdout_raw)}")
    
    def tokenize(example):
        full_text, prompt_text = format_example(example, tokenizer)
        if full_text is None:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # Tokenize full text
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=2048,
            padding=False,
        )
        
        # Tokenize prompt to find where assistant response starts
        prompt_encoded = tokenizer(
            prompt_text,
            truncation=True,
            max_length=2048,
            padding=False,
        )
        prompt_len = len(prompt_encoded["input_ids"])
        
        # Create labels: mask non-assistant tokens with -100
        labels = encoded["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        encoded["labels"] = labels
        
        return encoded
    
    # Tokenize only the train split (holdout is never used)
    tokenized = train_raw.map(tokenize, remove_columns=train_raw.column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
    
    # Split train into train/val for training monitoring (small val from train set)
    train_val_split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]
    
    print(f"Train: {len(train_dataset)}, Val (from train): {len(eval_dataset)}")
    
    # Training arguments with FSDP
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        per_device_train_batch_size=1,  # Small batch, FSDP handles sharding
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch = 1 * 8 * 8 GPUs = 64
        learning_rate=1e-6,  # 10x smaller to reduce catastrophic forgetting
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=None,  # Keep all checkpoints
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        report_to="wandb",
        run_name=f"sft_hf_qwen2_math_{TIMESTAMP}",
        # FSDP settings
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
            "fsdp_offload_params": False,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        },
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training with FSDP...")
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
