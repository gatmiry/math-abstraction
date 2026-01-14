#!/usr/bin/env python3
"""
Finetune model on distillation pairs.
Uses all 8 GPUs with FSDP via torchrun.

Run with:
  torchrun --standalone --nnodes=1 --nproc_per_node=8 finetune.py
"""

import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
DATA_PATH = "./distillation_data/sft_dataset"
OUTPUT_DIR = "./finetuned_model"


def tokenize_messages(example, tokenizer, max_length=4096):
    """Tokenize messages for causal LM training (only train on assistant response)."""
    messages = example["messages"]
    
    # Get full text with assistant response
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Get prompt only (without assistant response) to find where to mask
    prompt_messages = messages[:-1]  # All but assistant
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    full_tokens = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors=None)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=max_length, return_tensors=None)
    
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    
    # Create labels: -100 for prompt tokens (don't compute loss), actual ids for response
    labels = [-100] * len(prompt_tokens["input_ids"]) + input_ids[len(prompt_tokens["input_ids"]):]
    labels = labels[:len(input_ids)]  # Ensure same length
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset (saved by distill.py)
    print(f"Loading data from {args.data}")
    dataset = load_from_disk(args.data)
    print(f"Loaded {len(dataset)} examples")
    
    tokenized = dataset.map(
        lambda x: tokenize_messages(x, tokenizer, args.max_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    # Load model
    print(f"Loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Training arguments with FSDP
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fsdp="full_shard auto_wrap",
        fsdp_config={"fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer"},
        gradient_checkpointing=True,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(args.output, "final"))
    print(f"Model saved to {args.output}/final")


if __name__ == "__main__":
    main()

