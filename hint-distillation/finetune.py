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
import torch.distributed as dist
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
    
    # FIX: Don't train on tokens after the LAST EOS - this prevents the model from 
    # learning to continue generating after <|im_end|>, which causes degenerate repetition
    # Note: There are multiple EOS tokens (one per message), we want the LAST one
    eos_token_id = tokenizer.eos_token_id
    # Find the last occurrence of EOS token
    eos_idx = None
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] == eos_token_id:
            eos_idx = i
            break
    
    if eos_idx is not None:
        # Set all labels after the last EOS to -100 (don't train on them)
        for i in range(eos_idx + 1, len(labels)):
            labels[i] = -100
    
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
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Training arguments with DDP (FSDP has issues with LoRA + gradient_checkpointing)
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
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for LoRA
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
    
    # Save final model - merge LoRA weights and save full model
    final_path = os.path.join(args.output, "final")
    
    # Only save on rank 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        # Get the PEFT model (unwrap DDP if needed)
        if hasattr(trainer.model, "module"):
            peft_model = trainer.model.module
        else:
            peft_model = trainer.model
        
        # Merge LoRA weights into base model and save
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"Merged model saved to {final_path}")
    
    # Barrier to ensure all ranks wait for save to complete
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()

