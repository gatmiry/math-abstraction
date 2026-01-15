#!/usr/bin/env python3
"""Check training labels around EOS token."""

from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Math-7B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load example
with open("/mnt/task_wrapper/user_output/artifacts/iterative_distillation/round_1/distillation_pairs.json") as f:
    train_data = json.load(f)

example = train_data[0]
messages = example["messages"] + [{"role": "assistant", "content": example["target_response"]}]

# Simulate finetune.py tokenization
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
prompt_messages = messages[:-1]
prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

full_tokens = tokenizer(full_text, truncation=True, max_length=4096, return_tensors=None)
prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=4096, return_tensors=None)

input_ids = full_tokens["input_ids"]
labels = [-100] * len(prompt_tokens["input_ids"]) + input_ids[len(prompt_tokens["input_ids"]):]
labels = labels[:len(input_ids)]

print("=== TRAINING LABELS ANALYSIS ===")
print(f"Total tokens: {len(input_ids)}")
print(f"Prompt tokens (masked with -100): {len(prompt_tokens['input_ids'])}")
print(f"Response tokens (trained on): {len(input_ids) - len(prompt_tokens['input_ids'])}")

# Show last few tokens and labels
print("\n=== LAST 15 TOKENS AND LABELS ===")
for i in range(-15, 0):
    token = input_ids[i]
    label = labels[i]
    decoded = tokenizer.decode([token])
    label_str = f"{label}" if label != -100 else "MASKED"
    print(f"  {i}: token={token:6d} label={label_str:8s} -> {repr(decoded)}")

# Find EOS position
eos_pos = None
for i, t in enumerate(input_ids):
    if t == 151645:  # EOS token
        eos_pos = i
        
if eos_pos is not None:
    print(f"\n=== KEY FINDING ===")
    print(f"EOS token (151645) is at position: {eos_pos - len(input_ids)} from end (absolute: {eos_pos})")
    
    if eos_pos + 1 < len(input_ids):
        next_token = input_ids[eos_pos + 1]
        next_label = labels[eos_pos + 1]
        print(f"Token AFTER EOS: {next_token} -> {repr(tokenizer.decode([next_token]))}")
        print(f"Label for token after EOS: {next_label}")
        print()
        if next_label != -100:
            print("⚠️  WARNING: Model is being trained to predict tokens AFTER EOS!")
            print("   This teaches the model that EOS is not the end!")
    else:
        print("EOS is at the very end - good!")
else:
    print("EOS token not found!")

