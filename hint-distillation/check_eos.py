#!/usr/bin/env python3
"""Check if EOS token is properly included in training data."""

from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Math-7B-Instruct", trust_remote_code=True)

print("=== TOKENIZER INFO ===")
print(f"EOS token: {repr(tokenizer.eos_token)}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"PAD token: {repr(tokenizer.pad_token)}")

# Load a training example
with open("/mnt/task_wrapper/user_output/artifacts/iterative_distillation/round_1/distillation_pairs.json") as f:
    train_data = json.load(f)

example = train_data[0]
messages = example["messages"] + [{"role": "assistant", "content": example["target_response"]}]

print("\n=== MESSAGES STRUCTURE ===")
for m in messages:
    role = m["role"]
    content = m["content"][:50]
    print(f"  {role}: {content}...")

# Apply chat template WITHOUT tokenization
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

print("\n=== FULL TEXT ENDING (last 300 chars) ===")
print(repr(full_text[-300:]))

# Apply chat template WITH tokenization
tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

print(f"\n=== TOKEN CHECK ===")
print(f"Total tokens: {len(tokens)}")
print(f"Last 10 token IDs: {tokens[-10:]}")
print(f"Last 10 decoded: {[tokenizer.decode([t]) for t in tokens[-10:]]}")
print(f"EOS token ID ({tokenizer.eos_token_id}) in last 10? {tokenizer.eos_token_id in tokens[-10:]}")
print(f"EOS token ID at the very end? {tokens[-1] == tokenizer.eos_token_id}")

# Now check what finetune.py does
print("\n\n=== WHAT FINETUNE.PY DOES ===")

def tokenize_messages_simulation(messages, tokenizer, max_length=4096):
    """Simulate what finetune.py does."""
    # Get full text with assistant response
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Get prompt only (without assistant response) to find where to mask
    prompt_messages = messages[:-1]  # All but assistant
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    full_tokens = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors=None)
    
    input_ids = full_tokens["input_ids"]
    return input_ids

input_ids = tokenize_messages_simulation(messages, tokenizer)
print(f"Input IDs length: {len(input_ids)}")
print(f"Last 10 token IDs: {input_ids[-10:]}")
print(f"Last 10 decoded: {[tokenizer.decode([t]) for t in input_ids[-10:]]}")
print(f"EOS at end? {input_ids[-1] == tokenizer.eos_token_id}")

# Check if tokenizer adds EOS automatically
print("\n=== TOKENIZER ADD_EOS BEHAVIOR ===")
test_text = "Hello world"
tokens_test = tokenizer(test_text, return_tensors=None)["input_ids"]
print(f"Tokenizing '{test_text}': {tokens_test}")
print(f"EOS at end? {tokens_test[-1] == tokenizer.eos_token_id}")

