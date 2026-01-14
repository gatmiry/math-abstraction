
#!/usr/bin/env python3
"""
Hint Distillation: Find weakest working hint, distill its trace to weaker hints.

For each problem:
1. Generate proofs with all hints using vLLM
2. Find weakest hint i that gives correct answer
3. If hint i+1 exists (weaker, incorrect), create distillation pair:
   - Input: prompt with hint i+1 (weaker)
   - Target: response from hint i (stronger that worked)
4. Finetune model on these pairs
"""

import os
import re
import json
import argparse
from typing import List, Dict, Optional
from datasets import load_from_disk, Dataset
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

# Paths
HINTS_DATASET = "../newopenaioutputs/hints_dataset"
MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
OUTPUT_DIR = "./distillation_data"

SYSTEM_PROMPT = """You are learning to solve mathematics problems. You will be given a math problem, a partial proof or solution, and a hint. Your task is to carefully complete the proof or solution, step by step, providing clear reasoning at each stage (do not skip steps), making appropriate use of the hint. Only after finishing the complete reasoning, write the final answer at the end, clearly enclosed in the \\box{...} environment as is standard in LaTeX. 

- For each step, show the logical process and all intermediate computations or deductions.
- Use the provided hint as needed to help guide your reasoning.
- Only after reasoning is finished, put the final answer at the end, in its own line, using \\box{...}
- Use plain text with embedded LaTeX where mathematical symbols or equations are necessary."""


def format_prompt(problem: str, partial_proof: str, hint: str) -> List[Dict[str, str]]:
    """Format a prompt with hint."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}\n\nPartial proof: {partial_proof}\n\nHint: {hint}"}
    ]


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\box{...} or \\boxed{...}."""
    # Try \box{...} first (no backslash escape issues)
    patterns = [
        r'\\box\{([^}]+)\}',
        r'\\boxed\{([^}]+)\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()  # Return last match
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace and common LaTeX formatting
    ans = answer.strip()
    ans = re.sub(r'\s+', '', ans)
    ans = ans.replace('\\left', '').replace('\\right', '')
    ans = ans.replace('\\,', '').replace('\\;', '').replace('\\:', '')
    return ans.lower()


def check_answer(generated: str, ground_truth: str) -> bool:
    """Check if generated answer matches ground truth."""
    gen_ans = extract_boxed_answer(generated)
    if gen_ans is None:
        return False
    return normalize_answer(gen_ans) == normalize_answer(ground_truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--dataset", type=str, default=HINTS_DATASET)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = load_from_disk(args.dataset)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Loaded {len(dataset)} problems")

    # Initialize vLLM (tensor_parallel must divide attention heads, 28 for Qwen2-7B)
    # Use 4 GPUs for tensor parallelism (28 / 4 = 7)
    print(f"Loading model {args.model} with tensor_parallel_size=4")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Collect all prompts for batch generation
    # hints[0] = weakest, hints[-1] = strongest
    # We want to order from strongest (index 0) to weakest (index k-1)
    all_prompts = []
    prompt_metadata = []  # (problem_idx, hint_rank) where hint_rank 0 = strongest

    for prob_idx, row in enumerate(tqdm(dataset, desc="Preparing prompts")):
        problem = row["problem"]
        partial_proof = row["partial_proof"]
        hints = row["hints"]  # hints[0]=weakest, hints[-1]=strongest
        ground_truth = row["final_answer"]
        
        # Reverse to go from strongest to weakest
        hints_strong_to_weak = list(reversed(hints))
        
        for hint_rank, hint in enumerate(hints_strong_to_weak):
            messages = format_prompt(problem, partial_proof, hint)
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(prompt_text)
            prompt_metadata.append((prob_idx, hint_rank, hint, ground_truth))

    # Generate all responses in one batch
    print(f"Generating {len(all_prompts)} responses...")
    outputs = llm.generate(all_prompts, sampling_params)
    
    # Organize results by problem
    results_by_problem = {}
    for output, (prob_idx, hint_rank, hint, gt) in zip(outputs, prompt_metadata):
        response = output.outputs[0].text
        correct = check_answer(response, gt)
        
        if prob_idx not in results_by_problem:
            results_by_problem[prob_idx] = []
        results_by_problem[prob_idx].append({
            "hint_rank": hint_rank,  # 0 = strongest
            "hint": hint,
            "response": response,
            "correct": correct,
        })

    # Find distillation pairs
    distillation_pairs = []
    stats = {"total_problems": len(dataset), "pairs_found": 0, "all_correct": 0, "none_correct": 0}

    for prob_idx, row in enumerate(dataset):
        problem = row["problem"]
        partial_proof = row["partial_proof"]
        results = results_by_problem[prob_idx]
        
        # Sort by hint_rank (0=strongest first)
        results.sort(key=lambda x: x["hint_rank"])
        
        # Find weakest correct hint (largest hint_rank with correct=True)
        weakest_correct_rank = -1
        for r in results:
            if r["correct"]:
                weakest_correct_rank = r["hint_rank"]
        
        if weakest_correct_rank == -1:
            stats["none_correct"] += 1
            continue
        
        if weakest_correct_rank == len(results) - 1:
            stats["all_correct"] += 1
            continue  # Even weakest hint works, no distillation needed
        
        # Found boundary: hint at weakest_correct_rank works, hint at weakest_correct_rank+1 doesn't
        correct_result = results[weakest_correct_rank]
        incorrect_result = results[weakest_correct_rank + 1]
        
        # Create distillation pair
        # Input: prompt with weaker hint (that failed)
        # Target: response from stronger hint (that worked)
        weak_hint = incorrect_result["hint"]
        strong_response = correct_result["response"]
        
        messages = format_prompt(problem, partial_proof, weak_hint)
        
        distillation_pairs.append({
            "problem": problem,
            "partial_proof": partial_proof,
            "weak_hint": weak_hint,
            "strong_hint": correct_result["hint"],
            "target_response": strong_response,
            "messages": messages,
            "weak_hint_rank": incorrect_result["hint_rank"],
            "strong_hint_rank": correct_result["hint_rank"],
        })
        stats["pairs_found"] += 1

    # Save results
    print(f"\nStats: {stats}")
    
    # Save as HuggingFace Dataset for finetuning
    if distillation_pairs:
        # Format for SFT: messages with assistant response
        sft_data = []
        for pair in distillation_pairs:
            sft_data.append({
                "messages": pair["messages"] + [{"role": "assistant", "content": pair["target_response"]}],
                "problem": pair["problem"],
                "weak_hint_rank": pair["weak_hint_rank"],
                "strong_hint_rank": pair["strong_hint_rank"],
            })
        
        # Save as HuggingFace Dataset
        from datasets import Dataset as HFDataset
        sft_dataset = HFDataset.from_list(sft_data)
        dataset_path = os.path.join(args.output, "sft_dataset")
        sft_dataset.save_to_disk(dataset_path)
        print(f"Saved {len(sft_dataset)} distillation pairs to {dataset_path}")
        
        # Also save raw pairs as JSON for inspection
        json_path = os.path.join(args.output, "distillation_pairs.json")
        with open(json_path, "w") as f:
            json.dump(distillation_pairs, f, indent=2)
        print(f"Saved raw pairs to {json_path}")
    else:
        print("No distillation pairs found!")


if __name__ == "__main__":
    main()

