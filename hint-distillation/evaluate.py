#!/usr/bin/env python3
"""
Evaluate finetuned model on problems NOT used in training.
Compares performance with and without hints.
"""

import os
import json
import argparse
import re
from tqdm import tqdm
from datasets import load_from_disk
from vllm import LLM, SamplingParams

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

FINETUNED_MODEL = "./finetuned_model_v2/final"
HINTS_DATASET = "../newopenaioutputs/hints_dataset"
SFT_DATASET = "./distillation_data/sft_dataset"
TEST_INDICES = "./distillation_data/test_indices.json"


def extract_boxed(text):
    """Extract content from \\boxed{...} or \\box{...}."""
    patterns = [r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\\box\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}']
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return None


def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return None
    ans = str(ans).strip().lower()
    ans = re.sub(r'\s+', '', ans)
    ans = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', ans)
    return ans


def check_answer(response, ground_truth):
    """Check if response contains correct answer."""
    extracted = extract_boxed(response)
    if extracted is None:
        return False
    return normalize_answer(extracted) == normalize_answer(ground_truth)


def format_prompt(problem, partial_proof, hint):
    """Format prompt with hint."""
    system = """You are learning to solve mathematics problems. You will be given a math problem, a partial proof or solution, and a hint. Your task is to carefully complete the proof or solution, step by step, providing clear reasoning at each stage (do not skip steps), making appropriate use of the hint. Only after finishing the complete reasoning, write the final answer at the end, clearly enclosed in the \\box{...} environment as is standard in LaTeX. 

- For each step, show the logical process and all intermediate computations or deductions.
- Use the provided hint as needed to help guide your reasoning.
- Only after reasoning is finished, put the final answer at the end, in its own line, using \\box{...}
- Use plain text with embedded LaTeX where mathematical symbols or equations are necessary."""
    
    user = f"Problem: {problem}\n\nPartial proof: {partial_proof}\n\nHint: {hint}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=FINETUNED_MODEL)
    parser.add_argument("--hints-dataset", type=str, default=HINTS_DATASET)
    parser.add_argument("--test-indices", type=str, default=TEST_INDICES)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval problems")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    # Load test indices (problems held out from distillation)
    print(f"Loading test indices from {args.test_indices}...")
    with open(args.test_indices, "r") as f:
        eval_indices = json.load(f)
    print(f"Loaded {len(eval_indices)} test problem indices")

    # Load hints dataset
    print(f"Loading hints dataset from {args.hints_dataset}...")
    hints_dataset = load_from_disk(args.hints_dataset)
    print(f"Loaded {len(hints_dataset)} problems total")

    if args.limit:
        eval_indices = eval_indices[:args.limit]
        print(f"Limited to {len(eval_indices)} problems")

    # Initialize vLLM
    print(f"Loading model {args.model} with tensor_parallel_size={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_tokens,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # Prepare prompts for each problem with different hint levels
    all_prompts = []
    prompt_metadata = []  # (eval_idx, hint_rank)

    for eval_idx in tqdm(eval_indices, desc="Preparing prompts"):
        row = hints_dataset[eval_idx]
        problem = row["problem"]
        partial_proof = row["partial_proof"]
        hints = row["hints"]
        ground_truth = row["final_answer"]

        if not hints:
            continue

        # Test with weakest hint (hints[0]) and strongest hint (hints[-1])
        for hint_type, hint in [("weakest", hints[0]), ("strongest", hints[-1])]:
            messages = format_prompt(problem, partial_proof, hint)
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(prompt_text)
            prompt_metadata.append((eval_idx, hint_type, ground_truth))

    # Generate responses
    print(f"Generating {len(all_prompts)} responses...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Evaluate
    results = {"weakest": {"correct": 0, "total": 0}, "strongest": {"correct": 0, "total": 0}}
    detailed_results = []

    for output, (eval_idx, hint_type, gt) in zip(outputs, prompt_metadata):
        response = output.outputs[0].text
        correct = check_answer(response, gt)
        results[hint_type]["total"] += 1
        if correct:
            results[hint_type]["correct"] += 1
        
        detailed_results.append({
            "eval_idx": eval_idx,
            "hint_type": hint_type,
            "correct": correct,
            "ground_truth": gt,
            "extracted": extract_boxed(response),
        })

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for hint_type in ["strongest", "weakest"]:
        acc = results[hint_type]["correct"] / results[hint_type]["total"] * 100 if results[hint_type]["total"] > 0 else 0
        print(f"{hint_type.upper()} hint: {results[hint_type]['correct']}/{results[hint_type]['total']} = {acc:.1f}%")
    
    # Calculate improvement
    if results["strongest"]["total"] > 0 and results["weakest"]["total"] > 0:
        strong_acc = results["strongest"]["correct"] / results["strongest"]["total"]
        weak_acc = results["weakest"]["correct"] / results["weakest"]["total"]
        print(f"\nDistillation effect: {(weak_acc - strong_acc)*100:+.1f}% (weak - strong)")
        print("(Positive = model learned to do well with weaker hints)")

    # Save detailed results
    output_path = os.path.join(os.path.dirname(args.model), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({"summary": results, "detailed": detailed_results}, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()

