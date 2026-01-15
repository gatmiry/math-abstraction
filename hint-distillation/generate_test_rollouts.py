#!/usr/bin/env python3
"""Generate detailed rollouts on test set from a model checkpoint."""

import os
import json
import argparse
import re
from datasets import load_from_disk
from vllm import LLM, SamplingParams

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Paths
HINTS_DATASET = "../newopenaioutputs/hints_dataset"
TEST_INDICES_PATH = "/mnt/task_wrapper/user_output/artifacts/iterative_distillation/test_indices.json"


def format_prompt(problem: str, partial_proof: str, hint: str) -> list:
    """Format prompt as chat messages."""
    system = """You are learning to solve mathematics problems. You will be given a math problem, a partial proof or solution, and a hint. Your task is to carefully complete the proof or solution, step by step, providing clear reasoning at each stage (do not skip steps), making appropriate use of the hint. Only after finishing the complete reasoning, write the final answer at the end, clearly enclosed in the \\box{...} environment as is standard in LaTeX. 

- For each step, show the logical process and all intermediate computations or deductions.
- Use the provided hint as needed to help guide your reasoning.
- Only after reasoning is finished, put the final answer at the end, in its own line, using \\box{...}
- Use plain text with embedded LaTeX where mathematical symbols or equations are necessary."""
    
    user_content = f"Problem: {problem}\n\nPartial proof: {partial_proof}\n\nHint: {hint}"
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...} or \\box{...}."""
    patterns = [r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\\box\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}']
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return ""


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = ans.lower().strip()
    ans = re.sub(r'\s+', '', ans)
    ans = ans.replace('\\frac', '').replace('\\cdot', '*').replace('\\times', '*')
    return ans


def check_answer(response: str, ground_truth: str) -> bool:
    """Check if response contains correct answer."""
    extracted = extract_boxed(response)
    if not extracted:
        return False
    return normalize_answer(extracted) == normalize_answer(ground_truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default=HINTS_DATASET)
    parser.add_argument("--test-indices", type=str, default=TEST_INDICES_PATH)
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test problems")
    args = parser.parse_args()

    # Load dataset and test indices
    full_dataset = load_from_disk(args.dataset)
    with open(args.test_indices, "r") as f:
        test_indices = json.load(f)
    
    if args.limit:
        test_indices = test_indices[:args.limit]
    
    test_dataset = full_dataset.select(test_indices)
    print(f"Loaded {len(test_dataset)} test problems")

    # Initialize vLLM
    print(f"Loading model {args.model}...")
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

    # Prepare all prompts for all hints
    prompts = []
    prompt_metadata = []

    print("Preparing prompts...")
    for idx, row in enumerate(test_dataset):
        problem = row["problem"]
        partial_proof = row["partial_proof"]
        hints = row.get("hints", [])
        gt = row.get("final_answer", "")

        if not hints:
            continue

        # Generate for all hints (strongest = index 0, weakest = last index)
        for hint_rank, hint in enumerate(hints):
            messages = format_prompt(problem, partial_proof, hint)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            prompt_metadata.append({
                "test_idx": idx,
                "original_idx": test_indices[idx],
                "hint_rank": hint_rank,
                "total_hints": len(hints),
                "hint": hint,
                "ground_truth": gt,
                "problem": problem,
                "partial_proof": partial_proof,
            })

    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Organize results by test problem
    results = []
    results_by_idx = {}

    for output, meta in zip(outputs, prompt_metadata):
        response = output.outputs[0].text
        correct = check_answer(response, meta["ground_truth"])
        
        test_idx = meta["test_idx"]
        if test_idx not in results_by_idx:
            results_by_idx[test_idx] = {
                "test_idx": test_idx,
                "original_idx": meta["original_idx"],
                "problem": meta["problem"],
                "partial_proof": meta["partial_proof"],
                "ground_truth": meta["ground_truth"],
                "rollouts": []
            }
        
        results_by_idx[test_idx]["rollouts"].append({
            "hint_rank": meta["hint_rank"],
            "total_hints": meta["total_hints"],
            "hint": meta["hint"],
            "response": response,
            "correct": correct,
            "extracted_answer": extract_boxed(response),
        })

    # Sort rollouts by hint_rank and convert to list
    for test_idx in results_by_idx:
        results_by_idx[test_idx]["rollouts"].sort(key=lambda x: x["hint_rank"])
        results.append(results_by_idx[test_idx])
    
    # Sort results by test_idx
    results.sort(key=lambda x: x["test_idx"])

    # Compute summary stats
    strongest_correct = sum(1 for r in results if r["rollouts"] and r["rollouts"][0]["correct"])
    weakest_correct = sum(1 for r in results if r["rollouts"] and r["rollouts"][-1]["correct"])
    total_with_hints = len([r for r in results if r["rollouts"]])

    summary = {
        "model": args.model,
        "num_problems": len(results),
        "num_with_hints": total_with_hints,
        "strongest_hint_correct": strongest_correct,
        "weakest_hint_correct": weakest_correct,
        "strongest_hint_accuracy": strongest_correct / total_with_hints if total_with_hints > 0 else 0,
        "weakest_hint_accuracy": weakest_correct / total_with_hints if total_with_hints > 0 else 0,
    }

    output_data = {
        "summary": summary,
        "results": results,
    }

    # Save results
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

