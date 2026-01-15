#!/usr/bin/env python3
"""
Evaluate Qwen model on the first 100 problems in hints_dataset using the strongest hint.
The model receives: partial_proof + strongest hint (last hint in hints array)
"""

import argparse
import json
import re
from datasets import load_from_disk
from vllm import LLM, SamplingParams

HINTS_DATASET = "../newopenaioutputs/hints_dataset"
DEFAULT_MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
DEFAULT_OUTPUT = "eval_qwen_strongest_results.json"

def extract_boxed_answer(text):
    """Extract the answer from \\boxed{...}"""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None

def normalize_answer(ans):
    """Normalize answer for comparison"""
    if ans is None:
        return None
    ans = ans.strip()
    ans = ans.replace(" ", "")
    ans = ans.replace("\\$", "")
    ans = ans.replace("$", "")
    ans = ans.replace("\\%", "")
    ans = ans.replace("%", "")
    ans = ans.replace("\\text{", "").replace("}", "")
    ans = ans.replace("\\mathrm{", "").replace("}", "")
    ans = ans.replace("\\frac", "")
    ans = ans.replace("\\dfrac", "")
    return ans.lower()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen model on strongest hints.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model path or HF repo")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=100, help="Number of problems to evaluate")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="vLLM max_num_seqs override")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM gpu_memory_utilization override")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {HINTS_DATASET}...")
    dataset = load_from_disk(HINTS_DATASET)
    
    # Take first N problems
    problems = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Evaluating on {len(problems)} problems")
    
    # Load model
    print(f"Loading {args.model}...")
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": 4,
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )
    
    # Prepare prompts with partial_proof + strongest hint
    prompts = []
    valid_indices = []
    
    for i, prob in enumerate(problems):
        hints = prob.get("hints", [])
        if not hints:
            print(f"Problem {i}: No hints, skipping")
            continue
        
        valid_indices.append(i)
        question = prob["problem"]
        partial_proof = prob["partial_proof"]
        strongest_hint = hints[-1]  # Last hint is strongest (hint level 0)
        
        # Format prompt with partial proof and strongest hint
        prompt = (
            f"<|im_start|>system\n"
            f"You are a helpful math assistant. Solve the problem step by step and put your final answer in \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{question}\n\n"
            f"Partial solution:\n{partial_proof}\n\n"
            f"Hint: {strongest_hint}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(prompt)
    
    print(f"\nValid problems with hints: {len(valid_indices)}")
    
    # Generate responses
    print("\nGenerating responses with STRONGEST hint...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Evaluate
    correct = 0
    results = []
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for idx, output in enumerate(outputs):
        prob_idx = valid_indices[idx]
        prob = problems[prob_idx]
        ground_truth = prob["final_answer"]
        
        response = output.outputs[0].text
        predicted = extract_boxed_answer(response)
        
        norm_gt = normalize_answer(ground_truth)
        norm_pred = normalize_answer(predicted)
        
        is_correct = norm_pred == norm_gt if norm_pred and norm_gt else False
        
        if is_correct:
            correct += 1
        
        results.append({
            "problem_idx": prob_idx,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "hint": strongest_hint,
            "response": response
        })
        
        # Print first few examples
        if idx < 5:
            print(f"\n--- Problem {prob_idx} ---")
            print(f"Ground truth: {ground_truth}")
            print(f"Predicted: {predicted}")
            print(f"Correct: {is_correct}")
    
    total = len(valid_indices)
    accuracy = 100 * correct / total if total > 0 else 0
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS")
    print("="*60)
    print(f"Total problems evaluated: {total}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Save detailed results
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()

