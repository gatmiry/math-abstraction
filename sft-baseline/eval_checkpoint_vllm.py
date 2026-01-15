#!/usr/bin/env python3
"""
Evaluate a saved SFT checkpoint on the math evaluation dataset using vLLM for fast inference.

Loads a checkpoint and computes accuracy based on \boxed{} answer extraction.
"""

import os
import re
import sys
import json
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm

# Paths
DEFAULT_CHECKPOINT = "/mnt/task_wrapper/user_output/artifacts/sft_baseline_20260112_184137_from_qwen2-math-7b-instruct/global_step_100"
EVAL_DATASET_PATH = "../newopenaioutputs/hints_dataset"
VAL_SIZE = 256

# System prompt for evaluation
EVAL_SYSTEM_PROMPT = """You are a mathematics expert. Solve the given problem step by step, showing all your work and reasoning. Put your final answer in \\box{...} format at the end."""


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} at the end of text."""
    matches = list(re.finditer(r'\\boxed?\{', text))
    if not matches:
        return None
    
    start_pos = matches[-1].end()
    depth = 1
    i = start_pos
    
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return text[start_pos:i-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Basic normalization: strip whitespace and convert to lowercase
    normalized = answer.strip().lower()
    # Remove common LaTeX formatting
    normalized = normalized.replace("\\text{", "").replace("}", "")
    normalized = normalized.replace("\\mathrm{", "")
    normalized = normalized.replace("\\", "")
    normalized = normalized.replace(" ", "")
    return normalized


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Compute accuracy of predictions vs ground truths."""
    correct = 0
    results = []
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_answer = extract_boxed_answer(pred)
        pred_norm = normalize_answer(pred_answer)
        gt_norm = normalize_answer(gt)
        
        is_correct = pred_norm and gt_norm and pred_norm == gt_norm
        if is_correct:
            correct += 1
        
        results.append({
            "index": i,
            "ground_truth": gt,
            "predicted_answer": pred_answer,
            "gt_normalized": gt_norm,
            "pred_normalized": pred_norm,
            "correct": is_correct
        })
    
    accuracy = correct / len(predictions) if predictions else 0.0
    
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_samples": len(predictions),
        "results": results
    }


def create_eval_dataset(dataset_path: str, val_size: int = 256) -> List[Dict]:
    """Create evaluation dataset from hints_dataset (holdout).
    
    Uses the same random split as sft_hf.py (test_size=0.1, seed=42) to ensure
    no data leakage between training and evaluation.
    """
    from datasets import load_from_disk
    
    dataset = load_from_disk(dataset_path)
    
    # Use the same split as sft_hf.py to avoid data leakage
    # sft_hf.py uses: train_test_split(test_size=0.1, seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = split['test']
    
    print(f"[INFO] Using random split (test_size=0.1, seed=42) - same as training script")
    print(f"[INFO] Eval set size: {len(eval_dataset)} (val_size argument ignored)")
    
    eval_data = []
    for item in tqdm(eval_dataset, desc="Loading eval data"):
        problem = item.get('problem', '')
        final_answer = item.get('final_answer', '')
        
        if not problem or not final_answer:
            continue
        
        # Format prompt for generation
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"}
        ]
        
        eval_data.append({
            "messages": messages,
            "problem": problem,
            "ground_truth": final_answer
        })
    
    print(f"[INFO] Loaded {len(eval_data)} eval samples")
    return eval_data


def run_evaluation_vllm(
    model_path: str,
    eval_data: List[Dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
) -> Dict:
    """Run evaluation using vLLM for fast inference."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    # Load tokenizer for chat template
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Prepare all prompts
    print("Preparing prompts...")
    prompts = []
    ground_truths = []
    problems = []
    
    for item in tqdm(eval_data, desc="Formatting prompts"):
        prompt = tokenizer.apply_chat_template(
            item['messages'],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        ground_truths.append(item['ground_truth'])
        problems.append(item['problem'])
    
    # Initialize vLLM
    print(f"\nInitializing vLLM with model: {model_path}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    
    # Set up sampling parameters
    if temperature == 0:
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,
            top_p=1.0,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
        )
    
    # Generate all at once (vLLM handles batching internally)
    print(f"\nGenerating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract generated text
    predictions = []
    for output in outputs:
        generated_text = output.outputs[0].text
        predictions.append(generated_text)
    
    # Compute accuracy
    print("\nComputing accuracy...")
    metrics = compute_accuracy(predictions, ground_truths)
    
    # Add problems and full predictions to results
    for i, result in enumerate(metrics["results"]):
        result["problem"] = problems[i][:200] + "..." if len(problems[i]) > 200 else problems[i]
        result["full_prediction"] = predictions[i][:500] + "..." if len(predictions[i]) > 500 else predictions[i]
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT checkpoint using vLLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to the checkpoint to evaluate"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=VAL_SIZE,
        help="Number of evaluation samples"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for detailed results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("SFT Checkpoint Evaluation (vLLM)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eval samples: {args.eval_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 80)
    
    # Check checkpoint exists (skip check for HuggingFace model IDs)
    if not os.path.exists(args.checkpoint) and '/' not in args.checkpoint:
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create eval dataset
    print("\nLoading evaluation dataset...")
    eval_data = create_eval_dataset(EVAL_DATASET_PATH, val_size=args.eval_samples)
    
    # Run evaluation with vLLM
    metrics = run_evaluation_vllm(
        model_path=args.checkpoint,
        eval_data=eval_data,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['num_correct']}/{metrics['num_samples']})")
    print("=" * 80)
    
    # Print sample results
    print("\nSample Results (first 10):")
    print("-" * 80)
    for result in metrics["results"][:10]:
        status = "✓" if result["correct"] else "✗"
        print(f"{status} [{result['index']}] GT: {result['ground_truth'][:30]:30s} | Pred: {result['predicted_answer']}")
    
    # Count correct vs incorrect
    correct_examples = [r for r in metrics["results"] if r["correct"]]
    incorrect_examples = [r for r in metrics["results"] if not r["correct"]]
    
    print(f"\nCorrect: {len(correct_examples)}")
    print(f"Incorrect: {len(incorrect_examples)}")
    
    # Save detailed results if requested
    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "accuracy": metrics["accuracy"],
            "num_correct": metrics["num_correct"],
            "num_samples": metrics["num_samples"],
            "config": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "tensor_parallel_size": args.tensor_parallel_size
            },
            "results": metrics["results"]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[INFO] Detailed results saved to: {args.output}")
    
    print("\nEvaluation complete!")
    return metrics


if __name__ == "__main__":
    main()



