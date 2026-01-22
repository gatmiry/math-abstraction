#!/usr/bin/env python3
"""
Evaluate base model on the exact 8 validation problems from sbys_grpo.py.
Generates multiple samples per problem and computes best@8 and best@16 accuracy.
"""

import os
import sys
import random
from datasets import load_from_disk

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import math_checker from sbys_hinting directory
import importlib.util
_sbys_hinting_dir = os.path.dirname(os.path.dirname(__file__))  # sbys_hinting/
_math_checker_path = os.path.join(_sbys_hinting_dir, "math_checker.py")
_spec = importlib.util.spec_from_file_location("math_checker", _math_checker_path)
_math_checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math_checker)
check_answer = _math_checker.check_answer
extract_boxed_answer = _math_checker.extract_boxed_answer

# Configuration - must match sbys_grpo.py exactly
DATASET_PATH = os.path.join(_sbys_hinting_dir, "outputs", "hint_helped_dataset", "hint_helped_dataset")
VAL_SIZE = 8
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT_FILE = os.path.join(_sbys_hinting_dir, "system_prompt_full_solution.txt")

def load_system_prompt(name: str = "full_solution_simple"):
    """Load a named system prompt from file."""
    with open(SYSTEM_PROMPT_FILE, 'r') as f:
        content = f.read()
    
    prompts = {}
    current_name = None
    current_lines = []
    
    for line in content.split('\n'):
        if line.startswith('===PROMPT:') and line.endswith('==='):
            if current_name is not None:
                prompts[current_name] = '\n'.join(current_lines).strip()
            current_name = line[10:-3].strip()
            current_lines = []
        else:
            current_lines.append(line)
    
    if current_name is not None:
        prompts[current_name] = '\n'.join(current_lines).strip()
    
    return prompts.get(name, prompts.get("full_solution_simple", "You are a helpful math assistant."))


def get_validation_problems():
    """Get the exact 8 validation problems using the same split logic as sbys_grpo.py."""
    # Load dataset
    dataset = load_from_disk(DATASET_PATH)
    
    # Deduplicate by problem text (same as sbys_grpo.py)
    original_size = len(dataset)
    seen_problems = set()
    unique_indices = []
    for idx, example in enumerate(dataset):
        problem = example['problem']
        if problem not in seen_problems:
            seen_problems.add(problem)
            unique_indices.append(idx)
    if len(unique_indices) < original_size:
        dataset = dataset.select(unique_indices)
        print(f"[INFO] Deduplicated dataset: {original_size} -> {len(dataset)}")
    
    # Use the same random split as sbys_grpo.py
    total_size = len(dataset)
    val_size_actual = min(VAL_SIZE, total_size)
    
    indices = list(range(total_size))
    random.Random(42).shuffle(indices)  # Same seed as sbys_grpo.py
    val_indices = indices[:val_size_actual]
    
    val_dataset = dataset.select(val_indices)
    
    # Extract problems and answers
    val_problems = []
    for item in val_dataset:
        val_problems.append({
            "problem": item["problem"],
            "answer": item["answer"],
            "sbys_solution": item.get("sbys_solution", []),
        })
    
    return val_problems


def format_prompt(problem: str, system_prompt: str, sbys_solution: list = None):
    """Format prompt for the model with hints (all steps except last).
    
    Args:
        problem: The math problem
        system_prompt: System prompt for the model
        sbys_solution: List of solution steps. We include all except the last 
                       (which contains the final answer) as hints.
    """
    # Include all steps except the last one as hints
    if sbys_solution and len(sbys_solution) > 1:
        hint_steps = sbys_solution[:-1]  # All except last step
        partial_answer = "\n".join(hint_steps)
        user_content = (
            f"Problem: {problem}\n"
            f"Incomplete proof: {partial_answer}\n"
        )
        print(f"  [Hint] Using {len(hint_steps)}/{len(sbys_solution)} steps as hints")
    else:
        user_content = f"Problem: {problem}"
        print(f"  [Hint] No hints available (sbys_solution has {len(sbys_solution) if sbys_solution else 0} steps)")
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def generate_solutions_vllm(problems, system_prompt, n_samples=16):
    """Generate solutions using vLLM for efficiency."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    print(f"\n[INFO] Loading model {MODEL_PATH}...")
    
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Initialize vLLM
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=10752,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=8192,
    )
    
    # Prepare prompts with hints (all sbys_solution steps except last)
    prompts = []
    for i, p in enumerate(problems):
        print(f"\nPreparing prompt for problem {i+1}:")
        messages = format_prompt(p["problem"], system_prompt, p.get("sbys_solution", []))
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)
    
    print(f"[INFO] Generating {n_samples} solutions for each of {len(problems)} problems...")
    
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract solutions per problem
    all_solutions = []
    for output in outputs:
        solutions = [o.text for o in output.outputs]
        all_solutions.append(solutions)
    
    return all_solutions


def evaluate_solutions(problems, all_solutions):
    """Evaluate solutions and compute metrics."""
    results = []
    
    for i, (problem_data, solutions) in enumerate(zip(problems, all_solutions)):
        problem = problem_data["problem"]
        ground_truth = problem_data["answer"]
        
        # Check each solution
        correctness = []
        for j, sol in enumerate(solutions):
            is_correct = check_answer(sol, ground_truth)
            boxed = extract_boxed_answer(sol)
            correctness.append({
                "index": j,
                "correct": is_correct,
                "boxed_answer": boxed,
            })
        
        # Compute best@K
        n_correct = sum(1 for c in correctness if c["correct"])
        best_at_1 = 1 if correctness[0]["correct"] else 0
        best_at_8 = 1 if any(c["correct"] for c in correctness[:8]) else 0
        best_at_16 = 1 if any(c["correct"] for c in correctness[:16]) else 0
        
        results.append({
            "problem_idx": i,
            "problem": problem[:100] + "...",
            "ground_truth": ground_truth,
            "n_samples": len(solutions),
            "n_correct": n_correct,
            "best@1": best_at_1,
            "best@8": best_at_8,
            "best@16": best_at_16,
            "sample_correctness": correctness,
        })
        
        print(f"\nProblem {i+1}:")
        print(f"  Ground truth: {ground_truth[:50]}...")
        print(f"  Correct samples: {n_correct}/{len(solutions)}")
        print(f"  best@1: {best_at_1}, best@8: {best_at_8}, best@16: {best_at_16}")
        
        # Show some boxed answers
        unique_answers = {}
        for c in correctness:
            ans = c["boxed_answer"] or "None"
            if ans not in unique_answers:
                unique_answers[ans] = {"count": 0, "correct": c["correct"]}
            unique_answers[ans]["count"] += 1
        
        print(f"  Unique answers:")
        for ans, info in sorted(unique_answers.items(), key=lambda x: -x[1]["count"])[:5]:
            status = "✓" if info["correct"] else "✗"
            print(f"    {status} '{ans[:40]}...' (count: {info['count']})")
    
    return results


def main():
    print("=" * 80)
    print("Base Model Evaluation on Validation Problems")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Validation size: {VAL_SIZE}")
    print("=" * 80)
    
    # Load system prompt
    system_prompt = load_system_prompt("full_solution_simple")
    print(f"\nSystem prompt loaded ({len(system_prompt)} chars)")
    
    # Get validation problems
    print("\n[INFO] Loading validation problems...")
    val_problems = get_validation_problems()
    print(f"[INFO] Loaded {len(val_problems)} validation problems")
    
    for i, p in enumerate(val_problems):
        print(f"  {i+1}. {p['problem'][:80]}...")
        print(f"     Answer: {p['answer'][:50]}...")
    
    # Generate solutions (16 per problem for best@16)
    n_samples = 16
    all_solutions = generate_solutions_vllm(val_problems, system_prompt, n_samples=n_samples)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    results = evaluate_solutions(val_problems, all_solutions)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    n_problems = len(results)
    sum_best1 = sum(r["best@1"] for r in results)
    sum_best8 = sum(r["best@8"] for r in results)
    sum_best16 = sum(r["best@16"] for r in results)
    
    print(f"\nTotal problems: {n_problems}")
    print(f"best@1:  {sum_best1}/{n_problems} = {sum_best1/n_problems:.3f}")
    print(f"best@8:  {sum_best8}/{n_problems} = {sum_best8/n_problems:.3f}")
    print(f"best@16: {sum_best16}/{n_problems} = {sum_best16/n_problems:.3f}")
    
    print("\nNote: These are TRUE best@K metrics (max of K samples per problem),")
    print("      not the bootstrap estimates that verl uses.")


if __name__ == "__main__":
    main()

