#!/usr/bin/env python3
"""
Iterative Hint Distillation: Repeat distillation loop for multiple rounds.

Each round:
1. Generate proofs with current model using vLLM
2. Find pairs where model fails on weak hint but succeeds on strong hint
3. Finetune model on these pairs (weak hint context -> strong hint response)
4. Use finetuned model for next round
"""

import os
import re
import json
import argparse
import shutil
import gc
from typing import List, Dict, Optional
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import torch
import apple_bolt as bolt

# Paths
HINTS_DATASET = "../newopenaioutputs/hints_dataset"
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = os.path.join(bolt.ARTIFACT_DIR, "iterative_distillation")
TEST_SIZE = 512
SAVE_STEPS = 50  # Save checkpoint every N steps

SYSTEM_PROMPT_WITH_HINT = """You are a mathematics problem solver. You will be given a problem, the beginning of a solution, and a hint. Your task is to CONTINUE the solution EXACTLY from where it left off - do not restart or repeat what's already written. Pick up the reasoning from the last sentence and complete the proof.

- Continue directly from the partial solution provided - do not restate the problem or start over
- Use the hint to guide your continuation
- Show clear step-by-step reasoning for all remaining steps
- End with the final answer in \\box{...} format"""

SYSTEM_PROMPT_NO_HINT = """You are a mathematics problem solver. You will be given a problem and the beginning of a solution. Your task is to CONTINUE the solution EXACTLY from where it left off - do not restart or repeat what's already written. Pick up the reasoning from the last sentence and complete the proof.

- Continue directly from the partial solution provided - do not restate the problem or start over
- Show clear step-by-step reasoning for all remaining steps
- End with the final answer in \\box{...} format"""


def format_prompt(problem: str, partial_proof: str, hint: str = None) -> List[Dict[str, str]]:
    """Format a prompt with optional hint, explicitly asking model to continue from partial proof."""
    if hint:
        system_prompt = SYSTEM_PROMPT_WITH_HINT
        user_content = f"""Problem: {problem}

Here is the beginning of a solution. Continue EXACTLY from where it leaves off (do not restart):

{partial_proof}

Hint for continuation: {hint}

Now continue the solution from this point:"""
    else:
        system_prompt = SYSTEM_PROMPT_NO_HINT
        user_content = f"""Problem: {problem}

Here is the beginning of a solution. Continue EXACTLY from where it leaves off (do not restart):

{partial_proof}

Now continue the solution from this point:"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\box{...} or \\boxed{...} using proper brace matching."""
    # Find all occurrences of \box{ or \boxed{
    matches = list(re.finditer(r'\\box(ed)?\{', text))
    if not matches:
        return None
    
    # Use the last occurrence (final answer is typically at the end)
    start_pos = matches[-1].end()
    depth = 1
    i = start_pos
    
    # Match braces properly to handle nested expressions like \frac{...}{...}
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
    """Normalize answer for basic string comparison."""
    if answer is None:
        return ""
    ans = answer.strip()
    ans = re.sub(r'\s+', '', ans)
    ans = ans.replace('\\left', '').replace('\\right', '')
    ans = ans.replace('\\,', '').replace('\\;', '').replace('\\:', '')
    return ans.lower()


def parse_latex_to_sympy(latex_str: str):
    """Try to parse LaTeX string to SymPy expression using latex2sympy2."""
    try:
        from latex2sympy2 import latex2sympy
        
        # Clean up the LaTeX string
        latex_str = latex_str.strip()
        # Handle common LaTeX patterns
        latex_str = latex_str.replace('\\dfrac', '\\frac')
        latex_str = latex_str.replace('\\tfrac', '\\frac')
        
        expr = latex2sympy(latex_str)
        return expr
    except Exception:
        return None


def check_answer(generated: str, ground_truth: str) -> bool:
    """Check if generated answer matches ground truth using symbolic comparison."""
    gen_ans = extract_boxed_answer(generated)
    if gen_ans is None:
        return False
    
    # First try exact string match (after normalization)
    if normalize_answer(gen_ans) == normalize_answer(ground_truth):
        return True
    
    # Try symbolic comparison using SymPy
    try:
        from sympy import simplify, N
        
        gen_expr = parse_latex_to_sympy(gen_ans)
        gt_expr = parse_latex_to_sympy(ground_truth)
        
        if gen_expr is not None and gt_expr is not None:
            # Check if expressions are symbolically equal
            diff = simplify(gen_expr - gt_expr)
            if diff == 0:
                return True
            
            # Try numerical comparison for expressions that can't be simplified symbolically
            try:
                gen_val = complex(N(gen_expr))
                gt_val = complex(N(gt_expr))
                if abs(gen_val - gt_val) < 1e-9:
                    return True
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    
    return False


def generate_and_find_pairs(model_path: str, dataset, args):
    """Generate responses and find distillation pairs."""
    from vllm import LLM, SamplingParams
    
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    
    print(f"Loading model {model_path} with tensor_parallel_size=4")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # Prepare prompts
    all_prompts = []
    prompt_metadata = []

    for prob_idx, row in enumerate(tqdm(dataset, desc="Preparing prompts")):
        problem = row["problem"]
        partial_proof = row["partial_proof"]
        hints = row["hints"]
        ground_truth = row["final_answer"]
        
        if not hints:
            continue
            
        hints_strong_to_weak = list(reversed(hints))
        
        for hint_rank, hint in enumerate(hints_strong_to_weak):
            messages = format_prompt(problem, partial_proof, hint)
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(prompt_text)
            prompt_metadata.append((prob_idx, hint_rank, hint, ground_truth))

    # Generate responses
    print(f"Generating {len(all_prompts)} responses...")
    outputs = llm.generate(all_prompts, sampling_params)
    
    # Organize results
    results_by_problem = {}
    for output, (prob_idx, hint_rank, hint, gt) in zip(outputs, prompt_metadata):
        response = output.outputs[0].text
        correct = check_answer(response, gt)
        
        if prob_idx not in results_by_problem:
            results_by_problem[prob_idx] = []
        results_by_problem[prob_idx].append({
            "hint_rank": hint_rank,
            "hint": hint,
            "response": response,
            "correct": correct,
        })

    # Find distillation pairs
    distillation_pairs = []
    stats = {"total_problems": len(dataset), "pairs_found": 0, "all_correct": 0, "none_correct": 0}

    for prob_idx, row in enumerate(dataset):
        if prob_idx not in results_by_problem:
            continue
        problem = row["problem"]
        partial_proof = row["partial_proof"]
        results = results_by_problem[prob_idx]
        
        results.sort(key=lambda x: x["hint_rank"])
        
        weakest_correct_rank = -1
        for r in results:
            if r["correct"]:
                weakest_correct_rank = r["hint_rank"]
        
        if weakest_correct_rank == -1:
            stats["none_correct"] += 1
            continue
        
        if weakest_correct_rank == len(results) - 1:
            stats["all_correct"] += 1
            continue
        
        correct_result = results[weakest_correct_rank]
        incorrect_result = results[weakest_correct_rank + 1]
        
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

    # Clean up vLLM
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return distillation_pairs, stats


def evaluate_on_holdout(model_path: str, full_dataset, test_indices: List[int], args, 
                        eval_weakest: bool = False, eval_strongest: bool = False):
    """Evaluate model on holdout test set. Returns results dict and list of generations.
    
    By default only evaluates without hints. Use eval_weakest and eval_strongest flags
    to also evaluate with weakest/strongest hints.
    """
    from vllm import LLM, SamplingParams
    
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    
    # Determine which hint types to evaluate
    hint_types_to_eval = [("no_hint", None)]  # Always evaluate no_hint
    if eval_weakest:
        hint_types_to_eval.append(("weakest", "weakest"))
    if eval_strongest:
        hint_types_to_eval.append(("strongest", "strongest"))
    
    hint_type_names = [ht[0] for ht in hint_types_to_eval]
    print(f"Evaluating model {model_path} on {len(test_indices)} holdout problems...")
    print(f"  Hint types: {hint_type_names}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Prepare prompts based on selected hint types
    all_prompts = []
    prompt_metadata = []  # (test_idx, hint_type, ground_truth, problem, hint, partial_proof)

    for test_idx in tqdm(test_indices, desc="Preparing eval prompts"):
        row = full_dataset[test_idx]
        problem = row["problem"]
        partial_proof = row["partial_proof"]
        hints = row["hints"]
        ground_truth = row["final_answer"]

        if not hints:
            continue

        # Build list of (hint_type_name, actual_hint) for this problem
        prompts_for_problem = []
        for hint_type_name, hint_marker in hint_types_to_eval:
            if hint_marker is None:
                actual_hint = None
            elif hint_marker == "weakest":
                actual_hint = hints[0]
            elif hint_marker == "strongest":
                actual_hint = hints[-1]
            else:
                actual_hint = None
            prompts_for_problem.append((hint_type_name, actual_hint))
        
        for hint_type, hint in prompts_for_problem:
            messages = format_prompt(problem, partial_proof, hint)
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(prompt_text)
            prompt_metadata.append((test_idx, hint_type, ground_truth, problem, hint, partial_proof))

    # Generate responses
    print(f"Generating {len(all_prompts)} eval responses...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Evaluate and collect generations
    results = {}
    for hint_type_name in hint_type_names:
        results[hint_type_name] = {"correct": 0, "total": 0}
    generations = []  # List of all generations with metadata

    for output, (test_idx, hint_type, gt, problem, hint, partial_proof) in zip(outputs, prompt_metadata):
        response = output.outputs[0].text
        predicted_answer = extract_boxed_answer(response)
        correct = check_answer(response, gt)
        results[hint_type]["total"] += 1
        if correct:
            results[hint_type]["correct"] += 1
        
        generations.append({
            "test_idx": test_idx,
            "hint_type": hint_type,
            "problem": problem,
            "partial_proof": partial_proof,
            "hint": hint,
            "ground_truth": gt,
            "predicted_answer": predicted_answer,
            "response": response,
            "correct": correct,
        })

    # Calculate accuracies
    for hint_type in hint_type_names:
        total = results[hint_type]["total"]
        correct = results[hint_type]["correct"]
        results[hint_type]["accuracy"] = correct / total if total > 0 else 0.0

    # Clean up
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results, generations


def finetune_model(model_path: str, distillation_pairs: List[Dict], output_path: str, args):
    """Finetune model on distillation pairs using subprocess with torchrun."""
    import subprocess
    
    if not distillation_pairs:
        print("No pairs to finetune on!")
        return model_path
    
    # Prepare and save dataset for finetune.py
    sft_data = []
    for pair in distillation_pairs:
        sft_data.append({
            "messages": pair["messages"] + [{"role": "assistant", "content": pair["target_response"]}],
        })
    
    sft_dataset = Dataset.from_list(sft_data)
    temp_dataset_path = os.path.join(output_path, "temp_sft_dataset")
    sft_dataset.save_to_disk(temp_dataset_path)
    print(f"Saved {len(sft_dataset)} pairs to {temp_dataset_path}")
    
    # Call finetune.py with torchrun
    final_path = os.path.join(output_path, "final")
    finetune_script = os.path.join(os.path.dirname(__file__), "finetune.py")
    
    cmd = [
        "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=8",
        finetune_script,
        "--model", model_path,
        "--data", temp_dataset_path,
        "--output", output_path,
        "--epochs", str(args.epochs_per_round),
        "--lr", str(args.lr),
    ]
    
    print(f"Running finetuning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print(f"Finetuning failed with return code {result.returncode}")
        return model_path
    
    print(f"Finetuning complete. Model saved to {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--dataset", type=str, default=HINTS_DATASET)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--rounds", type=int, default=5, help="Number of distillation rounds")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-per-round", type=int, default=1, help="Epochs per finetuning round")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS, help="Save checkpoint every N steps")
    parser.add_argument("--eval-weakest", action="store_true", help="Also evaluate with weakest hint on holdout set")
    parser.add_argument("--eval-strongest", action="store_true", help="Also evaluate with strongest hint on holdout set")
    parser.add_argument("--test-freq", type=int, default=1, help="Evaluate on holdout every N rounds (default: every round)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # Create distillation_performance folders
    perf_dir = os.path.join(os.path.dirname(__file__), "distillation_performance")
    generations_dir = os.path.join(perf_dir, "mid_generations")
    os.makedirs(perf_dir, exist_ok=True)
    os.makedirs(generations_dir, exist_ok=True)

    # Load and split dataset
    print(f"Loading dataset from {args.dataset}")
    full_dataset = load_from_disk(args.dataset)
    if args.limit:
        full_dataset = full_dataset.select(range(min(args.limit, len(full_dataset))))
    print(f"Loaded {len(full_dataset)} problems")

    import random
    random.seed(args.seed)
    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)
    
    test_indices = all_indices[:args.test_size]
    train_indices = all_indices[args.test_size:]
    
    # Save test indices
    test_indices_path = os.path.join(args.output, "test_indices.json")
    with open(test_indices_path, "w") as f:
        json.dump(test_indices, f)
    
    dataset = full_dataset.select(train_indices)
    print(f"Train set: {len(dataset)} problems, Test set: {len(test_indices)} problems")

    # Iterative distillation loop
    current_model = args.model
    all_round_stats = []

    # Evaluate base model (round 0) before any training
    print(f"\n{'='*60}")
    print(f"ROUND 0 (BASE MODEL EVALUATION)")
    print(f"{'='*60}")
    print(f"Evaluating base model: {current_model}")
    
    base_eval_results, base_generations = evaluate_on_holdout(
        current_model, full_dataset, test_indices, args,
        eval_weakest=args.eval_weakest, eval_strongest=args.eval_strongest
    )
    
    base_stats = {
        "round": 0,
        "model": current_model,
        "pairs_found": 0,
        "eval_no_hint_acc": base_eval_results["no_hint"]["accuracy"],
        "eval_no_hint_correct": base_eval_results["no_hint"]["correct"],
        "eval_no_hint_total": base_eval_results["no_hint"]["total"],
    }
    if args.eval_weakest:
        base_stats["eval_weakest_acc"] = base_eval_results["weakest"]["accuracy"]
        base_stats["eval_weakest_correct"] = base_eval_results["weakest"]["correct"]
        base_stats["eval_weakest_total"] = base_eval_results["weakest"]["total"]
    if args.eval_strongest:
        base_stats["eval_strongest_acc"] = base_eval_results["strongest"]["accuracy"]
        base_stats["eval_strongest_correct"] = base_eval_results["strongest"]["correct"]
        base_stats["eval_strongest_total"] = base_eval_results["strongest"]["total"]
    all_round_stats.append(base_stats)
    
    print(f"[Round 0] Base Model Eval Results:")
    print(f"  NO HINT: {base_eval_results['no_hint']['correct']}/{base_eval_results['no_hint']['total']} = {base_eval_results['no_hint']['accuracy']*100:.1f}%")
    if args.eval_weakest:
        print(f"  WEAKEST hint: {base_eval_results['weakest']['correct']}/{base_eval_results['weakest']['total']} = {base_eval_results['weakest']['accuracy']*100:.1f}%")
    if args.eval_strongest:
        print(f"  STRONGEST hint: {base_eval_results['strongest']['correct']}/{base_eval_results['strongest']['total']} = {base_eval_results['strongest']['accuracy']*100:.1f}%")
    
    # Save base model performance and generations
    base_perf = {
        "round": 0,
        "model": current_model,
        "pairs_found": 0,
        "no_hint_accuracy": base_eval_results["no_hint"]["accuracy"],
        "no_hint_correct": base_eval_results["no_hint"]["correct"],
        "no_hint_total": base_eval_results["no_hint"]["total"],
    }
    if args.eval_weakest:
        base_perf["weakest_accuracy"] = base_eval_results["weakest"]["accuracy"]
        base_perf["weakest_correct"] = base_eval_results["weakest"]["correct"]
        base_perf["weakest_total"] = base_eval_results["weakest"]["total"]
    if args.eval_strongest:
        base_perf["strongest_accuracy"] = base_eval_results["strongest"]["accuracy"]
        base_perf["strongest_correct"] = base_eval_results["strongest"]["correct"]
        base_perf["strongest_total"] = base_eval_results["strongest"]["total"]
    perf_path = os.path.join(perf_dir, "round_0_performance.json")
    with open(perf_path, "w") as f:
        json.dump(base_perf, f, indent=2)
    print(f"  Performance saved to {perf_path}")
    
    gen_path = os.path.join(generations_dir, "round_0_generations.json")
    with open(gen_path, "w") as f:
        json.dump(base_generations, f, indent=2)
    print(f"  Generations saved to {gen_path}")

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{args.rounds}")
        print(f"{'='*60}")
        print(f"Current model: {current_model}")
        
        round_dir = os.path.join(args.output, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # Step 1: Generate and find pairs
        print(f"\n[Round {round_num}] Step 1: Generating responses and finding pairs...")
        pairs, stats = generate_and_find_pairs(current_model, dataset, args)
        
        stats["round"] = round_num
        stats["model"] = current_model
        all_round_stats.append(stats)
        
        print(f"\n[Round {round_num}] Stats: {stats}")
        
        # Save pairs for this round
        pairs_path = os.path.join(round_dir, "distillation_pairs.json")
        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"Saved {len(pairs)} pairs to {pairs_path}")
        
        if not pairs:
            print(f"[Round {round_num}] No pairs found, stopping early.")
            break
        
        # Step 2: Finetune
        print(f"\n[Round {round_num}] Step 2: Finetuning on {len(pairs)} pairs...")
        model_output_dir = os.path.join(round_dir, "model")
        new_model_path = finetune_model(current_model, pairs, model_output_dir, args)
        
        current_model = new_model_path
        print(f"[Round {round_num}] New model: {current_model}")
        
        # Step 3: Evaluate on holdout set (only every test_freq rounds)
        should_eval = (round_num % args.test_freq == 0) or (round_num == args.rounds)
        if should_eval:
            print(f"\n[Round {round_num}] Step 3: Evaluating on holdout set...")
            eval_results, eval_generations = evaluate_on_holdout(
                current_model, full_dataset, test_indices, args,
                eval_weakest=args.eval_weakest, eval_strongest=args.eval_strongest
            )
            
            # Add eval results to stats
            stats["eval_no_hint_acc"] = eval_results["no_hint"]["accuracy"]
            stats["eval_no_hint_correct"] = eval_results["no_hint"]["correct"]
            stats["eval_no_hint_total"] = eval_results["no_hint"]["total"]
            if args.eval_weakest:
                stats["eval_weakest_acc"] = eval_results["weakest"]["accuracy"]
                stats["eval_weakest_correct"] = eval_results["weakest"]["correct"]
                stats["eval_weakest_total"] = eval_results["weakest"]["total"]
            if args.eval_strongest:
                stats["eval_strongest_acc"] = eval_results["strongest"]["accuracy"]
                stats["eval_strongest_correct"] = eval_results["strongest"]["correct"]
                stats["eval_strongest_total"] = eval_results["strongest"]["total"]
            
            print(f"[Round {round_num}] Eval Results:")
            print(f"  NO HINT: {eval_results['no_hint']['correct']}/{eval_results['no_hint']['total']} = {eval_results['no_hint']['accuracy']*100:.1f}%")
            if args.eval_weakest:
                print(f"  WEAKEST hint: {eval_results['weakest']['correct']}/{eval_results['weakest']['total']} = {eval_results['weakest']['accuracy']*100:.1f}%")
            if args.eval_strongest:
                print(f"  STRONGEST hint: {eval_results['strongest']['correct']}/{eval_results['strongest']['total']} = {eval_results['strongest']['accuracy']*100:.1f}%")
            
            # Save eval results for this round
            eval_path = os.path.join(round_dir, "eval_results.json")
            with open(eval_path, "w") as f:
                json.dump(eval_results, f, indent=2)
            
            # Save performance and generations to distillation_performance folder
            round_perf = {
                "round": round_num,
                "model": current_model,
                "pairs_found": stats["pairs_found"],
                "no_hint_accuracy": eval_results["no_hint"]["accuracy"],
                "no_hint_correct": eval_results["no_hint"]["correct"],
                "no_hint_total": eval_results["no_hint"]["total"],
            }
            if args.eval_weakest:
                round_perf["weakest_accuracy"] = eval_results["weakest"]["accuracy"]
                round_perf["weakest_correct"] = eval_results["weakest"]["correct"]
                round_perf["weakest_total"] = eval_results["weakest"]["total"]
            if args.eval_strongest:
                round_perf["strongest_accuracy"] = eval_results["strongest"]["accuracy"]
                round_perf["strongest_correct"] = eval_results["strongest"]["correct"]
                round_perf["strongest_total"] = eval_results["strongest"]["total"]
            perf_path = os.path.join(perf_dir, f"round_{round_num}_performance.json")
            with open(perf_path, "w") as f:
                json.dump(round_perf, f, indent=2)
            print(f"  Performance saved to {perf_path}")
            
            # Save generations to mid_generations folder
            gen_path = os.path.join(generations_dir, f"round_{round_num}_generations.json")
            with open(gen_path, "w") as f:
                json.dump(eval_generations, f, indent=2)
            print(f"  Generations saved to {gen_path}")
        else:
            print(f"\n[Round {round_num}] Skipping evaluation (test_freq={args.test_freq})")

    # Save summary
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "rounds": args.rounds,
            "final_model": current_model,
            "round_stats": all_round_stats,
        }, f, indent=2)
    
    # Save consolidated evaluation results to hint-distillation folder
    eval_summary_path = os.path.join(os.path.dirname(__file__), "eval_summary.json")
    eval_summary = {
        "rounds": [],
        "final_model": current_model,
    }
    for s in all_round_stats:
        if "eval_no_hint_acc" in s:
            round_summary = {
                "round": s["round"],
                "pairs_found": s["pairs_found"],
                "no_hint_acc": s["eval_no_hint_acc"],
                "no_hint_correct": s["eval_no_hint_correct"],
                "no_hint_total": s["eval_no_hint_total"],
            }
            if "eval_weakest_acc" in s:
                round_summary["weakest_acc"] = s["eval_weakest_acc"]
                round_summary["weakest_correct"] = s["eval_weakest_correct"]
                round_summary["weakest_total"] = s["eval_weakest_total"]
            if "eval_strongest_acc" in s:
                round_summary["strongest_acc"] = s["eval_strongest_acc"]
                round_summary["strongest_correct"] = s["eval_strongest_correct"]
                round_summary["strongest_total"] = s["eval_strongest_total"]
            eval_summary["rounds"].append(round_summary)
    with open(eval_summary_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Evaluation summary saved to {eval_summary_path}")
    
    print(f"\n{'='*60}")
    print("ITERATIVE DISTILLATION COMPLETE")
    print(f"{'='*60}")
    print(f"Final model: {current_model}")
    print(f"Summary saved to {summary_path}")
    
    print("\nRound-by-round stats:")
    for s in all_round_stats:
        pairs_info = f"{s['pairs_found']} pairs"
        if "eval_no_hint_acc" in s:
            eval_parts = [f"no_hint={s['eval_no_hint_acc']*100:.1f}%"]
            if "eval_weakest_acc" in s:
                eval_parts.append(f"weak={s['eval_weakest_acc']*100:.1f}%")
            if "eval_strongest_acc" in s:
                eval_parts.append(f"strong={s['eval_strongest_acc']*100:.1f}%")
            eval_info = f"eval: {', '.join(eval_parts)}"
            print(f"  Round {s['round']}: {pairs_info}, {eval_info}")
        else:
            print(f"  Round {s['round']}: {pairs_info}")


if __name__ == "__main__":
    main()

