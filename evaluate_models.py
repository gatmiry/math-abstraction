#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen model vs baseline on math_solutions_dataset_2000.
Compares how many times each model gets the correct answer.

Uses vLLM for faster inference if available, otherwise falls back to transformers.
"""

import os
import json
import re
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Try to import vLLM for faster inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Always import transformers components
from transformers import AutoModelForCausalLM

# Configuration
BASELINE_MODEL = "Qwen/Qwen2.5-7B"
#FINETUNED_MODEL = "./math-abstraction/models/qwen_finetuned/qwen_finetuned"
FINETUNED_MODEL = "./models/qwen2-math-7b-instruct_finetuned_on_first_3542_transformed_omni_math_solutions"
DATASET_PATH = None  # Set to None to load from HuggingFace, or path to dataset
DATASET_NAME = "qwedsacf/competition_math"  # HuggingFace dataset name
PROBLEM_TYPE = "Algebra"  # Filter for this problem type
PROBLEM_LEVEL = "Level 4"  # Filter for this difficulty level
MAX_NEW_TOKENS = 1024
MAX_SAMPLES = None  # Set to a number to limit evaluation (e.g., 50 for quick test), None for all
USE_VLLM = True  # Use vLLM for faster inference with tensor parallelism
TENSOR_PARALLEL_SIZE = 4  # Use 4 GPUs per model (so both models can run on 8 GPUs total)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_answer(text):
    """Extract answer from text, looking for \\boxed{...} patterns."""
    if not text:
        return None
    
    # Try to find boxed{...} pattern (most flexible - works with or without backslash)
    # This handles cases where \boxed becomes boxed due to string processing
    boxed_pattern = r'boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()  # Return the last match
    
    # Try to find \boxed{...} pattern (with escaped backslash)
    boxed_pattern2 = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern2, text)
    if matches:
        return matches[-1].strip()
    
    # Try to find \\boxed{...} pattern (double escaped)
    boxed_pattern3 = r'\\\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern3, text)
    if matches:
        return matches[-1].strip()
    
    return None


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None
    
    # Remove whitespace
    answer = answer.strip()
    
    # Remove LaTeX commands that don't affect the answer
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\textbf\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathit\{([^}]+)\}', r'\1', answer)
    
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    
    # Normalize sqrt: \sqrt{a} -> sqrt(a) or sqrt a
    answer = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', answer)
    answer = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'sqrt[\1](\2)', answer)
    
    # Remove dollar signs
    answer = answer.replace('$', '')
    
    # Remove backslashes before common symbols
    answer = answer.replace('\\cdot', '*')
    answer = answer.replace('\\times', '*')
    answer = answer.replace('\\pm', '±')
    answer = answer.replace('\\mp', '∓')
    
    # Normalize spaces around operators and punctuation
    answer = re.sub(r'\s*,\s*', ', ', answer)
    answer = re.sub(r'\s*\.\s*', '.', answer)
    
    # Normalize multiple spaces to single space
    answer = ' '.join(answer.split())
    
    # Remove leading/trailing whitespace again
    answer = answer.strip()
    
    return answer


def format_prompt(problem, problem_type="math"):
    """Format problem using Qwen chat template."""
    type_str = problem_type.lower() if problem_type != "math" else "math"
    messages = [
            {"role": "system", "content": f"""You are a math tutor. Give a complete solution put the final answer in the format \\boxed{...}."""}, 
            {"role": "user", "content": f"""{problem}"""}
    ]
    return messages


def generate_response_vllm(llm, tokenizer, problem, problem_type="math"):
    """Generate response using vLLM (faster)."""
    messages = format_prompt(problem, problem_type)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )
    
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


def generate_response_transformers(model, tokenizer, problem, problem_type="math"):
    """Generate response using transformers (slower but more compatible)."""
    messages = format_prompt(problem, problem_type)
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    
    return response


def generate_responses_batch_vllm(llm, tokenizer, problems, problem_type="math"):
    """Generate responses for multiple problems in batch using vLLM."""
    prompts = []
    for problem in problems:
        messages = format_prompt(problem, problem_type)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def evaluate_models():
    """Evaluate both models on the dataset."""
    print("=" * 80)
    print("Model Evaluation: Fine-tuned vs Baseline")
    print("=" * 80)
    
    if USE_VLLM and not VLLM_AVAILABLE:
        print("\nWarning: vLLM requested but not available. Falling back to transformers.")
        use_vllm = False
    else:
        use_vllm = USE_VLLM and VLLM_AVAILABLE
    
    # Load dataset
    if DATASET_PATH:
        print(f"\nLoading dataset from {DATASET_PATH}...")
        dataset = load_from_disk(DATASET_PATH)
        print(f"Dataset loaded: {len(dataset)} examples")
        
        # Limit samples if specified
        if MAX_SAMPLES and MAX_SAMPLES < len(dataset):
            print(f"Limiting evaluation to {MAX_SAMPLES} samples for faster testing...")
            dataset = dataset.select(range(MAX_SAMPLES))
    else:
        print(f"\nLoading dataset from HuggingFace: {DATASET_NAME}...")
        full_dataset = load_dataset(DATASET_NAME, split="train")
        print(f"Full dataset loaded: {len(full_dataset)} examples")
        
        if PROBLEM_TYPE and PROBLEM_LEVEL:
            # Filter for problem type and level
            print(f"Filtering for {PROBLEM_TYPE} {PROBLEM_LEVEL} problems...")
            filtered_dataset = full_dataset.filter(
                lambda x: x.get('type') == PROBLEM_TYPE and x.get('level') == PROBLEM_LEVEL
            )
            print(f"Found {len(filtered_dataset)} {PROBLEM_TYPE} {PROBLEM_LEVEL} problems")
        else:
            filtered_dataset = full_dataset
        
        # Limit samples if specified
        if MAX_SAMPLES and MAX_SAMPLES < len(filtered_dataset):
            print(f"Limiting evaluation to {MAX_SAMPLES} samples...")
            dataset = filtered_dataset.select(range(MAX_SAMPLES))
        else:
            dataset = filtered_dataset
    
    # Prepare problems and ground truths first (needed for both paths)
    problems = []
    ground_truths = []
    valid_indices = []
    
    print("\nPreparing problems...")
    for i, example in enumerate(dataset):
        problem = example.get("problem", "")
        solution = example.get("solution", "")
        ground_truth = example.get("ground_truth", "")  # May not exist for competition_math
        
        # For competition_math, solution field contains the answer
        # For other datasets, ground_truth field may contain the answer
        if not problem:
            continue
        
        # Extract ground truth answer (competition_math uses "solution" field)
        gt_answer = extract_answer(solution) if solution else None
        if gt_answer is None and ground_truth:
            # Fallback to ground_truth field if available
            gt_answer = extract_answer(ground_truth)
        
        if gt_answer is None:
            continue
        
        problems.append(problem)
        ground_truths.append(gt_answer)
        valid_indices.append(i)
    
    print(f"Valid problems: {len(problems)}")
    
    # Load models
    if use_vllm:
        # Load and process baseline model first, then fine-tuned
        print(f"\nLoading baseline model with vLLM: {BASELINE_MODEL}...")
        baseline_llm = LLM(
            model=BASELINE_MODEL,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=0.7,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        baseline_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL, trust_remote_code=True)
        baseline_model = None
        print("Baseline model loaded")
        
        # Generate baseline responses
        problem_type_str = PROBLEM_TYPE if DATASET_PATH is None else "math"
        print("Generating baseline responses...")
        baseline_responses = generate_responses_batch_vllm(baseline_llm, baseline_tokenizer, problems, problem_type_str)
        
        # Clean up baseline model
        del baseline_llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load fine-tuned model
        print(f"\nLoading fine-tuned model with vLLM: {FINETUNED_MODEL}...")
        finetuned_llm = LLM(
            model=FINETUNED_MODEL,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=0.7,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL, trust_remote_code=True)
        finetuned_model = None
        print("Fine-tuned model loaded")
        
        # Generate fine-tuned responses
        print("Generating fine-tuned responses...")
        finetuned_responses = generate_responses_batch_vllm(finetuned_llm, finetuned_tokenizer, problems, problem_type_str)
        
        # Clean up
        del finetuned_llm
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\nLoading baseline model: {BASELINE_MODEL}...")
        baseline_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL, trust_remote_code=True)
        baseline_model = AutoModelForCausalLM.from_pretrained(
            BASELINE_MODEL,
            torch_dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        baseline_llm = None
        print("Baseline model loaded")
        
        print(f"\nLoading fine-tuned model: {FINETUNED_MODEL}...")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL, trust_remote_code=True)
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL,
            torch_dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        finetuned_llm = None
        print("Fine-tuned model loaded")
    
    # Generate responses
    print(f"\nGenerating responses...")
    if not use_vllm:
        # Generate one by one (slower)
        baseline_responses = []
        finetuned_responses = []
        problem_type_str = PROBLEM_TYPE if DATASET_PATH is None else "math"
        for problem in tqdm(problems, desc="Generating responses"):
            try:
                baseline_resp = generate_response_transformers(baseline_model, baseline_tokenizer, problem, problem_type_str)
                finetuned_resp = generate_response_transformers(finetuned_model, finetuned_tokenizer, problem, problem_type_str)
                baseline_responses.append(baseline_resp)
                finetuned_responses.append(finetuned_resp)
            except Exception as e:
                print(f"\nError generating responses: {str(e)}")
                baseline_responses.append("")
                finetuned_responses.append("")
    
    # Evaluate
    baseline_correct = 0
    finetuned_correct = 0
    total = len(problems)
    results = []
    
    print(f"\nEvaluating answers...")
    for i, (gt_answer, baseline_resp, finetuned_resp, orig_idx) in enumerate(zip(
        ground_truths, baseline_responses, finetuned_responses, valid_indices
    )):
        gt_answer_normalized = normalize_answer(gt_answer)
        
        # Extract answers from responses
        baseline_answer = extract_answer(baseline_resp)
        finetuned_answer = extract_answer(finetuned_resp)
        
        baseline_answer_normalized = normalize_answer(baseline_answer)
        finetuned_answer_normalized = normalize_answer(finetuned_answer)
        
        # Check correctness
        baseline_is_correct = (baseline_answer_normalized == gt_answer_normalized) if baseline_answer_normalized else False
        finetuned_is_correct = (finetuned_answer_normalized == gt_answer_normalized) if finetuned_answer_normalized else False
        
        if baseline_is_correct:
            baseline_correct += 1
        if finetuned_is_correct:
            finetuned_correct += 1
        
        # Store results
        results.append({
            "problem_idx": orig_idx,
            "problem": problems[i][:200] + "..." if len(problems[i]) > 200 else problems[i],
            "ground_truth_answer": gt_answer,
            "baseline_answer": baseline_answer,
            "finetuned_answer": finetuned_answer,
            "baseline_correct": baseline_is_correct,
            "finetuned_correct": finetuned_is_correct,
            "baseline_response": baseline_resp,
            "finetuned_response": finetuned_resp
        })
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total problems evaluated: {total}")
    print(f"\nBaseline Model ({BASELINE_MODEL}):")
    print(f"  Correct answers: {baseline_correct}/{total}")
    print(f"  Accuracy: {baseline_correct/total*100:.2f}%")
    print(f"\nFine-tuned Model ({FINETUNED_MODEL}):")
    print(f"  Correct answers: {finetuned_correct}/{total}")
    print(f"  Accuracy: {finetuned_correct/total*100:.2f}%")
    print(f"\nImprovement: {finetuned_correct - baseline_correct} more correct answers")
    print(f"Relative improvement: {(finetuned_correct - baseline_correct)/total*100:.2f}%")
    print("=" * 80)
    
    # Save detailed results
    problem_type_str = PROBLEM_TYPE.lower() if DATASET_PATH is None else "all"
    level_str = PROBLEM_LEVEL.replace(" ", "_").lower() if DATASET_PATH is None else ""
    output_file = f"outputs2/transformed_model_qwen2-math-7b-instruct_evaluation_results_{problem_type_str}_{level_str}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\nDetailed results saved to: {output_file}")
    
    return {
        "total": total,
        "baseline_correct": baseline_correct,
        "finetuned_correct": finetuned_correct,
        "baseline_accuracy": baseline_correct/total*100 if total > 0 else 0,
        "finetuned_accuracy": finetuned_correct/total*100 if total > 0 else 0
    }


if __name__ == "__main__":
    results = evaluate_models()

