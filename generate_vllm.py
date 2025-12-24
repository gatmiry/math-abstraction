#!/usr/bin/env python3
"""
Efficient parallel generation using vLLM on fine-tuned Qwen model.
Generates solutions for geometry level 4 problems from competition_math dataset.

Requirements:
    pip install vllm transformers datasets

Usage:
    python finetune_vllm.py
"""

from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os

# Configuration
MODEL_PATH = "Qwen/Qwen2.5-7B" #"./qwen_finetuned"
DATASET_NAME = "qwedsacf/competition_math"
PROBLEM_TYPE = "Geometry"
PROBLEM_LEVEL = "Level 4"
OUTPUT_FILE = "generated_solutions_baseline.jsonl"#"generated_solutions_level4.jsonl"
MAX_NEW_TOKENS = 1024
GPU_MEMORY_UTILIZATION = 0.9

def format_prompt(problem):
    """Format problem using Qwen chat template."""
    messages = [
        {
            "role": "user",
            "content": f"Solve this geometry problem without using any external tools. Put your solution in \\boxed{{...}} format.\n\n Here is the problem:\n\n{problem}"
        }
    ]
    return messages

def load_model_and_tokenizer(model_path):
    """Load vLLM model and tokenizer."""
    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype="bfloat16",  # Match training dtype
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return llm, tokenizer

def generate_solutions(llm, tokenizer, problems, output_file):
    """Generate solutions for problems using vLLM."""
    print(f"Generating solutions for {len(problems)} problems...")
    
    # Format prompts
    prompts = []
    for problem in problems:
        messages = format_prompt(problem['problem'])
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )
    
    # Generate in parallel
    outputs = llm.generate(prompts, sampling_params)
    
    # Save results
    with open(output_file, "w") as f:
        for i, output in enumerate(outputs):
            result = {
                "problem": problems[i]['problem'],
                "ground_truth": problems[i]['solution'],
                "type": problems[i]['type'],
                "level": problems[i]['level'],
                "generated_solution": output.outputs[0].text,
            }
            f.write(json.dumps(result) + "\n")
    
    print(f"Generated solutions saved to {output_file}")
    return outputs

def main():
    # Load dataset
    print(f"Loading dataset {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME)
    
    # Filter geometry level 4 problems
    #print(f"Filtering {PROBLEM_TYPE} {PROBLEM_LEVEL} problems...")
    #geometry_problems = ds['train'].filter(
    #    lambda x: x.get('type') == PROBLEM_TYPE and x.get('level') == PROBLEM_LEVEL
    #)
    problems = ds['train']
    #solutions = [example['solution'] for example in geometry_problems]
    print(f"Found {len(problems)} problems")
    
    if len(problems) == 0:
        print("No problems found. Exiting.")
        return
    
    # Load model
    llm, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # Generate solutions
    outputs = generate_solutions(llm, tokenizer, problems, OUTPUT_FILE)
    
    print(f"\nGeneration complete!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total problems processed: {len(outputs)}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()

