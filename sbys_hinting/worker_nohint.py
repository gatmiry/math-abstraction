"""Worker script for processing a chunk on a single GPU - NO HINT version."""

import json
import sys
import os

def main():
    gpu_id = sys.argv[1]
    chunk_file = sys.argv[2]
    output_file = sys.argv[3]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    from vllm import LLM, SamplingParams
    
    # full_solution_simple prompt - no hint, just solve the problem
    SYSTEM_PROMPT = """You are learning mathematics. Given a math problem, generate a clear, step-by-step solution with all necessary reasoning, and place the final answer at the very end, boxed in \\boxed{}.

- For each problem, first show step-by-step logical reasoning, explaining your process and each calculation or inference.
- Only reveal the final answer after the complete solution.
- Format the final answer as \\boxed{[FINAL ANSWER]} and place it as the last element of your response.
- Ensure the solution is thorough, easy to follow, and includes all necessary math justifications.
- Do not skip steps; be explicit in all calculations and logical connections.

Output Format:
- A full-paragraph or multi-step written solution, with clear reasoning steps.
- Always end the solution with the final answer in the format \\boxed{[FINAL ANSWER]} on its own line.

Example:

Input:
Problem: What is the sum of 12 and 30?

Output:
To find the sum of 12 and 30, start by adding the two numbers: 12 + 30 = 42. Therefore, the sum of 12 and 30 is
\\boxed{42}

(Reminder: Reasoning must always appear before the conclusion. The output is a clear step-by-step solution, ending with the boxed answer.)

Important instructions and objectives:
- Always provide detailed reasoning before the final answer.
- Only place the final answer at the end, boxed in \\boxed{}.
- Do not skip steps. Be explicit and clear in the solution."""

    # Load chunk
    with open(chunk_file) as f:
        chunk = json.load(f)
    
    print(f"GPU {gpu_id}: Loaded {len(chunk)} items, initializing model...")
    
    # Initialize vLLM
    llm = LLM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        max_model_len=16384,
        gpu_memory_utilization=0.95,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=8192)
    
    # Build messages
    messages_list = []
    for item in chunk:
        messages_list.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["user_msg"]},
        ])
    
    print(f"GPU {gpu_id}: Starting generation...")
    outputs = llm.chat(messages_list, sampling_params=sampling_params)
    
    # Save results
    results = []
    for item, output in zip(chunk, outputs):
        results.append({
            "problem": item["problem"],
            "sbys_solution": item["sbys_solution"],
            "final_answer": item["final_answer"],
            "generated_solution": output.outputs[0].text,
        })
    
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"GPU {gpu_id}: Done, saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()

