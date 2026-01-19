"""Worker script for processing a chunk on a single GPU."""

import json
import sys
import os

def main():
    gpu_id = sys.argv[1]
    chunk_file = sys.argv[2]
    output_file = sys.argv[3]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    from vllm import LLM, SamplingParams
    
    SYSTEM_PROMPT = """You are learning mathematics. For each task, you are given:
- A math problem.
- An incomplete solution or proof to the problem (this proof is CORRECT but missing the final conclusion).

Your job is to write a full solution following the SAME logical approach as the incomplete proof. The incomplete proof shows you the correct method and key steps - you must follow this same reasoning path and arrive at the same conclusion. Write the solution in your own words, but use the exact same mathematical approach, constructions, and logical steps shown in the incomplete proof.

**Critical:** The incomplete proof is mathematically correct. Trust it completely. Your solution must:
1. Follow the same proof strategy and logical structure
2. Use the same key constructions, lemmas, and techniques
3. Arrive at the same conclusion the proof is leading toward

**Output format:**
Write a complete, well-explained solution that follows the incomplete proof's approach. End with the final answer in \\boxed{}.

---

### Example

**Input:**
Problem: Let \\( f(x) = x^2 - 4x + 3 \\). Find all real solutions to \\( f(x) = 0 \\).
Incomplete proof: "We can factor f(x) as (x - 1)(x - 3). Setting each factor to zero..."

**Output:**
To solve \\( f(x) = 0 \\), we factor the quadratic \\( f(x) = x^2 - 4x + 3 = (x - 1)(x - 3) \\). Setting each factor equal to zero gives \\( x - 1 = 0 \\) or \\( x - 3 = 0 \\), so \\( x = 1 \\) or \\( x = 3 \\).
\\boxed{1 \\text{ and } 3}

---

**Important:**
- The incomplete proof shows the CORRECT approach - follow it exactly
- Do not deviate from the proof's logical structure or try a different method
- The proof leads to a specific answer - your solution must reach that same answer
- Final answer must be in \\boxed{}"""

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

