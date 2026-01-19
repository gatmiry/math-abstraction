"""Generate full solutions from Qwen3 4B using vLLM with data parallelism across 8 GPUs."""

import json
import os
import subprocess
import sys
from datasets import load_from_disk


def main():
    num_gpus = 8
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    worker_script = os.path.join(script_dir, "worker.py")
    
    # Load dataset
    dataset = load_from_disk(os.path.join(script_dir, "outputs/sbys_proofs_dataset"))
    
    # Build prompts
    all_items = []
    for row in dataset:
        problem = row["problem"]
        sbys_steps = row["sbys_solution"]
        incomplete_proof = "\n".join(sbys_steps[:-1]) if len(sbys_steps) > 1 else ""
        user_msg = f"Problem: {problem}\nIncomplete proof: {incomplete_proof}"
        all_items.append({
            "user_msg": user_msg,
            "problem": row["problem"],
            "sbys_solution": row["sbys_solution"],
            "final_answer": row["final_answer"],
        })
    
    # Split into chunks
    chunk_size = (len(all_items) + num_gpus - 1) // num_gpus
    chunks = [all_items[i:i + chunk_size] for i in range(0, len(all_items), chunk_size)]
    
    # Save chunks and start workers
    chunk_files = []
    output_files = []
    processes = []
    
    for gpu_id, chunk in enumerate(chunks):
        chunk_file = os.path.join(output_dir, f"tmp_chunk_{gpu_id}.json")
        output_file = os.path.join(output_dir, f"tmp_output_{gpu_id}.jsonl")
        
        with open(chunk_file, "w") as f:
            json.dump(chunk, f)
        
        chunk_files.append(chunk_file)
        output_files.append(output_file)
        
        print(f"Starting GPU {gpu_id} with {len(chunk)} items...")
        proc = subprocess.Popen(
            [sys.executable, worker_script, str(gpu_id), chunk_file, output_file],
        )
        processes.append(proc)
    
    # Wait for all workers
    print("Waiting for all workers to complete...")
    for gpu_id, proc in enumerate(processes):
        proc.wait()
        print(f"GPU {gpu_id} finished with exit code {proc.returncode}")
    
    # Merge results
    final_output = os.path.join(output_dir, "full_solutions_qwen3_4b.jsonl")
    with open(final_output, "w") as out:
        for output_file in output_files:
            if os.path.exists(output_file):
                with open(output_file) as f:
                    out.write(f.read())
                os.remove(output_file)
    
    # Cleanup chunk files
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
    
    print(f"Merged all results to {final_output}")


if __name__ == "__main__":
    main()
