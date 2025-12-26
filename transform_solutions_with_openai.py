"""
Transform Qwen-generated proofs using OpenAI API to new format.
Takes proofs from Qwen 7B model and transforms them using OpenAI.
"""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
import datasets
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def get_api_key():
    """Get API key from settings.json or environment variable."""
    settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            settings = json.load(f)
            api_key = settings.get("openai_api_key")
            if api_key:
                return api_key
    return os.getenv("OPENAI_API_KEY")


def transform_proof_with_openai(client: OpenAI, old_proof: str, problem: str) -> str:
    """Transform a proof using OpenAI API."""
    transform_prompt = """This is the problem:
{problem}

This is the proof:
{old_proof}
""".format(problem=problem, old_proof=old_proof)
    
    try:
        response = client.responses.create(
            prompt={
                "id": "pmpt_69475c3f5c9c8194b0a136dc663d15b30cc8e7bedee1d50b",
                "version": "2"
            },
            input=[{"role": "user", "content": transform_prompt}],
            reasoning={
                "summary": "auto"
            },
            include=[
                "reasoning.encrypted_content",
                "web_search_call.action.sources"
            ]
        )
        return response.output_text
    except Exception as e:
        return f"Error: {str(e)}"


def _transform_proof_worker(item_dict: Dict, api_key: str) -> Dict:
    """Worker function for parallel processing. Creates a new OpenAI client for each process.
    
    Args:
        item_dict: Dictionary with 'problem', 'generated_solution', and 'ground_truth' keys.
        api_key: OpenAI API key.
    
    Returns:
        Dictionary with 'problem', 'ground_truth', 'old_proof', and 'new_proof' keys.
    """
    # Create a new client for this process (can't share clients across processes)
    client = OpenAI(api_key=api_key)
    
    problem = item_dict.get("problem", "")
    old_proof = item_dict.get("generated_solution", "")
    ground_truth = item_dict.get("ground_truth", "")
    
    # Transform proof using OpenAI
    transform_prompt = """This is the problem:
{problem}

This is the proof:
{old_proof}
""".format(problem=problem, old_proof=old_proof)
    
    try:
        response = client.responses.create(
            prompt={
                "id": "pmpt_694bb78979a481969e18ba6a5ac61b81042befdd28a17945",
                "version": "7"
            },
            input=[{"role": "user", "content": transform_prompt}],
            reasoning={
                "summary": "auto"
            },
            include=[
                "reasoning.encrypted_content",
                "web_search_call.action.sources"
            ]
        )
        new_proof = response.output_text
    except Exception as e:
        new_proof = f"Error: {str(e)}"
    
    return {
        "problem": problem,
        "ground_truth": ground_truth,
        "old_proof": old_proof,  # Original Qwen proof
        "new_proof": new_proof,  # OpenAI-transformed proof
    }


def load_qwen_solutions(input_path: str) -> List[Dict]:
    """Load Qwen-generated solutions from JSONL file."""
    results = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def transform_solutions(
    input_path: str,
    output_jsonl_path: str,
    output_dataset_path: str = None,
    max_data_n: Optional[int] = None,
    num_processes: Optional[int] = None
):
    """Transform Qwen solutions using OpenAI API (sequential version)."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or add 'openai_api_key' to settings.json file."
        )
    
    client = OpenAI(api_key=api_key)
    
    # Load Qwen-generated solutions
    print(f"Loading solutions from {input_path}...")
    qwen_results = load_qwen_solutions(input_path)
    print(f"Loaded {len(qwen_results)} solutions")
    
    # Limit to first max_data_n rows if specified
    if max_data_n is not None:
        qwen_results = qwen_results[:max_data_n]
        print(f"Limited to first {len(qwen_results)} solutions")
    
    # Transform each solution
    transformed_results = []
    for item in tqdm(qwen_results, desc="Transforming"):
        problem = item.get("problem", "")
        old_proof = item.get("generated_solution", "")
        ground_truth = item.get("ground_truth", "")
        
        # Transform proof using OpenAI
        new_proof = transform_proof_with_openai(client, old_proof, problem)
        
        transformed_results.append({
            "problem": problem,
            "ground_truth": ground_truth,
            "old_proof": old_proof,  # Original Qwen proof
            "new_proof": new_proof,  # OpenAI-transformed proof
        })
    
    # Save as JSONL
    os.makedirs(os.path.dirname(output_jsonl_path) if os.path.dirname(output_jsonl_path) else ".", exist_ok=True)
    with open(output_jsonl_path, "w") as f:
        for result in transformed_results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved JSONL to {output_jsonl_path}")
    
    # Save as HuggingFace Dataset
    if output_dataset_path:
        dataset = datasets.Dataset.from_list(transformed_results)
        dataset.save_to_disk(output_dataset_path)
        print(f"Saved dataset to {output_dataset_path}")


def transform_solutions_parallel(
    input_path: str,
    output_jsonl_path: str,
    output_dataset_path: str = None,
    max_data_n: Optional[int] = None,
    num_processes: Optional[int] = None
):
    """Transform Qwen solutions using OpenAI API with parallel processing.
    
    Args:
        input_path: Path to input JSONL file with Qwen-generated solutions.
        output_jsonl_path: Path to save transformed results as JSONL.
        output_dataset_path: Optional path to save results as HuggingFace Dataset.
        max_data_n: Maximum number of rows to process (None for all).
        num_processes: Number of parallel processes to use. Defaults to CPU count.
    """
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or add 'openai_api_key' to settings.json file."
        )
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Load Qwen-generated solutions
    print(f"Loading solutions from {input_path}...")
    qwen_results = load_qwen_solutions(input_path)
    print(f"Loaded {len(qwen_results)} solutions")
    
    # Limit to first max_data_n rows if specified
    if max_data_n is not None:
        qwen_results = qwen_results[:max_data_n]
        print(f"Limited to first {len(qwen_results)} solutions")
    
    # Create worker function with API key
    worker_func = partial(
        _transform_proof_worker,
        api_key=api_key
    )
    
    # Process solutions in parallel
    with mp.Pool(processes=num_processes) as pool:
        transformed_results = list(tqdm(
            pool.imap(worker_func, qwen_results),
            total=len(qwen_results),
            desc="Transforming (parallel)"
        ))
    
    # Save as JSONL
    os.makedirs(os.path.dirname(output_jsonl_path) if os.path.dirname(output_jsonl_path) else ".", exist_ok=True)
    with open(output_jsonl_path, "w") as f:
        for result in transformed_results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved JSONL to {output_jsonl_path}")
    
    # Save as HuggingFace Dataset
    if output_dataset_path:
        dataset = datasets.Dataset.from_list(transformed_results)
        dataset.save_to_disk(output_dataset_path)
        print(f"Saved dataset to {output_dataset_path}")


def main():
    # Configuration
    input_path = "generated_solutions_baseline.jsonl"  # Qwen-generated solutions
    output_jsonl_path = "newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct.jsonl"
    output_dataset_path = "newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct_dataset"
    max_data_n = 5000  # Set to a number to limit processing (e.g., 100)
    num_processes = 100  # Set to a number or None for CPU count
    
    transform_solutions_parallel(
        input_path=input_path,
        output_jsonl_path=output_jsonl_path,
        output_dataset_path=output_dataset_path,
        max_data_n=max_data_n,
        num_processes=num_processes
    )


if __name__ == "__main__":
    main()

