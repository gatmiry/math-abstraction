"""
Generate step-by-step (sbys) proofs using OpenAI API with parallel processing.
Reads problems and solutions from OMNI-MATH HuggingFace dataset and generates
detailed step-by-step solutions with final answers.

Example output format:
{
  "sbys_solution": [
    "Let S be the sum of the first 10 positive integers.",
    "S = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10",
    "The sum S can be computed using the formula for the sum of an arithmetic series: S = n(n+1)/2, where n = 10.",
    ...
  ],
  "final_answer": "\\boxed{55}"
}
"""

import os
import json
import re
from typing import List, Dict, Optional
from openai import OpenAI
import datasets
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


def get_api_key():
    """Get API key from settings.json or environment variable."""
    settings_path = os.path.join(os.path.dirname(__file__), "..", "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            settings = json.load(f)
            api_key = settings.get("openai_api_key")
            if api_key:
                return api_key
    return os.getenv("OPENAI_API_KEY")


def _generate_sbys_proof_worker(item_dict: Dict, api_key: str, timeout: float = 600.0) -> Dict:
    """Worker function for parallel processing. Creates a new OpenAI client for each process.
    
    Args:
        item_dict: Dictionary with 'problem' and 'solution' keys from OMNI-MATH dataset.
        api_key: OpenAI API key.
        timeout: Timeout in seconds for API calls (default 600 = 10 minutes).
    
    Returns:
        Dictionary with original fields plus 'sbys_solution' and 'final_answer' fields.
    """
    # Create a new client with extended timeout for complex math proofs
    client = OpenAI(api_key=api_key) #, timeout=timeout)
    
    problem = item_dict.get("problem", "")
    solution = item_dict.get("solution", "")
    
    try:
        response = client.responses.create(
            prompt={
                "id": "pmpt_696c0ad540888190b19a6a6ecf30a9e0001c5361bee46a1c",
                "version": "3",
                "variables": {
                    "problem": problem,
                    "solution": solution
                }
            }
        )
        
        # Parse JSON from output text
        output_text = response.output_text
        print('output_text', output_text)
        # Try to extract JSON from the response (might be wrapped in markdown code blocks)
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            sbys_result = json.loads(json_match.group(0))
        else:
            # If no JSON found, try parsing the whole output
            sbys_result = json.loads(output_text)
        
        # Merge the sbys fields into the result
        result = item_dict.copy()
        result.update({
            "sbys_solution": sbys_result.get("sbys_solution", []),
            "final_answer": sbys_result.get("final_answer", ""),
        })
        
        return result
        
    except Exception as e:
        # Return original item with error message
        result = item_dict.copy()
        result.update({
            "sbys_solution": [f"Error: {str(e)}"],
            "final_answer": f"Error: {str(e)}",
        })
        return result


def load_omni_math_dataset(split: str = "test") -> List[Dict]:
    """Load OMNI-MATH dataset from HuggingFace.
    
    Args:
        split: Dataset split to load ('train', 'test', etc.). Defaults to 'test'.
    
    Returns:
        List of dictionaries with dataset items.
    """
    dataset = datasets.load_dataset("KbsdJames/Omni-MATH", split=split)
    # Convert to list of dicts
    return [dict(item) for item in dataset]


def generate_sbys_proofs(
    output_jsonl_path: str,
    output_dataset_path: str = None,
    dataset_split: str = "test",
    max_n: Optional[int] = None,
    num_threads: Optional[int] = None,
    test_fraction: float = 0.8,
    timeout: float = 600.0
):
    """Generate step-by-step proofs using OpenAI API with parallel processing.
    
    Args:
        output_jsonl_path: Path to save results as JSONL.
        output_dataset_path: Optional path to save results as HuggingFace Dataset.
        dataset_split: Dataset split to load from OMNI-MATH ('train', 'test', etc.). Defaults to 'test'.
        max_n: Maximum number of rows to process (None for all).
        num_threads: Number of parallel threads to use. Defaults to 10.
        test_fraction: Fraction of test split to use (default 0.8 = 80%).
        timeout: Timeout in seconds for each API call (default 600 = 10 minutes).
    """
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or add 'openai_api_key' to settings.json file."
        )
    
    if num_threads is None:
        num_threads = 10
    
    # Load OMNI-MATH dataset from HuggingFace
    print(f"Loading OMNI-MATH dataset (split: {dataset_split})...")
    input_data = load_omni_math_dataset(split=dataset_split)
    print(f"Loaded {len(input_data)} items")
    
    # For test split, use only specified fraction
    if dataset_split == "test" and test_fraction < 1.0:
        test_size = int(len(input_data) * test_fraction)
        input_data = input_data[:test_size]
        print(f"Using {test_fraction*100:.0f}% of test split: {len(input_data)} items")
    
    # Limit to first max_n rows if specified
    if max_n is not None:
        input_data = input_data[:max_n]
        print(f"Limited to first {len(input_data)} items")
    
    # Create worker function with API key and timeout
    worker_func = partial(
        _generate_sbys_proof_worker,
        api_key=api_key,
        timeout=timeout
    )
    
    # Process items in parallel using ThreadPoolExecutor (better for I/O-bound API calls)
    print(f"Processing with {num_threads} threads (timeout={timeout}s per request)...")
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker_func, item): i for i, item in enumerate(input_data)}
        for future in tqdm(as_completed(futures), total=len(input_data), desc="Generating step-by-step proofs"):
            results.append((futures[future], future.result()))
    
    # Sort by original order
    results = [r[1] for r in sorted(results, key=lambda x: x[0])]
    
    # Save as JSONL
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_jsonl_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved JSONL to {output_jsonl_path}")
    
    # Save as HuggingFace Dataset
    if output_dataset_path:
        dataset = datasets.Dataset.from_list(results)
        dataset.save_to_disk(output_dataset_path)
        print(f"Saved dataset to {output_dataset_path}")
    
    # Print summary statistics
    successful = sum(1 for r in results if not str(r.get("final_answer", "")).startswith("Error:"))
    print(f"\nSummary: {successful}/{len(results)} items processed successfully")


def main():
    # Configuration
    output_jsonl_path = "sbys_hinting/outputs/sbys_proofs.jsonl"
    output_dataset_path = "sbys_hinting/outputs/sbys_proofs_dataset"
    dataset_split = "test"  # 'train', 'test', etc.
    max_n = None  # Set to a number to limit processing (e.g., 100 for testing), None for all
    num_threads = 200  # Number of parallel API calls
    test_fraction = 1.0  # Use 80% of test split
    timeout = 600.0  # 10 minutes per API call (complex proofs take time)
    
    generate_sbys_proofs(
        output_jsonl_path=output_jsonl_path,
        output_dataset_path=output_dataset_path,
        dataset_split=dataset_split,
        max_n=max_n,
        num_threads=num_threads,
        test_fraction=test_fraction,
        timeout=timeout
    )


if __name__ == "__main__":
    main()

