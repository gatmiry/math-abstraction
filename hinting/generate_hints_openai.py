"""
Generate hints using OpenAI API with parallel processing.
Reads cut proofs from cut_proofs_dataset and generates hints for each.
"""

import os
import json
import re
import shutil
from typing import Any, List, Dict, Optional
from openai import OpenAI
import datasets
from tqdm import tqdm
import multiprocessing as mp
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


def _generate_hint_worker(item_dict: Dict, api_key: str) -> Dict:
    """Worker function for parallel processing. Creates a new OpenAI client for each process.
    
    Args:
        item_dict: Dictionary with fields from cut_proofs_dataset.
        api_key: OpenAI API key.
    
    Returns:
        Dictionary with original fields plus 'hint' field.
    """
    # Create a new client for this process (can't share clients across processes)
    client = OpenAI(api_key=api_key)
    
    # Map fields from dataset to OpenAI input
    problem = item_dict.get("problem", "")
    proof = item_dict.get("cut_version_1", "")  # Use cut_version_1 as proof
    ground_truth_solution = item_dict.get("solution", "")  # Use solution as ground_truth_solution
    final_answer = item_dict.get("answer", "")
    domain = item_dict.get("domain", [])
    difficulty = item_dict.get("difficulty", None)
    
    # Convert domain list to string if needed (for prompt variables)
    if isinstance(domain, list):
        domain_str = ", ".join(domain) if domain else ""
    else:
        domain_str = str(domain) if domain else ""
    
    # Convert difficulty to string if needed (for prompt variables)
    difficulty_str = str(difficulty) if difficulty is not None else ""
    
    try:
        response = client.responses.create(
            prompt={
                "id": "pmpt_6959ae5976bc8197a74a63927312f4750d94641387cb22a1",
                "version": "14",
                "variables": {
                    "problem": problem,
                    "proof": proof,
                    "ground_truth_solution": ground_truth_solution,
                    "final_answer": final_answer,
                    "domain": domain_str,
                    "difficulty": difficulty_str,
                }
            },
        )
        
        # Get the output text and parse JSON
        output_text = response.output_text
        
        # Try to extract JSON from the output (might be wrapped in markdown code blocks)
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = output_text
        
        # Parse the JSON response
        parsed_response = json.loads(json_str)
        
        # Merge the parsed fields into the result
        # Keep lists as lists, don't convert to strings
        #result = item_dict.copy()
        result = {}
        # Extract all fields from the parsed response and add to result
        # Use exact field names from the JSON response
        if "problem" in parsed_response:
            result["problem"] = parsed_response["problem"]
        if "domain" in parsed_response:
            result["domain"] = parsed_response["domain"]
        if "difficulty" in parsed_response:
            result["difficulty"] = parsed_response["difficulty"]
        if "partial_proof" in parsed_response:
            result["partial_proof"] = parsed_response["partial_proof"]
        if "full_proof" in parsed_response:
            result["full_proof"] = parsed_response["full_proof"]
        if "ground_truth_solution" in parsed_response:
            result["ground_truth_solution"] = parsed_response["ground_truth_solution"]
        if "final_answer" in parsed_response:
            result["final_answer"] = parsed_response["final_answer"]
        if "hints" in parsed_response:
            result["hints"] = parsed_response["hints"]
        
        # Also keep the raw output text for reference
        #result["raw_hint_output"] = output_text
        
        return result
        
    except json.JSONDecodeError as e:
        # Return error result with only expected fields
        result = {
            "problem": problem,
            "domain": domain_str,
            "difficulty": difficulty_str,
            "partial_proof": "",
            "full_proof": "",
            "ground_truth_solution": ground_truth_solution,
            "final_answer": final_answer,
            "hints": [],
            "raw_hint_output": response.output_text if 'response' in locals() else "",
            "parse_error": f"JSON decode error: {str(e)}"
        }
        return result
    except Exception as e:
        # Return error result with only expected fields
        result = {
            "problem": problem,
            "domain": domain_str,
            "difficulty": difficulty_str,
            "partial_proof": "",
            "full_proof": "",
            "ground_truth_solution": ground_truth_solution,
            "final_answer": final_answer,
            "hints": [],
            "raw_hint_output": response.output_text if 'response' in locals() else "",
            "parse_error": f"Error: {str(e)}"
        }
        return result


def load_cut_proofs_dataset(dataset_path: str) -> List[Dict]:
    """Load cut proofs dataset from disk.
    
    Args:
        dataset_path: Path to the HuggingFace dataset directory.
    
    Returns:
        List of dictionaries with dataset items.
    """
    dataset = datasets.load_from_disk(dataset_path)
    # Convert to list of dicts
    return [dict(item) for item in dataset]


def generate_hints(
    dataset_path: str,
    output_jsonl_path: str,
    output_dataset_path: str = None,
    max_data_n: Optional[int] = None,
    num_processes: Optional[int] = None
):
    """Generate hints using OpenAI API with parallel processing.
    
    Args:
        dataset_path: Path to the cut_proofs_dataset directory.
        output_jsonl_path: Path to save results as JSONL.
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
    
    # Load cut proofs dataset
    print(f"Loading cut proofs dataset from {dataset_path}...")
    input_data = load_cut_proofs_dataset(dataset_path)
    print(f"Loaded {len(input_data)} items")
    
    # Limit to first max_data_n rows if specified
    if max_data_n is not None:
        input_data = input_data[:max_data_n]
        print(f"Limited to first {len(input_data)} items")
    
    # Create worker function with API key
    worker_func = partial(
        _generate_hint_worker,
        api_key=api_key
    )
    
    # Process items in parallel
    print(f"Processing with {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(worker_func, input_data),
            total=len(input_data),
            desc="Generating hints (parallel)"
        ))
    
    # Save as JSONL
    os.makedirs(os.path.dirname(output_jsonl_path) if os.path.dirname(output_jsonl_path) else ".", exist_ok=True)
    with open(output_jsonl_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved JSONL to {output_jsonl_path}")
    
    # Save as HuggingFace Dataset
    if output_dataset_path:
        # Remove existing dataset if it exists to avoid feature mismatch errors
        if os.path.exists(output_dataset_path):
            print(f"Removing existing dataset at {output_dataset_path}...")
            shutil.rmtree(output_dataset_path)
            # Verify removal
            if os.path.exists(output_dataset_path):
                raise RuntimeError(f"Failed to remove existing dataset at {output_dataset_path}")
        
        # Normalize results: extract only the fields we want to keep
        # This ensures all results have the same schema
        normalized_results = []
        expected_fields = [
            "problem", "domain", "difficulty", "partial_proof", "full_proof",
            "ground_truth_solution", "final_answer", "hints", "raw_hint_output", "parse_error"
        ]
        
        for result in results:
            normalized = {}
            for field in expected_fields:
                if field in result:
                    normalized[field] = result[field]
                else:
                    # Set default values for missing fields
                    if field == "hints":
                        normalized[field] = []
                    elif field in ["raw_hint_output", "parse_error"]:
                        normalized[field] = ""
                    else:
                        normalized[field] = ""
            normalized_results.append(normalized)
        
        # Build Features schema that supports list fields
        # Collect all unique keys from normalized results
        all_keys = set[Any]()
        for result in normalized_results:
            all_keys.update(result.keys())
        
        # Build features dict - check all results to determine types
        features_dict = {}
        for key in all_keys:
            # Check all results to find the type
            for result in normalized_results:
                if key in result:
                    value = result[key]
                    if isinstance(value, list):
                        # If list is not empty, check element type
                        if value and isinstance(value[0], str):
                            features_dict[key] = datasets.Sequence(datasets.Value("string"))
                        else:
                            # Default to list of strings
                            features_dict[key] = datasets.Sequence(datasets.Value("string"))
                        break
                    elif isinstance(value, dict):
                        features_dict[key] = datasets.Value("string")
                        break
                    elif isinstance(value, (int, float)):
                        if isinstance(value, float):
                            features_dict[key] = datasets.Value("float64")
                        else:
                            features_dict[key] = datasets.Value("int64")
                        break
                    else:
                        features_dict[key] = datasets.Value("string")
                        break
            
            # If key not found in any result, default to string
            if key not in features_dict:
                features_dict[key] = datasets.Value("string")
        
        # Create Features object
        features = datasets.Features(features_dict)
        
        # Create dataset with explicit features
        dataset = datasets.Dataset.from_list(normalized_results, features=features)
        dataset.save_to_disk(output_dataset_path)
        print(f"Saved dataset to {output_dataset_path}")


def main():
    # Configuration
    dataset_path = "../newopenaioutputs/cut_proofs_dataset"  # Path to cut_proofs_dataset
    output_jsonl_path = "../newopenaioutputs/hints.jsonl"
    output_dataset_path = "../newopenaioutputs/hints_dataset"
    max_data_n = None  # Set to a number to limit processing (e.g., 100)
    num_processes = 50 # Set to a number or None for CPU count
    
    generate_hints(
        dataset_path=dataset_path,
        output_jsonl_path=output_jsonl_path,
        output_dataset_path=output_dataset_path,
        max_data_n=max_data_n,
        num_processes=num_processes
    )


if __name__ == "__main__":
    main()

