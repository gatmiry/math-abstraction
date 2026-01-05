"""
Generate 5 cut versions of proofs using OpenAI API with parallel processing.
Reads proofs from OMNI-MATH HuggingFace dataset and generates 5 different cut versions for each.
"""

import os
import json
import re
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


def get_schema():
    """Get the JSON schema for the 5 cut versions."""
    return {
        "name": "five_cut_proofs",
        "schema": {
            "type": "object",
            "properties": {
                "cut_version_1": {
                    "type": "object",
                    "properties": {"proof": {"type": "string"}},
                    "required": ["proof"],
                    "additionalProperties": False,
                },
                "cut_version_2": {
                    "type": "object",
                    "properties": {"proof": {"type": "string"}},
                    "required": ["proof"],
                    "additionalProperties": False,
                },
                "cut_version_3": {
                    "type": "object",
                    "properties": {"proof": {"type": "string"}},
                    "required": ["proof"],
                    "additionalProperties": False,
                },
                "cut_version_4": {
                    "type": "object",
                    "properties": {"proof": {"type": "string"}},
                    "required": ["proof"],
                    "additionalProperties": False,
                },
                "cut_version_5": {
                    "type": "object",
                    "properties": {"proof": {"type": "string"}},
                    "required": ["proof"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "cut_version_1",
                "cut_version_2",
                "cut_version_3",
                "cut_version_4",
                "cut_version_5",
            ],
            "additionalProperties": False,
        },
    }


def _generate_cut_proofs_worker(item_dict: Dict, api_key: str) -> Dict:
    """Worker function for parallel processing. Creates a new OpenAI client for each process.
    
    Args:
        item_dict: Dictionary with 'problem' and 'solution' keys from OMNI-MATH dataset.
        api_key: OpenAI API key.
    
    Returns:
        Dictionary with original fields plus 'cut_version_1' through 'cut_version_5' fields.
    """
    # Create a new client for this process (can't share clients across processes)
    client = OpenAI(api_key=api_key)
    
    problem = item_dict.get("problem", "")
    solution = item_dict.get("solution", "")
    
    schema = get_schema()
    
    # Prepare input for the prompt with 'problem' and 'solution' keywords
    prompt_input = {
        "problem": problem,
        "solution": solution,
    }
    
    try:
        response = client.responses.create(
            prompt={
                "id": "pmpt_695983d6d67881968f7ec6e53baca11c0cdb61de54e6239e",
                "version": "10",
                "variables": {
                    "problem": prompt_input["problem"],
                    "proof": prompt_input["solution"]
                }
            },
        )
        
        # Parse JSON from output text
        output_text = response.output_text
        # Try to extract JSON from the response (might be wrapped in markdown code blocks)
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            cut_proofs = json.loads(json_match.group(0))
        else:
            # If no JSON found, try parsing the whole output
            cut_proofs = json.loads(output_text)
        
        # Merge the cut proofs into the result
        result = item_dict.copy()
        result.update({
            "cut_version_1": cut_proofs.get("cut_version_1", {}).get("proof", ""),
            "cut_version_2": cut_proofs.get("cut_version_2", {}).get("proof", ""),
            "cut_version_3": cut_proofs.get("cut_version_3", {}).get("proof", ""),
            "cut_version_4": cut_proofs.get("cut_version_4", {}).get("proof", ""),
            "cut_version_5": cut_proofs.get("cut_version_5", {}).get("proof", ""),
        })
        
        return result
        
    except Exception as e:
        # Return original item with error message
        result = item_dict.copy()
        result.update({
            "cut_version_1": f"Error: {str(e)}",
            "cut_version_2": f"Error: {str(e)}",
            "cut_version_3": f"Error: {str(e)}",
            "cut_version_4": f"Error: {str(e)}",
            "cut_version_5": f"Error: {str(e)}",
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


def generate_cut_proofs(
    output_jsonl_path: str,
    output_dataset_path: str = None,
    dataset_split: str = "test",
    max_data_n: Optional[int] = None,
    num_processes: Optional[int] = None
):
    """Generate 5 cut versions of proofs using OpenAI API with parallel processing.
    
    Args:
        output_jsonl_path: Path to save results as JSONL.
        output_dataset_path: Optional path to save results as HuggingFace Dataset.
        dataset_split: Dataset split to load from OMNI-MATH ('train', 'test', etc.). Defaults to 'test'.
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
    
    # Load OMNI-MATH dataset from HuggingFace
    print(f"Loading OMNI-MATH dataset (split: {dataset_split})...")
    input_data = load_omni_math_dataset(split=dataset_split)
    print(f"Loaded {len(input_data)} items")
    
    # For test split, use only 0.8 fraction (80%)
    if dataset_split == "test":
        test_size = int(len(input_data) * 0.8)
        input_data = input_data[:test_size]
        print(f"Using 80% of test split: {len(input_data)} items")
    
    # Limit to first max_data_n rows if specified
    if max_data_n is not None:
        input_data = input_data[:max_data_n]
        print(f"Limited to first {len(input_data)} items")
    
    # Create worker function with API key
    worker_func = partial(
        _generate_cut_proofs_worker,
        api_key=api_key
    )
    
    # Process items in parallel
    print(f"Processing with {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(worker_func, input_data),
            total=len(input_data),
            desc="Generating cut proofs (parallel)"
        ))
    
    # Save as JSONL
    os.makedirs(os.path.dirname(output_jsonl_path) if os.path.dirname(output_jsonl_path) else ".", exist_ok=True)
    with open(output_jsonl_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved JSONL to {output_jsonl_path}")
    
    # Save as HuggingFace Dataset
    if output_dataset_path:
        dataset = datasets.Dataset.from_list(results)
        dataset.save_to_disk(output_dataset_path)
        print(f"Saved dataset to {output_dataset_path}")


def main():
    # Configuration
    output_jsonl_path = "newopenaioutputs/cut_proofs.jsonl"
    output_dataset_path = "newopenaioutputs/cut_proofs_dataset"
    dataset_split = "test"  # 'train', 'test', etc.
    max_data_n = 2  # Set to a number to limit processing (e.g., 100)
    num_processes = 2  # Set to a number or None for CPU count
    
    generate_cut_proofs(
        output_jsonl_path=output_jsonl_path,
        output_dataset_path=output_dataset_path,
        dataset_split=dataset_split,
        max_data_n=max_data_n,
        num_processes=num_processes
    )


if __name__ == "__main__":
    main()

