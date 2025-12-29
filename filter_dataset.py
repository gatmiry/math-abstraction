import json
import re
from datasets import Dataset

def extract_boxed_solution(text):
    """Extract solution from \\boxed{...} at the end of text."""
    # Find the last \\boxed{ and extract content handling nested braces
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None
    
    start_pos = matches[-1].end()
    depth = 1
    i = start_pos
    
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return text[start_pos:i-1]
    return None

def filter_dataset(input_file):
    """Filter dataset keeping only rows where boxed solution matches ground_truth."""
    filtered_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            boxed = extract_boxed_solution(data.get('new_proof', ''))
            ground_truth = data.get('ground_truth', '')
            
            if boxed and boxed.strip() == ground_truth.strip():
                filtered_data.append(data)
    
    return Dataset.from_list(filtered_data)

if __name__ == '__main__':
    dataset = filter_dataset('newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct.jsonl')
    print(f"Filtered dataset: {len(dataset)} examples")
    
    # Save as HuggingFace dataset
    dataset.save_to_disk('newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct_filtered')
    print("Dataset saved to newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct_filtered")
    
    # Also save as JSONL file
    output_jsonl = 'newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct_filtered.jsonl'
    with open(output_jsonl, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    print(f"JSONL file saved to {output_jsonl}")

