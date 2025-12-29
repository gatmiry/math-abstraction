import json
import re

def extract_boxed_solution(text):
    """Extract solution from \\boxed{...} at the end of text."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    return match.group(1) if match else None

def filter_dataset(input_file, output_file):
    """Filter dataset keeping only rows where boxed solution matches ground_truth."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            boxed = extract_boxed_solution(data.get('new_proof', ''))
            ground_truth = data.get('ground_truth', '')
            
            if boxed and boxed.strip() == ground_truth.strip():
                f_out.write(line)

if __name__ == '__main__':
    filter_dataset(
        'newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct.jsonl',
        'newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct_filtered.jsonl'
    )

