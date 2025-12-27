#!/usr/bin/env python3
"""
Test script for _find_lemma_theorem_positions function.
Tests the function on real data and prints tokens with their mask values.
"""

import json
import sys
import os
from transformers import AutoTokenizer

# Add the current directory to path to import from RL_on_lemmas
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RL_on_lemmas import LemmaTheoremDataset


def print_tokens_with_mask(tokenizer, solution: str, mask: list, max_tokens_to_print: int = 200):
    """
    Print tokens and their corresponding mask values in a readable format.
    
    Args:
        tokenizer: The tokenizer to use
        solution: The solution text
        mask: Boolean mask indicating which tokens should have gradients
        max_tokens_to_print: Maximum number of tokens to print (to avoid overwhelming output)
    """
    # Tokenize solution
    encoding = tokenizer(
        solution,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    tokens = encoding['input_ids']
    offsets = encoding['offset_mapping']
    
    print("\n" + "="*100)
    print("TOKENS AND MASK VALUES")
    print("="*100)
    print(f"Total tokens: {len(tokens)}")
    print(f"Tokens with mask=True: {sum(mask)}")
    print("\nFormat: [Token Index] Token_String -> Mask (True/False)")
    print("-"*100)
    
    # Print tokens with their masks
    tokens_to_print = min(len(tokens), max_tokens_to_print)
    for i in range(tokens_to_print):
        token_id = tokens[i]
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        # Replace newlines and tabs for readability
        token_str_display = token_str.replace('\n', '\\n').replace('\t', '\\t').replace(' ', '·')
        mask_value = mask[i] if i < len(mask) else False
        
        # Highlight tokens with mask=True
        marker = "★" if mask_value else " "
        print(f"[{i:4d}] {marker} '{token_str_display:30s}' -> {mask_value}")
    
    if len(tokens) > max_tokens_to_print:
        print(f"\n... (showing first {max_tokens_to_print} of {len(tokens)} tokens)")
    
    # Print summary of masked tokens
    print("\n" + "-"*100)
    print("SUMMARY OF MASKED TOKENS:")
    print("-"*100)
    masked_indices = [i for i, m in enumerate(mask) if m]
    if masked_indices:
        print(f"Masked token indices: {masked_indices[:50]}")  # Show first 50
        if len(masked_indices) > 50:
            print(f"... and {len(masked_indices) - 50} more")
        
        print("\nMasked token strings:")
        for idx in masked_indices[:80]:  # Show first 20
            if idx < len(tokens):
                token_str = tokenizer.decode([tokens[idx]], skip_special_tokens=True)
                token_str_display = token_str.replace('\n', '\\n').replace('\t', '\\t')
                print(f"  [{idx}] '{token_str_display}'")
        if len(masked_indices) > 20:
            print(f"  ... and {len(masked_indices) - 20} more masked tokens")
    else:
        print("No tokens are masked!")
    
    # Show context around masked tokens
    print("\n" + "-"*100)
    print("CONTEXT AROUND MASKED TOKENS:")
    print("-"*100)
    for idx in masked_indices[:10]:  # Show context for first 10 masked tokens
        if idx < len(tokens):
            start = max(0, idx - 3)
            end = min(len(tokens), idx + 4)
            print(f"\nAround token {idx}:")
            for j in range(start, end):
                token_str = tokenizer.decode([tokens[j]], skip_special_tokens=True)
                token_str_display = token_str.replace('\n', '\\n').replace('\t', '\\t')
                marker = "★" if mask[j] else " "
                print(f"  [{j}] {marker} '{token_str_display}'")


def test_on_examples(jsonl_path: str, model_path: str, num_examples: int = 3):
    """
    Test _find_lemma_theorem_positions on examples from the JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        model_path: Path to the model (for tokenizer)
        num_examples: Number of examples to test
    """
    print("="*100)
    print("TESTING _find_lemma_theorem_positions")
    print("="*100)
    print(f"Loading tokenizer from: {model_path}")
    print(f"Loading examples from: {jsonl_path}")
    print()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying to load from './qwen_finetuned'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("./qwen_finetuned")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e2:
            print(f"Error: {e2}")
            return
    
    # Load examples from JSONL
    examples = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples\n")
    
    # Create dataset instance to access the method
    # We'll create a dummy dataset just to access the method
    class TestDataset(LemmaTheoremDataset):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.max_length = 2048
    
    test_dataset = TestDataset(tokenizer)
    
    # Test on each example
    for example_idx, example in enumerate(examples):
        print("\n" + "="*100)
        print(f"EXAMPLE {example_idx + 1}/{len(examples)}")
        print("="*100)
        
        problem = example.get('problem', '')
        solution = example.get('solution', example.get('new_proof', example.get('generated_solution', '')))
        
        if not solution:
            print("No solution found in this example, skipping...")
            continue
        
        print("\nPROBLEM:")
        print("-"*100)
        print(problem[:500] + ("..." if len(problem) > 500 else ""))
        
        print("\n\nSOLUTION (first 2000 chars):")
        print("-"*100)
        print(solution[:2000] + ("..." if len(solution) > 2000 else ""))
        
        # Find lemma/theorem positions
        prompt = f"Solve this problem without using any external tools. Put your solution in \\boxed{{...}} format.\n\nHere is the problem:\n\n{problem}\n\nSolution:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        solution_tokens = tokenizer.encode(solution, add_special_tokens=False)
        
        print(f"\n\nPrompt length: {len(prompt_tokens)} tokens")
        print(f"Solution length: {len(solution_tokens)} tokens")
        
        # Call the function
        mask = test_dataset._find_lemma_theorem_positions(
            solution, len(prompt_tokens), len(solution_tokens)
        )
        
        # Print tokens with mask
        print_tokens_with_mask(tokenizer, solution, mask, max_tokens_to_print=150)
        
        # Also show the raw solution text with highlighted lemma/theorem starts
        print("\n" + "="*100)
        print("RAW SOLUTION WITH LEMMA/THEOREM MARKERS")
        print("="*100)
        
        # Find lemma/theorem blocks in the solution text
        import re
        lemma_patterns = [
            r'\\begin\{lemmatheorem\}',
            r'\\begin\{lemmatheorembox\}',
            r'\\begin\{intermediatederivation\}'
        ]
        wikipedia_pattern = r'https?://(?:www\.)?(?:[a-z]{2}\.)?wikipedia\.org/wiki/([^\s\)\}\\]+)'
        
        marked_solution = solution
        for pattern in lemma_patterns:
            for match in re.finditer(pattern, marked_solution):
                pos = match.start()
                marked_solution = marked_solution[:pos] + ">>>LEMMA_START<<<" + marked_solution[pos:match.end()] + marked_solution[match.end():]
        
        # Find Wikipedia URLs
        for match in re.finditer(wikipedia_pattern, marked_solution, re.IGNORECASE):
            url_start = match.start()
            core_start = url_start + match.group(0).find('/wiki/') + len('/wiki/')
            core_end = core_start + len(match.group(1))
            marked_solution = (
                marked_solution[:core_start] + 
                ">>>WIKI_CORE_START<<" + 
                marked_solution[core_start:core_end] + 
                ">>>WIKI_CORE_END<<" + 
                marked_solution[core_end:]
            )
        
        print(marked_solution[:2000] + ("..." if len(marked_solution) > 2000 else ""))


def main():
    """Main function to run tests."""
    # Configuration
    jsonl_path = "newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct.jsonl"
    model_path = "Qwen/Qwen2-Math-7B-Instruct"  # Try this first
    
    # Check if files exist
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}")
        print("Please check the path.")
        return
    
    # Test
    test_on_examples(jsonl_path, model_path, num_examples=2)


if __name__ == "__main__":
    main()

