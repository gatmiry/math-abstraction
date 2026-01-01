#!/usr/bin/env python3
"""
Create a LaTeX document from any JSONL file containing problem/solution dictionaries.
Similar to create_latex_new.py but accepts any JSONL file path.
"""

import json
import os
import sys
import argparse


def fix_double_backslashes(text):
    """Convert double backslashes to single backslashes."""
    if not text:
        return ""
    while '\\\\' in text:
        text = text.replace('\\\\', '\\')
    return text


def escape_latex_special_chars(text):
    """Escape LaTeX special characters that might appear in plain text."""
    # Only escape if not already in math mode or LaTeX commands
    # This is a simple version - more sophisticated escaping can be added if needed
    return text


def create_latex_from_jsonl(input_file, output_file=None, include_problem=True):
    """
    Create a LaTeX document from a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output LaTeX file (default: input_file with .tex extension)
        include_problem: Whether to include the problem statement in the LaTeX document
    """
    # Read the JSONL file
    problems = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    problems.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    if not problems:
        print(f"Error: No valid entries found in {input_file}")
        return
    
    # Determine output file name
    if output_file is None:
        base_name = os.path.basename(input_file)
        base_name = os.path.splitext(base_name)[0]
        # Create output directory
        output_dir = "output_openmathreasoning"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}.tex")
    else:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "output_openmathreasoning"
        if not output_dir:
            output_dir = "output_openmathreasoning"
        os.makedirs(output_dir, exist_ok=True)
    
    # Create LaTeX document
    latex_content = """\\documentclass[11pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsthm}
\\usepackage{geometry}
\\usepackage{hyperref}
\\geometry{margin=1in}

% Define custom environments
\\newenvironment{lemmatheorem}
  {\\begin{quote}}
  {\\end{quote}}

\\newenvironment{intermediatederivation}
  {\\begin{itshape}}
  {\\end{itshape}}

\\begin{document}

"""
    
    for i, item in enumerate(problems, 1):
        # Fix double backslashes for all string values
        for key, value in item.items():
            if isinstance(value, str):
                item[key] = fix_double_backslashes(value)
        
        # Add section
        latex_content += f"\\section{{Problem {i}}}\n\n"
        
        # Define the order and special handling for certain keys
        # Keys that should be displayed first (in order)
        priority_keys = ['problem', 'solution', 'ground_truth']
        
        # Keys to skip (already handled above or not needed)
        skip_keys = set()
        
        # Add problem statement if requested and available
        if include_problem and 'problem' in item and item['problem']:
            latex_content += "\\textbf{Problem:}\n\n"
            latex_content += item['problem']
            latex_content += "\n\n"
            skip_keys.add('problem')
        
        # Add solution
        if 'solution' in item and item['solution']:
            latex_content += "\\textbf{Solution:}\n\n"
            latex_content += item['solution']
            latex_content += "\n\n"
            skip_keys.add('solution')
        elif 'solution' in item:
            latex_content += "\\textbf{Solution:}\n\n"
            latex_content += "\\textit{No solution provided.}\n\n"
            skip_keys.add('solution')
        
        # Add all other keys with titles
        # First, handle remaining priority keys
        for key in priority_keys:
            if key in item and key not in skip_keys:
                value = item[key]
                if value:  # Only display if not empty
                    # Format key name (replace underscores, capitalize)
                    title = key.replace('_', ' ').title()
                    latex_content += f"\\textbf{{{title}:}}\n\n"
                    if isinstance(value, (dict, list)):
                        latex_content += "\\begin{verbatim}\n"
                        latex_content += json.dumps(value, indent=2)
                        latex_content += "\n\\end{verbatim}\n\n"
                    else:
                        latex_content += f"{value}\n\n"
                    skip_keys.add(key)
        
        # Then handle all other keys (sorted for consistent ordering)
        for key, value in sorted(item.items()):
            if key not in skip_keys and value:  # Skip empty values
                # Format key name (replace underscores, capitalize)
                title = key.replace('_', ' ').title()
                latex_content += f"\\textbf{{{title}:}}\n\n"
                
                # Handle different value types
                if isinstance(value, dict):
                    latex_content += "\\begin{verbatim}\n"
                    latex_content += json.dumps(value, indent=2)
                    latex_content += "\n\\end{verbatim}\n\n"
                elif isinstance(value, list):
                    latex_content += "\\begin{verbatim}\n"
                    latex_content += json.dumps(value, indent=2)
                    latex_content += "\n\\end{verbatim}\n\n"
                elif isinstance(value, (int, float, bool)):
                    latex_content += f"{value}\n\n"
                else:
                    # String value
                    latex_content += f"{value}\n\n"
        
        latex_content += "\\newpage\n\n"
    
    latex_content += "\\end{document}\n"
    
    # Write to file (directory already created above)
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX document created: {output_file}")
    print(f"Total problems: {len(problems)}")
    print(f"Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a LaTeX document from a JSONL file containing problems and solutions'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input JSONL file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output LaTeX file (default: input_file with .tex extension)'
    )
    parser.add_argument(
        '--no-problem',
        action='store_true',
        help='Do not include problem statements in the LaTeX document'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    create_latex_from_jsonl(
        args.input_file,
        output_file=args.output,
        include_problem=not args.no_problem
    )


if __name__ == '__main__':
    # If no arguments provided, use default file
    if len(sys.argv) == 1:
        # Default to the openmathreasoning file
        default_file = 'outputs2/transformed_solutions_qwen2-math-7b-instruct_filtered.jsonl'
        if os.path.exists(default_file):
            print(f"No arguments provided. Using default file: {default_file}")
            create_latex_from_jsonl(default_file)
        else:
            print("Usage: python create_latex_from_jsonl.py <input_file.jsonl> [-o output.tex] [--no-problem]")
            print("\nExample:")
            print("  python create_latex_from_jsonl.py outputs2/math_solutions_openmathreasoning_50.jsonl")
            print("  python create_latex_from_jsonl.py outputs/math_solutions.jsonl -o outputs/math_problems.tex")
            sys.exit(1)
    else:
        main()

