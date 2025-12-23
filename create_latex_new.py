#!/usr/bin/env python3
"""Minimal LaTeX document creator - fixes double backslashes and adds document structure."""

import json

def fix_double_backslashes(text):
    """Convert double backslashes to single backslashes."""
    if not text:
        return ""
    while '\\\\' in text:
        text = text.replace('\\\\', '\\')
    return text

def main():
    # Read the JSONL file
    problems = []
    with open('outputs/math_solutions.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
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
        solution = item.get('solution', '')
        
        # Fix double backslashes
        solution = fix_double_backslashes(solution)
        
        # Add section and solution
        latex_content += f"\\section{{Problem {i}}}\n\n"
        latex_content += solution
        latex_content += "\n\n\\newpage\n\n"
    
    latex_content += "\\end{document}\n"
    
    # Write to file
    output_file = 'outputs/math_problems.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX document created: {output_file}")
    print(f"Total problems: {len(problems)}")

if __name__ == '__main__':
    main()

