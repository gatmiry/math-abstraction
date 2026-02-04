#!/usr/bin/env python3
"""Create a LaTeX document from the math_solutions.jsonl file."""

import json
import re

def fix_double_backslashes(text):
    """Convert double backslashes to single backslashes in LaTeX commands."""
    if not text:
        return ""
    # Replace \\ (double backslash) with \ (single backslash)
    # This handles cases where the model generates \\begin instead of \begin
    # Use string replace instead of regex to avoid escape sequence issues
    # Iterate to handle cases like \\\\ (quadruple backslash) -> \\ -> \
    while '\\\\' in text:
        text = text.replace('\\\\', '\\')
    return text

def process_latex_math(text):
    """Convert various LaTeX math formats to standard $ format."""
    if not text:
        return ""
    # Replace \( and \) with $ for inline math
    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    return text

def process_markdown(text):
    """Convert markdown formatting to LaTeX."""
    if not text:
        return ""
    # Convert **bold** to \textbf{bold}
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    return text

def process_asy_code(text):
    """Handle Asymptote code blocks."""
    if not text:
        return ""
    # Replace [asy]...[/asy] with a comment or placeholder
    asy_pattern = r'\[asy\].*?\[/asy\]'
    text = re.sub(asy_pattern, '\\textit{[Diagram available in original source]}', text, flags=re.DOTALL)
    return text

def escape_latex(text):
    """Escape special LaTeX characters, but preserve LaTeX commands."""
    if not text:
        return ""
    # Escape special characters but preserve math mode and LaTeX commands
    # We'll be careful not to escape inside $...$ and not to escape \begin, \end, etc.
    special_chars = ['&', '%', '$', '#', '^', '_', '{', '}', '~', '\\']
    result = []
    in_math = False
    i = 0
    while i < len(text):
        if text[i] == '$' and (i == 0 or text[i-1] != '\\'):
            in_math = not in_math
            result.append('$')
            i += 1
        elif not in_math and text[i] == '\\':
            # Check if this is a LaTeX command
            if i + 1 < len(text):
                next_char = text[i + 1]
                if next_char.isalpha():
                    # It's a LaTeX command like \begin, \end, \textbf, etc.
                    # Copy the backslash and all following letters
                    result.append('\\')
                    i += 1
                    # Copy all following letters (command name)
                    while i < len(text) and text[i].isalpha():
                        result.append(text[i])
                        i += 1
                    continue  # Skip the i += 1 at the end since we already incremented
                else:
                    # Single character command like \$, \%, or special case - treat as command
                    result.append('\\')
            else:
                # Backslash at end of string - escape it as literal
                result.append('\\textbackslash{}')
            i += 1
        elif not in_math and text[i] in special_chars and text[i] != '$' and text[i] != '\\':
            if text[i] == '&':
                result.append('\\&')
            elif text[i] == '%':
                result.append('\\%')
            elif text[i] == '#':
                result.append('\\#')
            elif text[i] == '^':
                result.append('\\textasciicircum{}')
            elif text[i] == '_':
                result.append('\\_')
            elif text[i] == '{':
                result.append('\\{')
            elif text[i] == '}':
                result.append('\\}')
            elif text[i] == '~':
                result.append('\\textasciitilde{}')
            else:
                result.append(text[i])
            i += 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)

def parse_solution_json(solution_text):
    """Parse JSON from solution text, handling code blocks."""
    if not solution_text:
        return None
    
    # Try to extract JSON from code blocks
    json_match = re.search(r'```json\s*\n(.*?)\n```', solution_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try without code block markers
        json_match = re.search(r'\{.*\}', solution_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def format_lemma_ref(match):
    """Format a lemma reference for LaTeX."""
    full_match = match.group(0)
    lemma_id = match.group(1)
    topic = match.group(2)
    statement = match.group(3)
    
    # Escape LaTeX in statement
    statement = escape_latex(statement)
    statement = process_latex_math(statement)
    
    return f'\\textbf{{Lemma {lemma_id}}} ({topic}): {statement}'

def process_lemma_refs(text):
    """Process lemma references in the text."""
    if not text:
        return ""
    
    # Pattern: <lemma_ref id="LEMMA_ID" topic="TOPIC">STATEMENT</lemma_ref>
    pattern = r'<lemma_ref\s+id="([^"]+)"\s+topic="([^"]+)">([^<]+)</lemma_ref>'
    
    def replace_lemma(match):
        return format_lemma_ref(match)
    
    text = re.sub(pattern, replace_lemma, text)
    
    # Also handle single quotes
    pattern2 = r"<lemma_ref\s+id='([^']+)'\s+topic='([^']+)'>([^<]+)</lemma_ref>"
    text = re.sub(pattern2, replace_lemma, text)
    
    return text

def format_proof_structure(proof_data):
    """Format a structured proof into LaTeX."""
    latex = []
    
    # Definitions
    if 'definitions' in proof_data:
        definitions = proof_data['definitions']
        latex.append('\\textbf{Definitions:}')
        latex.append('\\begin{itemize}')
        if isinstance(definitions, list):
            for defn in definitions:
                defn_text = process_lemma_refs(str(defn))
                defn_text = process_latex_math(defn_text)
                defn_text = escape_latex(defn_text)
                latex.append(f'\\item {defn_text}')
        elif isinstance(definitions, dict):
            for key, value in definitions.items():
                defn_text = process_lemma_refs(str(value))
                defn_text = process_latex_math(defn_text)
                defn_text = escape_latex(defn_text)
                latex.append(f'\\item \\textbf{{{key}}}: {defn_text}')
        latex.append('\\end{itemize}')
        latex.append('')
    
    # Plan
    if 'plan' in proof_data:
        plan = proof_data['plan']
        latex.append('\\textbf{Plan:}')
        latex.append('\\begin{enumerate}')
        if isinstance(plan, list):
            for step in plan:
                step_text = process_lemma_refs(str(step))
                step_text = process_latex_math(step_text)
                step_text = escape_latex(step_text)
                latex.append(f'\\item {step_text}')
        latex.append('\\end{enumerate}')
        latex.append('')
    
    # Steps
    if 'steps' in proof_data:
        steps = proof_data['steps']
        latex.append('\\textbf{Proof:}')
        latex.append('\\begin{enumerate}')
        
        for step in steps:
            step_latex = []
            
            # Handle different step formats
            if isinstance(step, dict):
                # Step text
                if 'step' in step:
                    step_text = process_lemma_refs(str(step['step']))
                    step_text = process_latex_math(step_text)
                    step_text = escape_latex(step_text)
                    step_latex.append(step_text)
                
                # Justification/lemma references
                if 'justification' in step:
                    justification = step['justification']
                    if justification and justification != "Derived Step":
                        justification = process_lemma_refs(str(justification))
                        justification = process_latex_math(justification)
                        justification = escape_latex(justification)
                        step_latex.append(f'\\textit{{[Justification: {justification}]}}')
                    elif justification == "Derived Step":
                        step_latex.append('\\textit{[Derived Step]}')
                
                # Details/explanation
                if 'details' in step:
                    if isinstance(step['details'], list):
                        step_latex.append('\\begin{itemize}')
                        for detail in step['details']:
                            detail_text = process_lemma_refs(str(detail))
                            detail_text = process_latex_math(detail_text)
                            detail_text = escape_latex(detail_text)
                            step_latex.append(f'\\item {detail_text}')
                        step_latex.append('\\end{itemize}')
                    else:
                        detail_text = process_lemma_refs(str(step['details']))
                        detail_text = process_latex_math(detail_text)
                        detail_text = escape_latex(detail_text)
                        step_latex.append(f'\\textit{{[Details: {detail_text}]}}')
                
                # Explanation
                if 'explanation' in step:
                    explanation = process_lemma_refs(str(step['explanation']))
                    explanation = process_latex_math(explanation)
                    explanation = escape_latex(explanation)
                    step_latex.append(f'\\textit{{[Explanation: {explanation}]}}')
                
                # Result
                if 'result' in step:
                    result = process_lemma_refs(str(step['result']))
                    result = process_latex_math(result)
                    result = escape_latex(result)
                    step_latex.append(f'\\textbf{{Result:}} {result}')
                
                # Lemma ref object
                if 'lemma_ref' in step:
                    lemma_ref = step['lemma_ref']
                    if isinstance(lemma_ref, dict):
                        lemma_id = lemma_ref.get('id', '')
                        topic = lemma_ref.get('topic', '')
                        statement = lemma_ref.get('canonical_statement', '')
                        statement = escape_latex(statement)
                        statement = process_latex_math(statement)
                        step_latex.append(f'\\textbf{{Lemma {lemma_id}}} ({topic}): {statement}')
                
                # Derived step
                if 'derived_step' in step:
                    derived = step['derived_step']
                    if isinstance(derived, dict) and 'explanation' in derived:
                        explanation = process_lemma_refs(str(derived['explanation']))
                        explanation = process_latex_math(explanation)
                        explanation = escape_latex(explanation)
                        step_latex.append(f'\\textit{{[Derived Step: {explanation}]}}')
            
            elif isinstance(step, str):
                step_text = process_lemma_refs(step)
                step_text = process_latex_math(step_text)
                step_text = escape_latex(step_text)
                step_latex.append(step_text)
            
            if step_latex:
                latex.append(f'\\item {" ".join(step_latex)}')
        
        latex.append('\\end{enumerate}')
        latex.append('')
    
    # Conclusion
    if 'conclusion' in proof_data:
        conclusion = process_lemma_refs(str(proof_data['conclusion']))
        conclusion = process_latex_math(conclusion)
        conclusion = escape_latex(conclusion)
        latex.append(f'\\textbf{{Conclusion:}} {conclusion}')
        latex.append('')
    
    return '\n'.join(latex)

def format_solution(solution_text):
    """Format a solution text into LaTeX, handling both JSON and plain text."""
    if not solution_text:
        return "\\textit{[No solution provided]}"
    
    # Fix double backslashes in LaTeX commands (e.g., \\begin -> \begin)
    solution_text = fix_double_backslashes(solution_text)
    
    # Try to parse as JSON
    proof_data = parse_solution_json(solution_text)
    
    if proof_data:
        # Check if it has a proof structure
        if 'proof' in proof_data:
            proof = proof_data['proof']
            return format_proof_structure(proof)
        elif 'result' in proof_data:
            # Handle error/result messages
            result = process_lemma_refs(str(proof_data['result']))
            result = process_latex_math(result)
            result = escape_latex(result)
            return result
        else:
            # Fall back to plain text formatting
            text = json.dumps(proof_data, indent=2)
            text = process_lemma_refs(text)
            text = process_latex_math(text)
            text = escape_latex(text)
            return f'\\texttt{{{text}}}'
    else:
        # Plain text solution - process lemma refs and format
        text = process_lemma_refs(solution_text)
        text = process_latex_math(text)
        text = process_markdown(text)
        text = escape_latex(text)
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

% Define custom environment for lemmatheorem (allows structured lemma/theorem content)
\\newenvironment{lemmatheorem}
  {\\begin{quote}}
  {\\end{quote}}

% Define custom environment for intermediate derivations
\\newenvironment{intermediatederivation}
  {\\begin{itshape}}
  {\\end{itshape}}

\\title{Math Problems and Solutions}
\\author{Generated from Competition Math Dataset}
\\date{\\today}

\\begin{document}

\\maketitle

\\tableofcontents
\\newpage

"""
    
    for i, item in enumerate(problems, 1):
        problem = item['problem']
        ground_truth = item['ground_truth']
        solution = item['solution']
        
        # Fix double backslashes in solution (e.g., \\begin -> \begin)
        solution = fix_double_backslashes(solution)
        
        # Process Asymptote code
        problem = process_asy_code(problem)
        
        # Process LaTeX math formatting
        problem = process_latex_math(problem)
        ground_truth = process_latex_math(ground_truth)
        
        # Escape LaTeX special characters
        problem = escape_latex(problem)
        ground_truth = escape_latex(ground_truth)
        
        # Format solution with proof structure
        formatted_solution = format_solution(solution)
        
        # Create section for each problem
        latex_content += f"""
\\section{{Problem {i}}}

\\subsection{{Problem Statement}}
{problem}

\\subsection{{Ground Truth Solution}}
{ground_truth}

\\subsection{{Generated Solution}}
{formatted_solution}

\\newpage
"""
    
    latex_content += """
\\end{document}
"""
    
    # Write to file
    output_file = 'outputs/math_problems.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX document created: {output_file}")
    print(f"Total problems: {len(problems)}")

if __name__ == '__main__':
    main()
