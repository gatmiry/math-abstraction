import json
import re

def escape_latex_urls(text):
    """Replace URLs with url commands to handle underscores and special characters"""
    # Pattern to match URLs
    url_pattern = r'https?://[^\s\)]+'
    def replace_url(match):
        url = match.group(0)
        # Remove trailing punctuation that might not be part of URL
        url = url.rstrip('.,;:!?')
        return f'\\url{{{url}}}'
    return re.sub(url_pattern, replace_url, text)

problems = []
with open('../outputs2/new_transformed_model_qwen2-math-7b-instruct_evaluation_results_geometry_level_4.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['baseline_correct'] and not data['finetuned_correct']:
            problems.append(data)

code = """import json

problems = []
with open('../outputs2/new_transformed_model_qwen2-math-7b-instruct_evaluation_results_geometry_level_4.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['baseline_correct'] and not data['finetuned_correct']:
            problems.append(data)

with open('regressions.tex', 'w') as f:
    f.write('\\\\documentclass{article}\\\\n')
    f.write('\\\\usepackage{amsmath}\\\\n')
    f.write('\\\\usepackage{geometry}\\\\n')
    f.write('\\\\usepackage{fancyvrb}\\\\n')
    f.write('\\\\usepackage{xcolor}\\\\n')
    f.write('\\\\usepackage{tcolorbox}\\\\n')
    f.write('\\\\usepackage{hyperref}\\\\n')
    f.write('\\\\n')
    f.write('% Define intermediatederivation environment\\\\n')
    f.write('\\\\newtcolorbox{intermediatederivation}{\\\\n')
    f.write('    colback=blue!5!white,\\\\n')
    f.write('    colframe=blue!75!black,\\\\n')
    f.write('    boxrule=1pt,\\\\n')
    f.write('    arc=3pt,\\\\n')
    f.write('    left=5pt,\\\\n')
    f.write('    right=5pt,\\\\n')
    f.write('    top=5pt,\\\\n')
    f.write('    bottom=5pt\\\\n')
    f.write('}\\\\n')
    f.write('\\\\n')
    f.write('% Define lemmatheorembox environment\\\\n')
    f.write('\\\\newtcolorbox{lemmatheorembox}{\\\\n')
    f.write('    colback=green!5!white,\\\\n')
    f.write('    colframe=green!75!black,\\\\n')
    f.write('    boxrule=1pt,\\\\n')
    f.write('    arc=3pt,\\\\n')
    f.write('    left=5pt,\\\\n')
    f.write('    right=5pt,\\\\n')
    f.write('    top=5pt,\\\\n')
    f.write('    bottom=5pt,\\\\n')
    f.write('    title=Lemma/Theorem\\\\n')
    f.write('}\\\\n')
    f.write('\\\\n')
    f.write('\\\\begin{document}\\\\n\\\\n')
    for i, p in enumerate(problems, 1):
        f.write(f'\\\\section*{{Problem {i}}}\\\\n')
        f.write('\\\\textbf{Problem:}\\\\n')
        f.write(p['problem'])
        f.write('\\\\n\\\\n')
        gt = p['ground_truth_answer']
        ba = p['baseline_answer']
        fa = p['finetuned_answer']
        br = p.get('baseline_response', '')
        fr = p.get('finetuned_response', '')
        f.write(f'\\\\textbf{{Ground Truth:}} {gt}\\\\n\\\\n')
        f.write(f'\\\\textbf{{Baseline Answer:}} {ba}\\\\n\\\\n')
        f.write(f'\\\\textbf{{Finetuned Answer:}} {fa}\\\\n\\\\n')
        if br:
            f.write('\\\\textbf{Baseline Response:}\\\\n')
            f.write('\\\\begin{quote}\\\\n')
            # Escape backslashes for the code string representation
            f.write(br.replace('\\\\', '\\\\\\\\'))
            f.write('\\\\n\\\\end{quote}\\\\n\\\\n')
        if fr:
            f.write('\\\\textbf{Finetuned Response:}\\\\n')
            f.write('\\\\begin{quote}\\\\n')
            # Escape backslashes for the code string representation
            f.write(fr.replace('\\\\', '\\\\\\\\'))
            f.write('\\\\n\\\\end{quote}\\\\n\\\\n')
        f.write('\\\\newpage\\\\n\\\\n')
    f.write('\\\\end{document}\\\\n')
"""

with open('regressions.tex', 'w') as f:
    f.write('\\documentclass{article}\n')
    f.write('\\usepackage{amsmath}\n')
    f.write('\\usepackage{geometry}\n')
    f.write('\\usepackage{listings}\n')
    f.write('\\usepackage{fancyvrb}\n')
    f.write('\\usepackage{xcolor}\n')
    f.write('\\usepackage{tcolorbox}\n')
    f.write('\\usepackage{hyperref}\n')
    f.write('\n')
    f.write('% Define intermediatederivation environment\n')
    f.write('\\newtcolorbox{intermediatederivation}{\n')
    f.write('    colback=blue!5!white,\n')
    f.write('    colframe=blue!75!black,\n')
    f.write('    boxrule=1pt,\n')
    f.write('    arc=3pt,\n')
    f.write('    left=5pt,\n')
    f.write('    right=5pt,\n')
    f.write('    top=5pt,\n')
    f.write('    bottom=5pt\n')
    f.write('}\n')
    f.write('\n')
    f.write('% Define lemmatheorembox environment\n')
    f.write('\\newtcolorbox{lemmatheorembox}{\n')
    f.write('    colback=green!5!white,\n')
    f.write('    colframe=green!75!black,\n')
    f.write('    boxrule=1pt,\n')
    f.write('    arc=3pt,\n')
    f.write('    left=5pt,\n')
    f.write('    right=5pt,\n')
    f.write('    top=5pt,\n')
    f.write('    bottom=5pt,\n')
    f.write('    title=Lemma/Theorem\n')
    f.write('}\n')
    f.write('\n')
    f.write('\\begin{document}\n\n')
    f.write('\\section*{Python Code}\n')
    f.write('\\begin{lstlisting}[language=Python]\n')
    f.write(code)
    f.write('\\end{lstlisting}\n')
    f.write('\\newpage\n\n')
    
    for i, p in enumerate(problems, 1):
        f.write(f'\\section*{{Problem {i}}}\n')
        f.write('\\textbf{Problem:}\n')
        f.write(p['problem'])
        f.write('\n\n')
        gt = p['ground_truth_answer']
        ba = p['baseline_answer']
        fa = p['finetuned_answer']
        br = p.get('baseline_response', '')
        fr = p.get('finetuned_response', '')
        f.write(f'\\textbf{{Ground Truth:}} {gt}\n\n')
        f.write(f'\\textbf{{Baseline Answer:}} {ba}\n\n')
        f.write(f'\\textbf{{Finetuned Answer:}} {fa}\n\n')
        if br:
            f.write('\\textbf{Baseline Response:}\n')
            f.write('\\begin{quote}\n')
            # Escape URLs to handle underscores and special characters
            br_escaped = escape_latex_urls(br)
            f.write(br_escaped)
            f.write('\n\\end{quote}\n\n')
        if fr:
            f.write('\\textbf{Finetuned Response:}\n')
            f.write('\\begin{quote}\n')
            # Escape URLs to handle underscores and special characters
            fr_escaped = escape_latex_urls(fr)
            f.write(fr_escaped)
            f.write('\n\\end{quote}\n\n')
        f.write('\\newpage\n\n')
    
    f.write('\\end{document}\n')

print(f'Found {len(problems)} regressions')
