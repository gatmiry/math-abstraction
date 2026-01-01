import json

problems = []
with open('../outputs2/new_transformed_model_qwen2-math-7b-instruct_evaluation_results_geometry_level_4.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if not data['baseline_correct'] and data['finetuned_correct']:
            problems.append(data)

code = """import json

problems = []
with open('../outputs2/new_transformed_model_qwen2-math-7b-instruct_evaluation_results_geometry_level_4.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if not data['baseline_correct'] and data['finetuned_correct']:
            problems.append(data)

with open('improvements.tex', 'w') as f:
    f.write('\\\\documentclass{article}\\\\n')
    f.write('\\\\usepackage{amsmath}\\\\n')
    f.write('\\\\usepackage{geometry}\\\\n')
    f.write('\\\\begin{document}\\\\n\\\\n')
    for i, p in enumerate(problems, 1):
        f.write(f'\\\\section*{{Problem {i}}}\\\\n')
        f.write('\\\\textbf{Problem:}\\\\n')
        f.write(p['problem'])
        f.write('\\\\n\\\\n')
        gt = p['ground_truth_answer']
        ba = p['baseline_answer']
        fa = p['finetuned_answer']
        f.write(f'\\\\textbf{{Ground Truth:}} {gt}\\\\n\\\\n')
        f.write(f'\\\\textbf{{Baseline Answer:}} {ba}\\\\n\\\\n')
        f.write(f'\\\\textbf{{Finetuned Answer:}} {fa}\\\\n\\\\n')
        f.write('\\\\newpage\\\\n\\\\n')
    f.write('\\\\end{document}\\\\n')
"""

with open('improvements.tex', 'w') as f:
    f.write('\\documentclass{article}\n')
    f.write('\\usepackage{amsmath}\n')
    f.write('\\usepackage{geometry}\n')
    f.write('\\usepackage{listings}\n')
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
        f.write(f'\\textbf{{Ground Truth:}} {gt}\n\n')
        f.write(f'\\textbf{{Baseline Answer:}} {ba}\n\n')
        f.write(f'\\textbf{{Finetuned Answer:}} {fa}\n\n')
        f.write('\\newpage\n\n')
    
    f.write('\\end{document}\n')

print(f'Found {len(problems)} improvements')
