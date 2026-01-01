import json
import re
from collections import Counter

# Read JSONL and extract lemmatheorembox blocks
lemmas = []
with open('../newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct_filtered.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        proof = data.get('new_proof', '')
        # Extract all lemmatheorembox blocks
        matches = re.finditer(r'\\begin{lemmatheorembox}(.*?)\\end{lemmatheorembox}', proof, re.DOTALL)
        for match in matches:
            lemmas.append(match.group(0))

# Extract URLs and count usage
url_counts = Counter()
lemma_map = {}
for lemma in lemmas:
    url_match = re.search(r'\\textbf{URL:}\s*(https://[^\n]+)', lemma)
    if url_match:
        url = url_match.group(1).strip()
        url_counts[url] += 1
        if url not in lemma_map:
            lemma_map[url] = lemma

# Sort by usage count (descending)
sorted_lemmas = sorted(url_counts.items(), key=lambda x: x[1], reverse=True)

# Write LaTeX file
with open('lemmas.tex', 'w') as f:
    f.write('\\documentclass{article}\n')
    f.write('\\usepackage{amsmath}\n')
    f.write('\\begin{document}\n\n')
    
    for i, (url, count) in enumerate(sorted_lemmas, 1):
        lemma = lemma_map[url]
        f.write(f'\\section*{{Lemma {i}}}\n')
        f.write(lemma)
        f.write(f'\n\n\\textbf{{Usage count:}} {count}\n\n')
    
    f.write('\\end{document}\n')

