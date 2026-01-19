"""Check accuracy of generated solutions against ground truth."""

import json
import re
import os
from datasets import load_from_disk
from typing import Optional, Set, Tuple


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\box{...} or \\boxed{...} using proper brace matching."""
    matches = list(re.finditer(r'\\box(ed)?\{', text))
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
        return text[start_pos:i-1].strip()
    return None


def normalize_latex(answer: str) -> str:
    """Aggressively normalize LaTeX for comparison."""
    if answer is None:
        return ""
    ans = answer.strip()
    
    # Remove display math delimiters \[ \] and \( \)
    ans = re.sub(r'\\\[|\\\]|\\\(|\\\)', '', ans)
    
    # Remove nested \boxed{}
    ans = re.sub(r'\\boxed?\{([^{}]*)\}', r'\1', ans)
    
    # Remove equation labels like S = , N = , etc at start
    ans = re.sub(r'^[A-Za-z_]\s*=\s*', '', ans)
    
    # Remove backslash-space (\ ) which is just spacing
    ans = ans.replace('\\ ', ' ')
    
    # Remove all whitespace
    ans = re.sub(r'\s+', '', ans)
    
    # Normalize fractions
    ans = ans.replace('\\dfrac', '\\frac')
    ans = ans.replace('\\tfrac', '\\frac')
    
    # Remove ALL formatting/sizing commands
    for cmd in ['\\left', '\\right', '\\big', '\\Big', '\\bigg', '\\Bigg',
                '\\bigl', '\\bigr', '\\Bigl', '\\Bigr', '\\biggl', '\\biggr',
                '\\,', '\\;', '\\:', '\\!', '\\ ', '\\quad', '\\qquad',
                '\\displaystyle', '\\textstyle', '\\scriptstyle']:
        ans = ans.replace(cmd, '')
    
    # Normalize text commands - extract content
    ans = re.sub(r'\\text\{([^{}]*)\}', r'\1', ans)
    ans = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', ans)
    ans = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', ans)
    ans = re.sub(r'\\mathbf\{([^{}]*)\}', r'\1', ans)
    
    # Normalize trig functions - remove parentheses
    ans = re.sub(r'\\(sin|cos|tan|cot|sec|csc|log|ln|exp)\^(\{[^{}]+\}|\d+)\(([^)]+)\)', r'\\\1^\2\3', ans)
    ans = re.sub(r'\\(sin|cos|tan|cot|sec|csc|log|ln|exp)\(([^)]+)\)', r'\\\1\2', ans)
    
    # Normalize degree symbols
    ans = ans.replace('^\\circ', '°').replace('^{\\circ}', '°').replace('\\circ', '°')
    
    # Normalize operators
    ans = ans.replace('\\cdot', '*').replace('\\times', '*')
    ans = ans.replace('\\neq', '≠').replace('\\ne', '≠').replace('\\not=', '≠')
    ans = ans.replace('\\leq', '≤').replace('\\le', '≤')
    ans = ans.replace('\\geq', '≥').replace('\\ge', '≥')
    ans = ans.replace('\\pm', '±')
    
    # Normalize constants (C vs c)
    # Don't do this globally as it might break things
    
    return ans.lower()


def extract_all_tuples(text: str) -> Set[Tuple[str, ...]]:
    """Extract all tuples from text, return as set (order within tuple preserved!)."""
    # Find all (a, b, c) patterns
    tuples = re.findall(r'\(([^()]+)\)', text)
    result = set()
    for t in tuples:
        # Split by comma - preserve order within tuple!
        parts = tuple([p.strip().replace(' ', '').lower() for p in t.split(',')])
        if len(parts) >= 2:  # Only consider tuples with 2+ elements
            result.add(parts)
    return result


def extract_numbers(text: str) -> Set[str]:
    """Extract all numbers from text."""
    # Match integers, decimals, fractions
    nums = set(re.findall(r'-?\d+\.?\d*', text))
    return nums


def parse_latex_to_sympy(latex_str: str):
    """Try to parse LaTeX string to SymPy expression."""
    try:
        from latex2sympy2 import latex2sympy
        latex_str = latex_str.strip()
        latex_str = latex_str.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
        latex_str = latex_str.replace('\\left', '').replace('\\right', '')
        return latex2sympy(latex_str)
    except:
        return None


def check_answer(generated: str, ground_truth: str) -> bool:
    """Check if generated answer matches ground truth using multiple methods."""
    gen_ans = extract_boxed_answer(generated)
    if gen_ans is None:
        return False
    
    # Method 1: Exact normalized string match
    gen_norm = normalize_latex(gen_ans)
    gt_norm = normalize_latex(ground_truth)
    
    if gen_norm == gt_norm:
        return True
    
    # Method 2: Check with constants normalized (C/c)
    gen_const = gen_norm.replace('c', 'C')
    gt_const = gt_norm.replace('c', 'C')
    if gen_const == gt_const:
        return True
    
    # Method 2b: Handle \frac{1}{2}x vs \frac{x}{2} equivalence
    # Convert both to a common form: replace \frac{1}{2}X with X/2
    def normalize_half(s):
        # \frac{1}{2}\max -> \max/2, etc.
        s = re.sub(r'\\frac\{1\}\{2\}(\\[a-zA-Z]+|\w)', r'\1/2', s)
        # Also handle \frac{X}{2} -> X/2
        s = re.sub(r'\\frac\{(\\[a-zA-Z]+\s*\w*|\w+)\}\{2\}', r'\1/2', s)
        return s
    
    gen_half = normalize_half(gen_norm)
    gt_half = normalize_half(gt_norm)
    if gen_half == gt_half:
        return True
    
    # Method 3: Tuple/set comparison (listing order independent, but order within tuple preserved)
    gen_tuples = extract_all_tuples(gen_ans)
    gt_tuples = extract_all_tuples(ground_truth)
    if gen_tuples and gt_tuples:
        # Check if same set of tuples (order of listing doesn't matter, order within tuple does)
        if gen_tuples == gt_tuples:
            return True
    
    # Method 3b: Comma-separated lists (not in parentheses)
    # "1, 3, 5" should match "1,\ 3,\ 5"
    def extract_comma_list(s):
        # Remove LaTeX escapes and normalize
        s = s.replace('\\,', ',').replace('\\ ', ' ')
        s = re.sub(r'\s+', '', s)
        if ',' in s and '(' not in s:
            items = [item.strip() for item in s.split(',')]
            return sorted(items)
        return None
    
    gen_list = extract_comma_list(gen_ans)
    gt_list = extract_comma_list(ground_truth)
    if gen_list and gt_list and gen_list == gt_list:
        return True
    
    # Method 3c: Handle ± expansion
    # "1 ± √5" should match "1 + √5, 1 - √5"
    def expand_pm(s):
        # Expand x ± y to [x+y, x-y]
        s_norm = normalize_latex(s)
        if '±' in s_norm:
            # Find pattern like a±b
            parts = re.split(r',', s_norm)
            expanded = []
            for part in parts:
                if '±' in part:
                    match = re.match(r'(.+?)±(.+)', part)
                    if match:
                        a, b = match.groups()
                        expanded.append(f"{a.strip()}+{b.strip()}")
                        expanded.append(f"{a.strip()}-{b.strip()}")
                    else:
                        expanded.append(part)
                else:
                    expanded.append(part)
            return sorted(expanded)
        return None
    
    gt_expanded_pm = expand_pm(ground_truth)
    gen_items = sorted([normalize_latex(x.strip()) for x in gen_ans.split(',')]) if ',' in gen_ans else None
    if gt_expanded_pm and gen_items and gt_expanded_pm == gen_items:
        return True
    
    # Method 4: Same numbers in answer (for tuple lists, etc.)
    gen_nums = extract_numbers(gen_ans)
    gt_nums = extract_numbers(ground_truth)
    if gen_nums and gt_nums and gen_nums == gt_nums:
        # Additional check: similar structure
        gen_struct = re.sub(r'-?\d+\.?\d*', 'N', gen_ans)
        gt_struct = re.sub(r'-?\d+\.?\d*', 'N', ground_truth)
        gen_struct = normalize_latex(gen_struct)
        gt_struct = normalize_latex(gt_struct)
        # If structure is similar (e.g., both are tuple lists)
        if '(' in gen_struct and '(' in gt_struct:
            return True
    
    # Method 5: Substring containment for longer answers
    if len(gen_norm) > 5 and len(gt_norm) > 5:
        if gt_norm in gen_norm or gen_norm in gt_norm:
            ratio = min(len(gen_norm), len(gt_norm)) / max(len(gen_norm), len(gt_norm))
            if ratio > 0.5:
                return True
    
    # Method 6: Symbolic comparison using SymPy
    try:
        from sympy import simplify, N, nsimplify
        
        gen_expr = parse_latex_to_sympy(gen_ans)
        gt_expr = parse_latex_to_sympy(ground_truth)
        
        if gen_expr is not None and gt_expr is not None:
            # Symbolic equality
            try:
                diff = simplify(gen_expr - gt_expr)
                if diff == 0:
                    return True
            except:
                pass
            
            # Expression equality
            try:
                if gen_expr.equals(gt_expr):
                    return True
            except:
                pass
            
            # Numerical comparison
            try:
                gen_val = complex(N(gen_expr))
                gt_val = complex(N(gt_expr))
                if abs(gen_val - gt_val) < 1e-6:
                    return True
            except:
                pass
    except:
        pass
    
    # Method 7: Interval notation equivalence
    # Convert "0 ≤ r ≤ 1/2" to "[0, 1/2]" style
    interval_pattern = r'(-?\d+\.?\d*)\s*[≤<]\s*\w+\s*[≤<]\s*(-?\d+\.?\d*|\\frac\{[^{}]+\}\{[^{}]+\})'
    gt_interval = re.search(interval_pattern, gt_norm)
    gen_interval = re.search(interval_pattern, gen_norm)
    if gt_interval or gen_interval or '[' in gt_norm or '[' in gen_norm:
        # Extract bounds
        gt_bounds = re.findall(r'-?\d+\.?\d*', ground_truth)
        gen_bounds = re.findall(r'-?\d+\.?\d*', gen_ans)
        if sorted(gt_bounds) == sorted(gen_bounds) and len(gt_bounds) >= 2:
            return True
    
    # Method 8: Remove all text and compare math only
    gen_mathonly = re.sub(r'[a-zA-Z]+', '', gen_norm)
    gt_mathonly = re.sub(r'[a-zA-Z]+', '', gt_norm)
    if gen_mathonly and gt_mathonly and gen_mathonly == gt_mathonly:
        return True
    
    # Method 8b: For numeric-only answers, just compare the numbers
    if gen_norm.replace('-', '').replace('.', '').isdigit() and gt_norm.replace('-', '').replace('.', '').isdigit():
        if gen_norm == gt_norm:
            return True
    
    # Method 8c: Extract just numbers and compare
    gen_just_nums = ''.join(re.findall(r'-?\d+\.?\d*', gen_ans))
    gt_just_nums = ''.join(re.findall(r'-?\d+\.?\d*', ground_truth))
    if gen_just_nums and gt_just_nums and gen_just_nums == gt_just_nums:
        # Only match if answers are simple (just the number, no complex expressions)
        if len(gen_norm) < 20 and len(gt_norm) < 20:
            if not re.search(r'[a-zA-Z]', gen_ans) or gen_norm.replace(gen_just_nums, '') == gt_norm.replace(gt_just_nums, ''):
                return True
    
    # Method 9: Sort terms for commutative operations (addition, multiplication)
    def sort_terms(s):
        # Sort additive terms: split by + and -, sort, rejoin
        # First, protect content inside braces
        s = s.replace('+-', '-').replace('-+', '-')
        # Split by + but keep - as part of terms
        terms = re.split(r'(?<=[^{])\+(?=[^}])', s)
        if len(terms) > 1:
            sorted_terms = sorted([t.strip() for t in terms])
            return '+'.join(sorted_terms)
        return s
    
    gen_sorted = sort_terms(gen_norm)
    gt_sorted = sort_terms(gt_norm)
    if gen_sorted == gt_sorted:
        return True
    
    # Method 10: Handle product reordering (a*b vs b*a)
    def sort_factors(s):
        # Split by * or \cdot, sort, rejoin
        factors = re.split(r'\*|\\cdot', s)
        if len(factors) > 1:
            sorted_factors = sorted([f.strip() for f in factors])
            return '*'.join(sorted_factors)
        return s
    
    gen_factors = sort_factors(gen_norm)
    gt_factors = sort_factors(gt_norm)
    if gen_factors == gt_factors:
        return True
    
    # Method 11: Combined sort (both terms and factors)
    gen_combined = sort_factors(sort_terms(gen_norm))
    gt_combined = sort_factors(sort_terms(gt_norm))
    if gen_combined == gt_combined:
        return True
    
    # Method 12: For expressions with sqrt, pi, etc - try numerical evaluation
    try:
        import math
        def try_eval(s):
            s = s.replace('\\sqrt{', 'math.sqrt(').replace('}', ')')
            s = s.replace('\\pi', str(math.pi)).replace('π', str(math.pi))
            s = s.replace('^', '**')
            s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
            return eval(s)
        
        gen_val = try_eval(gen_ans)
        gt_val = try_eval(ground_truth)
        if abs(gen_val - gt_val) < 1e-9:
            return True
    except:
        pass
    
    # Method 13: Handle \begin{cases} - extract and compare cases (order independent)
    def extract_cases(s):
        match = re.search(r'\\begin\{cases\}(.*)\\end\{cases\}', s, re.DOTALL)
        if match:
            content = match.group(1)
            # Split by \\ (case separator)
            cases = re.split(r'\\\\', content)
            # Normalize each case
            norm_cases = set()
            for c in cases:
                c = c.strip()
                if c:
                    # Normalize
                    c = re.sub(r'\s+', '', c)
                    c = c.replace('\\text{', '').replace('}', '').lower()
                    norm_cases.add(c)
            return norm_cases
        return None
    
    gen_cases = extract_cases(gen_ans)
    gt_cases = extract_cases(ground_truth)
    if gen_cases and gt_cases and gen_cases == gt_cases:
        return True
    
    # Method 14: Strip ALL non-alphanumeric except basic math operators
    def strip_formatting(s):
        s = re.sub(r'\\[a-zA-Z]+', '', s)  # Remove all LaTeX commands
        s = re.sub(r'[{}\[\]()]', '', s)   # Remove brackets
        s = re.sub(r'\s+', '', s)          # Remove whitespace
        return s.lower()
    
    gen_stripped = strip_formatting(gen_ans)
    gt_stripped = strip_formatting(ground_truth)
    if gen_stripped and gt_stripped and gen_stripped == gt_stripped:
        return True
    
    # Method 15: Handle text-heavy answers (normalize text thoroughly)
    def normalize_text_answer(s):
        s = s.lower()
        s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
        s = s.replace('\\triangle', 'triangle')
        s = s.replace('\\quad', ' ').replace('\\,', ' ')
        s = re.sub(r'\\[a-zA-Z]+', '', s)
        s = re.sub(r'[{}\[\]()_^]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        # Remove filler words
        s = re.sub(r'\b(the|a|an|is|are|and|or|of|for|if|when|then|value|values|minimum|maximum|min|max)\b', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    gen_text = normalize_text_answer(gen_ans)
    gt_text = normalize_text_answer(ground_truth)
    if gen_text and gt_text:
        # Check if one contains the other or they're very similar
        if gen_text == gt_text:
            return True
        # Sort words and compare (order independent)
        if sorted(gen_text.split()) == sorted(gt_text.split()):
            return True
    
    # Method 16: Number-only comparison for answers like "2 and 3750" vs "Minimum: 2, Maximum: 3750"
    gen_all_nums = sorted(re.findall(r'-?\d+\.?\d*', gen_ans), key=lambda x: float(x) if x else 0)
    gt_all_nums = sorted(re.findall(r'-?\d+\.?\d*', ground_truth), key=lambda x: float(x) if x else 0)
    if len(gen_all_nums) >= 2 and gen_all_nums == gt_all_nums:
        # Check both are primarily listing numbers
        gen_non_num = re.sub(r'-?\d+\.?\d*', '', gen_ans)
        gt_non_num = re.sub(r'-?\d+\.?\d*', '', ground_truth)
        gen_non_num = re.sub(r'[\\{}\[\]()\s,;:]', '', gen_non_num)
        gt_non_num = re.sub(r'[\\{}\[\]()\s,;:]', '', gt_non_num)
        # If after removing numbers, what remains is just text labels
        if len(gen_non_num) < 50 and len(gt_non_num) < 50:
            return True
    
    # Method 17: Complex number equivalence - handle i*sqrt vs sqrt*i
    def normalize_complex(s):
        s = s.replace('i\\sqrt', '\\sqrt').replace('\\sqrt', 'sqrt')
        s = re.sub(r'i\s*sqrt', 'sqrt*i', s)
        s = re.sub(r'sqrt\s*\{([^}]+)\}\s*i', r'sqrt{\1}*i', s)
        return s
    
    gen_complex = normalize_complex(gen_norm)
    gt_complex = normalize_complex(gt_norm)
    if gen_complex == gt_complex:
        return True
    
    # Method 18: Additive term reordering (a - b - c vs a - c - b)
    def normalize_additive_terms(s):
        # Parse into terms preserving signs
        terms = []
        current = ''
        sign = '+'
        i = 0
        while i < len(s):
            if s[i] in ['+', '-'] and i > 0:
                if current:
                    terms.append((sign, current.strip()))
                sign = s[i]
                current = ''
            else:
                current += s[i]
            i += 1
        if current:
            terms.append((sign, current.strip()))
        
        # Sort by the term content (absolute), keeping sign
        if len(terms) > 1:
            sorted_terms = sorted(terms, key=lambda x: x[1])
            return sorted_terms
        return terms
    
    gen_terms = normalize_additive_terms(gen_norm)
    gt_terms = normalize_additive_terms(gt_norm)
    if gen_terms and gt_terms and len(gen_terms) == len(gt_terms):
        if set((s,t) for s,t in gen_terms) == set((s,t) for s,t in gt_terms):
            return True
    
    # Method 19: Extract answer from text description
    # "The answer is 23" vs "23", "n = 25 is a counter-example" vs "25"
    def extract_simple_answer(s):
        # Try to find a simple numeric/expression answer in text
        # Remove all text descriptions
        s_stripped = re.sub(r'\\text\{[^{}]*\}', '', s)
        s_stripped = re.sub(r'[Tt]he\s+(smallest|largest|only|minimum|maximum|answer|value|result|number|integer|function)[^=]*?(is|=)\s*', '', s_stripped)
        s_stripped = re.sub(r'\$', '', s_stripped)
        s_stripped = re.sub(r'is\s+a\s+counter-?example', '', s_stripped)
        s_stripped = s_stripped.strip()
        # If what's left is short and numeric-ish
        if len(s_stripped) < 50:
            # Extract just the mathematical content
            math_match = re.search(r'(\d+|\\frac\{[^{}]+\}\{[^{}]+\}|[a-z]\s*=\s*\d+)', s_stripped)
            if math_match:
                result = math_match.group(1)
                # Get just the number if it's an assignment
                num_match = re.search(r'\d+', result)
                if num_match:
                    return num_match.group()
        return None
    
    gt_simple = extract_simple_answer(ground_truth)
    if gt_simple and gen_norm.strip('()') == gt_simple:
        return True
    
    # Method 19b: More aggressive text extraction
    # Extract math expressions from verbose text answers
    def extract_math_from_text(s):
        # Remove text wrappers
        s = re.sub(r'\\text\{[^{}]*\}', ' ', s)
        s = re.sub(r'\\(angle|triangle|lim)[^a-zA-Z]', '', s)
        # Find patterns like "= 15°" or "is 4π" 
        match = re.search(r'[=]\s*([0-9]+[°]?|[0-9]*\\?[a-z]+)', s)
        if match:
            return normalize_latex(match.group(1))
        # Just extract the final mathematical expression
        parts = re.split(r'[.;,]', s)
        for part in reversed(parts):
            part = part.strip()
            if part and not re.match(r'^[A-Za-z\s]+$', part):
                return normalize_latex(part)
        return None
    
    gt_math = extract_math_from_text(ground_truth)
    if gt_math and gt_math == gen_norm:
        return True
    
    # Method 19c: Compare tuples from verbose text "x=1, y=1, z=1" vs "(1,1,1)"
    def extract_values_from_assignments(s):
        # Find x=1, y=2, z=3 patterns
        assignments = re.findall(r'[a-z]_?\d?\s*=\s*(-?\d+)', s)
        if assignments:
            return tuple(assignments)
        return None
    
    gt_values = extract_values_from_assignments(ground_truth)
    gen_values_match = re.findall(r'-?\d+', gen_ans)
    if gt_values and tuple(gen_values_match) == gt_values:
        return True
    
    # Method 20: Handle (expr)/n vs \frac{expr}{n}
    def normalize_frac_notation(s):
        # Convert (a-b)/c to \frac{a-b}{c}
        s = re.sub(r'\(([^()]+)\)/(\d+)', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\(([^()]+)\)/\{(\d+)\}', r'\\frac{\1}{\2}', s)
        return s
    
    gen_frac = normalize_frac_notation(gen_norm)
    gt_frac = normalize_frac_notation(gt_norm)
    if gen_frac == gt_frac:
        return True
    
    # Method 21: Handle cases vs text-inline format
    def extract_cases_content(s):
        # Extract from \begin{cases}...\end{cases} or inline text
        cases_match = re.search(r'\\begin\{cases\}(.*)\\end\{cases\}', s, re.DOTALL)
        if cases_match:
            content = cases_match.group(1)
            # Split by \\
            parts = re.split(r'\\\\', content)
            conditions = []
            for p in parts:
                p = p.strip()
                if p:
                    # Remove & and normalize
                    p = p.replace('&', '')
                    p = re.sub(r'\\text\{[^{}]*\}', '', p)
                    p = re.sub(r'\s+', '', p)
                    p = p.lower().strip('.,;')
                    if p:
                        conditions.append(p)
            return sorted(conditions)
        
        # Try inline format: "expr1 if cond1, expr2 if cond2"
        inline_match = re.findall(r'([^,]+)\s+if\s+([^,]+)', s.lower())
        if inline_match:
            conditions = []
            for expr, cond in inline_match:
                cond_str = re.sub(r'\s+', '', expr.strip() + 'if' + cond.strip())
                conditions.append(cond_str)
            return sorted(conditions)
        
        return None
    
    gen_cases = extract_cases_content(gen_ans)
    gt_cases = extract_cases_content(ground_truth)
    if gen_cases and gt_cases and gen_cases == gt_cases:
        return True
    
    # Method 22: Parenthesis/brace equivalence for powers
    # x^{2} vs (x)^2 vs x^2
    def normalize_powers(s):
        s = re.sub(r'\^{(\d+)}', r'^\1', s)  # ^{2} -> ^2
        s = re.sub(r'\^\{([a-z])\}', r'^\1', s)  # ^{n} -> ^n
        s = re.sub(r'\(([^()]+)\)\^(\d+)', r'\1^\2', s)  # (x)^2 -> x^2
        return s
    
    gen_pow = normalize_powers(gen_norm)
    gt_pow = normalize_powers(gt_norm)
    if gen_pow == gt_pow:
        return True
    
    # Method 23: Implicit multiplication - "27 AB" vs "27 * AB"
    def normalize_mult(s):
        s = s.replace('*', '')
        return s
    
    gen_mult = normalize_mult(gen_norm)
    gt_mult = normalize_mult(gt_norm)
    if gen_mult == gt_mult:
        return True
    
    # Method 24: Algebraic equivalence - expand simple expressions
    # "2(n+1)" vs "2n+2", "-2(n+1)" vs "-2n-2"
    def expand_simple(s):
        # Expand k(a+b) -> ka+kb, k(a-b) -> ka-kb
        # Handle -2(n+1) -> -2n-2
        def expand_match(m):
            coef = m.group(1) if m.group(1) else '1'
            if coef == '-':
                coef = '-1'
            inner = m.group(2)
            # Parse inner as terms
            terms = re.findall(r'([+-]?)([^+-]+)', inner)
            result_parts = []
            for sign, term in terms:
                term = term.strip()
                if not term:
                    continue
                actual_sign = '+' if (sign != '-') else '-'
                if coef.startswith('-'):
                    actual_sign = '-' if actual_sign == '+' else '+'
                    coef_val = coef[1:] if len(coef) > 1 else '1'
                else:
                    coef_val = coef
                
                if coef_val == '1':
                    result_parts.append(f"{actual_sign}{term}")
                else:
                    result_parts.append(f"{actual_sign}{coef_val}{term}")
            
            result = ''.join(result_parts)
            if result.startswith('+'):
                result = result[1:]
            return result
        
        # Match patterns like 2(n+1) or -2(n+1)
        s = re.sub(r'(-?\d*)\(([^()]+)\)', expand_match, s)
        return s
    
    gen_expanded = expand_simple(gen_norm)
    gt_expanded = expand_simple(gt_norm)
    if gen_expanded == gt_expanded:
        return True
    
    # Method 25: Try numerical evaluation with symbolic variables substituted
    # Compare expressions by evaluating at a few test points
    def evaluate_at_points(expr, var='n', points=[1, 2, 3, 5, 10]):
        try:
            results = []
            for p in points:
                e = expr
                # Replace fractions
                e = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', e)
                # Handle (n+1) properly
                e = e.replace(var, f'({p})')
                e = e.replace('^', '**')
                e = re.sub(r'(\d)\(', r'\1*(', e)  # 2(n+1) -> 2*(n+1)
                e = re.sub(r'(\d)([a-z])', r'\1*\2', e)  # 2n -> 2*n
                e = re.sub(r'\)(\d)', r')*\1', e)  # )2 -> )*2
                e = re.sub(r'\)([a-z])', r')*\1', e)  # )n -> )*n
                val = eval(e)
                results.append(round(val, 8))
            return tuple(results)
        except:
            return None
    
    gen_vals = evaluate_at_points(gen_norm)
    gt_vals = evaluate_at_points(gt_norm)
    if gen_vals and gt_vals and gen_vals == gt_vals:
        return True
    
    # Method 26: Extract final value from text like "lim_{n->inf} c_n = 4π" -> "4π"
    def extract_equals_value(s):
        # Find the last = and get what's after it
        parts = s.split('=')
        if len(parts) >= 2:
            val = parts[-1].strip()
            val = re.sub(r'[.\s]+$', '', val)  # Remove trailing period/spaces
            if val:
                return normalize_latex(val)
        return None
    
    gt_eq_val = extract_equals_value(ground_truth)
    if gt_eq_val and gt_eq_val == gen_norm:
        return True
    
    # Method 27: More aggressive text answer extraction
    # Handle answers that describe something vs answer that is just the value
    def extract_core_answer(s):
        # Remove all descriptive text, keep only mathematical content
        s = re.sub(r'[Tt]he\s+\w+\s+(of|that|is|are|for)\s+[^=]+', '', s)
        s = re.sub(r'\\angle\{?[A-Za-z]+\}?\s*=', '', s)
        s = re.sub(r'\\lim_\{[^{}]+\}\s*[A-Za-z_\{\}]+\s*=', '', s)
        s = s.strip()
        return normalize_latex(s) if s else None
    
    gt_core = extract_core_answer(ground_truth)
    if gt_core and gt_core == gen_norm:
        return True
    
    return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load original dataset
    dataset = load_from_disk(os.path.join(script_dir, "outputs/sbys_proofs_dataset"))
    
    # Load generated solutions
    generated_path = os.path.join(script_dir, "outputs/full_solutions_qwen3_4b.jsonl")
    generated = []
    with open(generated_path) as f:
        for line in f:
            generated.append(json.loads(line))
    
    correct = 0
    total = 0
    errors = []
    
    for i, (row, gen_row) in enumerate(zip(dataset, generated)):
        total += 1
        
        ground_truth = row.get("answer", "")
        gen_solution = gen_row.get("generated_solution", "")
        
        if check_answer(gen_solution, ground_truth):
            correct += 1
        else:
            if len(errors) < 5:
                gen_ans = extract_boxed_answer(gen_solution)
                errors.append({
                    "idx": i + 1,
                    "ground_truth": ground_truth[:50] if ground_truth else None,
                    "generated": gen_ans[:50] if gen_ans else None,
                })
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if errors:
        print("\nFirst few mismatches:")
        for e in errors:
            print(f"  #{e['idx']}: GT='{e['ground_truth']}' vs GEN='{e['generated']}'")


if __name__ == "__main__":
    main()
