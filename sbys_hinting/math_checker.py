"""Math answer verification using math_verify library (Hugging Face standard)."""

import re
from math_verify import parse, verify, LatexExtractionConfig, StringExtractionConfig, ExprExtractionConfig


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} using proper brace matching."""
    if not text:
        return None
        
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


def normalize_latex_for_parsing(s: str) -> str:
    """Normalize LaTeX commands before parsing."""
    if not s:
        return ""
    # Remove variable assignment prefixes (M=, V=, etc.)
    s = re.sub(r'^[A-Za-z]\s*=\s*', '', s)
    # Remove currency symbols
    s = s.replace('\\$', '').replace('$', '')
    # Normalize exponent braces: ^{1234} -> ^1234 (for numeric exponents only)
    # This helps math_verify parse exponents correctly
    s = re.sub(r'\^\{(\d+)\}', r'^\1', s)
    # Normalize fraction variants
    s = s.replace('\\dfrac', '\\frac')
    s = s.replace('\\tfrac', '\\frac')
    # Fix \frac followed by space and single char/digit: \frac 9{...} -> \frac{9}{...}
    s = re.sub(r'\\frac\s+(\d|\w)\s*\{', r'\\frac{\1}{', s)
    # Fix \sqrt followed by space and single char/digit: \sqrt 2 -> \sqrt{2}
    s = re.sub(r'\\sqrt\s+(\d|\w)(?![a-zA-Z0-9])', r'\\sqrt{\1}', s)
    # Normalize spacing commands
    s = s.replace('\\ ', ' ')
    s = s.replace('\\,', ' ')
    s = s.replace('\\;', ' ')
    s = s.replace('\\:', ' ')
    s = s.replace('\\!', '')
    s = s.replace('\\quad', ' ')
    s = s.replace('\\qquad', ' ')
    # Remove display math delimiters
    s = re.sub(r'\\\[|\\\]|\\\(|\\\)', '', s)
    # Normalize trig function parentheses: \sin^2(x) -> \sin^2 x
    s = re.sub(r'\\(sin|cos|tan|cot|sec|csc|log|ln)\^(\{[^{}]+\}|\d+)\(([^)]+)\)', r'\\\1^\2 \3', s)
    # Normalize \left\lceil and \right\rceil to plain \lceil \rceil
    s = s.replace('\\left\\lceil', '\\lceil').replace('\\right\\rceil', '\\rceil')
    s = s.replace('\\left\\lfloor', '\\lfloor').replace('\\right\\rfloor', '\\rfloor')
    return s


def normalize_for_comparison(s: str) -> str:
    """Aggressively normalize for string comparison."""
    if not s:
        return ""
    s = normalize_latex_for_parsing(s)
    # Convert \frac{a}{b} to a/b for comparison
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    # Remove all spaces
    s = re.sub(r'\s+', '', s)
    # Normalize degree symbol
    s = s.replace('^\\circ', '°').replace('^{\\circ}', '°').replace('\\circ', '°')
    # Remove parentheses around single arguments
    s = re.sub(r'\((\d+°?)\)', r'\1', s)
    # Simplify (1)/(x) to 1/x, 1/(x) to 1/x
    s = re.sub(r'\((\d+)\)/\(([^()]+)\)', r'\1/\2', s)
    s = re.sub(r'\(([^()]+)\)/\((\d+)\)', r'\1/\2', s)
    s = re.sub(r'(\d+)/\(([^()]+)\)', r'\1/\2', s)
    s = re.sub(r'\(([^()]+)\)/(\d+)', r'\1/\2', s)
    # Normalize operators
    s = s.replace('\\cdot', '*').replace('\\times', '*')
    # Lowercase
    s = s.lower()
    return s


def normalize_text(s: str) -> str:
    """Normalize text for comparison (handles \\text{}, whitespace, case)."""
    if not s:
        return ""
    s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)  # Remove LaTeX commands
    s = re.sub(r'[{}\[\]$]', '', s)     # Remove braces
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s


def check_answer(generated: str, ground_truth: str) -> bool:
    """
    Check if generated answer matches ground truth using math_verify.
    
    Args:
        generated: Full generated text (should contain \\boxed{...})
        ground_truth: The correct answer string
        
    Returns:
        True if answers match, False otherwise
    """
    # Extract boxed answer from generated text
    gen_ans = extract_boxed_answer(generated)
    if gen_ans is None:
        return False
    
    # Normalize LaTeX before parsing
    gt_normalized = normalize_latex_for_parsing(ground_truth)
    gen_normalized = normalize_latex_for_parsing(gen_ans)
    
    # FIRST: Check for tuple lists - these need special handling
    # because math_verify incorrectly extracts single numbers from tuple lists
    gt_tuples = extract_tuples(ground_truth)
    gen_tuples = extract_tuples(gen_ans)
    if gt_tuples and len(gt_tuples) >= 2:
        # Ground truth is a tuple list - use tuple comparison
        if gen_tuples and gt_tuples == gen_tuples:
            return True
        # If tuples exist but don't match, don't fall through to math_verify
        # (which would incorrectly match based on individual numbers)
        if gen_tuples:
            return False
    
    # Use latex, expr, and string extraction for comprehensive parsing
    config = [
        LatexExtractionConfig(boxed_match_priority=100),
        ExprExtractionConfig(),
        StringExtractionConfig()
    ]
    
    try:
        # Try parsing with normalized strings
        gt_parsed = parse(gt_normalized, extraction_config=config)
        gen_parsed = parse(gen_normalized, extraction_config=config)
        
        # If both parsed successfully, use math_verify
        if gt_parsed and gen_parsed:
            result = verify(gt_parsed[0], gen_parsed[0], strict=False)
            if result:
                return True
            
            # Try sympy simplification for exponential equivalence (e.g., 4^{2006} = 2^{4012})
            try:
                from sympy import simplify, Eq
                diff = simplify(gt_parsed[0] - gen_parsed[0])
                if diff == 0:
                    return True
                # Also try checking equality directly
                if simplify(Eq(gt_parsed[0], gen_parsed[0])) == True:
                    return True
            except:
                pass
        
        # If only one parsed, try numeric comparison
        if gt_parsed and not gen_parsed:
            try:
                gen_val = float(gen_ans.replace(',', ''))
                from sympy import N
                gt_val = float(N(gt_parsed[0]))
                if abs(gt_val - gen_val) < 1e-6:
                    return True
                # Check relative error for large numbers
                if gt_val != 0 and abs((gt_val - gen_val) / gt_val) < 1e-9:
                    return True
            except:
                pass
        
        if gen_parsed and not gt_parsed:
            try:
                gt_val = float(ground_truth.replace(',', ''))
                from sympy import N
                gen_val = float(N(gen_parsed[0]))
                if abs(gt_val - gen_val) < 1e-6:
                    return True
                if gen_val != 0 and abs((gt_val - gen_val) / gen_val) < 1e-9:
                    return True
            except:
                pass
        
        # Try both numeric if both failed to parse
        if not gt_parsed and not gen_parsed:
            try:
                gt_val = float(ground_truth.replace(',', ''))
                gen_val = float(gen_ans.replace(',', ''))
                if abs(gt_val - gen_val) < 1e-6:
                    return True
            except:
                pass
        
        # Fallback for text answers that don't parse (e.g., "Yes", "No")
        gt_clean = normalize_text(ground_truth)
        gen_clean = normalize_text(gen_ans)
        
        if gt_clean and gen_clean and gt_clean == gen_clean:
            return True
            
    except Exception:
        pass
    
    # Fallback: tuple set comparison (order-independent)
    gt_tuples = extract_tuples(ground_truth)
    gen_tuples = extract_tuples(gen_ans)
    if gt_tuples and gen_tuples and gt_tuples == gen_tuples:
        return True
    
    # Fallback: aggressive normalization (removes all spaces)
    gt_aggr = normalize_for_comparison(ground_truth)
    gen_aggr = normalize_for_comparison(gen_ans)
    if gt_aggr and gen_aggr and gt_aggr == gen_aggr:
        return True
    
    # DISABLED: latex2sympy2_extended fallback is too slow
    # The math_verify library already handles most cases
    
    # Final fallback: normalized string comparison
    return normalize_text(ground_truth) == normalize_text(gen_ans)


def extract_tuples(text: str) -> set | None:
    """Extract all tuples from text as a set (order-independent comparison)."""
    if not text:
        return None
    # Find all (a, b, ...) patterns
    tuples = re.findall(r'\(([^()]+)\)', text)
    if not tuples:
        return None
    result = set()
    for t in tuples:
        # Split by comma, normalize each element
        parts = tuple([p.strip().replace(' ', '').replace('\\', '').lower() for p in t.split(',')])
        if len(parts) >= 2:
            result.add(parts)
    return result if result else None


if __name__ == "__main__":
    # Quick test
    test_cases = [
        ('2(p!)^2', '\\boxed{(p!)^2}', False),
        ('1', '\\boxed{1}', True),
        ('\\text{No}', '\\boxed{\\text{No}}', True),
        ('\\frac{1}{2}', '\\boxed{0.5}', True),
        ('37/72', '\\boxed{\\dfrac{37}{72}}', True),  # dfrac test
        ('1/3', '\\boxed{\\dfrac{1}{3}}', True),  # dfrac test
        ('82 \\mathrm{~m}', '\\boxed{82}', True),  # unit test
        ('(2, 1), (3, 1), (1, 2), (1, 3)', '\\boxed{(1, 2), (1, 3), (2, 1), (3, 1)}', True),  # tuple order
        ('4-2\\sqrt{3}', '\\boxed{4 - 2\\sqrt{3}}', True),  # spacing
        ('\\frac{16 \\sqrt{2}}{9}', '\\boxed{\\dfrac{16\\sqrt{2}}{9}}', True),  # spacing in frac
        ('5n+1', '\\boxed{5n + 1}', True),  # spacing
        ('\\frac{4}{\\sqrt{3}} \\sin^2 80^\\circ', '\\boxed{\\frac{4}{\\sqrt{3}} \\sin^2(80^\\circ)}', True),  # trig parens
    ]
    
    print("Testing math_checker:")
    for gt, gen, expected in test_cases:
        result = check_answer(gen, gt)
        status = '✓' if result == expected else '✗'
        print(f'{status} GT={gt[:40]}, Expected={expected}, Got={result}')
