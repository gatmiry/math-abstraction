"""
Analyze Wikipedia URLs in math_solutions.jsonl to find similar URLs across solutions.
"""

import json
import re
from collections import defaultdict
from itertools import combinations
from urllib.parse import urlparse


def extract_wikipedia_urls(solution_text: str) -> list:
    """Extract all Wikipedia URLs from a solution text."""
    # Pattern to match Wikipedia URLs in LaTeX format
    # Looking for \url{https://en.wikipedia.org/...} or similar patterns
    patterns = [
        r'\\url\{([^}]+)\}',  # LaTeX \url{...}
        r'https://en\.wikipedia\.org/[^\s\)\}]+',  # Direct URL
        r'wikipedia_url:\s*([^\n]+)',  # If in structured format
    ]
    
    urls = []
    for pattern in patterns:
        matches = re.findall(pattern, solution_text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            if match and 'wikipedia.org' in match:
                urls.append(match.strip())
    
    # Also check for lemmatheorem blocks with Wikipedia URL field
    lemmatheorem_blocks = re.findall(
        r'\\begin\{lemmatheorem\}.*?\\textbf\{Wikipedia URL:\}.*?\\url\{([^}]+)\}.*?\\end\{lemmatheorem\}',
        solution_text,
        re.DOTALL
    )
    urls.extend(lemmatheorem_blocks)
    
    # Extract from text format like "wikipedia_url: https://..."
    text_urls = re.findall(r'wikipedia_url[:\s]+(https?://[^\s\n]+)', solution_text, re.IGNORECASE)
    urls.extend(text_urls)
    
    # Normalize URLs (remove trailing slashes, fragments, etc.)
    normalized_urls = []
    for url in urls:
        try:
            parsed = urlparse(url)
            # Keep scheme, netloc, and path, but normalize
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
            if normalized not in normalized_urls:
                normalized_urls.append(normalized)
        except:
            # If parsing fails, use as-is
            if url not in normalized_urls:
                normalized_urls.append(url)
    
    return normalized_urls


def normalize_url_for_comparison(url: str) -> str:
    """Normalize URL for similarity comparison."""
    try:
        parsed = urlparse(url)
        # Extract the main path without fragments or query params
        path = parsed.path.rstrip('/')
        # Remove common variations
        path = re.sub(r'#.*$', '', path)  # Remove fragments
        return f"{parsed.netloc}{path}"
    except:
        return url


def are_urls_similar(url1: str, url2: str) -> bool:
    """Check if two URLs are similar (same page, possibly different sections)."""
    norm1 = normalize_url_for_comparison(url1)
    norm2 = normalize_url_for_comparison(url2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return True
    
    # Check if they're the same Wikipedia page (different sections)
    # e.g., https://en.wikipedia.org/wiki/Triangle and https://en.wikipedia.org/wiki/Triangle#Area
    if 'wikipedia.org' in norm1 and 'wikipedia.org' in norm2:
        # Extract the main article name
        match1 = re.search(r'/wiki/([^/#]+)', norm1)
        match2 = re.search(r'/wiki/([^/#]+)', norm2)
        if match1 and match2:
            return match1.group(1) == match2.group(1)
    
    return False


def main():
    input_file = "outputs/math_solutions.jsonl"
    
    print(f"Reading solutions from {input_file}...")
    solutions = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                solutions.append({
                    'index': line_num,
                    'problem': data.get('problem', '')[:50] + '...' if len(data.get('problem', '')) > 50 else data.get('problem', ''),
                    'solution': data.get('solution', ''),
                    'urls': []
                })
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Found {len(solutions)} solutions")
    
    # Extract URLs from each solution
    print("\nExtracting Wikipedia URLs...")
    for sol in solutions:
        urls = extract_wikipedia_urls(sol['solution'])
        sol['urls'] = urls
        if urls:
            print(f"  Solution {sol['index']}: {len(urls)} URL(s)")
    
    # Count URLs per solution
    url_counts = [len(sol['urls']) for sol in solutions]
    print(f"\nURL statistics:")
    print(f"  Solutions with URLs: {sum(1 for c in url_counts if c > 0)}")
    print(f"  Solutions without URLs: {sum(1 for c in url_counts if c == 0)}")
    print(f"  Total URLs found: {sum(url_counts)}")
    
    # Find pairs with similar URLs
    print("\nFinding pairs with similar Wikipedia URLs...")
    similar_pairs = []
    
    for i, sol1 in enumerate(solutions):
        for j, sol2 in enumerate(solutions[i+1:], start=i+1):
            # Check if any URLs from sol1 are similar to any URLs from sol2
            for url1 in sol1['urls']:
                for url2 in sol2['urls']:
                    if are_urls_similar(url1, url2):
                        similar_pairs.append({
                            'pair': (i+1, j+1),
                            'solution1_index': i+1,
                            'solution2_index': j+1,
                            'url1': url1,
                            'url2': url2,
                            'normalized': normalize_url_for_comparison(url1)
                        })
                        break  # Only count each pair once
                else:
                    continue
                break
    
    # Remove duplicates (same pair with different URL matches)
    unique_pairs = {}
    for pair_info in similar_pairs:
        pair_key = tuple(sorted([pair_info['solution1_index'], pair_info['solution2_index']]))
        if pair_key not in unique_pairs:
            unique_pairs[pair_key] = pair_info
    
    similar_pairs = list(unique_pairs.values())
    
    print(f"\nResults:")
    print(f"  Total pairs of solutions: {len(solutions) * (len(solutions) - 1) // 2}")
    print(f"  Pairs with similar Wikipedia URLs: {len(similar_pairs)}")
    
    if similar_pairs:
        print(f"\nDetails of similar pairs:")
        for pair_info in similar_pairs:
            print(f"\n  Pair ({pair_info['solution1_index']}, {pair_info['solution2_index']}):")
            print(f"    URL 1: {pair_info['url1']}")
            print(f"    URL 2: {pair_info['url2']}")
            print(f"    Normalized: {pair_info['normalized']}")
    
    # Group by normalized URL to see which URLs are most common
    url_groups = defaultdict(list)
    for sol in solutions:
        for url in sol['urls']:
            normalized = normalize_url_for_comparison(url)
            url_groups[normalized].append(sol['index'])
    
    print(f"\n\nMost common Wikipedia URLs (appearing in multiple solutions):")
    common_urls = {url: indices for url, indices in url_groups.items() if len(indices) > 1}
    if common_urls:
        sorted_common = sorted(common_urls.items(), key=lambda x: len(x[1]), reverse=True)
        for url, indices in sorted_common[:10]:  # Top 10
            print(f"  {url}: appears in {len(indices)} solutions (indices: {indices})")
    else:
        print("  No URLs appear in multiple solutions")
    
    return len(similar_pairs)


if __name__ == "__main__":
    num_similar = main()
    print(f"\n{'='*60}")
    print(f"SUMMARY: {num_similar} pairs of solutions have similar Wikipedia URLs")


