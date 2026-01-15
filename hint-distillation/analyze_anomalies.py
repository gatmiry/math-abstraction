#!/usr/bin/env python3
"""Analyze anomalies in distillation_pairs_round1.json"""

import json
import re

with open("distillation_data/distillation_pairs_round1.json") as f:
    data = json.load(f)

print("=== DEGENERATE OUTPUT ANALYSIS ===\n")

# Count different types of degenerate endings
patterns = {
    "repeated_backslash_space": 0,
    "repeated_paren": 0,
    "repeated_boxed": 0,
    "enumeration_spam": 0,
    "normal": 0
}

degenerate_examples = []

for result in data["results"]:
    for rollout in result["rollouts"]:
        response = rollout["response"]
        ending = response[-500:]
        
        is_degenerate = False
        pattern_type = None
        
        # Check for repeated backslash space pattern
        if re.search(r'(\\ ){10,}', ending):
            patterns["repeated_backslash_space"] += 1
            is_degenerate = True
            pattern_type = "backslash_space"
        # Check for repeated paren pattern
        elif re.search(r'(\\?\) ){10,}', ending):
            patterns["repeated_paren"] += 1
            is_degenerate = True
            pattern_type = "paren"
        # Check for repeated boxed
        elif response.count("boxed") > 10:
            patterns["repeated_boxed"] += 1
            is_degenerate = True
            pattern_type = "repeated_boxed"
        # Check for enumeration spam
        elif " or " in ending and ending.count(" or ") > 10:
            patterns["enumeration_spam"] += 1
            is_degenerate = True
            pattern_type = "enumeration"
        else:
            patterns["normal"] += 1
            
        if is_degenerate and len(degenerate_examples) < 5:
            degenerate_examples.append({
                "problem": result["test_idx"],
                "hint": rollout["hint_rank"],
                "pattern": pattern_type,
                "ending": ending[-150:].replace('\n', ' ')
            })

print("Pattern counts:")
total = sum(patterns.values())
for k, v in patterns.items():
    pct = v / total * 100
    print(f"  {k}: {v} ({pct:.1f}%)")

degenerate_count = total - patterns["normal"]
print(f"\n=== DEGENERATE RATE: {degenerate_count}/{total} = {degenerate_count / total * 100:.1f}% ===")

print("\n=== DEGENERATE EXAMPLES ===")
for ex in degenerate_examples:
    print(f"\nProblem {ex['problem']}, Hint {ex['hint']} ({ex['pattern']}):")
    print(f"  ...{ex['ending']}")

# Check if the degenerate outputs are from training data leakage
print("\n=== CHECKING TRAINING DATA INFLUENCE ===")
# Look at what was in the training data
try:
    with open("distillation_data/distillation_pairs.json") as f:
        train_data = json.load(f)
    
    # Check endings of training targets
    train_endings = []
    for item in train_data[:10]:
        target = item.get("target_response", "")
        if target:
            train_endings.append(target[-100:].replace('\n', ' '))
    
    print("Sample training target endings:")
    for i, ending in enumerate(train_endings[:3]):
        print(f"  {i}: ...{ending}")
except Exception as e:
    print(f"Could not load training data: {e}")

