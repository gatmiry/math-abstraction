#!/usr/bin/env python3
"""
Continuous Hint Distillation with Dense Grouped Rewards

This module extends verl to support:
1. Grouped prompts - multiple hints per math problem processed together
2. Dense (token-level) rewards - per-token reward signals
3. Collective group evaluation - reward computed based on all hints in a group

MINIMAL CHANGES APPROACH:
- Uses verl's existing fast generation (vLLM) and training infrastructure
- Only overrides the reward function and dataset preparation
- Hooks into verl's token_level_scores which already supports per-token rewards

Key insight: verl already has token_level_scores (shape: [batch, response_length])
We just need to:
1. Group hints by problem in the dataset
2. Compute dense rewards with group awareness in the reward function
"""

import os
import re
import sys
import json
import datetime
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Iterator
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from transformers import AutoTokenizer
from datasets import load_from_disk
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# ============================================================================
# Configuration
# ============================================================================

# Model and dataset paths
MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
DATASET_PATH = "../newopenaioutputs/hints_dataset"
OUTPUT_DIR = "../../models/dense_hint_distillation"

# Training configuration
NUM_NODES = 1
GPUS_PER_NODE = 8
MAX_HINTS_PER_GROUP = 10
VAL_SIZE = 64

# Dense reward configuration
DENSE_REWARD_TYPE = "distributed"  # "distributed", "last_token", "weighted"
GROUP_CONSISTENCY_WEIGHT = 0.2
GROUP_DIVERSITY_WEIGHT = 0.1

# Special group ID for padding samples (these get zero reward)
PADDING_GROUP_ID = "__PADDING__"

# System prompt
SYSTEM_PROMPT = """You are learning to solve mathematics problems. You will be given a math problem, a partial proof, and a hint. Your task is to carefully complete the proof or solution, step by step, providing clear reasoning at each stage.

- Show step-by-step reasoning before the conclusion.
- Place final answer boxed (in \\boxed{...}) at the end."""


# ============================================================================
# Grouped Batch Sampler (ensures all hints from same group are in same batch)
# ============================================================================

class GroupedBatchSampler(Sampler):
    """
    Sampler that ensures all samples from the same group are in the same batch,
    while yielding FIXED-SIZE batches (required by verl).
    
    Strategy:
    1. Group samples by group_id (excluding PADDING_GROUP_ID)
    2. Pack complete groups into batches
    3. Pad incomplete batches using the padding sample index
    4. If a group is larger than batch_size, split it (with warning)
    
    Padding samples have group_id=PADDING_GROUP_ID and get zero reward.
    """
    
    def __init__(
        self,
        group_ids: List[str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        
        # Find padding sample index (has PADDING_GROUP_ID)
        self.padding_idx = None
        for idx, gid in enumerate(group_ids):
            if gid == PADDING_GROUP_ID:
                self.padding_idx = idx
                break
        
        if self.padding_idx is None:
            raise ValueError(f"No sample with group_id={PADDING_GROUP_ID} found. Add a padding sample to dataset.")
        
        # Build index of samples per group (excluding padding)
        self.group_to_indices = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            if gid != PADDING_GROUP_ID:
                self.group_to_indices[gid].append(idx)
        
        self.group_list = list(self.group_to_indices.keys())
        
        # Check for oversized groups
        oversized = [gid for gid, indices in self.group_to_indices.items() 
                     if len(indices) > batch_size]
        if oversized:
            print(f"[WARNING] {len(oversized)} groups larger than batch_size={batch_size}")
        
        print(f"[GroupedBatchSampler] {len(self.group_list)} groups, "
              f"{len(group_ids)-1} real samples, padding_idx={self.padding_idx}")
    
    def __iter__(self) -> Iterator[List[int]]:
        group_order = self.group_list.copy()
        if self.shuffle:
            self.rng.shuffle(group_order)
        
        batches = []
        current_batch = []
        
        for gid in group_order:
            group_indices = self.group_to_indices[gid].copy()
            if self.shuffle:
                self.rng.shuffle(group_indices)
            
            # Handle oversized groups: split them
            if len(group_indices) > self.batch_size:
                if current_batch:
                    batches.append(self._pad_batch(current_batch))
                    current_batch = []
                
                for i in range(0, len(group_indices), self.batch_size):
                    chunk = group_indices[i:i + self.batch_size]
                    if len(chunk) == self.batch_size:
                        batches.append(chunk)
                    elif not self.drop_last:
                        batches.append(self._pad_batch(chunk))
                continue
            
            # If adding this group would exceed batch_size, finalize current batch
            if len(current_batch) + len(group_indices) > self.batch_size:
                if current_batch:
                    batches.append(self._pad_batch(current_batch))
                    current_batch = []
            
            current_batch.extend(group_indices)
        
        # Handle remaining batch
        if current_batch:
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
            elif not self.drop_last:
                batches.append(self._pad_batch(current_batch))
        
        for batch in batches:
            yield batch
    
    def _pad_batch(self, batch: List[int]) -> List[int]:
        """Pad batch to exact batch_size using the padding sample."""
        if len(batch) >= self.batch_size:
            return batch[:self.batch_size]
        
        needed = self.batch_size - len(batch)
        padding = [self.padding_idx] * needed
        return batch + padding
    
    def __len__(self) -> int:
        # Compute exact number of batches
        total_samples = len(self.group_ids)
        if self.drop_last:
            return total_samples // self.batch_size
        return (total_samples + self.batch_size - 1) // self.batch_size


def extract_group_ids_from_dataset(data: List[Dict]) -> List[str]:
    """Extract group_id from each sample in the dataset."""
    group_ids = []
    for item in data:
        reward_model = item.get("reward_model", {})
        if isinstance(reward_model, dict):
            gid = reward_model.get("group_id", f"sample_{len(group_ids)}")
        else:
            gid = f"sample_{len(group_ids)}"
        group_ids.append(gid)
    return group_ids


# ============================================================================
# Helper Functions
# ============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...}."""
    matches = list(re.finditer(r'\\boxed\{', text))
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


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    return answer.strip().lower()


# ============================================================================
# Grouped Dataset Creation
# ============================================================================

def format_prompt(problem: str, partial_proof: str, hint: str) -> List[Dict[str, str]]:
    """Format a single hint prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}\n\nPartial proof: {partial_proof}\n\nHint: {hint}"}
    ]


def create_grouped_rl_dataset(
    tokenizer,
    dataset_path: str,
    max_samples: Optional[int] = None,
    val_size: int = 64,
    max_prompt_tokens: int = 2560,
    max_hints_per_group: int = 10,
):
    """
    Create RL dataset with hints grouped by problem.
    
    Each sample includes a group_id to enable group-aware reward computation.
    """
    dataset = load_from_disk(dataset_path)
    
    # Group by problem
    problem_groups = defaultdict(list)
    
    for idx, item in enumerate(dataset):
        problem = item.get('problem', '')
        partial_proof = item.get('partial_proof', '')
        hints = item.get('hints', [])
        ground_truth = item.get('final_answer', '') or item.get('answer', '')
        
        if not hints or not ground_truth:
            continue
        
        # Use problem hash as group key
        group_key = hash(problem + partial_proof)
        problem_groups[group_key].append({
            'problem': problem,
            'partial_proof': partial_proof,
            'hints': hints,
            'ground_truth': ground_truth,
        })
    
    # Flatten into samples with group metadata
    all_samples = []
    
    for group_idx, (group_key, items) in enumerate(problem_groups.items()):
        # Merge hints from items with same problem
        item = items[0]
        all_hints = []
        for i in items:
            all_hints.extend(i['hints'])
        
        # Limit hints per group
        all_hints = all_hints[:max_hints_per_group]
        
        # Create a sample for each hint in the group
        for hint_idx, hint in enumerate(all_hints):
            prompt = format_prompt(item['problem'], item['partial_proof'], hint)
            
            # Check prompt length
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            token_count = len(tokenizer.encode(prompt_text))
            if token_count > max_prompt_tokens:
                continue
            
            all_samples.append({
                "prompt": prompt,
                "reward_model": {
                    "ground_truth": item['ground_truth'],
                    "group_id": f"group_{group_idx}",
                    "hint_idx": hint_idx,
                    "total_hints_in_group": len(all_hints),
                    "problem": item['problem'],
                },
                "data_source": "hints_grouped",
            })
    
    # Limit samples if specified
    if max_samples is not None and max_samples > 0:
        all_samples = all_samples[-max_samples:]
    
    # Split train/val
    total = len(all_samples)
    val_size_actual = min(val_size, total)
    train_samples = all_samples[:-val_size_actual] if val_size_actual > 0 else all_samples
    val_samples = all_samples[-val_size_actual:] if val_size_actual > 0 else []
    
    # Add a padding sample at the end (used by GroupedBatchSampler for padding)
    # This sample has PADDING_GROUP_ID and will get zero reward
    padding_sample = {
        "prompt": [{"role": "user", "content": "padding"}],
        "reward_model": {
            "ground_truth": "",
            "group_id": PADDING_GROUP_ID,
            "hint_idx": -1,
            "total_hints_in_group": 0,
            "problem": "",
        },
        "data_source": "padding",
    }
    train_samples.append(padding_sample)
    val_samples.append(padding_sample)
    
    print(f"[INFO] Created {len(train_samples)-1} train samples + 1 padding, {len(val_samples)-1} val samples + 1 padding")
    print(f"[INFO] {len(problem_groups)} unique problems, up to {max_hints_per_group} hints each")
    
    return train_samples, val_samples


# ============================================================================
# Dense Group Reward Function (verl-compatible)
# ============================================================================

# Global storage for group-level reward computation
_group_responses = defaultdict(list)
_group_scores = defaultdict(list)


def compute_dense_group_score(
    data_source: str = None,
    solution_str: str = None,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """
    Compute reward score with group awareness.
    
    This function is called by verl's NaiveRewardManager for each sample.
    It returns a SPARSE score (float), but verl will distribute it to tokens
    via token_level_scores.
    
    For true dense rewards, we need to override the reward manager (see below).
    
    Args:
        data_source: Data source identifier
        solution_str: Generated solution string
        ground_truth: Ground truth answer (nested dict from dataset)
        extra_info: Extra info dictionary
    
    Returns:
        float: Reward score
    """
    # ground_truth might be the nested dict we stored
    if isinstance(ground_truth, dict):
        actual_answer = ground_truth.get('ground_truth', '')
        group_id = ground_truth.get('group_id', 'unknown')
        hint_idx = ground_truth.get('hint_idx', 0)
        total_hints = ground_truth.get('total_hints_in_group', 1)
    else:
        actual_answer = ground_truth
        group_id = 'unknown'
        hint_idx = 0
        total_hints = 1
    
    # Compute base correctness score
    boxed = extract_boxed_answer(solution_str)
    boxed_norm = normalize_answer(boxed)
    answer_norm = normalize_answer(actual_answer)
    
    base_score = 1.0 if (boxed_norm and answer_norm and boxed_norm == answer_norm) else 0.0
    
    # Store for group-level computation
    _group_responses[group_id].append(solution_str)
    _group_scores[group_id].append(base_score)
    
    # Compute group bonus once we have all responses from the group
    group_bonus = 0.0
    if len(_group_scores[group_id]) >= total_hints:
        scores = _group_scores[group_id]
        responses = _group_responses[group_id]
        
        # Consistency bonus: reward if multiple hints lead to same answer
        answers = [extract_boxed_answer(r) for r in responses]
        answers = [normalize_answer(a) for a in answers if a]
        
        if len(answers) >= 2:
            answer_counts = defaultdict(int)
            for a in answers:
                answer_counts[a] += 1
            max_count = max(answer_counts.values())
            consistency_bonus = (max_count / len(answers)) * GROUP_CONSISTENCY_WEIGHT
            group_bonus += consistency_bonus
        
        # Diversity bonus: based on response length variance
        lengths = [len(r) for r in responses]
        if len(lengths) >= 2 and np.mean(lengths) > 0:
            length_std = np.std(lengths) / (np.mean(lengths) + 1)
            diversity_bonus = min(length_std * GROUP_DIVERSITY_WEIGHT, GROUP_DIVERSITY_WEIGHT)
            group_bonus += diversity_bonus
        
        # Clear group storage after processing
        del _group_responses[group_id]
        del _group_scores[group_id]
    
    return base_score + group_bonus


# ============================================================================
# Custom Reward Manager for True Dense Rewards
# ============================================================================

def register_dense_grpo_advantage():
    """
    Register a custom advantage estimator that passes through dense rewards directly.
    
    No normalization, no baseline subtraction - just use the raw per-token rewards.
    Loss will be: loss_t = -reward_t * log_prob_t
    """
    from verl.trainer.ppo.core_algos import register_adv_est
    import torch
    import numpy as np
    
    @register_adv_est("dense_grpo")
    def compute_dense_grpo_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray = None,
        **kwargs,
    ):
        """
        Pass through dense rewards directly as advantages.
        
        No advantage calculation - just use raw per-token rewards.
        
        Args:
            token_level_rewards: shape (bs, response_length) - per-token rewards
            response_mask: shape (bs, response_length)
        
        Returns:
            advantages: same as token_level_rewards (masked)
            returns: same as advantages
        """
        # Just pass through the rewards directly, masked
        advantages = token_level_rewards * response_mask
        return advantages, advantages
    
    print("[INFO] Registered 'dense_grpo' advantage estimator (raw rewards, no normalization)")


def create_dense_reward_manager():
    """
    Create a custom reward manager that returns dense (per-token) rewards.
    
    This hooks into verl's reward manager system to provide token_level_scores
    instead of just a scalar reward.
    """
    from verl.workers.reward_manager import register
    from verl.workers.reward_manager.naive import NaiveRewardManager
    from verl import DataProto
    
    @register("dense_group")
    class DenseGroupRewardManager(NaiveRewardManager):
        """
        Reward manager that computes TRUE dense (per-token) rewards with group awareness.
        
        Each token gets its own reward value, and gradients are computed as:
            loss_t = -advantage_t * log_prob_t
        
        This allows different tokens to have different learning signals.
        """
        
        def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="reward_model", **kwargs):
            super().__init__(tokenizer, num_examine, compute_score, reward_fn_key, **kwargs)
            self.dense_reward_type = kwargs.get("dense_reward_type", DENSE_REWARD_TYPE)
        
        def __call__(self, data: DataProto, return_dict: bool = False):
            """
            Compute TRUE per-token dense rewards for all samples.
            
            Samples with group_id=PADDING_GROUP_ID get zero reward.
            """
            responses = data.batch["responses"]
            batch_size, response_length = responses.shape
            
            response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            reward_infos = data.non_tensor_batch.get(self.reward_fn_key, [{}] * batch_size)
            
            # Group responses by group_id
            group_data = defaultdict(list)
            num_padding = 0
            
            for i, (response_text, reward_info) in enumerate(zip(response_texts, reward_infos)):
                if isinstance(reward_info, dict):
                    group_id = reward_info.get('group_id', f'sample_{i}')
                    ground_truth = reward_info.get('ground_truth', '')
                    group_subindex = reward_info.get('hint_idx', 0)  # Hint index within group
                    total_hints = reward_info.get('total_hints_in_group', 1)
                else:
                    group_id = f'sample_{i}'
                    ground_truth = str(reward_info) if reward_info else ''
                    group_subindex = 0
                    total_hints = 1
                
                if group_id == PADDING_GROUP_ID:
                    num_padding += 1
                
                response_mask = data.batch["attention_mask"][i, -response_length:]
                actual_length = (response_mask > 0).sum().item()
                
                # Pre-compute sparse correctness score
                boxed = extract_boxed_answer(response_text)
                boxed_norm = normalize_answer(boxed)
                answer_norm = normalize_answer(ground_truth)
                is_correct = bool(boxed_norm and answer_norm and boxed_norm == answer_norm)
                sparse_score = 1.0 if is_correct else 0.0
                
                group_data[group_id].append({
                    'idx': i,
                    'response_text': response_text,
                    'response_tokens': responses[i],
                    'ground_truth': ground_truth,
                    'response_length': actual_length,
                    'response_mask': response_mask,
                    'is_correct': is_correct,
                    'sparse_score': sparse_score,
                    'extracted_answer': boxed_norm,
                    'group_subindex': group_subindex,  # Hint index within group
                    'total_hints': total_hints,        # Total hints in this group
                })
            
            if num_padding > 0:
                print(f"[DenseGroupRewardManager] {num_padding}/{batch_size} padding samples (zero reward)")
            
            # Compute rewards
            token_level_scores = torch.zeros(batch_size, response_length, dtype=torch.float32)
            reward_extra_info = defaultdict(list)
            
            for group_id, group_items in group_data.items():
                # Padding group gets zero reward
                if group_id == PADDING_GROUP_ID:
                    for item in group_items:
                        reward_extra_info['is_padding'].append(True)
                    continue
                
                # Compute dense rewards for real groups
                dense_rewards = self.compute_group_dense_rewards(
                    group_items=group_items,
                    tokenizer=self.tokenizer,
                    response_length=response_length,
                )
                
                for item, dense_reward in zip(group_items, dense_rewards):
                    i = item['idx']
                    token_level_scores[i] = dense_reward
                    reward_extra_info['mean_token_reward'].append(dense_reward[:item['response_length']].mean().item())
                    reward_extra_info['is_padding'].append(False)
            
            if return_dict:
                return {
                    "reward_tensor": token_level_scores,
                    "reward_extra_info": dict(reward_extra_info),
                }
            return token_level_scores
        
        def compute_group_dense_rewards(
            self,
            group_items: List[Dict],
            tokenizer,
            response_length: int,
        ) -> List[torch.Tensor]:
            """
            Compute TRUE per-token dense rewards for all responses in a group.
            
            THIS IS THE FUNCTION TO CUSTOMIZE for your specific reward logic.
            
            Args:
                group_items: List of dicts, each containing:
                    - 'idx': batch index
                    - 'response_text': decoded response string
                    - 'response_tokens': tensor of token IDs [response_length]
                    - 'ground_truth': expected answer
                    - 'response_length': actual length (non-padding)
                    - 'response_mask': attention mask for response
                tokenizer: tokenizer for decoding tokens
                response_length: max response length (for padding)
            
            Returns:
                List of tensors, each [response_length] with per-token rewards
            """
            dense_rewards = []
            
            # First, compute group-level info that informs per-token rewards
            group_info = self._analyze_group(group_items, tokenizer)
            
            for item in group_items:
                # Compute per-token rewards for this response
                token_rewards = self._compute_token_level_rewards(
                    item=item,
                    group_info=group_info,
                    tokenizer=tokenizer,
                    max_length=response_length,
                )
                dense_rewards.append(token_rewards)
            
            return dense_rewards
        
        def _analyze_group(self, group_items: List[Dict], tokenizer) -> Dict:
            """
            Analyze the group to extract information for per-token reward computation.
            
            Returns a dict with group-level statistics that inform token rewards.
            """
            # Compute correctness for each response
            correct_responses = []
            incorrect_responses = []
            all_answers = []
            
            for item in group_items:
                boxed = extract_boxed_answer(item['response_text'])
                boxed_norm = normalize_answer(boxed)
                answer_norm = normalize_answer(item['ground_truth'])
                is_correct = boxed_norm and answer_norm and boxed_norm == answer_norm
                
                if is_correct:
                    correct_responses.append(item)
                else:
                    incorrect_responses.append(item)
                
                if boxed_norm:
                    all_answers.append(boxed_norm)
            
            # Find most common answer (consensus)
            answer_counts = defaultdict(int)
            for a in all_answers:
                answer_counts[a] += 1
            consensus_answer = max(answer_counts.keys(), key=lambda x: answer_counts[x]) if answer_counts else None
            
            # Extract key tokens from correct responses (tokens that appear frequently in correct solutions)
            key_tokens = set()
            if correct_responses:
                token_freq = defaultdict(int)
                for item in correct_responses:
                    tokens = item['response_tokens'].tolist()
                    for t in set(tokens):  # unique tokens per response
                        token_freq[t] += 1
                
                # Tokens that appear in >50% of correct responses
                threshold = len(correct_responses) * 0.5
                key_tokens = {t for t, freq in token_freq.items() if freq >= threshold}
            
            return {
                'correct_responses': correct_responses,
                'incorrect_responses': incorrect_responses,
                'consensus_answer': consensus_answer,
                'key_tokens': key_tokens,
                'num_correct': len(correct_responses),
                'num_total': len(group_items),
                'accuracy': len(correct_responses) / len(group_items) if group_items else 0,
            }
        
        def _compute_token_level_rewards(
            self,
            item: Dict,
            group_info: Dict,
            tokenizer,
            max_length: int,
        ) -> torch.Tensor:
            """
            Compute TRUE per-token rewards for a single response.
            
            CUSTOMIZE THIS FUNCTION for your specific reward logic.
            
            Current implementation:
            1. Base reward from correctness
            2. Bonus for tokens that appear in correct solutions (key tokens)
            3. Bonus for reasoning markers (mathematical symbols, etc.)
            4. Position-aware scaling (more reward toward answer)
            """
            rewards = torch.zeros(max_length, dtype=torch.float32)
            seq_len = item['response_length']
            
            if seq_len <= 0:
                return rewards
            
            response_tokens = item['response_tokens'][:seq_len]
            response_text = item['response_text']
            
            # 1. Check if this response is correct
            boxed = extract_boxed_answer(response_text)
            boxed_norm = normalize_answer(boxed)
            answer_norm = normalize_answer(item['ground_truth'])
            is_correct = boxed_norm and answer_norm and boxed_norm == answer_norm
            
            # Base reward per token
            base_reward = 1.0 if is_correct else -0.1
            
            # 2. Per-token reward computation
            for t in range(seq_len):
                token_id = response_tokens[t].item()
                token_str = tokenizer.decode([token_id])
                
                # Start with base reward
                token_reward = base_reward
                
                # Bonus: Token appears in key tokens from correct solutions
                if token_id in group_info['key_tokens']:
                    token_reward += 0.2
                
                # Bonus: Mathematical reasoning tokens
                math_symbols = ['=', '+', '-', '*', '/', '\\', '^', '_', '{', '}', '(', ')', '[', ']']
                if any(sym in token_str for sym in math_symbols):
                    token_reward += 0.1
                
                # Bonus: Reasoning keywords
                reasoning_words = ['therefore', 'hence', 'thus', 'since', 'because', 'implies', 'if', 'then']
                token_lower = token_str.lower().strip()
                if any(word in token_lower for word in reasoning_words):
                    token_reward += 0.15
                
                # Position-aware scaling: higher reward toward end (where answer is)
                position_weight = 0.5 + 0.5 * (t / seq_len)  # 0.5 at start, 1.0 at end
                token_reward *= position_weight
                
                # Group consistency bonus: higher if group accuracy is high
                token_reward += group_info['accuracy'] * 0.1
                
                rewards[t] = token_reward
            
            return rewards
    
    return DenseGroupRewardManager


# ============================================================================
# Main Training Script
# ============================================================================

def get_verl_config_path():
    """Get path to verl's config directory."""
    import verl
    verl_path = os.path.dirname(verl.__file__)
    return os.path.join(verl_path, "trainer", "config")


def main():
    """Main function to run GRPO training with dense group rewards."""
    
    # Generate experiment name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"dense_hints_{timestamp}"
    output_dir = os.path.join(OUTPUT_DIR, experiment_name)
    
    print("=" * 80)
    print("Dense Grouped GRPO Training with verl")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {output_dir}")
    print(f"Dense reward type: {DENSE_REWARD_TYPE}")
    print(f"Max hints per group: {MAX_HINTS_PER_GROUP}")
    print("=" * 80)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create grouped dataset
    print("Creating grouped RL dataset...")
    train_data, val_data = create_grouped_rl_dataset(
        tokenizer,
        DATASET_PATH,
        max_samples=None,
        val_size=VAL_SIZE,
        max_hints_per_group=MAX_HINTS_PER_GROUP,
    )
    
    # Save to parquet
    print("Saving datasets to parquet...")
    os.makedirs('/mnt/tmp', exist_ok=True)
    
    df_train = pd.DataFrame(train_data)
    train_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_train.parquet', delete=False, dir='/mnt/tmp').name
    df_train.to_parquet(train_file, index=False)
    
    df_val = pd.DataFrame(val_data)
    val_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_val.parquet', delete=False, dir='/mnt/tmp').name
    df_val.to_parquet(val_file, index=False)
    
    print(f"Train: {train_file}")
    print(f"Val: {val_file}")
    
    # Register custom reward manager
    print("Registering dense group reward manager...")
    create_dense_reward_manager()
    
    # Register custom advantage estimator that preserves per-token rewards
    print("Registering dense_grpo advantage estimator...")
    register_dense_grpo_advantage()
    
    # Setup Hydra config
    print("Loading verl configuration...")
    config_path = get_verl_config_path()
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_path, version_base=None)
    
    # Compose overrides
    overrides = [
        # Model
        f"actor_rollout_ref.model.path={MODEL_PATH}",
        "actor_rollout_ref.model.trust_remote_code=true",
        
        # Rollout (vLLM)
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=4",  # Multiple generations for GRPO
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.85",
        "actor_rollout_ref.rollout.prompt_length=2560",
        "actor_rollout_ref.rollout.response_length=1536",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.enforce_eager=true",
        
        # Ray
        "++ray_kwargs.ray_init.address=auto",
        
        # Actor
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.ppo_epochs=1",
        
        # Algorithm - Dense GRPO (preserves per-token rewards)
        "algorithm.adv_estimator=dense_grpo",  # Custom: keeps per-token structure
        "algorithm.use_kl_in_reward=false",
        "algorithm.norm_adv_by_std_in_grpo=true",
        
        # Data
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.prompt_key=prompt",
        "data.max_prompt_length=2560",
        "data.max_response_length=1536",
        "data.train_batch_size=64",
        "data.val_batch_size=32",
        
        # Trainer
        f"trainer.project_name=dense-hint-distillation",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.nnodes={NUM_NODES}",
        f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.total_epochs=1",
        "trainer.save_freq=500",
        "trainer.val_before_train=false",
        "trainer.test_freq=50",
        "trainer.log_val_generations=3",
        
        # Use our custom reward manager for dense rewards
        # If you want sparse rewards with group bonus, use compute_dense_group_score instead
        f"++custom_reward_function.path={__file__}",
        "++custom_reward_function.name=compute_dense_group_score",
        
        # Or use the dense reward manager:
        # "++reward_model.reward_manager=dense_group",
        # f"++reward_model.dense_reward_type={DENSE_REWARD_TYPE}",
        
        # Disable external reward model
        "++reward_model.enable=false",
        
        # Disable critic (GRPO doesn't need it)
        "++critic.enable=false",
    ]
    
    try:
        config = compose(config_name="ppo_trainer", overrides=overrides)
    except Exception as e:
        print(f"Error composing config: {e}")
        raise
    
    print("Configuration loaded successfully")
    
    # =========================================================================
    # Create custom sampler that keeps groups together
    # =========================================================================
    print("Creating grouped batch sampler...")
    
    # Extract group_ids from train data
    train_group_ids = extract_group_ids_from_dataset(train_data)
    
    # Create the grouped sampler
    train_sampler = GroupedBatchSampler(
        group_ids=train_group_ids,
        batch_size=config.data.train_batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    # =========================================================================
    # Run training with custom sampler
    # =========================================================================
    print("Starting training with grouped batch sampler...")
    
    try:
        run_ppo_with_grouped_sampler(config, train_sampler, tokenizer)
        print("Training completed!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        for f in [train_file, val_file]:
            if os.path.exists(f):
                os.unlink(f)


def run_ppo_with_grouped_sampler(config, train_sampler, tokenizer):
    """
    Custom run_ppo that uses our grouped batch sampler.
    
    This ensures all hints from the same group are in the same batch,
    which is required for group-aware reward computation.
    """
    import ray
    from verl.trainer.main_ppo import (
        create_rl_dataset,
        RayPPOTrainer,
        Role,
        ResourcePoolManager,
    )
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.utils.dataset.rl_dataset import collate_fn
    
    # Initialize Ray
    ray_init_config = config.ray_kwargs.get("ray_init", {})
    ray.init(**ray_init_config)
    
    # Create datasets
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor=None)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor=None)
    
    # Setup resource pool
    n_gpus_per_node = config.trainer.n_gpus_per_node
    nnodes = config.trainer.nnodes
    
    resource_pool_spec = {
        "actor_rollout": [n_gpus_per_node] * nnodes,
    }
    
    mapping = {
        Role.ActorRollout: "actor_rollout",
        Role.ActorRolloutRef: "actor_rollout",
    }
    
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )
    
    role_worker_mapping = {
        Role.ActorRollout: ActorRolloutRefWorker,
        Role.ActorRolloutRef: ActorRolloutRefWorker,
    }
    
    # Create reward function
    reward_fn = load_reward_manager(
        config,
        tokenizer,
        num_examine=2,
    )
    
    # Create trainer with our custom sampler
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        reward_fn=reward_fn,
        val_reward_fn=reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,  # ← Our grouped sampler!
    )
    
    # Initialize workers and run
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
