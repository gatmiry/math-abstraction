#!/usr/bin/env python3
"""
RL training using GRPO (Group Relative Policy Optimization) on finetuned Qwen model.
Only computes gradients for tokens that are the starting token of a lemma/theorem.

Requirements:
    pip install torch transformers datasets accelerate peft
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, load_from_disk
import json
import os
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm


class LemmaTheoremDataset(Dataset):
    """Dataset for loading problems and solutions with lemma/theorem annotations."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data from JSONL or dataset
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            # Assume it's a HuggingFace dataset path
            dataset = load_from_disk(data_path) if os.path.isdir(data_path) else load_dataset(data_path)
            if isinstance(dataset, dict):
                dataset = dataset.get('train', dataset.get('test', list(dataset.values())[0]))
            self.data = [dataset[i] for i in range(len(dataset))]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        problem = item.get('problem', '')
        solution = item.get('solution', item.get('generated_solution', ''))
        
        # Format prompt
        prompt = f"Solve this problem without using any external tools. Put your solution in \\boxed{{...}} format.\n\nHere is the problem:\n\n{problem}\n\nSolution:"
        
        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        solution_tokens = self.tokenizer.encode(solution, add_special_tokens=False)
        
        # Combine and truncate
        full_text = prompt_tokens + solution_tokens
        if len(full_text) > self.max_length:
            full_text = full_text[:self.max_length]
        
        # Find lemma/theorem starting positions in solution
        lemma_theorem_mask = self._find_lemma_theorem_positions(
            solution, len(prompt_tokens), len(solution_tokens)
        )
        
        return {
            'input_ids': torch.tensor(full_text, dtype=torch.long),
            'prompt_length': len(prompt_tokens),
            'lemma_theorem_mask': torch.tensor(lemma_theorem_mask, dtype=torch.bool),
            'problem': problem,
            'solution': solution
        }
    
    def _find_lemma_theorem_positions(self, solution: str, prompt_len: int, solution_len: int) -> List[bool]:
        """
        Find positions in solution that are:
        1. Starting tokens of lemma/theorem blocks
        2. Core address tokens of Wikipedia URLs within lemma/theorem blocks
           (excluding redundant parts like "https://", "//", ".org", etc.)
        
        The core address is the part after "/wiki/" in URLs like:
        https://en.wikipedia.org/wiki/Dilation_(geometry) -> "Dilation_(geometry)"
        
        Returns a boolean mask of length solution_len.
        """
        mask = [False] * solution_len
        
        # Tokenize solution to find token positions more accurately
        # Use tokenizer's encoding with offsets
        encoding = self.tokenizer(
            solution,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']
        
        # Find all lemma/theorem block ranges
        # Pattern: \begin{lemmatheorem}, \begin{lemmatheorembox}, or \begin{intermediatederivation}
        lemma_patterns = [
            r'\\begin\{lemmatheorem\}',
            r'\\begin\{lemmatheorembox\}',
            r'\\begin\{intermediatederivation\}'
        ]
        end_patterns = [
            r'\\end\{lemmatheorem\}',
            r'\\end\{lemmatheorembox\}',
            r'\\end\{intermediatederivation\}'
        ]
        
        # Find all lemma/theorem block boundaries
        block_ranges = []
        for lemma_pattern, end_pattern in zip(lemma_patterns, end_patterns):
            lemma_matches = list(re.finditer(lemma_pattern, solution))
            end_matches = list(re.finditer(end_pattern, solution))
            
            for lemma_match in lemma_matches:
                # Find the corresponding end match
                lemma_start = lemma_match.start()
                lemma_end_char = lemma_match.end()
                
                # Find the matching \end{...}
                for end_match in end_matches:
                    if end_match.start() > lemma_end_char:
                        block_ranges.append((lemma_start, end_match.end()))
                        break
        
        # Mark ALL tokens that are part of the \begin{...} commands
        # (not just the first token, but all tokens spanning the entire command)
        for lemma_pattern in lemma_patterns:
            for match in re.finditer(lemma_pattern, solution):
                cmd_start_char = match.start()
                cmd_end_char = match.end()
                
                # Find all tokens that overlap with this command
                for i, (char_start, char_end) in enumerate(offsets):
                    if i >= solution_len:
                        continue
                    
                    # Check if this token overlaps with the command
                    token_overlaps = (char_start < cmd_end_char and char_end > cmd_start_char)
                    
                    if token_overlaps:
                        mask[i] = True
        
        # Find Wikipedia URLs within lemma/theorem blocks and mark core address tokens
        # Pattern to match Wikipedia URLs: https://en.wikipedia.org/wiki/Page_Name
        wikipedia_url_pattern = r'https?://(?:www\.)?(?:[a-z]{2}\.)?wikipedia\.org/wiki/([^\s\)\}\\]+)'
        
        for block_start, block_end in block_ranges:
            block_text = solution[block_start:block_end]
            
            # Find all Wikipedia URLs in this block
            for url_match in re.finditer(wikipedia_url_pattern, block_text, re.IGNORECASE):
                full_url = url_match.group(0)
                core_address = url_match.group(1)  # The part after /wiki/ (e.g., "Dilation_(geometry)")
                
                # Find the character position of the core address within the full solution
                url_start_in_solution = block_start + url_match.start()
                wiki_path_pos = full_url.find('/wiki/')
                if wiki_path_pos == -1:
                    continue
                
                # Calculate the exact character range of the core address
                core_address_start_char = url_start_in_solution + wiki_path_pos + len('/wiki/')
                core_address_end_char = core_address_start_char + len(core_address)
                
                # Find which tokens correspond to the core address
                # We need to exclude tokens that are part of "https://", "//", ".org", etc.
                for i, (char_start, char_end) in enumerate(offsets):
                    if i >= solution_len:
                        continue
                    
                    # Check if this token's character range overlaps with the core address range
                    token_overlaps = (char_start < core_address_end_char and char_end > core_address_start_char)
                    
                    if token_overlaps:
                        # Decode the token to check its content
                        token_text = self.tokenizer.decode([tokens[i]], skip_special_tokens=True)
                        
                        # Calculate how much of this token is within the core address
                        overlap_start = max(char_start, core_address_start_char)
                        overlap_end = min(char_end, core_address_end_char)
                        overlap_ratio = (overlap_end - overlap_start) / (char_end - char_start) if char_end > char_start else 0
                        
                        # Skip tokens that are clearly part of redundant URL parts
                        # These patterns match redundant parts we want to exclude
                        redundant_patterns = [
                            r'^https?',
                            r'^://',
                            r'^//',
                            r'www\.',
                            r'wikipedia',
                            r'\.org',
                            r'^/wiki/?$',
                            r'^/wiki/',
                        ]
                        
                        is_redundant = any(re.match(pattern, token_text, re.IGNORECASE) for pattern in redundant_patterns)
                        
                        # Mark token if:
                        # 1. It's not redundant
                        # 2. It overlaps with the core address
                        # 3. At least 50% of the token is within the core address (to avoid edge cases)
                        if not is_redundant and overlap_ratio > 0.5:
                            mask[i] = True
                        # Also mark if the token is fully within the core address (even if small)
                        elif (char_start >= core_address_start_char and char_end <= core_address_end_char):
                            if not is_redundant:
                                mask[i] = True
        
        return mask


class GRPOTrainer:
    """Group Relative Policy Optimization trainer for lemma/theorem tokens."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        learning_rate: float = 1e-6,
        batch_size: int = 4,
        group_size: int = 8,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Reference model for computing KL divergence
        self.ref_model = None
    
    def compute_reward(self, generated_text: str, ground_truth: str, problem: str) -> float:
        """
        Compute reward for generated solution by comparing answers in \\boxed{}.
        
        Args:
            generated_text: The generated solution text
            ground_truth: The ground truth solution text
            problem: The problem statement (not used currently)
        
        Returns:
            Reward value: 1.0 if answers match, 0.0 otherwise (with penalties)
        """
        def extract_boxed_answer(text: str) -> Optional[str]:
            """Extract answer from \\boxed{...} in text."""
            # Try to find \boxed{...} pattern
            # Handle nested braces by finding the matching closing brace
            pattern = r'\\boxed\{'
            matches = list(re.finditer(pattern, text))
            
            if not matches:
                return None
            
            # Get the last match (most likely the final answer)
            match = matches[-1]
            start_pos = match.end()
            
            # Find the matching closing brace
            brace_count = 1
            end_pos = start_pos
            while end_pos < len(text) and brace_count > 0:
                if text[end_pos] == '{':
                    brace_count += 1
                elif text[end_pos] == '}':
                    brace_count -= 1
                end_pos += 1
            
            if brace_count == 0:
                answer = text[start_pos:end_pos-1]  # -1 to exclude the closing brace
                return answer.strip()
            
            return None
        
        def normalize_answer(answer: str) -> str:
            """Normalize answer for comparison (remove extra whitespace, etc.)."""
            # Remove extra whitespace
            answer = ' '.join(answer.split())
            # Remove LaTeX commands that don't affect the answer value
            # (This is a simple normalization - can be extended)
            return answer
        
        # Extract answers from both texts
        correct_answer = extract_boxed_answer(ground_truth)
        generated_answer = extract_boxed_answer(generated_text)
        
        # If we can't find answers, return base reward
        if correct_answer is None:
            # No correct answer found in ground truth
            return 0.0
        
        if generated_answer is None:
            # Generated solution doesn't have a boxed answer
            return -0.5
        
        # Normalize and compare
        correct_normalized = normalize_answer(correct_answer)
        generated_normalized = normalize_answer(generated_answer)
        
        # Check if answers match
        if correct_normalized == generated_normalized:
            return 1.0
        
        # Partial credit: check if correct answer is contained in generated answer
        # (useful for cases where generated answer has extra formatting)
        if correct_normalized in generated_normalized or generated_normalized in correct_normalized:
            return 0.5
        
        # Penalty for wrong answer
        return -0.5
    
    def generate_with_model(self, prompts: List[str], max_new_tokens: int = 512) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate solutions from prompts and compute log probabilities.
        Returns: (generated_ids_list, log_probs_list)
        """
        self.model.eval()
        all_generated_ids = []
        all_log_probs = []
        
        for prompt in prompts:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            prompt_length = inputs['input_ids'].shape[1]
            
            # Generate with return_dict_in_generate to get scores
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_ids = outputs.sequences[0, prompt_length:]
            all_generated_ids.append(generated_ids)
            
            # Compute log probs from scores
            scores = outputs.scores  # List of [1, vocab_size] tensors
            log_probs_list = []
            
            for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
                # Get log prob of the generated token
                log_probs = F.log_softmax(score[0], dim=-1)
                log_prob = log_probs[token_id]
                log_probs_list.append(log_prob)
            
            if log_probs_list:
                all_log_probs.append(torch.stack(log_probs_list))
            else:
                all_log_probs.append(torch.tensor([], device=self.device))
        
        return all_generated_ids, all_log_probs
    
    def compute_grpo_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lemma_theorem_mask: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        prompt_lengths: List[int]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GRPO loss only for lemma/theorem starting tokens.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            lemma_theorem_mask: [batch_size, seq_len] - True for lemma/theorem start tokens
            rewards: [batch_size] - rewards for each sequence
            old_log_probs: [batch_size, seq_len] - log probs from reference model
            prompt_lengths: List of prompt lengths for each sequence
        """
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Compute log probs for generated tokens only
        batch_size, seq_len = input_ids.shape
        new_log_probs = []
        
        for b in range(batch_size):
            prompt_len = prompt_lengths[b]
            seq_log_probs = []
            
            for t in range(prompt_len, seq_len - 1):
                # Log prob of next token
                log_prob = F.log_softmax(logits[b, t], dim=-1)[input_ids[b, t + 1]]
                seq_log_probs.append(log_prob)
            
            # Pad to seq_len
            seq_log_probs = [torch.tensor(0.0, device=self.device)] * prompt_len + seq_log_probs
            if len(seq_log_probs) < seq_len:
                seq_log_probs += [torch.tensor(0.0, device=self.device)] * (seq_len - len(seq_log_probs))
            
            new_log_probs.append(torch.stack(seq_log_probs[:seq_len]))
        
        new_log_probs = torch.stack(new_log_probs)  # [batch_size, seq_len]
        
        # Compute ratio (importance sampling)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Mask: only compute loss for lemma/theorem starting tokens
        # Also mask out prompt tokens
        loss_mask = lemma_theorem_mask.clone()
        for b in range(batch_size):
            loss_mask[b, :prompt_lengths[b]] = False
        
        # Group rewards (for GRPO, we compare within groups)
        # Reshape rewards for group comparison
        num_groups = batch_size // self.group_size
        if num_groups == 0:
            num_groups = 1
        
        # Compute advantages using group-relative method
        advantages = self._compute_group_relative_advantages(rewards, num_groups)
        
        # Expand advantages to match sequence length (only for lemma/theorem tokens)
        expanded_advantages = advantages.unsqueeze(1).expand(-1, seq_len)
        expanded_advantages = expanded_advantages * loss_mask.float()
        
        # GRPO loss: clipped policy gradient
        policy_loss_1 = -expanded_advantages * ratio
        policy_loss_2 = -expanded_advantages * torch.clamp(
            ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        )
        policy_loss = torch.max(policy_loss_1, policy_loss_2)
        
        # Only apply to lemma/theorem tokens
        policy_loss = policy_loss * loss_mask.float()
        
        # Average over non-zero elements
        num_valid_tokens = loss_mask.sum().float()
        if num_valid_tokens > 0:
            loss = policy_loss.sum() / num_valid_tokens
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # KL penalty (optional, for stability)
        kl_penalty = 0.1 * (new_log_probs - old_log_probs).pow(2) * loss_mask.float()
        if num_valid_tokens > 0:
            kl_penalty = kl_penalty.sum() / num_valid_tokens
        else:
            kl_penalty = torch.tensor(0.0, device=self.device)
        
        total_loss = loss + kl_penalty
        
        metrics = {
            'policy_loss': loss.item(),
            'kl_penalty': kl_penalty.item(),
            'total_loss': total_loss.item(),
            'num_lemma_tokens': num_valid_tokens.item(),
            'mean_reward': rewards.mean().item()
        }
        
        return total_loss, metrics
    
    def _compute_group_relative_advantages(self, rewards: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        Compute group-relative advantages for GRPO.
        Advantages are computed relative to the mean reward in each group.
        """
        batch_size = len(rewards)
        group_size = batch_size // num_groups if num_groups > 0 else batch_size
        
        advantages = torch.zeros_like(rewards)
        
        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, batch_size)
            group_rewards = rewards[start_idx:end_idx]
            group_mean = group_rewards.mean()
            advantages[start_idx:end_idx] = group_rewards - group_mean
        
        return advantages
    
    def compute_log_probs_for_sequence(self, input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
        """Compute log probabilities for a sequence using the current model."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.unsqueeze(0))
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Compute log probs for each position
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Extract log probs for actual tokens (shift by 1)
            seq_log_probs = []
            for t in range(prompt_length, input_ids.shape[0] - 1):
                token_id = input_ids[t + 1]
                seq_log_probs.append(log_probs[t, token_id])
            
            # Pad with zeros for prompt tokens
            full_log_probs = torch.zeros(input_ids.shape[0], device=self.device)
            if seq_log_probs:
                full_log_probs[prompt_length:prompt_length + len(seq_log_probs)] = torch.stack(seq_log_probs)
            
            return full_log_probs
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        ground_truths: List[str],
        problems: List[str]
    ) -> Dict:
        """Perform one training step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        lemma_theorem_mask = batch['lemma_theorem_mask'].to(self.device)
        prompt_lengths = batch['prompt_length'].tolist()
        
        # Compute old log probs using reference model (or current model in eval mode)
        # For simplicity, we'll use the current model in eval mode as reference
        old_log_probs_list = []
        for b in range(input_ids.shape[0]):
            seq_ids = input_ids[b]
            prompt_len = prompt_lengths[b]
            old_log_probs = self.compute_log_probs_for_sequence(seq_ids, prompt_len)
            old_log_probs_list.append(old_log_probs)
        old_log_probs = torch.stack(old_log_probs_list)
        
        # Compute rewards based on ground truth solutions
        # In practice, you might want to generate new solutions and evaluate them
        rewards = torch.tensor([
            self.compute_reward(gt, gt, prob)  # Using ground truth as generated for now
            for gt, prob in zip(ground_truths, problems)
        ], device=self.device)
        
        # Compute loss
        loss, metrics = self.compute_grpo_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lemma_theorem_mask=lemma_theorem_mask,
            rewards=rewards,
            old_log_probs=old_log_probs,
            prompt_lengths=prompt_lengths
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return metrics


def main():
    # Configuration
    MODEL_PATH = "./qwen_finetuned"  # Path to finetuned model
    DATA_PATH = "newopenaioutputs/transformed_solutions_qwen2-math-7b-instruct_dataset"  # Dataset path
    OUTPUT_DIR = "./rl_lemma_model"
    BATCH_SIZE = 4
    GROUP_SIZE = 8
    LEARNING_RATE = 1e-6
    NUM_EPOCHS = 3
    MAX_LENGTH = 2048
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = LemmaTheoremDataset(DATA_PATH, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: {
            'input_ids': torch.nn.utils.rnn.pad_sequence(
                [item['input_ids'] for item in x],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            ),
            'attention_mask': torch.nn.utils.rnn.pad_sequence(
                [torch.ones(len(item['input_ids'])) for item in x],
                batch_first=True,
                padding_value=0
            ),
            'lemma_theorem_mask': torch.nn.utils.rnn.pad_sequence(
                [item['lemma_theorem_mask'] for item in x],
                batch_first=True,
                padding_value=False
            ),
            'prompt_length': torch.tensor([item['prompt_length'] for item in x]),
            'problem': [item['problem'] for item in x],
            'solution': [item['solution'] for item in x]
        }
    )
    
    # Initialize trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        device=device
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_metrics = {
            'policy_loss': [],
            'kl_penalty': [],
            'total_loss': [],
            'num_lemma_tokens': [],
            'mean_reward': []
        }
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            try:
                metrics = trainer.train_step(
                    batch=batch,
                    ground_truths=batch['solution'],
                    problems=batch['problem']
                )
                
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"\nBatch {batch_idx + 1}:")
                    for key, values in epoch_metrics.items():
                        if values:
                            print(f"  {key}: {np.mean(values[-10:]):.4f}")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        for key, values in epoch_metrics.items():
            if values:
                print(f"  Mean {key}: {np.mean(values):.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    print(f"\nSaving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")


if __name__ == "__main__":
    main()

