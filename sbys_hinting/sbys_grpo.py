#!/usr/bin/env python3
"""
GRPO training script for Omni-MATH dataset using verl with Ray for distributed training.
Configured for 2 nodes, each with 8 H100 GPUs (16 GPUs total).
Uses Hydra to properly load verl's default configs and override specific values.
"""

import os
import sys
import argparse
import apple_bolt as bolt

# Add parent directory to PYTHONPATH so 'sbys_hinting' module can be imported by verl workers
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
# Also set as environment variable so Ray workers inherit it
if "PYTHONPATH" in os.environ:
    if _parent_dir not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = f"{_parent_dir}:{os.environ['PYTHONPATH']}"
else:
    os.environ["PYTHONPATH"] = _parent_dir

# vLLM 0.8.5 V1 engine works fine, but keep this for compatibility with older versions
os.environ.setdefault("VLLM_USE_V1", "0")

# Tell wandb to save code with each run
os.environ["WANDB_SAVE_CODE"] = "true"

# Token file path (not tracked by git)
# Try sbys_hinting first, then fall back to hinting folder
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "hf_token.txt")
if not os.path.exists(TOKEN_FILE):
    TOKEN_FILE = os.path.join(os.path.dirname(__file__), "..", "hinting", "hf_token.txt")

# System prompt file path
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt_full_solution.txt")

# Which system prompt to use (from system_prompt_full_solution.txt)
SYSTEM_PROMPT_NAME = "full_solution_simple"

HINT_LEVEL = -1
VAL_SIZE = 128 
PROMPT_UPDATE_ENABLED = True
PROMPT_UPDATE_MAX_ASSISTANT_TURNS = 2  # Turn 1: inject hints, Turn 2: evaluate
PROMPT_UPDATE_MAX_USER_TURNS = 2
PROMPT_RESET_PREFIX = "__RESET_PROMPT__\n"
PROMPT_RESET_SYSTEM_TAG = "__SYSTEM__\n"
PROMPT_RESET_USER_TAG = "__USER__\n"

def load_system_prompt(name: str = None):
    """Load a named system prompt from file.
    
    File format: Multiple prompts separated by ===PROMPT: name=== headers.
    
    Args:
        name: Name of the prompt to load. If None, uses SYSTEM_PROMPT_NAME.
    
    Returns:
        The system prompt string.
    """
    if name is None:
        name = SYSTEM_PROMPT_NAME
    
    with open(SYSTEM_PROMPT_FILE, 'r') as f:
        content = f.read()
    
    # Parse prompts from file
    prompts = {}
    current_name = None
    current_lines = []
    
    for line in content.split('\n'):
        if line.startswith('===PROMPT:') and line.endswith('==='):
            # Save previous prompt if exists
            if current_name is not None:
                prompts[current_name] = '\n'.join(current_lines).strip()
            # Start new prompt
            current_name = line[10:-3].strip()
            current_lines = []
        else:
            current_lines.append(line)
    
    # Save last prompt
    if current_name is not None:
        prompts[current_name] = '\n'.join(current_lines).strip()
    
    if name not in prompts:
        available = list(prompts.keys())
        raise ValueError(f"System prompt '{name}' not found. Available: {available}")
    
    return prompts[name]


def load_tokens():
    """Load tokens from file. Format: KEY=VALUE per line."""
    tokens = {}
    if not os.path.exists(TOKEN_FILE):
        print(f"[WARNING] Token file not found: {TOKEN_FILE}")
        return tokens
    
    with open(TOKEN_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                tokens[key.strip()] = value.strip()
    
    return tokens


def login_huggingface(tokens):
    """Login to HuggingFace using token from dict."""
    token = tokens.get('HF_TOKEN')
    if not token:
        print("[WARNING] HF_TOKEN not found in token file")
        return False
    
    # Set environment variable for transformers/datasets
    os.environ["HF_TOKEN"] = token
    
    # Login via huggingface_hub
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[INFO] Successfully logged in to HuggingFace")
        return True
    except Exception as e:
        print(f"[WARNING] HuggingFace login failed: {e}")
        return False


def login_wandb(tokens):
    """Login to Weights & Biases using API key from dict."""
    api_key = tokens.get('WANDB_API_KEY')
    if not api_key or api_key == 'YOUR_WANDB_API_KEY_HERE':
        print("[WARNING] WANDB_API_KEY not configured in token file")
        return False
    
    # Set environment variable for wandb
    os.environ["WANDB_API_KEY"] = api_key
    
    try:
        import wandb
        wandb.login(key=api_key, relogin=True)
        print("[INFO] Successfully logged in to Wandb")
        return True
    except Exception as e:
        print(f"[WARNING] Wandb login failed: {e}")
        return False

# NCCL P2P workaround for hardware issues with NVLink
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

import re
import sys
import random
from verl.interactions.base import BaseInteraction
import ray
import tempfile
import datetime
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import load_from_disk
from typing import List, Dict, Optional, Any
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.schemas import AsyncRolloutRequest, BASE_CHAT_HISTORY, Message
# Ensure Ray runtime_env carries necessary env vars
try:
    from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
    PPO_RAY_RUNTIME_ENV.setdefault("env_vars", {})
    PPO_RAY_RUNTIME_ENV["env_vars"]["VLLM_USE_V1"] = "0"
    # NCCL P2P workaround for hardware issues
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_IGNORE_DISABLED_P2P"] = "1"
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_P2P_DISABLE"] = "1"
    # Add CUDA library paths for sglang subprocesses
    cuda_lib_paths = [
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cublas/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cufft/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/nccl/lib",
    ]
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    PPO_RAY_RUNTIME_ENV["env_vars"]["LD_LIBRARY_PATH"] = ":".join(cuda_lib_paths) + ":" + existing_ld_path
    # Add sglang to pip packages for workers
    PPO_RAY_RUNTIME_ENV.setdefault("pip", [])
    PPO_RAY_RUNTIME_ENV["pip"].append("sglang[all]==0.4.6.post1")
except ImportError:
    pass  # verl 0.4.x may not have this

# Configuration
DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = os.path.join(os.path.dirname(__file__), "outputs", "hint_helped_dataset", "hint_helped_dataset")
MAX_NUM = None  # Limit dataset to last MAX_NUM rows (None = use all data). Useful for testing.

# Training hyperparameters
TRAIN_BATCH_SIZE = 64  # Reduced for faster iteration
TOTAL_EPOCHS = 50
TEST_BATCH_SIZE = 128  # Reduced for faster validation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GRPO Training for math problem solving")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a local HuggingFace-format model to start training from. "
             "Must contain tokenizer files (vocab.json, etc.). "
             "If not provided, uses the default HuggingFace model."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a verl checkpoint directory to resume training from. "
             "e.g., /path/to/model_dir/global_step_50"
    )
    parser.add_argument(
        "--hint-level",
        type=int,
        default=None,
        help="Which hint level to use (overrides HINT_LEVEL constant)"
    )
    return parser.parse_args()

# Distributed training configuration
# 2 nodes, 8 GPUs per node = 16 GPUs total
NUM_NODES = 1  # Testing with single node first
GPUS_PER_NODE = 8


# Import the math_verify based answer checking logic from math_checker.py
# Uses Hugging Face's math_verify library for standardized verification
import importlib.util
import os as _os
_math_checker_path = _os.path.join(_os.path.dirname(__file__), "math_checker.py")
_spec = importlib.util.spec_from_file_location("math_checker", _math_checker_path)
_math_checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math_checker)
check_answer = _math_checker.check_answer
extract_boxed_answer = _math_checker.extract_boxed_answer


# Global counter for compute_score logging (tracks samples that bypass interaction)
_compute_score_log_counter = 0
_compute_score_log_limit = 5  # Log detailed info for first N samples

def compute_score(
    data_source: str = None,
    solution_str: str = None,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """Compute reward score for a single solution (verl 0.4.1 format).
    
    This function is called by verl's NaiveRewardManager for each sample.
    Uses math_verify library (Hugging Face standard) for answer verification.
    
    Args:
        data_source: Data source identifier (not used here)
        solution_str: Generated solution string
        ground_truth: Ground truth answer (from dataset 'answer' field)
        extra_info: Extra info dictionary
        **kwargs: Additional keyword arguments
    
    Returns:
        Reward score (1.0 if answer matches, 0.0 otherwise)
    """
    global _compute_score_log_counter
    
    # DEBUG: Log ALL parameters received by compute_score
    if _compute_score_log_counter < 3:
        print("\n" + "!" * 80)
        print("[compute_score DEBUG] ALL PARAMETERS RECEIVED:")
        print(f"  data_source = {repr(data_source)}")
        print(f"  solution_str = {repr(solution_str[:200] if solution_str else solution_str)}...")
        print(f"  ground_truth = {repr(ground_truth)}")
        print(f"  extra_info = {repr(extra_info)}")
        print(f"  kwargs keys = {list(kwargs.keys())}")
        for k, v in kwargs.items():
            v_str = repr(v)[:200] if v else repr(v)
            print(f"    kwargs[{k}] = {v_str}")
        print("!" * 80 + "\n")
    
    if ground_truth is None:
        print(f"[compute_score] WARNING: ground_truth is None! Returning 0.0")
        return 0.0
    
    if solution_str is None:
        print(f"[compute_score] WARNING: solution_str is None! Returning 0.0")
        return 0.0
    
    # Use math_checker.check_answer which uses math_verify library
    # It handles: LaTeX parsing, symbolic comparison, tuple matching,
    # numeric equivalence, text normalization, etc.
    is_correct = check_answer(solution_str, ground_truth)
    boxed_answer = extract_boxed_answer(solution_str)
    
    # ==================== DETAILED LOGGING for compute_score ====================
    # This logs samples that go through verl's reward function
    # If interaction was NOT called, this is the ONLY place we see the generation
    if _compute_score_log_counter < _compute_score_log_limit:
        _compute_score_log_counter += 1
        print("\n" + "#" * 80)
        print(f"[COMPUTE_SCORE DETAILED LOG #{_compute_score_log_counter}] - This is what verl's reward function receives")
        print("#" * 80)
        print(f"\n>>> DATA_SOURCE: {data_source}")
        print(f"\n>>> GROUND_TRUTH:\n{ground_truth[:300] if ground_truth else 'None'}...")
        print(f"\n>>> SOLUTION_STR (model generation - check if single or multi-turn):\n{solution_str[:2000] if solution_str else 'None'}...")
        print(f"\n>>> EXTRA_INFO: {extra_info}")
        print(f"\n>>> KWARGS: {list(kwargs.keys())}")
        print(f"\n>>> BOXED_ANSWER: {boxed_answer}")
        print(f">>> IS_CORRECT: {is_correct}")
        print("#" * 80 + "\n")
    
    # Debug logging (sample 1 in 50)
    if random.random() < 0.02:
        print(f"[compute_score DEBUG] ground_truth={ground_truth[:80] if ground_truth else None}...")
        print(f"[compute_score DEBUG] boxed_answer={boxed_answer[:80] if boxed_answer else None}...")
        print(f"[compute_score DEBUG] is_correct={is_correct}")
    
    return 1.0 if is_correct else 0.0


@ray.remote
class ProblemStateActor:
    """Ray Actor to maintain persistent problem state across all workers.
    
    This actor is shared across all workers in distributed training,
    ensuring that problem state persists across GRPO steps regardless
    of which worker processes the problem.
    """
    
    def __init__(self):
        self._problem_state = {}  # Keyed by problem text
    
    def get_state(self, problem_key: str) -> Optional[Dict[str, Any]]:
        """Get state for a problem, returns None if not exists."""
        return self._problem_state.get(problem_key)
    
    def set_state(self, problem_key: str, state: Dict[str, Any]) -> None:
        """Set state for a problem."""
        self._problem_state[problem_key] = state
    
    def update_state(self, problem_key: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific fields in problem state and return the updated state."""
        if problem_key not in self._problem_state:
            self._problem_state[problem_key] = {}
        self._problem_state[problem_key].update(updates)
        return self._problem_state[problem_key]
    
    def init_state_if_missing(self, problem_key: str, default_state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state for a problem if it doesn't exist, return current state."""
        if problem_key not in self._problem_state:
            self._problem_state[problem_key] = default_state
        return self._problem_state[problem_key]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all problems."""
        total_problems = len(self._problem_state)
        total_attempts = sum(s.get("total_attempts", 0) for s in self._problem_state.values())
        total_correct = sum(s.get("total_correct", 0) for s in self._problem_state.values())
        return {
            "total_problems": total_problems,
            "total_attempts": total_attempts,
            "total_correct": total_correct,
        }
    
    def initialize_all_problems(self, problems_data: List[Dict[str, Any]]) -> int:
        """Initialize state for all problems at once (called at start of training).
        
        Args:
            problems_data: List of dicts with 'problem' and 'sbys_solution' keys
        
        Returns:
            Number of problems initialized
        """
        count = 0
        for item in problems_data:
            problem_key = item.get("problem")
            sbys_solution = item.get("sbys_solution") or []
            if problem_key and problem_key not in self._problem_state:
                self._problem_state[problem_key] = {
                    "guide_steps_count": len(sbys_solution),
                    "unable_index": 0,
                    "able_index": len(sbys_solution),
                    "try_index": 0,
                    "total_attempts": 0,
                    "total_correct": 0,
                }
                count += 1
        return count


# Global name for the shared actor
PROBLEM_STATE_ACTOR_NAME = "problem_state_actor"


def get_or_create_problem_state_actor():
    """Get existing ProblemStateActor or create a new one."""
    try:
        # Try to get existing actor
        return ray.get_actor(PROBLEM_STATE_ACTOR_NAME)
    except ValueError:
        # Actor doesn't exist, create it
        return ProblemStateActor.options(
            name=PROBLEM_STATE_ACTOR_NAME,
            lifetime="detached",  # Survives driver failure
            max_concurrency=100,  # Handle many concurrent requests
        ).remote()


class PromptUpdateInteraction(BaseInteraction):
    """Update prompts between generations based on previous response/reward."""
    
    # Class-level counter for detailed logging (shared across instances)
    _sample_log_counter = 0
    _sample_log_limit = 5  # Log detailed info for first N samples per batch
    
    # Class-level dict to store Turn 1 info by problem_key (since instance_id changes between calls)
    _turn1_info_by_problem = {}

    def __init__(self, config):
        super().__init__(config)
        self._instance_state = {}  # Per-instance state (keyed by instance_id)
        # Use Ray Actor for persistent state across all workers
        self._state_actor = get_or_create_problem_state_actor()
    
    ##generate_response returns a 4‑tuple:
    # 1) should_terminate_sequence (bool)
    # 2) content (str) — the user message to add (or reset)
    # 3) reward (float) — score for that turn
    # 4) metrics (dict) — any extra info to log

    async def generate_response(self, instance_id, messages, **kwargs):
        """Two-turn interaction flow for training, single-turn for validation.
        
        Training (has interaction_kwargs):
            Turn 1: Ignore initial generation (it had no hints), inject prompt WITH hints
            Turn 2: Evaluate generation (with hints), update persistent state, terminate
        
        Validation (no interaction_kwargs):
            Single turn: Terminate immediately, let verl's custom_reward_function score
        """
        import math
        
        # Extract kwargs from interaction_kwargs
        ground_truth = kwargs.get("ground_truth")
        sbys_solution = kwargs.get("sbys_solution") or []
        problem = kwargs.get("problem")
        initial_state = kwargs.get("state", {})
        
        # Debug: Log what we received
        print(f"[PromptUpdateInteraction] generate_response called! instance_id={instance_id[:8]}...")
        print(f"[PromptUpdateInteraction] kwargs keys: {list(kwargs.keys())}")
        print(f"[PromptUpdateInteraction] ground_truth={'SET' if ground_truth else 'None'}, problem={'SET' if problem else 'None'}, sbys_solution_len={len(sbys_solution)}")
        
        # ==================== LOG RAW INPUT FROM VERL ====================
        # Log the raw messages and kwargs that verl passed to us
        if PromptUpdateInteraction._sample_log_counter < PromptUpdateInteraction._sample_log_limit:
            print("\n" + "-" * 60)
            print(f"[RAW VERL INPUT - Sample approaching limit {PromptUpdateInteraction._sample_log_counter + 1}/{PromptUpdateInteraction._sample_log_limit}]")
            print("-" * 60)
            print(f">>> MESSAGES FROM VERL ({len(messages)} messages):")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:800]
                print(f"  [{i}] {role.upper()}:")
                print(f"      {content}...")
            print(f"\n>>> KWARGS FROM VERL:")
            for key, val in kwargs.items():
                if isinstance(val, str):
                    print(f"  {key}: {val[:200]}...")
                elif isinstance(val, list):
                    print(f"  {key}: list with {len(val)} items")
                else:
                    print(f"  {key}: {val}")
            print("-" * 60 + "\n")
        
        
        # ==================== TRAINING: Two-turn interaction ====================
        # Verify we have the problem text (required for training)
        if not problem:
            assert False, "No problem text in kwargs for training! kwargs={kwargs}"
            
        
        # Use problem text as key for persistent binary search state (survives across steps)
        problem_key = problem if problem else instance_id
        
        # Get persistent state from Ray Actor (initialized at start of training)
        problem_state = ray.get(self._state_actor.get_state.remote(problem_key))
        
        # Fallback initialization if state wasn't pre-initialized (shouldn't happen normally)
        if problem_state is None:
            print(f"[PromptUpdateInteraction] WARNING: Problem state not pre-initialized for: {problem_key[:50]}...")
            problem_state = {
                "guide_steps_count": len(sbys_solution) if sbys_solution else 0,
                "unable_index": 0,
                "able_index": len(sbys_solution) if sbys_solution else 0,
                "try_index": 0,
                "total_attempts": 0,
                "total_correct": 0,
                "current_turn": 0,  # Track turn number per-problem (persists across instance_ids)
            }
            ray.get(self._state_actor.set_state.remote(problem_key, problem_state))
        
        # Ensure current_turn field exists (for backward compatibility with old state)
        if "current_turn" not in problem_state:
            problem_state["current_turn"] = 0
        
        # Increment turn counter (tracked by problem_key, not instance_id!)
        # This is critical because verl uses different instance_ids for continuation calls
        problem_state["current_turn"] = problem_state.get("current_turn", 0) + 1
        current_turn = problem_state["current_turn"]
        
        # Persist the incremented turn immediately
        ray.get(self._state_actor.set_state.remote(problem_key, problem_state))
        
        # Keep instance_state for storing Turn 1 info needed by Turn 2
        if instance_id not in self._instance_state:
            self._instance_state[instance_id] = {
                **(dict(initial_state) if isinstance(initial_state, dict) else {})
            }
        instance_state = self._instance_state[instance_id]
        # Link instance_id to problem_key so Turn 2 can find Turn 1's stored info
        instance_state["problem_key"] = problem_key
        
        print(f"[PromptUpdateInteraction] Turn {current_turn}, problem_key={problem_key[:50] if problem_key else None}...")
        print(f"[PromptUpdateInteraction] Persistent state: try_index={problem_state['try_index']}, able={problem_state['able_index']}, unable={problem_state['unable_index']}")
        
        # Update guide_steps_count if sbys_solution changed (shouldn't happen, but just in case)
        if problem_state["guide_steps_count"] == 0 and sbys_solution:
            print('guide_steps_count changed from 0 to ', len(sbys_solution), ' this was an error and should not happen')
            problem_state["guide_steps_count"] = len(sbys_solution)
            problem_state["able_index"] = len(sbys_solution)
            ray.get(self._state_actor.set_state.remote(problem_key, problem_state))
        
        # ==================== TURN 1: Inject hints ====================
        # Ignore the initial generation (it had no hints), return prompt WITH hints
        if current_turn == 1:
            print(f"[PromptUpdateInteraction] Turn 1, problem_key={problem_key[:50]}..., try_index={problem_state['try_index']}")
            print(f"[PromptUpdateInteraction] Turn 1: problem={problem[:80] if problem else 'None'}...")
            print(f"[PromptUpdateInteraction] Turn 1: sbys_solution has {len(sbys_solution)} steps")
            
            # Extract Turn 0 generation (the one we're discarding)
            turn0_generation = ""
            turn0_prompt = ""
            for msg in messages:
                if msg.get("role") == "user":
                    turn0_prompt = msg.get("content", "")[:500]
                elif msg.get("role") == "assistant":
                    turn0_generation = msg.get("content", "")
            
            # Construct prompt with current hint level from persistent state
            partial_answer = "\n".join(sbys_solution[:problem_state["try_index"]])
            if problem_state["try_index"] > 0:
                updated_prompt = (
                    f"Problem: {problem}\n"
                    f"Incomplete proof: {partial_answer}\n"
                )
            else:
                updated_prompt = (
                    f"Problem: {problem}\n"
                )
            
            print(f"[PromptUpdateInteraction] Turn 1: updated_prompt={updated_prompt[:100]}...")
            
            system_prompt_name = "full_solution_simple" if problem_state["try_index"] > 0 else "full_solution_with_hint"
            system_prompt = load_system_prompt(system_prompt_name)
            reset_payload = (
                f"{PROMPT_RESET_PREFIX}"
                f"{PROMPT_RESET_SYSTEM_TAG}{system_prompt}\n"
                f"{PROMPT_RESET_USER_TAG}{updated_prompt}"
            )
            
            # Store Turn 1 info for detailed logging in Turn 2
            # Store by problem_key (not instance_id) since verl uses new instance_id for continuation

            
            turn1_info = {
                "turn0_prompt": turn0_prompt,
                "turn0_generation": turn0_generation[:2000],  # Truncate for logging
                "turn1_prompt": updated_prompt,
                "try_index": problem_state["try_index"],
                "problem": problem[:500] if problem else "None",
                "ground_truth": ground_truth[:200] if ground_truth else "None",
            }
            
            # Return 0 reward for turn 1 (we're not evaluating this generation)
            return False, reset_payload, 0.0, {"turn": 1, "try_index": problem_state["try_index"]}
        
        # ==================== TURN 2: Evaluate and update state ====================
        print(f"[PromptUpdateInteraction] Turn 2: Evaluating generation for problem_key={problem_key[:50] if problem_key else None}...")
        
        # Extract the assistant's response
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        # Compute reward
        reward = compute_score(solution_str=last_assistant, ground_truth=ground_truth)
        boxed_answer = extract_boxed_answer(last_assistant) or "N/A"
        
        print(f"[PromptUpdateInteraction] ground_truth={ground_truth[:50] if ground_truth else None}...")
        print(f"[PromptUpdateInteraction] boxed_answer={boxed_answer[:50] if boxed_answer else 'N/A'}...")
        print(f"[PromptUpdateInteraction] reward={reward}")
        
        # Retrieve Turn 1 info stored by problem_key
        turn1_info = {}
        if hasattr(PromptUpdateInteraction, '_turn1_info_by_problem'):
            turn1_info = PromptUpdateInteraction._turn1_info_by_problem.get(problem_key, {})
        
        # ==================== DETAILED LOGGING for sample problems ====================
        # Log full details for first N samples to help debug (console only)
        if PromptUpdateInteraction._sample_log_counter < PromptUpdateInteraction._sample_log_limit:
            PromptUpdateInteraction._sample_log_counter += 1
            
            print("\n" + "=" * 80)
            print(f"[DETAILED SAMPLE LOG #{PromptUpdateInteraction._sample_log_counter}]")
            print("=" * 80)
            print(f"\n>>> PROBLEM:\n{turn1_info.get('problem', 'N/A')}")
            print(f"\n>>> GROUND TRUTH:\n{turn1_info.get('ground_truth', 'N/A')}")
            print(f"\n>>> TRY_INDEX (hints given): {turn1_info.get('try_index', 'N/A')}")
            print(f"\n>>> TURN 0 PROMPT (original, no hints):\n{turn1_info.get('turn0_prompt', 'N/A')[:500]}...")
            print(f"\n>>> TURN 0 GENERATION (DISCARDED):\n{turn1_info.get('turn0_generation', 'N/A')[:1000]}...")
            print(f"\n>>> TURN 1 PROMPT (with hints injected):\n{turn1_info.get('turn1_prompt', 'N/A')[:500]}...")
            print(f"\n>>> TURN 2 GENERATION (SCORED):\n{last_assistant[:1000]}...")
            print(f"\n>>> BOXED ANSWER: {boxed_answer}")
            print(f">>> REWARD: {reward}")
            print("=" * 80 + "\n")
        
        # Update persistent problem state statistics
        problem_state["total_attempts"] = problem_state.get("total_attempts", 0) + 1
        if reward > 0.5:
            problem_state["total_correct"] = problem_state.get("total_correct", 0) + 1

        # Check if solved without hints (try_index == 0 means no hints were given)
        if reward > 0.5 and problem_state["try_index"] == 0:
            print(f"[PromptUpdateInteraction] Correct without hints! attempts={problem_state['total_attempts']}, correct={problem_state['total_correct']}")
            # Reset current_turn for next time this problem is seen
            problem_state["current_turn"] = 0
            ray.get(self._state_actor.set_state.remote(problem_key, problem_state))
            # Clean up instance state and turn1_info
            if instance_id in self._instance_state:
                del self._instance_state[instance_id]
            if hasattr(PromptUpdateInteraction, '_turn1_info_by_problem') and problem_key in PromptUpdateInteraction._turn1_info_by_problem:
                del PromptUpdateInteraction._turn1_info_by_problem[problem_key]
            return True, "", reward, {"correct": True, "try_index": 0, "turn": 2}

        # Binary search logic to update try_index for NEXT time this problem is seen
        old_try_index = problem_state["try_index"]
        if problem_state["try_index"] <= problem_state["unable_index"] and reward > 0.5:
            problem_state["able_index"] = problem_state["try_index"]
            if problem_state["try_index"] == problem_state["unable_index"]:
                problem_state["try_index"] = problem_state["try_index"] - 1
            else:
                problem_state["try_index"] = problem_state["try_index"] - (problem_state["unable_index"] - problem_state["try_index"])
            problem_state["try_index"] = max(problem_state["try_index"], 0)
        elif problem_state["try_index"] >= problem_state["able_index"] and reward < 0.5:
            problem_state["unable_index"] = problem_state["try_index"]
            if problem_state["try_index"] == problem_state["able_index"]:
                problem_state["try_index"] = problem_state["try_index"] + 1
            else:
                problem_state["try_index"] = problem_state["try_index"] + (problem_state["try_index"] - problem_state["able_index"])
            problem_state["try_index"] = min(problem_state["try_index"], problem_state["guide_steps_count"])
        else:
            if reward < 0.5:
                problem_state["unable_index"] = problem_state["try_index"]
                problem_state["try_index"] = math.ceil((problem_state["try_index"] + problem_state["able_index"]) / 2)
            else:
                problem_state["able_index"] = problem_state["try_index"]
                problem_state["try_index"] = math.floor((problem_state["try_index"] + problem_state["unable_index"]) / 2)

        # Reset current_turn for next time this problem is seen
        problem_state["current_turn"] = 0
        
        # Persist updated state to Ray Actor
        ray.get(self._state_actor.set_state.remote(problem_key, problem_state))
        
        print(f"[PromptUpdateInteraction] Updated hint level: {old_try_index} -> {problem_state['try_index']} (able={problem_state['able_index']}, unable={problem_state['unable_index']})")

        # Clean up instance state and turn1_info
        if instance_id in self._instance_state:
            del self._instance_state[instance_id]
        if hasattr(PromptUpdateInteraction, '_turn1_info_by_problem') and problem_key in PromptUpdateInteraction._turn1_info_by_problem:
            del PromptUpdateInteraction._turn1_info_by_problem[problem_key]
        
        # Terminate the sequence - we're done with this problem for this GRPO step
        return True, "", reward, {"try_index": problem_state["try_index"], "turn": 2, "reward": reward}

def _reset_request_prompt(self, processing_class, new_user_content: str, new_system_content: Optional[str] = None) -> None:
    if new_system_content is None:
        system_msg = None
        for msg in self.messages:
            if msg.role == "system":
                system_msg = msg
                break
        if system_msg is None:
            system_msg = Message(role="system", content="You are a helpful assistant.")
    else:
        system_msg = Message(role="system", content=new_system_content)

    self.messages = [system_msg, Message(role="user", content=new_user_content)]

    tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
    multi_modal_data = self.multi_modal_data or {}
    messages = [msg.model_dump() for msg in self.messages]
    tokens_without_prompt = AsyncRolloutRequest._handle_apply_chat_template(
        processing_class,
        messages,
        multi_modal_data=multi_modal_data,
        tools=tools,
        add_generation_prompt=False,
        tokenize=True,
    )
    tokenization_dict_with_prompt = AsyncRolloutRequest._handle_apply_chat_template(
        processing_class,
        messages,
        multi_modal_data=multi_modal_data,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    self.input_ids = tokenization_dict_with_prompt["input_ids"]
    self.attention_mask = tokenization_dict_with_prompt["attention_mask"]
    self.prompt_ids = self.input_ids
    self.prompt_attention_mask = self.attention_mask
    self.position_ids = self.prompt_position_ids = compute_position_id_with_mask(
        torch.tensor(self.attention_mask)
    ).tolist()
    self.loss_mask = self.prompt_loss_mask = [0] * len(self.input_ids)
    self.generation_prompt_ids = self.input_ids[len(tokens_without_prompt) :]
    self.base_conv_wo_gen_prompt_end_pos = len(
        AsyncRolloutRequest._handle_apply_chat_template(
            processing_class,
            BASE_CHAT_HISTORY,
            multi_modal_data=multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        )
    )
    self.base_conv_with_gen_prompt_end_pos = len(
        AsyncRolloutRequest._handle_apply_chat_template(
            processing_class,
            BASE_CHAT_HISTORY,
            multi_modal_data=multi_modal_data,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    self.response_ids = []
    self.response_attention_mask = []
    self.response_position_ids = []
    self.response_loss_mask = []


def install_prompt_reset_hook():
    """Intercept add_user_message to reset the prompt when requested."""
    if getattr(AsyncRolloutRequest.add_user_message, "_prompt_reset_patched", False):
        return

    original_add_user_message = AsyncRolloutRequest.add_user_message

    def patched_add_user_message(self, processing_class, content: str):
        if content.startswith(PROMPT_RESET_PREFIX):
            payload = content[len(PROMPT_RESET_PREFIX) :]
            new_system_content = None
            new_user_content = payload
            if payload.startswith(PROMPT_RESET_SYSTEM_TAG):
                payload = payload[len(PROMPT_RESET_SYSTEM_TAG) :]
                if PROMPT_RESET_USER_TAG in payload:
                    system_part, user_part = payload.split(PROMPT_RESET_USER_TAG, 1)
                    new_system_content = system_part.rstrip("\n")
                    new_user_content = user_part
            _reset_request_prompt(self, processing_class, new_user_content, new_system_content)
            return
        return original_add_user_message(self, processing_class, content)

    patched_add_user_message._prompt_reset_patched = True
    AsyncRolloutRequest.add_user_message = patched_add_user_message


def format_prompt(problem: str, system_prompt: str = None) -> List[Dict[str, str]]:
    if system_prompt is None:
        system_prompt = load_system_prompt(SYSTEM_PROMPT_NAME)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
            "content": f"Problem: {problem}"
            }
        ]
        return messages

def create_rl_dataset(tokenizer, dataset_path: str, max_samples: Optional[int] = None, val_size: int = 128, max_prompt_tokens: int = 2560, hint_level: int = 0):
    """Create RL dataset in verl format with train/val split.
    
    Args:
        tokenizer: Tokenizer instance
        dataset_path: Path to dataset
        max_samples: Maximum number of samples (None = use all)
        val_size: Number of samples for validation (default 64)
        max_prompt_tokens: Maximum prompt length in tokens (default 2560)
    
    Returns:
        Tuple of (train_data, val_data) in verl format
    """
    import json
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Deduplicate by problem text to avoid @4 metrics issues
    original_size = len(dataset)
    seen_problems = set()
    unique_indices = []
    for idx, example in enumerate(dataset):
        problem = example['problem']
        if problem not in seen_problems:
            seen_problems.add(problem)
            unique_indices.append(idx)
    if len(unique_indices) < original_size:
        dataset = dataset.select(unique_indices)
        print(f"[INFO] Deduplicated dataset: {original_size} -> {len(dataset)} ({original_size - len(dataset)} duplicates removed)")
    
    # Format dataset - store messages as JSON for verl to parse
    def format_dataset(examples):
        """Format dataset examples for GRPO training."""
        prompts = []
        answers = []
        sbys_solutions = []
        problems = []

        for problem, answer, sbys_solution in zip(
            examples['problem'], examples['answer'], examples['sbys_solution']
        ):
            if answer:
                messages = format_prompt(problem)
                prompts.append(messages)
                answers.append(answer)
                sbys_solutions.append(sbys_solution)
                problems.append(problem)
        
        return {"prompt": prompts, "answer": answers, "sbys_solution": sbys_solutions, "problem": problems}
    
    # Split into train and val
    

    # Filter out None answers
    #formatted_dataset = formatted_dataset.filter(lambda x: x['answer'] is not None)
    
    # Filter out prompts that are too long
    def is_prompt_short_enough(example):
        """Check if prompt fits within max_prompt_tokens."""
        prompt_text = tokenizer.apply_chat_template(example['prompt'], tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer.encode(prompt_text))
        return token_count <= max_prompt_tokens
    
    
    # Limit dataset to last MAX_NUM rows if specified
    if max_samples is not None and max_samples > 0:
        original_size = len(dataset)
        start_idx = max(0, len(dataset) - max_samples)
        dataset = dataset.select(range(start_idx, len(dataset)))
        print(f"[INFO] Limited dataset from {original_size} to {len(dataset)} rows (using last {max_samples} rows)")
    
    
    
    
    
    total_size = len(dataset)
    val_size_actual = min(val_size, total_size)  # Don't exceed dataset size
    train_size = total_size - val_size_actual
    print(f"[INFO] Split dataset: {train_size} train, {val_size_actual} val")
    

    # Random split to avoid dataset ordering bias
    indices = list(range(total_size))
    random.Random(42).shuffle(indices)
    val_indices = indices[:val_size_actual]
    train_indices = indices[val_size_actual:]
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    # Process dataset
    from functools import partial
    train_dataset = train_dataset.map(
        format_dataset,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        format_dataset,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    original_count = len(train_dataset)
    train_dataset = train_dataset.filter(is_prompt_short_enough)
    filtered_count = original_count - len(train_dataset)
    if filtered_count > 0:
        print(f"[INFO] Filtered out {filtered_count} samples with prompts > {max_prompt_tokens} tokens for train dataset")
    original_count = len(val_dataset)
    val_dataset = val_dataset.filter(is_prompt_short_enough)
    filtered_count = original_count - len(val_dataset)
    if filtered_count > 0:
        print(f"[INFO] Filtered out {filtered_count} samples with prompts > {max_prompt_tokens} tokens for test dataset")

    # Convert to verl format
    # verl expects ground_truth nested under reward_model
    def to_verl_format(dataset_split, enable_interaction: bool):
        rl_data = []
        for item in dataset_split:
            extra_info = {}
            if enable_interaction and PROMPT_UPDATE_ENABLED:
                extra_info["interaction_kwargs"] = {
                    "ground_truth": item["answer"],
                    "state": {"attempt": 0},
                    "sbys_solution": item.get("sbys_solution"),
                    "problem": item.get("problem"),
                }
            row = {
                "prompt": item["prompt"],  # List of message dicts with 'role' and 'content'
                "ground_truth": item["answer"],  # Top-level for custom_reward_function
                "reward_model": {
                    "ground_truth": item["answer"],  # Keep for compatibility
                },
                "data_source": "omni_math",  # Identifier for reward function
            }
            if extra_info:
                row["extra_info"] = extra_info
            rl_data.append(row)
        return rl_data
    
    train_data = to_verl_format(train_dataset, enable_interaction=True)
    val_data = to_verl_format(val_dataset, enable_interaction=False)
    
    return train_data, val_data


def get_verl_config_path():
    """Get path to verl's config directory."""
    import verl
    verl_path = os.path.dirname(verl.__file__)
    config_path = os.path.join(verl_path, "trainer", "config")
    return config_path


def write_interaction_config() -> str:
    """Write a minimal interaction config for prompt updates and return its path."""
    config_text = (
        "interaction:\n"
        "  - class_name: sbys_hinting.sbys_grpo.PromptUpdateInteraction\n"
        "    config: {}\n"
    )
    target_path = os.path.join(os.path.dirname(__file__), "interaction_prompt_update.yaml")
    with open(target_path, "w") as f:
        f.write(config_text)
    return target_path


def distribute_checkpoint_to_all_nodes(checkpoint_dir: str, target_dir: str):
    """
    Distribute checkpoint shards from head node to all worker nodes.
    
    This is needed because verl's FSDP checkpoint loading expects each node
    to have ALL shards available locally.
    
    Args:
        checkpoint_dir: Path to the gathered checkpoint (with all shards)
        target_dir: Path where checkpoint should be placed on all nodes
    
    Returns:
        True if successful
    """
    actor_dir = os.path.join(checkpoint_dir, "actor")
    target_actor_dir = os.path.join(target_dir, "actor")
    
    # Read all checkpoint files from head node
    print(f"[Distribute] Reading checkpoint from {actor_dir}...")
    files_data = {}
    for filename in os.listdir(actor_dir):
        if filename.endswith(".pt"):
            filepath = os.path.join(actor_dir, filename)
            with open(filepath, 'rb') as f:
                files_data[filename] = f.read()
            print(f"[Distribute] Read {filename} ({len(files_data[filename]) // 1024 // 1024} MB)")
    
    # Also read data.pt if it exists
    data_pt_path = os.path.join(checkpoint_dir, "data.pt")
    data_pt_content = None
    if os.path.exists(data_pt_path):
        with open(data_pt_path, 'rb') as f:
            data_pt_content = f.read()
        print(f"[Distribute] Read data.pt")
    
    print(f"[Distribute] Total files to distribute: {len(files_data)}")
    
    # Define Ray remote function to write files on each node
    @ray.remote
    def write_checkpoint_files(files: dict, target_path: str, data_pt: bytes = None):
        """Write checkpoint files to this node."""
        import os
        os.makedirs(target_path, exist_ok=True)
        
        written = 0
        for filename, data in files.items():
            filepath = os.path.join(target_path, filename)
            with open(filepath, 'wb') as f:
                f.write(data)
            written += 1
        
        # Write data.pt to parent directory
        if data_pt is not None:
            parent_dir = os.path.dirname(target_path)
            data_pt_path = os.path.join(parent_dir, "data.pt")
            with open(data_pt_path, 'wb') as f:
                f.write(data_pt)
        
        import socket
        return f"{socket.gethostname()}: wrote {written} files to {target_path}"
    
    # Get all nodes
    nodes = ray.nodes()
    node_ips = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]
    print(f"[Distribute] Distributing to {len(node_ips)} nodes: {node_ips}")
    
    # Schedule write tasks on each node
    tasks = []
    for node_ip in node_ips:
        task = write_checkpoint_files.options(
            resources={f"node:{node_ip}": 0.001}
        ).remote(files_data, target_actor_dir, data_pt_content)
        tasks.append((node_ip, task))
    
    # Wait for all writes to complete
    for node_ip, task in tasks:
        try:
            result = ray.get(task, timeout=600)  # 10 min timeout for large files
            print(f"[Distribute] {result}")
        except Exception as e:
            print(f"[Distribute] Failed on {node_ip}: {e}")
            return False
    
    print(f"[Distribute] ✓ Checkpoint distributed to all nodes at: {target_dir}")
    return True


def gather_checkpoint_to_head_node(checkpoint_dir: str, output_dir: str, num_nodes: int, gpus_per_node: int):
    """
    Gather all FSDP checkpoint shards from worker nodes to the head node.
    
    This keeps the checkpoint in verl's native format so it can be:
    - Used directly with --resume-from for continued training
    - Loaded with verl's inference utilities
    
    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g., /path/to/global_step_50)
        output_dir: Where to save the gathered checkpoint on head node
        num_nodes: Number of nodes used in training
        gpus_per_node: GPUs per node
    
    Returns:
        True if successful, False otherwise
    """
    import shutil
    
    world_size = num_nodes * gpus_per_node
    actor_dir = os.path.join(checkpoint_dir, "actor")
    output_actor_dir = os.path.join(output_dir, "actor")
    
    print(f"[Gather] Collecting checkpoint shards from {num_nodes} nodes...")
    print(f"[Gather] World size: {world_size}")
    print(f"[Gather] Source: {actor_dir}")
    print(f"[Gather] Destination: {output_dir}")
    
    # Create output directory
    os.makedirs(output_actor_dir, exist_ok=True)
    
    # Define Ray remote function to read checkpoint files from each node
    @ray.remote
    def get_checkpoint_files(checkpoint_path: str):
        """Get list of checkpoint files and their contents from this node."""
        import os
        
        files_data = {}
        if os.path.exists(checkpoint_path):
            for filename in os.listdir(checkpoint_path):
                if filename.endswith(".pt"):
                    filepath = os.path.join(checkpoint_path, filename)
                    # Read file as bytes
                    with open(filepath, 'rb') as f:
                        files_data[filename] = f.read()
                    print(f"[Node] Read {filename} ({len(files_data[filename])} bytes)")
        return files_data
    
    # Get list of all node IPs from Ray
    nodes = ray.nodes()
    node_ips = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]
    print(f"[Gather] Found {len(node_ips)} alive nodes: {node_ips}")
    
    # Schedule gather tasks on each node
    gather_tasks = []
    for node_ip in node_ips:
        # Try to run on specific node using scheduling strategy
        task = get_checkpoint_files.options(
            resources={f"node:{node_ip}": 0.001}
        ).remote(actor_dir)
        gather_tasks.append((node_ip, task))
    
    # Also try without node constraint as fallback (will run on any available node)
    # This helps if node labels aren't set up
    fallback_task = get_checkpoint_files.remote(actor_dir)
    
    # Collect results
    all_files = {}
    
    # First try node-specific tasks
    for node_ip, task in gather_tasks:
        try:
            result = ray.get(task, timeout=300)  # 5 min timeout
            for filename, data in result.items():
                if filename not in all_files:
                    all_files[filename] = data
                    print(f"[Gather] Got {filename} from {node_ip}")
        except Exception as e:
            print(f"[Gather] Failed to get files from {node_ip}: {e}")
    
    # Try fallback if we don't have all files
    expected_model_files = world_size
    model_files_count = sum(1 for f in all_files if f.startswith("model_world_size_"))
    
    if model_files_count < expected_model_files:
        print(f"[Gather] Only got {model_files_count}/{expected_model_files} model shards, trying fallback...")
        try:
            result = ray.get(fallback_task, timeout=300)
            for filename, data in result.items():
                if filename not in all_files:
                    all_files[filename] = data
                    print(f"[Gather] Got {filename} from fallback")
        except Exception as e:
            print(f"[Gather] Fallback also failed: {e}")
    
    # Write all collected files to output directory
    print(f"[Gather] Writing {len(all_files)} files to {output_actor_dir}")
    for filename, data in all_files.items():
        output_path = os.path.join(output_actor_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(data)
        print(f"[Gather] Wrote {filename}")
    
    # Copy the latest_checkpointed_iteration.txt if it exists
    parent_dir = os.path.dirname(checkpoint_dir)
    iteration_file = os.path.join(parent_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(iteration_file):
        output_parent = os.path.dirname(output_dir)
        os.makedirs(output_parent, exist_ok=True)
        shutil.copy2(iteration_file, os.path.join(output_parent, "latest_checkpointed_iteration.txt"))
        print(f"[Gather] Copied latest_checkpointed_iteration.txt")
    
    # Verify we got all shards
    model_files = [f for f in all_files if f.startswith("model_world_size_")]
    optim_files = [f for f in all_files if f.startswith("optim_world_size_")]
    extra_files = [f for f in all_files if f.startswith("extra_state_world_size_")]
    
    print(f"\n[Gather] Summary:")
    print(f"  - Model shards: {len(model_files)}/{world_size}")
    print(f"  - Optim shards: {len(optim_files)}/{world_size}")
    print(f"  - Extra state shards: {len(extra_files)}/{world_size}")
    
    if len(model_files) == world_size:
        print(f"\n✓ Successfully gathered complete checkpoint to: {output_dir}")
        print(f"  Use with: python hinting2_grpo_ray.py --resume-from {output_dir}")
        return True
    else:
        missing_ranks = set(range(world_size)) - {int(f.split("_")[-1].replace(".pt", "")) for f in model_files}
        print(f"\n✗ Incomplete checkpoint - missing ranks: {sorted(missing_ranks)}")
        print(f"  The missing shards may be on nodes that are no longer accessible.")
        return False


def main():
    """Main function to run GRPO training with verl and Ray."""
    # Parse command line arguments
    args = parse_args()
    
    # Determine model path (command line arg or default)
    # --model-path: Use a different base model (must have tokenizer files)
    # --resume-from: Resume from a verl checkpoint (uses default model + loads weights)
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_PATH
    
    # Determine hint level (command line arg or constant)
    hint_level = args.hint_level if args.hint_level is not None else HINT_LEVEL
    
    # Extract base model name for output naming
    if args.resume_from:
        # Resuming from verl checkpoint
        path_parts = args.resume_from.rstrip('/').split('/')
        # Find the main model directory (skip "global_step_X", etc.)
        base_model_name = None
        for part in reversed(path_parts):
            if part and not part.startswith('global_step'):
                base_model_name = part
                break
        if not base_model_name:
            base_model_name = "checkpoint"
    elif args.model_path:
        # Using a different base model
        path_parts = args.model_path.rstrip('/').split('/')
        base_model_name = path_parts[-1].lower() if path_parts else "custom"
    else:
        # Using default HuggingFace model
        base_model_name = model_path.split('/')[-1].lower()
    
    # Generate timestamp and experiment name for this run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    special_name = f"hint_level_{hint_level}"
    
    # Add "resumed" to name if loading from checkpoint
    if args.resume_from:
        special_name = f"{special_name}_resumed"
    
    experiment_name = f"grpo_omni_math_{special_name}_{timestamp}"
    
    # Construct output directory with timestamp, special_name, and base model indicator
    model_name = f"qwen3-4b-instruct-2507_{special_name}_{timestamp}_from_{base_model_name}"
    output_dir = os.path.join(bolt.ARTIFACT_DIR, model_name)
    
    print("=" * 80)
    print("GRPO Training with verl and Ray")
    print("=" * 80)
    print(f"Model: {model_path}")
    if args.resume_from:
        print(f"  Resuming from checkpoint: {args.resume_from}")
    elif args.model_path:
        print(f"  (Using custom model path)")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {output_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Hint Level: {hint_level}")
    print(f"Nodes: {NUM_NODES}, GPUs per node: {GPUS_PER_NODE}")
    print("=" * 80)
    
    # Load tokens and login to services
    tokens = load_tokens()
    login_huggingface(tokens)
    login_wandb(tokens)
    
    # Load tokenizer (use default model for tokenizer if resuming from checkpoint)
    print("Loading tokenizer...")
    tokenizer_path = DEFAULT_MODEL_PATH  # Always use base model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create RL dataset with train/val split
    print("Creating RL dataset...")
    train_data, val_data = create_rl_dataset(tokenizer, DATASET_NAME, max_samples=MAX_NUM, val_size=VAL_SIZE, hint_level=hint_level)
    print(f"Created {len(train_data)} training samples, {len(val_data)} validation samples")
    
    # Calculate total training steps (same formula verl uses)
    num_batches_per_epoch = len(train_data) // TRAIN_BATCH_SIZE
    total_training_steps = num_batches_per_epoch * TOTAL_EPOCHS
    print(f"Calculated total_training_steps: {num_batches_per_epoch} batches/epoch × {TOTAL_EPOCHS} epochs = {total_training_steps}")
    
    # Save train dataset to parquet file
    print("Saving datasets to parquet...")
    df_train = pd.DataFrame(train_data)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='_train.parquet', delete=False, dir='/mnt/tmp') as f:
        df_train.to_parquet(f.name, index=False)
        train_dataset_file = f.name
    print(f"Train dataset saved to {train_dataset_file}")
    
    # Save val dataset to parquet file
    df_val = pd.DataFrame(val_data)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='_val.parquet', delete=False, dir='/mnt/tmp') as f:
        df_val.to_parquet(f.name, index=False)
        val_dataset_file = f.name
    print(f"Val dataset saved to {val_dataset_file}")
    
    # Load verl's default config using Hydra
    print("Loading verl configuration with Hydra...")
    config_path = get_verl_config_path()
    
    # Clear any previous Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with verl's config directory
    initialize_config_dir(config_dir=config_path, version_base=None)
    
    # Use Hydra's compose to load defaults and apply overrides
    # Always use base model for tokenizer (checkpoints don't have tokenizer files)
    tokenizer_path = DEFAULT_MODEL_PATH
    
    overrides = [
        # Model path
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=true",
        
        # Tokenizer path (always use base model, checkpoints don't have tokenizer)
        f"data.tokenizer={tokenizer_path}",
        f"reward_model.model.input_tokenizer={tokenizer_path}",
        f"critic.model.tokenizer_path={tokenizer_path}",
        
        # Rollout config for GRPO - using vLLM
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=4",  # Multiple generations for GRPO
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",  # Leave room for FSDP actor
        "actor_rollout_ref.rollout.prompt_length=2560",
        "actor_rollout_ref.rollout.response_length=8192",
        "actor_rollout_ref.rollout.max_model_len=10752",  # 2560 + 8192
        "actor_rollout_ref.rollout.max_num_batched_tokens=10752",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",  # Reduced from 4 to save GPU memory
        "actor_rollout_ref.rollout.load_format=auto",
        "actor_rollout_ref.rollout.enforce_eager=true",
        
        # Validation rollout config - 2 samples per prompt for @2 metrics
        "actor_rollout_ref.rollout.val_kwargs.n=2",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        
        # Ray cluster config - connect to existing cluster
        "++ray_kwargs.ray_init.address=auto",
        
        # Actor config
        f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_BATCH_SIZE}",  # Must be <= train_batch_size
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",  # Reduced from 4 to save GPU memory
        "actor_rollout_ref.actor.ppo_epochs=1",
        
        # Algorithm - GRPO
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.norm_adv_by_std_in_grpo=true",
        
        # Data config
        f"data.train_files={train_dataset_file}",
        f"data.val_files={val_dataset_file}",
        "data.prompt_key=prompt",
        "data.max_prompt_length=2560",
        "data.max_response_length=8192",
        f"data.train_batch_size={TRAIN_BATCH_SIZE}",
        f"data.val_batch_size={TEST_BATCH_SIZE}",
        "data.return_raw_chat=true",  # Required for sglang rollout
        
        # Trainer config
        f"trainer.project_name=grpo-omni-math",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.nnodes={NUM_NODES}",
        f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.total_epochs={TOTAL_EPOCHS}",
        "trainer.save_freq=50",  # Save checkpoint every N steps
        "trainer.val_before_train=true",  # Skip validation before training (too slow)
        "trainer.test_freq=5",  # Validate every 50 training steps
        "trainer.log_val_generations=3",  # Log N validation samples to wandb
        
        # Custom reward function (use ++ to add or override)
        f"++custom_reward_function.path={__file__}",
        "++custom_reward_function.name=compute_score",
        
        # Disable reward model loop (use custom function)
        "++reward_model.enable=false",
        
        # Disable critic (not needed for GRPO)
        "++critic.enable=false",
        
        # Custom script parameters (for wandb logging)
        f"++custom_params.hint_level={hint_level}",
        f"++custom_params.system_prompt_name={SYSTEM_PROMPT_NAME}",
        f"++custom_params.max_samples={MAX_NUM}",
        f"++custom_params.dataset_path={DATASET_NAME}",
        f"++custom_params.val_size={VAL_SIZE}",
        f"++custom_params.model_path={model_path}",
        f"++custom_params.resumed_from_checkpoint={args.resume_from is not None}",
    ]

    if PROMPT_UPDATE_ENABLED:
        interaction_config_path = write_interaction_config()
        overrides.extend([
            "actor_rollout_ref.rollout.name=sglang",
            "actor_rollout_ref.rollout.multi_turn.enable=true",
            f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_config_path}",
            f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={PROMPT_UPDATE_MAX_ASSISTANT_TURNS}",
            f"actor_rollout_ref.rollout.multi_turn.max_user_turns={PROMPT_UPDATE_MAX_USER_TURNS}",
        ])
    
    # Add resume_from_path if resuming from checkpoint
    if args.resume_from:
        # Validate that the path contains global_step_ (required by verl)
        if "global_step_" not in args.resume_from:
            print(f"[ERROR] --resume-from path must contain 'global_step_X' directory")
            print(f"        Given: {args.resume_from}")
            print(f"        Example: {args.resume_from}/global_step_50")
            sys.exit(1)
        
        # Initialize Ray early to distribute checkpoint to all nodes
        print("[INFO] Initializing Ray for checkpoint distribution...")
        # Runtime env ensures workers can import sbys_hinting module
        ray_runtime_env = {"env_vars": {"PYTHONPATH": _parent_dir}}
        if not ray.is_initialized():
            try:
                ray.init(address='auto', ignore_reinit_error=True, runtime_env=ray_runtime_env)
            except ConnectionError:
                print("[INFO] Could not connect to existing Ray cluster, starting new one...")
                ray.init(ignore_reinit_error=True, runtime_env=ray_runtime_env)
        
        # Distribute checkpoint from head node to all worker nodes
        # This is necessary because verl expects ALL shards to be available on each node
        print(f"[INFO] Distributing checkpoint to all nodes...")
        
        success = distribute_checkpoint_to_all_nodes(
            checkpoint_dir=args.resume_from,
            target_dir=args.resume_from  # Same path on all nodes
        )
        
        if not success:
            print("[ERROR] Failed to distribute checkpoint to all nodes")
            sys.exit(1)
        
        # Must set resume_mode to 'resume_path' for verl to use the checkpoint
        overrides.append("trainer.resume_mode=resume_path")
        overrides.append(f"trainer.resume_from_path={args.resume_from}")
        
        # Extract step number from checkpoint path and set total_training_steps
        # This is needed because verl calculates total_training_steps = dataset_size * epochs
        # and training exits when global_steps >= total_training_steps
        step_match = re.search(r'global_step_(\d+)', args.resume_from)
        if step_match:
            resume_step = int(step_match.group(1))
            # When resuming, add another full training cycle to the resume step
            new_total = resume_step + total_training_steps
            print(f"[INFO] Resuming from step {resume_step}, adding {total_training_steps} more steps")
            print(f"[INFO] New total_training_steps: {new_total}")
            overrides.append(f"trainer.total_training_steps={new_total}")
        
        print(f"[INFO] Will resume from checkpoint: {args.resume_from}")
    
    try:
        config = compose(config_name="ppo_trainer", overrides=overrides)
    except Exception as e:
        print(f"Error composing config: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Configuration loaded successfully")
    print(f"Config summary:")
    print(f"  - Model: {config.actor_rollout_ref.model.path}")
    print(f"  - Adv estimator: {config.algorithm.adv_estimator}")
    print(f"  - Train batch size: {config.data.train_batch_size}")
    print(f"  - Nodes: {config.trainer.nnodes}, GPUs/node: {config.trainer.n_gpus_per_node}")

    if PROMPT_UPDATE_ENABLED:
        install_prompt_reset_hook()
        
        # Initialize Ray if not already initialized (needed for ProblemStateActor)
        if not ray.is_initialized():
            try:
                ray.init(address='auto', ignore_reinit_error=True)
            except ConnectionError:
                print("[INFO] Could not connect to existing Ray cluster, starting new one...")
                ray.init(ignore_reinit_error=True)
        
        # Create the shared ProblemStateActor before training starts
        # This ensures all workers can access the same actor instance
        print("[INFO] Initializing shared ProblemStateActor for persistent problem state...")
        state_actor = get_or_create_problem_state_actor()
        print(f"[INFO] ProblemStateActor ready: {state_actor}")
        
        # Initialize all problem states at once (more efficient than initializing per-call)
        all_problems = []
        for item in train_data:
            extra_info = item.get("extra_info", {})
            interaction_kwargs = extra_info.get("interaction_kwargs", {})
            if interaction_kwargs.get("problem"):
                all_problems.append({
                    "problem": interaction_kwargs["problem"],
                    "sbys_solution": interaction_kwargs.get("sbys_solution", []),
                })
        print(f"[INFO] Initializing state for {len(all_problems)} problems...")
        num_initialized = ray.get(state_actor.initialize_all_problems.remote(all_problems))
        print(f"[INFO] Initialized {num_initialized} problem states")
    
    # Import run_ppo here to avoid import issues
    from verl.trainer.main_ppo import run_ppo
    
    # Run PPO training
    print("Starting GRPO training with verl...")
    try:
        run_ppo(config)
        print("Training completed successfully!")
        
        # After training, gather checkpoint shards from all nodes to head node
        print("\n" + "=" * 80)
        print("Gathering checkpoint shards to head node...")
        print("=" * 80)
        
        # Find the latest checkpoint
        latest_checkpoint_file = os.path.join(output_dir, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_checkpoint_file):
            with open(latest_checkpoint_file, 'r') as f:
                latest_step = f.read().strip()
            checkpoint_dir = os.path.join(output_dir, f"global_step_{latest_step}")
            
            # Gather to a consolidated directory on head node
            gathered_dir = os.path.join(output_dir, "gathered_checkpoint", f"global_step_{latest_step}")
            
            print(f"Found latest checkpoint at step {latest_step}")
            
            # Gather all shards to head node (keeps verl's native format)
            success = gather_checkpoint_to_head_node(
                checkpoint_dir=checkpoint_dir,
                output_dir=gathered_dir,
                num_nodes=NUM_NODES,
                gpus_per_node=GPUS_PER_NODE
            )
            
            if success:
                print(f"\n✓ Complete checkpoint gathered to: {gathered_dir}")
                print(f"  Resume training with: python hinting2_grpo_ray.py --resume-from {gathered_dir}")
            else:
                print("\n✗ Checkpoint gathering incomplete - some shards may be missing")
                print("  The partial checkpoint is still at the gathered location.")
        else:
            print(f"[WARNING] No checkpoint found at {latest_checkpoint_file}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if os.path.exists(train_dataset_file):
            os.unlink(train_dataset_file)
            print(f"Cleaned up temporary train dataset file: {train_dataset_file}")
        if os.path.exists(val_dataset_file):
            os.unlink(val_dataset_file)
            print(f"Cleaned up temporary val dataset file: {val_dataset_file}")


if __name__ == "__main__":
    main()
