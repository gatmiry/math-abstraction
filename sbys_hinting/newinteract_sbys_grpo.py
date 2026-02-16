#!/usr/bin/env python3
"""
Simplified GRPO training script for Omni-MATH dataset using verl with Ray.
Single-turn only: no hints, no prompt updates, no multi-turn interaction.
Prompts are fixed throughout the entire RL run.
"""

import os
import sys
import json
import argparse
import time
import uuid
import random
import socket
import traceback
import threading
import signal
import atexit
import datetime
import tempfile
import re

import numpy as np
import apple_bolt as bolt

# Add parent directory to PYTHONPATH so 'sbys_hinting' module can be imported by verl workers
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if "PYTHONPATH" in os.environ:
    if _parent_dir not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = f"{_parent_dir}:{os.environ['PYTHONPATH']}"
else:
    os.environ["PYTHONPATH"] = _parent_dir

# vLLM V1 engine compatibility
os.environ.setdefault("VLLM_USE_V1", "0")

# Use shared HF cache so all worker nodes can access cached models
os.environ.setdefault("HF_HOME", "/mnt/task_runtime/.hf_cache")

# Tell wandb to save code with each run
os.environ["WANDB_SAVE_CODE"] = "true"

# HuggingFace timeout settings
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

# Ray gRPC and health check timeouts - MUST be set BEFORE ray.init()
os.environ["RAY_grpc_keepalive_time_ms"] = "60000"
os.environ["RAY_grpc_keepalive_timeout_ms"] = "600000"
os.environ["RAY_health_check_initial_delay_ms"] = "60000"
os.environ["RAY_health_check_period_ms"] = "60000"
os.environ["RAY_health_check_timeout_ms"] = "600000"
os.environ["RAY_health_check_failure_threshold"] = "10"
print("[INFO] Set Ray keepalive/health check timeouts (10 min) for slow model init")

# NCCL P2P workaround for hardware issues
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Token file path
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "hf_token.txt")
if not os.path.exists(TOKEN_FILE):
    TOKEN_FILE = os.path.join(os.path.dirname(__file__), "..", "hinting", "hf_token.txt")

# System prompt file path
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt_full_solution.txt")
SYSTEM_PROMPT_NAME = "full_solution_simple"

# Training config
TRAIN_BATCH_SIZE = 32#128
TOTAL_EPOCHS = 50
VAL_SIZE = 32#128
NUM_NODES = 1
GPUS_PER_NODE = 8

# Model & dataset
DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = os.path.join(os.path.dirname(__file__), "outputs", "hint_helped_dataset", "hint_helped_dataset")
MAX_NUM = None  # Limit dataset to last MAX_NUM rows (None = use all)

# =============================================================================
# IMPORTS
# =============================================================================
import ray
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import load_from_disk
from typing import List, Dict, Optional, Any
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Import math_checker for reward computation
import importlib.util
_math_checker_path = os.path.join(os.path.dirname(__file__), "math_checker.py")
_spec = importlib.util.spec_from_file_location("math_checker", _math_checker_path)
_math_checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math_checker)
check_answer = _math_checker.check_answer
extract_boxed_answer = _math_checker.extract_boxed_answer


# =============================================================================
# CENTRALIZED CRASH LOGGING & HEARTBEAT SYSTEM
# =============================================================================
_CRASH_LOG_DIR = os.path.join(os.path.dirname(__file__), "crash_logs")
_CRASH_LOG_LOCK = threading.Lock()
_PROCESS_START_TIME = datetime.datetime.now()
_HEARTBEAT_THREAD = None
_HEARTBEAT_STOP = threading.Event()
_WORKER_STATE = {"phase": "init", "iteration": 0, "last_activity": None}


def get_node_identifier():
    """Get a unique identifier for this node."""
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "unknown"
    pid = os.getpid()
    return f"{hostname}_{ip}_{pid}"


def get_detailed_system_info():
    """Get detailed system info for crash diagnosis."""
    info = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cwd": os.getcwd(),
    }
    try:
        info["ip"] = socket.gethostbyname(socket.gethostname())
    except Exception:
        info["ip"] = "unknown"
    try:
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_current_device"] = torch.cuda.current_device()
            info["gpu_memory"] = []
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                info["gpu_memory"].append({
                    "device": i,
                    "allocated_gb": round(mem_allocated, 2),
                    "reserved_gb": round(mem_reserved, 2),
                    "total_gb": round(mem_total, 2),
                    "free_gb": round(mem_total - mem_reserved, 2),
                })
    except Exception as e:
        info["gpu_error"] = str(e)
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        info["memory_mb"] = rusage.ru_maxrss / 1024
    except Exception:
        pass
    info["thread_count"] = threading.active_count()
    info["thread_names"] = [t.name for t in threading.enumerate()]
    info["worker_state"] = dict(_WORKER_STATE)
    return info


def update_worker_state(phase: str = None, iteration: int = None, **kwargs):
    """Update the current worker state for crash diagnosis."""
    global _WORKER_STATE
    now = datetime.datetime.now()
    if phase is not None:
        old_phase = _WORKER_STATE.get("phase")
        if old_phase != phase:
            phase_start = _WORKER_STATE.get("_phase_start_time")
            if phase_start and old_phase:
                phase_duration = (now - datetime.datetime.fromisoformat(phase_start)).total_seconds()
                if phase_duration > 60:
                    print(f"[STUCK DETECTION] Phase '{old_phase}' took {phase_duration:.1f}s (>60s) before transitioning to '{phase}'")
            _WORKER_STATE["_prev_phase"] = old_phase
            _WORKER_STATE["_phase_start_time"] = now.isoformat()
        _WORKER_STATE["phase"] = phase
    if iteration is not None:
        _WORKER_STATE["iteration"] = iteration
    _WORKER_STATE["last_activity"] = now.isoformat()
    _WORKER_STATE.update(kwargs)


def log_crash(error_type: str, error_msg: str, stack_trace: str = None, extra_info: dict = None):
    """Log a crash/error to the shared crash log directory."""
    try:
        os.makedirs(_CRASH_LOG_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        node_id = get_node_identifier()
        system_info = get_detailed_system_info()
        log_entry = {
            "timestamp": timestamp,
            "node_id": node_id,
            "error_type": error_type,
            "error_msg": str(error_msg)[:2000],
            "stack_trace": (stack_trace or traceback.format_exc())[:10000],
            "extra_info": extra_info or {},
            "system_info": system_info,
            "uptime_seconds": (datetime.datetime.now() - _PROCESS_START_TIME).total_seconds(),
        }
        combined_log = os.path.join(_CRASH_LOG_DIR, "all_crashes.jsonl")
        with _CRASH_LOG_LOCK:
            with open(combined_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        node_log = os.path.join(_CRASH_LOG_DIR, f"crashes_{node_id}.log")
        with open(node_log, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{timestamp}] {error_type}\n")
            f.write(f"Node: {node_id}\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"Worker State: {_WORKER_STATE}\n")
            if stack_trace:
                f.write(f"Stack trace:\n{stack_trace}\n")
            if system_info.get("gpu_memory"):
                f.write(f"GPU Memory:\n")
                for gpu in system_info["gpu_memory"]:
                    f.write(f"  GPU {gpu['device']}: {gpu['allocated_gb']:.2f}GB allocated, "
                            f"{gpu['free_gb']:.2f}GB free / {gpu['total_gb']:.2f}GB total\n")
        print(f"[CRASH_LOG] Logged {error_type} to {combined_log}", flush=True)
    except Exception as e:
        print(f"[CRASH_LOG] Failed to write crash log: {e}", flush=True)


def log_heartbeat():
    """Log a heartbeat to show this worker is alive."""
    try:
        os.makedirs(_CRASH_LOG_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        node_id = get_node_identifier()
        system_info = get_detailed_system_info()
        heartbeat_entry = {
            "timestamp": timestamp,
            "node_id": node_id,
            "type": "HEARTBEAT",
            "uptime_seconds": (datetime.datetime.now() - _PROCESS_START_TIME).total_seconds(),
            "worker_state": dict(_WORKER_STATE),
            "gpu_memory": system_info.get("gpu_memory"),
            "thread_count": system_info.get("thread_count"),
        }
        heartbeat_log = os.path.join(_CRASH_LOG_DIR, "heartbeats.jsonl")
        with _CRASH_LOG_LOCK:
            with open(heartbeat_log, "a") as f:
                f.write(json.dumps(heartbeat_entry) + "\n")
    except Exception:
        pass


def start_heartbeat_thread(interval_seconds=15):
    """Start a background thread that logs heartbeats with stuck detection."""
    global _HEARTBEAT_THREAD, _HEARTBEAT_STOP

    def heartbeat_loop():
        while not _HEARTBEAT_STOP.is_set():
            log_heartbeat()
            # Stuck detection
            last_activity_str = _WORKER_STATE.get("last_activity")
            current_phase = _WORKER_STATE.get("phase", "unknown")
            if last_activity_str:
                try:
                    last_activity = datetime.datetime.fromisoformat(last_activity_str)
                    idle_time = (datetime.datetime.now() - last_activity).total_seconds()
                    if idle_time > 120:
                        print(f"[STUCK ALERT] Worker idle for {idle_time:.0f}s in phase '{current_phase}'!")
                        print(f"[STUCK ALERT] Worker state: {_WORKER_STATE}")
                        if idle_time > 300:
                            print(f"[STUCK CRITICAL] Worker CRITICALLY idle for {idle_time:.0f}s!")
                except Exception:
                    pass
            _HEARTBEAT_STOP.wait(interval_seconds)

    _HEARTBEAT_STOP.clear()
    _HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True, name="HeartbeatThread")
    _HEARTBEAT_THREAD.start()
    print(f"[INFO] Started heartbeat thread (interval={interval_seconds}s)", flush=True)


def install_global_exception_handler():
    """Install a global exception handler that logs uncaught exceptions."""
    original_excepthook = sys.excepthook

    def custom_excepthook(exc_type, exc_value, exc_tb):
        stack = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log_crash(
            error_type=f"UNCAUGHT_EXCEPTION:{exc_type.__name__}",
            error_msg=str(exc_value),
            stack_trace=stack,
            extra_info={"exc_type": str(exc_type)}
        )
        original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = custom_excepthook
    print(f"[INFO] Installed global exception handler, crash logs: {_CRASH_LOG_DIR}", flush=True)


def install_signal_handlers():
    """Install signal handlers to log when process receives termination signals."""
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        stack = "".join(traceback.format_stack(frame)) if frame else "No stack available"
        log_crash(
            error_type=f"SIGNAL_RECEIVED:{sig_name}",
            error_msg=f"Process received signal {signum} ({sig_name})",
            stack_trace=stack,
            extra_info={"signal_number": signum, "signal_name": sig_name}
        )
        print(f"[SIGNAL] Received {sig_name}, logged to crash_logs", flush=True)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGABRT]:
        try:
            signal.signal(sig, signal_handler)
        except Exception as e:
            print(f"[WARN] Could not install handler for {sig}: {e}", flush=True)
    print(f"[INFO] Installed signal handlers for SIGTERM, SIGINT, SIGABRT", flush=True)


def install_atexit_handler():
    """Install atexit handler to log when process exits."""
    def atexit_handler():
        log_crash(
            error_type="PROCESS_EXIT",
            error_msg="Process is exiting",
            stack_trace="".join(traceback.format_stack()),
            extra_info={"exit_type": "atexit"}
        )
        print(f"[ATEXIT] Process exiting, logged to crash_logs", flush=True)
    atexit.register(atexit_handler)
    print(f"[INFO] Installed atexit handler", flush=True)


# Install all handlers immediately when module is imported
install_global_exception_handler()
install_signal_handlers()
install_atexit_handler()


# =============================================================================
# TOKEN & LOGIN HELPERS
# =============================================================================

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
    os.environ["HF_TOKEN"] = token
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
    os.environ["WANDB_API_KEY"] = api_key
    try:
        import wandb
        wandb.login(key=api_key, relogin=True)
        print("[INFO] Successfully logged in to Wandb")
        return True
    except Exception as e:
        print(f"[WARNING] Wandb login failed: {e}")
        return False


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def load_system_prompt(name: str = None):
    """Load a named system prompt from file."""
    if name is None:
        name = SYSTEM_PROMPT_NAME
    with open(SYSTEM_PROMPT_FILE, 'r') as f:
        content = f.read()
    prompts = {}
    current_name = None
    current_lines = []
    for line in content.split('\n'):
        if line.startswith('===PROMPT:') and line.endswith('==='):
            if current_name is not None:
                prompts[current_name] = '\n'.join(current_lines).strip()
            current_name = line[10:-3].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_name is not None:
        prompts[current_name] = '\n'.join(current_lines).strip()
    if name not in prompts:
        available = list(prompts.keys())
        raise ValueError(f"System prompt '{name}' not found. Available: {available}")
    return prompts[name]


# =============================================================================
# REWARD FUNCTION
# =============================================================================

_compute_score_log_counter = 0
_compute_score_log_limit = 5


def compute_score(
    data_source: str = None,
    solution_str: str = None,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """Compute reward score for a single solution.

    Uses math_verify library for answer verification.
    Returns 1.0 if correct, 0.0 otherwise.
    """
    global _compute_score_log_counter

    if _compute_score_log_counter < _compute_score_log_limit:
        _compute_score_log_counter += 1
        print(f"\n[compute_score #{_compute_score_log_counter}] "
              f"ground_truth={repr(ground_truth[:80] if ground_truth else ground_truth)}... "
              f"solution_str={repr(solution_str[:200] if solution_str else solution_str)}...")

    if ground_truth is None or solution_str is None:
        return 0.0

    is_correct = check_answer(solution_str, ground_truth)
    boxed_answer = extract_boxed_answer(solution_str)

    if random.random() < 0.02:
        print(f"[compute_score] ground_truth={ground_truth[:80]}... "
              f"boxed={boxed_answer[:80] if boxed_answer else None}... "
              f"correct={is_correct}")

    return 1.0 if is_correct else 0.0


# =============================================================================
# DATASET CREATION
# =============================================================================

def format_prompt(problem: str) -> List[Dict[str, str]]:
    """Format a problem into chat messages (system + user)."""
    system_prompt = load_system_prompt(SYSTEM_PROMPT_NAME)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {problem}"},
    ]


def create_rl_dataset(tokenizer, dataset_path: str, max_samples: Optional[int] = None,
                       val_size: int = 128, max_prompt_tokens: int = 4096):
    """Create RL dataset in verl format with train/val split. No hints."""

    dataset = load_from_disk(dataset_path)

    # Deduplicate by problem text
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
        print(f"[INFO] Deduplicated dataset: {original_size} -> {len(dataset)}")

    def format_dataset(examples):
        prompts, answers = [], []
        for problem, answer in zip(examples['problem'], examples['answer']):
            if answer:
                prompts.append(format_prompt(problem))
                answers.append(answer)
        return {"prompt": prompts, "answer": answers}

    def is_prompt_short_enough(example):
        prompt_text = tokenizer.apply_chat_template(example['prompt'], tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(prompt_text)) <= max_prompt_tokens

    # Limit dataset size
    if max_samples is not None and max_samples > 0:
        start_idx = max(0, len(dataset) - max_samples)
        dataset = dataset.select(range(start_idx, len(dataset)))
        print(f"[INFO] Limited dataset to last {max_samples} rows: {len(dataset)} total")

    # Split into train/val
    total_size = len(dataset)
    val_size_actual = min(val_size, total_size)
    indices = list(range(total_size))
    random.Random(42).shuffle(indices)
    val_indices = indices[:val_size_actual]
    train_indices = indices[val_size_actual:]
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    print(f"[INFO] Split: {len(train_dataset)} train, {len(val_dataset)} val")

    # Format
    train_dataset = train_dataset.map(format_dataset, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_dataset, batched=True, remove_columns=val_dataset.column_names)

    # Filter long prompts
    orig = len(train_dataset)
    train_dataset = train_dataset.filter(is_prompt_short_enough)
    if orig - len(train_dataset) > 0:
        print(f"[INFO] Filtered {orig - len(train_dataset)} train samples > {max_prompt_tokens} tokens")
    orig = len(val_dataset)
    val_dataset = val_dataset.filter(is_prompt_short_enough)
    if orig - len(val_dataset) > 0:
        print(f"[INFO] Filtered {orig - len(val_dataset)} val samples > {max_prompt_tokens} tokens")

    # Convert to verl format
    def to_verl_format(dataset_split):
        rl_data = []
        for item in dataset_split:
            rl_data.append({
                "prompt": item["prompt"],
                "ground_truth": item["answer"],
                "reward_model": {"ground_truth": item["answer"]},
                "data_source": "omni_math",
            })
        return rl_data

    return to_verl_format(train_dataset), to_verl_format(val_dataset)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_verl_config_path():
    """Get path to verl's config directory."""
    import verl
    return os.path.join(os.path.dirname(verl.__file__), "trainer", "config")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple GRPO Training (no hints, single-turn)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to a local HuggingFace model. If not provided, uses default.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a verl checkpoint directory to resume from (e.g. /path/global_step_50)")
    return parser.parse_args()


def distribute_file_to_all_nodes(file_path: str):
    """Distribute a file to all nodes in the Ray cluster."""
    with open(file_path, 'rb') as f:
        content = f.read()
    files = [(content, file_path, os.path.basename(file_path))]

    # Also distribute math_checker.py
    script_dir = os.path.dirname(file_path)
    math_checker = os.path.join(script_dir, "math_checker.py")
    if os.path.exists(math_checker):
        with open(math_checker, 'rb') as f:
            files.append((f.read(), math_checker, "math_checker.py"))
    # Also distribute system prompt
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, 'rb') as f:
            files.append((f.read(), SYSTEM_PROMPT_FILE, os.path.basename(SYSTEM_PROMPT_FILE)))

    @ray.remote
    def write_files(files_list):
        import os, socket
        results = []
        for content, target, name in files_list:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'wb') as f:
                f.write(content)
            results.append(name)
        return f"{socket.gethostname()}: wrote {', '.join(results)}"

    nodes = ray.nodes()
    node_ips = [n["NodeManagerAddress"] for n in nodes if n["Alive"]]
    print(f"[INFO] Distributing files to {len(node_ips)} nodes")
    tasks = []
    for ip in node_ips:
        tasks.append((ip, write_files.options(resources={f"node:{ip}": 0.001}).remote(files)))
    for ip, task in tasks:
        try:
            result = ray.get(task, timeout=60)
            print(f"[INFO] {result}")
        except Exception as e:
            print(f"[ERROR] Failed to distribute to {ip}: {e}")


def distribute_checkpoint_to_all_nodes(checkpoint_dir: str, target_dir: str):
    """Distribute checkpoint shards from head node to all worker nodes."""
    if not os.path.exists(checkpoint_dir):
        print(f"[ERROR] Checkpoint dir not found: {checkpoint_dir}")
        return False

    # Collect all checkpoint files
    all_files = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, checkpoint_dir)
            with open(full_path, 'rb') as f:
                all_files.append((f.read(), rel_path))
    print(f"[INFO] Read {len(all_files)} checkpoint files")

    @ray.remote
    def write_checkpoint(files_data, base_dir):
        import os, socket
        written = 0
        for content, rel_path in files_data:
            target = os.path.join(base_dir, rel_path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'wb') as f:
                f.write(content)
            written += 1
        return f"{socket.gethostname()}: wrote {written} files"

    nodes = ray.nodes()
    node_ips = [n["NodeManagerAddress"] for n in nodes if n["Alive"]]
    tasks = []
    for ip in node_ips:
        tasks.append((ip, write_checkpoint.options(
            resources={f"node:{ip}": 0.001}
        ).remote(all_files, target_dir)))

    success = True
    for ip, task in tasks:
        try:
            result = ray.get(task, timeout=300)
            print(f"[INFO] {result}")
        except Exception as e:
            print(f"[ERROR] Checkpoint distribution failed on {ip}: {e}")
            success = False
    return success


def gather_checkpoint_to_head_node(checkpoint_dir: str, output_dir: str,
                                    num_nodes: int, gpus_per_node: int):
    """Gather checkpoint shards from all nodes to head node."""
    os.makedirs(output_dir, exist_ok=True)

    @ray.remote
    def read_checkpoint(base_dir):
        import os, socket
        files = []
        if not os.path.exists(base_dir):
            return socket.gethostname(), []
        for root, dirs, fnames in os.walk(base_dir):
            for fname in fnames:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, base_dir)
                with open(full_path, 'rb') as f:
                    files.append((rel_path, f.read()))
        return socket.gethostname(), files

    nodes = ray.nodes()
    node_ips = [n["NodeManagerAddress"] for n in nodes if n["Alive"]]
    tasks = []
    for ip in node_ips:
        tasks.append((ip, read_checkpoint.options(
            resources={f"node:{ip}": 0.001}
        ).remote(checkpoint_dir)))

    success = True
    for ip, task in tasks:
        try:
            hostname, files = ray.get(task, timeout=300)
            for rel_path, content in files:
                target = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, 'wb') as f:
                    f.write(content)
            print(f"[INFO] Gathered {len(files)} files from {hostname}")
        except Exception as e:
            print(f"[ERROR] Gather failed from {ip}: {e}")
            success = False
    return success


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run single-turn GRPO training with verl and Ray."""
    start_heartbeat_thread(interval_seconds=15)
    update_worker_state(phase="main_init", iteration=0)

    # Set distributed env vars
    distributed_env_vars = {
        "NCCL_TIMEOUT": "3600",
        "NCCL_DEBUG": "INFO",
        "NCCL_SOCKET_NTHREADS": "4",
        "NCCL_NSOCKS_PERTHREAD": "4",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IGNORE_DISABLED_P2P": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "GLOO_SOCKET_TIMEOUT": "3600000",
        "GLOO_TIMEOUT_MS": "2700000",
        "TORCH_DISTRIBUTED_TIMEOUT": "2700",
        "TORCH_DISTRIBUTED_DEBUG": "OFF",
        "RAY_grpc_keepalive_time_ms": "60000",
        "RAY_grpc_keepalive_timeout_ms": "600000",
        "RAY_health_check_initial_delay_ms": "60000",
        "RAY_health_check_period_ms": "60000",
        "RAY_health_check_timeout_ms": "600000",
        "RAY_health_check_failure_threshold": "10",
    }
    for key, value in distributed_env_vars.items():
        os.environ[key] = value
    print("[INFO] Set NCCL/Gloo/Ray environment variables")
    update_worker_state(phase="nccl_env_set")

    args = parse_args()
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_PATH

    # Experiment naming
    if args.resume_from:
        path_parts = args.resume_from.rstrip('/').split('/')
        base_model_name = next((p for p in reversed(path_parts) if p and not p.startswith('global_step')), "checkpoint")
    elif args.model_path:
        base_model_name = args.model_path.rstrip('/').split('/')[-1].lower()
    else:
        base_model_name = model_path.split('/')[-1].lower()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"grpo_simple_no_hints_{timestamp}"
    model_name = f"qwen3-4b-instruct-2507_simple_{timestamp}_from_{base_model_name}"
    output_dir = os.path.join(bolt.ARTIFACT_DIR, model_name)

    print("=" * 80)
    print("Simple GRPO Training (no hints, single-turn)")
    print("=" * 80)
    print(f"Model: {model_path}")
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {output_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Nodes: {NUM_NODES}, GPUs/node: {GPUS_PER_NODE}")
    print("=" * 80)

    # Load tokens and login
    tokens = load_tokens()
    login_huggingface(tokens)
    login_wandb(tokens)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    print("Creating RL dataset...")
    train_data, val_data = create_rl_dataset(tokenizer, DATASET_NAME, max_samples=MAX_NUM, val_size=VAL_SIZE)
    print(f"Created {len(train_data)} training, {len(val_data)} validation samples")

    # Calculate total training steps
    num_batches_per_epoch = len(train_data) // TRAIN_BATCH_SIZE
    total_training_steps = num_batches_per_epoch * TOTAL_EPOCHS
    print(f"Total training steps: {num_batches_per_epoch} batches/epoch x {TOTAL_EPOCHS} epochs = {total_training_steps}")

    # Save datasets to parquet
    print("Saving datasets to parquet...")
    df_train = pd.DataFrame(train_data)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='_train.parquet', delete=False, dir='/mnt/tmp') as f:
        df_train.to_parquet(f.name, index=False)
        train_dataset_file = f.name
    df_val = pd.DataFrame(val_data)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='_val.parquet', delete=False, dir='/mnt/tmp') as f:
        df_val.to_parquet(f.name, index=False)
        val_dataset_file = f.name
    print(f"Saved: {train_dataset_file}, {val_dataset_file}")

    # Load verl config
    print("Loading verl configuration...")
    config_path = get_verl_config_path()
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_path, version_base=None)
    tokenizer_path = DEFAULT_MODEL_PATH

    overrides = [
        # Model
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=true",
        # Tokenizer
        f"data.tokenizer={tokenizer_path}",
        f"critic.model.tokenizer_path={tokenizer_path}",
        # Rollout - sglang (vllm has import compatibility issues with installed version)
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.prompt_length=4096",
        "actor_rollout_ref.rollout.response_length=16384",
        "actor_rollout_ref.rollout.max_model_len=20480",
        "actor_rollout_ref.rollout.max_num_batched_tokens=1310720",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.load_format=auto",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        # B200 GPUs (SM 100) don't support FlashAttention v3 (SM 80-90 only), use flashinfer instead
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer",
        # Validation rollout
        "actor_rollout_ref.rollout.val_kwargs.n=2",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        # Ray cluster
        "++ray_kwargs.ray_init.address=auto",
        # Actor
        f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_BATCH_SIZE}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.ppo_epochs=2",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=false",
        "actor_rollout_ref.actor.fsdp_config.param_offload=true",
        "actor_rollout_ref.ref.entropy_from_logits_with_chunking=false",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        # Algorithm - GRPO
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.norm_adv_by_std_in_grpo=true",
        # Data
        f"data.train_files={train_dataset_file}",
        f"data.val_files={val_dataset_file}",
        "data.prompt_key=prompt",
        "data.max_prompt_length=4096",
        "data.max_response_length=16384",
        f"data.train_batch_size={TRAIN_BATCH_SIZE}",
        f"data.val_batch_size={VAL_SIZE}",
        "data.return_raw_chat=true",
        # Custom dataset class with on_batch_end hook for dynamic hint adjustment
        f"data.custom_cls.path={os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hint_dataset.py')}",
        "data.custom_cls.name=HintDataset",
        "data.dataloader_num_workers=0",
        # Trainer
        f"trainer.project_name=grpo-omni-math-simple",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.nnodes={NUM_NODES}",
        f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.total_epochs={TOTAL_EPOCHS}",
        "trainer.save_freq=15",
        "+trainer.save_checkpoint_before_train=true",
        "trainer.val_before_train=false",
        "trainer.test_freq=5",
        "trainer.log_val_generations=3",
        # Reward (new verl uses reward.custom_reward_function / reward.reward_model)
        f"reward.custom_reward_function.path={__file__}",
        "reward.custom_reward_function.name=compute_score",
        "reward.reward_model.enable=false",
        "++critic.enable=false",
        # Custom params for wandb
        f"++custom_params.model_path={model_path}",
        f"++custom_params.dataset_path={DATASET_NAME}",
        f"++custom_params.val_size={VAL_SIZE}",
        f"++custom_params.max_samples={MAX_NUM}",
        f"++custom_params.mode=simple_no_hints",
    ]

    # Handle checkpoint resume
    if args.resume_from:
        if "global_step_" not in args.resume_from:
            print(f"[ERROR] --resume-from must contain 'global_step_X' directory")
            sys.exit(1)

        print("[INFO] Initializing Ray for checkpoint distribution...")
        ray_runtime_env = {"env_vars": {"PYTHONPATH": _parent_dir}}
        _wandb_key = tokens.get("WANDB_API_KEY", "")
        if _wandb_key and _wandb_key != "YOUR_WANDB_API_KEY_HERE":
            ray_runtime_env["env_vars"]["WANDB_API_KEY"] = _wandb_key
        if not ray.is_initialized():
            try:
                ray.init(address='auto', ignore_reinit_error=True, runtime_env=ray_runtime_env)
            except ConnectionError:
                ray.init(ignore_reinit_error=True, runtime_env=ray_runtime_env)

        print(f"[INFO] Distributing checkpoint to all nodes...")
        if not distribute_checkpoint_to_all_nodes(args.resume_from, args.resume_from):
            print("[ERROR] Failed to distribute checkpoint")
            sys.exit(1)

        overrides.append("trainer.resume_mode=resume_path")
        overrides.append(f"trainer.resume_from_path={args.resume_from}")
        step_match = re.search(r'global_step_(\d+)', args.resume_from)
        if step_match:
            resume_step = int(step_match.group(1))
            new_total = resume_step + total_training_steps
            print(f"[INFO] Resuming from step {resume_step}, new total_training_steps: {new_total}")
            overrides.append(f"trainer.total_training_steps={new_total}")

    # Compose config
    try:
        config = compose(config_name="ppo_trainer", overrides=overrides)
    except Exception as e:
        print(f"Error composing config: {e}")
        traceback.print_exc()
        raise

    print("Configuration loaded successfully")
    print(f"  Model: {config.actor_rollout_ref.model.path}")
    print(f"  Train batch: {config.data.train_batch_size}")
    print(f"  Nodes: {config.trainer.nnodes}, GPUs/node: {config.trainer.n_gpus_per_node}")

    # Initialize Ray (if not already from resume path)
    if not ray.is_initialized():
        ray_runtime_env = {"env_vars": {"PYTHONPATH": _parent_dir}}
        _wandb_key = tokens.get("WANDB_API_KEY", "")
        if _wandb_key and _wandb_key != "YOUR_WANDB_API_KEY_HERE":
            ray_runtime_env["env_vars"]["WANDB_API_KEY"] = _wandb_key
            print("[INFO] WANDB_API_KEY added to Ray runtime_env")
        try:
            ray.init(address='auto', ignore_reinit_error=True, runtime_env=ray_runtime_env)
        except ConnectionError:
            ray.init(ignore_reinit_error=True, runtime_env=ray_runtime_env)

    # Distribute script and patches to all nodes
    print("[INFO] Distributing script to all nodes...")
    distribute_file_to_all_nodes(os.path.abspath(__file__))

    print("[INFO] Waiting 5 seconds for cluster to stabilize...")
    time.sleep(5)

    # Run training
    from verl.trainer.main_ppo import run_ppo

    print("Starting simple GRPO training (no hints, single-turn)...")
    update_worker_state(phase="training_start")
    try:
        run_ppo(config)
        update_worker_state(phase="training_complete")
        print("Training completed successfully!")

        # Gather checkpoint
        latest_checkpoint_file = os.path.join(output_dir, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_checkpoint_file):
            with open(latest_checkpoint_file, 'r') as f:
                latest_step = f.read().strip()
            checkpoint_dir = os.path.join(output_dir, f"global_step_{latest_step}")
            gathered_dir = os.path.join(output_dir, "gathered_checkpoint", f"global_step_{latest_step}")
            print(f"Gathering checkpoint (step {latest_step})...")
            success = gather_checkpoint_to_head_node(checkpoint_dir, gathered_dir, NUM_NODES, GPUS_PER_NODE)
            if success:
                print(f"Checkpoint gathered to: {gathered_dir}")
            else:
                print("Checkpoint gathering incomplete")
        else:
            print(f"[WARNING] No checkpoint found at {latest_checkpoint_file}")

    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        raise
    finally:
        if os.path.exists(train_dataset_file):
            os.unlink(train_dataset_file)
            print(f"Cleaned up: {train_dataset_file}")
        if os.path.exists(val_dataset_file):
            os.unlink(val_dataset_file)
            print(f"Cleaned up: {val_dataset_file}")


if __name__ == "__main__":
    main()
