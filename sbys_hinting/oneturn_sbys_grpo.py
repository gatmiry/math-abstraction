#!/usr/bin/env python3
# Note: HF offline mode ENABLED to prevent network timeouts blocking training
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline - model must be cached
os.environ["TRANSFORMERS_OFFLINE"] = "1"

"""
ONE-TURN GRPO training script for Omni-MATH dataset using verl with Ray.

INVESTIGATION RESULTS - How verl handles prompts:
==================================================

1. verl's SGLang rollout DOES call `start_interaction(instance_id, **interaction_kwargs)` 
   BEFORE the first generation (in `_handle_pending_state`, line 933 of sglang_rollout.py).
   
2. However, `start_interaction()` does NOT receive or return the `AsyncRolloutRequest` object,
   so we cannot directly modify the prompt messages from within start_interaction.

3. The prompt is set from the dataset in `_preprocess_prompt_to_async_rollout_requests()`:
   - `messages=raw_prompt.tolist()` comes from dataset's "prompt" field
   - `interaction_kwargs` comes from dataset's "extra_info.interaction_kwargs"

OPTIONS FOR ONE-TURN APPROACH:
==============================

Option A: DYNAMIC DATASET GENERATION (Recommended)
- Modify `to_verl_format()` to include hints in the initial prompt
- Query ProblemStateActor to get current `try_index` for each problem
- Requires per-epoch dataset regeneration or dynamic data loading

Option B: PATCH verl's _handle_pending_state
- Modify verl code to pass AsyncRolloutRequest to start_interaction
- More invasive but allows interaction to modify prompt directly

Option C: KEEP MULTI-TURN BUT OPTIMIZE (Current approach)
- Turn 0: Generate without hints (wasted computation)
- Turn 1: Reset prompt with hints, start real generation
- Turn 2: Evaluate and update state
- Slower due to wasted Turn 0, but works with existing verl

INJECTION POINTS IN VERL FOR DYNAMIC PROMPTS:
=============================================

1. RLHFDataset.__getitem__ (verl/utils/dataset/rl_dataset.py line 212)
   - Called for each sample when DataLoader fetches it
   - Can query ProblemStateActor here to get current try_index
   - Modify prompt to include appropriate hints
   - BEST for per-sample dynamic modification

2. collate_fn (verl/utils/dataset/rl_dataset.py line 37)
   - Called after __getitem__ to batch samples together
   - Can modify prompts at batch level
   - Passed to StatefulDataLoader at trainer line 503

3. Training loop (verl/trainer/ppo/ray_trainer.py line 959)
   - After: batch_dict = next(train_dataloader)
   - Before: batch = DataProto.from_single_dict(batch_dict)
   - Can modify batch_dict directly here

4. Epoch start (verl/trainer/ppo/ray_trainer.py line 945)
   - Before: for epoch in range(total_epochs)
   - Regenerate entire dataset with updated prompts per epoch

This file implements Option A: Custom Dataset with dynamic prompt generation.
"""

import os
import sys
import json
import argparse
import fcntl
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

# Use shared HF cache so all worker nodes can access cached models
os.environ.setdefault("HF_HOME", "/mnt/task_runtime/.hf_cache")

# HF_HUB_OFFLINE is set at the top of the file

# Set sglang watchdog timeout BEFORE any sglang imports
# Default is 300s (5 min) - if a forward batch takes longer, sglang kills the server
# 3 minutes should be plenty for any single generation with our batch sizes
# If generations are taking longer than 3 minutes, something is wrong and we want to know
try:
    from sglang.srt.server_args import ServerArgs
    ServerArgs.watchdog_timeout = 180  # 3 minutes - faster detection of stuck workers
    print(f"[INFO] Set sglang watchdog_timeout to {ServerArgs.watchdog_timeout}s (3 min)")
except ImportError:
    pass  # sglang not installed yet

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
VAL_SIZE = 256
ONE_TURN_MODE = True  # Enable one-turn mode: dynamic prompts, try_index updates in compute_score
ENABLE_VALIDATION_INTERACTION = False
TRAIN_BATCH_SIZE = 256  # Batch size for training
TOTAL_EPOCHS = 50
TEST_BATCH_SIZE = 256  # Reduced for faster validation



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
import socket
import traceback
import threading
import signal
import atexit
from verl.interactions.base import BaseInteraction
import ray
import tempfile
import datetime
import pandas as pd
import torch

# =============================================================================
# CENTRALIZED CRASH LOGGING - Writes to shared filesystem for all nodes
# All nodes write to /mnt/task_runtime/sbys_hinting/crash_logs/ via NFS
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
    except:
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
    
    # Get IP address
    try:
        info["ip"] = socket.gethostbyname(socket.gethostname())
    except:
        info["ip"] = "unknown"
    
    # Get GPU info
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
    
    # Get memory info
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        info["memory_mb"] = rusage.ru_maxrss / 1024  # Convert KB to MB
    except:
        pass
    
    # Get thread info
    info["thread_count"] = threading.active_count()
    info["thread_names"] = [t.name for t in threading.enumerate()]
    
    # Get worker state
    info["worker_state"] = dict(_WORKER_STATE)
    
    return info

def update_worker_state(phase: str = None, iteration: int = None, **kwargs):
    """Update the current worker state for crash diagnosis."""
    global _WORKER_STATE
    now = datetime.datetime.now()
    
    # Track phase transition timing for stuck detection
    if phase is not None:
        old_phase = _WORKER_STATE.get("phase")
        if old_phase != phase:
            # Log phase transition
            phase_start = _WORKER_STATE.get("_phase_start_time")
            if phase_start and old_phase:
                phase_duration = (now - datetime.datetime.fromisoformat(phase_start)).total_seconds()
                # Log if phase took too long (potential stuck indicator)
                if phase_duration > 60:  # More than 60 seconds in one phase
                    print(f"[STUCK DETECTION] Phase '{old_phase}' took {phase_duration:.1f}s (>60s) before transitioning to '{phase}'")
                    print(f"[STUCK DETECTION] Worker state: {_WORKER_STATE}")
            _WORKER_STATE["_prev_phase"] = old_phase
            _WORKER_STATE["_phase_start_time"] = now.isoformat()
        _WORKER_STATE["phase"] = phase
    
    if iteration is not None:
        _WORKER_STATE["iteration"] = iteration
    _WORKER_STATE["last_activity"] = now.isoformat()
    _WORKER_STATE.update(kwargs)

def log_crash(error_type: str, error_msg: str, stack_trace: str = None, extra_info: dict = None):
    """Log a crash/error to the shared crash log directory.
    
    This writes to /mnt/task_runtime/sbys_hinting/crash_logs/ which should be
    accessible from all nodes via NFS.
    """
    try:
        os.makedirs(_CRASH_LOG_DIR, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        node_id = get_node_identifier()
        system_info = get_detailed_system_info()
        
        log_entry = {
            "timestamp": timestamp,
            "node_id": node_id,
            "error_type": error_type,
            "error_msg": str(error_msg)[:2000],  # Limit size
            "stack_trace": (stack_trace or traceback.format_exc())[:10000],  # Limit size
            "extra_info": extra_info or {},
            "system_info": system_info,
            "uptime_seconds": (datetime.datetime.now() - _PROCESS_START_TIME).total_seconds(),
        }
        
        # Write to a single combined log file (thread-safe)
        combined_log = os.path.join(_CRASH_LOG_DIR, "all_crashes.jsonl")
        with _CRASH_LOG_LOCK:
            with open(combined_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        # Also write to node-specific file for easier debugging
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
                    f.write(f"  GPU {gpu['device']}: {gpu['allocated_gb']:.2f}GB allocated, {gpu['free_gb']:.2f}GB free / {gpu['total_gb']:.2f}GB total\n")
            if extra_info:
                f.write(f"Extra info: {json.dumps(extra_info, indent=2)}\n")
        
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
        
        # Write to heartbeat log (separate from crashes)
        heartbeat_log = os.path.join(_CRASH_LOG_DIR, "heartbeats.jsonl")
        with _CRASH_LOG_LOCK:
            with open(heartbeat_log, "a") as f:
                f.write(json.dumps(heartbeat_entry) + "\n")
    except:
        pass  # Don't crash on heartbeat failure

def start_heartbeat_thread(interval_seconds=30):
    """Start a background thread that logs heartbeats."""
    global _HEARTBEAT_THREAD, _HEARTBEAT_STOP
    
    def heartbeat_loop():
        while not _HEARTBEAT_STOP.is_set():
            log_heartbeat()
            _HEARTBEAT_STOP.wait(interval_seconds)
    
    _HEARTBEAT_STOP.clear()
    _HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True, name="HeartbeatThread")
    _HEARTBEAT_THREAD.start()
    print(f"[INFO] Started heartbeat thread (interval={interval_seconds}s)", flush=True)

def stop_heartbeat_thread():
    """Stop the heartbeat thread."""
    global _HEARTBEAT_STOP
    _HEARTBEAT_STOP.set()

def install_global_exception_handler():
    """Install a global exception handler that logs uncaught exceptions."""
    original_excepthook = sys.excepthook
    
    def custom_excepthook(exc_type, exc_value, exc_tb):
        # Log the crash
        stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log_crash(
            error_type=f"UNCAUGHT_EXCEPTION:{exc_type.__name__}",
            error_msg=str(exc_value),
            stack_trace=stack_trace,
            extra_info={"exc_type": str(exc_type)}
        )
        # Call original handler
        original_excepthook(exc_type, exc_value, exc_tb)
    
    sys.excepthook = custom_excepthook
    print(f"[INFO] Installed global exception handler, crash logs will be saved to: {_CRASH_LOG_DIR}", flush=True)

def install_signal_handlers():
    """Install signal handlers to log when process receives termination signals."""
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        stack_trace = "".join(traceback.format_stack(frame)) if frame else "No stack available"
        log_crash(
            error_type=f"SIGNAL_RECEIVED:{sig_name}",
            error_msg=f"Process received signal {signum} ({sig_name})",
            stack_trace=stack_trace,
            extra_info={"signal_number": signum, "signal_name": sig_name}
        )
        print(f"[SIGNAL] Received {sig_name}, logged to crash_logs", flush=True)
        # Re-raise the signal to allow normal termination
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    # Install handlers for common termination signals
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
# DISTRIBUTED CRASH MONITOR - Ray actor that tracks all workers across nodes
# =============================================================================
CRASH_MONITOR_ACTOR_NAME = "crash_monitor_actor"
_CRASH_MONITOR = None  # Will be set after ray.init()

@ray.remote
class CrashMonitorActor:
    """Central actor that monitors all workers across all nodes.
    
    Workers register with this actor and send heartbeats.
    If a worker stops sending heartbeats, we log its last known state.
    """
    
    def __init__(self):
        self._workers = {}  # worker_id -> {last_heartbeat, state, ip, etc.}
        self._dead_workers = []  # List of workers that stopped responding
        self._heartbeat_timeout = 120  # Consider dead after 2 min no heartbeat
        self._check_interval = 30  # Check for dead workers every 30s
        self._started = datetime.datetime.now()
        
        # Start background thread to check for dead workers
        self._stop_checker = threading.Event()
        self._checker_thread = threading.Thread(target=self._check_dead_workers, daemon=True)
        self._checker_thread.start()
        print(f"[CrashMonitor] Started on head node", flush=True)
    
    def register_worker(self, worker_id: str, worker_info: dict) -> bool:
        """Worker calls this on startup to register."""
        self._workers[worker_id] = {
            "registered_at": datetime.datetime.now().isoformat(),
            "last_heartbeat": datetime.datetime.now(),
            "heartbeat_count": 0,
            "state": worker_info.get("state", {}),
            "ip": worker_info.get("ip", "unknown"),
            "hostname": worker_info.get("hostname", "unknown"),
            "pid": worker_info.get("pid", 0),
            "gpu_info": worker_info.get("gpu_info", []),
        }
        print(f"[CrashMonitor] Worker registered: {worker_id} from {worker_info.get('ip')}", flush=True)
        return True
    
    def heartbeat(self, worker_id: str, state: dict, gpu_memory: list = None) -> bool:
        """Worker calls this periodically to report it's alive."""
        if worker_id not in self._workers:
            # Auto-register if not registered
            self._workers[worker_id] = {
                "registered_at": datetime.datetime.now().isoformat(),
                "last_heartbeat": datetime.datetime.now(),
                "heartbeat_count": 0,
                "state": state,
                "ip": state.get("ip", "unknown"),
            }
        
        worker = self._workers[worker_id]
        worker["last_heartbeat"] = datetime.datetime.now()
        worker["heartbeat_count"] += 1
        worker["state"] = state
        if gpu_memory:
            worker["gpu_memory"] = gpu_memory
        return True
    
    def report_crash(self, worker_id: str, error_type: str, error_msg: str, 
                     stack_trace: str = None, extra_info: dict = None) -> bool:
        """Worker calls this to report a crash before dying."""
        crash_info = {
            "worker_id": worker_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "error_type": error_type,
            "error_msg": error_msg[:2000],
            "stack_trace": stack_trace[:10000] if stack_trace else None,
            "extra_info": extra_info,
            "last_known_state": self._workers.get(worker_id, {}).get("state"),
        }
        
        # Log to file
        crash_log = os.path.join(_CRASH_LOG_DIR, "distributed_crashes.jsonl")
        os.makedirs(_CRASH_LOG_DIR, exist_ok=True)
        with open(crash_log, "a") as f:
            f.write(json.dumps(crash_info) + "\n")
        
        print(f"[CrashMonitor] CRASH REPORTED from {worker_id}: {error_type} - {error_msg[:100]}", flush=True)
        
        # Mark worker as dead
        if worker_id in self._workers:
            self._dead_workers.append({**self._workers[worker_id], "crash_info": crash_info})
            del self._workers[worker_id]
        
        return True
    
    def _check_dead_workers(self):
        """Background thread that checks for workers that stopped responding."""
        while not self._stop_checker.is_set():
            try:
                now = datetime.datetime.now()
                dead_workers = []
                
                for worker_id, info in list(self._workers.items()):
                    last_hb = info["last_heartbeat"]
                    if (now - last_hb).total_seconds() > self._heartbeat_timeout:
                        dead_workers.append((worker_id, info))
                
                for worker_id, info in dead_workers:
                    # Log the dead worker
                    dead_info = {
                        "worker_id": worker_id,
                        "timestamp": now.isoformat(),
                        "error_type": "HEARTBEAT_TIMEOUT",
                        "error_msg": f"Worker stopped responding after {self._heartbeat_timeout}s",
                        "last_heartbeat": info["last_heartbeat"].isoformat(),
                        "heartbeat_count": info["heartbeat_count"],
                        "last_known_state": info.get("state"),
                        "last_gpu_memory": info.get("gpu_memory"),
                        "ip": info.get("ip"),
                    }
                    
                    crash_log = os.path.join(_CRASH_LOG_DIR, "distributed_crashes.jsonl")
                    os.makedirs(_CRASH_LOG_DIR, exist_ok=True)
                    with open(crash_log, "a") as f:
                        f.write(json.dumps(dead_info) + "\n")
                    
                    print(f"[CrashMonitor] DEAD WORKER DETECTED: {worker_id} on {info.get('ip')}", flush=True)
                    print(f"[CrashMonitor] Last state: {info.get('state')}", flush=True)
                    print(f"[CrashMonitor] Last GPU memory: {info.get('gpu_memory')}", flush=True)
                    
                    self._dead_workers.append({**info, "death_type": "HEARTBEAT_TIMEOUT"})
                    del self._workers[worker_id]
                    
            except Exception as e:
                print(f"[CrashMonitor] Error checking dead workers: {e}", flush=True)
            
            self._stop_checker.wait(self._check_interval)
    
    def get_status(self) -> dict:
        """Get current status of all workers."""
        return {
            "active_workers": len(self._workers),
            "dead_workers": len(self._dead_workers),
            "workers": {k: {**v, "last_heartbeat": v["last_heartbeat"].isoformat()} 
                       for k, v in self._workers.items()},
            "dead_worker_ids": [d.get("worker_id", d.get("ip", "unknown")) for d in self._dead_workers],
        }
    
    def get_dead_workers(self) -> list:
        """Get list of dead workers with their last known state."""
        return self._dead_workers


def get_or_create_crash_monitor():
    """Get existing CrashMonitorActor or create a new one."""
    try:
        return ray.get_actor(CRASH_MONITOR_ACTOR_NAME, namespace="sbys_grpo")
    except ValueError:
        return CrashMonitorActor.options(
            name=CRASH_MONITOR_ACTOR_NAME,
            namespace="sbys_grpo",  # Use consistent namespace
            lifetime="detached",
            num_cpus=0.1,  # Minimal CPU usage
        ).remote()


def register_with_crash_monitor():
    """Register this worker with the crash monitor."""
    global _CRASH_MONITOR
    try:
        _CRASH_MONITOR = get_or_create_crash_monitor()
        worker_id = get_node_identifier()
        worker_info = {
            "ip": socket.gethostbyname(socket.gethostname()),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "state": dict(_WORKER_STATE),
        }
        try:
            if torch.cuda.is_available():
                worker_info["gpu_info"] = [
                    {"device": i, "name": torch.cuda.get_device_name(i)}
                    for i in range(torch.cuda.device_count())
                ]
        except:
            pass
        
        ray.get(_CRASH_MONITOR.register_worker.remote(worker_id, worker_info))
        print(f"[CrashMonitor] Registered with crash monitor as {worker_id}", flush=True)
    except Exception as e:
        print(f"[CrashMonitor] Failed to register: {e}", flush=True)


def send_heartbeat_to_monitor():
    """Send heartbeat to the crash monitor."""
    global _CRASH_MONITOR
    if _CRASH_MONITOR is None:
        return
    
    try:
        worker_id = get_node_identifier()
        state = dict(_WORKER_STATE)
        state["ip"] = socket.gethostbyname(socket.gethostname())
        
        # ==================== STUCK DETECTION ON HEARTBEAT ====================
        # Check if worker has been in the same phase for too long
        phase_start_str = _WORKER_STATE.get("_phase_start_time")
        current_phase = _WORKER_STATE.get("phase", "unknown")
        if phase_start_str:
            try:
                phase_start = datetime.datetime.fromisoformat(phase_start_str)
                phase_duration = (datetime.datetime.now() - phase_start).total_seconds()
                state["phase_duration_seconds"] = round(phase_duration, 1)
                
                # Log warning if stuck in a phase for too long
                if phase_duration > 120:  # 2 minutes
                    print(f"[STUCK ALERT] Worker stuck in phase '{current_phase}' for {phase_duration:.0f}s!")
                    print(f"[STUCK ALERT] Worker state: {state}")
                    if phase_duration > 300:  # 5 minutes
                        print(f"[STUCK CRITICAL] Worker CRITICALLY stuck in '{current_phase}' for {phase_duration:.0f}s!")
                        print(f"[STUCK CRITICAL] This may cause distributed barrier failures!")
            except Exception as e:
                pass  # Ignore parsing errors
        
        gpu_memory = None
        try:
            if torch.cuda.is_available():
                gpu_memory = []
                for i in range(torch.cuda.device_count()):
                    gpu_memory.append({
                        "device": i,
                        "allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                        "reserved_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
                    })
        except:
            pass
        
        # Fire-and-forget heartbeat (don't block on it)
        _CRASH_MONITOR.heartbeat.remote(worker_id, state, gpu_memory)
    except Exception as e:
        pass  # Don't fail on heartbeat errors


def report_crash_to_monitor(error_type: str, error_msg: str, stack_trace: str = None, extra_info: dict = None):
    """Report a crash to the central monitor."""
    global _CRASH_MONITOR
    if _CRASH_MONITOR is None:
        return
    
    try:
        worker_id = get_node_identifier()
        # Use ray.get with short timeout to ensure crash is reported before we die
        ray.get(_CRASH_MONITOR.report_crash.remote(
            worker_id, error_type, error_msg, stack_trace, extra_info
        ), timeout=5)
    except Exception as e:
        print(f"[CrashMonitor] Failed to report crash: {e}", flush=True)


# Override the heartbeat thread to also send to monitor
def start_heartbeat_thread_v2(interval_seconds=30):
    """Start a background thread that logs heartbeats to both file and monitor."""
    global _HEARTBEAT_THREAD, _HEARTBEAT_STOP
    
    def heartbeat_loop():
        while not _HEARTBEAT_STOP.is_set():
            log_heartbeat()  # Original file-based heartbeat
            send_heartbeat_to_monitor()  # New monitor-based heartbeat
            _HEARTBEAT_STOP.wait(interval_seconds)
    
    _HEARTBEAT_STOP.clear()
    _HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True, name="HeartbeatThread")
    _HEARTBEAT_THREAD.start()
    print(f"[INFO] Started heartbeat thread v2 (interval={interval_seconds}s)", flush=True)


# =============================================================================
# WORKER CRASH MONITORING - Initialize on remote workers
# =============================================================================
_WORKER_CRASH_MONITORING_INITIALIZED = False

def init_worker_crash_monitoring():
    """Initialize crash monitoring on a worker process.
    
    Safe to call multiple times - will only initialize once per process.
    """
    global _WORKER_CRASH_MONITORING_INITIALIZED, _CRASH_MONITOR
    
    if _WORKER_CRASH_MONITORING_INITIALIZED:
        return  # Already initialized
    
    try:
        # Try to connect to the existing CrashMonitorActor
        try:
            _CRASH_MONITOR = ray.get_actor(CRASH_MONITOR_ACTOR_NAME, namespace="sbys_grpo")
        except ValueError:
            # Actor doesn't exist yet, skip (head node will create it)
            print(f"[WorkerCrashMonitor] CrashMonitorActor not found, skipping registration", flush=True)
            _WORKER_CRASH_MONITORING_INITIALIZED = True
            return
        
        worker_id = get_node_identifier()
        ip = socket.gethostbyname(socket.gethostname())
        
        # Build worker_info dict with all relevant info
        worker_info = {
            "ip": ip,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "state": dict(_WORKER_STATE),
            "gpu_info": [],
        }
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    worker_info["gpu_info"].append({
                        "device": i,
                        "name": torch.cuda.get_device_name(i),
                    })
        except:
            pass
        
        # Register this worker
        try:
            ray.get(_CRASH_MONITOR.register_worker.remote(worker_id, worker_info), timeout=10)
            print(f"[WorkerCrashMonitor] Registered worker {worker_id} from {ip}", flush=True)
        except Exception as e:
            print(f"[WorkerCrashMonitor] Failed to register: {e}", flush=True)
        
        # Start heartbeat thread (if not already running)
        global _HEARTBEAT_THREAD
        if _HEARTBEAT_THREAD is None or not _HEARTBEAT_THREAD.is_alive():
            start_heartbeat_thread_v2(interval_seconds=15)
            print(f"[WorkerCrashMonitor] Started heartbeat thread on {worker_id}", flush=True)
        
        # Update worker state to show we're a remote worker
        update_worker_state(phase="worker_initialized", worker_type="remote_rollout_worker")
        
        _WORKER_CRASH_MONITORING_INITIALIZED = True
        print(f"[WorkerCrashMonitor] Crash monitoring initialized on {worker_id}", flush=True)
        
    except Exception as e:
        print(f"[WorkerCrashMonitor] Failed to initialize crash monitoring: {e}", flush=True)
        _WORKER_CRASH_MONITORING_INITIALIZED = True  # Don't retry on failure


# Override crash logger to also report to monitor
_original_log_crash = log_crash
def log_crash_v2(error_type: str, error_msg: str, stack_trace: str = None, extra_info: dict = None):
    """Log crash to both file and central monitor."""
    _original_log_crash(error_type, error_msg, stack_trace, extra_info)
    report_crash_to_monitor(error_type, error_msg, stack_trace, extra_info)

log_crash = log_crash_v2

# =============================================================================
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import load_from_disk
from typing import List, Dict, Optional, Any
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.schemas import AsyncRolloutRequest, BASE_CHAT_HISTORY, Message
# PPO_RAY_RUNTIME_ENV removed - not needed for single node

# Configuration
DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = os.path.join(os.path.dirname(__file__), "outputs", "hint_helped_dataset", "hint_helped_dataset")
MAX_NUM = None  # Limit dataset to last MAX_NUM rows (None = use all data). Useful for testing.

# Training hyperparameters


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
# 1 node, 8 GPUs = 8 GPUs total
NUM_NODES = 1
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
    
    # ONE-TURN MODE: Update try_index in ProblemStateActor directly from reward function
    if ONE_TURN_MODE and extra_info:
        interaction_kwargs = extra_info.get("interaction_kwargs", {})
        problem = interaction_kwargs.get("problem")
        if problem:
            state_actor = ray.get_actor(PROBLEM_STATE_ACTOR_NAME, namespace="sbys_grpo")
            N_SAMPLES_PER_PROMPT = 4  # Same as GRPO n parameter
            state_actor.try_claim_and_update.remote(problem, is_correct, N_SAMPLES_PER_PROMPT)
    
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
        self._validation_log_count = 0  # Shared counter for validation logging
    
    def get_next_validation_log_index(self) -> int:
        """Atomically increment and return the next validation log index.
        
        This is thread-safe across all distributed workers since it runs
        in a single Ray actor.
        """
        self._validation_log_count += 1
        return self._validation_log_count
    
    def get_state(self, problem_key: str) -> Optional[Dict[str, Any]]:
        """Get state for a problem, returns None if not exists."""
        return self._problem_state.get(problem_key)
    
    def get_states_batch(self, problem_keys: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get states for multiple problems in a single call (reduces actor contention)."""
        return {key: self._problem_state.get(key) for key in problem_keys}
    
    def set_state(self, problem_key: str, state: Dict[str, Any]) -> None:
        """Set state for a problem."""
        self._problem_state[problem_key] = state
    
    def process_turn1_batch(self, problem_keys: List[str], target_count: int = 8) -> Dict[str, int]:
        """Process Turn 1 for multiple problems in a single call.
        
        Increments turn_counter for each problem and returns the new counter values.
        Much more efficient than calling increment_turn_counter individually.
        """
        results = {}
        for key in problem_keys:
            if key in self._problem_state:
                state = self._problem_state[key]
                counter = state.get("turn_counter", 0) + 1
                state["turn_counter"] = counter
                results[key] = counter
        return results
    
    def process_turn2_batch(self, updates: List[Dict[str, Any]], target_count: int = 8) -> Dict[str, tuple]:
        """Process Turn 2 for multiple problems in a single call.
        
        Each update dict should contain: problem_key, is_correct
        Returns dict of problem_key -> (update_success, old_try_index, new_try_index, state)
        """
        results = {}
        for update in updates:
            key = update["problem_key"]
            is_correct = update["is_correct"]
            
            if key not in self._problem_state:
                results[key] = (False, 0, 0, {})
                continue
                
            state = self._problem_state[key]
            counter = state.get("turn_counter", 0) + 1
            state["turn_counter"] = counter
            
            # Check if this call claims the update (hits target_count)
            if counter == target_count:
                old_try_index = state.get("try_index", 0)
                able = state.get("able_index", 0)
                unable = state.get("unable_index", 0)
                
                if is_correct:
                    # Correct: move able_index down (need fewer hints)
                    able = old_try_index
                else:
                    # Incorrect: move unable_index up (need more hints)
                    unable = old_try_index + 1
                
                # Binary search: next try is midpoint
                new_try_index = (able + unable) // 2
                
                state["able_index"] = able
                state["unable_index"] = unable
                state["try_index"] = new_try_index
                state["total_attempts"] = state.get("total_attempts", 0) + 1
                if is_correct:
                    state["total_correct"] = state.get("total_correct", 0) + 1
                state["turn_counter"] = 0  # Reset for next step
                state["last_verified_attempts"] = state.get("total_attempts", 0)
                
                results[key] = (True, old_try_index, new_try_index, dict(state))
            else:
                results[key] = (False, state.get("try_index", 0), state.get("try_index", 0), dict(state))
        
        return results
    
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
    
    def verify_counters_reset(self, problem_keys: List[str]) -> Dict[str, Any]:
        """Verify that turn_counter is 0 for all specified problems.
        
        Called after each GRPO step to ensure all counters were properly reset.
        
        Returns:
            Dict with 'all_zero': bool, 'non_zero_problems': list of (key, counter) tuples
        """
        non_zero = []
        for key in problem_keys:
            if key in self._problem_state:
                counter = self._problem_state[key].get("turn_counter", 0)
                if counter != 0:
                    non_zero.append((key[:50], counter))
        
        return {
            "all_zero": len(non_zero) == 0,
            "non_zero_problems": non_zero,
            "checked_count": len(problem_keys),
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
                    "turn_counter": 0,  # For race-free state updates
                    "last_verified_attempts": -1,  # Sentinel for step completion verification
                }
                count += 1
        return count
    
    def increment_turn_counter(self, problem_key: str, target_count: int = 8) -> int:
        """Increment the turn counter. Called in both Turn 1 and Turn 2.
        
        Returns the counter value AFTER increment.
        
        VERIFICATION: When counter transitions from 0 to 1 (first increment of a new step),
        we check if total_attempts increased since the last time counter was 0.
        If not, the previous step didn't complete (counter never hit target_count).
        """
        if problem_key not in self._problem_state:
            return 0
        state = self._problem_state[problem_key]
        counter = state.get("turn_counter", 0)
        total_attempts = state.get("total_attempts", 0)
        last_verified_attempts = state.get("last_verified_attempts", -1)
        
        # VERIFICATION: First increment of a new step (counter == 0)
        if counter == 0:
            # Check if total_attempts increased since last time we started a step
            # If not (and this isn't the first step), the previous step didn't complete!
            if last_verified_attempts != -1 and total_attempts == last_verified_attempts:
                error_msg = (
                    f"FATAL ERROR: Previous step did not complete!\n"
                    f"Counter is 0 but total_attempts ({total_attempts}) did not increase since last step.\n"
                    f"This means counter never reached {target_count} in the previous step.\n"
                    f"Problem key: {problem_key[:100]}..."
                )
                print(f"[ProblemStateActor] {error_msg}")
                raise RuntimeError(error_msg)
            
            # Mark current total_attempts as verified (we'll check it increased at next step start)
            state["last_verified_attempts"] = total_attempts
        
        elif counter >= target_count:
            # Counter should have been reset when it hit target_count
            error_msg = (
                f"FATAL ERROR: turn_counter ({counter}) >= target ({target_count}) at start of increment!\n"
                f"Previous step did not reset counter properly.\n"
                f"Problem key: {problem_key[:100]}..."
            )
            print(f"[ProblemStateActor] {error_msg}")
            raise RuntimeError(error_msg)
        
        counter += 1
        state["turn_counter"] = counter
        return counter
    
    def try_claim_and_update(self, problem_key: str, is_correct: bool, target_count: int) -> tuple:
        """Try to claim this step's update. Only the generation that hits target_count wins.
        
        Counter-based approach (RACE-FREE for any interleaving):
        - Turn 1: Each of 4 generations increments counter (+4 total)
        - Turn 2: Each of 4 generations increments counter (+4 total)
        - Total per step = 8 (n_samples_per_prompt * 2)
        - Only the generation whose increment hits exactly target_count claims
        - After claiming, reset counter to 0 for next step
        
        Returns:
            (success, old_try_index, new_try_index, state)
        """
        import math
        
        state = self._problem_state.get(problem_key, {})
        old_try_index = state.get("try_index", 0)
        
        # Increment counter and check if we hit the target
        counter = state.get("turn_counter", 0) + 1
        state["turn_counter"] = counter
        
        # SANITY CHECK: Counter should NEVER exceed target_count
        # If it does, something is seriously wrong (duplicate calls, missing reset, etc.)
        if counter > target_count:
            error_msg = (
                f"FATAL ERROR: turn_counter ({counter}) exceeded target ({target_count})!\n"
                f"This indicates a bug: duplicate calls, missing reset, or race condition.\n"
                f"Problem key: {problem_key[:100]}...\n"
                f"State: total_attempts={state.get('total_attempts')}, "
                f"try_index={state.get('try_index')}, "
                f"able_index={state.get('able_index')}, "
                f"unable_index={state.get('unable_index')}"
            )
            print(f"[ProblemStateActor] {error_msg}")
            # Save state for debugging before crashing
            self._problem_state[problem_key] = state
            raise RuntimeError(error_msg)
        
        if counter != target_count:
            # Not the last one, don't claim
            self._problem_state[problem_key] = state
            return False, old_try_index, old_try_index, state
        
        # We hit the target! Claim and reset counter for next step
        state["turn_counter"] = 0
        
        # Apply updates
        state["total_attempts"] = state.get("total_attempts", 0) + 1
        if is_correct:
            state["total_correct"] = state.get("total_correct", 0) + 1
        
        # Binary search logic to update try_index
        try_index = state.get("try_index", 0)
        able_index = state.get("able_index", 0)
        unable_index = state.get("unable_index", 0)
        guide_steps_count = state.get("guide_steps_count", 0)
        
        if try_index <= unable_index and is_correct:
            state["able_index"] = try_index
            if try_index == unable_index:
                state["try_index"] = try_index - 1
            else:
                state["try_index"] = try_index - (unable_index - try_index)
            state["try_index"] = max(state["try_index"], 0)
        elif try_index >= able_index and not is_correct:
            state["unable_index"] = try_index
            if try_index == able_index:
                state["try_index"] = try_index + 1
            else:
                state["try_index"] = try_index + (try_index - able_index)
            state["try_index"] = min(state["try_index"], guide_steps_count)
        else:
            if not is_correct:
                state["unable_index"] = try_index
                state["try_index"] = math.ceil((try_index + able_index) / 2)
            else:
                state["able_index"] = try_index
                state["try_index"] = math.floor((try_index + unable_index) / 2)
        
        new_try_index = state["try_index"]
        self._problem_state[problem_key] = state
        
        return True, old_try_index, new_try_index, state


# Global name for the shared actor
PROBLEM_STATE_ACTOR_NAME = "problem_state_actor"


def get_or_create_problem_state_actor():
    """Get existing ProblemStateActor or create a new one."""
    try:
        # Try to get existing actor
        return ray.get_actor(PROBLEM_STATE_ACTOR_NAME, namespace="sbys_grpo")
    except ValueError:
        # Actor doesn't exist, create it
        return ProblemStateActor.options(
            name=PROBLEM_STATE_ACTOR_NAME,
            namespace="sbys_grpo",  # Use consistent namespace
            lifetime="detached",  # Survives driver failure
            max_concurrency=2000,  # Must handle TRAIN_BATCH_SIZE * n concurrent requests (256 * 4 = 1024+)
        ).remote()


# =============================================================================
# DYNAMIC HINT DATASET - Queries state actor at __getitem__ time (runs on head node)
# =============================================================================

# Global actor name for lookup (avoids storing actor reference in dataset)
PROBLEM_STATE_ACTOR_NAME = "ProblemStateActor"

class DynamicHintDataset(torch.utils.data.Dataset):
    """
    Dataset that queries ProblemStateActor for try_index and builds prompts WITH hints.
    
    KEY DESIGN: __getitem__ runs on HEAD NODE where Ray is accessible.
    We DON'T store the actor reference - we look it up by name using ray.get_actor().
    This avoids serialization issues when the dataset metadata is sent to workers.
    
    The batch (with tokenized prompts including hints) is then sent to workers as tensors.
    """
    
    def __init__(self, base_data: list, tokenizer, max_prompt_length: int = 4096):
        """
        Args:
            base_data: List of dicts with 'problem', 'answer', 'sbys_solution' keys
            tokenizer: HuggingFace tokenizer
            max_prompt_length: Maximum prompt length in tokens
        """
        self.base_data = base_data
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        
        # System prompts (loaded once)
        self._system_prompt_no_hints = load_system_prompt("full_solution_simple")
        self._system_prompt_with_hints = load_system_prompt("full_solution_with_hint")
        
        # Cache for state lookups within a batch/epoch
        self._state_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._log_count = 0
        
        print(f"[DynamicHintDataset] Initialized with {len(base_data)} samples (no actor reference stored)")
    
    def __len__(self):
        return len(self.base_data)
    
    def _get_state_actor(self):
        """Look up ProblemStateActor by name. Called at __getitem__ time on head node."""
        try:
            return ray.get_actor(PROBLEM_STATE_ACTOR_NAME, namespace="sbys_grpo")
        except ValueError:
            return None
    
    def _get_try_index(self, problem: str) -> int:
        """Get current try_index for a problem, using cache."""
        if problem in self._state_cache:
            self._cache_hits += 1
            return self._state_cache[problem]
        
        self._cache_misses += 1
        
        actor = self._get_state_actor()
        if actor is None:
            self._state_cache[problem] = 0
            return 0
        
        try:
            state = ray.get(actor.get_state.remote(problem), timeout=2)
            try_index = (state or {}).get("try_index", 0)
            self._state_cache[problem] = try_index
            return try_index
        except Exception:
            self._state_cache[problem] = 0
            return 0
    
    def _build_prompt_with_hints(self, problem: str, sbys_solution: list, try_index: int) -> list:
        """Build chat messages with hints based on try_index."""
        if try_index > 0 and sbys_solution and len(sbys_solution) >= try_index:
            partial_answer = "\n".join(sbys_solution[:try_index])
            user_content = f"Problem: {problem}\nIncomplete proof: {partial_answer}\n"
            system_prompt = self._system_prompt_with_hints
        else:
            user_content = f"Problem: {problem}\n"
            system_prompt = self._system_prompt_no_hints
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def __getitem__(self, idx):
        """Get sample with prompt built using CURRENT try_index from ProblemStateActor."""
        import time as _time
        _start = _time.time()
        
        # DEBUG: Log every 10th sample to see progress
        if idx % 10 == 0:
            print(f"[DynamicHintDataset] __getitem__({idx}) START", flush=True)
        
        item = self.base_data[idx]
        problem = item["problem"]
        answer = item["answer"]
        sbys_solution = item.get("sbys_solution", [])
        
        # Query current try_index (cached within batch/epoch)
        _query_start = _time.time()
        try_index = self._get_try_index(problem)
        _query_time = _time.time() - _query_start
        if _query_time > 0.5:
            print(f"[DynamicHintDataset] SLOW _get_try_index: {_query_time:.2f}s for idx={idx}", flush=True)
        
        # Build prompt WITH hints
        messages = self._build_prompt_with_hints(problem, sbys_solution, try_index)
        
        # Tokenize
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        
        # Truncate/pad
        from verl.utils.torch_functional import postprocess_data
        input_ids, attention_mask = postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="left",
        )
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
        
        # Log first few samples and every 10th
        self._log_count += 1
        _total_time = _time.time() - _start
        if self._log_count <= 5 or idx % 10 == 0:
            print(f"[DynamicHintDataset] #{self._log_count} idx={idx}: try_index={try_index}, prompt_len={len(raw_prompt_ids)}, time={_total_time:.2f}s", flush=True)
        
        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "raw_prompt": messages,
            "ground_truth": answer,
            "data_source": "omni_math",
            "index": idx,
            "tools_kwargs": {},
            "interaction_kwargs": {
                "ground_truth": answer,
                "problem": problem,
                "sbys_solution": sbys_solution,
                "try_index": try_index,
                "is_validation": False,
            },
        }
    
    def invalidate_cache(self):
        """Clear cache - call at epoch/step boundaries to refresh try_index values."""
        hits = self._cache_hits
        misses = self._cache_misses
        hit_rate = hits / max(1, hits + misses) * 100
        print(f"[DynamicHintDataset] Cache: {hits} hits, {misses} misses ({hit_rate:.1f}% hit rate)")
        self._state_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._log_count = 0
        print(f"[DynamicHintDataset] Cache invalidated")


# =============================================================================


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

def create_rl_dataset(tokenizer, dataset_path: str, max_samples: Optional[int] = None, val_size: int = 128, max_prompt_tokens: int = 4096, hint_level: int = 0):
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
    def to_verl_format(dataset_split, enable_interaction: bool, is_validation: bool = False):
        rl_data = []
        for item in dataset_split:
            extra_info = {}
            # Provide interaction_kwargs for compute_score to access problem info
            extra_info["interaction_kwargs"] = {
                "ground_truth": item["answer"],
                "state": {"attempt": 0},
                "sbys_solution": item.get("sbys_solution") if enable_interaction else [],
                "problem": item.get("problem") if enable_interaction else "",
                "is_validation": is_validation,
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
    
    train_data = to_verl_format(train_dataset, enable_interaction=True, is_validation=False)
    val_data = to_verl_format(val_dataset, enable_interaction=ENABLE_VALIDATION_INTERACTION, is_validation=True)
    
    return train_data, val_data


def get_verl_config_path():
    """Get path to verl's config directory."""
    import verl
    verl_path = os.path.dirname(verl.__file__)
    config_path = os.path.join(verl_path, "trainer", "config")
    return config_path


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


# =============================================================================
# ONE-TURN CUSTOM TRAINING LOOP
# =============================================================================

def run_ppo_one_turn(config, tokenizer, base_train_data: list):
    """One-turn training using verl's standard run_ppo (no multi-turn interaction)."""
    from verl.trainer.main_ppo import run_ppo
    run_ppo(config)


def main():
    """Main function to run GRPO training with verl and Ray."""
    # Start heartbeat thread for crash diagnosis
    start_heartbeat_thread_v2(interval_seconds=15)  # More frequent heartbeats for better crash detection
    
    # Register this worker with the central crash monitor
    register_with_crash_monitor()
    update_worker_state(phase="main_init", iteration=0)
    
    # Set NCCL and Gloo environment variables EARLY - before any distributed init
    # These help prevent socket errors during multi-node training
    distributed_env_vars = {
        # NCCL settings
        "NCCL_TIMEOUT": "3600",  # 1 hour timeout (default 30 min)
        "NCCL_DEBUG": "INFO",  # Full debug logging to catch network issues
        "NCCL_SOCKET_NTHREADS": "4",  # More socket threads
        "NCCL_NSOCKS_PERTHREAD": "4",  # More sockets per thread
        "NCCL_IB_DISABLE": "1",  # Disable InfiniBand, use TCP
        "NCCL_P2P_DISABLE": "1",  # Disable P2P (can cause issues)
        "NCCL_IGNORE_DISABLED_P2P": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",  # Better error handling
        # Gloo settings (used for CPU barriers)
        "GLOO_SOCKET_TIMEOUT": "3600000",  # 1 hour in milliseconds
        # PyTorch distributed timeout
        "TORCH_DISTRIBUTED_DEBUG": "OFF",  # Set to DETAIL for debugging
    }
    for key, value in distributed_env_vars.items():
        os.environ[key] = value
    print(f"[INFO] Set NCCL/Gloo environment variables for network stability")
    update_worker_state(phase="nccl_env_set")
    
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
    
    # Clear validation log files (workers will append to them)
    validation_log_dir = os.path.join(os.path.dirname(__file__), "validation_prompts")
    os.makedirs(validation_log_dir, exist_ok=True)
    validation_log_file = os.path.join(validation_log_dir, "verl_finalized_prompts.jsonl")
    validation_state_file = os.path.join(validation_log_dir, "validation_state.jsonl")
    with open(validation_log_file, 'w') as f:
        f.write("")  # Clear file at start of run
    with open(validation_state_file, 'w') as f:
        f.write("")  # Clear file at start of run
    print(f"Cleared validation logs: {validation_log_file}, {validation_state_file}")
    
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
    
    # For ONE_TURN_MODE, we'll store base data and create DynamicHintDataset later
    # The DynamicHintDataset will query ProblemStateActor for each sample's current try_index
    if ONE_TURN_MODE:
        print("[ONE_TURN_MODE] Will use DynamicHintDataset for dynamic prompt generation")
        # Extract base data (problem, answer, sbys_solution) from train_data
        base_train_data = []
        for item in train_data:
            extra_info = item.get("extra_info", {})
            interaction_kwargs = extra_info.get("interaction_kwargs", {})
            base_train_data.append({
                "problem": interaction_kwargs.get("problem", ""),
                "answer": item.get("ground_truth", ""),
                "sbys_solution": interaction_kwargs.get("sbys_solution", []),
            })
        print(f"[ONE_TURN_MODE] Extracted {len(base_train_data)} problems for dynamic dataset")
    
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
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",  # Reduced to 50% - more headroom for FSDP
        "actor_rollout_ref.rollout.prompt_length=4096",  # Increased from 2560 to handle multi-turn prompt updates
        "actor_rollout_ref.rollout.response_length=16384",  # Increased from 12288 to further reduce LENGTH truncation
        "actor_rollout_ref.rollout.max_model_len=20480",  # 4096 + 16384
        "actor_rollout_ref.rollout.max_num_batched_tokens=1310720",  # 20480*16 = ~102GB KV cache total, ~12.8GB per GPU with TP=8
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",  # Halved to avoid OOM in entropy calculation
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
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",  # Halved to avoid OOM in entropy calculation
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=false",  # Disabled - using smaller micro batch instead
        "+actor_rollout_ref.actor.offload_param=true",  # Offload actor params to CPU to reduce GPU memory pressure
        # "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",  # Only needed when use_kl_in_reward=true
        "actor_rollout_ref.ref.entropy_from_logits_with_chunking=false",  # Disabled - using smaller micro batch instead
        
        # Enable gradient checkpointing to reduce memory usage during policy update
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        
        # Algorithm - GRPO
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",  # Disabled to test if ref model init is causing crashes
        # "algorithm.kl_ctrl.kl_coef=0.01",  # KL coefficient - disabled for now
        "algorithm.norm_adv_by_std_in_grpo=true",
        
        # Data config
        f"data.train_files={train_dataset_file}",
        f"data.val_files={val_dataset_file}",
        "data.prompt_key=prompt",
        "data.max_prompt_length=4096",  # Match increased prompt_length for multi-turn
        "data.max_response_length=16384",  # Match rollout.response_length
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

    # ONE_TURN_MODE: Standard single-turn rollout with sglang (NO multi_turn)
    # try_index updates happen in compute_score reward function
    if ONE_TURN_MODE:
        print("[ONE_TURN_MODE] Using sglang single-turn rollout")
        print("[ONE_TURN_MODE] try_index updates happen in compute_score")
        overrides.append("actor_rollout_ref.rollout.name=sglang")
        # NO multi_turn - just single-turn generation
        overrides.append("+data.dataloader_num_workers=0")  # For DynamicHintDataset
        overrides.append("trainer.val_before_train=false")  # Skip initial validation
    
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
    
    # Small delay to ensure Ray cluster is fully synced before starting distributed training
    import time
    print("[INFO] Waiting 5 seconds for Ray cluster to stabilize...")
    time.sleep(5)
    
    # Run PPO training
    print("Starting GRPO training with verl...")
    update_worker_state(phase="training_start")
    try:
        # ONE-TURN MODE: Use run_ppo_one_turn (calls verl's standard run_ppo)
        # try_index updates happen in compute_score reward function
        print("[ONE_TURN_MODE] Using run_ppo_one_turn")
        run_ppo_one_turn(config, tokenizer, base_train_data)
        update_worker_state(phase="training_complete")
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
