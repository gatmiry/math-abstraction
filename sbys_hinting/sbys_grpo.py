#!/usr/bin/env python3
"""
GRPO training script for Omni-MATH dataset using verl with Ray for distributed training.
Configured for 2 nodes, each with 8 H100 GPUs (16 GPUs total).
Uses Hydra to properly load verl's default configs and override specific values.
"""

import os
import sys
import json
import argparse
import fcntl
import time
import uuid
import numpy as np
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

# Set sglang watchdog timeout BEFORE any sglang imports
# Default is 300s (5 min) - if a forward batch takes longer, sglang kills the server
# 3 minutes should be plenty for any single generation with our batch sizes
# If generations are taking longer than 3 minutes, something is wrong and we want to know
try:
    from sglang.srt.server_args import ServerArgs
    ServerArgs.watchdog_timeout = 1200  # 20 minutes - allow time for long sequences (16k tokens)
    print(f"[INFO] Set sglang watchdog_timeout to {ServerArgs.watchdog_timeout}s (20 min)")
except ImportError:
    pass  # sglang not installed yet

# Tell wandb to save code with each run
os.environ["WANDB_SAVE_CODE"] = "true"

# HuggingFace timeout settings - prevent 504 timeouts from slow HF servers
# Don't use offline mode as sglang needs to check if quant config exists
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 min timeout for downloads
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"  # 1 min timeout for metadata checks

# Ray gRPC and health check timeouts - MUST be set BEFORE ray.init()
# These prevent Ray from killing actors during slow model initialization
os.environ["RAY_grpc_keepalive_time_ms"] = "60000"      # Send keepalive ping every 60s (default 10s)
os.environ["RAY_grpc_keepalive_timeout_ms"] = "600000"  # Wait 10 min for keepalive response (default 20s)
os.environ["RAY_health_check_initial_delay_ms"] = "60000"   # Delay first health check by 60s
os.environ["RAY_health_check_period_ms"] = "60000"          # Health check every 60s
os.environ["RAY_health_check_timeout_ms"] = "600000"        # 10 min timeout for health checks
os.environ["RAY_health_check_failure_threshold"] = "10"     # Allow 10 failures before considering dead
print("[INFO] Set Ray keepalive/health check timeouts (10 min) for slow model init")

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
DYNAMIC_HINT_SELECTION = True  # If True, use state actor for binary search hint selection. If False, use fixed try_index = len(sbys_solution) // 2
PROMPT_UPDATE_MAX_ASSISTANT_TURNS = 2  # Turn 1: inject hints, Turn 2: evaluate
PROMPT_UPDATE_MAX_USER_TURNS = 2
PROMPT_RESET_PREFIX = "__RESET_PROMPT__\n"
PROMPT_RESET_SYSTEM_TAG = "__SYSTEM__\n"
PROMPT_RESET_USER_TAG = "__USER__\n"
PROMPT_LOG_VALIDATION_MARKER = "__LOG_VALIDATION__\n"  # Marker to enable logging in _reset_request_prompt
ENABLE_VALIDATION_INTERACTION = False
#
TRAIN_BATCH_SIZE = 128  # Minimum for 8 GPUs - debugging multi-turn
TOTAL_EPOCHS = 50
TEST_BATCH_SIZE = 128  # Minimum for 8 GPUs



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
        # Check if worker has been inactive for too long (no activity, not phase duration)
        last_activity_str = _WORKER_STATE.get("last_activity")
        current_phase = _WORKER_STATE.get("phase", "unknown")
        
        # Also track phase duration for informational purposes
        phase_start_str = _WORKER_STATE.get("_phase_start_time")
        if phase_start_str:
            try:
                phase_start = datetime.datetime.fromisoformat(phase_start_str)
                phase_duration = (datetime.datetime.now() - phase_start).total_seconds()
                state["phase_duration_seconds"] = round(phase_duration, 1)
            except:
                pass
        
        # Use last_activity for stuck detection (not phase start time!)
        if last_activity_str:
            try:
                last_activity = datetime.datetime.fromisoformat(last_activity_str)
                idle_time = (datetime.datetime.now() - last_activity).total_seconds()
                state["idle_seconds"] = round(idle_time, 1)
                
                # Log warning if no activity for too long
                if idle_time > 120:  # 2 minutes without any activity
                    print(f"[STUCK ALERT] Worker idle for {idle_time:.0f}s in phase '{current_phase}'!")
                    print(f"[STUCK ALERT] Worker state: {state}")
                    if idle_time > 300:  # 5 minutes without any activity
                        print(f"[STUCK CRITICAL] Worker CRITICALLY idle for {idle_time:.0f}s!")
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


# =============================================================================
# SGLANG HEALTH CHECK - Detect when sglang scheduler processes die
# =============================================================================
_SGLANG_DEATH_DETECTED = False

def check_sglang_scheduler_health():
    """Check if sglang scheduler processes are alive.
    
    Detects defunct/zombie scheduler processes which indicate sglang has crashed.
    If detected, logs the error and crashes the program.
    
    Returns:
        True if sglang is healthy, raises RuntimeError if dead
    """
    global _SGLANG_DEATH_DETECTED
    
    # Only check once - if already detected, skip (we're about to crash anyway)
    if _SGLANG_DEATH_DETECTED:
        return False
    
    try:
        import subprocess
        # Check for sglang scheduler processes
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        lines = result.stdout.split('\n')
        scheduler_lines = [l for l in lines if 'sglang::schedul' in l]
        
        if not scheduler_lines:
            # No scheduler processes found at all - might be too early or already dead
            return True
        
        # Check if any schedulers are defunct (zombie)
        defunct_schedulers = [l for l in scheduler_lines if '<defunct>' in l or ' Z' in l.split()[7:8]]
        alive_schedulers = [l for l in scheduler_lines if '<defunct>' not in l and ' Z' not in l.split()[7:8]]
        
        if defunct_schedulers and not alive_schedulers:
            # ALL schedulers are dead!
            _SGLANG_DEATH_DETECTED = True
            
            error_msg = f"FATAL: All sglang scheduler processes are DEAD (defunct/zombie)!\n"
            error_msg += f"Found {len(defunct_schedulers)} defunct schedulers, 0 alive.\n"
            error_msg += "This means generation cannot proceed. The training will hang indefinitely.\n"
            error_msg += "\nDefunct processes:\n"
            for line in defunct_schedulers[:5]:  # Show first 5
                error_msg += f"  {line}\n"
            
            print("\n" + "!" * 80)
            print("[SGLANG CRASH DETECTED]")
            print(error_msg)
            print("!" * 80 + "\n", flush=True)
            
            # Log to crash monitor
            log_crash(
                error_type="SGLANG_SCHEDULER_DEAD",
                error_msg=error_msg,
                extra_info={
                    "defunct_count": len(defunct_schedulers),
                    "alive_count": len(alive_schedulers),
                }
            )
            
            # CRASH THE PROGRAM
            raise RuntimeError(f"[FATAL] sglang scheduler processes are dead! {len(defunct_schedulers)} defunct, 0 alive. Training cannot continue.")
        
        elif defunct_schedulers:
            # Some schedulers are dead but others alive - warn but don't crash yet
            print(f"[SGLANG WARNING] {len(defunct_schedulers)}/{len(scheduler_lines)} scheduler processes are defunct!", flush=True)
        
        return True
        
    except subprocess.TimeoutExpired:
        print("[SGLANG HEALTH] ps command timed out", flush=True)
        return True
    except RuntimeError:
        # Re-raise our own RuntimeError
        raise
    except Exception as e:
        # Don't crash on health check errors
        print(f"[SGLANG HEALTH] Error checking health: {e}", flush=True)
        return True


# Override the heartbeat thread to also send to monitor
def start_heartbeat_thread_v2(interval_seconds=30):
    """Start a background thread that logs heartbeats to both file and monitor."""
    global _HEARTBEAT_THREAD, _HEARTBEAT_STOP
    
    def heartbeat_loop():
        while not _HEARTBEAT_STOP.is_set():
            log_heartbeat()  # Original file-based heartbeat
            send_heartbeat_to_monitor()  # New monitor-based heartbeat
            
            # Check sglang health - will crash if schedulers are dead
            try:
                check_sglang_scheduler_health()
            except RuntimeError as e:
                print(f"\n[FATAL] sglang health check failed: {e}", flush=True)
                # Force exit the entire process
                import os
                os._exit(1)
            
            _HEARTBEAT_STOP.wait(interval_seconds)
    
    _HEARTBEAT_STOP.clear()
    _HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True, name="HeartbeatThread")
    _HEARTBEAT_THREAD.start()
    print(f"[INFO] Started heartbeat thread v2 (interval={interval_seconds}s) with sglang health monitoring", flush=True)


# =============================================================================
# WORKER CRASH MONITORING - Initialize on remote workers
# =============================================================================
_WORKER_CRASH_MONITORING_INITIALIZED = False

def init_worker_crash_monitoring():
    """Initialize crash monitoring on a worker process.
    
    This should be called from PromptUpdateInteraction.__init__ to ensure
    remote workers also register with the CrashMonitorActor and send heartbeats.
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
# Ensure Ray runtime_env carries necessary env vars
try:
    from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
    PPO_RAY_RUNTIME_ENV.setdefault("env_vars", {})
    PPO_RAY_RUNTIME_ENV["env_vars"]["VLLM_USE_V1"] = "0"
    # NCCL P2P workaround for hardware issues
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_IGNORE_DISABLED_P2P"] = "1"
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_P2P_DISABLE"] = "1"
    # NCCL network stability settings - fix for socket errors (code 3)
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_TIMEOUT"] = "3600"  # 1 hour timeout (default is 30 min)
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_DEBUG"] = "INFO"  # Full debug logging to catch network issues
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_SOCKET_NTHREADS"] = "4"  # More socket threads for stability
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_NSOCKS_PERTHREAD"] = "4"  # More sockets per thread
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand, use TCP sockets
    # Torch distributed settings for better error reporting
    PPO_RAY_RUNTIME_ENV["env_vars"]["TORCH_DISTRIBUTED_DEBUG"] = "OFF"  # Set to DETAIL for debugging
    PPO_RAY_RUNTIME_ENV["env_vars"]["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Better error handling
    # Gloo timeout for CPU barriers (prevents Connection closed by peer errors)
    PPO_RAY_RUNTIME_ENV["env_vars"]["GLOO_SOCKET_TIMEOUT"] = "3600000"  # 1 hour in milliseconds
    # Add PYTHONPATH so verl workers can import sbys_hinting module
    PPO_RAY_RUNTIME_ENV["env_vars"]["PYTHONPATH"] = _parent_dir
    # Add CUDA library paths for sglang subprocesses
    cuda_lib_paths = [
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cublas/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cufft/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/curand/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cusolver/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/cusparse/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/nccl/lib",
        "/mnt/task_runtime/myenv/lib/python3.12/site-packages/nvidia/nvjitlink/lib",
    ]
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    PPO_RAY_RUNTIME_ENV["env_vars"]["LD_LIBRARY_PATH"] = ":".join(cuda_lib_paths) + ":" + existing_ld_path
    # Use triton attention backend instead of flashinfer JIT (system CUDA doesn't support sm_100a)
    PPO_RAY_RUNTIME_ENV["env_vars"]["SGLANG_ATTENTION_BACKEND"] = "triton"
    # Enable sglang debug logging to diagnose multi-turn issues
    PPO_RAY_RUNTIME_ENV["env_vars"]["SGLANG_LOG_LEVEL"] = "DEBUG"
    PPO_RAY_RUNTIME_ENV["env_vars"]["SGLANG_SHOW_TIME_COST"] = "1"
    # HuggingFace timeout settings - prevent 504 timeouts from slow HF servers
    PPO_RAY_RUNTIME_ENV["env_vars"]["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 min timeout
    PPO_RAY_RUNTIME_ENV["env_vars"]["HF_HUB_ETAG_TIMEOUT"] = "60"  # 1 min for metadata
    # Add WANDB_API_KEY to Ray workers so wandb logger can authenticate
    try:
        _early_tokens = load_tokens()
        _wandb_key = _early_tokens.get('WANDB_API_KEY', '')
        if _wandb_key and _wandb_key != 'YOUR_WANDB_API_KEY_HERE':
            PPO_RAY_RUNTIME_ENV["env_vars"]["WANDB_API_KEY"] = _wandb_key
            print(f"[INFO] WANDB_API_KEY added to Ray runtime_env for workers")
    except Exception as e:
        print(f"[WARNING] Could not add WANDB_API_KEY to Ray runtime_env: {e}")
    # Add sglang to pip packages for workers
    PPO_RAY_RUNTIME_ENV.setdefault("pip", [])
    PPO_RAY_RUNTIME_ENV["pip"].append("sglang[all]==0.4.6.post1")
except ImportError:
    pass  # verl 0.4.x may not have this

# Configuration
# Use HuggingFace model name (will download if not cached)
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
NUM_NODES = 1  # 1 node (8 GPUs)
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
    
    IMPORTANT: Uses run_id to detect and crash on stale state from previous runs.
    """
    
    def __init__(self, run_id: str = None):
        self._problem_state = {}  # Keyed by problem text
        self._validation_log_count = 0  # Shared counter for validation logging
        self._run_id = run_id or str(uuid.uuid4())
        self._created_at = time.time()
        # Step-level hint snapshots: {step: {problem_key: try_index}}
        # Ensures all samples in a step see the same hint level regardless of timing
        self._step_hint_snapshots = {}
        self._snapshot_cleanup_threshold = 3  # Keep snapshots for last N steps
        print(f"[ProblemStateActor] Initialized with run_id={self._run_id[:16]}... at {self._created_at}")
    
    def get_run_id(self) -> str:
        """Get the run_id this actor was created with."""
        return self._run_id
    
    def verify_run_id(self, expected_run_id: str) -> bool:
        """Verify the run_id matches. Returns False if mismatched (stale actor)."""
        if self._run_id != expected_run_id:
            print(f"[ProblemStateActor] FATAL: run_id mismatch! "
                  f"Actor has {self._run_id[:16]}..., expected {expected_run_id[:16]}...")
            return False
        return True
    
    def reset_all_counters(self) -> int:
        """Reset all turn_counters and step_correct_counts to 0. Called at start of each run.
        
        Returns the number of problems that had non-zero counters.
        """
        reset_count = 0
        for key, state in self._problem_state.items():
            had_stale = False
            if state.get("turn_counter", 0) != 0:
                old_counter = state["turn_counter"]
                state["turn_counter"] = 0
                had_stale = True
                print(f"[COUNTER_RESET] Reset stale turn_counter for {key[:40]}... "
                      f"(was {old_counter}, now 0)")
            if state.get("step_correct_count", 0) != 0:
                old_correct = state["step_correct_count"]
                state["step_correct_count"] = 0
                had_stale = True
                print(f"[COUNTER_RESET] Reset stale step_correct_count for {key[:40]}... "
                      f"(was {old_correct}, now 0)")
            if had_stale:
                reset_count += 1
        if reset_count > 0:
            print(f"[ProblemStateActor] Reset {reset_count} stale counters at run start")
        return reset_count
    
    
    
    def get_state(self, problem_key: str) -> Optional[Dict[str, Any]]:
        """Get state for a problem, returns None if not exists."""
        return self._problem_state.get(problem_key)
    
    def check_hint_level_consistency(self, step: int, problem_key: str, observed_hint_level: int) -> dict:
        """Check if the observed hint level is consistent with the snapshot from PREVIOUS step.
        
        Uses snapshot from step-1 (which is guaranteed finalized/stable) to compare against.
        This prevents race conditions where current step's snapshot could be affected by updates.
        
        RAISES RuntimeError if mismatch detected (crash the training).
        
        Returns a dict with:
        - 'expected': the snapshotted hint level
        - 'from_step': which step the snapshot came from
        """
        # FIRST: Try to use snapshot from PREVIOUS step (step - 1)
        # This is guaranteed to be finalized since step-1 is complete
        prev_step = step - 1
        if prev_step >= 0 and prev_step in self._step_hint_snapshots:
            prev_snapshots = self._step_hint_snapshots[prev_step]
            if problem_key in prev_snapshots:
                expected_level = prev_snapshots[problem_key]['level']
                if observed_hint_level != expected_level:
                    error_msg = (f"[HINT_LEVEL_MISMATCH] FATAL: Step {step}, problem={problem_key[:60]}... "
                                f"observed hint_level={observed_hint_level} but step {prev_step} snapshot has {expected_level}! "
                                f"Hint level was updated between steps - this indicates a race condition.")
                    print(error_msg)
                    raise RuntimeError(error_msg)
                # Match! Return success
                return {'expected': expected_level, 'from_step': prev_step}
        
        # FALLBACK: No snapshot from previous step (step 0, or problem not in prev batch)
        # Create/use snapshot for current step
        if step not in self._step_hint_snapshots:
            self._step_hint_snapshots[step] = {}
            # Cleanup old snapshots (keep last N steps)
            old_steps = [s for s in self._step_hint_snapshots.keys() 
                        if s < step - self._snapshot_cleanup_threshold]
            for old_step in old_steps:
                del self._step_hint_snapshots[old_step]
        
        step_snapshots = self._step_hint_snapshots[step]
        
        if problem_key not in step_snapshots:
            # First access for this (step, problem_key) - snapshot the OBSERVED value
            step_snapshots[problem_key] = {'level': observed_hint_level, 'count': 1}
            print(f"[HINT_SNAPSHOT] Step {step}, problem={problem_key[:40]}... snapshotted observed_hint_level={observed_hint_level} (no prev step snapshot)")
            return {'expected': observed_hint_level, 'from_step': step}
        
        # Check consistency with current step's snapshot
        tracker = step_snapshots[problem_key]
        tracker['count'] += 1
        expected_level = tracker['level']
        
        if observed_hint_level != expected_level:
            error_msg = (f"[HINT_LEVEL_MISMATCH] FATAL: Step {step}, problem={problem_key[:60]}... "
                        f"sample #{tracker['count']} has hint_level={observed_hint_level} but first sample had {expected_level}! "
                        f"All samples of the same problem in one step MUST see the same hint level.")
            print(error_msg)
            raise RuntimeError(error_msg)
        
        return {'expected': expected_level, 'from_step': step, 'sample_count': tracker['count']}
    
    def set_state(self, problem_key: str, state: Dict[str, Any]) -> None:
        """Set state for a problem."""
        self._problem_state[problem_key] = state
    
    
    
    def update_state(self, problem_key: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific fields in problem state and return the updated state."""
        if problem_key not in self._problem_state:
            self._problem_state[problem_key] = {}
        self._problem_state[problem_key].update(updates)
        return self._problem_state[problem_key]
    
    
    
    def get_hint_level_stats(self) -> Dict[str, Any]:
        """Get hint level (try_index) distribution stats for wandb logging."""
        if not self._problem_state:
            return {"hint_level/count": 0}
        
        try_indices = [s.get("try_index", 0) for s in self._problem_state.values()]
        guide_steps = [s.get("guide_steps_count", 0) for s in self._problem_state.values()]
        
        # Calculate percentages of max hint level used
        hint_pcts = []
        for ti, gs in zip(try_indices, guide_steps):
            if gs > 0:
                hint_pcts.append(ti / gs)
            else:
                hint_pcts.append(0.0)
        
        try_indices = np.array(try_indices)
        hint_pcts = np.array(hint_pcts)
        
        return {
            "hint_level/count": len(try_indices),
            "hint_level/mean": float(np.mean(try_indices)),
            "hint_level/median": float(np.median(try_indices)),
            "hint_level/max": float(np.max(try_indices)),
            "hint_level/min": float(np.min(try_indices)),
            "hint_level/std": float(np.std(try_indices)),
            "hint_level/pct_zero": float((try_indices == 0).mean()),
            "hint_level/pct_low_1_5": float(((try_indices >= 1) & (try_indices <= 5)).mean()),
            "hint_level/pct_mid_6_15": float(((try_indices >= 6) & (try_indices <= 15)).mean()),
            "hint_level/pct_high_16plus": float((try_indices >= 16).mean()),
            "hint_level/pct_of_max_mean": float(np.mean(hint_pcts)),  # How much of available hints used
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
                    "step_correct_count": 0,  # Number of correct responses in current step (0-4)
                    "last_verified_attempts": -1,  # Sentinel for step completion verification
                }
                count += 1
        return count
    
    def increment_turn_counter(self, problem_key: str, target_count: int = 8, sent_at: float = 0, sent_step: int = -1) -> int:
        """Increment the turn counter. Called in both Turn 1 and Turn 2.
        
        Returns the counter value AFTER increment.
        
        If counter is stale (from a previous incomplete batch), reset it.
        
        Debug params:
        - sent_at: timestamp when the call was sent (for latency tracking)
        - sent_step: the step number when call was sent (to detect stale calls)
        """
        import time
        
        if problem_key not in self._problem_state:
            return 0
        state = self._problem_state[problem_key]
        counter = state.get("turn_counter", 0)
        last_counter_time = state.get("_last_counter_time", 0)
        now = time.time()
        current_step = state.get("total_attempts", 0)
        
        # DEBUG: Check if this call is from a previous step (race condition!)
        if sent_step >= 0 and sent_step < current_step:
            print(f"[RACE_DETECTED] increment_turn_counter arrived LATE! "
                  f"sent_step={sent_step}, current_step={current_step}, "
                  f"latency={now - sent_at:.3f}s, problem={problem_key[:40]}...")
            return counter  # Ignore stale call
        
        # DEBUG: Log latency for fire-and-forget calls
        if sent_at > 0:
            latency = now - sent_at
            if latency > 1.0:  # Log if > 1 second latency
                print(f"[SLOW_CALL] increment_turn_counter latency={latency:.3f}s for {problem_key[:40]}...")
        
        # If counter is non-zero but it's been more than 5 minutes since last increment,
        # the previous batch probably timed out. Reset the counter.
        STALE_TIMEOUT = 300  # 5 minutes
        if counter > 0 and (now - last_counter_time) > STALE_TIMEOUT:
            print(f"[COUNTER_RESET] Stale counter detected for {problem_key[:40]}... "
                  f"counter={counter}, last_activity={now - last_counter_time:.0f}s ago. Resetting to 0.")
            counter = 0
            state["turn_counter"] = 0
        
        # If counter >= target, previous step didn't reset properly. Reset it.
        if counter >= target_count:
            print(f"[COUNTER_RESET] Counter overflow for {problem_key[:40]}... "
                  f"counter={counter} >= target={target_count}. Resetting to 0.")
            counter = 0
            state["turn_counter"] = 0
        
        counter += 1
        state["turn_counter"] = counter
        state["_last_counter_time"] = now
        total_attempts = state.get("total_attempts", 0)
        print(f"[COUNTER_DEBUG] increment_turn_counter: {problem_key[:40]}... counter={counter}/{target_count}, problem_step={total_attempts}")
        return counter
    
    def try_claim_and_update(self, problem_key: str, is_correct: bool, target_count: int) -> tuple:
        """Try to claim this step's update. Only the generation that hits target_count wins.
        
        Counter-based approach (RACE-FREE for any interleaving):
        - Turn 1: Each of 4 generations increments counter (+4 total)
        - Turn 2: Each of 4 generations increments counter (+4 total)
        - Total per step = 8 (n_samples_per_prompt * 2)
        - Only the generation whose increment hits exactly target_count claims
        - After claiming, reset counter to 0 for next step
        
        Also tracks step_correct_count: number of correct responses in this step (0-4).
        
        Returns:
            (success, old_try_index, new_try_index, state)
        """
        import math
        
        state = self._problem_state.get(problem_key, {})
        old_try_index = state.get("try_index", 0)
        
        # Increment counter and check if we hit the target
        counter = state.get("turn_counter", 0) + 1
        state["turn_counter"] = counter
        
        # Track number of correct responses in this step (only Turn 2 calls have is_correct info)
        # Turn 1 increments counter via increment_turn_counter, Turn 2 via try_claim_and_update
        # So step_correct_count only gets incremented here (Turn 2)
        if is_correct:
            state["step_correct_count"] = state.get("step_correct_count", 0) + 1
        
        total_attempts = state.get("total_attempts", 0)
        step_correct = state.get("step_correct_count", 0)
        print(f"[COUNTER_DEBUG] try_claim_and_update: {problem_key[:40]}... counter={counter}/{target_count}, problem_step={total_attempts}, step_correct={step_correct}")
        
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
        
        # We hit the target! Claim and reset counters for next step
        old_step = state.get("total_attempts", 0)
        final_step_correct_count = state.get("step_correct_count", 0)  # Save before reset
        state["turn_counter"] = 0
        state["step_correct_count"] = 0  # Reset for next step
        print(f"[STEP_CORRECT] problem={problem_key[:40]}... step {old_step} had {final_step_correct_count}/4 correct responses")
        
       

        # Apply updates
        state["total_attempts"] = old_step + 1
        new_step = state["total_attempts"]
        print(f"[STEP_COMPLETE] problem={problem_key[:40]}... step {old_step} → {new_step}, counter reset to 0")
        if is_correct:
            state["total_correct"] = state.get("total_correct", 0) + 1
        
        ## check if we might not need to update the try_index and yet go to the next step
        if final_step_correct_count != 4 or final_step_correct_count != 0:
            print(f"[STEP_COMPLETE] problem={problem_key[:40]}... step {old_step} → {new_step}, counter reset to 0, but no try_index update needed")
            return True, old_try_index, old_try_index, state
        ## check if we might not need to update the try_index and yet go to the next step
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
        
        # Log hint level changes
        if old_try_index != new_try_index:
            print(f"[HINT_LEVEL_UPDATE] problem={problem_key[:60]}... is_correct={is_correct} "
                  f"try_index: {old_try_index} → {new_try_index} "
                  f"(able={state.get('able_index')}, unable={state.get('unable_index')}, max={guide_steps_count})")
        else:
            print(f"[HINT_LEVEL_SAME] problem={problem_key[:60]}... is_correct={is_correct} "
                  f"try_index={old_try_index} (no change)")
        
        return True, old_try_index, new_try_index, state


# Global name for the shared actor
PROBLEM_STATE_ACTOR_NAME = "problem_state_actor"

# Global run_id for this training run - set once at startup
_CURRENT_RUN_ID = None


def set_current_run_id(run_id: str):
    """Set the global run_id for this training run."""
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = run_id
    print(f"[RUN_ID] Set current run_id to {run_id[:16]}...")


def get_current_run_id() -> str:
    """Get the global run_id for this training run."""
    global _CURRENT_RUN_ID
    if _CURRENT_RUN_ID is None:
        _CURRENT_RUN_ID = str(uuid.uuid4())
        print(f"[RUN_ID] Generated new run_id: {_CURRENT_RUN_ID[:16]}...")
    return _CURRENT_RUN_ID


def get_or_create_problem_state_actor(run_id: str = None):
    """Get existing ProblemStateActor or create a new one.
    
    Uses get-or-create pattern with retries to handle race conditions.
    
    IMPORTANT: 
    - If run_id is provided, creates actor with that run_id (main process)
    - If run_id is None (worker process), gets existing actor and adopts its run_id
    - If existing actor has different run_id than provided, CRASHES
    
    Args:
        run_id: The run_id for this training run. If None, adopts from existing actor.
    """
    import time as _time
    
    # Try to get or create the actor with retries (handles race conditions)
    for attempt in range(10):
        try:
            # First check if actor already exists
            actor = ray.get_actor(PROBLEM_STATE_ACTOR_NAME, namespace="sbys_grpo")
            if attempt == 0:
                print(f"[ProblemStateActor] Found existing actor")
            
            # Get the actor's run_id
            actor_run_id = ray.get(actor.get_run_id.remote(), timeout=30)
            
            if run_id is not None:
                # Main process: verify the run_id matches
                if actor_run_id != run_id:
                    error_msg = (
                        f"FATAL: Stale ProblemStateActor detected!\n"
                        f"  Actor run_id: {actor_run_id[:16]}...\n"
                        f"  Expected run_id: {run_id[:16]}...\n"
                        f"This indicates a previous run's actor is still alive.\n"
                        f"The actor should have been killed at startup.\n"
                        f"Please manually kill it: ray.kill(ray.get_actor('problem_state_actor', namespace='sbys_grpo'))"
                    )
                    print(f"[ProblemStateActor] {error_msg}")
                    raise RuntimeError(error_msg)
                print(f"[ProblemStateActor] run_id verified ✓ ({run_id[:16]}...)")
            else:
                # Worker process: adopt the actor's run_id
                set_current_run_id(actor_run_id)
                print(f"[ProblemStateActor] Adopted run_id from existing actor: {actor_run_id[:16]}...")
            
            return actor
        except ValueError:
            # Actor doesn't exist
            if run_id is None:
                # Worker trying to get actor that doesn't exist - FATAL
                error_msg = (
                    f"FATAL: ProblemStateActor does not exist!\n"
                    f"Worker processes should not create the actor.\n"
                    f"The main process should have created it before workers start.\n"
                    f"This indicates a startup race condition or main process failure."
                )
                print(f"[ProblemStateActor] {error_msg}")
                raise RuntimeError(error_msg)
            
            # Main process: try to create it
            try:
                print(f"[ProblemStateActor] Creating new actor (attempt {attempt + 1}) with run_id={run_id[:16]}...")
                # NOTE: Removed lifetime="detached" to prevent persistence across runs
                return ProblemStateActor.options(
                    name=PROBLEM_STATE_ACTOR_NAME,
                    namespace="sbys_grpo",
                    max_concurrency=2000,
                ).remote(run_id=run_id)
            except ray.exceptions.ActorAlreadyExistsError:
                # Another worker created it, retry to get it
                print(f"[ProblemStateActor] Race condition, retrying...")
                _time.sleep(0.5)
                continue
    
    # Final fallback - get actor and verify
    actor = ray.get_actor(PROBLEM_STATE_ACTOR_NAME, namespace="sbys_grpo")
    actor_run_id = ray.get(actor.get_run_id.remote(), timeout=30)
    if run_id is not None and actor_run_id != run_id:
        raise RuntimeError(f"FATAL: Stale actor run_id mismatch after retries: {actor_run_id} != {run_id}")
    if run_id is None:
        set_current_run_id(actor_run_id)
    return actor


class PromptUpdateInteraction(BaseInteraction):
    """Update prompts between generations based on previous response/reward."""
    
    # Class-level counter for detailed logging (shared across instances)
    _sample_log_counter = 0
    _sample_log_limit = 5  # Log detailed info for first N samples per batch
    
    # Class-level dict to store Turn 1 info by problem_key (since instance_id changes between calls)
    _turn1_info_by_problem = {}
    # Class-level dict to track Turn 1 start times for timing measurement
    _turn1_start_times = {}
    # Aggregated timing stats for periodic logging
    _timing_stats = {
        "turn1_count": 0,
        "turn2_count": 0,
        "turn1_durations": [],
        "turn2_durations": [],
        "generation_times": [],  # Time between Turn 1 end and Turn 2 start
        "state_actor_durations": [],
        "try_claim_durations": [],
        "last_summary_at": 0,
        "summary_interval": 50,  # Log summary every N turn2 completions
    }
    
    def __init__(self, config):
        super().__init__(config)
        self._instance_state = {}  # Per-instance state (keyed by instance_id)
        # Use Ray Actor for persistent state across all workers (only if dynamic hint selection is enabled)
        if DYNAMIC_HINT_SELECTION:
            self._state_actor = get_or_create_problem_state_actor()
        else:
            self._state_actor = None
            print(f"[PromptUpdateInteraction] DYNAMIC_HINT_SELECTION=False, using fixed try_index = len(sbys_solution) // 2")
        # Install the prompt reset hook on this worker (needed for multi-turn with prompt reset)
        install_prompt_reset_hook()
        
        # Initialize distributed crash monitoring on this worker (registers with CrashMonitorActor)
        # This ensures remote workers also send heartbeats and report crashes
        init_worker_crash_monitoring()
        
        # Cache our system prompts for turn detection
        self._our_system_prompts = {
            load_system_prompt("full_solution_simple_turn2"),  # No hints, Turn 2 variant
            load_system_prompt("full_solution_with_hint"),      # With hints
        }
    
    def _detect_turn_from_messages(self, messages) -> int:
        """Detect if this is Turn 1 or Turn 2 by checking the system prompt.
        
        Turn 1: System prompt is the ORIGINAL from dataset (full_solution_simple or similar)
        Turn 2: System prompt is one of OURS (full_solution_simple_turn2 or full_solution_with_hint)
        
        Returns: 1 for Turn 1, 2 for Turn 2
        """
        # Count assistant messages to detect actual turn number
        assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
        
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
                # Debug: check if it matches our prompts
                is_our_prompt = system_content in self._our_system_prompts
                # Also check substring match as fallback
                is_turn2_keyword = "studying mathematics" in system_content or "Incomplete proof:" in system_content
                detected_turn = 2 if (is_our_prompt or is_turn2_keyword) else 1
                
                # SANITY CHECK: assistant_count should match detected turn
                # Turn 1: 1 assistant message (the one we're discarding)
                # Turn 2: 1 assistant message (the one with hints we're scoring)
                if assistant_count > 1:
                    print(f"[TURN_DETECT WARNING] assistant_count={assistant_count} > 1! "
                          f"detected_turn={detected_turn}, is_our_prompt={is_our_prompt}, first50={system_content[:50]!r}")
                
                print(f"[TURN_DETECT] detected_turn={detected_turn}, assistant_count={assistant_count}, "
                      f"is_our_prompt={is_our_prompt}, is_turn2_keyword={is_turn2_keyword}", flush=True)
                return detected_turn
        return 1  # Default to Turn 1 if no system message found
    
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
        
        Validation (is_validation=True):
            Single turn: Terminate immediately, let verl's custom_reward_function score
        """
        import math
        import traceback
        
        # Catch exceptions to log them, but always re-raise to crash
        try:
            update_worker_state(phase="generate_response", instance_id=instance_id[:16])
            return await self._generate_response_impl(instance_id, messages, **kwargs)
        except Exception as e:
            # Log the error to centralized crash log AND console
            stack_trace = traceback.format_exc()
            print(f"[PromptUpdateInteraction] FATAL ERROR in generate_response: {e}")
            print(stack_trace)
            log_crash(
                error_type="INTERACTION_ERROR",
                error_msg=str(e),
                stack_trace=stack_trace,
                extra_info={"instance_id": instance_id, "kwargs_keys": list(kwargs.keys())}
            )
            # Re-raise to crash the worker
            raise
    
    async def _actor_call(self, coro, timeout_seconds=300, call_name="unknown"):
        """Await a Ray actor call with timeout. Crashes on timeout after long wait."""
        import asyncio
        import time as _time
        _call_start = _time.time()
        try:
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            _call_duration = _time.time() - _call_start
            if _call_duration > 5.0:
                print(f"[ACTOR CALL SLOW] {call_name} took {_call_duration:.2f}s (>5s)")
            return result
        except asyncio.TimeoutError:
            _call_duration = _time.time() - _call_start
            error_msg = f"FATAL: Actor call '{call_name}' TIMED OUT after {_call_duration:.2f}s (limit={timeout_seconds}s)"
            print(f"[ACTOR CALL TIMEOUT] {error_msg}")
            print(f"[ACTOR CALL TIMEOUT] Worker state: {dict(_WORKER_STATE)}")
            log_crash("ACTOR_TIMEOUT", error_msg,
                     extra_info={"call_name": call_name, "timeout": timeout_seconds})
            raise RuntimeError(error_msg)  # Crash on timeout
        except Exception as e:
            _call_duration = _time.time() - _call_start
            print(f"[ACTOR CALL FAILED] {call_name} failed after {_call_duration:.2f}s: {e}")
            log_crash("ACTOR_CALL_FAILED", str(e), traceback.format_exc(),
                     extra_info={"call_name": call_name})
            raise  # Re-raise to crash on non-timeout errors
    
    async def _generate_response_impl(self, instance_id, messages, **kwargs):
        """Internal implementation of generate_response with actual logic."""
        import math
        import time as _time
        
        # ==================== SGLANG HEALTH CHECK ====================
        # Check if sglang scheduler is alive before attempting any generation
        # This prevents hanging on a dead scheduler
        check_sglang_scheduler_health()  # Will raise RuntimeError if dead
        
        # ==================== DETAILED TIMING START ====================
        _impl_start = _time.time()
        _rollout_step = kwargs.get("_rollout_step", -1)  # Global step from sglang_rollout
        _timing_log = {
            "instance_id": instance_id[:16],
            "impl_start": _impl_start,
            "state_actor_calls": [],  # Track all state actor calls
            "rollout_step": _rollout_step,
        }
        
        # Track detailed state for crash diagnosis
        update_worker_state(
            phase="generate_response_start",
            instance_id=instance_id[:16],
            message_count=len(messages),
        )
        
        # Extract kwargs from interaction_kwargs
        ground_truth = kwargs.get("ground_truth")
        sbys_solution = kwargs.get("sbys_solution") or []
        problem = kwargs.get("problem")
        initial_state = kwargs.get("state", {})
        is_validation = kwargs.get("is_validation", False)
        
        # ==================== VALIDATION: Terminate immediately ====================
        # For validation, we don't want multi-turn interaction - just single generation
        # with scoring handled by custom_reward_function. Terminate immediately.
        if is_validation:
            update_worker_state(phase="validation_skip")
            print(f"[PromptUpdateInteraction] VALIDATION mode - terminating immediately")
            # Return: (should_terminate=True, content="", reward=0.0, metrics={})
            # The actual reward will be computed by compute_score in custom_reward_function
            return True, "", 0.0, {"mode": "validation", "skipped": True}
        
        # Debug: Log what we received (training only)
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
            # Debug: Show what we received
            print(f"[ERROR] No problem text! is_validation={is_validation}, kwargs_keys={list(kwargs.keys())}")
            print(f"[ERROR] Full kwargs: {kwargs}")
            # If this is actually a validation sample that somehow has is_validation=False,
            # treat it as validation (terminate immediately)
            if not sbys_solution or len(sbys_solution) == 0:
                print(f"[WARNING] Empty problem AND empty sbys_solution - treating as validation")
                return True, "", 0.0, {"mode": "validation_fallback", "skipped": True}
            raise ValueError(f"No problem text in kwargs for training! is_validation={is_validation}, kwargs={kwargs}")
            
        
        # Use problem text as key for persistent binary search state (survives across steps)
        problem_key = problem if problem else instance_id
        
        # Get problem state - either from state actor (dynamic) or use fixed hint level (static)
        if DYNAMIC_HINT_SELECTION and self._state_actor is not None:
            # Get persistent state from Ray Actor (initialized at start of training)
            # Use safe wrapper with timeout to avoid blocking and handle actor issues gracefully
            _get_state_start = _time.time()
            problem_state = await self._actor_call(
                self._state_actor.get_state.remote(problem_key),
                timeout_seconds=300,  # 5 minutes - wait long, then crash
                call_name="get_state"
            )
            _get_state_duration = _time.time() - _get_state_start
            _timing_log["state_actor_calls"].append(("get_state", _get_state_duration))
            if _get_state_duration > 1.0:
                print(f"[TIMING WARNING] get_state took {_get_state_duration:.3f}s (>1s) for {instance_id[:8]}...")
            
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
                    "step_correct_count": 0,
                }
                _set_state_start = _time.time()
                await self._actor_call(
                    self._state_actor.set_state.remote(problem_key, problem_state),
                    timeout_seconds=300,  # 5 minutes - wait long, then crash
                    call_name="set_state_init"
                )
                _set_state_duration = _time.time() - _set_state_start
                _timing_log["state_actor_calls"].append(("set_state_init", _set_state_duration))
                if _set_state_duration > 1.0:
                    print(f"[TIMING WARNING] set_state (init) took {_set_state_duration:.3f}s (>1s)")
        else:
            # STATIC HINT SELECTION: Use fixed try_index = len(sbys_solution) // 2
            # No state actor calls, no persistent state tracking
            guide_steps_count = len(sbys_solution) if sbys_solution else 0
            fixed_try_index = guide_steps_count // 2
            problem_state = {
                "guide_steps_count": guide_steps_count,
                "unable_index": 0,
                "able_index": guide_steps_count,
                "try_index": fixed_try_index,
                "total_attempts": 0,
                "total_correct": 0,
                "step_correct_count": 0,
            }
            print(f"[PromptUpdateInteraction] STATIC HINT: try_index={fixed_try_index} (half of {guide_steps_count} steps)")
        
        # Detect turn by checking system prompt content (not by counter!)
        # Turn 1: System prompt is ORIGINAL from dataset
        # Turn 2: System prompt is OURS (from Turn 1's reset)
        current_turn = self._detect_turn_from_messages(messages)
        print(f"[PromptUpdateInteraction] Detected turn {current_turn} via system prompt content")
        
        # Keep instance_state for storing Turn 1 info needed by Turn 2
        if instance_id not in self._instance_state:
            self._instance_state[instance_id] = {
                **(dict(initial_state) if isinstance(initial_state, dict) else {})
            }
        instance_state = self._instance_state[instance_id]
        # Link instance_id to problem_key so Turn 2 can find Turn 1's stored info
        instance_state["problem_key"] = problem_key
        
        import time as _time
        _turn_start_time = _time.time()
        print(f"[PromptUpdateInteraction] Turn {current_turn}, problem_key={problem_key[:50] if problem_key else None}... [timestamp={_turn_start_time:.3f}]")
        print(f"[PromptUpdateInteraction] Persistent state: try_index={problem_state['try_index']}, able={problem_state['able_index']}, unable={problem_state['unable_index']}")
        _timing_log["turn_detected"] = current_turn
        _timing_log["turn_detection_time"] = _time.time() - _impl_start
        
        # Note: Validation state logging moved to Turn 2 where we have the reward
        
        # Update guide_steps_count if sbys_solution changed (only if dynamic hint selection)
        if DYNAMIC_HINT_SELECTION and self._state_actor is not None:
            if problem_state["guide_steps_count"] == 0 and sbys_solution:
                print('guide_steps_count changed from 0 to ', len(sbys_solution), ' this was an error and should not happen')
                problem_state["guide_steps_count"] = len(sbys_solution)
                problem_state["able_index"] = len(sbys_solution)
                await self._actor_call(
                    self._state_actor.set_state.remote(problem_key, problem_state),
                    timeout_seconds=300,  # 5 minutes - wait long, then crash
                    call_name="set_state_guide_steps_fix"
                )
        
        # ==================== TURN 1: Inject hints ====================
        # Ignore the initial generation (it had no hints), return prompt WITH hints
        if current_turn == 1:
            _turn1_start = _time.time()
            _timing_log["turn1_start"] = _turn1_start
            update_worker_state(
                phase="turn1_processing",
                problem_key=problem_key[:50] if problem_key else None,
                try_index=problem_state.get("try_index", 0),
            )
            print(f"[TIMING] Turn 1 START at {_turn1_start:.3f}, setup_time={_turn1_start - _impl_start:.3f}s")
            print(f"[PromptUpdateInteraction] ROLLOUT_STEP={_rollout_step} Turn 1, problem_key={problem_key[:50]}..., try_index={problem_state['try_index']}")
            print(f"[PromptUpdateInteraction] Turn 1: problem={problem[:80] if problem else 'None'}...")
            print(f"[PromptUpdateInteraction] Turn 1: sbys_solution has {len(sbys_solution)} steps")
            
            # Check hint level consistency across workers (CRASH on mismatch!)
            # This uses a centralized snapshot in ProblemStateActor to detect race conditions
            hint_level = problem_state['try_index']
            if DYNAMIC_HINT_SELECTION and self._state_actor is not None:
                # Await consistency check - CRASH if mismatch detected (race condition = fatal error)
                await self._actor_call(
                    self._state_actor.check_hint_level_consistency.remote(_rollout_step, problem_key, hint_level),
                    timeout_seconds=60,  # Short timeout - this should be fast
                    call_name="check_hint_level_consistency"
                )
            print(f"[HINT_LEVEL_CHECK] problem={problem_key[:40]}... step={_rollout_step} hint_level={hint_level}")
            
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
            
            # Use turn2 variant of simple prompt so we can detect Turn 2 by checking system prompt
            # "full_solution_simple" = original from dataset (Turn 1)
            # "full_solution_simple_turn2" = our variant with "studying" instead of "learning" (Turn 2)
            # "full_solution_with_hint" = our hint prompt (Turn 2)
            system_prompt_name = "full_solution_with_hint" if problem_state["try_index"] > 0 else "full_solution_simple_turn2"
            system_prompt = load_system_prompt(system_prompt_name)
            # Add validation marker if this is a validation sample (enables logging in _reset_request_prompt)
            validation_marker = PROMPT_LOG_VALIDATION_MARKER if is_validation else ""
            if is_validation:
                print(f"[generate_response] VALIDATION SAMPLE - adding marker to reset_payload")
            reset_payload = (
                f"{validation_marker}"
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
            # Actually save turn1_info to class dict for retrieval in Turn 2
            PromptUpdateInteraction._turn1_info_by_problem[problem_key] = turn1_info
            
            # Increment turn counter - FIRE AND FORGET with debug params (only if dynamic hint selection)
            # The state actor will detect if this call arrives late (after next step started)
            if DYNAMIC_HINT_SELECTION and self._state_actor is not None:
                N_SAMPLES_PER_PROMPT = 4  # Must match actor_rollout_ref.rollout.n
                TARGET_COUNT = N_SAMPLES_PER_PROMPT * 2  # 4 Turn 1s + 4 Turn 2s = 8
                current_step = problem_state.get("total_attempts", 0)
                sent_at = _time.time()
                print(f"[COUNTER_DEBUG] Turn 1 INCREMENT (fire-and-forget) for {problem_key[:40]}... instance={instance_id[:8]}, step={current_step}")
                self._state_actor.increment_turn_counter.remote(problem_key, TARGET_COUNT, sent_at, current_step)
            else:
                print(f"[STATIC_HINT] Turn 1 skipping state actor call, fixed try_index={problem_state['try_index']}")
            
            # Record Turn 1 end time for timing measurement
            _turn1_end = _time.time()
            _turn1_duration = _turn1_end - _turn1_start
            PromptUpdateInteraction._turn1_start_times[problem_key] = _turn1_end
            _timing_log["turn1_duration"] = _turn1_duration
            _timing_log["turn1_end"] = _turn1_end
            
            # Log Turn 1 timing summary
            print(f"[TIMING] Turn 1 END: total={_turn1_duration:.3f}s, instance={instance_id[:8]}")
            if _turn1_duration > 5.0:
                print(f"[TIMING WARNING] Turn 1 took {_turn1_duration:.3f}s (>5s) for {instance_id[:8]}")
                print(f"[TIMING WARNING] State actor calls: {_timing_log['state_actor_calls']}")
            
            # Record timing stats (class-level aggregation)
            PromptUpdateInteraction._timing_stats["turn1_count"] += 1
            PromptUpdateInteraction._timing_stats["turn1_durations"].append(_turn1_duration)
            # Keep only last 100 samples
            if len(PromptUpdateInteraction._timing_stats["turn1_durations"]) > 100:
                PromptUpdateInteraction._timing_stats["turn1_durations"] = PromptUpdateInteraction._timing_stats["turn1_durations"][-100:]
            
            # Return 0 reward for turn 1 (we're not evaluating this generation)
            # NOTE: The finalized prompt is logged in _reset_request_prompt when verl processes our reset_payload
            return False, reset_payload, 0.0, {"turn": 1, "try_index": problem_state["try_index"], "turn1_duration": _turn1_duration}
        
        # ==================== TURN 2: Evaluate and update state ====================
        _turn2_start = _time.time()
        _timing_log["turn2_start"] = _turn2_start
        
        # Calculate time since Turn 1 ended (this is approximately the generation time)
        _turn1_end_time = PromptUpdateInteraction._turn1_start_times.get(problem_key, 0)
        _generation_time_approx = _turn2_start - _turn1_end_time if _turn1_end_time > 0 else -1
        _timing_log["generation_time_approx"] = _generation_time_approx
        
        update_worker_state(
            phase="turn2_evaluating",
            problem_key=problem_key[:50] if problem_key else None,
            try_index=problem_state.get("try_index", 0),
        )
        print(f"[TIMING] Turn 2 START at {_turn2_start:.3f}, gen_time_approx={_generation_time_approx:.3f}s")
        print(f"[PromptUpdateInteraction] ROLLOUT_STEP={_rollout_step} Turn 2: Evaluating generation for problem_key={problem_key[:50] if problem_key else None}...")
        
        # Extract the assistant's response
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        # Compute reward (time this)
        _compute_score_start = _time.time()
        reward = compute_score(solution_str=last_assistant, ground_truth=ground_truth)
        boxed_answer = extract_boxed_answer(last_assistant) or "N/A"
        _compute_score_duration = _time.time() - _compute_score_start
        _timing_log["compute_score_duration"] = _compute_score_duration
        if _compute_score_duration > 1.0:
            print(f"[TIMING WARNING] compute_score took {_compute_score_duration:.3f}s (>1s)")
        
        print(f"[PromptUpdateInteraction] ground_truth={ground_truth[:50] if ground_truth else None}...")
        print(f"[PromptUpdateInteraction] boxed_answer={boxed_answer[:50] if boxed_answer else 'N/A'}...")
        print(f"[PromptUpdateInteraction] reward={reward}")
        
        # Log Turn 2 evaluation with hint level used
        is_correct = reward > 0
        print(f"[TURN2_EVAL] problem={problem_key[:40]}... hint_level={problem_state.get('try_index', 0)} "
              f"is_correct={is_correct} boxed={boxed_answer[:30] if boxed_answer else 'N/A'}")
        
        # Log validation state with reward (only for validation, at Turn 2)
        # NOTE: Avoid blocking operations (ray.get, file locking) here to prevent deadlocks
        # during distributed training when workers finish at different times
        if is_validation:
            global _VALIDATION_STATE_LOG_FILE
            if _VALIDATION_STATE_LOG_FILE is None:
                log_dir = os.path.join(os.path.dirname(__file__), "validation_prompts")
                os.makedirs(log_dir, exist_ok=True)
                _VALIDATION_STATE_LOG_FILE = os.path.join(log_dir, "validation_state.jsonl")
                print(f"[generate_response] Will log validation state to: {_VALIDATION_STATE_LOG_FILE}")
            
            # Use local counter per worker to avoid blocking ray.get() calls
            # This prevents deadlocks when some workers finish before others
            if not hasattr(self, '_local_validation_log_counter'):
                self._local_validation_log_counter = 0
            self._local_validation_log_counter += 1
            log_index = self._local_validation_log_counter
            
            worker_pid = os.getpid()
            state_record = {
                "log_index": log_index,
                "worker_pid": worker_pid,  # Track which worker wrote this
                "problem": problem,
                "try_index": problem_state["try_index"],  # The try_index used for this generation
                "able_index": problem_state["able_index"],
                "unable_index": problem_state["unable_index"],
                "guide_steps_count": problem_state.get("guide_steps_count", 0),
                "ground_truth": ground_truth,
                "reward": reward,  # The reward for this generation
                "boxed_answer": boxed_answer[:200] if boxed_answer else "N/A",
            }
            
            # Write ONLY to per-worker file (no locking needed - each worker has its own file)
            # Avoid shared file with locking as it can cause deadlocks in distributed training
            worker_log_file = os.path.join(os.path.dirname(_VALIDATION_STATE_LOG_FILE), f"worker_{worker_pid}.jsonl")
            with open(worker_log_file, 'a') as f:
                f.write(json.dumps(state_record, indent=2) + '\n\n')
            
            print(f"[generate_response] Logged validation record #{log_index} from worker {worker_pid}")
        
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
        
        # ==================== STATE UPDATE (only if dynamic hint selection) ====================
        is_correct = reward > 0.5
        
        if DYNAMIC_HINT_SELECTION and self._state_actor is not None:
            # Counter-based approach (works regardless of interleaving):
            # - Turn 1: Each of 4 generations increments counter (+4)
            # - Turn 2: Each of 4 generations increments counter (+4)
            # - Total per step = 8 (N_SAMPLES_PER_PROMPT * 2)
            # - The generation whose increment reaches exactly 8 claims and updates
            # - After claiming, counter resets to 0 for next step
            N_SAMPLES_PER_PROMPT = 4  # Must match actor_rollout_ref.rollout.n
            TARGET_COUNT = N_SAMPLES_PER_PROMPT * 2  # 4 Turn 1s + 4 Turn 2s = 8
            
            # Await this call - it updates try_index which must complete before next step
            # Only 256 Turn 2 calls per batch (not 768), so this is acceptable
            _try_claim_start = _time.time()
            print(f"[COUNTER_DEBUG] Turn 2 try_claim_and_update for {problem_key[:40]}... instance={instance_id[:8]}")
            print(f"[TIMING] try_claim_and_update STARTING at {_try_claim_start:.3f} for {instance_id[:8]}...")
            result = await self._actor_call(
                self._state_actor.try_claim_and_update.remote(problem_key, is_correct, TARGET_COUNT),
                timeout_seconds=300,  # 5 minutes - wait long, then crash if still stuck
                call_name="try_claim_and_update"
            )
            _try_claim_duration = _time.time() - _try_claim_start
            _timing_log["state_actor_calls"].append(("try_claim_and_update", _try_claim_duration))
            _timing_log["try_claim_duration"] = _try_claim_duration
            if _try_claim_duration > 2.0:
                print(f"[TIMING WARNING] try_claim_and_update took {_try_claim_duration:.3f}s (>2s) for {instance_id[:8]}")
                print(f"[TIMING WARNING] This is a BLOCKING state actor call!")
            
            update_success, old_try_index, new_try_index, updated_state = result
            
            if update_success:
                print(f"[PromptUpdateInteraction] WON update claim! Updated hint level: {old_try_index} -> {new_try_index}")
            else:
                print(f"[PromptUpdateInteraction] Turn 2 complete: try_index={old_try_index}, is_correct={is_correct}")
        else:
            # STATIC HINT SELECTION: No state actor calls, try_index stays fixed
            old_try_index = problem_state.get("try_index", 0)
            new_try_index = old_try_index  # Never changes in static mode
            update_success = False
            print(f"[STATIC_HINT] Turn 2 complete: fixed try_index={old_try_index}, is_correct={is_correct}, reward={reward}")

            # Clean up instance state and turn1_info
            if instance_id in self._instance_state:
                del self._instance_state[instance_id]
            if hasattr(PromptUpdateInteraction, '_turn1_info_by_problem') and problem_key in PromptUpdateInteraction._turn1_info_by_problem:
                del PromptUpdateInteraction._turn1_info_by_problem[problem_key]
        if problem_key in PromptUpdateInteraction._turn1_start_times:
            del PromptUpdateInteraction._turn1_start_times[problem_key]
        
        # ==================== TURN 2 TIMING SUMMARY ====================
        _turn2_end = _time.time()
        _turn2_duration = _turn2_end - _turn2_start
        _total_duration = _turn2_end - _impl_start
        _timing_log["turn2_duration"] = _turn2_duration
        _timing_log["turn2_end"] = _turn2_end
        _timing_log["total_impl_duration"] = _total_duration
        
        print(f"[TIMING] Turn 2 END: turn2={_turn2_duration:.3f}s, total_impl={_total_duration:.3f}s, instance={instance_id[:8]}")
        
        # Log warning if any phase took too long
        if _turn2_duration > 5.0 or _total_duration > 10.0:
            print(f"[TIMING WARNING] Turn 2 or total took too long!")
            print(f"[TIMING WARNING] turn2_duration={_turn2_duration:.3f}s, total={_total_duration:.3f}s")
            print(f"[TIMING WARNING] gen_time_approx={_timing_log.get('generation_time_approx', -1):.3f}s")
            print(f"[TIMING WARNING] try_claim_duration={_timing_log.get('try_claim_duration', -1):.3f}s")
            print(f"[TIMING WARNING] State actor calls: {_timing_log['state_actor_calls']}")
        
        # ==================== AGGREGATED TIMING STATS ====================
        stats = PromptUpdateInteraction._timing_stats
        stats["turn2_count"] += 1
        stats["turn2_durations"].append(_turn2_duration)
        if _timing_log.get("generation_time_approx", -1) > 0:
            stats["generation_times"].append(_timing_log["generation_time_approx"])
        if _timing_log.get("try_claim_duration", -1) > 0:
            stats["try_claim_durations"].append(_timing_log["try_claim_duration"])
        
        # Keep only last 100 samples
        for key in ["turn2_durations", "generation_times", "try_claim_durations"]:
            if len(stats[key]) > 100:
                stats[key] = stats[key][-100:]
        
        # Log summary periodically
        if stats["turn2_count"] - stats["last_summary_at"] >= stats["summary_interval"]:
            stats["last_summary_at"] = stats["turn2_count"]
            def _avg(lst): return sum(lst) / len(lst) if lst else 0
            def _max(lst): return max(lst) if lst else 0
            print(f"\n{'='*60}")
            print(f"[TIMING SUMMARY] After {stats['turn2_count']} Turn 2 completions (worker pid={os.getpid()})")
            print(f"  Turn 1: count={stats['turn1_count']}, avg={_avg(stats['turn1_durations']):.2f}s, max={_max(stats['turn1_durations']):.2f}s")
            print(f"  Turn 2: count={stats['turn2_count']}, avg={_avg(stats['turn2_durations']):.2f}s, max={_max(stats['turn2_durations']):.2f}s")
            print(f"  Generation (T1->T2): avg={_avg(stats['generation_times']):.2f}s, max={_max(stats['generation_times']):.2f}s")
            print(f"  try_claim_and_update: avg={_avg(stats['try_claim_durations']):.3f}s, max={_max(stats['try_claim_durations']):.3f}s")
            
            # Log hint level stats to wandb (from state actor) - only if dynamic hint selection
            if DYNAMIC_HINT_SELECTION and self._state_actor is not None:
                try:
                    hint_stats = ray.get(self._state_actor.get_hint_level_stats.remote(), timeout=5.0)
                    print(f"  [HINT_LEVEL_STATS] mean={hint_stats.get('hint_level/mean', 0):.2f}, "
                          f"pct_zero={hint_stats.get('hint_level/pct_zero', 0)*100:.1f}%, "
                          f"pct_high_16+={hint_stats.get('hint_level/pct_high_16plus', 0)*100:.1f}%")
                    
                    # Log to wandb if available (only one worker should do this to avoid spam)
                    import wandb
                    if wandb.run is not None:
                        wandb.log(hint_stats, commit=False)
                except Exception as e:
                    print(f"  [HINT_LEVEL_STATS] Failed to get stats: {e}")
            else:
                print(f"  [STATIC_HINT] Using fixed try_index = len(sbys_solution) // 2 for all problems")
            
            print(f"{'='*60}\n")
        
        # Terminate the sequence - we're done with this problem for this GRPO step
        return True, "", reward, {
            "try_index": new_try_index, 
            "turn": 2, 
            "reward": reward, 
            "update_won": update_success,
            "turn2_duration": _turn2_duration,
            "try_claim_duration": _timing_log.get("try_claim_duration", -1),
            "generation_time_approx": _timing_log.get("generation_time_approx", -1),
        }

_RESET_PROMPT_LOG_FILE = None
_RESET_PROMPT_LOG_COUNT = 0

_VALIDATION_STATE_LOG_FILE = None
# Note: Validation log counter is now stored in ProblemStateActor for distributed consistency

def _reset_request_prompt(self, processing_class, new_user_content: str, new_system_content: Optional[str] = None, should_log: bool = False) -> None:
    import time as _time
    _reset_start = _time.time()
    
    global _RESET_PROMPT_LOG_FILE, _RESET_PROMPT_LOG_COUNT
    
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
    
    # === LOG THE FINALIZED PROMPT THAT VERL SENDS TO THE MODEL (validation only) ===
    if should_log:
        # Get the full prompt string (what the model actually sees)
        full_prompt_str = processing_class.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize log file path (all workers append to same file)
        if _RESET_PROMPT_LOG_FILE is None:
            log_dir = os.path.join(os.path.dirname(__file__), "validation_prompts")
            os.makedirs(log_dir, exist_ok=True)
            _RESET_PROMPT_LOG_FILE = os.path.join(log_dir, "verl_finalized_prompts.jsonl")
            print(f"[_reset_request_prompt] Will log VALIDATION finalized prompts to: {_RESET_PROMPT_LOG_FILE}")
        
        _RESET_PROMPT_LOG_COUNT += 1
        record = {
            "log_index": _RESET_PROMPT_LOG_COUNT,
            "messages": messages,
            "full_prompt_sent_to_model": full_prompt_str,
            "token_count": len(tokenization_dict_with_prompt["input_ids"]),
        }
        with open(_RESET_PROMPT_LOG_FILE, 'a') as f:
            f.write(json.dumps(record, indent=2) + '\n\n')  # Pretty print with separator
        
        # Print preview for first few
        if _RESET_PROMPT_LOG_COUNT <= 5:
            print(f"\n[VERL VALIDATION PROMPT #{_RESET_PROMPT_LOG_COUNT}] tokens={len(tokenization_dict_with_prompt['input_ids'])}")
            print(f"  Messages: {len(messages)} messages")
            for i, msg in enumerate(messages):
                print(f"    [{i}] {msg['role']}: {msg['content'][:150]}...")
            print(f"  Full prompt preview: {full_prompt_str[:400]}...")
    # === END LOGGING ===
    
    self.input_ids = tokenization_dict_with_prompt["input_ids"]
    self.attention_mask = tokenization_dict_with_prompt["attention_mask"]
    # IMPORTANT: Use list() to create COPIES, not references!
    # Otherwise _update_input_ids (called by add_assistant_message) would extend both
    self.prompt_ids = list(self.input_ids)
    self.prompt_attention_mask = list(self.attention_mask)
    position_ids_list = compute_position_id_with_mask(
        torch.tensor(self.attention_mask)
    ).tolist()
    self.position_ids = list(position_ids_list)
    self.prompt_position_ids = list(position_ids_list)
    loss_mask_list = [0] * len(self.input_ids)
    self.loss_mask = list(loss_mask_list)
    self.prompt_loss_mask = list(loss_mask_list)
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
    
    _reset_duration = _time.time() - _reset_start
    print(f"[TIMING] _reset_request_prompt completed in {_reset_duration:.4f}s, new input_ids len={len(self.input_ids)}, messages={len(self.messages)}")


def install_prompt_reset_hook():
    """Intercept add_user_message to reset the prompt when requested."""
    if getattr(AsyncRolloutRequest.add_user_message, "_prompt_reset_patched", False):
        return

    original_add_user_message = AsyncRolloutRequest.add_user_message

    def patched_add_user_message(self, processing_class, content: str):
        import traceback
        import time as _time
        _patch_start = _time.time()
        try:
            # Check for validation logging marker first
            should_log = False
            if content.startswith(PROMPT_LOG_VALIDATION_MARKER):
                should_log = True
                content = content[len(PROMPT_LOG_VALIDATION_MARKER):]
                print(f"[patched_add_user_message] DETECTED VALIDATION MARKER! should_log={should_log}")
            
            if content.startswith(PROMPT_RESET_PREFIX):
                print(f"[TIMING] patched_add_user_message DETECTED RESET at {_patch_start:.3f}, current input_ids len={len(self.input_ids)}")
                payload = content[len(PROMPT_RESET_PREFIX):]
                new_system_content = None
                new_user_content = payload
                if payload.startswith(PROMPT_RESET_SYSTEM_TAG):
                    payload = payload[len(PROMPT_RESET_SYSTEM_TAG):]
                    if PROMPT_RESET_USER_TAG in payload:
                        system_part, user_part = payload.split(PROMPT_RESET_USER_TAG, 1)
                        new_system_content = system_part.rstrip("\n")
                        new_user_content = user_part
                _reset_request_prompt(self, processing_class, new_user_content, new_system_content, should_log)
                print(f"[TIMING] patched_add_user_message RESET COMPLETE in {_time.time() - _patch_start:.4f}s")
                return
            return original_add_user_message(self, processing_class, content)
        except Exception as e:
            stack_trace = traceback.format_exc()
            print(f"[patched_add_user_message] ERROR: {e}")
            print(stack_trace)
            log_crash(
                error_type="PATCHED_ADD_USER_MESSAGE_ERROR",
                error_msg=str(e),
                stack_trace=stack_trace,
                extra_info={"content_preview": content[:200] if content else None}
            )
            # Re-raise to let verl handle it - but at least we logged it
            raise

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
            # IMPORTANT: When multi_turn.enable=true is set globally, verl's sglang_rollout
            # expects interaction_kwargs to exist in non_tensor_batch, even for validation.
            # If missing, it raises KeyError causing worker crash. We MUST provide it.
            # Also, AsyncRolloutRequest is a Pydantic model that requires interaction_kwargs
            # to be a dict (not None), so we must provide the same schema for both.
            if PROMPT_UPDATE_ENABLED:
                # Always provide full interaction_kwargs with same schema
                # For validation, is_validation=True tells the interaction to terminate immediately
                extra_info["interaction_kwargs"] = {
                    "ground_truth": item["answer"],
                    "state": {"attempt": 0},
                    "sbys_solution": item.get("sbys_solution") if enable_interaction else [],
                    "problem": item.get("problem") if enable_interaction else "",
                    "is_validation": is_validation,  # Flag to skip interaction for validation
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


def patch_verl_on_all_nodes():
    """
    Apply patched verl files to all nodes in the Ray cluster.
    
    Patches:
    1. sglang_rollout.py - watchdog_timeout=1200 (20 min), Turn 1 optimization (32 tokens)
    2. fsdp_workers.py - distributed timeout=3600s (was 30min default causing timeouts)
    3. ray_trainer.py - save_checkpoint_before_train option to save step 0 checkpoint
    """
    import shutil
    patches_dir = os.path.join(os.path.dirname(__file__), "patches")
    
    # ==================== PATCH LOCAL NODE FIRST ====================
    # This ensures the head node has the patches before any workers start
    local_patches = [
        (
            os.path.join(patches_dir, "sglang_rollout_patched.py"),
            "/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
        ),
        (
            os.path.join(patches_dir, "fsdp_workers_patched.py"),
            "/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/workers/fsdp_workers.py"
        ),
        (
            os.path.join(patches_dir, "ray_trainer_patched.py"),
            "/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/trainer/ppo/ray_trainer.py"
        ),
    ]
    print("[INFO] Patching LOCAL node first...")
    for source, target in local_patches:
        if os.path.exists(source):
            shutil.copy2(source, target)
            print(f"[INFO] LOCAL: Copied {os.path.basename(source)} -> {target}")
        else:
            print(f"[WARN] LOCAL: Patch not found: {source}")
    
    # Define patches to apply: (source_file, target_path)
    patches = [
        (
            os.path.join(patches_dir, "sglang_rollout_patched.py"),
            "/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
        ),
        (
            os.path.join(patches_dir, "fsdp_workers_patched.py"),
            "/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/workers/fsdp_workers.py"
        ),
        (
            os.path.join(patches_dir, "ray_trainer_patched.py"),
            "/mnt/task_runtime/myenv/lib/python3.12/site-packages/verl/trainer/ppo/ray_trainer.py"
        ),
    ]
    
    # Read all patches
    patch_data = []
    for source_file, target_path in patches:
        if not os.path.exists(source_file):
            print(f"[WARN] Patch file not found: {source_file}, skipping")
            continue
        with open(source_file, 'rb') as f:
            content = f.read()
        patch_data.append((content, target_path, os.path.basename(source_file)))
        print(f"[INFO] Read {os.path.basename(source_file)} ({len(content)} bytes)")
    
    if not patch_data:
        print("[WARN] No patches to apply")
        return False
    
    @ray.remote
    def apply_patches(patches_list: list) -> str:
        """Apply all patches on this node."""
        import os
        import socket
        results = []
        for content, target, name in patches_list:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'wb') as f:
                f.write(content)
            results.append(name)
        return f"{socket.gethostname()}: patched {', '.join(results)}"
    
    # Get all nodes
    nodes = ray.nodes()
    node_ips = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]
    print(f"[INFO] Patching verl on {len(node_ips)} nodes: {node_ips}")
    
    # Apply patches on all nodes
    tasks = []
    for node_ip in node_ips:
        task = apply_patches.options(
            resources={f"node:{node_ip}": 0.001}
        ).remote(patch_data)
        tasks.append((node_ip, task))
    
    # Wait for all patches to complete
    success = True
    for node_ip, task in tasks:
        try:
            result = ray.get(task, timeout=60)
            print(f"[INFO] {result}")
        except Exception as e:
            print(f"[ERROR] Failed to patch on {node_ip}: {e}")
            success = False
    
    if success:
        print(f"[INFO] ✓ All verl patches applied on all nodes")
    return success


# Keep old name as alias for backward compatibility
def patch_sglang_on_all_nodes():
    return patch_verl_on_all_nodes()


def distribute_sbys_grpo_to_all_nodes():
    """
    Distribute the sbys_grpo.py file to all nodes in the Ray cluster.
    
    This ensures all workers have the latest version of the training script,
    even if the shared filesystem has caching issues or delays.
    """
    current_file = os.path.abspath(__file__)
    target_path = current_file  # Same path on all nodes
    
    # Read the current file
    with open(current_file, 'rb') as f:
        file_content = f.read()
    print(f"[INFO] Read {current_file} ({len(file_content)} bytes)")
    
    # Also distribute helper files in the same directory
    script_dir = os.path.dirname(current_file)
    files_to_distribute = [
        (file_content, target_path, "sbys_grpo.py"),
    ]
    
    # Add math_checker.py if it exists
    math_checker_path = os.path.join(script_dir, "math_checker.py")
    if os.path.exists(math_checker_path):
        with open(math_checker_path, 'rb') as f:
            files_to_distribute.append((f.read(), math_checker_path, "math_checker.py"))
        print(f"[INFO] Read math_checker.py")
    
    # Add system_prompt file if it exists
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, 'rb') as f:
            files_to_distribute.append((f.read(), SYSTEM_PROMPT_FILE, os.path.basename(SYSTEM_PROMPT_FILE)))
        print(f"[INFO] Read {os.path.basename(SYSTEM_PROMPT_FILE)}")
    
    @ray.remote
    def write_files(files_list: list) -> str:
        """Write files on this node."""
        import os
        import socket
        results = []
        for content, target, name in files_list:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'wb') as f:
                f.write(content)
            results.append(name)
        return f"{socket.gethostname()}: wrote {', '.join(results)}"
    
    # Get all nodes
    nodes = ray.nodes()
    node_ips = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]
    print(f"[INFO] Distributing sbys_grpo.py and dependencies to {len(node_ips)} nodes: {node_ips}")
    
    # Distribute to all nodes
    tasks = []
    for node_ip in node_ips:
        task = write_files.options(
            resources={f"node:{node_ip}": 0.001}
        ).remote(files_to_distribute)
        tasks.append((node_ip, task))
    
    # Wait for all writes to complete
    success = True
    for node_ip, task in tasks:
        try:
            result = ray.get(task, timeout=60)
            print(f"[INFO] {result}")
        except Exception as e:
            print(f"[ERROR] Failed to distribute to {node_ip}: {e}")
            success = False
    
    if success:
        print(f"[INFO] ✓ sbys_grpo.py distributed to all nodes")
    return success


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
    
    # Verify files exist on all nodes
    @ray.remote
    def verify_checkpoint_files(target_path: str, expected_files: list):
        """Verify checkpoint files exist on this node."""
        import os
        import socket
        missing = []
        size_mismatch = []
        for filename, expected_size in expected_files:
            filepath = os.path.join(target_path, filename)
            if not os.path.exists(filepath):
                missing.append(filename)
            elif os.path.getsize(filepath) != expected_size:
                size_mismatch.append((filename, os.path.getsize(filepath), expected_size))
        return {
            'hostname': socket.gethostname(),
            'missing': missing,
            'size_mismatch': size_mismatch,
            'total_checked': len(expected_files)
        }
    
    # Build expected files list with sizes
    expected_files = [(f, len(d)) for f, d in files_data.items()]
    
    print(f"[Distribute] Verifying files on all nodes...")
    verify_tasks = []
    for node_ip in node_ips:
        task = verify_checkpoint_files.options(
            resources={f"node:{node_ip}": 0.001}
        ).remote(target_actor_dir, expected_files)
        verify_tasks.append((node_ip, task))
    
    all_verified = True
    for node_ip, task in verify_tasks:
        try:
            result = ray.get(task, timeout=120)
            if result['missing'] or result['size_mismatch']:
                print(f"[Distribute] VERIFICATION FAILED on {result['hostname']}!")
                if result['missing']:
                    print(f"  Missing files: {result['missing'][:5]}...")
                if result['size_mismatch']:
                    print(f"  Size mismatch: {result['size_mismatch'][:3]}...")
                all_verified = False
            else:
                print(f"[Distribute] ✓ Verified {result['total_checked']} files on {result['hostname']}")
        except Exception as e:
            print(f"[Distribute] Verification failed on {node_ip}: {e}")
            all_verified = False
    
    if not all_verified:
        print("[Distribute] ABORT: File verification failed!")
        return False
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
        "GLOO_TIMEOUT_MS": "2700000",  # 45 min - Gloo collective timeout
        # PyTorch distributed timeout (affects init_process_group default)
        "TORCH_DISTRIBUTED_TIMEOUT": "2700",  # 45 min in seconds
        "TORCH_DISTRIBUTED_DEBUG": "OFF",  # Set to DETAIL for debugging
        # Ray gRPC keepalive settings - prevent "keepalive watchdog timeout" during slow model init
        "RAY_grpc_keepalive_time_ms": "60000",  # Send keepalive ping every 60s (default 10s)
        "RAY_grpc_keepalive_timeout_ms": "600000",  # Wait 10 min for keepalive response (default 20s)
        "RAY_health_check_initial_delay_ms": "60000",  # Delay first health check by 60s
        "RAY_health_check_period_ms": "60000",  # Health check every 60s
        "RAY_health_check_timeout_ms": "600000",  # 10 min timeout for health checks
        "RAY_health_check_failure_threshold": "10",  # Allow 10 failures before considering dead
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
    print(f"Hint Selection: {'DYNAMIC (binary search via ProblemStateActor)' if DYNAMIC_HINT_SELECTION else 'STATIC (fixed at len(sbys_solution) // 2)'}")
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
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",  # High utilization - FSDP offloads weights to CPU
        "actor_rollout_ref.rollout.prompt_length=4096",  # Increased from 2560 to handle multi-turn prompt updates
        "actor_rollout_ref.rollout.response_length=16384",  # 16k for Turn 2
        "actor_rollout_ref.rollout.max_model_len=20480",  # 4096 + 16384
        "actor_rollout_ref.rollout.max_num_batched_tokens=1310720",  # 20480*64 (intentionally large)
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",  # Reduced to avoid OOM in actor update
        "actor_rollout_ref.rollout.load_format=auto",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.free_cache_engine=true",  # Free KV cache between generation and FSDP update phases
        
        # Validation rollout config - 2 samples per prompt for @2 metrics
        "actor_rollout_ref.rollout.val_kwargs.n=2",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        
        # Ray cluster config - connect to existing cluster
        "++ray_kwargs.ray_init.address=auto",
        
        # Actor config
        f"actor_rollout_ref.actor.ppo_mini_batch_size={TRAIN_BATCH_SIZE}",  # Must be <= train_batch_size
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",  # Reduced to avoid OOM in actor update (lm_head needs ~32GB)
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=false",  # Disabled - using smaller micro batch instead
        "+actor_rollout_ref.actor.offload_param=true",  # Offload weights to CPU - frees GPU for sglang KV cache
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
        "data.max_response_length=16384",  # Match response_length for Turn 2 (16k)
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
        "trainer.save_freq=15",  # Save checkpoint every N steps
        "+trainer.save_checkpoint_before_train=true",  # Save step 0 checkpoint as sanity check
        "trainer.val_before_train=true",  # Initial evaluation DISABLED to save time
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
            #"actor_rollout_ref.rollout.mode=async",  # Use AsyncLLM for better request scheduling
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
        
        # Distribute sbys_grpo.py to all nodes FIRST (ensures all workers have latest code)
        print("[INFO] Distributing sbys_grpo.py to all nodes...")
        distribute_sbys_grpo_to_all_nodes()
        
        # Only initialize ProblemStateActor if DYNAMIC_HINT_SELECTION is enabled
        if DYNAMIC_HINT_SELECTION:
            # Generate unique run_id for this training run
            import time as _time
            run_id = f"run_{int(_time.time())}_{uuid.uuid4().hex[:8]}"
            set_current_run_id(run_id)
            print(f"[INFO] Generated run_id: {run_id}")
            
            # Kill any existing ProblemStateActor to ensure fresh state
            # This is CRITICAL - even without detached lifetime, actors may persist
            # in some edge cases. Always kill to be safe.
            print("[INFO] Killing any existing ProblemStateActor to ensure fresh state...")
            try:
                existing_actor = ray.get_actor(PROBLEM_STATE_ACTOR_NAME, namespace="sbys_grpo")
                # Check if it's a stale actor before killing
                try:
                    old_run_id = ray.get(existing_actor.get_run_id.remote(), timeout=10)
                    print(f"[INFO] Found existing actor with run_id={old_run_id[:16]}...")
                except Exception as e:
                    old_run_id = "unknown"
                    print(f"[INFO] Could not get run_id from existing actor: {e}")
                ray.kill(existing_actor)
                print(f"[INFO] Killed existing ProblemStateActor (was run_id={old_run_id[:16] if old_run_id else 'unknown'}...)")
                _time.sleep(2)  # Give Ray time to clean up
            except ValueError:
                print("[INFO] No existing ProblemStateActor found (good)")
            
            # Create the shared ProblemStateActor before training starts
            # This ensures all workers can access the same actor instance
            print(f"[INFO] Creating fresh ProblemStateActor with run_id={run_id}...")
            state_actor = get_or_create_problem_state_actor(run_id=run_id)
            print(f"[INFO] ProblemStateActor ready: {state_actor}")
            
            # Reset all counters to ensure clean state (belt and suspenders)
            print("[INFO] Resetting all counters to ensure clean state...")
            reset_count = ray.get(state_actor.reset_all_counters.remote())
            if reset_count > 0:
                print(f"[WARNING] Reset {reset_count} stale counters - this indicates previous run crashed mid-step")
            else:
                print("[INFO] No stale counters found (clean state)")
            
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
        else:
            print("[INFO] DYNAMIC_HINT_SELECTION=False, skipping ProblemStateActor initialization")
            print("[INFO] Using fixed try_index = len(sbys_solution) // 2 for all problems")
    
    # Import run_ppo here to avoid import issues
    from verl.trainer.main_ppo import run_ppo
    
    # Small delay to ensure Ray cluster is fully synced before starting distributed training
    import time
    print("[INFO] Waiting 5 seconds for Ray cluster to stabilize...")
    time.sleep(5)
    
    # Patch sglang_rollout.py on all nodes (increases watchdog timeout for long sequences)
    print("[INFO] Patching sglang_rollout.py on all nodes...")
    patch_sglang_on_all_nodes()
    
    # Run PPO training
    print("Starting GRPO training with verl...")
    update_worker_state(phase="training_start")
    try:
        run_ppo(config)
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
