#!/usr/bin/env python3
"""
Crash Log Viewer - View crash logs from all nodes in one place.

Usage:
    python view_crash_logs.py              # View recent crashes
    python view_crash_logs.py --all        # View all crashes
    python view_crash_logs.py --heartbeat  # View heartbeats (shows which nodes are alive)
    python view_crash_logs.py --timeline   # View crashes sorted by time across all nodes
    python view_crash_logs.py --node <ip>  # View crashes from specific node
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict

CRASH_LOG_DIR = os.path.join(os.path.dirname(__file__), "crash_logs")

def load_jsonl(filepath):
    """Load JSONL file and return list of entries."""
    entries = []
    if not os.path.exists(filepath):
        return entries
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries

def format_timestamp(ts):
    """Format timestamp for display."""
    try:
        # Format: 20260126_125210_715371
        dt = datetime.strptime(ts[:15], "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ts

def get_node_short_name(node_id):
    """Extract short node name from full node_id."""
    # node_id format: hostname_ip_pid
    parts = node_id.split("_")
    if len(parts) >= 2:
        return parts[1]  # Return IP
    return node_id

def view_crashes(args):
    """View crash logs."""
    crashes_file = os.path.join(CRASH_LOG_DIR, "all_crashes.jsonl")
    entries = load_jsonl(crashes_file)
    
    if not entries:
        print("No crash logs found.")
        return
    
    # Filter by node if specified
    if args.node:
        entries = [e for e in entries if args.node in e.get("node_id", "")]
    
    # Filter to recent if not --all
    if not args.all:
        entries = entries[-20:]  # Last 20 entries
    
    # Sort by timestamp
    entries.sort(key=lambda x: x.get("timestamp", ""))
    
    print(f"\n{'='*80}")
    print(f"CRASH LOGS ({len(entries)} entries)")
    print(f"{'='*80}\n")
    
    for entry in entries:
        ts = format_timestamp(entry.get("timestamp", "unknown"))
        node = get_node_short_name(entry.get("node_id", "unknown"))
        error_type = entry.get("error_type", "unknown")
        error_msg = entry.get("error_msg", "")[:200]
        uptime = entry.get("uptime_seconds", 0)
        
        # Get GPU info if available
        gpu_info = ""
        system_info = entry.get("system_info", {})
        if system_info.get("gpu_memory"):
            gpus = system_info["gpu_memory"]
            gpu_info = f" | GPU: {', '.join([f'{g[\"allocated_gb\"]:.1f}/{g[\"total_gb\"]:.1f}GB' for g in gpus])}"
        
        # Get worker state
        worker_state = ""
        if system_info.get("worker_state"):
            ws = system_info["worker_state"]
            worker_state = f" | Phase: {ws.get('phase', 'unknown')}"
        
        print(f"[{ts}] {node} | {error_type}")
        print(f"  Uptime: {uptime:.0f}s{gpu_info}{worker_state}")
        print(f"  Error: {error_msg}")
        
        # Show stack trace snippet if available
        if args.verbose and entry.get("stack_trace"):
            stack = entry["stack_trace"]
            # Get last 5 lines of stack trace
            lines = stack.strip().split("\n")
            if len(lines) > 5:
                print(f"  Stack (last 5 lines):")
                for line in lines[-5:]:
                    print(f"    {line}")
            else:
                print(f"  Stack: {stack[:500]}")
        
        print()

def view_heartbeats(args):
    """View heartbeat logs to see which nodes are alive."""
    heartbeat_file = os.path.join(CRASH_LOG_DIR, "heartbeats.jsonl")
    entries = load_jsonl(heartbeat_file)
    
    if not entries:
        print("No heartbeat logs found.")
        return
    
    # Group by node
    nodes = defaultdict(list)
    for entry in entries:
        node = get_node_short_name(entry.get("node_id", "unknown"))
        nodes[node].append(entry)
    
    print(f"\n{'='*80}")
    print(f"HEARTBEAT STATUS ({len(nodes)} nodes)")
    print(f"{'='*80}\n")
    
    for node, heartbeats in sorted(nodes.items()):
        last_hb = heartbeats[-1]
        ts = format_timestamp(last_hb.get("timestamp", "unknown"))
        uptime = last_hb.get("uptime_seconds", 0)
        worker_state = last_hb.get("worker_state", {})
        phase = worker_state.get("phase", "unknown")
        
        # GPU info
        gpu_info = ""
        if last_hb.get("gpu_memory"):
            gpus = last_hb["gpu_memory"]
            gpu_info = f" | GPU: {', '.join([f'{g[\"allocated_gb\"]:.1f}/{g[\"total_gb\"]:.1f}GB' for g in gpus])}"
        
        print(f"Node: {node}")
        print(f"  Last heartbeat: {ts}")
        print(f"  Uptime: {uptime:.0f}s | Phase: {phase}{gpu_info}")
        print(f"  Total heartbeats: {len(heartbeats)}")
        print()

def view_timeline(args):
    """View crashes in timeline format across all nodes."""
    crashes_file = os.path.join(CRASH_LOG_DIR, "all_crashes.jsonl")
    entries = load_jsonl(crashes_file)
    
    if not entries:
        print("No crash logs found.")
        return
    
    # Sort by timestamp
    entries.sort(key=lambda x: x.get("timestamp", ""))
    
    # Group by timestamp (within 1 second)
    print(f"\n{'='*80}")
    print(f"CRASH TIMELINE")
    print(f"{'='*80}\n")
    
    current_time = None
    for entry in entries:
        ts = entry.get("timestamp", "")[:15]  # Group by second
        node = get_node_short_name(entry.get("node_id", "unknown"))
        error_type = entry.get("error_type", "unknown")
        
        if ts != current_time:
            current_time = ts
            print(f"\n[{format_timestamp(ts)}]")
        
        # Color-code by error type
        prefix = "  "
        if "SIGNAL" in error_type:
            prefix = "  ⚡"
        elif "EXCEPTION" in error_type:
            prefix = "  ❌"
        elif "TIMEOUT" in error_type:
            prefix = "  ⏰"
        
        print(f"{prefix} {node}: {error_type}")

def clear_logs(args):
    """Clear all crash logs."""
    if not args.force:
        confirm = input("Are you sure you want to clear all crash logs? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    files_to_clear = [
        os.path.join(CRASH_LOG_DIR, "all_crashes.jsonl"),
        os.path.join(CRASH_LOG_DIR, "heartbeats.jsonl"),
    ]
    
    # Also clear node-specific logs
    if os.path.exists(CRASH_LOG_DIR):
        for f in os.listdir(CRASH_LOG_DIR):
            if f.startswith("crashes_") and f.endswith(".log"):
                files_to_clear.append(os.path.join(CRASH_LOG_DIR, f))
    
    for filepath in files_to_clear:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Removed: {filepath}")
    
    print("Crash logs cleared.")

def main():
    parser = argparse.ArgumentParser(description="View crash logs from all nodes")
    parser.add_argument("--all", "-a", action="store_true", help="Show all crashes (default: last 20)")
    parser.add_argument("--heartbeat", "-b", action="store_true", help="Show heartbeat status")
    parser.add_argument("--timeline", "-t", action="store_true", help="Show crash timeline")
    parser.add_argument("--node", "-n", type=str, help="Filter by node IP")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show stack traces")
    parser.add_argument("--clear", action="store_true", help="Clear all crash logs")
    parser.add_argument("--force", "-f", action="store_true", help="Force clear without confirmation")
    
    args = parser.parse_args()
    
    if args.clear:
        clear_logs(args)
    elif args.heartbeat:
        view_heartbeats(args)
    elif args.timeline:
        view_timeline(args)
    else:
        view_crashes(args)

if __name__ == "__main__":
    main()

