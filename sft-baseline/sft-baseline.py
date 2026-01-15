#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) baseline script for Qwen Math model.

Trains on cut_proofs_dataset (problem + solution pairs) with periodic evaluation
on a holdout set from the Omni-MATH dataset.

Uses verl's FSDP SFT trainer with custom evaluation callback.
"""

import os
import re
import sys
import json
import argparse
import datetime
import tempfile
import torch
import pandas as pd
from typing import List, Dict, Optional, Any
from tqdm import tqdm

# Environment setup
os.environ["WANDB_SAVE_CODE"] = "true"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce memory fragmentation

import apple_bolt as bolt

# Token file path (in hinting folder, not tracked by git)
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "../hinting/hf_token.txt")

# Paths (relative to task_runtime)
DEFAULT_MODEL_PATH = "Qwen/Qwen2-Math-7B-Instruct"
TRAIN_DATASET_PATH = "../newopenaioutputs/cut_proofs_dataset"
EVAL_DATASET_PATH = "../newopenaioutputs/hints_dataset"

# Training configuration
NUM_NODES = 1
GPUS_PER_NODE = 8
VAL_SIZE = 256  # Number of samples for evaluation
MAX_TRAIN_SAMPLES = None  # None = use all

# System prompt for evaluation (problem solving without hints)
EVAL_SYSTEM_PROMPT = """You are a mathematics expert. Solve the given problem step by step, showing all your work and reasoning. Put your final answer in \\box{...} format at the end."""


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


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} at the end of text."""
    matches = list(re.finditer(r'\\boxed?\{', text))
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


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute accuracy of predictions vs ground truths."""
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_answer = extract_boxed_answer(pred)
        pred_norm = normalize_answer(pred_answer)
        gt_norm = normalize_answer(gt)
        
        if pred_norm and gt_norm and pred_norm == gt_norm:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def create_sft_dataset(tokenizer, dataset_path: str, max_samples: Optional[int] = None):
    """Create SFT training dataset from cut_proofs_dataset.
    
    Format: problem -> solution (for supervised fine-tuning)
    """
    from datasets import load_from_disk
    
    dataset = load_from_disk(dataset_path)
    
    if max_samples is not None and max_samples > 0:
        original_size = len(dataset)
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"[INFO] Limited dataset from {original_size} to {len(dataset)} samples")
    
    # Format for SFT: messages format with system, user (problem), assistant (solution)
    sft_data = []
    for item in tqdm(dataset, desc="Formatting SFT data"):
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        answer = item.get('answer', '')
        
        if not problem or not solution:
            continue
        
        # Ensure solution ends with boxed answer if we have the answer
        if answer and '\\box' not in solution:
            solution = solution.strip() + f"\n\n\\boxed{{{answer}}}"
        
        # Create chat format
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
            {"role": "assistant", "content": solution}
        ]
        
        sft_data.append({
            "messages": messages,
            "problem": problem,
            "answer": answer
        })
    
    print(f"[INFO] Created {len(sft_data)} SFT samples")
    return sft_data


def create_eval_dataset(tokenizer, dataset_path: str, val_size: int = 256):
    """Create evaluation dataset from hints_dataset (holdout).
    
    Uses problems without hints for evaluation.
    """
    from datasets import load_from_disk
    
    dataset = load_from_disk(dataset_path)
    
    # Take last val_size samples as holdout
    total_size = len(dataset)
    start_idx = max(0, total_size - val_size)
    eval_dataset = dataset.select(range(start_idx, total_size))
    
    eval_data = []
    for item in tqdm(eval_dataset, desc="Formatting eval data"):
        problem = item.get('problem', '')
        final_answer = item.get('final_answer', '')
        
        if not problem or not final_answer:
            continue
        
        # Format prompt for generation
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"}
        ]
        
        eval_data.append({
            "messages": messages,
            "problem": problem,
            "ground_truth": final_answer
        })
    
    print(f"[INFO] Created {len(eval_data)} eval samples")
    return eval_data


def run_evaluation(
    model,
    tokenizer,
    eval_data: List[Dict],
    device: str = "cuda",
    max_new_tokens: int = 1536,
    batch_size: int = 4,
    temperature: float = 0.0,
) -> Dict[str, float]:
    """Run evaluation on the model and compute accuracy.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        eval_data: List of eval samples with 'messages' and 'ground_truth'
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        temperature: Sampling temperature (0 for greedy)
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    predictions = []
    ground_truths = []
    
    # Process in batches
    for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
        batch = eval_data[i:i + batch_size]
        
        # Prepare prompts
        prompts = []
        for item in batch:
            prompt = tokenizer.apply_chat_template(
                item['messages'],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
            ground_truths.append(item['ground_truth'])
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2560
        ).to(device)
        
        # Generate
        with torch.no_grad():
            if temperature == 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        
        # Decode only the generated part
        for j, output in enumerate(outputs):
            input_len = inputs['input_ids'][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            predictions.append(generated)
    
    # Compute accuracy
    accuracy = compute_accuracy(predictions, ground_truths)
    
    # Log some examples
    print("\n[Eval] Sample predictions:")
    for i in range(min(3, len(predictions))):
        pred_answer = extract_boxed_answer(predictions[i])
        print(f"  GT: {ground_truths[i][:50]}... | Pred: {pred_answer}")
    
    return {
        "eval/accuracy": accuracy,
        "eval/num_samples": len(predictions),
        "eval/num_correct": int(accuracy * len(predictions))
    }


def save_sft_parquet(data: List[Dict], output_path: str, tokenizer):
    """Save SFT data in parquet format compatible with verl."""
    # Convert to format verl expects
    records = []
    for item in data:
        # verl's SFTDataset expects 'prompt' and 'response' or 'messages'
        # For chat format, we use messages which will be processed by chat_template
        records.append({
            "messages": item["messages"]
        })
    
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved {len(records)} samples to {output_path}")
    return output_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SFT Baseline Training")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to base model (default: Qwen2-Math-7B-Instruct)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Global training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,  # Reduced by 10x from 1e-5
        help="Learning rate"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1,
        help="Run evaluation every N steps"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,  # Reduced from 4096 to fit in GPU memory
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,  # Reduced for memory safety during generation
        help="Batch size for evaluation generation"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=VAL_SIZE,
        help="Number of evaluation samples"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (None = all)"
    )
    parser.add_argument(
        "--skip-generation-eval",
        action="store_true",
        help="Skip generation-based evaluation (only use loss-based validation)"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    model_path = args.model_path or DEFAULT_MODEL_PATH
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_model_name = model_path.split('/')[-1].lower()
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            bolt.ARTIFACT_DIR,
            f"sft_baseline_{timestamp}_from_{base_model_name}"
        )
    
    experiment_name = f"sft_baseline_{timestamp}"
    
    print("=" * 80)
    print("SFT Baseline Training")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Eval frequency: {args.eval_freq} steps")
    print(f"Eval samples: {args.eval_samples}")
    print("=" * 80)
    
    # Login to services
    tokens = load_tokens()
    login_huggingface(tokens)
    login_wandb(tokens)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("\nCreating datasets...")
    train_data = create_sft_dataset(
        tokenizer, 
        TRAIN_DATASET_PATH,
        max_samples=args.max_train_samples
    )
    
    # Create eval data for generation eval (skip if flag is set)
    if args.skip_generation_eval:
        print("[INFO] Skipping generation evaluation (--skip-generation-eval)")
        eval_data = []
    else:
        eval_data = create_eval_dataset(
            tokenizer,
            EVAL_DATASET_PATH,
            val_size=args.eval_samples
        )
    
    # Save datasets to parquet
    print("\nSaving datasets to parquet...")
    os.makedirs('/mnt/tmp', exist_ok=True)
    
    train_parquet = f'/mnt/tmp/sft_train_{timestamp}.parquet'
    val_parquet = f'/mnt/tmp/sft_val_{timestamp}.parquet'
    
    save_sft_parquet(train_data, train_parquet, tokenizer)
    # For validation loss, we use a subset of training data
    val_subset = train_data[:min(500, len(train_data))]
    save_sft_parquet(val_subset, val_parquet, tokenizer)
    
    # Store eval_data for later use in evaluation callback
    eval_data_path = f'/mnt/tmp/eval_data_{timestamp}.json'
    with open(eval_data_path, 'w') as f:
        json.dump(eval_data, f)
    print(f"[INFO] Saved eval data to {eval_data_path}")
    
    # Build Hydra config for verl SFT trainer
    print("\nLoading verl configuration...")
    
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import verl
    
    verl_path = os.path.dirname(verl.__file__)
    config_path = os.path.join(verl_path, "trainer", "config")
    
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_path, version_base=None)
    
    overrides = [
        # Model
        f"model.partial_pretrain={model_path}",
        "model.trust_remote_code=true",
        "model.enable_gradient_checkpointing=true",
        "model.strategy=fsdp",
# NOTE: CPU offloading is disabled because it makes generation eval extremely slow (CPU inference)
        # If you hit OOM, reduce max_length or batch_size instead
        # "model.fsdp_config.cpu_offload=true",  # Offload optimizer to CPU to save GPU memory
        # "model.fsdp_config.offload_params=true",  # Offload params to CPU
        
        # Data
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        f"data.train_batch_size={args.batch_size}",
        "data.micro_batch_size_per_gpu=1",
        f"data.max_length={args.max_length}",
        "data.truncation=left",
        # Use multiturn format for messages
        "data.multiturn.enable=true",
        "data.multiturn.messages_key=messages",
        
        # Optimizer
        f"optim.lr={args.lr}",
        "optim.warmup_steps_ratio=0.1",
        "optim.clip_grad=1.0",
        
        # Trainer
        f"trainer.project_name=sft-math-baseline",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.total_epochs={args.epochs}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.eval_freq}",
        f"trainer.nnodes={NUM_NODES}",
        f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
        "trainer.logger=['console','wandb']",
    ]
    
    config = compose(config_name="sft_trainer", overrides=overrides)
    
    print("Configuration loaded successfully")
    print(f"  - Model: {config.model.partial_pretrain}")
    print(f"  - Train batch size: {config.data.train_batch_size}")
    print(f"  - Epochs: {config.trainer.total_epochs}")
    
    # Custom SFT trainer with evaluation callback
    print("\n" + "=" * 80)
    print("Starting SFT training with evaluation...")
    print("=" * 80)
    
    # Import and run verl's SFT trainer with our evaluation extension
    from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, run_sft, create_sft_dataset as verl_create_sft_dataset
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer
    from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
    from verl.utils.device import get_device_name
    from torch.distributed.device_mesh import init_device_mesh
    
    # Create a custom trainer that adds evaluation
    class SFTTrainerWithEval(FSDPSFTTrainer):
        """Extended SFT trainer with generation-based evaluation."""
        
        def __init__(self, *args, eval_data=None, eval_batch_size=4, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_data = eval_data or []
            self.eval_batch_size = eval_batch_size
            self._wandb_run = None
        
        def run_generation_eval(self, step: int):
            """Run generation-based evaluation and log to wandb."""
            if not self.eval_data:
                return {}
            
            if self.device_mesh.get_rank() != 0:
                return {}
            
            print(f"\n[Step {step}] Running generation evaluation on {len(self.eval_data)} samples...")
            
            # Run evaluation
            metrics = run_evaluation(
                model=self.fsdp_model,
                tokenizer=self.tokenizer,
                eval_data=self.eval_data,
                device=self.device_name,
                max_new_tokens=1536,
                batch_size=self.eval_batch_size,
                temperature=0.0
            )
            
            print(f"[Step {step}] Eval accuracy: {metrics['eval/accuracy']:.4f} "
                  f"({metrics['eval/num_correct']}/{metrics['eval/num_samples']})")
            
            return metrics
        
        def fit(self):
            """Override fit to add generation evaluation."""
            rank = self.device_mesh.get_rank()
            
            # Import tracking
            from verl.utils.tracking import Tracking
            
            if rank == 0:
                tracking = Tracking(
                    project_name=self.config.trainer.project_name,
                    experiment_name=self.config.trainer.experiment_name,
                    default_backend=self.config.trainer.logger,
                )
            
            global_step = 0
            last_valid_metric = None
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            
            if self.config.trainer.total_training_steps is not None:
                total_training_steps = self.config.trainer.total_training_steps
            
            self.total_training_steps = total_training_steps
            print(f"Total training steps: {self.total_training_steps}")
            
            for epoch in range(self.config.trainer.total_epochs):
                self.train_sampler.set_epoch(epoch=epoch)
                
                for data in tqdm(
                    self.train_dataloader,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0
                ):
                    global_step += 1
                    from tensordict import TensorDict
                    data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                    metric = self.training_step(data)
                    
                    if rank == 0:
                        tracking.log(data=metric, step=global_step)
                    
                    is_last_step = global_step >= self.total_training_steps
                    is_valid_step = global_step % self.config.trainer.test_freq == 0
                    is_save_step = global_step % self.config.trainer.save_freq == 0
                    
                    # Validation (loss-based)
                    if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                        val_losses = []
                        for val_data in self.val_dataloader:
                            val_data = TensorDict(
                                val_data, 
                                batch_size=self.config.data.micro_batch_size_per_gpu
                            ).to(self.device_name)
                            val_loss = self.validation_step(val_data)
                            val_losses.append(val_loss)
                        
                        if rank == 0:
                            val_loss = torch.mean(torch.stack(val_losses))
                            metric = {"val/loss": val_loss.detach().item()}
                            tracking.log(data=metric, step=global_step)
                            last_valid_metric = metric
                        
                        # Generation-based evaluation
                        eval_metrics = self.run_generation_eval(global_step)
                        if rank == 0 and eval_metrics:
                            tracking.log(data=eval_metrics, step=global_step)
                        
                        torch.distributed.barrier()
                    
                    # Save checkpoint
                    if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                        self.save_checkpoint(step=global_step)
                    
                    if is_last_step:
                        if rank == 0:
                            print(f"Final validation metrics: {last_valid_metric}")
                        return
    
    # Run custom trainer
    def run_sft_with_eval(config, eval_data, eval_batch_size):
        """Run SFT training with generation-based evaluation."""
        device_name = get_device_name()
        local_rank, rank, world_size = initialize_global_process_group()
        
        device_mesh = init_device_mesh(
            device_type=device_name,
            mesh_shape=(world_size,),
            mesh_dim_names=("fsdp",)
        )
        
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(
            device_type=device_name,
            mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
            mesh_dim_names=("dp", "sp")
        )
        
        # Load tokenizer
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
        
        # Create datasets
        train_dataset = verl_create_sft_dataset(config.data.train_files, config.data, tokenizer)
        val_dataset = verl_create_sft_dataset(config.data.val_files, config.data, tokenizer)
        
        # Create trainer with eval
        trainer = SFTTrainerWithEval(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_data=eval_data,
            eval_batch_size=eval_batch_size
        )
        
        trainer.fit()
        destroy_global_process_group()
    
    try:
        run_sft_with_eval(config, eval_data, args.eval_batch_size)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        for f in [train_parquet, val_parquet, eval_data_path]:
            if os.path.exists(f):
                os.unlink(f)
                print(f"Cleaned up: {f}")


if __name__ == "__main__":
    main()

