#!/usr/bin/env python3
"""
Train LoRA adapters with pre-batched data for Cauldron.

Production-ready training script that uses JSON configuration files
for all training parameters. Supports:
- Pre-batched dataset loading (no retokenization)
- Multiple optimizers: AdamW, Muon, NorMuon
- Advanced schedulers: constant, linear, cosine, wWSD, epoch shock absorber
- LoRA layer configuration with simple substring matching or regex patterns
- Split and single eval dataset support
- Gradient debugging for PEFT bug detection
- Automatic output directory naming from config
- Graceful wandb handling

Usage:
    python train_lora.py --config granite-h-hybrid-r256.json

    # With optional overrides:
    python train_lora.py \\
        --config granite-h-hybrid-r256.json \\
        --postfix experiment-1 \\
        --debug-gradients mamba_out \\
        --wandb-project my-project
"""

# Fix CUDA memory fragmentation BEFORE importing torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
import torch.nn.functional as F
import pickle
import json
import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import sys

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    TrainerCallback, EarlyStoppingCallback,
    AutoConfig, GenerationConfig,
    get_scheduler
)
from peft import get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset, DataLoader

import math
from torch.optim.lr_scheduler import LambdaLR


# ============================================================================
# wWSD Scheduler (warmup-Warmup-Stable-Decay)
# ============================================================================

def get_wwsd_schedule_with_warmup(
    optimizer,
    num_prewarmup_steps: int,
    prewarmup_lr_ratio: float,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
    """
    Create a schedule with prewarmup, warmup, stable, and cosine decay phases.

    Phases:
    1. Prewarmup: Linear ramp from 0 to prewarmup_lr_ratio * LR (e.g., 0 -> 0.1 * LR)
    2. Warmup: Linear ramp from prewarmup_lr_ratio to full LR (e.g., 0.1 * LR -> LR)
    3. Stable: Constant at full LR
    4. Decay: Cosine decay from full LR back to prewarmup_lr_ratio * LR (symmetric)

    Args:
        optimizer: Optimizer
        num_prewarmup_steps: Steps for prewarmup phase
        prewarmup_lr_ratio: Target LR for prewarmup as ratio of full LR (e.g., 0.1 = 10% of peak LR)
                           Also used as minimum LR for decay phase (symmetric design)
        num_warmup_steps: Steps for warmup phase (from prewarmup_lr to full LR)
        num_stable_steps: Steps for stable phase
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (default: 0.5 for half cycle)

    Returns:
        LambdaLR scheduler
    """
    # Calculate decay steps
    num_decay_steps = num_training_steps - num_prewarmup_steps - num_warmup_steps - num_stable_steps

    if num_decay_steps < 0:
        raise ValueError(
            f"Invalid step configuration: prewarmup({num_prewarmup_steps}) + "
            f"warmup({num_warmup_steps}) + stable({num_stable_steps}) > "
            f"total({num_training_steps})"
        )

    def lr_lambda(current_step: int):
        # Phase 1: Prewarmup (0 -> prewarmup_lr_ratio)
        if current_step < num_prewarmup_steps:
            if num_prewarmup_steps == 0:
                return prewarmup_lr_ratio
            return float(current_step) / float(max(1, num_prewarmup_steps)) * prewarmup_lr_ratio

        # Phase 2: Warmup (prewarmup_lr_ratio -> 1.0)
        elif current_step < num_prewarmup_steps + num_warmup_steps:
            progress = float(current_step - num_prewarmup_steps) / float(max(1, num_warmup_steps))
            return prewarmup_lr_ratio + (1.0 - prewarmup_lr_ratio) * progress

        # Phase 3: Stable (1.0)
        elif current_step < num_prewarmup_steps + num_warmup_steps + num_stable_steps:
            return 1.0

        # Phase 4: Cosine Decay (1.0 -> prewarmup_lr_ratio)
        # Symmetric: decay back to the same ratio where prewarmup ended
        else:
            progress = float(current_step - num_prewarmup_steps - num_warmup_steps - num_stable_steps)
            progress = progress / float(max(1, num_decay_steps))

            # Cosine decay from 1.0 to prewarmup_lr_ratio
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
            return prewarmup_lr_ratio + (1.0 - prewarmup_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class EpochShockAbsorberScheduler:
    """
    Wrapper scheduler that applies epoch shock absorption on top of a base scheduler.

    At epoch boundaries, applies a 0.5x LR multiplier, then linearly recovers
    to 1.0x over shock_steps.
    """

    def __init__(self, base_scheduler, steps_per_epoch: int, shock_steps: int):
        self.base_scheduler = base_scheduler
        self.steps_per_epoch = steps_per_epoch
        self.shock_steps = shock_steps
        self.current_step = 0
        self.optimizer = base_scheduler.optimizer

    def get_shock_multiplier(self, step):
        """Calculate shock multiplier for given step."""
        # No shock before we've completed one epoch
        if step <= self.steps_per_epoch:
            return 1.0

        # Check if we're at an epoch boundary (step = N+1, 2N+1, 3N+1, ...)
        # This happens when step % steps_per_epoch == 1
        steps_since_shock = (step % self.steps_per_epoch) - 1

        if steps_since_shock < 0:
            # step % steps_per_epoch == 0 means we're at step N, 2N, 3N, ...
            # Not at a boundary, just continue with last shock state
            return 1.0

        if steps_since_shock == 0:
            # We're at an epoch boundary! Apply shock
            return 0.5
        elif steps_since_shock <= self.shock_steps:
            # Linearly recover from 0.5 to 1.0
            progress = float(steps_since_shock) / float(self.shock_steps)
            return 0.5 + 0.5 * progress
        else:
            # Normal operation
            return 1.0

    def step(self):
        """Step both schedulers."""
        # Step the base scheduler
        self.base_scheduler.step()
        self.current_step += 1

        # Apply shock multiplier to all param groups
        multiplier = self.get_shock_multiplier(self.current_step)
        base_lrs = self.base_scheduler.get_last_lr()

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = base_lrs[i] * multiplier

    def get_last_lr(self):
        """Get current LR with shock multiplier applied."""
        multiplier = self.get_shock_multiplier(self.current_step)
        base_lrs = self.base_scheduler.get_last_lr()
        return [lr * multiplier for lr in base_lrs]

    def state_dict(self):
        """Return state dict."""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.current_step = state_dict['current_step']


# ============================================================================
# Configuration Loading and Validation
# ============================================================================

def load_training_config(config_path: str) -> dict:
    """
    Load and validate training configuration file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Validated config dictionary

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Validate required sections
    required_sections = ["model", "data", "lora", "optimizer", "scheduler", "training", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: {section}")

    # Set defaults
    if config["output"].get("prefix") is None:
        config["output"]["prefix"] = config_path.stem  # Filename without .json

    return config


# ============================================================================
# Output Directory and Wandb Setup
# ============================================================================

def setup_output_directory(config: dict, postfix: Optional[str] = None) -> Path:
    """
    Create and return the output directory path based on config.

    Args:
        config: Training configuration
        postfix: Optional postfix to append to directory name

    Returns:
        Path to output directory

    Raises:
        ValueError: If output config is invalid
    """
    root_dir = Path(config["output"]["root_dir"])
    prefix = config["output"]["prefix"]

    if not prefix:
        raise ValueError("Output prefix must be set in config")

    # Construct directory name
    if postfix:
        dir_name = f"{prefix}-{postfix}"
    else:
        dir_name = prefix

    output_dir = root_dir / dir_name

    # Create directory
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def setup_wandb(
    config: dict,
    output_dir: Path,
    project_override: Optional[str] = None,
    disable: bool = False
) -> bool:
    """
    Initialize Weights & Biases logging with graceful error handling.

    Args:
        config: Training configuration
        output_dir: Output directory path
        project_override: Optional wandb project name override
        disable: If True, skip wandb initialization

    Returns:
        True if wandb was successfully initialized, False otherwise

    Notes:
        - Does not raise errors on wandb initialization failure
        - Logs warning if wandb is not available or fails to init
        - Sets WANDB_DISABLED=true environment variable if disabled
    """
    if disable:
        os.environ["WANDB_DISABLED"] = "true"
        print("[Wandb] Disabled via --no-wandb flag")
        return False

    # Get project name: explicit override > config > output dir name
    project = project_override or config.get("wandb_project") or output_dir.name

    try:
        import wandb

        # Check for API key before attempting init
        if not os.environ.get("WANDB_API_KEY") and not wandb.api.api_key:
            print("[Wandb] Warning: No API key found (WANDB_API_KEY not set)")
            print("  Continuing without wandb logging")
            os.environ["WANDB_DISABLED"] = "true"
            return False

        # Initialize wandb
        wandb.init(
            project=project,
            name=output_dir.name,
            config={
                "model": config.get("model", {}),
                "lora": config.get("lora", {}),
                "optimizer": config.get("optimizer", {}),
                "scheduler": config.get("scheduler", {}),
                "training": config.get("training", {}),
            },
            dir=str(output_dir),
        )

        print(f"[Wandb] Initialized successfully")
        print(f"  Project: {project}")
        print(f"  Run name: {output_dir.name}")
        return True

    except ImportError:
        print("[Wandb] Warning: wandb package not installed")
        print("  Install with: pip install wandb")
        os.environ["WANDB_DISABLED"] = "true"
        return False

    except Exception as e:
        print(f"[Wandb] Warning: Failed to initialize wandb: {e}")
        print("  Continuing without wandb logging")
        os.environ["WANDB_DISABLED"] = "true"
        return False


def save_config_to_output_dir(config: dict, output_dir: Path) -> None:
    """
    Save the training configuration to the output directory.

    Args:
        config: Training configuration
        output_dir: Output directory path
    """
    config_path = output_dir / "training_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[Config] Saved to {config_path}")


# ============================================================================
# Dataset Loading
# ============================================================================

class PreBatchedDataset(Dataset):
    """
    Dataset that unbatches pre-batched data into individual sequences.

    Each effective batch file contains multiple on-device batches, each with
    multiple sequences. This dataset yields individual sequences so the
    DataLoader can re-batch them and Trainer's gradient accumulation works
    correctly across on-device batches.
    """

    def __init__(self, data_dir: Path, num_epochs: int, on_device_batch_size: int, grad_accum_steps: int):
        """
        Args:
            data_dir: Base directory containing epoch_XX subdirectories
            num_epochs: Number of epochs to load
            on_device_batch_size: Sequences per on-device batch
            grad_accum_steps: Expected on-device batches per effective batch
        """
        self.data_dir = Path(data_dir)
        self.on_device_batch_size = on_device_batch_size
        # Index: (file, on_device_batch_idx, sequence_idx_within_batch)
        self.sequence_index = []
        # Single-item file cache to avoid re-reading for consecutive sequences
        self._last_file = None
        self._last_data = None

        for epoch in range(num_epochs):
            epoch_dir = self.data_dir / f"epoch_{epoch:02d}"

            if not epoch_dir.exists():
                raise ValueError(f"Epoch directory not found: {epoch_dir}")

            effective_batch_files = sorted(epoch_dir.glob("effective_batch_*.pkl"))

            if not effective_batch_files:
                raise ValueError(f"No batch files found in {epoch_dir}")

            start_idx = len(self.sequence_index)

            for eb_file in effective_batch_files:
                with open(eb_file, "rb") as f:
                    effective_batch = pickle.load(f)
                num_odb = len(effective_batch["on_device_batches"])

                if num_odb != grad_accum_steps:
                    raise ValueError(
                        f"BROKEN BATCH FILE: {eb_file}\n"
                        f"  Expected {grad_accum_steps} on-device batches (from metadata)\n"
                        f"  Found {num_odb} on-device batches in file\n"
                        f"  This means the batching script produced inconsistent output!"
                    )

                for odb_idx in range(grad_accum_steps):
                    for seq_idx in range(on_device_batch_size):
                        self.sequence_index.append((eb_file, odb_idx, seq_idx))

            end_idx = len(self.sequence_index)
            print(f"[Dataset] Indexed epoch {epoch}: {len(effective_batch_files)} effective batches "
                  f"→ {end_idx - start_idx} sequences")

        print(f"[Dataset] Total: {len(self.sequence_index)} sequences across {num_epochs} epochs")

    def __len__(self):
        return len(self.sequence_index)

    def __getitem__(self, idx):
        """Load and return one sequence."""
        eb_file, odb_idx, seq_idx = self.sequence_index[idx]

        if eb_file != self._last_file:
            with open(eb_file, "rb") as f:
                effective_batch = pickle.load(f)
            self._last_file = eb_file
            self._last_data = effective_batch["on_device_batches"]

        on_device_batch = self._last_data[odb_idx]

        return {
            "input_ids": on_device_batch["input_ids"][seq_idx],
            "labels": on_device_batch["labels"][seq_idx],
            "sequence_length": on_device_batch["sequence_lengths"][seq_idx],
        }


class PreBatchedEvalDataset(Dataset):
    """
    Dataset for pre-batched evaluation data.

    Yields entire pre-batched on-device batches (no unbatching) to preserve
    the original bucketing and avoid double padding.
    """

    def __init__(self, data_dir: Path, on_device_batch_size: int):
        self.data_dir = Path(data_dir)
        self.on_device_batch_size = on_device_batch_size
        self.batch_files = sorted(data_dir.glob("batch_*.pkl"))

        if not self.batch_files:
            raise ValueError(f"No batch files found in {data_dir}")

        print(f"[Eval Dataset] Found {len(self.batch_files)} batches (batch_size={on_device_batch_size})")

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        batch_file = self.batch_files[idx]

        with open(batch_file, "rb") as f:
            batch = pickle.load(f)

        return {
            "input_ids": batch["input_ids"],
            "labels": batch["labels"],
            "sequence_lengths": batch["sequence_lengths"],
        }


# ============================================================================
# LoRA Configuration
# ============================================================================

def build_lora_config(config: dict) -> LoraConfig:
    """
    Build LoRA configuration from config dictionary.

    Args:
        config: Training configuration

    Returns:
        LoraConfig instance
    """
    lora_cfg = config["lora"]

    # Get base parameters
    base_rank = lora_cfg["base_rank"]
    base_alpha = lora_cfg["base_alpha"]
    dropout = lora_cfg.get("dropout", 0.0)
    use_rslora = lora_cfg.get("use_rslora", False)

    # Build target modules list
    target_modules = []
    layer_groups = lora_cfg.get("layer_groups", [])

    for group in layer_groups:
        if not group.get("enabled", True):
            continue

        target_modules.extend(group["modules"])

    if not target_modules:
        raise ValueError("No LoRA target modules enabled in config")

    # Build LoRA config
    return LoraConfig(
        r=base_rank,
        lora_alpha=base_alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=use_rslora,
    )


def apply_layer_specific_lora(model, config: dict) -> None:
    """
    Apply layer-specific LoRA rank/alpha to specific groups.

    This modifies the LoRA adapters after PEFT model creation to use
    different ranks/alphas for different layer groups.

    Args:
        model: PEFT model with LoRA adapters
        config: Training configuration
    """
    lora_cfg = config["lora"]
    layer_groups = lora_cfg.get("layer_groups", [])

    base_rank = lora_cfg["base_rank"]
    base_alpha = lora_cfg["base_alpha"]

    # Apply custom ranks/alphas to enabled groups
    for group in layer_groups:
        if not group.get("enabled", True):
            continue

        custom_rank = group.get("rank")
        custom_alpha = group.get("alpha")

        if custom_rank is None and custom_alpha is None:
            continue  # Use base values

        # Find matching modules
        for name, module in model.named_modules():
            # Check if this module matches any of the group's module patterns
            if any(pattern in name for pattern in group["modules"]):
                if hasattr(module, "lora_A"):
                    # Update rank dimension
                    if custom_rank is not None and custom_rank != base_rank:
                        # Re-initialize with new rank
                        # This is a simplified approach - in practice you may need
                        # more sophisticated handling
                        pass

                    # Update alpha
                    if custom_alpha is not None:
                        module.scaling = custom_alpha / (custom_rank or base_rank)


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(config: dict, bf16_patch: bool = False):
    """
    Load model and tokenizer from configuration.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = config["model"]["name"]

    print(f"[Model] Loading {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Apply BF16 logits patch if requested via CLI flag or config
    if bf16_patch or config["model"].get("bf16_logits_patch", False):
        print("[Model] Applying BF16 logits patch")
        apply_bf16_logits_patch(model)

    return model, tokenizer


def apply_bf16_logits_patch(model):
    """
    Patch model to ensure logits are in BF16.

    Some models return FP32 logits which can cause memory issues.
    This ensures logits stay in BF16.
    """
    original_forward = model.forward

    def patched_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if hasattr(outputs, "logits") and outputs.logits.dtype == torch.float32:
            outputs.logits = outputs.logits.to(torch.bfloat16)
        return outputs

    model.forward = patched_forward


# ============================================================================
# Custom Data Collator
# ============================================================================

class PreBatchedCollator:
    """
    Collator for pre-batched training and evaluation data.

    Training: receives individual sequences from PreBatchedDataset, stacks them
    into a batch, and builds an attention mask from stored sequence lengths.

    Eval: receives a single pre-batched item (sequence_lengths plural) from
    PreBatchedEvalDataset, converts numpy arrays to torch, and builds the mask.
    """

    def __call__(self, batch_list):
        # Eval path: single pre-batched item with sequence_lengths (plural)
        if len(batch_list) == 1 and "sequence_lengths" in batch_list[0]:
            item = batch_list[0]
            input_ids = torch.from_numpy(item["input_ids"]).long()
            labels = torch.from_numpy(item["labels"]).long()
            seq_lengths = item["sequence_lengths"]

            batch_size, seq_len = input_ids.shape
            attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
            for j, length in enumerate(seq_lengths):
                attention_mask[j, :length] = 1

            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        # Training path: list of individual sequences with sequence_length (singular)
        input_ids_list = []
        labels_list = []
        seq_lengths = []

        for item in batch_list:
            input_ids_list.append(torch.from_numpy(item["input_ids"]).long())
            labels_list.append(torch.from_numpy(item["labels"]).long())
            seq_lengths.append(item["sequence_length"])

        input_ids = torch.stack(input_ids_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        seq_len = input_ids.shape[1]
        attention_mask = torch.zeros(len(batch_list), seq_len, dtype=torch.long)
        for j, length in enumerate(seq_lengths):
            attention_mask[j, :length] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ============================================================================
# Custom Loss and Trainer
# ============================================================================

def compute_label_smoothed_loss(logits, labels, epsilon, ignore_index=-100):
    """
    Compute label-smoothed loss with sum reduction (unnormalized).

    Fixes the transformers LabelSmoother issue where loss is normalized
    per micro-batch instead of per effective batch across all gradient
    accumulation steps.

    Returns:
        loss_sum: Unnormalized loss sum (caller normalizes by total tokens)
        num_tokens: Number of non-padding tokens in this batch
    """
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    log_probs = -F.log_softmax(logits, dim=-1)

    padding_mask = labels.eq(ignore_index)
    labels_clamped = torch.clamp(labels, min=0)

    nll_loss = log_probs.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
    smoothed_loss = log_probs.sum(dim=-1)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    vocab_size = log_probs.shape[-1]
    nll_loss_sum = nll_loss.sum()
    smoothed_loss_sum = smoothed_loss.sum() / vocab_size
    num_tokens = (~padding_mask).sum()

    loss_sum = (1 - epsilon) * nll_loss_sum + epsilon * smoothed_loss_sum

    return loss_sum, num_tokens


class PreBatchedTrainer(Trainer):
    """
    Custom Trainer for pre-batched data.

    Uses SequentialSampler so the DataLoader respects the bucket ordering
    from the prebatcher — data is already shuffled at prep time.

    Also fixes label-smoothing normalization across gradient accumulation
    steps (the default transformers LabelSmoother normalizes per micro-batch,
    giving different effective loss weights for different GA splits).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing_factor = self.args.label_smoothing_factor

    def _get_train_sampler(self, train_dataset=None):
        from torch.utils.data import SequentialSampler
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    def _get_eval_sampler(self, eval_dataset):
        from torch.utils.data import SequentialSampler
        return SequentialSampler(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.label_smoothing_factor > 0:
            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            loss_sum, num_tokens_this_batch = compute_label_smoothed_loss(
                logits, labels,
                epsilon=self.label_smoothing_factor,
                ignore_index=-100
            )

            if num_items_in_batch is None:
                num_items_in_batch = num_tokens_this_batch

            if not hasattr(self, '_logged_label_smoothing_info'):
                print(f"[Label Smoothing] ε={self.label_smoothing_factor}, "
                      f"num_items_in_batch={num_items_in_batch}, tokens_this_batch={num_tokens_this_batch}")
                self._logged_label_smoothing_info = True

            loss = loss_sum / num_items_in_batch
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


# ============================================================================
# Training Setup
# ============================================================================

def setup_training(config: dict, output_dir: Path, num_epochs: int, on_device_batch_size: int, grad_accum_steps: int, bf16_patch: bool = False):
    """
    Complete training setup from configuration.

    Args:
        config: Training configuration
        output_dir: Output directory
        num_epochs: Number of epochs (from pre-batch metadata)
        on_device_batch_size: Sequences per on-device batch (from pre-batch metadata)
        grad_accum_steps: Gradient accumulation steps (from pre-batch metadata)

    Returns:
        Tuple of (model, tokenizer, train_dataset, eval_dataset)
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, bf16_patch=bf16_patch)

    # Apply LoRA
    print("[LoRA] Applying adapters")
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    print(f"[LoRA] Trainable layers ({len(trainable)} total):")
    for name, count in trainable[:20]:
        print(f"  {name}  ({count:,} params)")
    if len(trainable) > 20:
        remaining = len(trainable) - 20
        remaining_params = sum(c for _, c in trainable[20:])
        print(f"  ... and {remaining} more layers ({remaining_params:,} params)")

    # Load datasets
    train_dir = Path(config["data"]["train_dir"])
    eval_dir = Path(config["data"]["eval_dir"])

    train_dataset = PreBatchedDataset(
        train_dir,
        num_epochs=num_epochs,
        on_device_batch_size=on_device_batch_size,
        grad_accum_steps=grad_accum_steps,
    )
    eval_dataset = PreBatchedEvalDataset(eval_dir, on_device_batch_size=on_device_batch_size)

    return model, tokenizer, train_dataset, eval_dataset


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train LoRA adapters with pre-batched data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration JSON file"
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default=None,
        help="Optional postfix to append to output directory name"
    )
    parser.add_argument(
        "--debug-gradients",
        type=str,
        default=None,
        help="Enable gradient debugging for specific layer (e.g., 'out_proj')"
    )
    parser.add_argument(
        "--bf16-patch",
        action="store_true",
        help="Apply BF16 logits patch (saves VRAM on some models)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint: 'auto' for latest, a step number, or a full path. "
             "Overrides checkpoint.resume in config."
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (overrides config)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    args = parser.parse_args()

    # Load and validate configuration
    from config_utils import load_training_config, print_config_summary
    from optimizer_builder import build_optimizer, print_optimizer_summary
    from scheduler_builder import build_scheduler, print_scheduler_summary

    print("\n" + "="*80)
    print("CAULDRON LORA TRAINING")
    print("="*80)

    config = load_training_config(args.config)
    print_config_summary(config)

    # Setup output directory
    output_dir = setup_output_directory(config, postfix=args.postfix)
    print(f"\n[Output] Directory: {output_dir}")

    # Resolve checkpoint to resume from (CLI overrides config)
    resume_spec = args.resume or config.get("checkpoint", {}).get("resume")
    checkpoint_path = None
    if resume_spec:
        if str(resume_spec).lower() == "auto":
            checkpoints = sorted(
                output_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1])
            )
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                print(f"[Resume] Auto-detected checkpoint: {checkpoint_path}")
            else:
                print("[Resume] No checkpoints found, starting fresh")
        elif str(resume_spec).isdigit():
            checkpoint_path = output_dir / f"checkpoint-{resume_spec}"
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint not found: {checkpoint_path}")
            print(f"[Resume] Resuming from step: {checkpoint_path}")
        else:
            checkpoint_path = Path(resume_spec)
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint not found: {checkpoint_path}")
            print(f"[Resume] Resuming from: {checkpoint_path}")

    # Save config to output directory
    save_config_to_output_dir(config, output_dir)

    # Setup wandb
    wandb_enabled = setup_wandb(
        config,
        output_dir,
        project_override=args.wandb_project,
        disable=args.no_wandb
    )

    # Setup training
    # Load metadata before dataset creation — batch params come from here
    train_metadata_path = Path(config["data"]["train_dir"]) / "metadata.json"
    with open(train_metadata_path) as f:
        train_metadata = json.load(f)

    num_epochs = train_metadata["num_epochs"]
    on_device_batch_size = train_metadata["on_device_batch_size"]
    grad_accum_steps = train_metadata["grad_accum_steps"]

    print(f"\n[Metadata] Epochs: {num_epochs}, on-device batch: {on_device_batch_size}, grad accum: {grad_accum_steps}")

    model, tokenizer, train_dataset, eval_dataset = setup_training(
        config, output_dir,
        num_epochs=num_epochs,
        on_device_batch_size=on_device_batch_size,
        grad_accum_steps=grad_accum_steps,
        bf16_patch=args.bf16_patch,
    )

    # Dataset yields individual sequences; one optimizer step = on_device_batch_size * grad_accum_steps sequences
    total_steps = len(train_dataset) // (on_device_batch_size * grad_accum_steps)
    steps_per_epoch = total_steps // num_epochs

    print(f"\n[Training] Total sequences: {len(train_dataset)}")
    print(f"[Training] Total steps: {total_steps}")
    print(f"[Training] Steps per epoch: {steps_per_epoch}")
    print(f"[Training] Num epochs: {num_epochs}")

    # Build optimizer
    from optimizer_builder import build_optimizer
    optimizer = build_optimizer(config, model.parameters())
    print_optimizer_summary(optimizer)

    # Build scheduler
    from scheduler_builder import build_scheduler
    scheduler = build_scheduler(
        config,
        optimizer,
        num_training_steps=total_steps,
        steps_per_epoch=steps_per_epoch
    )
    print_scheduler_summary(scheduler, total_steps)

    # Setup callbacks
    from callbacks import (
        LiveLogCallback,
        CheckpointFlagCallback,
        FinalEvalCallback,
        GradientDebugCallback
    )

    callbacks = [
        LiveLogCallback(),
        FinalEvalCallback(),
    ]

    # Add gradient debug callback if requested
    if args.debug_gradients:
        callbacks.append(
            GradientDebugCallback(debug_target=args.debug_gradients, verbose=True)
        )

    # Add checkpoint flag callback if configured
    flag_file = config.get("checkpoint", {}).get("flag_file")
    if flag_file:
        callbacks.append(CheckpointFlagCallback(flag_file))

    # Create training arguments
    training_config = config["training"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,  # Dataset already spans all epochs
        per_device_train_batch_size=on_device_batch_size,
        per_device_eval_batch_size=1,  # Eval batches are pre-formed, don't rebatch
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        save_strategy="steps",
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=training_config.get("save_steps", 500),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_smoothing_factor=training_config.get("label_smoothing", 0.0),
        report_to="wandb" if wandb_enabled else "none",
    )

    # Create Trainer
    trainer = PreBatchedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PreBatchedCollator(),
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
    )

    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    trainer.train(resume_from_checkpoint=str(checkpoint_path) if checkpoint_path else None)

    # Save final model
    print("\n" + "="*80)
    print("SAVING FINAL MODEL")
    print("="*80)

    final_output_dir = output_dir / "final"
    trainer.save_model(final_output_dir)
    print(f"[Save] Model saved to {final_output_dir}")

    # Auto-merge if requested
    if config["output"].get("auto_merge", False):
        print("\n[Merge] Auto-merging LoRA adapters")
        merged_model = model.merge_and_unload()
        merged_output_dir = output_dir / "merged"
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        gen_config = GenerationConfig(top_p=0.95, temperature=0.6, do_sample=True)
        gen_config.save_pretrained(str(merged_output_dir))

        # PEFT merge saves the inner text subconfig (e.g. Qwen3_5TextConfig) instead
        # of the full wrapper config (Qwen3_5Config) that vLLM expects. Re-save the
        # original base model config to restore the correct config.json.
        from transformers import AutoConfig, AutoProcessor
        base_config = AutoConfig.from_pretrained(config["model"]["name"], trust_remote_code=True)
        base_config.save_pretrained(str(merged_output_dir))

        # Multimodal models (e.g. Qwen3.5) require preprocessor_config.json for vLLM
        # even when only language layers were trained. Save it if the base model has one.
        try:
            processor = AutoProcessor.from_pretrained(config["model"]["name"], trust_remote_code=True)
            processor.save_pretrained(str(merged_output_dir))
        except Exception:
            pass

        # AutoModelForCausalLM only loads the language model portion of the checkpoint.
        # Components like the visual encoder and MTP head are skipped at load time and
        # therefore absent from the merged output. Copy any such missing weights from the
        # base model so vLLM sees a complete checkpoint. If a future transformers version
        # loads these components automatically, the diff will be empty and we skip cleanly.
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file as _st_save
            import json as _json

            _base = config["model"]["name"]
            _midx_path = merged_output_dir / "model.safetensors.index.json"
            _msf_path = merged_output_dir / "model.safetensors"

            # Locate base model directory without network access
            _base_dir = None
            if Path(_base).is_dir():
                _base_dir = Path(_base)
            else:
                try:
                    from huggingface_hub import snapshot_download as _snap
                    _base_dir = Path(_snap(_base, local_files_only=True))
                except Exception:
                    pass

            if _base_dir is None:
                print("[Merge] Warning: could not locate base model directory; "
                      "skipping component copy. Run fix_merged_visual.py manually.")
            else:
                # Collect keys already in merged output
                _merged_keys = set()
                if _midx_path.exists():
                    with open(_midx_path) as _f:
                        _merged_keys = set(_json.load(_f)["weight_map"])
                elif _msf_path.exists():
                    with safe_open(str(_msf_path), framework="pt", device="cpu") as _sf:
                        _merged_keys = set(_sf.keys())

                # Collect keys in base model
                _base_keys = set()
                for _sf in sorted(_base_dir.glob("*.safetensors")):
                    with safe_open(str(_sf), framework="pt", device="cpu") as _f:
                        _base_keys.update(_f.keys())

                _missing = _base_keys - _merged_keys
                if not _missing:
                    print("[Merge] Merged output already contains all base model components")
                else:
                    # Load the missing tensors from base
                    _extra = {}
                    for _sf in sorted(_base_dir.glob("*.safetensors")):
                        with safe_open(str(_sf), framework="pt", device="cpu") as _f:
                            for k in _f.keys():
                                if k in _missing:
                                    _extra[k] = _f.get_tensor(k)
                        if len(_extra) == len(_missing):
                            break

                    _eshard = "model_extra_components.safetensors"
                    _st_save(_extra, str(merged_output_dir / _eshard))

                    # Create or update the weight index
                    if _midx_path.exists():
                        with open(_midx_path) as _f:
                            _midx = _json.load(_f)
                    else:
                        _midx = {"metadata": {}, "weight_map": {}}
                        if _msf_path.exists():
                            with safe_open(str(_msf_path), framework="pt", device="cpu") as _sf2:
                                for k in _sf2.keys():
                                    _midx["weight_map"][k] = "model.safetensors"
                    for k in _extra:
                        _midx["weight_map"][k] = _eshard
                    with open(_midx_path, "w") as _f:
                        _json.dump(_midx, _f)

                    # Summarise by depth-2 prefix
                    _groups = {}
                    for k in _extra:
                        _groups.setdefault(".".join(k.split(".")[:2]), 0)
                        _groups[".".join(k.split(".")[:2])] += 1
                    _summary = ", ".join(f"{p}.* ({n})" for p, n in sorted(_groups.items()))
                    print(f"[Merge] Copied {len(_extra)} tensors from base model: {_summary}")
        except Exception as _e:
            print(f"[Merge] Warning: could not copy base model components: {_e}")

        print(f"[Merge] Merged model saved to {merged_output_dir}")

        if flag_file:
            try:
                with open(flag_file, "w") as f:
                    f.write("MERGED")
                print(f"[Upload Signal] Merged model ready, wrote MERGED to {flag_file}")
            except Exception as e:
                print(f"[Upload Signal] Warning: Could not write MERGED to flag file: {e}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
