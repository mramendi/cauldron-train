#!/usr/bin/env python3
"""
Prepare pre-batched training data for chat-formatted supervised fine-tuning.

This script tokenizes conversational data, organizes it into effective batches,
and applies dynamic bucketing for efficient training. It supports:
- Multiple datasets with configurable upsampling ratios
- Chat templates with {% generation %} markers for proper token masking
- Multi-epoch generation with reshuffling
- Dynamic bucketing to minimize padding waste
- Verification of EOS token placement

Usage:
    python prepare_training_batches.py \\
        --config datasets_config.json \\
        --model-name ibm-granite/granite-4.0-h-1b \\
        --output-dir ./prepared_data \\
        --on-device-batch-size 3 \\
        --grad-accum-steps 10 \\
        --num-epochs 10 \\
        --max-length 32768
"""

import argparse
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import sys

# Import tokenizer utilities
from tokenizer_utils import (
    tokenize_conversation,
    validate_chat_template,
    check_default_system_prompt
)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_jsonl_dataset(
    path: str,
    name: str,
    tokenizer,
    max_length: int = 32768
) -> List[Dict]:
    """
    Load and tokenize a JSONL dataset.

    Args:
        path: Path to JSONL file
        name: Dataset name
        tokenizer: Tokenizer
        max_length: Max sequence length

    Returns:
        List of tokenized examples
    """
    print(f"\nLoading {name} from {path}...")

    examples = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)

                if 'messages' not in row:
                    print(f"  Warning: Line {line_num} missing 'messages', skipping")
                    continue

                messages = row['messages']

                # Tokenize (no system prompt - they're in the messages if needed)
                input_ids, labels = tokenize_conversation(
                    tokenizer=tokenizer,
                    messages=messages,
                    system_prompt=None,
                    max_length=max_length
                )

                examples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "original_length": len(input_ids),
                    "dataset": name,
                    "row_id": row.get("row_id", None)
                })

            except json.JSONDecodeError as e:
                print(f"  Warning: Malformed JSON at line {line_num}: {e}")
                continue

    # Print statistics
    if not examples:
        raise ValueError(f"No valid examples loaded from {path}")

    lengths = [ex["original_length"] for ex in examples]
    unmasked_counts = [sum(1 for label in ex["labels"] if label != -100) for ex in examples]

    print(f"  Loaded {len(examples)} examples")
    print(f"  Sequence length (tokens): min={min(lengths)}, mean={np.mean(lengths):.0f}, max={max(lengths)}")
    print(f"  Unmasked tokens: mean={np.mean(unmasked_counts):.0f} ({np.mean(unmasked_counts)/np.mean(lengths)*100:.1f}%)")

    return examples


def upsample_dataset(
    examples: List[Dict],
    factor: int,
    rng: random.Random
) -> List[Dict]:
    """
    Upsample dataset by integer factor with reshuffling between copies.

    Args:
        examples: Original examples
        factor: Integer upsampling factor
        rng: Random number generator

    Returns:
        Upsampled examples list
    """
    if factor == 1:
        return examples.copy()

    upsampled = []
    for i in range(factor):
        # Shuffle each copy differently
        copy = examples.copy()
        rng.shuffle(copy)
        upsampled.extend(copy)

    return upsampled


# ============================================================================
# EOS Verification
# ============================================================================

def verify_eos_before_padding(
    input_ids: List[int],
    labels: List[int],
    eos_token_id: int
) -> bool:
    """
    Verify that the sequence ends with an unmasked EOS token before any padding.

    Args:
        input_ids: Token IDs
        labels: Label IDs (-100 for masked)
        eos_token_id: EOS token ID

    Returns:
        True if valid, False otherwise
    """
    # Find the last unmasked token
    last_unmasked_idx = -1
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] != -100:
            last_unmasked_idx = i
            break

    if last_unmasked_idx == -1:
        # No unmasked tokens (shouldn't happen, but check)
        return False

    # Check if last unmasked token is EOS
    if input_ids[last_unmasked_idx] == eos_token_id:
        return True

    # Also check second-to-last unmasked token (handles EOS+newline case)
    if last_unmasked_idx >= 1:
        second_last_idx = -1
        for i in range(last_unmasked_idx - 1, -1, -1):
            if labels[i] != -100:
                second_last_idx = i
                break
        if second_last_idx >= 0 and input_ids[second_last_idx] == eos_token_id:
            return True

    return False


def ensure_eos_termination(
    input_ids: List[int],
    labels: List[int],
    eos_token_id: int
) -> Tuple[List[int], List[int]]:
    """
    Ensure sequence ends with unmasked EOS. Append if missing.

    Args:
        input_ids: Token IDs
        labels: Label IDs
        eos_token_id: EOS token ID

    Returns:
        Modified (input_ids, labels) tuple
    """
    if verify_eos_before_padding(input_ids, labels, eos_token_id):
        return input_ids, labels

    # Append EOS
    input_ids = input_ids + [eos_token_id]
    labels = labels + [eos_token_id]  # Unmask the EOS

    return input_ids, labels


# ============================================================================
# Balanced Effective Batch Creation
# ============================================================================

def create_proportional_effective_batch(
    dataset_pools: Dict[str, List[Dict]],
    dataset_iters: Dict[str, any],
    cumulative_target: Dict[str, float],
    cumulative_actual: Dict[str, int],
    proportions: Dict[str, float],
    effective_batch_size: int,
    rng: random.Random
) -> List[Dict]:
    """
    Create one effective batch by sampling proportionally from datasets.

    Uses cumulative tracking to avoid rounding errors accumulating over batches.
    Each dataset contributes proportionally to its size in the upsampled pool.

    Args:
        dataset_pools: Original dataset pools (for reshuffling)
        dataset_iters: Current iterators for each dataset
        cumulative_target: Cumulative target count for each dataset
        cumulative_actual: Cumulative actual count for each dataset
        proportions: Proportion of total for each dataset
        effective_batch_size: Target EBS
        rng: Random number generator

    Returns:
        List of examples for this effective batch
    """
    batch_examples = []

    # Update cumulative targets
    for name in dataset_pools.keys():
        cumulative_target[name] += proportions[name] * effective_batch_size

    # Calculate how many each dataset should contribute this batch
    # Use "largest remainder method" to fairly distribute rounding
    dataset_names = sorted(dataset_pools.keys())

    # Calculate exact needed (can be fractional)
    exact_needed = {}
    for name in dataset_names:
        exact_needed[name] = cumulative_target[name] - cumulative_actual[name]

    # Floor all of them
    floored_needed = {name: int(exact_needed[name]) for name in dataset_names}

    # Calculate remainder to distribute
    total_floored = sum(floored_needed.values())
    remainder = effective_batch_size - total_floored

    # Give remainder slots to datasets with largest fractional parts
    fractional_parts = {name: exact_needed[name] - floored_needed[name] for name in dataset_names}
    sorted_by_fraction = sorted(fractional_parts.items(), key=lambda x: x[1], reverse=True)

    actual_needed = floored_needed.copy()
    for i in range(remainder):
        name = sorted_by_fraction[i][0]
        actual_needed[name] += 1

    # Sample according to actual_needed
    for name in dataset_names:
        needed = actual_needed[name]

        for _ in range(needed):
            try:
                example = next(dataset_iters[name])
            except StopIteration:
                # Reshuffle and restart this dataset
                shuffled = dataset_pools[name].copy()
                rng.shuffle(shuffled)
                dataset_iters[name] = iter(shuffled)
                example = next(dataset_iters[name])

            batch_examples.append(example)
            cumulative_actual[name] += 1

    # Verify we got exactly the right number
    if len(batch_examples) != effective_batch_size:
        raise ValueError(f"Created batch with {len(batch_examples)} samples, expected {effective_batch_size}")

    # Shuffle within batch to avoid dataset clustering
    rng.shuffle(batch_examples)

    return batch_examples


def create_on_device_batch(
    examples: List[Dict],
    bucket_size: int,
    pad_token_id: int
) -> Dict:
    """
    Create a single on-device batch from examples.

    Args:
        examples: Examples to batch
        bucket_size: Padding length
        pad_token_id: Pad token ID

    Returns:
        Batch dict with numpy arrays
    """
    input_ids_batch = []
    labels_batch = []
    sequence_lengths = []
    datasets = []
    row_ids = []

    for ex in examples:
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        seq_len = len(input_ids)

        # Pad
        padding_length = bucket_size - seq_len
        input_ids_padded = input_ids + [pad_token_id] * padding_length
        labels_padded = labels + [-100] * padding_length

        input_ids_batch.append(input_ids_padded)
        labels_batch.append(labels_padded)
        sequence_lengths.append(seq_len)
        datasets.append(ex["dataset"])
        row_ids.append(ex.get("row_id", None))

    return {
        "input_ids": np.array(input_ids_batch, dtype=np.int32),
        "labels": np.array(labels_batch, dtype=np.int32),
        "sequence_lengths": np.array(sequence_lengths, dtype=np.int16),
        "datasets": datasets,
        "row_ids": row_ids,
        "padded_length": bucket_size,
    }


def create_effective_batch_with_dynamic_bucketing(
    batch_examples: List[Dict],
    on_device_batch_size: int,
    pad_token_id: int
) -> List[Dict]:
    """
    Split effective batch into on-device batches with dynamic bucketing.

    Args:
        batch_examples: Examples for this effective batch
        on_device_batch_size: On-device batch size
        pad_token_id: Pad token ID

    Returns:
        List of on-device batches
    """
    # Sort by length for efficient bucketing
    sorted_examples = sorted(batch_examples, key=lambda x: x["original_length"])

    on_device_batches = []
    num_on_device = len(sorted_examples) // on_device_batch_size

    for i in range(num_on_device):
        start_idx = i * on_device_batch_size
        end_idx = start_idx + on_device_batch_size

        batch_exs = sorted_examples[start_idx:end_idx]

        # Calculate bucket (round up to multiple of 8)
        max_len = max(ex["original_length"] for ex in batch_exs)
        bucket_size = ((max_len + 7) // 8) * 8

        batch = create_on_device_batch(batch_exs, bucket_size, pad_token_id)
        batch["bucket"] = bucket_size
        on_device_batches.append(batch)

    return on_device_batches


# ============================================================================
# Multi-Epoch Generation
# ============================================================================

def generate_epoch_batches(
    datasets: Dict[str, List[Dict]],
    epoch_num: int,
    effective_batch_size: int,
    on_device_batch_size: int,
    pad_token_id: int,
    eos_token_id: int,
    seed: int
) -> List[Dict]:
    """
    Generate all effective batches for one epoch.

    Args:
        datasets: Dict of upsampled datasets
        epoch_num: Epoch number (for seeding)
        effective_batch_size: EBS
        on_device_batch_size: On-device batch size
        pad_token_id: Pad token ID
        eos_token_id: EOS token ID
        seed: Base seed

    Returns:
        List of effective batches
    """
    # Epoch-specific seed
    epoch_seed = seed + epoch_num
    rng = random.Random(epoch_seed)

    # Create dataset pools and iterators
    dataset_pools = {}
    dataset_iters = {}

    for name, examples in datasets.items():
        # Shuffle for this epoch
        shuffled = examples.copy()
        rng.shuffle(shuffled)
        dataset_pools[name] = shuffled
        dataset_iters[name] = iter(shuffled)

    # Calculate total number of effective batches
    total_examples = sum(len(exs) for exs in datasets.values())
    num_effective_batches = total_examples // effective_batch_size

    # Calculate proportions for proportional sampling
    proportions = {name: len(exs) / total_examples for name, exs in datasets.items()}

    # Initialize cumulative tracking (to avoid rounding error accumulation)
    cumulative_target = {name: 0.0 for name in datasets.keys()}
    cumulative_actual = {name: 0 for name in datasets.keys()}

    print(f"\nEpoch {epoch_num}:")
    print(f"  Total examples: {total_examples:,}")
    print(f"  Effective batches: {num_effective_batches}")
    print(f"  Proportions:")
    for name in sorted(datasets.keys()):
        print(f"    {name}: {proportions[name]*100:.1f}% ({len(datasets[name])} examples)")

    effective_batches = []

    for eb_idx in tqdm(range(num_effective_batches), desc=f"Epoch {epoch_num}"):
        # Create proportional effective batch
        batch_examples = create_proportional_effective_batch(
            dataset_pools=dataset_pools,
            dataset_iters=dataset_iters,
            cumulative_target=cumulative_target,
            cumulative_actual=cumulative_actual,
            proportions=proportions,
            effective_batch_size=effective_batch_size,
            rng=rng
        )

        # Verify EOS termination
        for ex in batch_examples:
            ex["input_ids"], ex["labels"] = ensure_eos_termination(
                ex["input_ids"],
                ex["labels"],
                eos_token_id
            )
            # Update length if EOS was added
            ex["original_length"] = len(ex["input_ids"])

        # Create on-device batches
        on_device_batches = create_effective_batch_with_dynamic_bucketing(
            batch_examples=batch_examples,
            on_device_batch_size=on_device_batch_size,
            pad_token_id=pad_token_id
        )

        effective_batch = {
            "on_device_batches": on_device_batches,
            "effective_batch_idx": eb_idx,
            "num_on_device_batches": len(on_device_batches),
            "epoch": epoch_num
        }

        effective_batches.append(effective_batch)

    # Verify proportions achieved
    print(f"\n  Actual distribution achieved:")
    total_sampled = sum(cumulative_actual.values())
    for name in sorted(datasets.keys()):
        actual_count = cumulative_actual[name]
        actual_pct = (actual_count / total_sampled * 100) if total_sampled > 0 else 0
        target_pct = proportions[name] * 100
        print(f"    {name}: {actual_count:,}/{total_sampled:,} ({actual_pct:.1f}%, target: {target_pct:.1f}%)")

    return effective_batches


# ============================================================================
# Save/Load
# ============================================================================

def save_epoch_batches(
    effective_batches: List[Dict],
    output_dir: Path,
    epoch_num: int
):
    """
    Save one epoch's effective batches.

    Args:
        effective_batches: List of effective batches
        output_dir: Base output directory
        epoch_num: Epoch number
    """
    epoch_dir = output_dir / f"epoch_{epoch_num:02d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    for eb in effective_batches:
        batch_idx = eb["effective_batch_idx"]
        batch_path = epoch_dir / f"effective_batch_{batch_idx:06d}.pkl"

        with open(batch_path, "wb") as f:
            pickle.dump(eb, f)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare batched training data for chat-formatted supervised fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python prepare_training_batches.py \\
      --config datasets_config.json \\
      --model-name ibm-granite/granite-4.0-h-1b \\
      --output-dir ./prepared_data \\
      --on-device-batch-size 3 \\
      --grad-accum-steps 10 \\
      --num-epochs 10 \\
      --max-length 32768

The datasets config JSON should have the following format:
  {
    "datasets": [
      {"name": "dataset1", "path": "data1.jsonl", "upsample": 1},
      {"name": "dataset2", "path": "data2.jsonl", "upsample": 2}
    ]
  }
"""
    )

    # Config
    parser.add_argument("--config", type=str, required=True,
                       help="JSON config with datasets and integer upsample factors")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for prepared batches")

    # Tokenizer and Model
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name for tokenizer (e.g., ibm-granite/granite-4.0-h-1b)")
    parser.add_argument("--chat-template", type=str, default=None,
                       help="Path to custom chat template file (.jinja). If not provided, uses the model's default template")
    parser.add_argument("--max-length", type=int, default=32768,
                       help="Maximum sequence length in tokens (default: 32768)")

    # Batching
    parser.add_argument("--on-device-batch-size", type=int, default=3,
                       help="On-device batch size (default: 3)")
    parser.add_argument("--grad-accum-steps", type=int, default=10,
                       help="Gradient accumulation steps (default: 10)")

    # Multi-epoch
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of epochs to generate (default: 10)")

    # Seed
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CAULDRON TRAINING DATA PREPARATION")
    print("="*80)

    # Load config
    print(f"\nLoading dataset configuration from {args.config}...")
    with open(args.config) as f:
        config = json.load(f)

    dataset_configs = config["datasets"]

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load chat template if provided
    if args.chat_template is not None:
        template_path = Path(args.chat_template)
        if not template_path.exists():
            print(f"\nERROR: Chat template file not found: {template_path}")
            sys.exit(1)

        print(f"  Loading custom chat template from {template_path}")
        with open(template_path, 'r') as f:
            custom_template = f.read()

        # Validate the custom template
        is_valid, _, error_msg = validate_chat_template(tokenizer, custom_template)
        if not is_valid:
            print(f"\nERROR: Invalid chat template:")
            print(error_msg)
            sys.exit(1)

        # Check for default system prompt
        warning = check_default_system_prompt(custom_template)
        if warning:
            print(f"\n{warning}")

        # Apply the template
        tokenizer.chat_template = custom_template
    else:
        # Use model's default template
        print(f"  Using chat template from model (no custom template provided)")

        # Validate the model's template
        is_valid, template, error_msg = validate_chat_template(tokenizer)
        if not is_valid:
            print(f"\nERROR: Model's chat template is not compatible with this script:")
            print(error_msg)
            print("\nPlease provide a custom chat template using --chat-template that includes")
            print("{% generation %} markers around assistant responses.")
            sys.exit(1)

        # Check for default system prompt
        warning = check_default_system_prompt(template)
        if warning:
            print(f"\n{warning}")

    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")
    print(f"  Maximum sequence length: {args.max_length} tokens")

    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    base_datasets = {}
    for ds_config in dataset_configs:
        name = ds_config["name"]
        path = ds_config["path"]

        examples = load_jsonl_dataset(
            path=path,
            name=name,
            tokenizer=tokenizer,
            max_length=args.max_length
        )

        base_datasets[name] = {
            "examples": examples,
            "upsample": ds_config.get("upsample", 1)
        }

    # Upsample datasets
    print("\n" + "="*80)
    print("UPSAMPLING DATASETS")
    print("="*80)

    rng = random.Random(args.seed)
    upsampled_datasets = {}

    total_examples = 0
    total_tokens = 0

    for name, ds_info in base_datasets.items():
        examples = ds_info["examples"]
        factor = ds_info["upsample"]

        upsampled = upsample_dataset(examples, factor, rng)
        upsampled_datasets[name] = upsampled

        tokens = sum(ex["original_length"] for ex in upsampled)
        total_examples += len(upsampled)
        total_tokens += tokens

        print(f"\n{name}:")
        print(f"  Original: {len(examples):,} examples")
        print(f"  Upsample: {factor}x")
        print(f"  Final: {len(upsampled):,} examples")
        print(f"  Tokens: {tokens:,}")
        print(f"  Proportion: {len(upsampled)/total_examples*100:.1f}% examples, {tokens/total_tokens*100:.1f}% tokens")

    # Batch config
    effective_batch_size = args.on_device_batch_size * args.grad_accum_steps

    print("\n" + "="*80)
    print("BATCH CONFIGURATION")
    print("="*80)
    print(f"On-device batch size: {args.on_device_batch_size}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Maximum sequence length: {args.max_length} tokens")

    # Generate epochs
    print("\n" + "="*80)
    print("GENERATING EPOCHS")
    print("="*80)

    all_epoch_stats = []

    for epoch in range(args.num_epochs):
        effective_batches = generate_epoch_batches(
            datasets=upsampled_datasets,
            epoch_num=epoch,
            effective_batch_size=effective_batch_size,
            on_device_batch_size=args.on_device_batch_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            seed=args.seed
        )

        # Save
        save_epoch_batches(effective_batches, output_dir, epoch)

        # Stats
        num_on_device = sum(eb["num_on_device_batches"] for eb in effective_batches)

        epoch_stats = {
            "epoch": epoch,
            "effective_batches": len(effective_batches),
            "on_device_batches": num_on_device
        }
        all_epoch_stats.append(epoch_stats)

        print(f"  Saved {len(effective_batches)} effective batches → {num_on_device} on-device batches")

    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "max_sequence_length_tokens": args.max_length,
        "num_epochs": args.num_epochs,
        "effective_batch_size": effective_batch_size,
        "on_device_batch_size": args.on_device_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "custom_chat_template": args.chat_template,
        "datasets": {
            name: {
                "original_examples": len(base_datasets[name]["examples"]),
                "upsample_factor": base_datasets[name]["upsample"],
                "final_examples": len(upsampled_datasets[name])
            }
            for name in base_datasets.keys()
        },
        "epoch_stats": all_epoch_stats,
        "total_examples": total_examples,
        "total_tokens": total_tokens,
        "seed": args.seed
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Maximum sequence length: {args.max_length} tokens")
    print(f"Epochs: {args.num_epochs}")
    print(f"Total effective batches: {sum(s['effective_batches'] for s in all_epoch_stats)}")
    print(f"Total on-device batches: {sum(s['on_device_batches'] for s in all_epoch_stats)}")
    print(f"Metadata saved to: {metadata_path}")
    print("="*80)


if __name__ == "__main__":
    main()
