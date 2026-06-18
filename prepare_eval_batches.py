#!/usr/bin/env python3
"""
Prepare evaluation batches for model evaluation during training.

This script is simpler than training batch preparation:
- No upsampling
- No multi-epoch generation (single pass)
- Deterministic ordering (no shuffling)
- No gradient accumulation (just on-device batches)

Usage:
    python prepare_eval_batches.py \\
        --eval-file ./data/eval.jsonl \\
        --model-name ibm-granite/granite-4.0-h-1b \\
        --output-dir ./prepared_data/eval \\
        --on-device-batch-size 3 \\
        --max-length 32768
"""

import argparse
import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

# Import tokenizer utilities
from tokenizer_utils import (
    tokenize_conversation,
    validate_chat_template,
    check_default_system_prompt
)


def load_and_tokenize_eval(
    path: str,
    tokenizer,
    max_length: int = 32768
) -> List[Dict]:
    """
    Load and tokenize evaluation dataset.

    Args:
        path: Path to JSONL file
        tokenizer: Tokenizer with validated chat template
        max_length: Max sequence length

    Returns:
        List of tokenized examples
    """
    print(f"Loading evaluation data from {path}...")

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

                # Tokenize
                input_ids, labels = tokenize_conversation(
                    tokenizer=tokenizer,
                    messages=messages,
                    system_prompt=None,
                    max_length=max_length
                )

                # Ensure EOS termination
                # Check if labels[-1] is masked (shouldn't happen for assistant messages)
                if labels[-1] == -100:
                    print(f"  Warning: Line {line_num} has masked final token (assistant message should be unmasked)")

                # Check if EOS is in last two positions (handles EOS+newline case)
                has_eos = (
                    input_ids[-1] == tokenizer.eos_token_id or
                    (len(input_ids) >= 2 and input_ids[-2] == tokenizer.eos_token_id)
                )

                if not has_eos:
                    # Append EOS
                    input_ids.append(tokenizer.eos_token_id)
                    labels.append(tokenizer.eos_token_id)

                examples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "original_length": len(input_ids),
                    "dataset": row.get("dataset", "eval"),
                    "row_id": row.get("row_id", None)
                })

            except json.JSONDecodeError as e:
                print(f"  Warning: Malformed JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing line {line_num}: {e}")
                continue

    if not examples:
        raise ValueError(f"No valid examples loaded from {path}")

    lengths = [ex["original_length"] for ex in examples]
    unmasked_counts = [sum(1 for label in ex["labels"] if label != -100) for ex in examples]

    print(f"  Loaded: {len(examples)} examples")
    print(f"  Sequence length (tokens): min={min(lengths)}, mean={np.mean(lengths):.0f}, max={max(lengths)}")
    print(f"  Unmasked tokens: mean={np.mean(unmasked_counts):.0f} ({np.mean(unmasked_counts)/np.mean(lengths)*100:.1f}%)")

    return examples


def create_on_device_batch(
    examples: List[Dict],
    bucket_size: int,
    pad_token_id: int
) -> Dict:
    """
    Create a single on-device batch.

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


def create_eval_batches(
    examples: List[Dict],
    on_device_batch_size: int,
    pad_token_id: int
) -> List[Dict]:
    """
    Create evaluation batches with deterministic ordering (no shuffling).

    Args:
        examples: All eval examples
        on_device_batch_size: Batch size
        pad_token_id: Pad token ID

    Returns:
        List of on-device batches
    """
    # Sort by length for efficient bucketing (deterministic)
    sorted_examples = sorted(examples, key=lambda x: x["original_length"])

    on_device_batches = []
    num_batches = len(sorted_examples) // on_device_batch_size

    for i in range(num_batches):
        start_idx = i * on_device_batch_size
        end_idx = start_idx + on_device_batch_size

        batch_exs = sorted_examples[start_idx:end_idx]

        # Calculate bucket (round up to multiple of 8)
        max_len = max(ex["original_length"] for ex in batch_exs)
        bucket_size = ((max_len + 7) // 8) * 8

        batch = create_on_device_batch(batch_exs, bucket_size, pad_token_id)
        batch["bucket"] = bucket_size
        on_device_batches.append(batch)

    # Handle remainder
    remainder = len(sorted_examples) % on_device_batch_size
    if remainder > 0:
        print(f"\nNote: {remainder} examples don't fit in full batches and will be skipped")
        print(f"  (You can adjust --on-device-batch-size to minimize waste)")

    return on_device_batches


def save_eval_batches(
    on_device_batches: List[Dict],
    output_dir: Path
):
    """
    Save evaluation batches (one on-device batch per file).

    Args:
        on_device_batches: List of batches to save
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(on_device_batches):
        batch_path = output_dir / f"batch_{i:06d}.pkl"

        with open(batch_path, "wb") as f:
            pickle.dump(batch, f)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare evaluation batches for model evaluation during training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python prepare_eval_batches.py \\
      --eval-file ./data/eval.jsonl \\
      --model-name ibm-granite/granite-4.0-h-1b \\
      --output-dir ./prepared_data/eval \\
      --on-device-batch-size 3 \\
      --max-length 32768

Note: Evaluation uses deterministic ordering and no shuffling.
"""
    )

    # Required arguments
    parser.add_argument("--eval-file", type=str, required=True,
                       help="Path to evaluation JSONL file")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name for tokenizer (e.g., ibm-granite/granite-4.0-h-1b)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for prepared batches")

    # Optional arguments
    parser.add_argument("--chat-template", type=str, default=None,
                       help="Path to custom chat template file (.jinja). If not provided, uses the model's default template")
    parser.add_argument("--max-length", type=int, default=32768,
                       help="Maximum sequence length in tokens (default: 32768)")
    parser.add_argument("--on-device-batch-size", type=int, default=3,
                       help="On-device batch size for memory efficiency (default: 3)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("="*80)
    print("CAULDRON EVALUATION DATA PREPARATION")
    print("="*80)

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

    # Load and tokenize
    print("\n" + "="*80)
    print("LOADING EVALUATION DATA")
    print("="*80)

    examples = load_and_tokenize_eval(
        path=args.eval_file,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    # Create on-device batches
    print("\n" + "="*80)
    print("CREATING BATCHES")
    print("="*80)
    print(f"On-device batch size: {args.on_device_batch_size}")
    print(f"Maximum sequence length: {args.max_length} tokens")
    print(f"(No gradient accumulation for eval - just processing batches)")

    on_device_batches = create_eval_batches(
        examples=examples,
        on_device_batch_size=args.on_device_batch_size,
        pad_token_id=tokenizer.pad_token_id
    )

    print(f"\nCreated {len(on_device_batches)} on-device batches")

    # Calculate stats
    total_tokens = 0
    total_padding = 0

    for odb in on_device_batches:
        seq_lengths = odb["sequence_lengths"]
        bucket = odb["padded_length"]

        total_tokens += seq_lengths.sum()
        total_padding += (bucket * len(seq_lengths) - seq_lengths.sum())

    efficiency = total_tokens / (total_tokens + total_padding) if total_tokens > 0 else 0

    print(f"\nPadding efficiency: {efficiency*100:.1f}%")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Wasted padding: {total_padding:,}")

    # Save
    print("\n" + "="*80)
    print("SAVING BATCHES")
    print("="*80)

    save_eval_batches(on_device_batches, output_dir)
    print(f"Saved {len(on_device_batches)} batches to {output_dir}")

    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "max_sequence_length_tokens": args.max_length,
        "custom_chat_template": args.chat_template,
        "split": False,
        "best": True,  # Single eval is always the best
        "num_batches": len(on_device_batches),
        "on_device_batch_size": args.on_device_batch_size,
        "total_eval_examples": len(examples),
        "total_tokens": int(total_tokens),
        "total_padding": int(total_padding),
        "padding_efficiency": float(efficiency),
        "deterministic": True,
        "no_shuffling": True,
        "no_gradient_accumulation": True
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
    print(f"Eval batches: {len(on_device_batches)}")
    print(f"Eval examples: {len(examples)}")
    print(f"Metadata saved to: {metadata_path}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
