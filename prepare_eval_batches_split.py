#!/usr/bin/env python3
"""
Prepare evaluation batches split by category.

Creates separate batched datasets for different evaluation categories,
each with its own batch size. One category must be designated as the
"best" category for early stopping during training.

Usage:
    python prepare_eval_batches_split.py \\
        --model-name ibm-granite/granite-4.0-h-1b \\
        --output-base ./prepared_data/eval_split \\
        --category ifeval:./datasets/ifeval_eval.jsonl:8 \\
        --category general:./datasets/general_eval.jsonl:6 \\
        --category coding:./datasets/coding_eval.jsonl:4 \\
        --best-category ifeval

    Each --category takes format: name:input_file:batch_size

    This creates:
    - ./prepared_data/eval_split/metadata.json (split=True, best_category=ifeval)
    - ./prepared_data/eval_split/ifeval/ (with metadata.json: best=True)
    - ./prepared_data/eval_split/general/ (with metadata.json: best=False)
    - ./prepared_data/eval_split/coding/ (with metadata.json: best=False)
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


def load_and_tokenize_category(
    path: str,
    category: str,
    tokenizer,
    max_length: int = 32768
) -> List[Dict]:
    """
    Load and tokenize a single category's evaluation dataset.

    Args:
        path: Path to JSONL file
        category: Category name
        tokenizer: Tokenizer with validated chat template
        max_length: Max sequence length

    Returns:
        List of tokenized examples
    """
    print(f"\nLoading {category} from {path}...")

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
                    "dataset": row.get("dataset", category),
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

    # Print stats
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
        print(f"  Note: {remainder} examples don't fit in full batches and will be skipped")
        print(f"    (You can adjust batch size to minimize waste)")

    return on_device_batches


def save_eval_batches(
    on_device_batches: List[Dict],
    output_dir: Path,
    category: str,
    is_best: bool = False
):
    """
    Save evaluation batches (one on-device batch per file).

    Args:
        on_device_batches: List of batches to save
        output_dir: Output directory for this category
        category: Category name
        is_best: Whether this is the best category for early stopping
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(on_device_batches):
        batch_path = output_dir / f"batch_{i:06d}.pkl"

        with open(batch_path, "wb") as f:
            pickle.dump(batch, f)

    # Calculate stats
    total_tokens = 0
    total_padding = 0

    for odb in on_device_batches:
        seq_lengths = odb["sequence_lengths"]
        bucket = odb["padded_length"]

        total_tokens += seq_lengths.sum()
        total_padding += (bucket * len(seq_lengths) - seq_lengths.sum())

    efficiency = total_tokens / (total_tokens + total_padding) if total_tokens > 0 else 0

    # Save metadata - preserve exact format from original
    metadata = {
        "category": category,
        "best": is_best,
        "num_batches": len(on_device_batches),
        "on_device_batch_size": len(on_device_batches[0]["input_ids"]) if on_device_batches else 0,
        "total_eval_examples": sum(len(odb["input_ids"]) for odb in on_device_batches),
        "total_tokens": int(total_tokens),
        "padding_efficiency": float(efficiency),
        "deterministic": True,
        "no_shuffling": True,
        "no_gradient_accumulation": True
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    best_marker = " (BEST - used for early stopping)" if is_best else ""
    print(f"\n{category.upper()}{best_marker} batches saved to {output_dir}")
    print(f"  Batches: {len(on_device_batches)}")
    print(f"  Examples: {metadata['total_eval_examples']}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Padding efficiency: {efficiency*100:.1f}%")


def parse_category_spec(spec: str) -> tuple:
    """
    Parse a category specification string.

    Format: "name:input_file:batch_size"
    Example: "ifeval:./datasets/ifeval_eval.jsonl:8"

    Returns:
        (name, input_file, batch_size)

    Raises:
        ValueError: If specification is invalid
    """
    parts = spec.split(':')
    if len(parts) != 3:
        raise ValueError(
            f"Invalid category spec '{spec}'. "
            f"Expected format: 'name:input_file:batch_size' "
            f"Example: 'ifeval:./datasets/ifeval_eval.jsonl:8'"
        )

    name = parts[0].strip()
    input_file = parts[1].strip()

    try:
        batch_size = int(parts[2].strip())
    except ValueError:
        raise ValueError(f"Invalid batch_size in category spec '{spec}'. Must be an integer.")

    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size} in '{spec}'")

    if not name:
        raise ValueError(f"Category name cannot be empty in '{spec}'")

    if not input_file:
        raise ValueError(f"Input file cannot be empty in '{spec}'")

    return name, input_file, batch_size


def main():
    parser = argparse.ArgumentParser(
        description="Prepare evaluation batches split by category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Three categories with different batch sizes
  python prepare_eval_batches_split.py \\
      --model-name ibm-granite/granite-4.0-h-1b \\
      --output-base ./prepared_data/eval_split \\
      --category ifeval:./datasets/ifeval_eval.jsonl:8 \\
      --category general:./datasets/general_eval.jsonl:6 \\
      --category coding:./datasets/coding_eval.jsonl:4 \\
      --best-category ifeval \\
      --max-length 32768

  # Two categories (minimal example)
  python prepare_eval_batches_split.py \\
      --model-name ibm-granite/granite-4.0-h-1b \\
      --output-base ./eval \\
      --category holdout:./holdout.jsonl:8 \\
      --category test:./test.jsonl:8 \\
      --best-category holdout

  # With custom chat template
  python prepare_eval_batches_split.py \\
      --model-name google/gemma-3-1b-it \\
      --chat-template ../gemma3_chat_template.jinja \\
      --output-base ./eval \\
      --category ifeval:./ifeval.jsonl:8 \\
      --best-category ifeval

Category specification format: name:input_file:batch_size
  - name: Short identifier for this category
  - input_file: Path to JSONL file
  - batch_size: On-device batch size (integer > 0)
        """
    )

    # Required arguments
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name for tokenizer (e.g., ibm-granite/granite-4.0-h-1b)")
    parser.add_argument("--category", action="append", dest="categories", required=True,
                       help="Category specification in format 'name:input_file:batch_size'. Can be repeated for multiple categories.")
    parser.add_argument("--output-base", type=str, required=True,
                       help="Base output directory (e.g., ./prepared_data/eval_split)")
    parser.add_argument("--best-category", type=str, required=True,
                       help="Category to use for early stopping (must match one of the category names)")

    # Optional arguments
    parser.add_argument("--chat-template", type=str, default=None,
                       help="Path to custom chat template file (.jinja). If not provided, uses the model's default template")
    parser.add_argument("--max-length", type=int, default=32768,
                       help="Maximum sequence length in tokens (default: 32768)")

    args = parser.parse_args()

    output_base = Path(args.output_base)

    print("="*80)
    print("CAULDRON SPLIT EVALUATION DATA PREPARATION")
    print("="*80)

    # Parse category specifications
    print("\n" + "="*80)
    print("PARSING CATEGORY SPECIFICATIONS")
    print("="*80)

    category_configs = {}
    for spec in args.categories:
        name, input_file, batch_size = parse_category_spec(spec)

        if name in category_configs:
            raise ValueError(f"Duplicate category name: '{name}'")

        # Verify input file exists
        if not Path(input_file).exists():
            print(f"\nERROR: Input file not found: {input_file}")
            print(f"  (from category specification: {spec})")
            sys.exit(1)

        category_configs[name] = {
            "file": input_file,
            "batch_size": batch_size
        }
        print(f"  {name}: file={input_file}, batch_size={batch_size}")

    # Validate best_category
    if args.best_category not in category_configs:
        available = ', '.join(f"'{c}'" for c in sorted(category_configs.keys()))
        print(f"\nERROR: --best-category '{args.best_category}' is not valid.")
        print(f"Must be one of the category names: {available}")
        sys.exit(1)

    best_category = args.best_category
    print(f"\nBest category for early stopping: {best_category}")

    # Load tokenizer
    print("\n" + "="*80)
    print("LOADING TOKENIZER")
    print("="*80)
    print(f"Model: {args.model_name}")
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

    # Load and tokenize each category
    print("\n" + "="*80)
    print("LOADING EVALUATION DATA")
    print("="*80)

    examples_by_category = {}
    for category, config in category_configs.items():
        examples_by_category[category] = load_and_tokenize_category(
            path=config["file"],
            category=category,
            tokenizer=tokenizer,
            max_length=args.max_length
        )

    # Process each category
    print("\n" + "="*80)
    print("CREATING BATCHES BY CATEGORY")
    print("="*80)

    for category, examples in examples_by_category.items():
        batch_size = category_configs[category]["batch_size"]
        print(f"\n--- Processing {category.upper()} ---")
        print(f"  Batch size: {batch_size}")

        on_device_batches = create_eval_batches(
            examples=examples,
            on_device_batch_size=batch_size,
            pad_token_id=tokenizer.pad_token_id
        )

        print(f"  Created {len(on_device_batches)} on-device batches")

        # Save to category-specific directory inside base
        output_dir = output_base / category
        is_best = (category == best_category)
        save_eval_batches(on_device_batches, output_dir, category, is_best=is_best)

    # Create parent metadata.json indicating this is split eval
    # Preserve exact format from original
    print("\n" + "="*80)
    print("CREATING PARENT METADATA")
    print("="*80)

    parent_metadata = {
        "split": True,
        "categories": sorted(examples_by_category.keys()),
        "best_category": best_category
    }

    parent_dir = output_base
    parent_dir.mkdir(parents=True, exist_ok=True)

    parent_metadata_path = parent_dir / "metadata.json"
    with open(parent_metadata_path, "w") as f:
        json.dump(parent_metadata, f, indent=2)

    print(f"Created parent metadata at {parent_metadata_path}")
    print(f"  split: {parent_metadata['split']}")
    print(f"  categories: {parent_metadata['categories']}")
    print(f"  best_category: {parent_metadata['best_category']}")

    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print("="*80)
    print(f"Created split eval structure in: {output_base}")
    print(f"\nDirectory structure:")
    print(f"  {output_base}/")
    print(f"    metadata.json (split=True, best_category={best_category})")
    for category in sorted(examples_by_category.keys()):
        output_dir = output_base / category
        best_marker = " <- BEST (early stopping)" if category == best_category else ""
        print(f"    {category}/{best_marker}")
        print(f"      metadata.json (best={category == best_category})")
        print(f"      batch_*.pkl files")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
