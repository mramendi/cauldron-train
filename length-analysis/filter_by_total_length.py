#!/usr/bin/env python3
"""
Filter JSONL by total tokenized conversation length.

Uses the same tokenization as training (via tokenizer_utils.tokenize_conversation)
to filter examples by their full tokenized length including all messages.

Usage:
    python filter_by_total_length.py \
        --input datasets/combined.jsonl \
        --output datasets/combined-short.jsonl \
        --max-length 4096 \
        --model-name ibm-granite/granite-4.0-h-1b \
        --filtered-output datasets/combined-long.jsonl \
        --show-distribution
"""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
from tokenizer_utils import tokenize_conversation


def get_total_length(messages, tokenizer):
    """Get the total tokenized length of the conversation."""
    input_ids, _ = tokenize_conversation(
        tokenizer=tokenizer,
        messages=messages,
        system_prompt=None,
        max_length=999999  # No truncation for measuring
    )
    return len(input_ids)


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL by total tokenized conversation length")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file (kept examples)")
    parser.add_argument("--filtered-output", type=str, default=None,
                       help="Optional output file for filtered-out examples")
    parser.add_argument("--max-length", type=int, required=True,
                       help="Maximum total tokenized length")
    parser.add_argument("--min-length", type=int, default=0,
                       help="Minimum total tokenized length (default: 0)")
    parser.add_argument("--model-name", type=str, default="ibm-granite/granite-4.0-h-1b",
                       help="Model name for tokenizer")
    parser.add_argument("--show-distribution", action="store_true",
                       help="Show length distribution before filtering")

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print()

    # First pass: collect lengths for distribution
    total_lengths = []

    print("Analyzing total tokenized conversation lengths...")
    with open(args.input, 'r') as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if 'messages' not in row:
                    continue
                total_len = get_total_length(row['messages'], tokenizer)
                total_lengths.append(total_len)
            except Exception as e:
                continue

    if args.show_distribution:
        import numpy as np
        print(f"\nTotal tokenized length distribution ({len(total_lengths)} examples):")
        print(f"  Min:    {min(total_lengths)}")
        print(f"  P10:    {np.percentile(total_lengths, 10):.0f}")
        print(f"  P25:    {np.percentile(total_lengths, 25):.0f}")
        print(f"  Median: {np.median(total_lengths):.0f}")
        print(f"  Mean:   {np.mean(total_lengths):.0f}")
        print(f"  P75:    {np.percentile(total_lengths, 75):.0f}")
        print(f"  P90:    {np.percentile(total_lengths, 90):.0f}")
        print(f"  P95:    {np.percentile(total_lengths, 95):.0f}")
        print(f"  P99:    {np.percentile(total_lengths, 99):.0f}")
        print(f"  Max:    {max(total_lengths)}")
        print()

    # Second pass: filter
    kept = 0
    filtered = 0

    print(f"Filtering examples with total length outside [{args.min_length}, {args.max_length}] tokens...")

    # Open filtered output file if specified
    filtered_file = open(args.filtered_output, 'w') if args.filtered_output else None

    try:
        with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
            for line in tqdm(infile):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                    if 'messages' not in row:
                        print(f"Warning: Row missing 'messages', skipping")
                        continue

                    total_len = get_total_length(row['messages'], tokenizer)

                    if args.min_length <= total_len <= args.max_length:
                        outfile.write(line)
                        kept += 1
                    else:
                        if filtered_file:
                            filtered_file.write(line)
                        filtered += 1
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
    finally:
        if filtered_file:
            filtered_file.close()

    print()
    print("="*80)
    print("FILTERING RESULTS")
    print("="*80)
    print(f"Input:                  {args.input}")
    print(f"Output (kept):          {args.output}")
    if args.filtered_output:
        print(f"Output (filtered):      {args.filtered_output}")
    print(f"Length range:           [{args.min_length}, {args.max_length}] tokens")
    print(f"Kept:                   {kept:6,} examples")
    print(f"Filtered:               {filtered:6,} examples")
    print(f"Total:                  {kept + filtered:6,} examples")
    if kept + filtered > 0:
        print(f"Kept percentage:        {kept/(kept+filtered)*100:5.1f}%")
    print("="*80)


if __name__ == "__main__":
    import sys
    sys.exit(main())
