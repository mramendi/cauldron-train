#!/usr/bin/env python3
"""
Filter JSONL by minimum assistant message length across all turns.

For multiturn conversations, checks ALL assistant messages and uses the shortest one
(unless --first is specified, which only checks the first assistant message).

Usage:
    python filter_by_min_assistant_length.py \
        --input datasets/split_data/ifevlike-train1000.jsonl \
        --output datasets/split_data/ifevlike-train1000-long-enough.jsonl \
        --min-asst-len 50 \
        --model-name ibm-granite/granite-4.0-h-1b \
        --filtered-output datasets/split_data/ifevlike-train1000-too-short.jsonl

    # Only check first assistant response:
    python filter_by_min_assistant_length.py \
        --input input.jsonl \
        --output output.jsonl \
        --min-asst-len 50 \
        --first
"""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm


def get_min_assistant_length(messages, tokenizer, first_only=False):
    """Get the minimum length of assistant message(s) in tokens.

    Args:
        messages: List of conversation messages
        tokenizer: Tokenizer to use for encoding
        first_only: If True, only check the first assistant message
    """
    if first_only:
        # Only check the first assistant message
        for msg in messages:
            if msg['role'] == 'assistant':
                return len(tokenizer.encode(msg['content']))
        return 0
    else:
        # Check all assistant messages and return minimum
        min_len = float('inf')
        for msg in messages:
            if msg['role'] == 'assistant':
                msg_len = len(tokenizer.encode(msg['content']))
                min_len = min(min_len, msg_len)
        return min_len if min_len != float('inf') else 0


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL by minimum assistant message length")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file (kept examples)")
    parser.add_argument("--filtered-output", type=str, default=None,
                       help="Optional output file for filtered-out examples")
    parser.add_argument("--min-asst-len", type=int, required=True,
                       help="Minimum assistant message length in tokens")
    parser.add_argument("--model-name", type=str, default="ibm-granite/granite-4.0-h-1b",
                       help="Model name for tokenizer")
    parser.add_argument("--show-distribution", action="store_true",
                       help="Show length distribution before filtering")
    parser.add_argument("--first", action="store_true",
                       help="Only check the first assistant message (ignore subsequent ones)")

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print()

    # First pass: collect lengths for distribution
    asst_lengths = []

    mode_desc = "first assistant message" if args.first else "minimum assistant message lengths (across all turns)"
    print(f"Analyzing {mode_desc}...")
    with open(args.input, 'r') as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                asst_len = get_min_assistant_length(row['messages'], tokenizer, first_only=args.first)
                if asst_len > 0:
                    asst_lengths.append(asst_len)
            except Exception as e:
                continue

    if args.show_distribution:
        import numpy as np
        if args.first:
            print(f"\nFirst assistant message length distribution ({len(asst_lengths)} examples):")
        else:
            print(f"\nMinimum assistant message length distribution ({len(asst_lengths)} examples):")
            print(f"  (For multiturn, this is the SHORTEST assistant message in the conversation)")
        print(f"  Min:    {min(asst_lengths)}")
        print(f"  P10:    {np.percentile(asst_lengths, 10):.0f}")
        print(f"  P25:    {np.percentile(asst_lengths, 25):.0f}")
        print(f"  Median: {np.median(asst_lengths):.0f}")
        print(f"  P75:    {np.percentile(asst_lengths, 75):.0f}")
        print(f"  P90:    {np.percentile(asst_lengths, 90):.0f}")
        print(f"  P95:    {np.percentile(asst_lengths, 95):.0f}")
        print(f"  Max:    {max(asst_lengths)}")
        print()

    # Second pass: filter
    kept = 0
    filtered = 0

    if args.first:
        print(f"Filtering examples with first assistant message < {args.min_asst_len} tokens...")
    else:
        print(f"Filtering examples with any assistant message < {args.min_asst_len} tokens...")

    # Open filtered output file if specified
    filtered_file = open(args.filtered_output, 'w') if args.filtered_output else None

    try:
        with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
            for line in tqdm(infile):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                    asst_len = get_min_assistant_length(row['messages'], tokenizer, first_only=args.first)

                    if asst_len >= args.min_asst_len:
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
    print(f"Input:                 {args.input}")
    print(f"Output (kept):         {args.output}")
    if args.filtered_output:
        print(f"Output (filtered):     {args.filtered_output}")
    print(f"Mode:                  {'First assistant only' if args.first else 'All assistants (minimum)'}")
    print(f"Min assistant length:  {args.min_asst_len} tokens")
    print(f"Kept:                  {kept:6,} examples")
    print(f"Filtered:              {filtered:6,} examples")
    print(f"Total:                 {kept + filtered:6,} examples")
    print(f"Kept percentage:       {kept/(kept+filtered)*100:5.1f}%")
    print("="*80)


if __name__ == "__main__":
    import sys
    sys.exit(main())
