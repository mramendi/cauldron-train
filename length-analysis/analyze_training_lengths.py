#!/usr/bin/env python3
"""
Analyze actual prompt and response lengths in training data.

For multiturn conversations, tracks ALL assistant message lengths
across all conversations.

Usage:
    python analyze_training_lengths.py \
        --files datasets/split_data/kimify-500_train.jsonl \
                datasets/split_data/ifeval-like-500_train.jsonl \
                datasets/split_data/h4-inst-500_train.jsonl
"""

import argparse
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Import tokenization utilities for accurate conversation length
try:
    from tokenizer_utils import tokenize_conversation
    HAS_TOKENIZER_UTILS = True
except ImportError:
    HAS_TOKENIZER_UTILS = False


def analyze_file(file_path: str, tokenizer, max_examples: int = None):
    """Analyze prompt/response length distribution in a file."""

    prompt_lens = []
    response_lens = []
    total_lens = []

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if max_examples is not None and line_num > max_examples:
                break

            if not line.strip():
                continue

            try:
                row = json.loads(line)
                msgs = row['messages']

                # Use proper tokenization if available, otherwise approximate
                if HAS_TOKENIZER_UTILS:
                    # Use the actual tokenization function (includes chat template, special tokens, etc.)
                    input_ids, labels = tokenize_conversation(
                        tokenizer=tokenizer,
                        messages=msgs,
                        system_prompt=None,
                        max_length=65536  # Large to avoid truncation
                    )
                    total_tokens = len(input_ids)
                else:
                    # Fallback: approximate by concatenating content
                    full_conversation = " ".join(msg['content'] for msg in msgs)
                    total_tokens = len(tokenizer.encode(full_conversation))

                # Track first user message and all assistant messages
                first_user_len = 0
                asst_lens = []
                found_first_user = False

                for msg in msgs:
                    content = msg['content']
                    if msg['role'] == 'user' and not found_first_user:
                        first_user_len = len(tokenizer.encode(content))
                        found_first_user = True
                    elif msg['role'] == 'assistant':
                        asst_len = len(tokenizer.encode(content))
                        asst_lens.append(asst_len)

                if found_first_user and asst_lens:
                    prompt_lens.append(first_user_len)
                    # Append ALL assistant message lengths, not just max
                    response_lens.extend(asst_lens)
                    total_lens.append(total_tokens)

            except Exception as e:
                continue

    return {
        'file': file_path,
        'count': len(prompt_lens),
        'prompt_lens': prompt_lens,
        'response_lens': response_lens,
        'total_lens': total_lens
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze training data length distribution")
    parser.add_argument("--files", nargs="+", required=True,
                       help="JSONL files to analyze")
    parser.add_argument("--model-name", type=str, default="ibm-granite/granite-4.0-h-1b",
                       help="Model name for tokenizer")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Max examples to analyze per file (default: all examples)")

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print()

    # Analyze each file
    results = []
    for file_path in tqdm(args.files, desc="Analyzing files"):
        if not Path(file_path).exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        result = analyze_file(file_path, tokenizer, args.max_examples)
        results.append(result)

    # Print results
    print("\n" + "="*80)
    print("TRAINING DATA LENGTH ANALYSIS")
    print("="*80)
    if HAS_TOKENIZER_UTILS:
        print("(Using tokenize_conversation for accurate full-conversation lengths)")
    else:
        print("(Warning: tokenizer_utils not found, using approximate lengths)")
    print()

    for result in results:
        print(f"\n{Path(result['file']).name}:")
        print(f"  Examples analyzed: {result['count']}")

        if result['count'] > 0:
            print(f"  First user msg:    min={min(result['prompt_lens']):4d}, "
                  f"mean={np.mean(result['prompt_lens']):6.0f}, "
                  f"max={max(result['prompt_lens']):4d}")
            print(f"  All asst msgs:     min={min(result['response_lens']):4d}, "
                  f"mean={np.mean(result['response_lens']):6.0f}, "
                  f"max={max(result['response_lens']):4d}")
            print(f"  FULL conversation: min={min(result['total_lens']):4d}, "
                  f"mean={np.mean(result['total_lens']):6.0f}, "
                  f"max={max(result['total_lens']):4d}")

            # Generation position analysis
            print(f"\n  Generation starts at positions: {min(result['prompt_lens'])}-{max(result['prompt_lens'])}")
            print(f"  Generation ends at positions:   {min(result['total_lens'])}-{max(result['total_lens'])}")

    # Combined stats
    all_prompt_lens = []
    all_response_lens = []
    all_total_lens = []

    for result in results:
        all_prompt_lens.extend(result['prompt_lens'])
        all_response_lens.extend(result['response_lens'])
        all_total_lens.extend(result['total_lens'])

    if all_prompt_lens:
        print(f"\n{'='*80}")
        print("COMBINED STATISTICS")
        print("="*80)
        print(f"Total examples: {len(all_prompt_lens)}")
        print(f"\nFirst user message lengths:")
        print(f"  Min:    {min(all_prompt_lens)}")
        print(f"  P10:    {np.percentile(all_prompt_lens, 10):.0f}")
        print(f"  P20:    {np.percentile(all_prompt_lens, 20):.0f}")
        print(f"  Mean:   {np.mean(all_prompt_lens):.0f}")
        print(f"  Median: {np.median(all_prompt_lens):.0f}")
        print(f"  P90:    {np.percentile(all_prompt_lens, 90):.0f}")
        print(f"  Max:    {max(all_prompt_lens)}")

        print(f"\nAll assistant message lengths (every assistant turn):")
        print(f"  Min:    {min(all_response_lens)}")
        print(f"  P10:    {np.percentile(all_response_lens, 10):.0f}")
        print(f"  P20:    {np.percentile(all_response_lens, 20):.0f}")
        print(f"  Mean:   {np.mean(all_response_lens):.0f}")
        print(f"  Median: {np.median(all_response_lens):.0f}")
        print(f"  P90:    {np.percentile(all_response_lens, 90):.0f}")
        print(f"  Max:    {max(all_response_lens)}")

        print(f"\nFULL conversation lengths (all turns):")
        print(f"  Min:    {min(all_total_lens)}")
        print(f"  P10:    {np.percentile(all_total_lens, 10):.0f}")
        print(f"  P20:    {np.percentile(all_total_lens, 20):.0f}")
        print(f"  Mean:   {np.mean(all_total_lens):.0f}")
        print(f"  Median: {np.median(all_total_lens):.0f}")
        print(f"  P90:    {np.percentile(all_total_lens, 90):.0f}")
        print(f"  Max:    {max(all_total_lens)}")

        print(f"\n{'='*80}")
        print("KEY INSIGHT")
        print("="*80)
        print(f"During training, model practices generating:")
        print(f"  Starting at positions:  {min(all_prompt_lens)} to {max(all_prompt_lens)}")
        print(f"  Response lengths:       {min(all_response_lens)} to {max(all_response_lens)} tokens")
        print(f"    P10-P20 range:        {np.percentile(all_response_lens, 10):.0f} to {np.percentile(all_response_lens, 20):.0f} tokens")
        print(f"    P50 (median):         {np.median(all_response_lens):.0f} tokens")
        print(f"  Ending at positions:    {min(all_total_lens)} to {max(all_total_lens)}")
        print(f"\nIf training has many long assistant responses, the model learns to be verbose.")
        print(f"For small models, this can cause loops when trying to hit learned response lengths.")
        print("="*80)


if __name__ == "__main__":
    import sys
    sys.exit(main())
