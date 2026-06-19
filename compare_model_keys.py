#!/usr/bin/env python3
"""
Compare safetensors keys between two model directories.

Reports what keys are present in the reference (base) model but missing
from the target (merged) model, and vice versa.

Usage:
    python compare_model_keys.py <target_dir> [--base Qwen/Qwen3.5-2B]
"""

import argparse
import json
import sys
from pathlib import Path
from safetensors import safe_open


def find_model_dir(model_name):
    if Path(model_name).is_dir():
        return Path(model_name)
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_name, local_files_only=True))
    except Exception as e:
        raise RuntimeError(f"Could not locate '{model_name}' locally: {e}")


def collect_keys(model_dir):
    """Return {key: shape} for all tensors in a model directory."""
    keys = {}
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No .safetensors files in {model_dir}")
    for sf in shards:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for k in f.keys():
                keys[k] = tuple(f.get_slice(k).get_shape())
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target_dir", help="Model directory to check (e.g. merged model)")
    parser.add_argument("--base", default="Qwen/Qwen3.5-2B",
                        help="Reference model: HF name or local path (default: Qwen/Qwen3.5-2B)")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        print(f"ERROR: {target_dir} is not a directory")
        sys.exit(1)

    print(f"Reference (base): {args.base}")
    base_dir = find_model_dir(args.base)
    print(f"  resolved to: {base_dir}")
    print(f"Target (merged):  {target_dir}\n")

    print("Loading base keys ...")
    base_keys = collect_keys(base_dir)
    print(f"  {len(base_keys)} tensors\n")

    print("Loading target keys ...")
    target_keys = collect_keys(target_dir)
    print(f"  {len(target_keys)} tensors\n")

    only_in_base   = sorted(set(base_keys) - set(target_keys))
    only_in_target = sorted(set(target_keys) - set(base_keys))
    shape_mismatch = sorted(
        k for k in set(base_keys) & set(target_keys)
        if base_keys[k] != target_keys[k]
    )

    print("=" * 70)
    print(f"Only in BASE   (missing from target): {len(only_in_base)}")
    print(f"Only in TARGET (not in base):         {len(only_in_target)}")
    print(f"Shape mismatches (same key, diff shape): {len(shape_mismatch)}")
    print("=" * 70)

    if only_in_base:
        print(f"\n--- Missing from target ({len(only_in_base)} keys) ---")
        # Group by depth-2 prefix for readability
        groups = {}
        for k in only_in_base:
            prefix = ".".join(k.split(".")[:2])
            groups.setdefault(prefix, []).append(k)
        for prefix, ks in sorted(groups.items()):
            print(f"  {prefix}.*  ({len(ks)} tensors)")
            for k in ks[:3]:
                print(f"    {k}  {base_keys[k]}")
            if len(ks) > 3:
                print(f"    ... and {len(ks) - 3} more")

    if only_in_target:
        print(f"\n--- Only in target ({len(only_in_target)} keys) ---")
        groups = {}
        for k in only_in_target:
            prefix = ".".join(k.split(".")[:2])
            groups.setdefault(prefix, []).append(k)
        for prefix, ks in sorted(groups.items()):
            print(f"  {prefix}.*  ({len(ks)} tensors)")
            for k in ks[:3]:
                print(f"    {k}  {target_keys[k]}")
            if len(ks) > 3:
                print(f"    ... and {len(ks) - 3} more")

    if shape_mismatch:
        print(f"\n--- Shape mismatches ({len(shape_mismatch)} keys) ---")
        for k in shape_mismatch:
            print(f"  {k}: base={base_keys[k]}  target={target_keys[k]}")

    if not only_in_base and not only_in_target and not shape_mismatch:
        print("\nNo differences found — models have identical key sets and shapes.")


if __name__ == "__main__":
    main()
