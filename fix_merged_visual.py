#!/usr/bin/env python3
"""
Patch a merged model directory to include weights that AutoModelForCausalLM
does not load (visual encoder, MTP head, etc.).

Compares the base model's safetensors against the merged model's safetensors
and copies any keys that are present in the base but absent from the merged
output into a new shard, then updates (or creates) the weight index.

Usage:
    python fix_merged_visual.py <merged_dir> [--base Qwen/Qwen3.5-2B]
"""

import argparse
import json
import sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file


def find_model_dir(model_name):
    if Path(model_name).is_dir():
        return Path(model_name)
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_name, local_files_only=True))
    except Exception as e:
        raise RuntimeError(
            f"Could not locate '{model_name}' locally.\n"
            f"  If it's a local directory, check the path.\n"
            f"  If it's a HF model, make sure it's downloaded (error: {e})"
        )


def collect_keys(model_dir):
    keys = set()
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            keys.update(f.keys())
    return keys


def load_tensors(model_dir, keys_to_load):
    """Load specific tensors from a model directory."""
    tensors = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in keys_to_load:
                    tensors[k] = f.get_tensor(k)
        if len(tensors) == len(keys_to_load):
            break
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("merged_dir", help="Path to the merged model directory to patch")
    parser.add_argument("--base", default="Qwen/Qwen3.5-2B",
                        help="Reference model: HF name or local path (default: Qwen/Qwen3.5-2B)")
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    if not merged_dir.is_dir():
        print(f"ERROR: {merged_dir} is not a directory")
        sys.exit(1)

    print(f"Locating base model: {args.base}")
    base_dir = find_model_dir(args.base)
    print(f"  resolved to: {base_dir}")

    base_keys = collect_keys(base_dir)
    merged_keys = collect_keys(merged_dir)

    missing = base_keys - merged_keys
    if not missing:
        print("No missing keys — merged model already matches base. Nothing to do.")
        sys.exit(0)

    # Group by depth-2 prefix for reporting
    groups = {}
    for k in sorted(missing):
        prefix = ".".join(k.split(".")[:2])
        groups.setdefault(prefix, []).append(k)

    print(f"\nFound {len(missing)} tensors in base that are missing from merged:")
    for prefix, ks in sorted(groups.items()):
        print(f"  {prefix}.*  ({len(ks)} tensors)")

    print("\nLoading missing tensors from base model ...")
    extra = load_tensors(base_dir, missing)
    print(f"  loaded {len(extra)} tensors")

    shard_name = "model_extra_components.safetensors"
    save_file(extra, str(merged_dir / shard_name))
    print(f"  saved to {shard_name}")

    # Update or create the safetensors index
    idx_path = merged_dir / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            idx = json.load(f)
    else:
        # Build index from existing single-shard merged file
        idx = {"metadata": {}, "weight_map": {}}
        msf = merged_dir / "model.safetensors"
        if msf.exists():
            with safe_open(str(msf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    idx["weight_map"][k] = "model.safetensors"

    for k in extra:
        idx["weight_map"][k] = shard_name

    with open(idx_path, "w") as f:
        json.dump(idx, f)
    print(f"  updated {idx_path.name}")

    print("\nDone. Run compare_model_keys.py to verify.")


if __name__ == "__main__":
    main()
