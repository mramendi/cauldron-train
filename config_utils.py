#!/usr/bin/env python3
"""
Configuration helper utilities for Cauldron training scripts.

Provides functions for loading, validating, and accessing configuration values
with sensible defaults and type checking.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


def load_json_config(config_path: Union[str, Path]) -> dict:
    """
    Load a JSON configuration file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return json.load(f)


def get_config_value(
    config: dict,
    path: str,
    default: Any = None,
    required: bool = False,
    value_type: Optional[type] = None
) -> Any:
    """
    Get a nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        path: Dot-separated path to value (e.g., "model.name")
        default: Default value if path doesn't exist
        required: If True, raise error if value not found
        value_type: Expected type of value (for validation)

    Returns:
        Configuration value or default

    Raises:
        ConfigError: If required value is missing or has wrong type

    Examples:
        >>> config = {"model": {"name": "granite", "size": "1b"}}
        >>> get_config_value(config, "model.name")
        'granite'
        >>> get_config_value(config, "model.dtype", default="bf16")
        'bf16'
    """
    keys = path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if required:
                raise ConfigError(f"Required config value missing: {path}")
            return default

    # Type validation if requested
    if value_type is not None and value is not None:
        if not isinstance(value, value_type):
            raise ConfigError(
                f"Config value '{path}' has wrong type. "
                f"Expected {value_type.__name__}, got {type(value).__name__}"
            )

    return value


def validate_required_sections(config: dict, required_sections: List[str]) -> None:
    """
    Validate that all required top-level sections exist in config.

    Args:
        config: Configuration dictionary
        required_sections: List of required section names

    Raises:
        ConfigError: If any required section is missing
    """
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"Config missing required sections: {', '.join(missing)}")


def validate_model_config(config: dict) -> None:
    """
    Validate model configuration section.

    Required fields:
        - model.name: Model identifier

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If model config is invalid
    """
    if "model" not in config:
        raise ConfigError("Config missing 'model' section")

    model_cfg = config["model"]

    if "name" not in model_cfg:
        raise ConfigError("Model config missing required field: 'name'")

    # Validate optional boolean fields
    for field in ["bf16_logits_patch"]:
        if field in model_cfg and not isinstance(model_cfg[field], bool):
            raise ConfigError(f"Model config field '{field}' must be boolean")


def validate_data_config(config: dict) -> None:
    """
    Validate data configuration section.

    Required fields:
        - data.train_dir OR datasets (for training)
        - data.eval_dir (for evaluation)

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If data config is invalid
    """
    if "data" not in config:
        raise ConfigError("Config missing 'data' section")

    data_cfg = config["data"]

    # Check for either train_dir or datasets
    has_train_dir = "train_dir" in data_cfg
    has_datasets = "datasets" in config

    if not (has_train_dir or has_datasets):
        raise ConfigError(
            "Data config must have either 'train_dir' or top-level 'datasets' section"
        )


def validate_lora_config(config: dict) -> None:
    """
    Validate LoRA configuration section.

    Required fields:
        - lora.base_rank: Base rank for LoRA adapters
        - lora.base_alpha: Base alpha for LoRA adapters

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If LoRA config is invalid
    """
    if "lora" not in config:
        raise ConfigError("Config missing 'lora' section")

    lora_cfg = config["lora"]

    # Validate base parameters
    for field in ["base_rank", "base_alpha"]:
        if field not in lora_cfg:
            raise ConfigError(f"LoRA config missing required field: '{field}'")

        value = lora_cfg[field]
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"LoRA config field '{field}' must be positive integer")

    # Validate dropout if present
    if "dropout" in lora_cfg:
        dropout = lora_cfg["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            raise ConfigError("LoRA dropout must be a number between 0 and 1")

    # Validate layer groups if present
    if "layer_groups" in lora_cfg:
        if not isinstance(lora_cfg["layer_groups"], list):
            raise ConfigError("LoRA 'layer_groups' must be a list")

        for i, group in enumerate(lora_cfg["layer_groups"]):
            if not isinstance(group, dict):
                raise ConfigError(f"LoRA layer_groups[{i}] must be a dictionary")

            if "name" not in group:
                raise ConfigError(f"LoRA layer_groups[{i}] missing 'name' field")

            if "modules" not in group:
                raise ConfigError(f"LoRA layer_groups[{i}] missing 'modules' field")

            if not isinstance(group["modules"], list):
                raise ConfigError(f"LoRA layer_groups[{i}] 'modules' must be a list")


def validate_optimizer_config(config: dict) -> None:
    """
    Validate optimizer configuration section.

    Required fields:
        - optimizer.type: Optimizer type (adamw, muon, normuon)
        - optimizer.learning_rate: Learning rate

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If optimizer config is invalid
    """
    if "optimizer" not in config:
        raise ConfigError("Config missing 'optimizer' section")

    opt_cfg = config["optimizer"]

    # Validate type
    if "type" not in opt_cfg:
        raise ConfigError("Optimizer config missing required field: 'type'")

    opt_type = opt_cfg["type"].lower()
    valid_types = ["adamw", "muon", "normuon"]
    if opt_type not in valid_types:
        raise ConfigError(
            f"Optimizer type must be one of {valid_types}, got '{opt_type}'"
        )

    # Validate learning rate
    if "learning_rate" not in opt_cfg:
        raise ConfigError("Optimizer config missing required field: 'learning_rate'")

    lr = opt_cfg["learning_rate"]
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ConfigError("Optimizer learning_rate must be a positive number")

    # Validate weight decay if present
    if "weight_decay" in opt_cfg:
        wd = opt_cfg["weight_decay"]
        if not isinstance(wd, (int, float)) or wd < 0:
            raise ConfigError("Optimizer weight_decay must be non-negative")


def validate_scheduler_config(config: dict) -> None:
    """
    Validate scheduler configuration section.

    Required fields:
        - scheduler.type: Scheduler type (constant, linear, cosine, wwsd)

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If scheduler config is invalid
    """
    if "scheduler" not in config:
        raise ConfigError("Config missing 'scheduler' section")

    sched_cfg = config["scheduler"]

    # Validate type
    if "type" not in sched_cfg:
        raise ConfigError("Scheduler config missing required field: 'type'")

    sched_type = sched_cfg["type"].lower()
    valid_types = ["constant", "linear", "cosine", "wwsd"]
    if sched_type not in valid_types:
        raise ConfigError(
            f"Scheduler type must be one of {valid_types}, got '{sched_type}'"
        )

    # Validate wWSD-specific fields
    if sched_type == "wwsd":
        required_wwsd = ["warmup_steps", "stable_steps"]
        for field in required_wwsd:
            if field not in sched_cfg:
                raise ConfigError(
                    f"wWSD scheduler requires '{field}' field"
                )

            value = sched_cfg[field]
            if not isinstance(value, int) or value < 0:
                raise ConfigError(
                    f"Scheduler '{field}' must be non-negative integer"
                )


def validate_training_config(config: dict) -> None:
    """
    Validate training configuration section.

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If training config is invalid
    """
    if "training" not in config:
        raise ConfigError("Config missing 'training' section")

    train_cfg = config["training"]

    # Validate max_grad_norm if present
    if "max_grad_norm" in train_cfg:
        mgn = train_cfg["max_grad_norm"]
        if mgn is not None and (not isinstance(mgn, (int, float)) or mgn <= 0):
            raise ConfigError("Training max_grad_norm must be positive number or null")

    # Validate label_smoothing if present
    if "label_smoothing" in train_cfg:
        ls = train_cfg["label_smoothing"]
        if not isinstance(ls, (int, float)) or not (0 <= ls < 1):
            raise ConfigError("Training label_smoothing must be between 0 and 1")

    # Validate early stopping if present
    if "early_stopping" in train_cfg:
        es_cfg = train_cfg["early_stopping"]
        if not isinstance(es_cfg, dict):
            raise ConfigError("Training early_stopping must be a dictionary")

        if "patience" in es_cfg:
            patience = es_cfg["patience"]
            if not isinstance(patience, int) or patience < 1:
                raise ConfigError("Early stopping patience must be positive integer")


def validate_output_config(config: dict) -> None:
    """
    Validate output configuration section.

    Required fields:
        - output.root_dir: Root directory for outputs

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If output config is invalid
    """
    if "output" not in config:
        raise ConfigError("Config missing 'output' section")

    out_cfg = config["output"]

    if "root_dir" not in out_cfg:
        raise ConfigError("Output config missing required field: 'root_dir'")

    # Validate auto_merge if present
    if "auto_merge" in out_cfg:
        if not isinstance(out_cfg["auto_merge"], bool):
            raise ConfigError("Output auto_merge must be boolean")


def load_training_config(config_path: Union[str, Path]) -> dict:
    """
    Load and validate a complete training configuration file.

    Performs comprehensive validation of all config sections.

    Args:
        config_path: Path to JSON config file

    Returns:
        Validated config dictionary with defaults applied

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigError: If config is invalid
    """
    config_path = Path(config_path)
    config = load_json_config(config_path)

    # Validate all required sections exist
    required_sections = [
        "model", "data", "lora", "optimizer",
        "scheduler", "training", "output"
    ]
    validate_required_sections(config, required_sections)

    # Validate each section
    validate_model_config(config)
    validate_data_config(config)
    validate_lora_config(config)
    validate_optimizer_config(config)
    validate_scheduler_config(config)
    validate_training_config(config)
    validate_output_config(config)

    # Apply defaults
    if config["output"].get("prefix") is None:
        config["output"]["prefix"] = config_path.stem  # Filename without .json

    return config


def load_dataset_config(config_path: Union[str, Path]) -> dict:
    """
    Load and validate a dataset configuration file.

    Expected format:
    {
        "datasets": [
            {"name": "...", "path": "...", "upsample": 1}
        ]
    }

    Args:
        config_path: Path to JSON config file

    Returns:
        Validated config dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigError: If config is invalid
    """
    config = load_json_config(config_path)

    if "datasets" not in config:
        raise ConfigError("Dataset config missing 'datasets' list")

    datasets = config["datasets"]
    if not isinstance(datasets, list):
        raise ConfigError("'datasets' must be a list")

    if len(datasets) == 0:
        raise ConfigError("'datasets' list cannot be empty")

    # Validate each dataset entry
    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ConfigError(f"datasets[{i}] must be a dictionary")

        required_fields = ["name", "path"]
        for field in required_fields:
            if field not in dataset:
                raise ConfigError(f"datasets[{i}] missing required field: '{field}'")

        # Validate upsample if present
        if "upsample" in dataset:
            upsample = dataset["upsample"]
            if not isinstance(upsample, int) or upsample < 1:
                raise ConfigError(
                    f"datasets[{i}] upsample must be positive integer, got {upsample}"
                )
        else:
            # Set default
            dataset["upsample"] = 1

    return config


def merge_configs(base: dict, override: dict) -> dict:
    """
    Merge two configuration dictionaries, with override taking precedence.

    Performs deep merge: nested dictionaries are merged recursively,
    lists and other values are replaced.

    Args:
        base: Base configuration
        override: Configuration with override values

    Returns:
        Merged configuration dictionary

    Examples:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"d": 4}, "e": 5}
        >>> merge_configs(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 4}, 'e': 5}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Replace value
            result[key] = value

    return result


def get_output_dir(config: dict, postfix: Optional[str] = None) -> Path:
    """
    Generate output directory path from configuration.

    Output directory is constructed as:
        <root_dir>/<prefix>[-<postfix>]

    Args:
        config: Configuration dictionary
        postfix: Optional postfix to append to directory name

    Returns:
        Path object for output directory

    Examples:
        >>> config = {"output": {"root_dir": "./results", "prefix": "exp1"}}
        >>> get_output_dir(config)
        PosixPath('results/exp1')
        >>> get_output_dir(config, postfix="run2")
        PosixPath('results/exp1-run2')
    """
    root_dir = Path(get_config_value(config, "output.root_dir", required=True))
    prefix = get_config_value(config, "output.prefix", required=True)

    if postfix:
        dir_name = f"{prefix}-{postfix}"
    else:
        dir_name = prefix

    return root_dir / dir_name


def print_config_summary(config: dict) -> None:
    """
    Print a human-readable summary of the configuration.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)

    # Model
    print(f"\nModel: {get_config_value(config, 'model.name', default='N/A')}")
    if get_config_value(config, "model.bf16_logits_patch", default=False):
        print("  - BF16 logits patch: ENABLED")

    # Data
    train_dir = get_config_value(config, "data.train_dir")
    eval_dir = get_config_value(config, "data.eval_dir")
    if train_dir:
        print(f"\nTraining data: {train_dir}")
    if eval_dir:
        print(f"Evaluation data: {eval_dir}")

    # LoRA
    base_rank = get_config_value(config, "lora.base_rank")
    base_alpha = get_config_value(config, "lora.base_alpha")
    dropout = get_config_value(config, "lora.dropout", default=0.0)
    print(f"\nLoRA: rank={base_rank}, alpha={base_alpha}, dropout={dropout}")

    layer_groups = get_config_value(config, "lora.layer_groups", default=[])
    if layer_groups:
        print("  Layer groups:")
        for group in layer_groups:
            enabled = group.get("enabled", True)
            status = "✓" if enabled else "✗"
            rank = group.get("rank") or base_rank
            alpha = group.get("alpha") or base_alpha
            print(f"    {status} {group['name']}: rank={rank}, alpha={alpha}")

    # Optimizer
    opt_type = get_config_value(config, "optimizer.type", default="N/A")
    lr = get_config_value(config, "optimizer.learning_rate", default=0)
    wd = get_config_value(config, "optimizer.weight_decay", default=0)
    print(f"\nOptimizer: {opt_type}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Weight decay: {wd}")

    # Scheduler
    sched_type = get_config_value(config, "scheduler.type", default="N/A")
    print(f"\nScheduler: {sched_type}")
    if sched_type == "wwsd":
        warmup = get_config_value(config, "scheduler.warmup_steps", default=0)
        prewarmup = get_config_value(config, "scheduler.prewarmup_steps", default=0)
        stable = get_config_value(config, "scheduler.stable_steps", default=0)
        print(f"  - Prewarmup: {prewarmup} steps")
        print(f"  - Warmup: {warmup} steps")
        print(f"  - Stable: {stable} steps")

    # Output
    root_dir = get_config_value(config, "output.root_dir", default="N/A")
    prefix = get_config_value(config, "output.prefix", default="N/A")
    auto_merge = get_config_value(config, "output.auto_merge", default=False)
    print(f"\nOutput: {root_dir}/{prefix}")
    if auto_merge:
        print("  - Auto-merge adapters: ENABLED")

    print("=" * 70 + "\n")
