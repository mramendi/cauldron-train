#!/usr/bin/env python3
"""
Optimizer builder for Cauldron training scripts.

Provides factory functions for creating optimizers from configuration dictionaries.
Supports AdamW, Muon, and NorMuon optimizers with sensible defaults.
"""

import torch
from torch.optim import AdamW
from typing import Dict, Any, Iterable
from config_utils import get_config_value, ConfigError


def build_optimizer(
    config: dict,
    model_parameters: Iterable[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    """
    Build an optimizer from configuration.

    Args:
        config: Configuration dictionary with 'optimizer' section
        model_parameters: Model parameters to optimize

    Returns:
        Configured optimizer instance

    Raises:
        ConfigError: If optimizer config is invalid
        ImportError: If required optimizer library is not available

    Example config:
        {
            "optimizer": {
                "type": "adamw",
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8
            }
        }
    """
    if "optimizer" not in config:
        raise ConfigError("Config missing 'optimizer' section")

    opt_cfg = config["optimizer"]
    opt_type = get_config_value(opt_cfg, "type", required=True).lower()
    learning_rate = get_config_value(opt_cfg, "learning_rate", required=True)

    if opt_type == "adamw":
        return build_adamw(opt_cfg, model_parameters, learning_rate)
    elif opt_type == "muon":
        return build_muon(opt_cfg, model_parameters, learning_rate)
    elif opt_type == "normuon":
        return build_normuon(opt_cfg, model_parameters, learning_rate)
    else:
        raise ConfigError(
            f"Unknown optimizer type: {opt_type}. "
            f"Supported types: adamw, muon, normuon"
        )


def build_adamw(
    config: dict,
    model_parameters: Iterable[torch.nn.Parameter],
    learning_rate: float
) -> AdamW:
    """
    Build AdamW optimizer from configuration.

    Args:
        config: Optimizer configuration dictionary
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate

    Returns:
        AdamW optimizer instance

    Config options:
        - weight_decay: Weight decay coefficient (default: 0.0)
        - beta1: Adam beta1 coefficient (default: 0.9)
        - beta2: Adam beta2 coefficient (default: 0.999)
        - epsilon: Adam epsilon for numerical stability (default: 1e-8)
    """
    weight_decay = get_config_value(config, "weight_decay", default=0.0)
    beta1 = get_config_value(config, "beta1", default=0.9)
    beta2 = get_config_value(config, "beta2", default=0.999)
    epsilon = get_config_value(config, "epsilon", default=1e-8)

    return AdamW(
        model_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay
    )


def build_muon(
    config: dict,
    model_parameters: Iterable[torch.nn.Parameter],
    learning_rate: float
) -> "Muon":
    """
    Build Muon optimizer from configuration.

    Muon is a momentum-based optimizer designed for transformer training.
    Requires the muon optimizer package to be installed.

    Args:
        config: Optimizer configuration dictionary
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate

    Returns:
        Muon optimizer instance

    Raises:
        ImportError: If muon package is not installed

    Config options:
        - momentum: Momentum coefficient (default: 0.95)
        - nesterov: Use Nesterov momentum (default: True)
        - backend: Backend implementation (default: "newtonschulz5")
        - backend_steps: Backend iteration steps (default: 5)
    """
    try:
        from muon import Muon
    except ImportError:
        raise ImportError(
            "Muon optimizer not found. Install with: pip install muon-optimizer"
        )

    momentum = get_config_value(config, "momentum", default=0.95)
    nesterov = get_config_value(config, "nesterov", default=True)
    backend = get_config_value(config, "backend", default="newtonschulz5")
    backend_steps = get_config_value(config, "backend_steps", default=5)

    return Muon(
        model_parameters,
        lr=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
        backend=backend,
        backend_steps=backend_steps
    )


def build_normuon(
    config: dict,
    model_parameters: Iterable[torch.nn.Parameter],
    learning_rate: float
) -> "SingleDeviceNorMuonWithAuxAdam":
    """
    Build NorMuon optimizer from configuration.

    NorMuon is a normalized Muon variant that applies NorMuon to 2D+ parameters
    and Adam to 1D parameters via SingleDeviceNorMuonWithAuxAdam.
    Requires: pip install git+https://github.com/zichongli5/NorMuon.git

    Args:
        config: Optimizer configuration dictionary
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate

    Returns:
        SingleDeviceNorMuonWithAuxAdam optimizer instance

    Raises:
        ImportError: If normuon package is not installed

    Config options:
        - weight_decay: Weight decay coefficient (default: 0.0)
    """
    try:
        from normuon import SingleDeviceNorMuonWithAuxAdam
    except ImportError:
        raise ImportError(
            "NorMuon not installed. Install with: "
            "pip install git+https://github.com/zichongli5/NorMuon.git"
        )

    weight_decay = get_config_value(config, "weight_decay", default=0.0)

    normuon_params = []
    adam_params = []
    for param in model_parameters:
        if param.requires_grad:
            if param.ndim >= 2:
                normuon_params.append(param)
            else:
                adam_params.append(param)

    print(f"\nOptimizer parameter split:")
    print(f"  NorMuon (2D+): {len(normuon_params)} parameters")
    print(f"  Adam (1D): {len(adam_params)} parameters")

    param_groups = []
    if normuon_params:
        param_groups.append({
            "params": normuon_params,
            "use_muon": True,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        })
    if adam_params:
        param_groups.append({
            "params": adam_params,
            "use_muon": False,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        })

    return SingleDeviceNorMuonWithAuxAdam(param_groups)


def get_optimizer_param_groups(
    model: torch.nn.Module,
    config: dict
) -> list:
    """
    Create parameter groups for optimizer with optional custom settings.

    This allows different learning rates or weight decay for different
    parameter groups (e.g., no decay on biases and layer norms).

    Args:
        model: PyTorch model
        config: Configuration dictionary

    Returns:
        List of parameter group dictionaries

    Example config:
        {
            "optimizer": {
                "type": "adamw",
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "param_groups": {
                    "no_decay": {
                        "patterns": ["bias", "LayerNorm", "layer_norm"],
                        "weight_decay": 0.0
                    }
                }
            }
        }
    """
    opt_cfg = config.get("optimizer", {})
    param_groups_cfg = get_config_value(opt_cfg, "param_groups", default=None)

    if param_groups_cfg is None:
        # No custom parameter groups, return all parameters
        return [{"params": model.parameters()}]

    # Build parameter groups
    param_groups = []
    assigned_params = set()

    # Process custom groups
    for group_name, group_cfg in param_groups_cfg.items():
        patterns = get_config_value(group_cfg, "patterns", default=[])
        if not patterns:
            continue

        # Collect parameters matching patterns
        group_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and id(param) not in assigned_params:
                if any(pattern in name for pattern in patterns):
                    group_params.append(param)
                    assigned_params.add(id(param))

        if group_params:
            # Create param group with custom settings
            param_group = {"params": group_params}

            # Copy relevant optimizer settings
            if "weight_decay" in group_cfg:
                param_group["weight_decay"] = group_cfg["weight_decay"]
            if "learning_rate" in group_cfg:
                param_group["lr"] = group_cfg["learning_rate"]

            param_groups.append(param_group)

    # Add remaining parameters to default group
    remaining_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in assigned_params
    ]

    if remaining_params:
        param_groups.append({"params": remaining_params})

    return param_groups


def print_optimizer_summary(optimizer: torch.optim.Optimizer) -> None:
    """
    Print a human-readable summary of the optimizer configuration.

    Args:
        optimizer: PyTorch optimizer instance
    """
    print(f"\nOptimizer: {optimizer.__class__.__name__}")
    print(f"  Parameter groups: {len(optimizer.param_groups)}")

    for i, group in enumerate(optimizer.param_groups):
        num_params = len(group["params"])
        total_params = sum(p.numel() for p in group["params"])

        print(f"\n  Group {i}:")
        print(f"    Parameters: {num_params:,} tensors ({total_params:,} elements)")
        print(f"    Learning rate: {group.get('lr', 'N/A')}")
        print(f"    Weight decay: {group.get('weight_decay', 0.0)}")

        # Print optimizer-specific settings
        if hasattr(optimizer, 'defaults'):
            for key, value in optimizer.defaults.items():
                if key not in ['lr', 'weight_decay', 'params']:
                    print(f"    {key}: {value}")
