#!/usr/bin/env python3
"""
Learning rate scheduler builder for Cauldron training scripts.

Provides factory functions for creating learning rate schedulers from configuration.
Supports constant, linear, cosine, wWSD (warmup-Warmup-Stable-Decay), and
epoch shock absorber schedulers.
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ConstantLR
from typing import Optional
from config_utils import get_config_value, ConfigError


def build_scheduler(
    config: dict,
    optimizer: Optimizer,
    num_training_steps: int,
    steps_per_epoch: Optional[int] = None
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a learning rate scheduler from configuration.

    Args:
        config: Configuration dictionary with 'scheduler' section
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps
        steps_per_epoch: Steps per epoch (required for epoch shock absorber)

    Returns:
        Configured scheduler instance

    Raises:
        ConfigError: If scheduler config is invalid

    Example config:
        {
            "scheduler": {
                "type": "wwsd",
                "warmup_steps": 100,
                "prewarmup_steps": 50,
                "prewarmup_lr_ratio": 0.1,
                "stable_steps": 200,
                "num_cycles": 1,
                "epoch_shock_absorber": {
                    "enabled": true,
                    "steps_per_epoch": 1000,
                    "shock_recovery_steps": 100
                }
            }
        }
    """
    if "scheduler" not in config:
        raise ConfigError("Config missing 'scheduler' section")

    sched_cfg = config["scheduler"]
    sched_type = get_config_value(sched_cfg, "type", required=True).lower()

    # Build base scheduler
    if sched_type == "constant":
        base_scheduler = build_constant_scheduler(sched_cfg, optimizer)
    elif sched_type == "linear":
        base_scheduler = build_linear_scheduler(
            sched_cfg, optimizer, num_training_steps
        )
    elif sched_type == "cosine":
        base_scheduler = build_cosine_scheduler(
            sched_cfg, optimizer, num_training_steps
        )
    elif sched_type == "wwsd":
        base_scheduler = build_wwsd_scheduler(
            sched_cfg, optimizer, num_training_steps
        )
    else:
        raise ConfigError(
            f"Unknown scheduler type: {sched_type}. "
            f"Supported types: constant, linear, cosine, wwsd"
        )

    # Optionally wrap with epoch shock absorber
    shock_cfg = get_config_value(sched_cfg, "epoch_shock_absorber", default=None)
    if shock_cfg and get_config_value(shock_cfg, "enabled", default=False):
        # Get steps_per_epoch from shock config or function argument
        spe = get_config_value(shock_cfg, "steps_per_epoch", default=steps_per_epoch)
        if spe is None:
            raise ConfigError(
                "Epoch shock absorber requires 'steps_per_epoch' in config "
                "or as function argument"
            )

        shock_steps = get_config_value(
            shock_cfg, "shock_recovery_steps", default=100
        )

        return EpochShockAbsorberScheduler(
            base_scheduler=base_scheduler,
            steps_per_epoch=spe,
            shock_steps=shock_steps
        )

    return base_scheduler


def build_constant_scheduler(
    config: dict,
    optimizer: Optimizer
) -> ConstantLR:
    """
    Build constant learning rate scheduler.

    The learning rate stays constant throughout training, optionally with
    a warmup period.

    Args:
        config: Scheduler configuration dictionary
        optimizer: Optimizer to schedule

    Returns:
        Constant LR scheduler instance

    Config options:
        - warmup_steps: Number of warmup steps (default: 0)
          If > 0, LR linearly increases from 0 to full LR over warmup_steps
    """
    warmup_steps = get_config_value(config, "warmup_steps", default=0)

    if warmup_steps > 0:
        # Use lambda scheduler for warmup + constant
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)
    else:
        # Pure constant (factor=1.0 means no change)
        return ConstantLR(optimizer, factor=1.0, total_iters=1)


def build_linear_scheduler(
    config: dict,
    optimizer: Optimizer,
    num_training_steps: int
) -> LambdaLR:
    """
    Build linear decay scheduler with optional warmup.

    The learning rate linearly increases during warmup, then linearly
    decreases to 0 over the remaining steps.

    Args:
        config: Scheduler configuration dictionary
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps

    Returns:
        Linear decay scheduler instance

    Config options:
        - warmup_steps: Number of warmup steps (default: 0)
        - min_lr_ratio: Minimum LR as ratio of initial LR (default: 0.0)
    """
    warmup_steps = get_config_value(config, "warmup_steps", default=0)
    min_lr_ratio = get_config_value(config, "min_lr_ratio", default=0.0)

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Linear decay phase
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda)


def build_cosine_scheduler(
    config: dict,
    optimizer: Optimizer,
    num_training_steps: int
) -> LambdaLR:
    """
    Build cosine annealing scheduler with optional warmup.

    The learning rate linearly increases during warmup, then follows
    a cosine decay curve.

    Args:
        config: Scheduler configuration dictionary
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps

    Returns:
        Cosine annealing scheduler instance

    Config options:
        - warmup_steps: Number of warmup steps (default: 0)
        - num_cycles: Number of cosine cycles (default: 0.5 for half cycle)
        - min_lr_ratio: Minimum LR as ratio of initial LR (default: 0.0)
    """
    warmup_steps = get_config_value(config, "warmup_steps", default=0)
    num_cycles = get_config_value(config, "num_cycles", default=0.5)
    min_lr_ratio = get_config_value(config, "min_lr_ratio", default=0.0)

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine annealing phase
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def build_wwsd_scheduler(
    config: dict,
    optimizer: Optimizer,
    num_training_steps: int
) -> LambdaLR:
    """
    Build wWSD (warmup-Warmup-Stable-Decay) scheduler.

    Four-phase scheduler:
    1. Prewarmup: Linear ramp from 0 to prewarmup_lr_ratio * LR
    2. Warmup: Linear ramp from prewarmup_lr_ratio to full LR
    3. Stable: Constant at full LR
    4. Decay: Cosine decay from full LR back to prewarmup_lr_ratio * LR

    Args:
        config: Scheduler configuration dictionary
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps

    Returns:
        wWSD scheduler instance

    Config options:
        - warmup_steps: Steps for warmup phase (required)
        - stable_steps: Steps for stable phase (required)
        - prewarmup_steps: Steps for prewarmup phase (default: 0)
        - prewarmup_lr_ratio: Target LR for prewarmup as ratio (default: 0.1)
        - num_cycles: Number of cosine cycles in decay (default: 0.5)
    """
    # Required parameters
    num_warmup_steps = get_config_value(config, "warmup_steps", required=True)
    num_stable_steps = get_config_value(config, "stable_steps", required=True)

    # Optional parameters
    num_prewarmup_steps = get_config_value(config, "prewarmup_steps", default=0)
    prewarmup_lr_ratio = get_config_value(config, "prewarmup_lr_ratio", default=0.1)
    num_cycles = get_config_value(config, "num_cycles", default=0.5)

    return get_wwsd_schedule_with_warmup(
        optimizer=optimizer,
        num_prewarmup_steps=num_prewarmup_steps,
        prewarmup_lr_ratio=prewarmup_lr_ratio,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles
    )


def get_wwsd_schedule_with_warmup(
    optimizer: Optimizer,
    num_prewarmup_steps: int,
    prewarmup_lr_ratio: float,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """
    Create a wWSD schedule with prewarmup, warmup, stable, and cosine decay phases.

    Phases:
    1. Prewarmup: Linear ramp from 0 to prewarmup_lr_ratio * LR (e.g., 0 -> 0.1 * LR)
    2. Warmup: Linear ramp from prewarmup_lr_ratio to full LR (e.g., 0.1 * LR -> LR)
    3. Stable: Constant at full LR
    4. Decay: Cosine decay from full LR back to prewarmup_lr_ratio * LR (symmetric)

    Args:
        optimizer: Optimizer
        num_prewarmup_steps: Steps for prewarmup phase
        prewarmup_lr_ratio: Target LR for prewarmup as ratio of full LR (e.g., 0.1 = 10%)
                           Also used as minimum LR for decay phase (symmetric design)
        num_warmup_steps: Steps for warmup phase (from prewarmup_lr to full LR)
        num_stable_steps: Steps for stable phase
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (default: 0.5 for half cycle)

    Returns:
        LambdaLR scheduler
    """
    # Calculate decay steps
    num_decay_steps = (
        num_training_steps - num_prewarmup_steps - num_warmup_steps - num_stable_steps
    )

    if num_decay_steps < 0:
        raise ValueError(
            f"Invalid step configuration: prewarmup({num_prewarmup_steps}) + "
            f"warmup({num_warmup_steps}) + stable({num_stable_steps}) > "
            f"total({num_training_steps})"
        )

    def lr_lambda(current_step: int):
        # Phase 1: Prewarmup (0 -> prewarmup_lr_ratio)
        if current_step < num_prewarmup_steps:
            if num_prewarmup_steps == 0:
                return prewarmup_lr_ratio
            return (
                float(current_step) / float(max(1, num_prewarmup_steps))
                * prewarmup_lr_ratio
            )

        # Phase 2: Warmup (prewarmup_lr_ratio -> 1.0)
        elif current_step < num_prewarmup_steps + num_warmup_steps:
            progress = float(current_step - num_prewarmup_steps) / float(
                max(1, num_warmup_steps)
            )
            return prewarmup_lr_ratio + (1.0 - prewarmup_lr_ratio) * progress

        # Phase 3: Stable (1.0)
        elif current_step < num_prewarmup_steps + num_warmup_steps + num_stable_steps:
            return 1.0

        # Phase 4: Cosine Decay (1.0 -> prewarmup_lr_ratio)
        else:
            progress = float(
                current_step - num_prewarmup_steps - num_warmup_steps - num_stable_steps
            )
            progress = progress / float(max(1, num_decay_steps))

            # Cosine decay from 1.0 to prewarmup_lr_ratio
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
            return prewarmup_lr_ratio + (1.0 - prewarmup_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class EpochShockAbsorberScheduler:
    """
    Wrapper scheduler that applies epoch shock absorption on top of a base scheduler.

    At epoch boundaries, applies a 0.5x LR multiplier, then linearly recovers
    to 1.0x over shock_steps. This helps stabilize training at epoch boundaries
    where data distribution may shift.

    Args:
        base_scheduler: Base learning rate scheduler
        steps_per_epoch: Number of training steps per epoch
        shock_steps: Number of steps to recover from shock (linear ramp 0.5x -> 1.0x)

    Example:
        # Create base scheduler
        base = get_cosine_schedule_with_warmup(optimizer, 100, 1000)

        # Wrap with shock absorber
        scheduler = EpochShockAbsorberScheduler(
            base_scheduler=base,
            steps_per_epoch=200,
            shock_steps=50
        )

        # Use as normal scheduler
        for step in range(num_steps):
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        base_scheduler: torch.optim.lr_scheduler._LRScheduler,
        steps_per_epoch: int,
        shock_steps: int
    ):
        self.base_scheduler = base_scheduler
        self.steps_per_epoch = steps_per_epoch
        self.shock_steps = shock_steps
        self.current_step = 0
        self.optimizer = base_scheduler.optimizer

    def get_shock_multiplier(self, step: int) -> float:
        """
        Calculate shock multiplier for given step.

        Returns:
            Multiplier in range [0.5, 1.0]
        """
        # No shock before we've completed one epoch
        if step <= self.steps_per_epoch:
            return 1.0

        # Check if we're at an epoch boundary (step = N+1, 2N+1, 3N+1, ...)
        # This happens when step % steps_per_epoch == 1
        steps_since_shock = (step % self.steps_per_epoch) - 1

        if steps_since_shock < 0:
            # step % steps_per_epoch == 0 means we're at step N, 2N, 3N, ...
            # Not at a boundary, just continue with last shock state
            return 1.0

        if steps_since_shock == 0:
            # We're at an epoch boundary! Apply shock
            return 0.5
        elif steps_since_shock <= self.shock_steps:
            # Linearly recover from 0.5 to 1.0
            progress = float(steps_since_shock) / float(self.shock_steps)
            return 0.5 + 0.5 * progress
        else:
            # Normal operation
            return 1.0

    def step(self):
        """Step both base scheduler and apply shock multiplier."""
        # Step the base scheduler
        self.base_scheduler.step()
        self.current_step += 1

        # Apply shock multiplier to all param groups
        multiplier = self.get_shock_multiplier(self.current_step)
        base_lrs = self.base_scheduler.get_last_lr()

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = base_lrs[i] * multiplier

    def get_last_lr(self):
        """Get current LR with shock multiplier applied."""
        multiplier = self.get_shock_multiplier(self.current_step)
        base_lrs = self.base_scheduler.get_last_lr()
        return [lr * multiplier for lr in base_lrs]

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.current_step = state_dict['current_step']


def print_scheduler_summary(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_training_steps: int
) -> None:
    """
    Print a human-readable summary of the scheduler configuration.

    Args:
        scheduler: Learning rate scheduler instance
        num_training_steps: Total number of training steps
    """
    scheduler_name = scheduler.__class__.__name__

    # Handle wrapped schedulers
    if isinstance(scheduler, EpochShockAbsorberScheduler):
        print(f"\nScheduler: {scheduler_name}")
        print(f"  Base: {scheduler.base_scheduler.__class__.__name__}")
        print(f"  Steps per epoch: {scheduler.steps_per_epoch}")
        print(f"  Shock recovery steps: {scheduler.shock_steps}")
        scheduler = scheduler.base_scheduler
    else:
        print(f"\nScheduler: {scheduler_name}")

    # Get initial LR
    initial_lr = scheduler.optimizer.param_groups[0]['lr']
    print(f"  Initial LR: {initial_lr}")

    # Print current LR
    try:
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Current LR: {current_lr}")
    except Exception:
        pass

    print(f"  Total steps: {num_training_steps}")


def get_scheduler_lr_at_step(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    base_lr: float = 1.0
) -> float:
    """
    Calculate what the learning rate would be at a specific step.

    Useful for previewing the learning rate schedule.

    Args:
        scheduler: Learning rate scheduler
        step: Step number to query
        base_lr: Base learning rate (multiplied by schedule)

    Returns:
        Learning rate at the given step

    Note:
        This function works for LambdaLR schedulers. For other scheduler types,
        it may not be accurate.
    """
    if isinstance(scheduler, LambdaLR):
        # LambdaLR stores the lambda function
        lr_lambda = scheduler.lr_lambdas[0]
        return base_lr * lr_lambda(step)
    elif isinstance(scheduler, EpochShockAbsorberScheduler):
        # Get base LR from inner scheduler
        base_sched = scheduler.base_scheduler
        if isinstance(base_sched, LambdaLR):
            lr_lambda = base_sched.lr_lambdas[0]
            base_lr_at_step = base_lr * lr_lambda(step)
            # Apply shock multiplier
            multiplier = scheduler.get_shock_multiplier(step)
            return base_lr_at_step * multiplier
        else:
            raise NotImplementedError(
                f"Cannot query LR for scheduler type: {type(base_sched)}"
            )
    else:
        raise NotImplementedError(
            f"Cannot query LR for scheduler type: {type(scheduler)}"
        )
