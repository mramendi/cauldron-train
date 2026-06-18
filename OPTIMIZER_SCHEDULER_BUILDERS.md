# Optimizer and Scheduler Builders

This document describes the optimizer and scheduler builder utilities for Cauldron training scripts.

## Overview

The builder modules provide factory functions for creating optimizers and learning rate schedulers from JSON configuration dictionaries. This allows for clean separation between configuration and implementation, making it easy to experiment with different optimization strategies.

**Files:**
- `optimizer_builder.py` - Optimizer factory functions
- `scheduler_builder.py` - Scheduler factory functions
- `test_optimizer_scheduler_builders.py` - Comprehensive test suite

## Optimizer Builder

### Supported Optimizers

1. **AdamW** - Standard AdamW optimizer with decoupled weight decay
2. **Muon** - Momentum-based optimizer for transformers (requires `muon-optimizer` package)
3. **NorMuon** - Normalized variant of Muon (requires `muon-optimizer` package)

### Usage

```python
from optimizer_builder import build_optimizer

# Configure optimizer in JSON
config = {
    "optimizer": {
        "type": "adamw",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8
    }
}

# Build optimizer
optimizer = build_optimizer(config, model.parameters())
```

### Configuration Options

#### AdamW
```json
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
```

#### Muon
```json
{
    "optimizer": {
        "type": "muon",
        "learning_rate": 0.02,
        "momentum": 0.95,
        "nesterov": true,
        "backend": "newtonschulz5",
        "backend_steps": 5
    }
}
```

Requires: `pip install muon-optimizer`

#### NorMuon
```json
{
    "optimizer": {
        "type": "normuon",
        "learning_rate": 0.02,
        "weight_decay": 0.0
    }
}
```

NorMuon applies `SingleDeviceNorMuonWithAuxAdam`: NorMuon to 2D+ parameters (weight matrices) and Adam to 1D parameters (biases, norms). Requires: `pip install git+https://github.com/zichongli5/NorMuon.git`

### Parameter Groups

You can specify different optimizer settings for different parameter groups (e.g., no weight decay on biases and layer norms):

```json
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
```

```python
from optimizer_builder import build_optimizer, get_optimizer_param_groups

# Create parameter groups
param_groups = get_optimizer_param_groups(model, config)

# Build optimizer with custom groups
optimizer = build_optimizer(config, param_groups)
```

## Scheduler Builder

### Supported Schedulers

1. **Constant** - Constant learning rate (optionally with warmup)
2. **Linear** - Linear warmup followed by linear decay
3. **Cosine** - Linear warmup followed by cosine annealing
4. **wWSD** - Warmup-Warmup-Stable-Decay (4-phase scheduler)
5. **Epoch Shock Absorber** - Wrapper that applies LR reduction at epoch boundaries

### Usage

```python
from scheduler_builder import build_scheduler

# Configure scheduler in JSON
config = {
    "scheduler": {
        "type": "cosine",
        "warmup_steps": 100,
        "num_cycles": 0.5,
        "min_lr_ratio": 0.0
    }
}

# Build scheduler
scheduler = build_scheduler(
    config=config,
    optimizer=optimizer,
    num_training_steps=1000
)

# Training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate
    optimizer.zero_grad()
```

### Configuration Options

#### Constant Scheduler
```json
{
    "scheduler": {
        "type": "constant",
        "warmup_steps": 100
    }
}
```

#### Linear Scheduler
```json
{
    "scheduler": {
        "type": "linear",
        "warmup_steps": 100,
        "min_lr_ratio": 0.1
    }
}
```

#### Cosine Scheduler
```json
{
    "scheduler": {
        "type": "cosine",
        "warmup_steps": 100,
        "num_cycles": 0.5,
        "min_lr_ratio": 0.0
    }
}
```

#### wWSD Scheduler

The wWSD (warmup-Warmup-Stable-Decay) scheduler has four phases:

1. **Prewarmup**: Linear ramp from 0 to `prewarmup_lr_ratio * LR` (e.g., 0 → 0.1 * LR)
2. **Warmup**: Linear ramp from `prewarmup_lr_ratio` to full LR (e.g., 0.1 * LR → LR)
3. **Stable**: Constant at full LR
4. **Decay**: Cosine decay back to `prewarmup_lr_ratio * LR` (symmetric)

```json
{
    "scheduler": {
        "type": "wwsd",
        "prewarmup_steps": 50,
        "prewarmup_lr_ratio": 0.1,
        "warmup_steps": 100,
        "stable_steps": 200,
        "num_cycles": 1
    }
}
```

**Timeline Example** (1000 total steps):
- Steps 0-49: Prewarmup (0 → 0.1x LR)
- Steps 50-149: Warmup (0.1x → 1.0x LR)
- Steps 150-349: Stable (1.0x LR)
- Steps 350-999: Decay (1.0x → 0.1x LR, cosine)

### Epoch Shock Absorber

The epoch shock absorber applies a learning rate reduction at epoch boundaries to stabilize training when the data distribution shifts:

```json
{
    "scheduler": {
        "type": "cosine",
        "warmup_steps": 100,
        "epoch_shock_absorber": {
            "enabled": true,
            "steps_per_epoch": 200,
            "shock_recovery_steps": 50
        }
    }
}
```

**Behavior:**
- At the first step of each new epoch: LR multiplied by 0.5
- Over next `shock_recovery_steps`: LR linearly recovers from 0.5x to 1.0x
- Remainder of epoch: Normal base scheduler behavior

## Utility Functions

### Print Optimizer Summary
```python
from optimizer_builder import print_optimizer_summary

print_optimizer_summary(optimizer)
```

Output:
```
Optimizer: AdamW
  Parameter groups: 2

  Group 0:
    Parameters: 3 tensors (45 elements)
    Learning rate: 2e-05
    Weight decay: 0.0
    ...

  Group 1:
    Parameters: 3 tensors (320 elements)
    Learning rate: 2e-05
    Weight decay: 0.01
    ...
```

### Print Scheduler Summary
```python
from scheduler_builder import print_scheduler_summary

print_scheduler_summary(scheduler, num_training_steps=1000)
```

Output:
```
Scheduler: LambdaLR
  Initial LR: 0.0
  Current LR: 2e-05
  Total steps: 1000
```

### Preview Learning Rate Schedule
```python
from scheduler_builder import get_scheduler_lr_at_step

# Preview LR at specific steps
for step in [0, 100, 500, 1000]:
    lr = get_scheduler_lr_at_step(scheduler, step, base_lr=2e-5)
    print(f"Step {step}: LR = {lr:.2e}")
```

## Complete Example

```python
import torch
import torch.nn as nn
from optimizer_builder import build_optimizer, print_optimizer_summary
from scheduler_builder import build_scheduler, print_scheduler_summary

# Define model
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Configuration
config = {
    "optimizer": {
        "type": "adamw",
        "learning_rate": 2e-5,
        "weight_decay": 0.01
    },
    "scheduler": {
        "type": "wwsd",
        "prewarmup_steps": 50,
        "prewarmup_lr_ratio": 0.1,
        "warmup_steps": 100,
        "stable_steps": 200,
        "num_cycles": 1
    }
}

# Build optimizer and scheduler
optimizer = build_optimizer(config, model.parameters())
scheduler = build_scheduler(config, optimizer, num_training_steps=1000)

# Print summaries
print_optimizer_summary(optimizer)
print_scheduler_summary(scheduler, num_training_steps=1000)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Optimizer and scheduler steps
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step {step}: LR = {current_lr:.2e}, Loss = {loss.item():.4f}")
```

## Testing

Run the comprehensive test suite:

```bash
python test_optimizer_scheduler_builders.py
```

The test suite validates:
- All optimizer types
- All scheduler types
- Parameter group handling
- Configuration validation
- Step-by-step scheduler behavior
- Epoch shock absorber mechanics
- Full integration with example configs

## Integration with Training Config

The builders are designed to work seamlessly with Cauldron training configs:

```json
{
    "name": "granite-hybrid-experiment",
    
    "model": {
        "name": "ibm-granite/granite-4.0-h-1b"
    },
    
    "optimizer": {
        "type": "adamw",
        "learning_rate": 2e-5,
        "weight_decay": 0.01
    },
    
    "scheduler": {
        "type": "wwsd",
        "warmup_steps": 100,
        "prewarmup_steps": 50,
        "prewarmup_lr_ratio": 0.1,
        "stable_steps": 200,
        "num_cycles": 1
    }
}
```

```python
from config_utils import load_training_config
from optimizer_builder import build_optimizer
from scheduler_builder import build_scheduler

# Load config
config = load_training_config("granite-hybrid-experiment.json")

# Build optimizer and scheduler
optimizer = build_optimizer(config, model.parameters())
scheduler = build_scheduler(config, optimizer, num_training_steps=1000)
```

## Error Handling

The builders provide clear error messages for invalid configurations:

```python
# Missing required field
config = {"optimizer": {"type": "adamw"}}  # Missing learning_rate
build_optimizer(config, model.parameters())
# ConfigError: Optimizer config missing required field: 'learning_rate'

# Invalid optimizer type
config = {"optimizer": {"type": "sgd", "learning_rate": 0.01}}
build_optimizer(config, model.parameters())
# ConfigError: Unknown optimizer type: sgd. Supported types: adamw, muon, normuon

# Invalid wWSD config
config = {"scheduler": {"type": "wwsd"}}  # Missing required fields
build_scheduler(config, optimizer, 1000)
# ConfigError: Required config value missing: warmup_steps
```

## Dependencies

- **Required**: PyTorch (`torch`)
- **For Muon**: `pip install muon-optimizer`
- **For NorMuon**: `pip install git+https://github.com/zichongli5/NorMuon.git`

## License

See main Cauldron LICENSE file.
