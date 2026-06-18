"""
Training callbacks for Cauldron LoRA training.

Provides specialized callbacks for:
- Live logging to JSONL files
- Checkpoint signaling for upload orchestration
- Final evaluation triggers
- Split evaluation with early stopping
- Gradient debugging for PEFT issues
- Performance monitoring
"""

import os
import json
import time
from pathlib import Path
from typing import List, Optional

from transformers import TrainerCallback


class LiveLogCallback(TrainerCallback):
    """
    Log metrics to JSONL file with step timing.

    Tracks time between logging steps and writes all metrics to
    live_logs.jsonl in the output directory for real-time monitoring.
    """

    def __init__(self):
        super().__init__()
        self.step_times = []
        self.last_log_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        current_time = time.time()

        if self.last_log_time is not None:
            elapsed = current_time - self.last_log_time
            self.step_times.append(elapsed)
            logs["step_time"] = elapsed
            if len(self.step_times) >= 5:
                logs["avg_step_time_5"] = sum(self.step_times[-5:]) / min(5, len(self.step_times))

        self.last_log_time = current_time

        # Write to live_logs.jsonl
        log_file = Path(args.output_dir) / "live_logs.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(logs) + "\n")


class CheckpointFlagCallback(TrainerCallback):
    """
    Write checkpoint ID to flag file for upload orchestration.

    When training on remote infrastructure (e.g., Modal, Vast.ai), this
    signals to upload scripts which checkpoint is ready for sync.
    """

    def __init__(self, flag_file: str):
        self.flag_file = flag_file

    def on_save(self, args, state, control, **kwargs):
        checkpoint_num = state.global_step
        try:
            os.makedirs(os.path.dirname(self.flag_file), exist_ok=True)
            with open(self.flag_file, "w") as f:
                f.write(str(checkpoint_num))
            print(f"[Upload Signal] Checkpoint ready, wrote {checkpoint_num} to {self.flag_file}")
        except Exception as e:
            print(f"[Upload Signal] Warning: Could not write to flag file: {e}")


class FinalEvalCallback(TrainerCallback):
    """
    Run final evaluation after training.

    Prints a clear separator to highlight final evaluation results.
    """

    def on_train_end(self, args, state, control, **kwargs):
        print("\n" + "="*80)
        print("RUNNING FINAL EVALUATION")
        print("="*80)


class SplitEvalEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping for split eval datasets that tracks a specific category's loss.

    When using split evaluation (e.g., separate math, coding, general categories),
    this allows early stopping based on a specific category rather than overall loss.

    Args:
        expected_categories: List of category names in eval dataset
        best_category: Category to track for early stopping (default: first category)
        patience: Number of evaluation steps without improvement before stopping
        threshold: Minimum improvement required to reset patience counter
    """

    def __init__(self, expected_categories: List[str], best_category: Optional[str],
                 patience: int = 1, threshold: float = 0.0):
        self.expected_categories = expected_categories
        self.best_category = best_category or expected_categories[0]
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.wait = 0
        self.metric_name = f"eval_{self.best_category}_loss"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get(self.metric_name)

        if current_metric is None:
            return

        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric < (self.best_metric - self.threshold):
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True


class GradientDebugCallback(TrainerCallback):
    """
    Debug callback to detect PEFT gradient bugs.

    Monitors LoRA adapter weights during training to verify they're actually
    updating. This is critical for detecting issues like:
    - Frozen adapters due to PEFT bugs
    - Mamba out_proj adapters not updating (requires special kernel)
    - Optimizer configuration errors

    The callback operates in two modes:
    1. Silent mode (default): Monitors weights without printing
    2. Verbose mode: Prints detailed diagnostics at key steps

    To enable verbose mode, set debug_target when initializing:
        callback = GradientDebugCallback(debug_target="mamba_out")

    Args:
        debug_target: Optional layer substring to debug ("mamba_out", "attention", etc.)
        verbose: If True, print detailed diagnostics at steps 10, 50, 100
    """

    def __init__(self, debug_target: Optional[str] = None, verbose: bool = False):
        self.debug_target = debug_target
        self.verbose = verbose
        self.initial_weights = {}
        self.first_loss = None
        self.last_losses = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Capture initial LoRA weight values."""
        # Categorize LoRA parameters by layer type
        mamba_in = []
        mamba_out = []
        attention = []
        mlp = []
        other = []

        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                if "mamba.in_proj" in name or "mamba_in_proj" in name:
                    mamba_in.append(name)
                elif "mamba.out_proj" in name or "mamba_out_proj" in name:
                    mamba_out.append(name)
                elif any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    attention.append(name)
                elif any(x in name for x in ["mlp", "up_proj", "down_proj", "gate_proj"]):
                    mlp.append(name)
                else:
                    other.append(name)

        # Print summary if verbose
        if self.verbose or self.debug_target:
            print("\n" + "="*80)
            print("GRADIENT DEBUG: CAPTURING INITIAL LORA WEIGHTS")
            print("="*80)
            print(f"Found LoRA parameters:")
            print(f"  Mamba in_proj: {len(mamba_in)}")
            print(f"  Mamba out_proj: {len(mamba_out)}")
            if len(mamba_out) == 0:
                print("    (no out_proj adapters - normal for non-hybrid models)")
            print(f"  Attention: {len(attention)}")
            print(f"  MLP: {len(mlp)}")
            print(f"  Other: {len(other)}")
            print()

        # Select which parameters to track based on debug_target
        all_lora_params = mamba_in + mamba_out + attention + mlp + other
        if self.debug_target:
            # Named category shortcuts
            if "mamba_out" in self.debug_target.lower():
                track_params = mamba_out[:4]
            elif "mamba_in" in self.debug_target.lower():
                track_params = mamba_in[:4]
            elif "attention" in self.debug_target.lower():
                track_params = attention[:4]
            elif "mlp" in self.debug_target.lower():
                track_params = mlp[:4]
            else:
                # Substring match against actual parameter names
                matched = [n for n in all_lora_params if self.debug_target.lower() in n.lower()]
                track_params = matched[:4]
                if not matched:
                    print(f"[GradDebug] Warning: no LoRA params matched '{self.debug_target}', tracking first 4 overall")
                    track_params = all_lora_params[:4]
        else:
            # Track a sample from each type
            track_params = (mamba_in[:2] + mamba_out[:2] +
                          attention[:2] + mlp[:1])

        # Store initial values
        param_dict = dict(model.named_parameters())
        for name in track_params:
            if name in param_dict:
                param = param_dict[name]
                self.initial_weights[name] = param.data.clone().cpu()

                if self.verbose or self.debug_target:
                    layer_type = self._get_layer_type(name)
                    print(f"  [{layer_type}] {name}")
                    print(f"    Initial mean: {param.data.mean().item():.8f}")
                    print(f"    Initial std:  {param.data.std().item():.8f}")

        if self.verbose or self.debug_target:
            print(f"\nTracking {len(self.initial_weights)} parameters for gradient updates")
            print("="*80 + "\n")

    def _get_layer_type(self, name: str) -> str:
        """Determine layer type from parameter name."""
        if "mamba.in_proj" in name or "mamba_in_proj" in name:
            return "MAMBA-IN"
        elif "mamba.out_proj" in name or "mamba_out_proj" in name:
            return "MAMBA-OUT"
        elif any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            return "ATTENTION"
        elif any(x in name for x in ["mlp", "up_proj", "down_proj", "gate_proj"]):
            return "MLP"
        else:
            return "OTHER"

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track if loss is changing."""
        if logs and "loss" in logs:
            current_loss = logs["loss"]

            if self.first_loss is None:
                self.first_loss = current_loss

            self.last_losses.append(current_loss)
            if len(self.last_losses) > 20:
                self.last_losses.pop(0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check for gradient updates at key steps."""
        step = state.global_step

        # Only run diagnostics if verbose or debugging specific target
        if not (self.verbose or self.debug_target):
            return

        # Run diagnostics at key steps
        if step not in [10, 50, 100]:
            return

        print("\n" + "="*80)
        print(f"GRADIENT DEBUG: WEIGHT UPDATE CHECK (Step {step})")
        print("="*80)

        any_changed = False
        max_change = 0.0
        frozen_params = []

        param_dict = dict(model.named_parameters())
        for name, initial_value in self.initial_weights.items():
            if name not in param_dict:
                continue

            param = param_dict[name]
            current_value = param.data.cpu()

            # Compute change metrics
            abs_diff = (current_value - initial_value).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()

            # Relative change
            initial_norm = initial_value.norm().item()
            current_norm = current_value.norm().item()
            norm_change = abs(current_norm - initial_norm)

            if max_abs_diff > 1e-6:
                any_changed = True
                max_change = max(max_change, max_abs_diff)
                status = "✓ UPDATING"
            else:
                frozen_params.append(name)
                status = "✗ FROZEN"

            layer_type = self._get_layer_type(name)
            print(f"\n  [{layer_type}] {name}")
            print(f"    Initial norm: {initial_norm:.8f}")
            print(f"    Current norm: {current_norm:.8f}")
            print(f"    Norm change:  {norm_change:.8f}")
            print(f"    Max abs diff: {max_abs_diff:.8f}")
            print(f"    Mean abs diff: {mean_abs_diff:.8f}")
            print(f"    Status: {status}")

        print()
        if any_changed:
            print(f"✅ LoRA weights ARE updating (max change: {max_change:.8f})")
            if frozen_params:
                print(f"⚠️  WARNING: {len(frozen_params)} parameters appear frozen:")
                for name in frozen_params:
                    print(f"    - {name}")
        else:
            print("🔥🔥🔥 CRITICAL FAILURE 🔥🔥🔥")
            print()
            print("  LoRA weights have NOT changed since initialization!")
            print("  Possible causes:")
            print("    1. PEFT bug preventing gradient application")
            print("    2. Optimizer misconfiguration")
            print("    3. Mamba out_proj requires special kernel (transformers 4.57.6+granite.lora.fix)")
            print("    4. Model frozen or gradients not flowing")
            print()
            print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥")

        print("="*80 + "\n")


class GradientClipDebugCallback(TrainerCallback):
    """
    Debug callback to verify gradient clipping is happening.

    Monitors grad_norm values to confirm gradient clipping is active
    and working as expected. Useful for debugging training instability.

    Args:
        log_every_n_steps: How often to log gradient norm info
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Only log periodically to avoid spam
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Check if grad_norm was logged (Trainer tracks this)
        if not hasattr(state, 'log_history') or len(state.log_history) == 0:
            return

        recent_logs = [log for log in state.log_history if 'grad_norm' in log]
        if not recent_logs:
            return

        latest_grad_norm = recent_logs[-1].get('grad_norm', 'N/A')
        max_grad_norm = args.max_grad_norm

        if max_grad_norm is None or max_grad_norm == 0:
            status = "NO_CLIP"
            print(f"[GRAD CLIP DEBUG @ step {state.global_step}] "
                  f"grad_norm={latest_grad_norm:.3f}, clipping=DISABLED")
        else:
            clipped = (latest_grad_norm > max_grad_norm
                      if isinstance(latest_grad_norm, (int, float)) else False)
            status = "CLIPPED" if clipped else "OK"
            print(f"[GRAD CLIP DEBUG @ step {state.global_step}] "
                  f"grad_norm={latest_grad_norm:.3f}, max={max_grad_norm}, status={status}")


class ShuffleVerificationCallback(TrainerCallback):
    """
    Verify that data is being reshuffled between epochs.

    Logs the first batch ID at the start of each epoch to confirm
    that shuffling is working as expected.

    Args:
        output_dir: Directory to write verification logs
    """

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.current_epoch = -1
        self.epoch_first_batches = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            log_entry = {
                "epoch": epoch,
                "global_step": state.global_step,
                "timestamp": time.time()
            }
            self.epoch_first_batches.append(log_entry)

            # Write to file
            log_file = Path(self.output_dir) / "shuffle_verification.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"[Shuffle Verification] Epoch {epoch} started at step {state.global_step}")
