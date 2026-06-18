# Cauldron Training Data Preparation

Production-ready scripts for preparing pre-batched training data for chat-formatted supervised fine-tuning.

## Overview

This system tokenizes conversational training data, organizes it into effective batches with dynamic bucketing, and prepares it for efficient multi-epoch training. Because batch contents are locked in at prep time, padding is minimized once — not re-computed each training step. With gradient accumulation ≥ 2, effective padding rates can be rather low, compared to dynamic batching where every batch pads to its longest sequence.

Every epoch is also reshuffled, and if you include multiple datasets, they are well mixed together - thus, "cauldron"!

**NOT FULLY TESTED**. These scripts are *based on* a battle-tested setup, but changed into a "production" model-independent version by sheer vibe-coding and some things can well be off. SOme minimal testing was done but please do report any issues so I can fix them!

### Key Features

- **Minimal padding via pre-batching**: Sequences are bucketed by length once at prep time. Training sees near-zero wasted tokens, and throughput is maximized because the model always processes full, dense batches.
- **Chat template validation**: Ensures templates contain `{% generation %}` markers for proper token masking
- **Multiple datasets**: Support for multiple datasets with configurable upsampling ratios
- **Multi-epoch generation**: Generates multiple epochs with proper reshuffling
- **EOS verification**: Ensures all sequences end with proper EOS tokens
- **Proportional sampling**: Maintains dataset proportions across batches

## Files

### Data Preparation

- `prepare_training_batches.py` - Tokenize and pre-batch training data with dynamic bucketing
- `prepare_eval_batches.py` - Tokenize and pre-batch a single evaluation set
- `prepare_eval_batches_split.py` - Tokenize and pre-batch multiple evaluation categories
- `tokenizer_utils.py` - Tokenization utilities with chat template validation

### Training

- `train_lora.py` - LoRA fine-tuning script that loads pre-batched data and trains
- `optimizer_builder.py` - Factory for AdamW, Muon, and NorMuon optimizers
- `scheduler_builder.py` - Factory for constant, linear, cosine, and wWSD schedulers
- `callbacks.py` - Training callbacks (live logging, checkpoint signaling, split eval)
- `config_utils.py` - Configuration loading and validation utilities

## Package Requirements

- `torch` - PyTorch
- `transformers` - HuggingFace Transformers
- `peft` - Parameter-Efficient Fine-Tuning (required by `train_lora.py`)
- `numpy` - Array operations
- `tqdm` - Progress bars
- `wandb` (optional) - Experiment tracking; pass `--no-wandb` to skip

For the Muon optimizer:
```bash
pip install muon-optimizer
```

For the NorMuon optimizer:
```bash
pip install git+https://github.com/zichongli5/NorMuon.git
```

## Chat Template Requirements

The chat template must contain `{% generation %}` and `{% endgeneration %}` tags around assistant responses. These tags are used to identify which tokens should be unmasked during training (i.e., which tokens the model should learn to predict).

Example template structure:
```jinja
{% if message.role == 'assistant' %}
  {{ '<start_of_turn>model\n' }}{% generation %}{{ message.content }}{{ '<end_of_turn>\n' }}{% endgeneration %}
{% endif %}
```

## Usage

### Basic Usage

```bash
python prepare_training_batches.py \
    --config datasets_config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./prepared_data \
    --on-device-batch-size 3 \
    --grad-accum-steps 10 \
    --num-epochs 10 \
    --max-length 32768
```

### With Custom Chat Template

```bash
python prepare_training_batches.py \
    --config datasets_config.json \
    --model-name google/gemma-3-1b-it \
    --chat-template ../gemma3_chat_template.jinja \
    --output-dir ./prepared_data \
    --on-device-batch-size 3 \
    --grad-accum-steps 10 \
    --num-epochs 10 \
    --max-length 32768
```

## Arguments

### Required Arguments

- `--config PATH` - JSON configuration file specifying datasets and upsampling factors
- `--model-name NAME` - HuggingFace model name for tokenizer (e.g., `ibm-granite/granite-4.0-h-1b`)
- `--output-dir PATH` - Output directory for prepared batches

### Optional Arguments

- `--chat-template PATH` - Path to custom chat template (.jinja file). If not provided, uses the model's default template
- `--max-length TOKENS` - Maximum sequence length in tokens (default: 32768)
- `--on-device-batch-size N` - On-device batch size (default: 3)
- `--grad-accum-steps N` - Gradient accumulation steps (default: 10)
- `--num-epochs N` - Number of epochs to generate (default: 10)
- `--seed N` - Random seed for reproducibility (default: 42)

## Dataset Configuration Format

Create a JSON file with the following structure:

```json
{
  "datasets": [
    {
      "name": "dataset1",
      "path": "path/to/dataset1.jsonl",
      "upsample": 1
    },
    {
      "name": "dataset2",
      "path": "path/to/dataset2.jsonl",
      "upsample": 2
    }
  ]
}
```

Each dataset entry:
- `name` - Human-readable name for the dataset
- `path` - Path to JSONL file containing the dataset
- `upsample` - Integer upsampling factor (default: 1)

## Input Data Format

Each line in the JSONL file should contain a conversation in the following format:

```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ],
  "row_id": "optional_identifier"
}
```

The `row_id` field is optional and can be used to track individual examples.

## Output Structure

The script creates the following directory structure:

```
output_dir/
├── metadata.json
├── epoch_00/
│   ├── effective_batch_000000.pkl
│   ├── effective_batch_000001.pkl
│   └── ...
├── epoch_01/
│   └── ...
└── ...
```

### metadata.json

Contains configuration and statistics:
```json
{
  "model_name": "ibm-granite/granite-4.0-h-1b",
  "max_sequence_length_tokens": 32768,
  "num_epochs": 10,
  "effective_batch_size": 30,
  "on_device_batch_size": 3,
  "grad_accum_steps": 10,
  "custom_chat_template": null,
  "datasets": {
    "dataset1": {
      "original_examples": 1000,
      "upsample_factor": 1,
      "final_examples": 1000
    }
  },
  "epoch_stats": [...],
  "total_examples": 1000,
  "total_tokens": 500000,
  "seed": 42
}
```

### Batch Files (.pkl)

Each pickle file contains one effective batch with the following structure:
```python
{
  "on_device_batches": [
    {
      "input_ids": np.array,      # Shape: (on_device_batch_size, padded_length)
      "labels": np.array,          # Shape: (on_device_batch_size, padded_length)
      "sequence_lengths": np.array,# Shape: (on_device_batch_size,)
      "datasets": list,            # Dataset name for each example
      "row_ids": list,             # Row IDs for each example
      "padded_length": int,        # Bucket size
      "bucket": int                # Same as padded_length
    },
    ...
  ],
  "effective_batch_idx": int,
  "num_on_device_batches": int,
  "epoch": int
}
```

## Error Handling

The script will exit with an error message if:

1. **Missing `{% generation %}` tags**: The chat template must contain these markers
   ```
   ERROR: Chat template does not contain {% generation %} markers.
   These markers are required to identify which tokens should be unmasked during training.
   ```

2. **No chat template found**: The model has no default template and none was provided
   ```
   ERROR: Model's chat template is not compatible with this script:
   No chat template found in tokenizer and none provided
   ```

The script will warn (but not exit) if:

1. **Default system prompt detected**: The template may contain a default system prompt
   ```
   WARNING: Chat template appears to contain a default system prompt.
   This may override or conflict with system messages in your training data.
   ```

## Examples

### Example 1: Single Dataset

```bash
# Configuration file: config.json
{
  "datasets": [
    {"name": "conversations", "path": "data.jsonl", "upsample": 1}
  ]
}

# Run preparation
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches \
    --on-device-batch-size 4 \
    --grad-accum-steps 8 \
    --num-epochs 5
```

### Example 2: Multiple Datasets with Upsampling

```bash
# Configuration file: config.json
{
  "datasets": [
    {"name": "general", "path": "general.jsonl", "upsample": 1},
    {"name": "specialized", "path": "specialized.jsonl", "upsample": 3}
  ]
}

# Run preparation
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches \
    --on-device-batch-size 2 \
    --grad-accum-steps 16
```

### Example 3: Gemma with Custom Template

```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name google/gemma-3-1b-it \
    --chat-template ../gemma3_chat_template.jinja \
    --output-dir ./gemma_batches \
    --max-length 8192
```

## Preparing Evaluation Batches

Evaluation batches are simpler than training batches:
- No upsampling
- No multi-epoch generation (single pass)
- Deterministic ordering (no shuffling)
- No gradient accumulation

### Basic Usage

```bash
python prepare_eval_batches.py \
    --eval-file ./data/eval.jsonl \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./prepared_data/eval \
    --on-device-batch-size 3 \
    --max-length 32768
```

### With Custom Chat Template

```bash
python prepare_eval_batches.py \
    --eval-file ./data/eval.jsonl \
    --model-name google/gemma-3-1b-it \
    --chat-template ../gemma3_chat_template.jinja \
    --output-dir ./prepared_data/eval
```

### Output Structure

```
output_dir/
├── metadata.json
├── batch_000000.pkl
├── batch_000001.pkl
└── ...
```

The metadata includes:
- `max_sequence_length_tokens`: Maximum sequence length
- `total_eval_examples`: Number of evaluation examples
- `padding_efficiency`: Percentage of non-padding tokens
- `deterministic`: Always true (no shuffling)

## Preparing Split Evaluation Batches

For evaluation with multiple categories (e.g., different task types), use the split version:

### Basic Usage

```bash
python prepare_eval_batches_split.py \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-base ./prepared_data/eval_split \
    --category ifeval:./datasets/ifeval_eval.jsonl:8 \
    --category general:./datasets/general_eval.jsonl:6 \
    --category coding:./datasets/coding_eval.jsonl:4 \
    --best-category ifeval \
    --max-length 32768
```

### Category Specification Format

Each `--category` argument uses the format: `name:input_file:batch_size`

- `name`: Short identifier for this category
- `input_file`: Path to JSONL file
- `batch_size`: On-device batch size (can differ per category)

### Best Category (Required)

The `--best-category` argument specifies which category is used for early stopping during training. This must match one of your category names.

### Output Structure

```
output_base/
├── metadata.json              # Parent metadata (split=True, best_category=...)
├── ifeval/
│   ├── metadata.json         # Category metadata (best=True)
│   ├── batch_000000.pkl
│   └── ...
├── general/
│   ├── metadata.json         # Category metadata (best=False)
│   └── ...
└── coding/
    ├── metadata.json         # Category metadata (best=False)
    └── ...
```

The parent `metadata.json` contains:
```json
{
  "split": true,
  "categories": ["coding", "general", "ifeval"],
  "best_category": "ifeval"
}
```

Each category's `metadata.json` contains:
```json
{
  "category": "ifeval",
  "best": true,
  "num_batches": 50,
  "total_eval_examples": 400,
  ...
}
```

## Tokenizer Utilities API

The `tokenizer_utils.py` module can also be used independently:

```python
from tokenizer_utils import (
    tokenize_conversation,
    validate_chat_template,
    check_default_system_prompt
)

# Validate a chat template
is_valid, template, error_msg = validate_chat_template(tokenizer)
if not is_valid:
    print(f"Error: {error_msg}")

# Check for default system prompt
warning = check_default_system_prompt(template)
if warning:
    print(warning)

# Tokenize a conversation
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]
input_ids, labels = tokenize_conversation(tokenizer, messages, max_length=2048)
```

## Running Training

After preparing training and evaluation batches, use `train_lora.py` to train LoRA adapters. All parameters — model, LoRA rank, optimizer, scheduler, and data paths — come from a JSON config file.

```bash
python train_lora.py --config example_train_config.json
```

Example configs are provided for Granite (`granite-h-attention-only.json`, `granite-h-full-mlp.json`, `granite-h-shallow-attention.json`) and Qwen (`qwen_train_config.json`).

```bash
# Optional: append a postfix to the output directory name
python train_lora.py --config granite-h-attention-only.json --postfix run-1

# Disable wandb
python train_lora.py --config granite-h-attention-only.json --no-wandb

# Apply BF16 logits patch (saves VRAM on some models)
python train_lora.py --config granite-h-attention-only.json --bf16-patch
```

The training script reads `metadata.json` from the pre-batched data directory to verify that batch parameters (on-device batch size, gradient accumulation steps, number of epochs) match the config. Prepared batches are consumed in the order the prebatcher wrote them, preserving bucket structure and avoiding re-padding.

See `OPTIMIZER_SCHEDULER_BUILDERS.md` for full documentation of optimizer and scheduler options.

## Notes

- The effective batch size is calculated as `on_device_batch_size × grad_accum_steps`
- Bucket sizes are rounded up to multiples of 8 for efficiency
- Each epoch is shuffled independently using a deterministic seed
- The `max_sequence_length_tokens` value is clearly reported in the metadata and console output
- All sequences are verified to end with proper EOS tokens
