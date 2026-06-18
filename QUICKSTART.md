# Quick Start Guide

Get started with Cauldron training data preparation in 5 minutes.

## 1. Prepare Your Data

Create a JSONL file where each line contains a conversation:

```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
```

Save this as `my_data.jsonl`.

## 2. Create a Configuration File

Create `config.json`:

```json
{
  "datasets": [
    {
      "name": "my_dataset",
      "path": "my_data.jsonl",
      "upsample": 1
    }
  ]
}
```

## 3. Choose Your Model and Template

### Option A: Use Model's Default Template (Granite)

```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches
```

**Before running:** Make sure the model's default chat template has `{% generation %}` tags. The script will tell you if it doesn't.

### Option B: Provide a Custom Template

If you have a custom template (e.g., `my_template.jinja`):

```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --chat-template my_template.jinja \
    --output-dir ./batches
```

## 4. Common Use Cases

### For Gemma Models

```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name google/gemma-3-1b-it \
    --chat-template ../gemma3_chat_template.jinja \
    --output-dir ./batches
```

### With Multiple Datasets and Upsampling

**config.json:**
```json
{
  "datasets": [
    {"name": "general", "path": "general.jsonl", "upsample": 1},
    {"name": "code", "path": "code.jsonl", "upsample": 3}
  ]
}
```

```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches
```

This will use 3× more code examples than general examples in training.

### With Custom Sequence Length

```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches \
    --max-length 8192
```

### With Custom Batch Configuration

```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches \
    --on-device-batch-size 4 \
    --grad-accum-steps 16
```

This creates an effective batch size of 4 × 16 = 64.

## 5. Understanding the Output

After running, you'll see:

```
batches/
├── metadata.json          # Configuration and statistics
├── epoch_00/
│   ├── effective_batch_000000.pkl
│   ├── effective_batch_000001.pkl
│   └── ...
├── epoch_01/
│   └── ...
└── ...
```

The `metadata.json` file contains important information:
- `max_sequence_length_tokens`: Maximum sequence length used
- `total_examples`: Total number of training examples
- `total_tokens`: Total number of tokens
- Dataset statistics and proportions

## Common Issues

### Issue: "Chat template does not contain {% generation %} markers"

**Solution:** Your chat template needs `{% generation %}` tags to mark which tokens should be learned. Either:
1. Modify your template to include these tags around assistant responses
2. Use a different model with a compatible template
3. Provide a custom template using `--chat-template`

### Issue: "No chat template found in tokenizer"

**Solution:** The model doesn't have a default chat template. Provide one:
```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name your-model-name \
    --chat-template path/to/template.jinja \
    --output-dir ./batches
```

### Issue: Warning about default system prompt

**This is just a warning, not an error.** The script detected that your template might add a default system prompt. Verify that this doesn't conflict with your training data. The script will continue running.

## Preparing Evaluation Data

### Single Evaluation Set

For a single evaluation set during training:

```bash
python prepare_eval_batches.py \
    --eval-file my_eval.jsonl \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches/eval \
    --on-device-batch-size 3
```

### Split Evaluation (Multiple Categories)

For evaluation with multiple categories (e.g., different task types):

```bash
python prepare_eval_batches_split.py \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-base ./batches/eval_split \
    --category ifeval:./ifeval.jsonl:8 \
    --category general:./general.jsonl:6 \
    --best-category ifeval
```

The `--best-category` specifies which category is used for early stopping.

Key differences from training batches:
- No upsampling
- No multi-epoch generation
- Deterministic ordering (no shuffling)
- No gradient accumulation

## Running Training

Once training and evaluation batches are prepared, launch training with `train_lora.py`:

```bash
python train_lora.py --config example_train_config.json
```

All training parameters (model name, LoRA rank, optimizer, scheduler, data paths) are read from the JSON config file. The script reads `metadata.json` from the prepared data directory and validates that batch parameters match.

```bash
# Append a postfix to distinguish runs
python train_lora.py --config example_train_config.json --postfix run-1

# Disable wandb
python train_lora.py --config example_train_config.json --no-wandb
```

Example config files for Granite and Qwen are included in the repo. See `OPTIMIZER_SCHEDULER_BUILDERS.md` for full optimizer and scheduler options.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- See [OPTIMIZER_SCHEDULER_BUILDERS.md](OPTIMIZER_SCHEDULER_BUILDERS.md) for optimizer and scheduler configuration

## Getting Help

Run with `--help` to see all options:
```bash
python prepare_training_batches.py --help
python train_lora.py --help
```
