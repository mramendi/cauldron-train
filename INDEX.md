# Cauldron Training Data Preparation - Release Directory

Production-ready scripts for preparing pre-batched training and evaluation data for chat-formatted supervised fine-tuning.

## Quick Navigation

- **New User?** Start with [QUICKSTART.md](QUICKSTART.md)
- **Full Documentation?** See [README.md](README.md)
- **Optimizer / Scheduler options?** See [OPTIMIZER_SCHEDULER_BUILDERS.md](OPTIMIZER_SCHEDULER_BUILDERS.md)

## Files Overview

### Data Preparation Scripts

| File | Purpose |
|------|---------|
| `prepare_training_batches.py` | Tokenize and pre-batch training data with dynamic bucketing and multi-epoch support |
| `prepare_eval_batches.py` | Tokenize and pre-batch a single evaluation set (deterministic, no shuffling) |
| `prepare_eval_batches_split.py` | Tokenize and pre-batch multiple evaluation categories with early-stopping support |
| `tokenizer_utils.py` | Tokenization utilities with chat template validation; importable as a library |

### Training Scripts

| File | Purpose |
|------|---------|
| `train_lora.py` | LoRA fine-tuning script; loads pre-batched data and trains via HuggingFace Trainer |
| `optimizer_builder.py` | Factory for AdamW, Muon, and NorMuon optimizers |
| `scheduler_builder.py` | Factory for constant, linear, cosine, and wWSD schedulers |
| `callbacks.py` | Training callbacks: live logging, checkpoint signaling, split eval, gradient debug |
| `config_utils.py` | Configuration loading and validation utilities |

### Documentation

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | 5-minute getting started guide |
| `README.md` | Comprehensive documentation with API reference |
| `OPTIMIZER_SCHEDULER_BUILDERS.md` | Optimizer and scheduler configuration reference |
| `INDEX.md` | This file — directory overview |

### Configuration

| File | Purpose |
|------|---------|
| `example_config.json` | Example dataset config for the prebatcher |
| `example_train_config.json` | Example training config for `train_lora.py` |
| `granite-h-attention-only.json` | Granite hybrid: LoRA on attention layers only |
| `granite-h-full-mlp.json` | Granite hybrid: LoRA on attention + MLP layers |
| `granite-h-shallow-attention.json` | Granite hybrid: LoRA on shallow attention layers |
| `qwen_train_config.json` | Qwen training config |
| `granite/chat_template.jinja` | Ready-to-use template for Granite 4.0-h models (`<\|start_of_role\|>` / `<\|end_of_text\|>` format). Supports tool and document injection. The validation script warns about a default system message (`g4_default_system_message`) but that variable is defined and never applied — the warning is a false positive and can be ignored. Pass with `--chat-template granite/chat_template.jinja`. |
| `qwen/chat_template.jinja` | Ready-to-use template derived from Qwen 3.5 2B; expected to work for other Qwen 3.5 models and likely Qwen 3.6. ChatML format (`<\|im_start\|>` / `<\|im_end\|>`); supports thinking blocks, vision tokens, and tool calling. Pass with `--chat-template qwen/chat_template.jinja`. |

## Typical Workflow

### 1. Prepare Your Data

Create JSONL files with conversations:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

You can have several datasets; prebatching will ensure even distrubution of every dataset in the training process.

You can have diverse lengths; adaptive bucketing ensures minimal use of padding (provided your gradient accumulation is 2 or more).

### 2. Prepare Training Batches

```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches/train
```

### 3. Prepare Evaluation Batches

**Option A: Single evaluation set**
```bash
python prepare_eval_batches.py \
    --eval-file eval.jsonl \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches/eval
```

**Option B: Split evaluation by category**
```bash
python prepare_eval_batches_split.py \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-base ./batches/eval_split \
    --category ifeval:./ifeval.jsonl:8 \
    --category general:./general.jsonl:6 \
    --best-category ifeval
```

### 4. Run Training

```bash
python train_lora.py --config example_train_config.json
```

The training script reads batch parameters (on-device batch size, gradient accumulation, epochs) from the prepared data's `metadata.json` and validates them against your config. No re-tokenization or re-padding happens at training time.

## Key Features

### Both Scripts
- ✅ Reqyures chat template validation with `{% generation %}` markers. Capable AI harnesses such as Claude Code can easily mangle a chat template to add that.
- ✅ Mandatory model name (no implicit defaults)
- ✅ Clear error messages and warnings
- ✅ Maximum sequence length clearly reported
- ✅ Support for custom chat templates

### Training Script (`prepare_training_batches.py`)
- Multiple datasets with integer upsampling
- Multi-epoch generation with reshuffling
- Proportional sampling across datasets
- Gradient accumulation support
- EOS token verification

### Evaluation Script (`prepare_eval_batches.py`)
- Deterministic ordering (no shuffling)
- Single-pass processing
- Simpler configuration
- Memory-efficient batching

## Requirements

### Python Packages
- `torch` - PyTorch
- `transformers` - HuggingFace Transformers
- `peft` - Parameter-Efficient Fine-Tuning (required by `train_lora.py`)
- `numpy` - Array operations
- `tqdm` - Progress bars
- `wandb` (optional) - Experiment tracking

For the Muon optimizer: `pip install muon-optimizer`

For the NorMuon optimizer: `pip install git+https://github.com/zichongli5/NorMuon.git`

### Chat Template Requirements
Your chat template must contain `{% generation %}` and `{% endgeneration %}` markers around assistant responses:

```jinja
{% if message.role == 'assistant' %}
  {{ '<start>assistant\n' }}{% generation %}{{ message.content }}{{ '<end>\n' }}{% endgeneration %}
{% endif %}
```

## Common Use Cases

### Single Dataset Training
```bash
python prepare_training_batches.py \
    --config single_dataset_config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches
```

### Multiple Datasets with Upsampling
```bash
# config.json:
# {
#   "datasets": [
#     {"name": "general", "path": "general.jsonl", "upsample": 1},
#     {"name": "code", "path": "code.jsonl", "upsample": 3}
#   ]
# }

python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches
```

### Custom Chat Template (e.g., Gemma)
```bash
python prepare_training_batches.py \
    --config config.json \
    --model-name google/gemma-3-1b-it \
    --chat-template ../gemma3_chat_template.jinja \
    --output-dir ./batches
```

### Different Sequence Lengths
```bash
# Short sequences (8K tokens)
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches \
    --max-length 8192

# Long sequences (32K tokens)
python prepare_training_batches.py \
    --config config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./batches \
    --max-length 32768
```

## Troubleshooting

### Error: "Chat template does not contain {% generation %} markers"

**Solution:** Your template needs these markers. Either:
1. Modify your template to include them around assistant responses
2. Use a model with a compatible default template
3. Provide a custom template with `--chat-template`

### Error: "No chat template found in tokenizer"

**Solution:** Provide a custom template:
```bash
--chat-template path/to/template.jinja
```

### Warning: "Chat template appears to contain a default system prompt"

**This is just a warning.** Verify the template doesn't conflict with your data. The script continues running.

### Examples Don't Fit in Batches (Eval)

Adjust `--on-device-batch-size` to minimize waste:
```bash
# If you have 100 examples and batch size 3, you waste 1 example
# Try batch size 2, 4, 5, etc. to minimize waste
```

## File Sizes

Typical file sizes for prepared batches:
- Each effective batch pickle: ~100KB - 10MB (depends on sequence length and batch size)
- metadata.json: ~1-5KB
- Total training data: Scales with dataset size and number of epochs

## Getting Help

Run any script with `--help` for detailed usage:
```bash
python prepare_training_batches.py --help
python prepare_eval_batches.py --help
python train_lora.py --help
```
