# Changes from Development Version

This document summarizes the changes made from `step1_prepare_batches_chat_v3.py` to the production-ready release version.

## File Renames

| Development Version | Release Version | Purpose |
|---------------------|-----------------|---------|
| `step1_prepare_batches_chat_v3.py` | `prepare_training_batches.py` | Clear, descriptive name without version number |
| `prepare_eval_batches.py` | `prepare_eval_batches.py` | Kept same name (already clear) |
| `prepare_eval_batches_split_flexible.py` | `prepare_eval_batches_split.py` | Removed "flexible" suffix (it's the standard now) |
| `tokenizer_utils.py` | `tokenizer_utils.py` | Kept same name (no version suffix) |

## Major Changes

### 1. Model Name Now Mandatory

**Before:**
```python
parser.add_argument("--model-name", type=str, default=None, ...)
# Had default fallback to granite or gemma based on --gemma flag
```

**After:**
```python
parser.add_argument("--model-name", type=str, required=True, ...)
# No defaults - user must explicitly specify model
```

**Rationale:** Explicit is better than implicit for production use. Users should consciously choose their model.

### 2. Chat Template Validation

**New Features:**
- `validate_chat_template()` function checks for required `{% generation %}` markers
- Script exits with clear error message if template is invalid
- Warning (non-fatal) if template contains default system prompt
- Support for custom chat templates via `--chat-template` argument

**Example Error Messages:**
```
ERROR: Chat template does not contain {% generation %} markers.
These markers are required to identify which tokens should be unmasked during training.
The template must use {% generation %}...{% endgeneration %} around assistant responses.
```

```
WARNING: Chat template appears to contain a default system prompt (found: 'g4_default_system_message').
This may override or conflict with system messages in your training data.
Verify that the template behaves as expected with your data.
```

### 3. Removed Model-Specific Flags

**Removed:**
```python
parser.add_argument("--gemma", action="store_true", ...)
```

**Replaced with:**
```python
parser.add_argument("--chat-template", type=str, default=None, ...)
```

**Rationale:** More flexible - supports any model and template combination. Users specify the model name and optionally provide a custom template.

### 4. Enhanced Output Information

**Added to metadata.json:**
```json
{
  "model_name": "ibm-granite/granite-4.0-h-1b",
  "max_sequence_length_tokens": 32768,
  "custom_chat_template": null,
  ...
}
```

**Console output now includes:**
```
Maximum sequence length: 32768 tokens
Sequence length (tokens): min=..., mean=..., max=...
```

**Rationale:** Makes it immediately clear what the maximum sequence length is, both in output and in saved metadata.

### 5. Improved Error Handling in tokenizer_utils.py

**New validation functions:**
```python
def validate_chat_template(tokenizer, chat_template=None):
    """Validate template contains {% generation %} markers."""
    ...

def check_default_system_prompt(template):
    """Check if template contains default system prompt."""
    ...
```

**Better error messages:**
- Specific guidance on what's wrong
- Actionable suggestions for fixing issues
- Distinguishes between fatal errors and warnings

### 6. Enhanced Documentation

**Before:** Basic docstring at top of file

**After:**
- Comprehensive README.md with usage examples
- Detailed API documentation
- Error handling guide
- Example configurations
- Test suite (test_template_validation.py)

## Backward Compatibility

### Breaking Changes

1. **`--model-name` is now required** - scripts without this argument will fail
2. **`--gemma` flag removed** - use `--model-name google/gemma-3-1b-it --chat-template path/to/template.jinja` instead
3. **Chat template must contain `{% generation %}` tags** - templates without these will be rejected

### Migration Guide

#### Training Batches

**Old command:**
```bash
python step1_prepare_batches_chat_v3.py \
    --config datasets_config.json \
    --output-dir ./prepared_data \
    --on-device-batch-size 3 \
    --grad-accum-steps 10
```

**New command:**
```bash
python prepare_training_batches.py \
    --config datasets_config.json \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./prepared_data \
    --on-device-batch-size 3 \
    --grad-accum-steps 10
```

**Old command with --gemma:**
```bash
python step1_prepare_batches_chat_v3.py \
    --config datasets_config.json \
    --output-dir ./prepared_data \
    --gemma
```

**New command:**
```bash
python prepare_training_batches.py \
    --config datasets_config.json \
    --model-name google/gemma-3-1b-it \
    --chat-template ../gemma3_chat_template.jinja \
    --output-dir ./prepared_data
```

#### Evaluation Batches

**Old command:**
```bash
python prepare_eval_batches.py \
    --eval-file eval.jsonl \
    --output-dir ./prepared_data/eval \
    --on-device-batch-size 3
```

**New command:**
```bash
python prepare_eval_batches.py \
    --eval-file eval.jsonl \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-dir ./prepared_data/eval \
    --on-device-batch-size 3
```

**Old command with --gemma:**
```bash
python prepare_eval_batches.py \
    --eval-file eval.jsonl \
    --output-dir ./prepared_data/eval \
    --gemma
```

**New command:**
```bash
python prepare_eval_batches.py \
    --eval-file eval.jsonl \
    --model-name google/gemma-3-1b-it \
    --chat-template ../gemma3_chat_template.jinja \
    --output-dir ./prepared_data/eval
```

#### Split Evaluation Batches

**Old command:**
```bash
python prepare_eval_batches_split_flexible.py \
    --output-base ./prepared_data/eval_split \
    --category ifeval:./ifeval.jsonl:8 \
    --category general:./general.jsonl:6 \
    --best-category ifeval
```

**New command:**
```bash
python prepare_eval_batches_split.py \
    --model-name ibm-granite/granite-4.0-h-1b \
    --output-base ./prepared_data/eval_split \
    --category ifeval:./ifeval.jsonl:8 \
    --category general:./general.jsonl:6 \
    --best-category ifeval
```

**Old command with --gemma:**
```bash
python prepare_eval_batches_split_flexible.py \
    --output-base ./prepared_data/eval_split \
    --category ifeval:./ifeval.jsonl:8 \
    --gemma \
    --best-category ifeval
```

**New command:**
```bash
python prepare_eval_batches_split.py \
    --model-name google/gemma-3-1b-it \
    --chat-template ../gemma3_chat_template.jinja \
    --output-base ./prepared_data/eval_split \
    --category ifeval:./ifeval.jsonl:8 \
    --best-category ifeval
```

## Code Quality Improvements

1. **Better error messages**: All errors now include context and suggestions
2. **Input validation**: Template validation happens before processing any data
3. **Clearer output**: Console messages use clear headers and formatting
4. **Type hints preserved**: All type annotations maintained
5. **Comprehensive testing**: Test suite added for validation logic

## Files Added

- `release/prepare_training_batches.py` - Training batch preparation script
- `release/prepare_eval_batches.py` - Single evaluation batch preparation script
- `release/prepare_eval_batches_split.py` - Split evaluation batch preparation script
- `release/tokenizer_utils.py` - Utility functions
- `release/README.md` - Comprehensive documentation
- `release/QUICKSTART.md` - Quick start guide
- `release/INDEX.md` - Directory overview and navigation
- `release/CHANGES.md` - This file
- `release/example_config.json` - Example configuration file
- `release/test_template_validation.py` - Test suite

## No Changes To

1. **Core algorithm**: Batching, bucketing, and upsampling logic unchanged
2. **Output format**: Pickle files have identical structure
3. **Dataset format**: JSONL input format unchanged
4. **Tokenization**: Uses same `apply_chat_template` approach
