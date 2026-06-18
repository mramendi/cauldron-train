#!/usr/bin/env python3
"""
Tokenizer utilities for chat-formatted training data.

Supports chat templates with {% generation %} tags for proper assistant token masking.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys


def validate_chat_template(tokenizer, chat_template: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate that the chat template contains {% generation %} markers.

    Args:
        tokenizer: HuggingFace tokenizer
        chat_template: Optional explicit chat template string. If None, uses tokenizer's template

    Returns:
        Tuple of (is_valid, template_to_use, error_message)
        - is_valid: True if template contains {% generation %} tags
        - template_to_use: The validated template string
        - error_message: Error description if not valid, None otherwise
    """
    template = chat_template if chat_template is not None else getattr(tokenizer, 'chat_template', None)

    if template is None:
        return False, None, "No chat template found in tokenizer and none provided"

    if '{% generation %}' not in template:
        return False, template, (
            "Chat template does not contain {% generation %} markers.\n"
            "These markers are required to identify which tokens should be unmasked during training.\n"
            "The template must use {% generation %}...{% endgeneration %} around assistant responses."
        )

    if '{% endgeneration %}' not in template:
        return False, template, (
            "Chat template contains {% generation %} but not {% endgeneration %}.\n"
            "Both tags are required to properly delimit assistant responses."
        )

    return True, template, None


def check_default_system_prompt(template: str) -> Optional[str]:
    """
    Check if the chat template contains a default system prompt.

    Args:
        template: Chat template string

    Returns:
        Warning message if default system prompt detected, None otherwise
    """
    # Common patterns for default system prompts in templates
    default_prompt_indicators = [
        'default_system_message',
        'g4_default_system_message',
        'You are a helpful assistant',
    ]

    for indicator in default_prompt_indicators:
        if indicator in template:
            return (
                f"WARNING: Chat template appears to contain a default system prompt (found: '{indicator}').\n"
                f"This may override or conflict with system messages in your training data.\n"
                f"Verify that the template behaves as expected with your data."
            )

    return None


def tokenize_conversation(
    tokenizer,
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    max_length: int = 32768
) -> Tuple[List[int], List[int]]:
    """
    Tokenize a single conversation using apply_chat_template.

    The tokenizer must have a chat template with {% generation %} tags already loaded.
    Use validate_chat_template() to verify before calling this function.

    Args:
        tokenizer: HuggingFace tokenizer with validated chat template
        messages: List of message dicts with 'role' and 'content' keys
        system_prompt: Optional system prompt (IGNORED - template handles system messages)
        max_length: Maximum sequence length (default 32768)

    Returns:
        Tuple of (input_ids, labels) where labels has -100 for non-assistant tokens

    Raises:
        ValueError: If chat template is not valid (missing {% generation %} tags)
    """
    # Verify template is valid
    is_valid, _, error_msg = validate_chat_template(tokenizer)
    if not is_valid:
        raise ValueError(f"Invalid chat template: {error_msg}")

    # Use apply_chat_template with assistant mask
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_length
        )
    except Exception as e:
        raise ValueError(
            f"Failed to apply chat template. This may indicate the template is incompatible "
            f"with the tokenizer or the messages are malformed.\nOriginal error: {e}"
        )

    input_ids = result['input_ids']
    masks = result['assistant_masks']

    # Convert masks to labels: token_id for assistant (mask=1), -100 for others (mask=0)
    labels = [input_ids[i] if masks[i] == 1 else -100 for i in range(len(input_ids))]

    return input_ids, labels


def tokenize_conversations(
    tokenizer,
    examples: Dict[str, List],
    system_prompt: Optional[str] = None,
    max_length: int = 32768
) -> Dict[str, List]:
    """
    Tokenize a batch of conversations using apply_chat_template (for use with datasets.map()).

    The tokenizer must have a chat template with {% generation %} tags already loaded.
    Use validate_chat_template() to verify before calling this function.

    Args:
        tokenizer: HuggingFace tokenizer with validated chat template
        examples: Dict with 'messages' key containing list of conversations
        system_prompt: Optional system prompt (IGNORED - template handles system messages)
        max_length: Maximum sequence length

    Returns:
        Dict with 'input_ids' and 'labels' keys, plus 'row_id' if present

    Raises:
        ValueError: If chat template is not valid (missing {% generation %} tags)
    """
    # Verify template is valid
    is_valid, _, error_msg = validate_chat_template(tokenizer)
    if not is_valid:
        raise ValueError(f"Invalid chat template: {error_msg}")

    all_input_ids = []
    all_labels = []

    for conversation in examples["messages"]:
        # Use apply_chat_template with assistant mask
        try:
            result = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_assistant_tokens_mask=True,
                truncation=True,
                max_length=max_length
            )
        except Exception as e:
            raise ValueError(
                f"Failed to apply chat template to conversation. "
                f"Original error: {e}"
            )

        input_ids = result['input_ids']
        masks = result['assistant_masks']

        # Convert masks to labels
        labels = [input_ids[i] if masks[i] == 1 else -100 for i in range(len(input_ids))]

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    # Return result with optional row_id passthrough
    result = {"input_ids": all_input_ids, "labels": all_labels}
    if "row_id" in examples:
        result["row_id"] = examples["row_id"]

    return result


def count_tokens_in_conversation(
    tokenizer,
    messages: List[Dict[str, str]],
    exclude_system: bool = True
) -> Tuple[int, int]:
    """
    Count total and unmasked tokens in a conversation.

    Args:
        tokenizer: HuggingFace tokenizer with validated chat template
        messages: List of message dicts
        exclude_system: If True, exclude system messages from count (IGNORED - template always adds system)

    Returns:
        Tuple of (total_tokens, unmasked_tokens)
    """
    # Use tokenize_conversation to get accurate counts
    input_ids, labels = tokenize_conversation(tokenizer, messages)

    total_tokens = len(input_ids)
    unmasked_tokens = sum(1 for label in labels if label != -100)

    return total_tokens, unmasked_tokens
