"""Tokenizer helpers for text and message formatting.

The core model remains token-ID based. This module keeps tokenizer
loading, chat-template formatting, text encoding, and ID decoding at
the application boundary so the inference engine does not depend on a
specific tokenizer runtime.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

import mlx.core as mx


def load_tokenizer(name_or_path: str) -> Any:
    """Load a HuggingFace tokenizer by model ID or local path."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Tokenizer support requires the 'transformers' package. "
            "Install it with `python -m pip install transformers` or "
            "use --prompt-ids to provide raw token IDs."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
    return tokenizer


def tokenizer_supports_chat_templates(tokenizer: Any) -> bool:
    """Return whether the tokenizer exposes ``apply_chat_template``."""
    return callable(getattr(tokenizer, "apply_chat_template", None))


def get_tokenizer_capabilities(tokenizer: Any) -> dict[str, bool]:
    """Describe the small capability set used by the application layer."""
    return {
        "chat_template": tokenizer_supports_chat_templates(tokenizer),
        "decode": callable(getattr(tokenizer, "decode", None)),
    }


def _normalize_token_ids(token_ids: Any) -> List[int]:
    """Normalize tokenizer outputs into a flat list of integer IDs."""
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict) and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]

    if hasattr(token_ids, "ids"):
        token_ids = token_ids.ids

    if not isinstance(token_ids, (list, tuple)):
        raise TypeError("Tokenizer output must expose input_ids or ids")

    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise ValueError("Only batch_size=1 tokenizer output is supported")
        token_ids = token_ids[0]

    return [int(token_id) for token_id in token_ids]


def validate_token_ids(token_ids: Iterable[int], vocab_size: int) -> List[int]:
    """Validate token IDs against the configured model vocabulary size."""
    normalized = [int(token_id) for token_id in token_ids]
    if not normalized:
        raise ValueError("Prompt must encode to at least one token")

    invalid = [token_id for token_id in normalized if token_id < 0 or token_id >= vocab_size]
    if invalid:
        preview = ", ".join(str(token_id) for token_id in invalid[:8])
        raise ValueError(
            f"Tokenizer produced token IDs outside model vocab_size={vocab_size}: {preview}"
        )
    return normalized


def prompt_ids_from_list(token_ids: Iterable[int], vocab_size: int) -> mx.array:
    """Convert a token-id iterable into a batch-1 MLX int32 tensor."""
    normalized = validate_token_ids(token_ids, vocab_size=vocab_size)
    return mx.array([normalized], dtype=mx.int32)


def encode_prompt_text(
    tokenizer: Any,
    text: str,
    *,
    vocab_size: int,
    add_special_tokens: bool = True,
) -> mx.array:
    """Encode a text prompt into a batch-1 MLX int32 token tensor."""
    if hasattr(tokenizer, "__call__"):
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    elif hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    else:
        raise TypeError("Tokenizer must provide __call__() or encode()")

    return prompt_ids_from_list(_normalize_token_ids(encoded), vocab_size=vocab_size)


def apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render chat messages through the tokenizer's template, if available."""
    template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        raise ValueError("Tokenizer does not support chat templates")

    rendered = template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return str(rendered)


def encode_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    vocab_size: int,
    add_generation_prompt: bool = True,
) -> mx.array:
    """Encode structured messages using a tokenizer chat template."""
    template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        raise ValueError("Tokenizer does not support chat templates")

    try:
        encoded = template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        rendered = apply_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
        )
        return encode_prompt_text(tokenizer, rendered, vocab_size=vocab_size)

    return prompt_ids_from_list(_normalize_token_ids(encoded), vocab_size=vocab_size)


def decode_token_ids(
    tokenizer: Any,
    token_ids: Any,
    *,
    skip_special_tokens: bool = True,
) -> str:
    """Decode a batch-1 token tensor or token list back to text."""
    if isinstance(token_ids, mx.array):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be an MLX array or list of integers")

    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Only batch_size=1 decode is supported")
        token_ids = token_ids[0]

    normalized = [int(token_id) for token_id in token_ids]
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(normalized, skip_special_tokens=skip_special_tokens))
    raise TypeError("Tokenizer must provide decode()")


def decode_tokens(
    tokenizer: Any,
    token_ids: Any,
    *,
    skip_special_tokens: bool = True,
) -> str:
    """Alias for decode_token_ids with the V11 tokenizer-manager naming."""
    return decode_token_ids(
        tokenizer,
        token_ids,
        skip_special_tokens=skip_special_tokens,
    )


def materialize_generated_sequence(prompt_ids: mx.array, generated: Any) -> mx.array:
    """Normalize `model.generate()` outputs into a full batch-1 token tensor."""
    if isinstance(generated, list):
        if not generated:
            return prompt_ids
        return mx.concatenate(
            [prompt_ids] + [token[:, None] for token in generated],
            axis=1,
        )
    return generated