"""Exact attention primitives.

This module provides helper routines for computing causal attention
using MLX's built in fast scaled dot product attention.  It exposes
functions for building causal masks as well as full prefill and
single–token decode attention.  These helpers are only used for the
exact path.  Compressed paths should reconstruct the full key/value
tensors and then call the same exact helper to ensure consistent
semantics.

The causal mask uses an additive mask with ``0.0`` for allowed
positions and ``-1e9`` for masked positions.  This aligns with
MLX's expectation that very negative values effectively zero the
resulting softmax.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import mlx.core as mx


AttentionSegment = Tuple[mx.array, mx.array, int]


def build_causal_mask(
    q_len: int,
    k_len: int,
    q_start_pos: int,
    k_start_pos: int,
) -> mx.array:
    """Construct an additive causal mask.

    A causal mask ensures that each query position can only attend
    to key positions at or before its own absolute position.

    Parameters
    ----------
    q_len: int
        Length of the query sequence.
    k_len: int
        Length of the key sequence.
    q_start_pos: int
        Absolute start position of the query sequence in the layer stream.
    k_start_pos: int
        Absolute start position of the key sequence in the layer stream.

    Returns
    -------
    mask: mx.array
        Mask of shape ``(1, 1, q_len, k_len)`` with dtype
        ``float32``.  Allowed positions are zero and masked positions
        are ``-1e9``.
    """
    # Compute absolute positions
    q_positions = mx.arange(q_start_pos, q_start_pos + q_len)[:, None]
    k_positions = mx.arange(k_start_pos, k_start_pos + k_len)[None, :]
    allowed = k_positions <= q_positions
    mask = mx.where(allowed, 0.0, -1e9).astype(mx.float32)
    return mask[None, None, :, :]


def run_exact_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_start_pos: int,
    k_start_pos: int,
) -> mx.array:
    """Perform causal attention on the provided key/value context.

    This helper constructs the appropriate causal mask and calls
    MLX's fused scaled dot product attention.  It can be used for
    both prefill and decode by choosing the appropriate shapes for
    ``q`` and ``k``.

    Parameters
    ----------
    q: mx.array
        Query tensor shaped ``(B, H, Q, D)``.
    k: mx.array
        Key tensor shaped ``(B, H, K, D)``.
    v: mx.array
        Value tensor shaped ``(B, H, K, D)``.
    q_start_pos: int
        Absolute start position of the query sequence.
    k_start_pos: int
        Absolute start position of the key sequence.

    Returns
    -------
    out: mx.array
        Attention output shaped ``(B, H, Q, D)``.
    """
    mask = build_causal_mask(
        q_len=q.shape[2],
        k_len=k.shape[2],
        q_start_pos=q_start_pos,
        k_start_pos=k_start_pos,
    )
    return mx.fast.scaled_dot_product_attention(
        q=q,
        k=k,
        v=v,
        scale=1.0 / (q.shape[-1] ** 0.5),
        mask=mask,
    )


def exact_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_start_pos: int,
    k_start_pos: int,
) -> mx.array:
    """Compatibility wrapper for the contiguous exact attention path."""
    return run_exact_attention(q, k, v, q_start_pos, k_start_pos)


def run_segmented_attention(
    q: mx.array,
    segments: Sequence[AttentionSegment],
    q_start_pos: int,
) -> mx.array:
    """Run exact causal attention over a small set of K/V segments.

    The segmented path avoids rebuilding a full contiguous K/V tensor
    when archived context is already cached separately from the rolling
    hot buffer.
    """
    if not segments:
        raise RuntimeError("run_segmented_attention requires at least one KV segment")
    if len(segments) == 1:
        k, v, k_start = segments[0]
        return run_exact_attention(q, k, v, q_start_pos, k_start)

    scale = 1.0 / (q.shape[-1] ** 0.5)
    q_f32 = q.astype(mx.float32)
    global_max = None
    segment_logits: List[Tuple[mx.array, mx.array]] = []
    for k, v, k_start in segments:
        logits = (q_f32 @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
        logits = logits + build_causal_mask(
            q_len=q.shape[2],
            k_len=k.shape[2],
            q_start_pos=q_start_pos,
            k_start_pos=k_start,
        )
        segment_logits.append((logits, v.astype(mx.float32)))
        local_max = mx.max(logits, axis=-1, keepdims=True)
        global_max = local_max if global_max is None else mx.maximum(global_max, local_max)

    numerator = None
    denominator = None
    for logits, v in segment_logits:
        weights = mx.exp(logits - global_max)
        weighted_v = weights @ v
        weight_sum = mx.sum(weights, axis=-1, keepdims=True)
        numerator = weighted_v if numerator is None else numerator + weighted_v
        denominator = weight_sum if denominator is None else denominator + weight_sum

    return (numerator / mx.maximum(denominator, 1e-9)).astype(q.dtype)


def exact_prefill_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_start_pos: int,
    k_start_pos: int,
) -> mx.array:
    """Compute batched exact causal attention for prefill.

    This simply forwards to :func:`run_exact_attention`.
    """
    return run_exact_attention(q, k, v, q_start_pos, k_start_pos)


def exact_decode_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_abs_pos: int,
    k_start_pos: int,
) -> mx.array:
    """Compute exact causal attention for a single decode step.

    ``q`` is expected to have sequence length ``1``.  The absolute
    position of the query corresponds to ``q_abs_pos``.  Keys are
    provided starting at ``k_start_pos``.
    """
    # Reuse run_exact_attention: q has Q=1 so q_start_pos == q_abs_pos
    return run_exact_attention(q, k, v, q_abs_pos, k_start_pos)