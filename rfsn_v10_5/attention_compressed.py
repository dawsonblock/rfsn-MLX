"""Compressed attention backend.

This module implements the compressed attention path for RFSN.  It
reconstructs archived warm and cold context blocks via the cache and
codec and then applies the same exact causal attention used in the
hot path.  Using a shared helper ensures that exact and compressed
paths remain semantically aligned.

The compressed backend is separate from the hot path so that the
costs of reconstructing archived context do not leak into the fast
execution path.  Consumers should call the methods on this class
instead of reaching into cache internals directly.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

from .attention_exact import AttentionSegment, run_segmented_attention
from .cache import LayerKVCache
from .codec import HybridKeyCodec


def compressed_attention(
    q: mx.array,
    segments: Sequence[AttentionSegment],
    q_start_pos: int,
) -> mx.array:
    """Run mixed-context attention over cached archived and hot segments."""
    return run_segmented_attention(q, segments, q_start_pos)


class CompressedAttentionBackend:
    """Backend for mixed–context attention.

    This class provides two methods: one for computing prefill
    attention when some context has been archived, and one for
    computing single–token decode attention.  Both methods
    reconstruct the full key/value context by delegating to the
    cache, then call the exact attention helper.  No separate
    attention math is implemented here.
    """

    def __init__(self, codec: HybridKeyCodec) -> None:
        self.codec = codec

    def materialize_reconstructed_context(self, cache: LayerKVCache) -> tuple[mx.array, mx.array, int, int]:
        """Reconstruct the full context across cold, warm and hot tiers.

        Parameters
        ----------
        cache: LayerKVCache
            Cache from which to reconstruct context.

        Returns
        -------
        k: mx.array
            Keys shaped ``(B, H, T, D)``.
        v: mx.array
            Values shaped ``(B, H, T, D)``.
        start_pos: int
            Absolute start position of the reconstructed context.
        end_pos: int
            Absolute end position (exclusive) of the reconstructed context.
        """
        return cache.materialize_mixed_context(self.codec)

    def prefill_over_mixed_context(
        self,
        q: mx.array,
        cache: LayerKVCache,
        q_start_pos: int,
    ) -> mx.array:
        """Compute prefill attention over reconstructed context.

        Parameters
        ----------
        q: mx.array
            Query tensor shaped ``(B, H, Q, D)``.
        cache: LayerKVCache
            Layer cache containing archived context.
        q_start_pos: int
            Absolute start position of the query sequence.

        Returns
        -------
        out: mx.array
            Attention output shaped ``(B, H, Q, D)``.
        """
        segments = cache.get_mixed_attention_segments(self.codec)
        return compressed_attention(q, segments, q_start_pos)

    def attention_over_mixed_context(
        self,
        q: mx.array,
        cache: LayerKVCache,
        q_abs_pos: int,
    ) -> mx.array:
        """Compute decode attention over reconstructed context.

        Parameters
        ----------
        q: mx.array
            Query tensor shaped ``(B, H, 1, D)``.
        cache: LayerKVCache
            Layer cache containing archived context.
        q_abs_pos: int
            Absolute position of the query token.

        Returns
        -------
        out: mx.array
            Attention output shaped ``(B, H, 1, D)``.
        """
        segments = cache.get_mixed_attention_segments(self.codec)
        return compressed_attention(q, segments, q_abs_pos)