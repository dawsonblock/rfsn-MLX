"""Single transformer layer for the RFSN v10.5 implementation.

This module defines ``RFSNLayerMLX``, which wires together:

- Pre-norm RMS LayerNorm (applied before attention and FFN).
- Q, K, V linear projections. In GQA mode, K and V project to
  ``num_kv_heads * head_dim`` while Q projects to ``num_heads * head_dim``.
- Rotary Position Embedding (RoPE) applied to Q and K after projection.
- KV cache append with evict-before-append in COMPRESSED mode.
- Routing to ``attention_exact`` or ``attention_compressed`` depending on
  ``RuntimeMode`` and whether archived context exists.
- Output projection and residual connection.
- SwiGLU FFN (gate * silu(up) -> down) with pre-norm.

It also exports ``build_rope_tables``, a standalone function that
pre-computes the cosine and sine tables for RoPE. The model calls this
once per forward pass and passes the tables to every layer, avoiding
redundant trigonometric computation (RoPE hoisting).

Pass 5 changes (retained)
--------------------------
- ``build_rope_tables`` exported for RoPE hoisting.
- EXACT mode raises ``RuntimeError`` before append if hot tier would overflow.

Pass 6 changes
--------------
Fix 1 - Evict-before-append in COMPRESSED mode
  The Pass 5 design called ``append_exact`` first and ``maybe_evict``
  after. Because ``append_exact`` now raises on overflow, the compressed
  engine could never cross the hot-tier boundary.
  Fix: In COMPRESSED mode, call ``cache.evict_for_append(codec, L)``
  BEFORE ``cache.append_exact(k, v)``. This guarantees the buffer has
  room before the append. For prompts longer than ``hot_capacity``,
  raise an explicit ``RuntimeError`` with a clear message (chunked
  prefill is a future feature).

Fix 2 - GQA (Grouped-Query Attention)
  K and V projections now output ``num_kv_heads * head_dim``. After
  projection and RoPE, K and V are repeated along the head axis by
  ``kv_groups = num_heads // num_kv_heads`` before being passed to
  attention. The KV cache stores the un-repeated (compact) tensors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention_compressed import compressed_attention
from .attention_exact import exact_attention
from .cache import LayerKVCache
from .codec import HybridKeyCodec
from .config import RFSNConfig, RuntimeMode, resolve_dtype


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def build_rope_tables(
    head_dim: int,
    rope_base: float,
    seq_len: int,
    start_pos: int = 0,
    dtype: mx.Dtype = mx.float32,
) -> Tuple[mx.array, mx.array]:
    """Pre-compute RoPE cosine and sine tables.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head. Must be even.
    rope_base : float
        Base for the geometric progression of frequencies.
    seq_len : int
        Number of positions to compute tables for.
    start_pos : int
        Absolute position offset (for incremental decode).
    dtype : mx.Dtype
        Output dtype for the tables.

    Returns
    -------
    cos_t : mx.array, shape (seq_len, head_dim)
    sin_t : mx.array, shape (seq_len, head_dim)
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        rope_base ** (mx.arange(0, half, dtype=mx.float32) / half)
    )
    positions = mx.arange(start_pos, start_pos + seq_len, dtype=mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)  # (L, head_dim)
    cos_t = mx.cos(emb).astype(dtype)
    sin_t = mx.sin(emb).astype(dtype)
    return cos_t, sin_t


def _apply_rope(
    x: mx.array,
    cos_t: mx.array,
    sin_t: mx.array,
) -> mx.array:
    """Apply rotary embeddings to x (B, H, L, D). cos_t/sin_t shape: (L, D)."""
    half = x.shape[-1] // 2
    x1 = x[:, :, :, :half]
    x2 = x[:, :, :, half:]
    cos = cos_t[None, None, :, :]
    sin = sin_t[None, None, :, :]
    return mx.concatenate(
        [x1 * cos[:, :, :, :half] - x2 * sin[:, :, :, :half],
         x1 * sin[:, :, :, half:] + x2 * cos[:, :, :, half:]],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# RMS Layer norm
# ---------------------------------------------------------------------------

class RFSNLayerNorm(nn.Module):
    """RMS LayerNorm (no bias, matches LLaMA/Mistral)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * (x / rms)


# ---------------------------------------------------------------------------
# Main layer
# ---------------------------------------------------------------------------

class RFSNLayerMLX(nn.Module):
    """Single RFSN transformer layer.

    Supports both standard MHA (``num_kv_heads == num_heads``) and GQA
    (``num_kv_heads < num_heads``).
    """

    def __init__(self, config: RFSNConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = resolve_dtype(config.model_dtype)

        H    = config.num_heads
        Hkv  = config.num_kv_heads
        D    = config.head_dim
        hidden  = config.hidden_dim
        ffn_dim = config.ffn_dim

        # Projections
        self.q_proj = nn.Linear(hidden, H * D,   bias=False)
        self.k_proj = nn.Linear(hidden, Hkv * D, bias=False)
        self.v_proj = nn.Linear(hidden, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, hidden,   bias=False)

        # FFN (SwiGLU)
        self.gate_proj = nn.Linear(hidden, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(hidden, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden, bias=False)

        # Norms
        self.attn_norm = RFSNLayerNorm(hidden, eps=config.norm_eps)
        self.ffn_norm  = RFSNLayerNorm(hidden, eps=config.norm_eps)

        # Codec (one per layer so codebooks can specialise per depth)
        self.codec = HybridKeyCodec(config)

    def _ffn(self, x: mx.array) -> mx.array:
        """SwiGLU feed-forward network."""
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    def __call__(
        self,
        x: mx.array,
        cache: LayerKVCache,
        rope_tables: Optional[Tuple[mx.array, mx.array]] = None,
        start_pos: int = 0,
    ) -> mx.array:
        """Forward pass for a single layer.

        Parameters
        ----------
        x : mx.array, shape (B, L, hidden_dim)
        cache : LayerKVCache
        rope_tables : (cos_t, sin_t) pre-computed by the model, or None
            to compute on the fly.
        start_pos : int
            Absolute position of the first token in x (for RoPE).
        """
        B, L, _ = x.shape
        H   = self.config.num_heads
        Hkv = self.config.num_kv_heads
        D   = self.config.head_dim

        # ---- Attention pre-norm ----
        residual = x
        x_norm = self.attn_norm(x)

        # ---- Projections ----
        q = self.q_proj(x_norm).reshape(B, L, H,   D).transpose(0, 2, 1, 3)
        k = self.k_proj(x_norm).reshape(B, L, Hkv, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(B, L, Hkv, D).transpose(0, 2, 1, 3)

        q = q.astype(self.dtype)
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)

        # ---- RoPE ----
        if rope_tables is None:
            cos_t, sin_t = build_rope_tables(
                D, self.config.rope_base, L, start_pos=start_pos, dtype=self.dtype
            )
        else:
            cos_t, sin_t = rope_tables

        q = _apply_rope(q, cos_t, sin_t)
        k = _apply_rope(k, cos_t, sin_t)

        # ---- Cache update (evict-before-append in COMPRESSED mode) ----
        if self.config.runtime_mode == RuntimeMode.EXACT:
            projected_hot_len = cache.hot_seq_len + L
            if projected_hot_len > self.config.hot_capacity:
                raise RuntimeError(
                    f"EXACT mode: projected hot length {projected_hot_len} exceeds "
                    f"hot_capacity={self.config.hot_capacity}. "
                    "Switch to RuntimeMode.COMPRESSED for long contexts."
                )
            cache.append_exact(k, v)
        else:
            # COMPRESSED mode: make room BEFORE appending (Pass 6 fix)
            if L > self.config.hot_capacity:
                raise RuntimeError(
                    f"Compressed prefill with L={L} > hot_capacity="
                    f"{self.config.hot_capacity} is not implemented yet. "
                    "Process the prompt in chunks of at most hot_capacity tokens."
                )
            cache.evict_for_append(self.codec, L)   # evict first
            cache.append_exact(k, v)                 # then append safely
            cache.maybe_evict(self.codec)            # warm->cold spill

        # ---- Attention ----
        has_archive = bool(cache.warm_blocks or cache.cold_blocks)
        if has_archive and self.config.runtime_mode == RuntimeMode.COMPRESSED:
            segments = cache.get_mixed_attention_segments(self.codec)
        else:
            segments = cache.get_hot_attention_segments()

        if self.config.kv_groups > 1:
            expanded_segments = []
            for seg_k, seg_v, seg_start in segments:
                expanded_segments.append(
                    (
                        mx.repeat(seg_k, self.config.kv_groups, axis=1),
                        mx.repeat(seg_v, self.config.kv_groups, axis=1),
                        seg_start,
                    )
                )
            segments = expanded_segments

        if len(segments) == 1:
            ctx_k, ctx_v, ctx_start = segments[0]
            attn_out = exact_attention(
                q,
                ctx_k,
                ctx_v,
                q_start_pos=start_pos,
                k_start_pos=ctx_start,
            )
        else:
            attn_out = compressed_attention(q, segments, q_start_pos=start_pos)

        # ---- Output projection + residual ----
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, H * D)
        x = residual + self.o_proj(attn_out)

        # ---- FFN with pre-norm ----
        x = x + self._ffn(self.ffn_norm(x))
        return x
