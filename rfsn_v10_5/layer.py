"""Single transformer layer for the exact block-managed runtime.

Phase 1 removes the archive reconstruction branch. Every forward pass
uses the same exact attention semantics over a combination of archived
exact segments and the current hot window.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention_exact import exact_attention, run_segmented_attention
from .cache import LayerKVCache
from .config import RFSNConfig, resolve_dtype


def build_rope_tables(
    head_dim: int,
    rope_base: float,
    seq_len: int,
    start_pos: int = 0,
    dtype: mx.Dtype = mx.float32,
) -> Tuple[mx.array, mx.array]:
    """Pre-compute RoPE cosine and sine tables."""
    half = head_dim // 2
    inv_freq = 1.0 / (rope_base ** (mx.arange(0, half, dtype=mx.float32) / half))
    positions = mx.arange(start_pos, start_pos + seq_len, dtype=mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos_t = mx.cos(emb).astype(dtype)
    sin_t = mx.sin(emb).astype(dtype)
    return cos_t, sin_t


def _apply_rope(x: mx.array, cos_t: mx.array, sin_t: mx.array) -> mx.array:
    """Apply rotary embeddings to x shaped ``(B, H, L, D)``."""
    half = x.shape[-1] // 2
    x1 = x[:, :, :, :half]
    x2 = x[:, :, :, half:]
    cos = cos_t[None, None, :, :]
    sin = sin_t[None, None, :, :]
    return mx.concatenate(
        [
            x1 * cos[:, :, :, :half] - x2 * sin[:, :, :, :half],
            x1 * sin[:, :, :, half:] + x2 * cos[:, :, :, half:],
        ],
        axis=-1,
    )


class RFSNLayerNorm(nn.Module):
    """RMS LayerNorm (no bias, matches LLaMA/Mistral)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * (x / rms)


class RFSNLayerMLX(nn.Module):
    """Single RFSN transformer layer using exact segmented attention."""

    def __init__(self, config: RFSNConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = resolve_dtype(config.model_dtype)

        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads
        head_dim = config.head_dim
        hidden_dim = config.hidden_dim
        ffn_dim = config.ffn_dim

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

        self.attn_norm = RFSNLayerNorm(hidden_dim, eps=config.norm_eps)
        self.ffn_norm = RFSNLayerNorm(hidden_dim, eps=config.norm_eps)

    def _ffn(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    def __call__(
        self,
        x: mx.array,
        cache: LayerKVCache,
        rope_tables: Optional[Tuple[mx.array, mx.array]] = None,
        start_pos: int = 0,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        num_heads = self.config.num_heads
        num_kv_heads = self.config.num_kv_heads
        head_dim = self.config.head_dim

        residual = x
        x_norm = self.attn_norm(x)

        q = self.q_proj(x_norm).reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x_norm).reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = q.astype(self.dtype)
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)

        if rope_tables is None:
            cos_t, sin_t = build_rope_tables(
                head_dim,
                self.config.rope_base,
                seq_len,
                start_pos=start_pos,
                dtype=self.dtype,
            )
        else:
            cos_t, sin_t = rope_tables

        q = _apply_rope(q, cos_t, sin_t)
        k = _apply_rope(k, cos_t, sin_t)

        cache.evict_for_append(seq_len)
        cache.append_exact(k, v)
        if seq_len == 1:
            cache.maybe_prefetch_for_decode(start_pos)

        segments = cache.get_attention_segments()
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
            context_k, context_v, context_start = segments[0]
            attn_out = exact_attention(
                q,
                context_k,
                context_v,
                q_start_pos=start_pos,
                k_start_pos=context_start,
            )
        else:
            attn_out = run_segmented_attention(q, segments, start_pos)

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)
        x = residual + self.o_proj(attn_out)
        x = x + self._ffn(self.ffn_norm(x))
        return x
