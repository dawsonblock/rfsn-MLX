"""Full RFSN transformer model.

This module implements ``RFSNMLX``, the top-level model class.

Pass 4 features (retained)
--------------------------
- Token embedding, layer stack, final LayerNorm, tied LM head.
- ``prefill``, ``decode_step``, and ``generate`` with temperature/top-p.

Pass 5 changes
--------------
Fix 1 - RoPE hoisting
  In Pass 4, every layer call recomputed ``mx.cos`` and ``mx.sin`` for
  the current sequence positions. For a 32-layer model this means 32
  trigonometric evaluations per forward pass. The model now computes
  the RoPE tables once per ``prefill`` / ``decode_step`` call and passes
  them down to every layer via the ``rope_tables`` argument.

Fix 2 - Top-K sampling
  Added ``top_k`` parameter to ``generate`` and ``_sample``. When set,
  the logit distribution is truncated to the top-k entries before
  applying temperature and top-p. This matches the standard LLM
  sampling pipeline (top_k → top_p → temperature → categorical).

Fix 3 - Repetition penalty
  Added ``repetition_penalty`` parameter. Logits for tokens that have
  already appeared in the context are divided by the penalty factor
  (values > 1.0 discourage repetition). Applied before top-k/top-p.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .cache import RFSNCache
from .config import RFSNConfig, RuntimeMode
from .layer import RFSNLayerMLX, build_rope_tables


class RFSNMLX(nn.Module):
    """Full RFSN transformer model.

    Parameters
    ----------
    config : RFSNConfig
    """

    def __init__(self, config: RFSNConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer layers
        self.layers = [
            RFSNLayerMLX(config, layer_idx=i)
            for i in range(config.num_layers)
        ]

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)

    def _lm_head(self, x: mx.array) -> mx.array:
        """Project hidden states to vocabulary logits (weight-tied)."""
        return x @ self.embed_tokens.weight.T

    def _context_end(self, cache: RFSNCache) -> int:
        return cache.layer(0).hot_end

    def _validate_prefill_bounds(self, cache: RFSNCache, incoming_tokens: int) -> int:
        context_end = self._context_end(cache)
        projected = context_end + incoming_tokens
        if self.config.max_position_embeddings > 0 and projected > self.config.max_position_embeddings:
            raise ValueError(
                f"Prompt would exceed max_position_embeddings={self.config.max_position_embeddings} "
                f"(requested {projected} tokens total)"
            )
        if self.config.runtime_mode == RuntimeMode.EXACT and projected > self.config.hot_capacity:
            raise RuntimeError(
                "runtime_mode='exact' is hot-window only; the prompt would exceed "
                f"hot_capacity={self.config.hot_capacity}"
            )
        return context_end

    def _validate_decode_bounds(self, cache: RFSNCache, pos: int) -> None:
        context_end = self._context_end(cache)
        if pos != context_end:
            raise ValueError(
                f"decode_step expected pos={context_end} from cache state, got pos={pos}"
            )
        projected = pos + 1
        if self.config.max_position_embeddings > 0 and projected > self.config.max_position_embeddings:
            raise ValueError(
                f"Decode would exceed max_position_embeddings={self.config.max_position_embeddings}"
            )
        if self.config.runtime_mode == RuntimeMode.EXACT and projected > self.config.hot_capacity:
            raise RuntimeError(
                "runtime_mode='exact' is hot-window only; the decode step would exceed "
                f"hot_capacity={self.config.hot_capacity}"
            )

    def _forward(
        self,
        x: mx.array,
        cache: RFSNCache,
        start_pos: int,
    ) -> mx.array:
        """Shared forward pass used by both prefill and decode_step.

        RoPE tables are computed once here and passed to every layer.
        """
        B, L, _ = x.shape
        D = self.config.head_dim

        # Hoist RoPE table computation (Pass 5 Fix 1)
        rope_tables = build_rope_tables(D, self.config.rope_base, L, start_pos)

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                cache.layer(i),
                rope_tables=rope_tables,
                start_pos=start_pos,
            )

        x = self.norm(x)
        return self._lm_head(x)

    def prefill(
        self,
        tokens: mx.array,
        cache: RFSNCache,
    ) -> mx.array:
        """Process a full prompt and populate the KV cache.

        Parameters
        ----------
        tokens : mx.array, shape (B, L) int32
        cache : RFSNCache

        Returns
        -------
        logits : mx.array, shape (B, L, vocab_size)
        """
        _, seq_len = tokens.shape
        if seq_len == 0:
            raise ValueError("prefill requires at least one token")

        self._validate_prefill_bounds(cache, seq_len)
        chunk_size = max(1, min(self.config.hot_capacity, seq_len))
        logits_chunks: List[mx.array] = []
        chunk_start = 0
        while chunk_start < seq_len:
            chunk_end = min(seq_len, chunk_start + chunk_size)
            token_chunk = tokens[:, chunk_start:chunk_end]
            start_pos = self._context_end(cache)
            x = self.embed_tokens(token_chunk)
            logits_chunks.append(self._forward(x, cache, start_pos))
            chunk_start = chunk_end

        return logits_chunks[0] if len(logits_chunks) == 1 else mx.concatenate(logits_chunks, axis=1)

    def decode_step(
        self,
        token_id: mx.array,
        cache: RFSNCache,
        pos: int,
    ) -> mx.array:
        """Single autoregressive decode step.

        Parameters
        ----------
        token_id : mx.array, shape (B,) int32
        cache : RFSNCache
        pos : int
            Absolute position of the token being decoded.

        Returns
        -------
        logits : mx.array, shape (B, vocab_size)
        """
        self._validate_decode_bounds(cache, pos)
        x = self.embed_tokens(token_id[:, None])  # (B, 1, hidden)
        logits = self._forward(x, cache, pos)      # (B, 1, vocab_size)
        return logits[:, 0, :]                     # (B, vocab_size)

    def generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int,
        cache: Optional[RFSNCache] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> List[mx.array]:
        """Generate tokens autoregressively.

        Parameters
        ----------
        prompt_ids : mx.array, shape (B, L) int32
        max_new_tokens : int
        cache : RFSNCache, optional
        temperature : float
            Softmax temperature. 1.0 = no scaling.
        top_p : float
            Nucleus sampling threshold. 1.0 = disabled.
        top_k : int
            Hard cutoff on the top-k logits. 0 = disabled.
        repetition_penalty : float
            Values > 1.0 penalise repeated tokens. 1.0 = disabled.

        Returns
        -------
        generated : list of mx.array, each shape (B,)
        """
        B, L = prompt_ids.shape
        if cache is None:
            cache = RFSNCache(self.config, batch_size=B)

        context_end = self._context_end(cache)
        requested_total = context_end + L + max_new_tokens
        if self.config.max_position_embeddings > 0 and requested_total > self.config.max_position_embeddings:
            raise ValueError(
                f"Prompt plus requested generation would exceed "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )
        if self.config.runtime_mode == RuntimeMode.EXACT and requested_total > self.config.hot_capacity:
            raise RuntimeError(
                "runtime_mode='exact' is hot-window only; prompt plus requested generation would exceed "
                f"hot_capacity={self.config.hot_capacity}"
            )

        # Track all token ids seen so far for repetition penalty
        seen_ids: mx.array = prompt_ids  # (B, L)

        logits = self.prefill(prompt_ids, cache)
        mx.eval(logits)

        generated: List[mx.array] = []
        next_token = self._sample(
            logits[:, -1, :], temperature, top_p, top_k, repetition_penalty, seen_ids
        )
        generated.append(next_token)
        mx.eval(next_token)
        seen_ids = mx.concatenate([seen_ids, next_token[:, None]], axis=1)

        pos = L
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_token, cache, pos)
            mx.eval(logits)
            next_token = self._sample(
                logits, temperature, top_p, top_k, repetition_penalty, seen_ids
            )
            generated.append(next_token)
            mx.eval(next_token)
            seen_ids = mx.concatenate([seen_ids, next_token[:, None]], axis=1)
            pos += 1

        return generated

    @staticmethod
    def _sample(
        logits: mx.array,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seen_ids: mx.array,
    ) -> mx.array:
        """Sample a token with optional repetition penalty, top-k, top-p, temperature.

        Parameters
        ----------
        logits : mx.array, shape (B, vocab_size)
        temperature : float
        top_p : float
        top_k : int
        repetition_penalty : float
        seen_ids : mx.array, shape (B, S) int32

        Returns
        -------
        token_ids : mx.array, shape (B,) int32
        """
        # Greedy shortcut
        if temperature == 0.0:
            return mx.argmax(logits, axis=-1).astype(mx.int32)

        # Step 1: repetition penalty
        if repetition_penalty != 1.0:
            B, V = logits.shape
            # Build a one-hot mask for seen tokens: (B, V)
            # We do this in numpy to avoid dynamic-shape issues in MLX
            import numpy as np
            seen_np = np.array(seen_ids, dtype=np.int32)
            mask_np = np.zeros((B, V), dtype=np.float32)
            for b in range(B):
                mask_np[b, seen_np[b]] = 1.0
            mask = mx.array(mask_np)
            # Divide positive logits, multiply negative logits
            logits = mx.where(
                logits > 0,
                logits / (mask * (repetition_penalty - 1.0) + 1.0),
                logits * (mask * (repetition_penalty - 1.0) + 1.0),
            )

        # Step 2: top-k truncation
        if top_k > 0:
            # Keep only the top-k logits; set the rest to -inf
            topk_vals = mx.topk(logits, top_k, axis=-1)          # (B, k)
            threshold = topk_vals[:, -1:] if top_k > 1 else topk_vals  # (B, 1)
            logits = mx.where(logits >= threshold, logits, mx.full(logits.shape, float("-inf")))

        # Step 3: temperature scaling
        scaled = logits / temperature

        # Step 4: top-p (nucleus) sampling
        if top_p < 1.0:
            probs = mx.softmax(scaled, axis=-1)
            sorted_idx = mx.argsort(-probs, axis=-1)
            sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
            cumsum = mx.cumsum(sorted_probs, axis=-1)
            nucleus_mask = (cumsum - sorted_probs) < top_p
            sorted_probs = sorted_probs * nucleus_mask
            sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
            sampled_local = mx.random.categorical(mx.log(sorted_probs + 1e-10))
            token_ids = mx.take_along_axis(sorted_idx, sampled_local[:, None], axis=-1)[:, 0]
        else:
            token_ids = mx.random.categorical(scaled)

        return token_ids.astype(mx.int32)
