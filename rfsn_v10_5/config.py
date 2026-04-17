"""Configuration definitions for the RFSN v10.5 implementation.

This module declares the ``RFSNConfig`` dataclass which collects all
hyperparameters and runtime toggles for a transformer instance. The
configuration enforces several invariants at construction time:

- ``hidden_dim`` must equal ``num_heads * head_dim``. This ensures the
  multi-head projection splits the hidden dimension evenly across heads.
- ``num_subspaces * subspace_dim`` must equal ``head_dim``. The product
  quantiser slices the head dimension into subspaces before quantising;
  mismatched values will lead to shape errors.
- ``hot_capacity <= warm_capacity <= cold_capacity``. These tiers
  represent the maximum number of tokens each cache tier can store.
  Violating this ordering would make cache promotion logic ambiguous.
- ``block_size_seq > 0``. This parameter controls the block size used in
  blockwise attention. A non-positive value would break the block loop.
- ``num_kv_heads`` must divide ``num_heads`` evenly. This is required for
  Grouped-Query Attention (GQA) head broadcasting.

Two enums are also defined:
- ``RuntimeMode`` selects between the pure exact path and the compressed
  path. In exact mode, prefill and decode will raise a ``RuntimeError``
  if the context would exceed ``hot_capacity``. In compressed mode the
  archive is consulted when the hot tier overflows.
- ``SafetyMode`` controls codec validation behaviour. In strict mode,
  invalid codes or coordinates raise immediately. In clamp mode, invalid
  values are clipped and counted, allowing best-effort decoding.

Pass 5 changes (retained)
-------------------------
- ``model_dtype``: configurable tensor dtype, default ``"bfloat16"``.

Pass 7 changes
--------------
- ``ffn_dim``: optional explicit FFN hidden size. When set, this takes
  precedence over ``hidden_dim * ffn_multiplier`` so checkpoints with
  non-4x SwiGLU expansions can be loaded faithfully.

Pass 6 changes
--------------
- ``num_kv_heads``: number of KV heads for Grouped-Query Attention.
  Defaults to ``num_heads`` (standard MHA). Set to a smaller value
  (e.g. 8 for Mistral-7B, 8 for LLaMA-3-8B) to enable GQA.
  Must satisfy ``num_heads % num_kv_heads == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import mlx.core as mx


_VALID_MODEL_DTYPES = {
  "float16": mx.float16,
  "bfloat16": mx.bfloat16,
  "float32": mx.float32,
}
_VALID_CACHE_DTYPES = frozenset((*_VALID_MODEL_DTYPES.keys(), "fp8_e4m3"))


def resolve_dtype(name: str) -> mx.Dtype:
  """Resolve a supported tensor dtype name into an MLX dtype."""
  try:
    return _VALID_MODEL_DTYPES[name]
  except KeyError as exc:
    raise ValueError(
      f"dtype must be one of {set(_VALID_MODEL_DTYPES)}, got '{name}'"
    ) from exc


class RuntimeMode(str, Enum):
    """Execution mode for the model.

    ``EXACT``: Only use the exact hot cache. Prefill and decode will
      raise a ``RuntimeError`` if the projected context length would
      exceed ``hot_capacity`` for any layer.
    ``COMPRESSED``: Allow evicted contexts to be reconstructed from
      archived blocks. The exact hot cache remains the default but
      archived context is consulted when the hot tier overflows.
    """

    EXACT = "exact"
    COMPRESSED = "compressed"


class SafetyMode(str, Enum):
    """Safety behaviour for the codec.

    ``STRICT``: Any invalid PQ or RVQ code or coordinate results in an
      immediate error.
    ``CLAMP``: Invalid values are clamped to the nearest valid value
      and counted in the statistics object. This allows best-effort
      decoding without aborting.
    """

    STRICT = "strict"
    CLAMP = "clamp"


@dataclass
class RFSNConfig:
    """Configuration object for the RFSN model.

    Attributes correspond to the model architecture (hidden_dim,
    num_heads, num_kv_heads, head_dim, num_layers), the product
    quantiser settings (num_subspaces, subspace_dim, pq_bits), the
    residual VQ settings (num_rvq_layers, rvq_codebook_size,
    rvq_sparsity_threshold), the tier capacities (hot_capacity,
    warm_capacity, cold_capacity), the block size for blockwise
    attention (block_size_seq), runtime mode (runtime_mode), safety
    mode (safety_mode), and tensor dtype (model_dtype).
    """

    hidden_dim: int = 512
    num_heads: int = 8
    # Pass 6: GQA support. Defaults to num_heads (standard MHA).
    num_kv_heads: int = 0   # 0 -> resolved to num_heads in __post_init__
    head_dim: int = 64
    num_layers: int = 4

    num_subspaces: int = 4
    subspace_dim: int = 16
    pq_bits: int = 8

    num_rvq_layers: int = 2
    rvq_codebook_size: int = 256
    rvq_sparsity_threshold: float = 0.003

    hot_capacity: int = 512
    warm_capacity: int = 2048
    cold_capacity: int = 8192
    block_size_seq: int = 64

    runtime_mode: RuntimeMode = RuntimeMode.EXACT
    safety_mode: SafetyMode = SafetyMode.STRICT

    vocab_size: int = 50000
    disk_cache_dir: str = "./rfsn_disk_cache"

    # Pass 4 additions
    rope_base: float = 10000.0
    ffn_multiplier: int = 4
    ffn_dim: int = 0
    norm_eps: float = 1e-5
    rvq_max_active: int = 0  # 0 -> resolved to block_size_seq * num_heads

    # Pass 5 addition
    model_dtype: str = "bfloat16"
    cache_dtype: str = ""

    def __post_init__(self) -> None:
        # Resolve GQA default
        if self.num_kv_heads <= 0:
            self.num_kv_heads = self.num_heads

        # Validate hidden/head relationship
        if self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must equal num_heads * head_dim "
                f"({self.num_heads * self.head_dim})"
            )
        # Validate GQA divisibility
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        # Validate subspace/head relationship
        if self.num_subspaces * self.subspace_dim != self.head_dim:
            raise ValueError(
                f"num_subspaces * subspace_dim ({self.num_subspaces * self.subspace_dim})"
                f" must equal head_dim ({self.head_dim})"
            )
        # Validate tier ordering
        if not (self.hot_capacity <= self.warm_capacity <= self.cold_capacity):
            raise ValueError(
                "Cache capacities must satisfy hot_capacity <= warm_capacity <= cold_capacity"
            )
        # Validate block size
        if self.block_size_seq <= 0:
            raise ValueError("block_size_seq must be > 0")
        # Resolve rvq_max_active default
        if self.rvq_max_active <= 0:
            self.rvq_max_active = self.block_size_seq * self.num_heads
        # Resolve FFN size default
        if self.ffn_dim <= 0:
          self.ffn_dim = self.hidden_dim * self.ffn_multiplier
        # Resolve cache dtype default
        if not self.cache_dtype:
          self.cache_dtype = self.model_dtype
        # Validate dtype string
        if self.model_dtype not in _VALID_MODEL_DTYPES:
          raise ValueError(
            f"model_dtype must be one of {set(_VALID_MODEL_DTYPES)}, "
            f"got '{self.model_dtype}'"
          )
        if self.cache_dtype not in _VALID_CACHE_DTYPES:
          raise ValueError(
            f"cache_dtype must be one of {_VALID_CACHE_DTYPES}, "
            f"got '{self.cache_dtype}'"
          )

    @property
    def kv_groups(self) -> int:
        """Number of Q heads per KV head (GQA repeat factor)."""
        return self.num_heads // self.num_kv_heads
