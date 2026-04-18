"""Configuration for the exact archived-context runtime.

The active runtime has two real execution modes:

- ``RuntimeMode.EXACT`` keeps the entire live context in the hot window
  and raises when a prompt or decode step would exceed
  ``hot_capacity``.
- ``RuntimeMode.ARCHIVED`` keeps the hot window exact and spills sealed
  prefixes into exact archived blocks that can live in RAM or on disk.

``RFSNConfig`` validates the core architectural and runtime invariants
used by the model, cache, launcher, and API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

import mlx.core as mx


_VALID_MODEL_DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def resolve_dtype(name: str) -> mx.Dtype:
    """Resolve a supported tensor dtype name into an MLX dtype."""
    try:
        return _VALID_MODEL_DTYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"model_dtype must be one of {set(_VALID_MODEL_DTYPES)}, got '{name}'"
        ) from exc


def validate_session_id(session_id: str) -> str:
    """Validate and normalize a user-facing cache session identifier."""
    normalized = session_id.strip()
    if not normalized:
        return ""
    if not _SESSION_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "session_id must contain only letters, numbers, '.', '_' or '-'"
        )
    return normalized


class RuntimeMode(str, Enum):
    """Real execution modes for the retained-context runtime."""

    EXACT = "exact"
    ARCHIVED = "archived"


@dataclass
class RFSNConfig:
    """Configuration object for the exact archived-context runtime."""

    hidden_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 0
    head_dim: int = 64
    num_layers: int = 4

    hot_capacity: int = 512
    warm_capacity: int = 2048
    cold_capacity: int = 8192
    block_size_seq: int = 64
    runtime_mode: RuntimeMode = RuntimeMode.ARCHIVED

    vocab_size: int = 50000
    max_position_embeddings: int = 0
    disk_cache_dir: str = "./rfsn_disk_cache"
    session_id: str = ""

    rope_base: float = 10000.0
    ffn_multiplier: int = 4
    ffn_dim: int = 0
    norm_eps: float = 1e-5
    model_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.num_kv_heads <= 0:
            self.num_kv_heads = self.num_heads
        if not isinstance(self.runtime_mode, RuntimeMode):
            self.runtime_mode = RuntimeMode(str(self.runtime_mode))

        if self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must equal num_heads * head_dim "
                f"({self.num_heads * self.head_dim})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if not (self.hot_capacity <= self.warm_capacity <= self.cold_capacity):
            raise ValueError(
                "Cache capacities must satisfy hot_capacity <= warm_capacity <= cold_capacity"
            )
        if self.block_size_seq <= 0:
            raise ValueError("block_size_seq must be > 0")
        if self.max_position_embeddings < 0:
            raise ValueError("max_position_embeddings must be >= 0")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be > 0")
        if not self.disk_cache_dir or not self.disk_cache_dir.strip():
            raise ValueError("disk_cache_dir must be a non-empty path")

        self.session_id = validate_session_id(self.session_id)
        if self.ffn_dim <= 0:
            self.ffn_dim = self.hidden_dim * self.ffn_multiplier
        if self.model_dtype not in _VALID_MODEL_DTYPES:
            raise ValueError(
                f"model_dtype must be one of {set(_VALID_MODEL_DTYPES)}, "
                f"got '{self.model_dtype}'"
            )

    @property
    def kv_groups(self) -> int:
        """Number of Q heads per KV head."""
        return self.num_heads // self.num_kv_heads
