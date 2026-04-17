"""Lightweight dataclasses for representing encoded and archived blocks.

These structures encapsulate key/value payloads and their associated
metadata. They are intentionally kept free of any tensor operations.
Instead, they serve as containers for data moving between the codec,
cache and attention modules.

The hot path now stores archived RVQ metadata in fixed-width tensor
buffers so the codec can stay inside MLX for both encode and decode.
The older coordinate-based sparse entry types are retained for
compatibility with any external callers that may still construct them,
but the engine itself uses tensor-backed metadata on ``EncodedKeyBlock``.

Pass 4 changes
--------------
- ``ArchivedKVBlock`` gains two optional fields, ``reconstructed_k``
  and ``reconstructed_v``, which cache the decoded key/value tensors
  after the first materialisation. Subsequent decode steps reuse the
  cached tensors without re-running PQ/RVQ decoding or disk I/O.
- ``ArchivedKVBlock`` gains a ``loaded`` flag that is set to ``True``
  once the block's payload has been loaded from disk into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class CacheTier(str, Enum):
    """Enumeration for cache tier identifiers."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class RVQCoord:
    """Coordinate identifying a sparse RVQ entry.

    ``batch_idx``: Which batch in the mini–batch (0 <= batch_idx < B).
    ``head_idx``: Which attention head (0 <= head_idx < H).
    ``time_idx``: Absolute token index within the layer stream. This
      value must be in the half–open interval [start_pos, end_pos) for
      the block in which it appears.
    """

    batch_idx: int
    head_idx: int
    time_idx: int


@dataclass
class RVQSparseEntry:
    """RVQ entry holding codes and their coordinate."""

    coord: RVQCoord
    codes: Any  # expected shape: (num_rvq_layers,)


@dataclass
class EncodedKeyBlock:
    """Container for an encoded key block.

    ``start_pos``: Absolute start position of the block in the layer
      stream (inclusive).
    ``end_pos``: Absolute end position of the block in the layer
      stream (exclusive).
    ``pq_codes``: Product quantiser codes of shape (B, H, L, S).
    ``rvq_flat_indices``: Fixed-width local flat indices shaped
      ``(max_active,)`` into the flattened ``(B * H * L)`` block.
    ``rvq_codes``: Fixed-width RVQ codes shaped
      ``(max_active, num_rvq_layers)``.
    ``rvq_mask``: Boolean validity mask shaped ``(max_active,)``.
    """

    start_pos: int
    end_pos: int
    pq_codes: Any
    rvq_flat_indices: Any
    rvq_codes: Any
    rvq_mask: Any


@dataclass
class ExactKVBlock:
    """Exact key/value block used for warm and hot tiers."""

    start_pos: int
    end_pos: int
    k: Any  # (B, H, L, D)
    v: Any  # (B, H, L, D)


@dataclass
class ArchivedKVBlock:
    """Archived key/value block stored in warm or cold tiers.

    Pass 4 additions
    ----------------
    ``loaded``: Set to ``True`` once the block's payload has been
      loaded from disk. Warm blocks are always loaded; cold blocks
      start unloaded and are loaded on first access.
    ``reconstructed_k`` / ``reconstructed_v``: Optional legacy fields
      for decoded per-block tensors. The current memory-optimized path
      keeps the combined archived context on ``LayerKVCache`` instead,
      so these fields are cleared aggressively to avoid duplicate
      residency.
    """

    start_pos: int
    end_pos: int
    tier: CacheTier
    encoded_k: EncodedKeyBlock
    v_payload: Any | None
    path: Optional[str] = None
    # Pass 4: disk-load bookkeeping
    loaded: bool = False
    # Pass 4: decoded-tensor cache (avoids re-decoding every token)
    reconstructed_k: Any | None = None
    reconstructed_v: Any | None = None


@dataclass
class CodecStats:
    """Statistics tracking invalid or clamped codec operations."""

    rvq_clamp_events: int = 0
    invalid_offset_events: int = 0
    invalid_code_events: int = 0