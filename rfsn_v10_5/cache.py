"""Per-layer cache for key/value storage with tiered eviction.

This module defines two classes:

- ``LayerKVCache`` manages the cache for a single transformer layer. It
  stores exact keys and values in the hot tier (on the device) and
  supports eviction of the oldest tokens into a warm tier (in RAM).
  When the warm tier grows beyond its capacity, blocks are spilled into
  a cold tier on disk. The cache uses the provided codec to compress
  keys when evicting and to decode them when reconstructing archived
  context.
- ``RFSNCache`` is a thin wrapper around a list of per-layer caches.

The cache design follows these rules:

1. The newest appended chunk of tokens must always remain in the hot
   tier. Only the oldest prefix can be evicted when the hot tier
   exceeds ``hot_capacity``.
2. Warm blocks hold an ``EncodedKeyBlock`` and a compact ``V`` payload.
    In the default path values are stored as numpy float16 arrays; when
    ``cache_dtype='fp8_e4m3'`` they are stored as packed uint8 bytes.
    Warm blocks live in memory.
3. Cold blocks are stored on disk. The encoded keys and compact value
    payload are serialised to disk. ``v_payload`` is retained in memory
   (``loaded=True``) to avoid disk I/O in the decode loop; a future
   RAM-pressure manager may set it to ``None`` to reclaim memory.
4. The cache tracks the total length of stored context across all tiers
   via ``get_total_length()``.
5. ``materialize_mixed_context()`` reconstructs the full context by
   decoding warm and cold blocks and concatenating them with the current
   hot block. It enforces continuity by verifying that the blocks are
   contiguous in time.

Pass 4 changes (retained)
--------------------------
- Cold blocks loaded eagerly at eviction time (no disk I/O in decode loop).
- Archived context cached at the combined tier level on
    ``LayerKVCache._archived_k/_archived_v`` to avoid repeated PQ/RVQ
    decoding on every token step without keeping duplicate per-block
    reconstructed tensors alive.

Pass 5 changes (retained)
--------------------------
- Pre-allocated hot-tier buffer: O(1) in-place update via ``.at[].set()``.
- ``maybe_evict`` evicts at least ``block_size_seq`` tokens (fragmentation fix).
- Configurable dtype via ``config.model_dtype``.

Pass 6 changes
--------------
Fix 1 - cold_capacity enforcement
  The config validated ``hot <= warm <= cold`` but the cache never
  actually enforced the cold tier limit. ``maybe_evict`` now counts the
  cold tier length and emits a ``RuntimeWarning`` when it is exceeded.
  Full eviction of cold blocks (freeing disk space) is left to a future
  pass because it requires a policy decision about which blocks to drop.

Fix 2 - GQA-aware KV cache dimensions
  ``LayerKVCache`` now accepts ``num_kv_heads`` separately from
  ``num_heads``. The hot buffer is shaped ``(B, num_kv_heads, hot_capacity, D)``
  instead of ``(B, num_heads, hot_capacity, D)``. This matches the
  projected KV tensors from the GQA layer. The codec and archived blocks
  also use ``num_kv_heads``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
import mlx.core as mx

from .config import RFSNConfig, RuntimeMode, resolve_dtype
from .codec import HybridKeyCodec
from .fp8 import native_fp8_e4m3_dtype, pack_fp8_e4m3, unpack_fp8_e4m3
from .types import (
    ArchivedKVBlock,
    CacheTier,
    EncodedKeyBlock,
    ExactKVBlock,
)


class LayerKVCache:
    """Cache for a single transformer layer.

    Pass 5: hot tier is a pre-allocated fixed-size buffer.
    Pass 6: GQA-aware — buffer shaped for ``num_kv_heads``.
    """

    def __init__(
        self,
        config: RFSNConfig,
        batch_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.dtype = resolve_dtype(config.model_dtype)
        self.cache_dtype_name = config.cache_dtype
        self.cache_storage_dtype = self._resolve_cache_storage_dtype(config.cache_dtype)
        self.uses_packed_fp8 = (
            config.cache_dtype == "fp8_e4m3" and self.cache_storage_dtype == mx.uint8
        )

        D = config.head_dim
        # Hot buffer shaped for KV heads (GQA: num_kv_heads <= num_heads)
        self.hot_k: mx.array = mx.zeros(
            (batch_size, self.num_kv_heads, config.hot_capacity, D),
            dtype=self.cache_storage_dtype,
        )
        self.hot_v: mx.array = mx.zeros(
            (batch_size, self.num_kv_heads, config.hot_capacity, D),
            dtype=self.cache_storage_dtype,
        )
        self.hot_seq_len: int = 0   # number of valid slots in the buffer
        self.hot_start: int = 0     # absolute position of slot 0
        self.hot_head_index: int = 0

        # Warm and cold tiers
        self.warm_blocks: List[ArchivedKVBlock] = []
        self.cold_blocks: List[ArchivedKVBlock] = []

        self._archived_context_dirty: bool = True
        self._archived_k: Optional[mx.array] = None
        self._archived_v: Optional[mx.array] = None
        self._archived_start: Optional[int] = None
        self._archived_end: Optional[int] = None

        # Disk storage
        self.cache_dir = Path(config.disk_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_cache_storage_dtype(name: str) -> mx.Dtype:
        if name == "fp8_e4m3":
            native_dtype = native_fp8_e4m3_dtype()
            return native_dtype if native_dtype is not None else mx.uint8
        return resolve_dtype(name)

    def _encode_cache_tensor(self, tensor: mx.array) -> mx.array:
        if self.uses_packed_fp8:
            return pack_fp8_e4m3(tensor)
        return tensor.astype(self.cache_storage_dtype)

    def _decode_cache_tensor(self, tensor: mx.array) -> mx.array:
        if self.uses_packed_fp8:
            return unpack_fp8_e4m3(tensor, dtype=self.dtype)
        if tensor.dtype == self.dtype:
            return tensor
        return tensor.astype(self.dtype)

    def _encode_archived_v_payload(self, tensor: mx.array) -> np.ndarray:
        if self.cache_dtype_name == "fp8_e4m3":
            return np.asarray(pack_fp8_e4m3(tensor))
        return np.asarray(tensor.astype(mx.float16))

    def _decode_archived_v_payload(self, payload: Any) -> mx.array:
        if self.cache_dtype_name == "fp8_e4m3":
            return unpack_fp8_e4m3(mx.array(payload, dtype=mx.uint8), dtype=self.dtype)
        return mx.array(payload, dtype=mx.float16).astype(self.dtype)

    def _mark_archived_context_dirty(self) -> None:
        self._archived_context_dirty = True
        self._archived_k = None
        self._archived_v = None
        self._archived_start = None
        self._archived_end = None
        self._clear_block_reconstruction_cache()

    def _clear_block_reconstruction_cache(self) -> None:
        for block in self.warm_blocks + self.cold_blocks:
            block.reconstructed_k = None
            block.reconstructed_v = None

    @property
    def hot_write_index(self) -> int:
        return (self.hot_head_index + self.hot_seq_len) % self.config.hot_capacity

    @property
    def hot_end(self) -> int:
        """Absolute position one past the last valid hot slot."""
        return self.hot_start + self.hot_seq_len

    def _empty_context(self) -> Tuple[mx.array, mx.array, int, int]:
        empty = mx.zeros(
            (self.batch_size, self.num_kv_heads, 0, self.config.head_dim),
            dtype=self.dtype,
        )
        return empty, empty, self.hot_start, self.hot_start

    def _hot_span_segments(
        self,
        logical_start: int = 0,
        length: Optional[int] = None,
    ) -> List[Tuple[mx.array, mx.array, int]]:
        if self.hot_seq_len == 0:
            return []
        if length is None:
            length = self.hot_seq_len - logical_start
        if length <= 0:
            return []

        capacity = self.config.hot_capacity
        physical_start = (self.hot_head_index + logical_start) % capacity
        absolute_start = self.hot_start + logical_start
        first_len = min(length, capacity - physical_start)
        segments: List[Tuple[mx.array, mx.array, int]] = []
        if first_len > 0:
            segments.append(
                (
                    self.hot_k[:, :, physical_start:physical_start + first_len, :],
                    self.hot_v[:, :, physical_start:physical_start + first_len, :],
                    absolute_start,
                )
            )
        remaining = length - first_len
        if remaining > 0:
            segments.append(
                (
                    self.hot_k[:, :, :remaining, :],
                    self.hot_v[:, :, :remaining, :],
                    absolute_start + first_len,
                )
            )
        return segments

    def _materialize_hot_span(
        self,
        logical_start: int = 0,
        length: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array, int, int]:
        segments = self._hot_span_segments(logical_start=logical_start, length=length)
        if not segments:
            return self._empty_context()
        ks = [self._decode_cache_tensor(seg_k) for seg_k, _, _ in segments]
        vs = [self._decode_cache_tensor(seg_v) for _, seg_v, _ in segments]
        start = segments[0][2]
        total_len = sum(seg_k.shape[2] for seg_k, _, _ in segments)
        k = ks[0] if len(ks) == 1 else mx.concatenate(ks, axis=2)
        v = vs[0] if len(vs) == 1 else mx.concatenate(vs, axis=2)
        return k, v, start, start + total_len

    def get_hot_attention_segments(self) -> List[Tuple[mx.array, mx.array, int]]:
        segments = []
        for seg_k, seg_v, start in self._hot_span_segments():
            segments.append(
                (
                    self._decode_cache_tensor(seg_k),
                    self._decode_cache_tensor(seg_v),
                    start,
                )
            )
        return segments

    def _archive_hot_prefix(self, codec: HybridKeyCodec, prefix_len: int) -> None:
        k_prefix, v_prefix, prefix_start, prefix_end = self._materialize_hot_span(
            logical_start=0,
            length=prefix_len,
        )
        encoded = codec.encode_keys(k_prefix, start_pos=prefix_start)

        warm_block = ArchivedKVBlock(
            start_pos=prefix_start,
            end_pos=prefix_end,
            tier=CacheTier.WARM,
            encoded_k=encoded,
            v_payload=self._encode_archived_v_payload(v_prefix),
            path=None,
            loaded=True,
            reconstructed_k=None,
            reconstructed_v=None,
        )
        self.warm_blocks.append(warm_block)
        self.hot_head_index = (self.hot_head_index + prefix_len) % self.config.hot_capacity
        self.hot_start = prefix_end
        self.hot_seq_len -= prefix_len
        self._mark_archived_context_dirty()

    def _write_hot_segment(
        self,
        buffer: mx.array,
        start_index: int,
        values: mx.array,
    ) -> mx.array:
        indices = mx.broadcast_to(
            mx.arange(start_index, start_index + values.shape[2], dtype=mx.int32)[
                None, None, :, None
            ],
            values.shape,
        )
        return mx.put_along_axis(buffer, indices, values, axis=2)

    # ------------------------------------------------------------------
    # Append and eviction helpers
    # ------------------------------------------------------------------

    def append_exact(self, k: mx.array, v: mx.array) -> None:
        """Append exact keys and values to the hot tier.

        Uses in-place indexed update into the pre-allocated buffer.
        Raises if the buffer would overflow (caller must evict first).

        Parameters
        ----------
        k : mx.array, shape (B, num_kv_heads, L, D)
        v : mx.array, shape (B, num_kv_heads, L, D)
        """
        _, _, L, _ = k.shape
        end = self.hot_seq_len + L
        if end > self.config.hot_capacity:
            raise RuntimeError(
                f"append_exact: hot buffer overflow ({end} > {self.config.hot_capacity}). "
                "Call evict_for_append before appending in COMPRESSED mode."
            )
        k_cast = self._encode_cache_tensor(k)
        v_cast = self._encode_cache_tensor(v)
        write_index = self.hot_write_index
        first_len = min(L, self.config.hot_capacity - write_index)
        if first_len > 0:
            self.hot_k = self._write_hot_segment(
                self.hot_k,
                write_index,
                k_cast[:, :, :first_len, :],
            )
            self.hot_v = self._write_hot_segment(
                self.hot_v,
                write_index,
                v_cast[:, :, :first_len, :],
            )
        remaining = L - first_len
        if remaining > 0:
            self.hot_k = self._write_hot_segment(
                self.hot_k,
                0,
                k_cast[:, :, first_len:, :],
            )
            self.hot_v = self._write_hot_segment(
                self.hot_v,
                0,
                v_cast[:, :, first_len:, :],
            )
        self.hot_seq_len = end

    def evict_for_append(self, codec: HybridKeyCodec, incoming_len: int) -> None:
        """Make room in the hot buffer for ``incoming_len`` new tokens.

        Pass 6: called by the layer BEFORE ``append_exact`` in COMPRESSED
        mode. Evicts at least ``block_size_seq`` tokens if the buffer
        would overflow, ensuring the fragmentation fix always fires.

        Parameters
        ----------
        codec : HybridKeyCodec
        incoming_len : int
            Number of tokens about to be appended.
        """
        projected = self.hot_seq_len + incoming_len
        while projected > self.config.hot_capacity and self.hot_seq_len > 0:
            need = projected - self.config.hot_capacity
            chunk = max(need, self.config.block_size_seq)
            chunk = min(chunk, self.hot_seq_len)  # defensive
            if chunk > 0:
                self._evict_hot_to_warm(codec, chunk)
            projected = self.hot_seq_len + incoming_len

    def _evict_hot_to_warm(self, codec: HybridKeyCodec, prefix_len: int) -> None:
        """Evict the oldest ``prefix_len`` tokens from hot into warm."""
        if prefix_len <= 0 or self.hot_seq_len == 0:
            return
        remaining = min(prefix_len, self.hot_seq_len)
        while remaining > 0:
            chunk = self.config.block_size_seq if remaining > self.config.block_size_seq else remaining
            self._archive_hot_prefix(codec, chunk)
            remaining -= chunk

    def _evict_warm_to_cold(self) -> None:
        """Move the oldest warm block to cold storage on disk."""
        if not self.warm_blocks:
            return
        block = self.warm_blocks.pop(0)
        if block.path is not None:
            self.cold_blocks.append(block)
            return
        if block.v_payload is None:
            return
        block_id = len(self.cold_blocks)
        path = self.cache_dir / f"layer_{id(self)}_block_{block_id}_{block.start_pos}.npz"
        np.savez_compressed(
            path,
            start_pos=block.encoded_k.start_pos,
            end_pos=block.encoded_k.end_pos,
            pq_codes=np.array(block.encoded_k.pq_codes),
            v=block.v_payload,
            rvq_flat_indices=np.array(block.encoded_k.rvq_flat_indices, dtype=np.int32),
            rvq_codes=np.array(block.encoded_k.rvq_codes, dtype=np.uint16),
            rvq_mask=np.array(block.encoded_k.rvq_mask, dtype=np.bool_),
        )
        block.path = str(path)
        block.tier = CacheTier.COLD
        self.cold_blocks.append(block)
        self._mark_archived_context_dirty()

    def maybe_evict(self, codec: HybridKeyCodec) -> None:
        """Perform bounded eviction from warm and cold tiers.

        Note: hot-tier eviction is now handled by ``evict_for_append``
        which is called BEFORE ``append_exact`` in COMPRESSED mode.
        This method handles warm->cold spill and cold_capacity warnings.
        """
        warm_len = sum(b.end_pos - b.start_pos for b in self.warm_blocks)
        while warm_len > self.config.warm_capacity and self.warm_blocks:
            self._evict_warm_to_cold()
            warm_len = sum(b.end_pos - b.start_pos for b in self.warm_blocks)

        # Pass 6: cold_capacity enforcement (warn; full drop is a future pass)
        cold_len = sum(b.end_pos - b.start_pos for b in self.cold_blocks)
        if cold_len > self.config.cold_capacity:
            warnings.warn(
                f"Cold tier length {cold_len} exceeds cold_capacity "
                f"{self.config.cold_capacity}. "
                "Oldest cold blocks should be dropped to enforce the limit. "
                "Full cold eviction is not yet implemented.",
                RuntimeWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Materialisation helpers
    # ------------------------------------------------------------------

    def materialize_exact_context(self) -> Tuple[mx.array, mx.array, int, int]:
        """Return the valid slice of the hot buffer as (k, v, start, end)."""
        if self.hot_seq_len == 0:
            return self._empty_context()
        return self._materialize_hot_span(logical_start=0, length=self.hot_seq_len)

    def materialize_archived_context(
        self, codec: HybridKeyCodec
    ) -> Tuple[mx.array, mx.array, int, int]:
        """Return cached archived context across cold and warm tiers."""
        if not (self.warm_blocks or self.cold_blocks):
            raise RuntimeError("No archived context available")
        if (
            not self._archived_context_dirty
            and self._archived_k is not None
            and self._archived_v is not None
            and self._archived_start is not None
            and self._archived_end is not None
        ):
            return (
                self._archived_k,
                self._archived_v,
                self._archived_start,
                self._archived_end,
            )

        decoded_ks: List[mx.array] = []
        decoded_vs: List[mx.array] = []
        start_positions: List[int] = []

        def _decode_block(block: ArchivedKVBlock) -> Tuple[mx.array, mx.array]:
            if block.v_payload is None and block.path:
                data = np.load(block.path, allow_pickle=True)
                block.encoded_k = EncodedKeyBlock(
                    start_pos=int(data["start_pos"]),
                    end_pos=int(data["end_pos"]),
                    pq_codes=mx.array(data["pq_codes"], dtype=mx.uint8),
                    rvq_flat_indices=mx.array(data["rvq_flat_indices"], dtype=mx.int32),
                    rvq_codes=mx.array(data["rvq_codes"], dtype=mx.uint16),
                    rvq_mask=mx.array(data["rvq_mask"], dtype=mx.bool_),
                )
                block.v_payload = data["v"]
                block.loaded = True

            k_decoded = codec.decode_keys(
                block.encoded_k,
                batch_size=self.batch_size,
                num_heads=self.num_kv_heads,
            ).astype(self.dtype)
            v_tensor = self._decode_archived_v_payload(block.v_payload)
            return k_decoded, v_tensor

        for block in self.cold_blocks + self.warm_blocks:
            k_dec, v_dec = _decode_block(block)
            decoded_ks.append(k_dec)
            decoded_vs.append(v_dec)
            start_positions.append(block.start_pos)

        blocks = list(zip(start_positions, decoded_ks, decoded_vs))
        blocks.sort(key=lambda item: item[0])
        for i in range(1, len(blocks)):
            prev_start, prev_k, _ = blocks[i - 1]
            cur_start, _, _ = blocks[i]
            prev_end = prev_start + prev_k.shape[2]
            if cur_start != prev_end:
                raise RuntimeError(
                    f"Non-contiguous archive: previous block end {prev_end}, "
                    f"next block start {cur_start}"
                )

        self._archived_k = blocks[0][1] if len(blocks) == 1 else mx.concatenate(
            [block[1] for block in blocks], axis=2
        )
        self._archived_v = blocks[0][2] if len(blocks) == 1 else mx.concatenate(
            [block[2] for block in blocks], axis=2
        )
        self._archived_start = blocks[0][0]
        self._archived_end = self._archived_start + self._archived_k.shape[2]
        self._archived_context_dirty = False
        self._clear_block_reconstruction_cache()
        return self._archived_k, self._archived_v, self._archived_start, self._archived_end

    def get_mixed_attention_segments(
        self,
        codec: HybridKeyCodec,
    ) -> List[Tuple[mx.array, mx.array, int]]:
        segments: List[Tuple[mx.array, mx.array, int]] = []
        if self.warm_blocks or self.cold_blocks:
            archived_k, archived_v, archived_start, _ = self.materialize_archived_context(codec)
            segments.append((archived_k, archived_v, archived_start))
        segments.extend(self.get_hot_attention_segments())
        return segments

    def materialize_mixed_context(
        self, codec: HybridKeyCodec
    ) -> Tuple[mx.array, mx.array, int, int]:
        """Reconstruct full context from cold + warm + hot tiers."""
        segments = self.get_mixed_attention_segments(codec)
        if not segments:
            raise RuntimeError("No context available for mixed materialisation")

        for i in range(1, len(segments)):
            prev_start = segments[i - 1][2]
            prev_end = prev_start + segments[i - 1][0].shape[2]
            cur_start = segments[i][2]
            if cur_start != prev_end:
                raise RuntimeError(
                    f"Non-contiguous archive: previous block end {prev_end}, "
                    f"next block start {cur_start}"
                )

        full_start = segments[0][2]
        full_k = segments[0][0] if len(segments) == 1 else mx.concatenate(
            [segment[0] for segment in segments], axis=2
        )
        full_v = segments[0][1] if len(segments) == 1 else mx.concatenate(
            [segment[1] for segment in segments], axis=2
        )
        full_end = full_start + full_k.shape[2]
        return full_k, full_v, full_start, full_end

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def get_total_length(self) -> int:
        """Total number of tokens stored across all tiers."""
        warm_len = sum(b.end_pos - b.start_pos for b in self.warm_blocks)
        cold_len = sum(b.end_pos - b.start_pos for b in self.cold_blocks)
        return self.hot_seq_len + warm_len + cold_len

    def invalidate_reconstructed_cache(self) -> None:
        """Clear cached decoded tensors on all archived blocks."""
        self._mark_archived_context_dirty()


class RFSNCache:
    """Wrapper for per-layer caches."""

    def __init__(self, config: RFSNConfig, batch_size: int) -> None:
        self.layers = [
            LayerKVCache(
                config,
                batch_size=batch_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
            )
            for _ in range(config.num_layers)
        ]

    def layer(self, idx: int) -> LayerKVCache:
        return self.layers[idx]
