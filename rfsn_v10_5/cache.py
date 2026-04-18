"""Exact per-layer KV cache with block-managed archival storage.

Phase 1 replaces the old archive-and-reconstruct design with a single
exact path:

- the hot tier is a rolling exact window in device memory,
- sealed prefixes become exact archived blocks registered in a
  ``BlockManager``,
- warm archived blocks stay in RAM,
- cold archived blocks live only on disk and are loaded lazily,
- attention consumes narrow segment views rather than a monolithic
  reconstructed archive tensor.

The runtime authority for archived context is the page table owned by
``BlockManager``. Any concatenation helpers in this module are debug-only
views built on top of that metadata and are not used by the hot path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Optional
import uuid

import numpy as np
import mlx.core as mx

from .attention_exact import AttentionSegment
from .block_manager import BlockId, BlockLocation, BlockManager, BlockManifest, BlockSpan
from .config import RFSNConfig, RuntimeMode, resolve_dtype, validate_session_id
from .residency import ResidencyManager
from .storage import BlockStorage


_EXACT_CODEC_VERSION = "v11-exact-kv"


def derive_model_id(config: RFSNConfig) -> str:
    """Derive a deterministic model identity from stable config fields."""
    payload = {
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "num_layers": config.num_layers,
        "vocab_size": config.vocab_size,
        "rope_base": config.rope_base,
        "ffn_dim": config.ffn_dim,
        "norm_eps": config.norm_eps,
        "model_dtype": config.model_dtype,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _resolve_session_id(
    config: RFSNConfig,
    session_id: Optional[str],
    *,
    restore: bool,
) -> str:
    candidate = validate_session_id(session_id if session_id is not None else config.session_id)
    if candidate:
        return candidate
    if restore:
        raise ValueError(
            "restore_cache requires an explicit session_id so the persisted archive can be resolved safely"
        )
    return uuid.uuid4().hex


class LayerKVCache:
    """Exact KV cache for one transformer layer."""

    def __init__(
        self,
        config: RFSNConfig,
        batch_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        *,
        layer_id: int = 0,
        model_id: Optional[str] = None,
        session_id: str,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.layer_id = layer_id
        self.model_id = model_id or derive_model_id(config)
        self.session_id = session_id
        self.dtype = resolve_dtype(config.model_dtype)
        self.storage_array_dtype = mx.float32 if self.dtype == mx.bfloat16 else self.dtype

        head_dim = config.head_dim
        self.hot_k: mx.array = mx.zeros(
            (batch_size, self.num_kv_heads, config.hot_capacity, head_dim),
            dtype=self.dtype,
        )
        self.hot_v: mx.array = mx.zeros(
            (batch_size, self.num_kv_heads, config.hot_capacity, head_dim),
            dtype=self.dtype,
        )
        self.hot_seq_len: int = 0
        self.hot_start: int = 0
        self.hot_head_index: int = 0

        self.block_manager = BlockManager(self.model_id)
        layer_storage_dir = Path(config.disk_cache_dir) / self.model_id / self.session_id / f"layer_{layer_id}"
        self.storage = BlockStorage(layer_storage_dir)
        self.residency_manager = ResidencyManager(
            prefetch_margin_tokens=max(1, config.block_size_seq // 2),
        )
        self._resident_blocks: dict[BlockId, tuple[mx.array, mx.array]] = {}
        self._block_serial: int = 0

    @property
    def model_root(self) -> Path:
        return Path(self.config.disk_cache_dir) / self.model_id

    @property
    def session_root(self) -> Path:
        return self.model_root / self.session_id

    @property
    def hot_write_index(self) -> int:
        return (self.hot_head_index + self.hot_seq_len) % self.config.hot_capacity

    @property
    def hot_end(self) -> int:
        return self.hot_start + self.hot_seq_len

    def reset(self, *, clear_persisted: bool = True) -> None:
        self.hot_k = mx.zeros(self.hot_k.shape, dtype=self.dtype)
        self.hot_v = mx.zeros(self.hot_v.shape, dtype=self.dtype)
        self.hot_seq_len = 0
        self.hot_start = 0
        self.hot_head_index = 0
        self.block_manager.clear()
        self._resident_blocks.clear()
        self._block_serial = 0
        self.residency_manager.reset()
        if clear_persisted:
            shutil.rmtree(self.storage.root_dir, ignore_errors=True)
            self.storage.root_dir.mkdir(parents=True, exist_ok=True)
            self.storage.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def restore_from_disk(self) -> list[BlockManifest]:
        self.hot_k = mx.zeros(self.hot_k.shape, dtype=self.dtype)
        self.hot_v = mx.zeros(self.hot_v.shape, dtype=self.dtype)
        self.hot_seq_len = 0
        self.hot_head_index = 0
        self.hot_start = 0
        self.block_manager.clear()
        self._resident_blocks.clear()
        manifests = self.storage.rebuild_manager(self.block_manager)
        contiguous_end = 0
        for manifest in manifests:
            if manifest.residency == BlockLocation.MISSING or not manifest.materializable:
                raise RuntimeError(
                    f"Persisted archive for session '{self.session_id}' contains an unreadable block "
                    f"at layer {self.layer_id}; restore cannot safely continue"
                )
            if manifest.logical_start != contiguous_end:
                raise RuntimeError(
                    f"Persisted archive for session '{self.session_id}' contains a gap at layer "
                    f"{self.layer_id}: expected {contiguous_end}, found {manifest.logical_start}"
                )
            contiguous_end = manifest.logical_end
        self.hot_start = contiguous_end
        return manifests

    def _empty_context(self) -> tuple[mx.array, mx.array, int, int]:
        empty = mx.zeros(
            (self.batch_size, self.num_kv_heads, 0, self.config.head_dim),
            dtype=self.dtype,
        )
        return empty, empty, self.hot_start, self.hot_start

    def _hot_span_segments(
        self,
        logical_start: int = 0,
        length: Optional[int] = None,
    ) -> list[AttentionSegment]:
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
        segments: list[AttentionSegment] = []
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
    ) -> tuple[mx.array, mx.array, int, int]:
        segments = self._hot_span_segments(logical_start=logical_start, length=length)
        if not segments:
            return self._empty_context()
        ks = [segment[0] for segment in segments]
        vs = [segment[1] for segment in segments]
        start = int(segments[0][2])
        total_len = sum(segment[0].shape[2] for segment in segments)
        hot_k = ks[0] if len(ks) == 1 else mx.concatenate(ks, axis=2)
        hot_v = vs[0] if len(vs) == 1 else mx.concatenate(vs, axis=2)
        return hot_k, hot_v, start, start + total_len

    def _next_block_manifest(self, start: int, end: int, dtype_name: str) -> BlockManifest:
        block_id = BlockId(self.model_id, self.layer_id, f"block-{self._block_serial:08d}")
        self._block_serial += 1
        token_count = end - start
        shape = (self.batch_size, self.num_kv_heads, token_count, self.config.head_dim)
        return BlockManifest(
            block_id=block_id,
            span=BlockSpan(start, end),
            codec_version=_EXACT_CODEC_VERSION,
            dtype=dtype_name,
            shape_metadata={"keys": shape, "values": shape},
            residency=BlockLocation.WARM_RAM,
        )

    def _write_hot_segment(self, buffer: mx.array, start_index: int, values: mx.array) -> mx.array:
        indices = mx.broadcast_to(
            mx.arange(start_index, start_index + values.shape[2], dtype=mx.int32)[None, None, :, None],
            values.shape,
        )
        return mx.put_along_axis(buffer, indices, values, axis=2)

    def _warm_manifests(self) -> list[BlockManifest]:
        return [
            manifest
            for manifest in self.block_manager.iter_blocks(layer_id=self.layer_id)
            if manifest.residency == BlockLocation.WARM_RAM
        ]

    def _cold_manifests(self) -> list[BlockManifest]:
        return [
            manifest
            for manifest in self.block_manager.iter_blocks(layer_id=self.layer_id)
            if manifest.residency == BlockLocation.COLD_DISK
        ]

    def _warm_token_count(self) -> int:
        return sum(manifest.token_count for manifest in self._warm_manifests())

    def _resident_payload_to_numpy(self, tensor: mx.array) -> np.ndarray:
        return np.asarray(tensor.astype(self.storage_array_dtype))

    def _storage_dtype_name(self, tensor: mx.array) -> str:
        return self._resident_payload_to_numpy(tensor).dtype.name

    def _spill_manifest_to_disk(self, manifest: BlockManifest) -> None:
        resident = self._resident_blocks.pop(manifest.block_id, None)
        if resident is None:
            self.block_manager.mark_unmaterializable(
                manifest.block_id,
                reason="warm payload missing from resident set",
            )
            return

        keys, values = resident
        payload = {
            "keys": self._resident_payload_to_numpy(keys),
            "values": self._resident_payload_to_numpy(values),
        }
        manifest.dtype = payload["keys"].dtype.name
        self.storage.persist_block(manifest, payload)

    def demote_manifest_to_cold(self, manifest: BlockManifest) -> None:
        self._spill_manifest_to_disk(manifest)

    def _spill_warm_if_needed(self) -> None:
        self.residency_manager.evict_warm_excess(self)

    def is_manifest_resident(self, manifest: BlockManifest) -> bool:
        return manifest.block_id in self._resident_blocks

    def load_manifest_payload_only(self, manifest: BlockManifest) -> Optional[dict[str, np.ndarray]]:
        return self.storage.load_block(manifest)

    def promote_manifest_from_payload(
        self,
        manifest: BlockManifest,
        payload: dict[str, np.ndarray],
    ) -> None:
        self._resident_blocks[manifest.block_id] = (
            mx.array(payload["keys"], dtype=self.dtype),
            mx.array(payload["values"], dtype=self.dtype),
        )
        manifest.residency = BlockLocation.WARM_RAM
        manifest.touch()

    def maybe_prefetch_for_decode(self, query_abs_pos: int) -> None:
        self.residency_manager.maybe_schedule_prefetch(self, query_abs_pos)

    def _seal_hot_prefix(self, prefix_len: int) -> None:
        hot_k, hot_v, prefix_start, prefix_end = self._materialize_hot_span(
            logical_start=0,
            length=prefix_len,
        )
        manifest = self._next_block_manifest(
            prefix_start,
            prefix_end,
            self._storage_dtype_name(hot_k),
        )
        self.block_manager.register_block(manifest)
        self._resident_blocks[manifest.block_id] = (hot_k, hot_v)
        self.hot_head_index = (self.hot_head_index + prefix_len) % self.config.hot_capacity
        self.hot_start = prefix_end
        self.hot_seq_len -= prefix_len
        self._spill_warm_if_needed()

    def append_exact(self, k: mx.array, v: mx.array) -> None:
        _, _, seq_len, _ = k.shape
        end = self.hot_seq_len + seq_len
        if end > self.config.hot_capacity:
            raise RuntimeError(
                f"append_exact: hot buffer overflow ({end} > {self.config.hot_capacity}). "
                "Chunk prefill should evict before append."
            )

        k_cast = k.astype(self.dtype)
        v_cast = v.astype(self.dtype)
        write_index = self.hot_write_index
        first_len = min(seq_len, self.config.hot_capacity - write_index)
        if first_len > 0:
            self.hot_k = self._write_hot_segment(self.hot_k, write_index, k_cast[:, :, :first_len, :])
            self.hot_v = self._write_hot_segment(self.hot_v, write_index, v_cast[:, :, :first_len, :])
        remaining = seq_len - first_len
        if remaining > 0:
            self.hot_k = self._write_hot_segment(self.hot_k, 0, k_cast[:, :, first_len:, :])
            self.hot_v = self._write_hot_segment(self.hot_v, 0, v_cast[:, :, first_len:, :])
        self.hot_seq_len = end

    def evict_for_append(self, incoming_len: int) -> None:
        if incoming_len > self.config.hot_capacity:
            raise RuntimeError(
                f"Chunk length {incoming_len} exceeds hot_capacity={self.config.hot_capacity}. "
                "Call model.prefill() with chunking rather than appending oversized segments."
            )

        projected = self.hot_seq_len + incoming_len
        if self.config.runtime_mode == RuntimeMode.EXACT and projected > self.config.hot_capacity:
            raise RuntimeError(
                "runtime_mode='exact' keeps all live context in the hot window; "
                f"the requested append would exceed hot_capacity={self.config.hot_capacity}"
            )
        while projected > self.config.hot_capacity and self.hot_seq_len > 0:
            need = projected - self.config.hot_capacity
            chunk = max(need, self.config.block_size_seq)
            chunk = min(chunk, self.hot_seq_len)
            if chunk <= 0:
                break
            self._seal_hot_prefix(chunk)
            projected = self.hot_seq_len + incoming_len

    def _load_archived_payload(self, manifest: BlockManifest) -> Optional[tuple[mx.array, mx.array]]:
        self.residency_manager.drain_completed_prefetches(self)
        resident = self._resident_blocks.get(manifest.block_id)
        if resident is not None:
            manifest.touch()
            return resident

        if manifest.residency == BlockLocation.WARM_RAM:
            self.block_manager.mark_unmaterializable(
                manifest.block_id,
                reason="warm payload missing from resident set",
            )
            return None

        if manifest.residency == BlockLocation.COLD_DISK:
            self.residency_manager.note_sync_load()
            payload = self.load_manifest_payload_only(manifest)
            if payload is None:
                return None
            self.promote_manifest_from_payload(manifest, payload)
            self.residency_manager.evict_warm_excess(self, protected_block_id=manifest.block_id)
            return self._resident_blocks[manifest.block_id]

        return None

    def get_hot_attention_segments(self) -> list[AttentionSegment]:
        return list(self._hot_span_segments())

    def get_archived_attention_segments(self) -> list[AttentionSegment]:
        self.residency_manager.drain_completed_prefetches(self)
        segments: list[AttentionSegment] = []
        for manifest in self.block_manager.iter_blocks(layer_id=self.layer_id):
            if manifest.residency == BlockLocation.MISSING or not manifest.materializable:
                continue
            payload = self._load_archived_payload(manifest)
            if payload is None:
                continue
            segments.append((payload[0], payload[1], manifest.logical_start))
        return segments

    def get_attention_segments(self) -> list[AttentionSegment]:
        segments = self.get_archived_attention_segments()
        segments.extend(self.get_hot_attention_segments())
        return segments

    def get_mixed_attention_segments(self, _unused: Any = None) -> list[AttentionSegment]:
        return self.get_attention_segments()

    def _concatenate_segments(
        self,
        segments: list[AttentionSegment],
        *,
        require_contiguous: bool,
    ) -> tuple[mx.array, mx.array, int, int]:
        if not segments:
            raise RuntimeError("No context available")
        ordered = sorted(segments, key=lambda segment: segment[2])
        if require_contiguous:
            for previous, current in zip(ordered, ordered[1:]):
                previous_end = previous[2] + previous[0].shape[2]
                if current[2] != previous_end:
                    raise RuntimeError(
                        f"Non-contiguous context: previous block end {previous_end}, next block start {current[2]}"
                    )
        start = int(ordered[0][2])
        full_k = ordered[0][0] if len(ordered) == 1 else mx.concatenate([segment[0] for segment in ordered], axis=2)
        full_v = ordered[0][1] if len(ordered) == 1 else mx.concatenate([segment[1] for segment in ordered], axis=2)
        return full_k, full_v, start, start + full_k.shape[2]

    def materialize_exact_context(self) -> tuple[mx.array, mx.array, int, int]:
        if self.hot_seq_len == 0:
            return self._empty_context()
        return self._materialize_hot_span(logical_start=0, length=self.hot_seq_len)

    def materialize_archived_context(self, _unused: Any = None) -> tuple[mx.array, mx.array, int, int]:
        return self._concatenate_segments(
            self.get_archived_attention_segments(),
            require_contiguous=True,
        )

    def materialize_mixed_context(self, _unused: Any = None) -> tuple[mx.array, mx.array, int, int]:
        return self._concatenate_segments(
            self.get_attention_segments(),
            require_contiguous=True,
        )

    def get_total_length(self) -> int:
        archived = sum(
            manifest.token_count
            for manifest in self.block_manager.iter_blocks(layer_id=self.layer_id)
            if manifest.residency != BlockLocation.MISSING and manifest.materializable
        )
        return self.hot_seq_len + archived

    def get_block_stats(self) -> dict[str, Any]:
        stats = self.block_manager.get_residency_stats()
        stats["hot_tokens"] = self.hot_seq_len
        stats["hot_start"] = self.hot_start
        stats["hot_end"] = self.hot_end
        stats["residency_metrics"] = self.residency_manager.get_metrics()
        return stats


class RFSNCache:
    """Wrapper for per-layer exact caches."""

    def __init__(
        self,
        config: RFSNConfig,
        batch_size: int,
        *,
        model_id: Optional[str] = None,
        session_id: Optional[str] = None,
        restore: bool = False,
    ) -> None:
        self.config = config
        self.model_id = model_id or derive_model_id(config)
        self.session_id = _resolve_session_id(config, session_id, restore=restore)
        self.layers = [
            LayerKVCache(
                config,
                batch_size=batch_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                layer_id=layer_index,
                model_id=self.model_id,
                session_id=self.session_id,
            )
            for layer_index in range(config.num_layers)
        ]

    @property
    def model_root(self) -> Path:
        return Path(self.config.disk_cache_dir) / self.model_id

    @property
    def session_root(self) -> Path:
        return self.model_root / self.session_id

    def layer(self, idx: int) -> LayerKVCache:
        return self.layers[idx]

    def reset(self, *, clear_persisted: bool = True) -> None:
        for layer_cache in self.layers:
            layer_cache.reset(clear_persisted=clear_persisted)

    def restore_from_disk(self) -> list[list[BlockManifest]]:
        if not self.session_id:
            raise ValueError(
                "restore_cache requires an explicit session_id so the persisted archive can be resolved safely"
            )
        if not self.model_root.exists():
            raise FileNotFoundError(
                f"Restore requested for session '{self.session_id}' but no persisted archive exists "
                f"for model '{self.model_id}'"
            )
        if not self.session_root.exists():
            known_sessions = sorted(path.name for path in self.model_root.iterdir() if path.is_dir())
            if known_sessions:
                raise FileNotFoundError(
                    f"Restore requested for unknown session '{self.session_id}' for model '{self.model_id}'. "
                    f"Available sessions: {', '.join(known_sessions)}"
                )
            raise FileNotFoundError(
                f"Restore requested for session '{self.session_id}' but no persisted archive exists "
                f"for model '{self.model_id}'"
            )

        restored = [layer_cache.restore_from_disk() for layer_cache in self.layers]
        restored_block_count = sum(len(layer_manifests) for layer_manifests in restored)
        if restored_block_count == 0:
            raise FileNotFoundError(
                f"Restore requested for session '{self.session_id}' but the persisted archive is empty"
            )
        return restored
