"""Simple residency and prefetch policy for exact KV blocks.

Phase 2 keeps the policy explicit and intentionally conservative:

- choose warm-RAM eviction victims by oldest access time,
- keep a bounded warm set,
- prefetch one adjacent cold block for sequential decode,
- surface metrics for evictions, prefetches, and failure states.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .block_manager import BlockId, BlockLocation, BlockManifest


if TYPE_CHECKING:
    from .cache import LayerKVCache


@dataclass
class ResidencyMetrics:
    warm_evictions: int = 0
    prefetch_requests: int = 0
    prefetch_completions: int = 0
    sync_loads: int = 0
    missing_blocks: int = 0
    unmaterializable_blocks: int = 0


class ResidencyManager:
    """Manage warm-set sizing and simple sequential prefetch."""

    def __init__(self, *, prefetch_margin_tokens: int = 1) -> None:
        self.prefetch_margin_tokens = max(1, prefetch_margin_tokens)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rfsn-prefetch")
        self._prefetch_futures: dict[BlockId, Future[Optional[dict[str, np.ndarray]]]] = {}
        self.metrics = ResidencyMetrics()

    def reset(self) -> None:
        for future in self._prefetch_futures.values():
            future.cancel()
        self._prefetch_futures.clear()
        self.metrics = ResidencyMetrics()

    def choose_warm_eviction_victim(
        self,
        manifests: list[BlockManifest],
        *,
        protected_block_id: Optional[BlockId] = None,
    ) -> Optional[BlockManifest]:
        eligible = [
            manifest
            for manifest in manifests
            if manifest.residency == BlockLocation.WARM_RAM
            and manifest.materializable
            and manifest.pin_count == 0
            and manifest.block_id != protected_block_id
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda manifest: (manifest.last_accessed_at, manifest.logical_start))

    def evict_warm_excess(
        self,
        cache: "LayerKVCache",
        *,
        protected_block_id: Optional[BlockId] = None,
    ) -> None:
        while cache._warm_token_count() > cache.config.warm_capacity:
            victim = self.choose_warm_eviction_victim(
                cache._warm_manifests(),
                protected_block_id=protected_block_id,
            )
            if victim is None:
                return
            cache.demote_manifest_to_cold(victim)
            self.metrics.warm_evictions += 1

    def maybe_schedule_prefetch(self, cache: "LayerKVCache", query_abs_pos: int) -> None:
        self.drain_completed_prefetches(cache)
        manifests = [
            manifest
            for manifest in cache.block_manager.iter_blocks(layer_id=cache.layer_id)
            if manifest.materializable
        ]
        if len(manifests) < 2:
            return

        candidate: Optional[BlockManifest] = None
        for current, next_manifest in zip(manifests, manifests[1:]):
            if current.residency == BlockLocation.WARM_RAM and next_manifest.residency == BlockLocation.COLD_DISK:
                if query_abs_pos >= current.logical_end - self.prefetch_margin_tokens:
                    candidate = next_manifest
                    break

        if candidate is None:
            for previous, current in zip(manifests, manifests[1:]):
                if previous.residency == BlockLocation.COLD_DISK and current.residency == BlockLocation.WARM_RAM:
                    if query_abs_pos >= current.logical_end - self.prefetch_margin_tokens:
                        candidate = previous
                        break

        if candidate is None:
            return
        if candidate.block_id in self._prefetch_futures:
            return
        if cache.is_manifest_resident(candidate):
            return

        self._prefetch_futures[candidate.block_id] = self._executor.submit(
            cache.load_manifest_payload_only,
            candidate,
        )
        self.metrics.prefetch_requests += 1

    def drain_completed_prefetches(self, cache: "LayerKVCache") -> None:
        for block_id, future in list(self._prefetch_futures.items()):
            if not future.done():
                continue

            self._prefetch_futures.pop(block_id, None)
            manifest = cache.block_manager.get_block(block_id)
            try:
                payload = future.result()
            except Exception as exc:
                cache.block_manager.mark_missing(block_id, reason=f"prefetch failed: {exc}")
                self.metrics.missing_blocks += 1
                continue

            if payload is None:
                if manifest.residency == BlockLocation.MISSING:
                    self.metrics.missing_blocks += 1
                if not manifest.materializable:
                    self.metrics.unmaterializable_blocks += 1
                continue

            cache.promote_manifest_from_payload(manifest, payload)
            self.evict_warm_excess(cache, protected_block_id=manifest.block_id)
            self.metrics.prefetch_completions += 1

    def wait_for_prefetches(self, cache: "LayerKVCache", *, timeout: float = 1.0) -> None:
        for future in list(self._prefetch_futures.values()):
            future.result(timeout=timeout)
        self.drain_completed_prefetches(cache)

    def note_sync_load(self) -> None:
        self.metrics.sync_loads += 1

    def get_metrics(self) -> dict[str, int]:
        return asdict(self.metrics)
