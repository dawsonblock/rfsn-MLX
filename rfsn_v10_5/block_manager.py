"""Block metadata and page-table authority for RFSN-MLX V11.

This module introduces the inspectable source of truth for archived
context. Runtime code will move to these abstractions in later phases,
but Phase 0 focuses on metadata correctness, serialization, and restart
rebuild behavior.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Iterable, Optional


class BlockLocation(str, Enum):
    """Explicit residency state for a physical KV block."""

    HOT = "hot"
    WARM_RAM = "warm_ram"
    COLD_DISK = "cold_disk"
    MISSING = "missing"


@dataclass(frozen=True, order=True)
class BlockId:
    """Stable identity for a persisted or resident block."""

    model_id: str
    layer_id: int
    block_id: str

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("BlockId.model_id must be non-empty")
        if self.layer_id < 0:
            raise ValueError("BlockId.layer_id must be >= 0")
        if not self.block_id:
            raise ValueError("BlockId.block_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "layer_id": self.layer_id,
            "block_id": self.block_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockId":
        return cls(
            model_id=str(payload["model_id"]),
            layer_id=int(payload["layer_id"]),
            block_id=str(payload["block_id"]),
        )


@dataclass(frozen=True, order=True)
class BlockSpan:
    """Logical token span covered by a physical block."""

    logical_start: int
    logical_end: int

    def __post_init__(self) -> None:
        if self.logical_start < 0:
            raise ValueError("BlockSpan.logical_start must be >= 0")
        if self.logical_end <= self.logical_start:
            raise ValueError("BlockSpan.logical_end must be greater than logical_start")

    @property
    def token_count(self) -> int:
        return self.logical_end - self.logical_start

    def overlaps(self, start: int, end: int) -> bool:
        return self.logical_start < end and start < self.logical_end

    def to_dict(self) -> dict[str, int]:
        return {
            "logical_start": self.logical_start,
            "logical_end": self.logical_end,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockSpan":
        return cls(
            logical_start=int(payload["logical_start"]),
            logical_end=int(payload["logical_end"]),
        )


@dataclass
class BlockManifest:
    """Inspectable metadata for one physical KV block."""

    block_id: BlockId
    span: BlockSpan
    codec_version: str
    dtype: str
    shape_metadata: dict[str, tuple[int, ...]]
    checksum: str = ""
    residency: BlockLocation = BlockLocation.HOT
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    pin_count: int = 0
    ref_count: int = 0
    payload_format: str = "npz"
    payload_path: Optional[str] = None
    manifest_path: Optional[str] = None
    materializable: bool = True
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.token_count != self.span.token_count:
            raise ValueError("BlockManifest token_count must match span length")
        if not self.codec_version:
            raise ValueError("BlockManifest.codec_version must be non-empty")
        if not self.dtype:
            raise ValueError("BlockManifest.dtype must be non-empty")
        if self.pin_count < 0:
            raise ValueError("BlockManifest.pin_count must be >= 0")
        if self.ref_count < 0:
            raise ValueError("BlockManifest.ref_count must be >= 0")
        if self.payload_format not in {"npz", "safetensors"}:
            raise ValueError("payload_format must be 'npz' or 'safetensors'")
        normalized_shapes: dict[str, tuple[int, ...]] = {}
        for name, shape in self.shape_metadata.items():
            normalized_shapes[str(name)] = tuple(int(dim) for dim in shape)
        self.shape_metadata = normalized_shapes

    @property
    def model_id(self) -> str:
        return self.block_id.model_id

    @property
    def layer_id(self) -> int:
        return self.block_id.layer_id

    @property
    def logical_start(self) -> int:
        return self.span.logical_start

    @property
    def logical_end(self) -> int:
        return self.span.logical_end

    @property
    def token_count(self) -> int:
        return self.span.token_count

    def touch(self, *, timestamp: Optional[float] = None) -> None:
        self.last_accessed_at = time.time() if timestamp is None else float(timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.block_id.to_dict(),
            **self.span.to_dict(),
            "codec_version": self.codec_version,
            "dtype": self.dtype,
            "shape_metadata": {
                name: list(shape) for name, shape in sorted(self.shape_metadata.items())
            },
            "checksum": self.checksum,
            "residency": self.residency.value,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "pin_count": self.pin_count,
            "ref_count": self.ref_count,
            "payload_format": self.payload_format,
            "payload_path": self.payload_path,
            "manifest_path": self.manifest_path,
            "materializable": self.materializable,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockManifest":
        return cls(
            block_id=BlockId.from_dict(payload),
            span=BlockSpan.from_dict(payload),
            codec_version=str(payload["codec_version"]),
            dtype=str(payload["dtype"]),
            shape_metadata={
                str(name): tuple(int(dim) for dim in shape)
                for name, shape in dict(payload.get("shape_metadata", {})).items()
            },
            checksum=str(payload.get("checksum", "")),
            residency=BlockLocation(str(payload.get("residency", BlockLocation.HOT.value))),
            created_at=float(payload.get("created_at", time.time())),
            last_accessed_at=float(payload.get("last_accessed_at", time.time())),
            pin_count=int(payload.get("pin_count", 0)),
            ref_count=int(payload.get("ref_count", 0)),
            payload_format=str(payload.get("payload_format", "npz")),
            payload_path=payload.get("payload_path"),
            manifest_path=payload.get("manifest_path"),
            materializable=bool(payload.get("materializable", True)),
            failure_reason=payload.get("failure_reason"),
        )


@dataclass
class PageTable:
    """Sorted logical-span index over physical block manifests."""

    _entries_by_layer: dict[int, list[BlockManifest]] = field(default_factory=dict)

    def register(self, manifest: BlockManifest) -> None:
        entries = self._entries_by_layer.setdefault(manifest.layer_id, [])
        starts = [entry.logical_start for entry in entries]
        insert_at = bisect_left(starts, manifest.logical_start)

        if insert_at > 0:
            previous = entries[insert_at - 1]
            if previous.logical_end > manifest.logical_start:
                raise ValueError(
                    "Block span overlaps previous page-table entry: "
                    f"{previous.span} vs {manifest.span}"
                )
        if insert_at < len(entries):
            following = entries[insert_at]
            if manifest.logical_end > following.logical_start:
                raise ValueError(
                    "Block span overlaps next page-table entry: "
                    f"{manifest.span} vs {following.span}"
                )

        entries.insert(insert_at, manifest)

    def remove(self, block_id: BlockId) -> None:
        entries = self._entries_by_layer.get(block_id.layer_id, [])
        for index, manifest in enumerate(entries):
            if manifest.block_id == block_id:
                del entries[index]
                if not entries:
                    self._entries_by_layer.pop(block_id.layer_id, None)
                return
        raise KeyError(f"Unknown block_id: {block_id}")

    def locate(self, layer_id: int, start: int, end: int) -> list[BlockManifest]:
        if start < 0 or end <= start:
            raise ValueError("locate() requires 0 <= start < end")
        return [
            manifest
            for manifest in self._entries_by_layer.get(layer_id, [])
            if manifest.span.overlaps(start, end)
        ]

    def iter_manifests(self, *, layer_id: Optional[int] = None) -> Iterable[BlockManifest]:
        if layer_id is not None:
            yield from list(self._entries_by_layer.get(layer_id, []))
            return

        for current_layer in sorted(self._entries_by_layer):
            yield from self._entries_by_layer[current_layer]

    def clear(self) -> None:
        self._entries_by_layer.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": {
                str(layer_id): [manifest.to_dict() for manifest in manifests]
                for layer_id, manifests in sorted(self._entries_by_layer.items())
            }
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageTable":
        table = cls()
        for _, manifest_dicts in dict(payload.get("layers", {})).items():
            for manifest_dict in manifest_dicts:
                table.register(BlockManifest.from_dict(manifest_dict))
        return table


class BlockManager:
    """Single authority for archived-context metadata and residency state."""

    def __init__(self, model_id: str) -> None:
        if not model_id:
            raise ValueError("BlockManager.model_id must be non-empty")
        self.model_id = model_id
        self.page_table = PageTable()
        self._blocks: dict[BlockId, BlockManifest] = {}

    def clear(self) -> None:
        self.page_table.clear()
        self._blocks.clear()

    def register_block(self, manifest: BlockManifest) -> BlockManifest:
        if manifest.model_id != self.model_id:
            raise ValueError(
                f"Manifest model_id '{manifest.model_id}' does not match manager model_id "
                f"'{self.model_id}'"
            )
        if manifest.block_id in self._blocks:
            raise ValueError(f"Block {manifest.block_id} is already registered")

        self.page_table.register(manifest)
        self._blocks[manifest.block_id] = manifest
        return manifest

    def get_block(self, block_id: BlockId) -> BlockManifest:
        try:
            return self._blocks[block_id]
        except KeyError as exc:
            raise KeyError(f"Unknown block_id: {block_id}") from exc

    def iter_blocks(self, *, layer_id: Optional[int] = None) -> list[BlockManifest]:
        return list(self.page_table.iter_manifests(layer_id=layer_id))

    def locate_blocks_for_range(self, layer_id: int, start: int, end: int) -> list[BlockManifest]:
        return self.page_table.locate(layer_id, start, end)

    def promote_block(self, block_id: BlockId, target_location: BlockLocation) -> BlockManifest:
        if target_location == BlockLocation.MISSING:
            raise ValueError("Use mark_missing() for MISSING state transitions")
        manifest = self.get_block(block_id)
        manifest.residency = target_location
        manifest.touch()
        return manifest

    def demote_block(self, block_id: BlockId, target_location: BlockLocation) -> BlockManifest:
        if target_location == BlockLocation.MISSING:
            raise ValueError("Use mark_missing() for MISSING state transitions")
        manifest = self.get_block(block_id)
        manifest.residency = target_location
        manifest.touch()
        return manifest

    def mark_missing(self, block_id: BlockId, *, reason: Optional[str] = None) -> BlockManifest:
        manifest = self.get_block(block_id)
        manifest.residency = BlockLocation.MISSING
        if reason is not None:
            manifest.failure_reason = reason
        manifest.touch()
        return manifest

    def mark_unmaterializable(
        self,
        block_id: Optional[BlockId] = None,
        *,
        layer_id: Optional[int] = None,
        span: Optional[BlockSpan] = None,
        reason: Optional[str] = None,
    ) -> list[BlockManifest]:
        if block_id is not None:
            targets = [self.get_block(block_id)]
        else:
            if layer_id is None or span is None:
                raise ValueError("mark_unmaterializable() requires either block_id or layer_id+span")
            targets = self.locate_blocks_for_range(layer_id, span.logical_start, span.logical_end)
            if not targets:
                raise KeyError(
                    f"No blocks registered for layer {layer_id} in span {span.logical_start}:{span.logical_end}"
                )

        for manifest in targets:
            manifest.materializable = False
            if reason is not None:
                manifest.failure_reason = reason
            manifest.touch()
        return targets

    def get_residency_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "total_blocks": len(self._blocks),
            "total_tokens": 0,
            "unmaterializable_blocks": 0,
            "by_location": {
                location.value: {"blocks": 0, "tokens": 0}
                for location in BlockLocation
            },
        }

        for manifest in self.iter_blocks():
            location_stats = stats["by_location"][manifest.residency.value]
            location_stats["blocks"] += 1
            location_stats["tokens"] += manifest.token_count
            stats["total_tokens"] += manifest.token_count
            if not manifest.materializable:
                stats["unmaterializable_blocks"] += 1

        return stats

    def serialize_metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "page_table": self.page_table.to_dict(),
            "blocks": [manifest.to_dict() for manifest in self.iter_blocks()],
        }

    @classmethod
    def deserialize_metadata(cls, payload: dict[str, Any]) -> "BlockManager":
        manager = cls(model_id=str(payload["model_id"]))
        manifests = [
            BlockManifest.from_dict(manifest_dict)
            for manifest_dict in payload.get("blocks", [])
        ]
        manager.rebuild_from_manifests(manifests)
        return manager

    def rebuild_from_manifests(self, manifests: Iterable[BlockManifest]) -> None:
        self.clear()
        for manifest in sorted(
            manifests,
            key=lambda current: (current.layer_id, current.logical_start, current.logical_end),
        ):
            self.register_block(manifest)
