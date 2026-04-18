"""Durable disk-backed KV block storage for RFSN-MLX V11.

Phase 0 uses ``.npz`` payloads plus JSON sidecars because they are
already available in-repo, inspectable, and easy to harden with atomic
write and checksum validation.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional

import numpy as np

from .block_manager import BlockLocation, BlockManager, BlockManifest


LOGGER = logging.getLogger(__name__)


class BlockStorage:
    """Persist, load, quarantine, and scan exact KV blocks."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.quarantine_dir = self.root_dir / "quarantine"

    def persist_block(
        self,
        manifest: BlockManifest,
        payload: Mapping[str, np.ndarray],
    ) -> BlockManifest:
        normalized_payload = self._normalize_payload(payload, manifest)
        payload_bytes = self._serialize_payload(normalized_payload, manifest.payload_format)
        checksum = hashlib.sha256(payload_bytes).hexdigest()

        payload_path, manifest_path = self._resolve_paths(manifest)
        manifest.payload_path = str(payload_path)
        manifest.manifest_path = str(manifest_path)
        manifest.checksum = checksum
        manifest.residency = BlockLocation.COLD_DISK
        manifest.materializable = True
        manifest.failure_reason = None
        manifest.touch()

        try:
            self._atomic_write_bytes(payload_path, payload_bytes)
            self.write_manifest(manifest)
        except Exception:
            payload_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise

        return manifest

    def write_manifest(self, manifest: BlockManifest) -> BlockManifest:
        _, manifest_path = self._resolve_paths(manifest)
        manifest.manifest_path = str(manifest_path)
        manifest_bytes = json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        self._atomic_write_bytes(manifest_path, manifest_bytes)
        return manifest

    def load_block(self, manifest: BlockManifest) -> Optional[dict[str, np.ndarray]]:
        payload_path = self._payload_path(manifest)
        if payload_path is None or not payload_path.exists():
            LOGGER.warning(
                "Block %s is missing payload file at %s; marking it missing",
                manifest.block_id,
                payload_path,
            )
            self._mark_failed_block(manifest, "payload file missing")
            return None

        try:
            payload_bytes = payload_path.read_bytes()
        except OSError as exc:
            LOGGER.warning(
                "Failed to read block payload %s for %s: %s",
                payload_path,
                manifest.block_id,
                exc,
            )
            self._mark_failed_block(manifest, f"payload read failed: {exc}")
            return None

        actual_checksum = hashlib.sha256(payload_bytes).hexdigest()
        if manifest.checksum and actual_checksum != manifest.checksum:
            LOGGER.warning(
                "Checksum mismatch for block %s at %s; expected %s got %s",
                manifest.block_id,
                payload_path,
                manifest.checksum,
                actual_checksum,
            )
            self.quarantine_block(manifest)
            self._mark_failed_block(manifest, "checksum mismatch")
            return None

        try:
            payload = self._deserialize_payload(payload_bytes, manifest.payload_format)
            self._validate_loaded_payload(manifest, payload)
        except Exception as exc:
            LOGGER.warning(
                "Failed to deserialize block %s from %s: %s",
                manifest.block_id,
                payload_path,
                exc,
            )
            self.quarantine_block(manifest)
            self._mark_failed_block(manifest, f"deserialization failed: {exc}")
            return None

        manifest.touch()
        self.write_manifest(manifest)
        return payload

    def delete_block(self, manifest: BlockManifest, *, remove_manifest: bool = False) -> None:
        payload_path = self._payload_path(manifest)
        if payload_path is not None:
            payload_path.unlink(missing_ok=True)

        manifest_path = self._manifest_path(manifest)
        if remove_manifest and manifest_path is not None:
            manifest_path.unlink(missing_ok=True)
            return

        self._mark_failed_block(manifest, "payload deleted")

    def quarantine_block(self, manifest: BlockManifest) -> Optional[Path]:
        payload_path = self._payload_path(manifest)
        if payload_path is None or not payload_path.exists():
            return None

        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        quarantined_path = self.quarantine_dir / f"{payload_path.name}.{timestamp}.bad"
        payload_path.replace(quarantined_path)
        self._fsync_directory(self.quarantine_dir)
        return quarantined_path

    def scan_manifests(self, *, model_id: Optional[str] = None) -> list[BlockManifest]:
        manifests: list[BlockManifest] = []
        for manifest_path in sorted(self.root_dir.glob("*.manifest.json")):
            try:
                payload = json.loads(manifest_path.read_text())
                manifest = BlockManifest.from_dict(payload)
            except Exception as exc:
                LOGGER.warning("Ignoring unreadable manifest %s: %s", manifest_path, exc)
                continue

            manifest.manifest_path = str(manifest_path)
            if model_id is not None and manifest.model_id != model_id:
                continue

            payload_path = self._payload_path(manifest)
            if payload_path is None or not payload_path.exists():
                manifest.residency = BlockLocation.MISSING
                manifest.materializable = False
                manifest.failure_reason = manifest.failure_reason or "payload file missing"
                self.write_manifest(manifest)

            manifests.append(manifest)

        manifests.sort(key=lambda manifest: (manifest.layer_id, manifest.logical_start, manifest.logical_end))
        return manifests

    def rebuild_manager(self, manager: BlockManager) -> list[BlockManifest]:
        manifests = self.scan_manifests(model_id=manager.model_id)
        manager.rebuild_from_manifests(manifests)
        return manifests

    def _normalize_payload(
        self,
        payload: Mapping[str, np.ndarray],
        manifest: BlockManifest,
    ) -> dict[str, np.ndarray]:
        if not payload:
            raise ValueError("persist_block() requires a non-empty payload mapping")

        normalized = {name: np.asarray(array) for name, array in payload.items()}
        dtypes = {array.dtype.name for array in normalized.values()}
        if len(dtypes) != 1:
            raise ValueError("All payload arrays must share the same dtype")

        dtype_name = next(iter(dtypes))
        if manifest.dtype and manifest.dtype != dtype_name:
            raise ValueError(
                f"Manifest dtype '{manifest.dtype}' does not match payload dtype '{dtype_name}'"
            )
        manifest.dtype = dtype_name

        expected_shapes = manifest.shape_metadata or {
            name: tuple(int(dim) for dim in array.shape)
            for name, array in normalized.items()
        }
        for name, array in normalized.items():
            actual_shape = tuple(int(dim) for dim in array.shape)
            if name not in expected_shapes:
                raise ValueError(f"Payload array '{name}' missing from shape_metadata")
            if tuple(expected_shapes[name]) != actual_shape:
                raise ValueError(
                    f"Payload shape mismatch for '{name}': expected {expected_shapes[name]} got {actual_shape}"
                )
        manifest.shape_metadata = {
            name: tuple(int(dim) for dim in shape)
            for name, shape in expected_shapes.items()
        }
        return normalized

    def _serialize_payload(self, payload: Mapping[str, np.ndarray], payload_format: str) -> bytes:
        if payload_format != "npz":
            raise ValueError(f"Unsupported payload_format '{payload_format}'")

        buffer = io.BytesIO()
        savez_payload: dict[str, Any] = {name: array for name, array in payload.items()}
        np.savez_compressed(buffer, **savez_payload)
        return buffer.getvalue()

    def _deserialize_payload(
        self,
        payload_bytes: bytes,
        payload_format: str,
    ) -> dict[str, np.ndarray]:
        if payload_format != "npz":
            raise ValueError(f"Unsupported payload_format '{payload_format}'")

        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}

    def _validate_loaded_payload(
        self,
        manifest: BlockManifest,
        payload: Mapping[str, np.ndarray],
    ) -> None:
        missing_arrays = set(manifest.shape_metadata) - set(payload)
        if missing_arrays:
            raise ValueError(f"Missing arrays in payload: {sorted(missing_arrays)}")

        for name, expected_shape in manifest.shape_metadata.items():
            array = np.asarray(payload[name])
            actual_shape = tuple(int(dim) for dim in array.shape)
            if actual_shape != tuple(expected_shape):
                raise ValueError(
                    f"Payload shape mismatch for '{name}': expected {expected_shape} got {actual_shape}"
                )
            if array.dtype.name != manifest.dtype:
                raise ValueError(
                    f"Payload dtype mismatch for '{name}': expected {manifest.dtype} got {array.dtype.name}"
                )

    def _mark_failed_block(self, manifest: BlockManifest, reason: str) -> None:
        manifest.residency = BlockLocation.MISSING
        manifest.materializable = False
        manifest.failure_reason = reason
        manifest.touch()
        if manifest.manifest_path is not None:
            self.write_manifest(manifest)

    def _resolve_paths(self, manifest: BlockManifest) -> tuple[Path, Path]:
        payload_path = self._payload_path(manifest)
        if payload_path is None:
            payload_name = self._default_payload_name(manifest)
            payload_path = self.root_dir / payload_name

        manifest_path = self._manifest_path(manifest)
        if manifest_path is None:
            manifest_path = self.root_dir / f"{payload_path.name}.manifest.json"

        return payload_path, manifest_path

    def _payload_path(self, manifest: BlockManifest) -> Optional[Path]:
        if manifest.payload_path is None:
            return None
        return Path(manifest.payload_path)

    def _manifest_path(self, manifest: BlockManifest) -> Optional[Path]:
        if manifest.manifest_path is None:
            return None
        return Path(manifest.manifest_path)

    def _default_payload_name(self, manifest: BlockManifest) -> str:
        safe_model = self._sanitize_path_fragment(manifest.model_id)
        safe_block = self._sanitize_path_fragment(manifest.block_id.block_id)
        return f"{safe_model}.layer{manifest.layer_id}.{safe_block}.{manifest.payload_format}"

    @staticmethod
    def _sanitize_path_fragment(value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)

    def _atomic_write_bytes(self, target_path: Path, payload: bytes) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target_path)
            self._fsync_directory(target_path.parent)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            directory_fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)
