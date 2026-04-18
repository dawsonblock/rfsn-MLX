from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfsn_v10_5.block_manager import (
    BlockId,
    BlockLocation,
    BlockManager,
    BlockManifest,
    BlockSpan,
)
from rfsn_v10_5.storage import BlockStorage


class ColdStorageIntegrityTest(unittest.TestCase):
    def _manifest(self, block_name: str, start: int = 0, end: int = 4) -> BlockManifest:
        return BlockManifest(
            block_id=BlockId("model-a", 0, block_name),
            span=BlockSpan(start, end),
            codec_version="v11-exact-kv",
            dtype="float32",
            shape_metadata={
                "keys": (1, 1, end - start, 2),
                "values": (1, 1, end - start, 2),
            },
            residency=BlockLocation.COLD_DISK,
        )

    def _payload(self, token_count: int) -> dict[str, np.ndarray]:
        keys = np.arange(token_count * 2, dtype=np.float32).reshape(1, 1, token_count, 2)
        values = (keys + 100.0).astype(np.float32)
        return {"keys": keys, "values": values}

    def test_persist_and_restart_rebuild_restore_manifest_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = BlockStorage(tmpdir)
            manager = BlockManager("model-a")
            manifest = self._manifest("blk0")
            payload = self._payload(manifest.token_count)

            storage.persist_block(manifest, payload)
            manager.register_block(manifest)

            restored_manager = BlockManager("model-a")
            scanned = storage.rebuild_manager(restored_manager)
            loaded = storage.load_block(scanned[0])

            self.assertEqual(len(scanned), 1)
            self.assertEqual(
                restored_manager.locate_blocks_for_range(0, 0, 4)[0].checksum,
                manifest.checksum,
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            np.testing.assert_allclose(loaded["keys"], payload["keys"])
            np.testing.assert_allclose(loaded["values"], payload["values"])

    def test_missing_payload_marks_block_missing_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = BlockStorage(tmpdir)
            manager = BlockManager("model-a")
            manifest = self._manifest("blk-missing")

            storage.persist_block(manifest, self._payload(manifest.token_count))
            manager.register_block(manifest)
            assert manifest.payload_path is not None
            Path(manifest.payload_path).unlink()

            loaded = storage.load_block(manifest)
            missing_reason = manager.get_block(manifest.block_id).failure_reason

            self.assertIsNone(loaded)
            self.assertEqual(manager.get_block(manifest.block_id).residency, BlockLocation.MISSING)
            self.assertFalse(manager.get_block(manifest.block_id).materializable)
            assert missing_reason is not None
            self.assertIn("missing", missing_reason)

    def test_checksum_mismatch_quarantines_payload_and_marks_block_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = BlockStorage(tmpdir)
            manager = BlockManager("model-a")
            manifest = self._manifest("blk-bad-checksum")

            storage.persist_block(manifest, self._payload(manifest.token_count))
            manager.register_block(manifest)
            assert manifest.payload_path is not None
            Path(manifest.payload_path).write_bytes(b"tampered-payload")

            loaded = storage.load_block(manifest)
            failure_reason = manager.get_block(manifest.block_id).failure_reason

            self.assertIsNone(loaded)
            self.assertFalse(Path(manifest.payload_path).exists())
            self.assertTrue(any(storage.quarantine_dir.iterdir()))
            self.assertEqual(manager.get_block(manifest.block_id).residency, BlockLocation.MISSING)
            self.assertFalse(manager.get_block(manifest.block_id).materializable)
            assert failure_reason is not None
            self.assertIn("checksum mismatch", failure_reason)

    def test_deserialize_failure_quarantines_payload_and_marks_block_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = BlockStorage(tmpdir)
            manager = BlockManager("model-a")
            manifest = self._manifest("blk-bad-npz")

            storage.persist_block(manifest, self._payload(manifest.token_count))
            manager.register_block(manifest)

            invalid_payload = b"not-a-valid-npz"
            assert manifest.payload_path is not None
            Path(manifest.payload_path).write_bytes(invalid_payload)
            manifest.checksum = hashlib.sha256(invalid_payload).hexdigest()
            storage.write_manifest(manifest)

            loaded = storage.load_block(manifest)
            failure_reason = manager.get_block(manifest.block_id).failure_reason

            self.assertIsNone(loaded)
            self.assertFalse(Path(manifest.payload_path).exists())
            self.assertTrue(any(storage.quarantine_dir.iterdir()))
            self.assertEqual(manager.get_block(manifest.block_id).residency, BlockLocation.MISSING)
            self.assertFalse(manager.get_block(manifest.block_id).materializable)
            assert failure_reason is not None
            self.assertIn("deserialization failed", failure_reason)


if __name__ == "__main__":
    unittest.main()