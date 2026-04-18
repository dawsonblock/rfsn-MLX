from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfsn_v10_5.block_manager import BlockLocation, BlockManager
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.storage import BlockStorage
from tests._helpers import build_config, build_manifest, build_payload


class ArchiveIntegrityTest(unittest.TestCase):
    def test_persist_and_restart_rebuild_restore_manifest_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = BlockStorage(tmpdir)
            manager = BlockManager("model-a")
            manifest = build_manifest("model-a")
            payload = build_payload(manifest.token_count)

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
            manifest = build_manifest("model-a", block_name="blk-missing")

            storage.persist_block(manifest, build_payload(manifest.token_count))
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
            manifest = build_manifest("model-a", block_name="blk-bad-checksum")

            storage.persist_block(manifest, build_payload(manifest.token_count))
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
            manifest = build_manifest("model-a", block_name="blk-bad-npz")

            storage.persist_block(manifest, build_payload(manifest.token_count))
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

    def test_restore_rejects_gap_in_persisted_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(disk_cache_dir=tmpdir, session_id="gap-session")
            cache = RFSNCache(config, batch_size=1)
            layer_cache = cache.layer(0)

            first = build_manifest(cache.model_id, block_name="blk0", start=0, end=4)
            second = build_manifest(cache.model_id, block_name="blk1", start=8, end=12)
            layer_cache.storage.persist_block(first, build_payload(first.token_count))
            layer_cache.storage.persist_block(second, build_payload(second.token_count))

            with self.assertRaisesRegex(RuntimeError, "contains a gap"):
                RFSNCache(config, batch_size=1, restore=True).restore_from_disk()

    def test_restore_rejects_unreadable_persisted_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(disk_cache_dir=tmpdir, session_id="bad-archive")
            cache = RFSNCache(config, batch_size=1)
            layer_cache = cache.layer(0)

            manifest = build_manifest(cache.model_id, block_name="blk0", start=0, end=4)
            layer_cache.storage.persist_block(manifest, build_payload(manifest.token_count))
            assert manifest.payload_path is not None
            Path(manifest.payload_path).unlink()

            with self.assertRaisesRegex(RuntimeError, "contains an unreadable block"):
                RFSNCache(config, batch_size=1, restore=True).restore_from_disk()


if __name__ == "__main__":
    unittest.main()