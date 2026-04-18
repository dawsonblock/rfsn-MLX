from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfsn_v10_5.cache import RFSNCache, derive_model_id
from rfsn_v10_5.model import RFSNMLX
from tests._helpers import build_config, persist_all_context_to_disk, seed_archived_context


class SessionPersistenceTest(unittest.TestCase):
    def test_restore_requires_explicit_session_id(self) -> None:
        config = build_config()

        with self.assertRaisesRegex(ValueError, "explicit session_id"):
            RFSNCache(config, batch_size=1, restore=True)

    def test_restore_is_isolated_by_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_a = build_config(disk_cache_dir=tmpdir, session_id="session-a")
            session_b = build_config(disk_cache_dir=tmpdir, session_id="session-b")
            model = RFSNMLX(session_a)

            cache_a = RFSNCache(session_a, batch_size=1)
            seed_archived_context(model, cache_a)
            persist_all_context_to_disk(cache_a)

            restored = RFSNCache(session_a, batch_size=1, restore=True)
            manifests = restored.restore_from_disk()
            self.assertGreater(sum(len(layer_manifests) for layer_manifests in manifests), 0)

            with self.assertRaisesRegex(FileNotFoundError, "unknown session 'session-b'"):
                RFSNCache(session_b, batch_size=1, restore=True).restore_from_disk()

            model_id = derive_model_id(session_a)
            self.assertTrue((Path(tmpdir) / model_id / "session-a").exists())
            self.assertFalse((Path(tmpdir) / model_id / "session-b").exists())

    def test_restore_rejects_empty_session_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(disk_cache_dir=tmpdir, session_id="empty-session")
            model_id = derive_model_id(config)
            (Path(tmpdir) / model_id / "empty-session" / "layer_0").mkdir(parents=True, exist_ok=True)

            with self.assertRaisesRegex(FileNotFoundError, "persisted archive is empty"):
                RFSNCache(config, batch_size=1, restore=True).restore_from_disk()


if __name__ == "__main__":
    unittest.main()