from __future__ import annotations

import tempfile
import unittest

import mlx.core as mx

from rfsn_v10_5.block_manager import BlockLocation
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RuntimeMode, resolve_dtype
from rfsn_v10_5.model import RFSNMLX
from tests._helpers import build_config, seed_archived_context


class ArchivedRuntimeTest(unittest.TestCase):
    def test_exact_mode_rejects_overflow_before_archival_spill(self) -> None:
        config = build_config(runtime_mode=RuntimeMode.EXACT)
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)

        with self.assertRaisesRegex(RuntimeError, "hot-window only"):
            model.prefill(mx.arange(0, 12, dtype=mx.int32)[None, :], cache)

    def test_archived_mode_spills_and_continues_when_hot_capacity_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(disk_cache_dir=tmpdir)
            model = RFSNMLX(config)
            cache = RFSNCache(config, batch_size=1)

            logits = seed_archived_context(model, cache)

            layer_cache = cache.layer(0)
            stats = layer_cache.get_block_stats()
            segments = layer_cache.get_attention_segments()

            self.assertEqual(logits.shape, (1, 12, config.vocab_size))
            self.assertLessEqual(layer_cache.hot_seq_len, config.hot_capacity)
            self.assertGreaterEqual(stats["by_location"][BlockLocation.COLD_DISK.value]["blocks"], 1)
            self.assertGreaterEqual(len(segments), 2)

    def test_archived_segments_use_exact_model_dtype(self) -> None:
        config = build_config()
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)

        seed_archived_context(model, cache)

        layer_cache = cache.layer(0)
        expected_dtype = resolve_dtype(config.model_dtype)
        segments = layer_cache.get_attention_segments()
        archived_manifests = list(layer_cache.block_manager.iter_blocks(layer_id=0))

        self.assertEqual(layer_cache.hot_k.dtype, expected_dtype)
        self.assertTrue(archived_manifests)
        self.assertTrue(any(segment[0].dtype == expected_dtype for segment in segments))
        self.assertGreaterEqual(
            layer_cache.get_block_stats()["by_location"][BlockLocation.COLD_DISK.value]["blocks"],
            1,
        )

    def test_runtime_mode_setting_changes_real_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archived_config = build_config(disk_cache_dir=tmpdir, session_id="archived-session")
            exact_config = build_config(
                disk_cache_dir=tmpdir,
                session_id="exact-session",
                runtime_mode=RuntimeMode.EXACT,
            )
            archived_model = RFSNMLX(archived_config)
            exact_model = RFSNMLX(exact_config)
            archived_cache = RFSNCache(archived_config, batch_size=1)
            exact_cache = RFSNCache(exact_config, batch_size=1)
            prompt = mx.arange(0, 12, dtype=mx.int32)[None, :]

            with self.assertRaisesRegex(RuntimeError, "hot-window only"):
                exact_model.prefill(prompt, exact_cache)

            logits = archived_model.prefill(prompt, archived_cache)
            mx.eval(logits)

            self.assertEqual(logits.shape, (1, 12, archived_config.vocab_size))
            self.assertGreater(archived_cache.layer(0).get_block_stats()["total_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
