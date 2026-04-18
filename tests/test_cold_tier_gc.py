from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from rfsn_v10_5.block_manager import BlockLocation
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode
from rfsn_v10_5.model import RFSNMLX


class ArchivedDiskSpillTest(unittest.TestCase):
    def test_archived_blocks_spill_to_disk_and_remain_queryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RFSNConfig(
                hidden_dim=32,
                num_heads=2,
                num_kv_heads=2,
                head_dim=16,
                num_layers=1,
                vocab_size=128,
                hot_capacity=4,
                warm_capacity=4,
                cold_capacity=32,
                block_size_seq=4,
                model_dtype="float16",
                runtime_mode=RuntimeMode.ARCHIVED,
                disk_cache_dir=tmpdir,
            )
            model = RFSNMLX(config)
            cache = RFSNCache(config, batch_size=1)

            prompt = mx.arange(0, 12, dtype=mx.int32)[None, :]
            logits = model.prefill(prompt, cache)
            mx.eval(logits)

            layer_cache = cache.layer(0)
            cold_manifests = [
                manifest
                for manifest in layer_cache.block_manager.iter_blocks(layer_id=0)
                if manifest.residency == BlockLocation.COLD_DISK
            ]
            self.assertTrue(cold_manifests)
            self.assertTrue(all(manifest.payload_path for manifest in cold_manifests))
            self.assertTrue(all(Path(manifest.payload_path).exists() for manifest in cold_manifests if manifest.payload_path))

            segments = layer_cache.get_attention_segments()
            archived_k, archived_v, archived_start, archived_end = layer_cache.materialize_archived_context()
            mx.eval(archived_k, archived_v)

            self.assertGreaterEqual(len(segments), 2)
            self.assertEqual(archived_start, 0)
            self.assertEqual(archived_end, 8)
            self.assertEqual(layer_cache.get_total_length(), 12)


if __name__ == "__main__":
    unittest.main()
