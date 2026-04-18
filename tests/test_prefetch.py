from __future__ import annotations

import tempfile
import unittest

import mlx.core as mx

from rfsn_v10_5.block_manager import BlockLocation
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode
from rfsn_v10_5.model import RFSNMLX


class PrefetchTest(unittest.TestCase):
    def test_sequential_prefetch_promotes_adjacent_cold_block_before_demand(self) -> None:
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
            manifests = list(layer_cache.block_manager.iter_blocks(layer_id=0))
            self.assertEqual([manifest.residency for manifest in manifests], [BlockLocation.COLD_DISK, BlockLocation.WARM_RAM])

            layer_cache.maybe_prefetch_for_decode(query_abs_pos=11)
            layer_cache.residency_manager.wait_for_prefetches(layer_cache)

            prefetched = layer_cache.block_manager.get_block(manifests[0].block_id)
            metrics = layer_cache.get_block_stats()["residency_metrics"]
            self.assertEqual(prefetched.residency, BlockLocation.WARM_RAM)
            self.assertTrue(layer_cache.is_manifest_resident(prefetched))
            self.assertEqual(metrics["prefetch_requests"], 1)
            self.assertEqual(metrics["prefetch_completions"], 1)


if __name__ == "__main__":
    unittest.main()