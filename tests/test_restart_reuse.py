from __future__ import annotations

import tempfile
import unittest

import mlx.core as mx

from rfsn_v10_5.block_manager import BlockLocation
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode
from rfsn_v10_5.model import RFSNMLX


class RestartReuseTest(unittest.TestCase):
    def test_persisted_blocks_restore_after_restart_for_matching_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RFSNConfig(
                hidden_dim=32,
                num_heads=2,
                num_kv_heads=2,
                head_dim=16,
                num_layers=1,
                vocab_size=128,
                num_subspaces=4,
                subspace_dim=4,
                hot_capacity=4,
                warm_capacity=4,
                cold_capacity=32,
                block_size_seq=4,
                model_dtype="float16",
                runtime_mode=RuntimeMode.COMPRESSED,
                disk_cache_dir=tmpdir,
            )
            model = RFSNMLX(config)
            cache = RFSNCache(config, batch_size=1)

            prompt = mx.arange(0, 16, dtype=mx.int32)[None, :]
            logits = model.prefill(prompt, cache)
            mx.eval(logits)

            layer_cache = cache.layer(0)
            layer_cache.evict_for_append(config.hot_capacity)
            for manifest in list(layer_cache.block_manager.iter_blocks(layer_id=0)):
                if manifest.residency == BlockLocation.WARM_RAM:
                    layer_cache.demote_manifest_to_cold(manifest)

            original_blocks = list(layer_cache.block_manager.iter_blocks(layer_id=0))
            self.assertTrue(original_blocks)
            self.assertTrue(all(manifest.residency == BlockLocation.COLD_DISK for manifest in original_blocks))

            restarted_cache = RFSNCache(config, batch_size=1)
            restored = restarted_cache.restore_from_disk()
            restored_layer = restarted_cache.layer(0)

            self.assertEqual(len(restored[0]), len(original_blocks))
            restored_archived_k, restored_archived_v, start, end = restored_layer.materialize_archived_context()
            mx.eval(restored_archived_k, restored_archived_v)

            self.assertEqual((start, end), (0, 16))
            logits = model.decode_step(mx.zeros((1,), dtype=mx.int32), restarted_cache, pos=16)
            mx.eval(logits)
            self.assertEqual(logits.shape, (1, config.vocab_size))


if __name__ == "__main__":
    unittest.main()