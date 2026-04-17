from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode
from rfsn_v10_5.model import RFSNMLX


class ColdTierGCTest(unittest.TestCase):
    def test_cold_capacity_prunes_oldest_block_and_deletes_file(self) -> None:
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
                cold_capacity=4,
                block_size_seq=4,
                rvq_max_active=8,
                model_dtype="float16",
                runtime_mode=RuntimeMode.COMPRESSED,
                disk_cache_dir=tmpdir,
            )
            model = RFSNMLX(config)
            cache = RFSNCache(config, batch_size=1)

            prompt = mx.zeros((1, 4), dtype=mx.int32)
            logits = model.prefill(prompt, cache)
            mx.eval(logits)

            token = mx.zeros((1,), dtype=mx.int32)
            for pos in range(4, 13):
                logits = model.decode_step(token, cache, pos)
                mx.eval(logits)
                token = mx.argmax(logits, axis=-1).astype(mx.int32)

            layer_cache = cache.layer(0)
            self.assertEqual(len(layer_cache.cold_blocks), 1)
            survivor = layer_cache.cold_blocks[0]
            self.assertEqual((survivor.start_pos, survivor.end_pos), (4, 8))

            deleted_path = Path(tmpdir) / f"layer_{id(layer_cache)}_block_0_0.npz"
            surviving_path = Path(survivor.path)
            self.assertFalse(deleted_path.exists())
            self.assertTrue(surviving_path.exists())

            archived_k, archived_v, archived_start, archived_end = layer_cache.materialize_archived_context(
                model.layers[0].codec
            )
            mx.eval(archived_k, archived_v)
            self.assertEqual(archived_start, 4)
            self.assertEqual(archived_end, 12)
            self.assertEqual(layer_cache.get_total_length(), 9)


if __name__ == "__main__":
    unittest.main()