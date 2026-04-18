from __future__ import annotations

import unittest

import mlx.core as mx

from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode
from rfsn_v10_5.model import RFSNMLX


class ChunkedPrefillTest(unittest.TestCase):
    def test_prompt_longer_than_hot_capacity_completes(self) -> None:
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
            warm_capacity=8,
            cold_capacity=32,
            block_size_seq=4,
            model_dtype="float16",
            runtime_mode=RuntimeMode.EXACT,
        )
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)
        prompt = mx.arange(0, 12, dtype=mx.int32)[None, :]

        logits = model.prefill(prompt, cache)
        mx.eval(logits)

        layer_cache = cache.layer(0)
        stats = layer_cache.get_block_stats()
        self.assertEqual(logits.shape, (1, 12, config.vocab_size))
        self.assertLessEqual(layer_cache.hot_seq_len, config.hot_capacity)
        self.assertGreater(stats["total_blocks"], 0)
        self.assertGreater(
            stats["by_location"]["warm_ram"]["blocks"]
            + stats["by_location"]["cold_disk"]["blocks"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
