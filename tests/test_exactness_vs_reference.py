from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np

from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode
from rfsn_v10_5.model import RFSNMLX


class ExactnessVsReferenceTest(unittest.TestCase):
    def test_chunked_prefill_matches_no_archive_reference_within_tolerance(self) -> None:
        archived_config = RFSNConfig(
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
            model_dtype="float32",
            runtime_mode=RuntimeMode.ARCHIVED,
        )
        reference_config = RFSNConfig(
            hidden_dim=32,
            num_heads=2,
            num_kv_heads=2,
            head_dim=16,
            num_layers=1,
            vocab_size=128,
            hot_capacity=32,
            warm_capacity=32,
            cold_capacity=64,
            block_size_seq=4,
            model_dtype="float32",
            runtime_mode=RuntimeMode.ARCHIVED,
        )

        archived_model = RFSNMLX(archived_config)
        reference_model = RFSNMLX(reference_config)
        reference_model.update(archived_model.parameters())
        mx.eval(reference_model.parameters())

        archived_cache = RFSNCache(archived_config, batch_size=1)
        reference_cache = RFSNCache(reference_config, batch_size=1)
        prompt = mx.arange(0, 12, dtype=mx.int32)[None, :]

        archived_logits = archived_model.prefill(prompt, archived_cache)
        reference_logits = reference_model.prefill(prompt, reference_cache)
        mx.eval(archived_logits, reference_logits)

        self.assertGreater(archived_cache.layer(0).get_block_stats()["total_blocks"], 0)
        np.testing.assert_allclose(
            np.array(archived_logits),
            np.array(reference_logits),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()