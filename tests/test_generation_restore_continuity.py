from __future__ import annotations

import tempfile
import unittest

import numpy as np
import mlx.core as mx

from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.model import RFSNMLX
from tests._helpers import build_config, persist_all_context_to_disk, prompt_tokens


class GenerationRestoreContinuityTest(unittest.TestCase):
    def test_restored_continuation_matches_uninterrupted_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(disk_cache_dir=tmpdir, session_id="restart-session")
            model = RFSNMLX(config)
            cache = RFSNCache(config, batch_size=1)

            prompt = prompt_tokens(16)
            continuation = prompt_tokens(4, start=16)
            logits = model.prefill(prompt, cache)
            mx.eval(logits)
            persist_all_context_to_disk(cache)

            restarted_cache = RFSNCache(config, batch_size=1, restore=True)
            restored = restarted_cache.restore_from_disk()
            restored_layer = restarted_cache.layer(0)

            self.assertEqual(len(restored[0]), 4)
            restored_archived_k, restored_archived_v, start, end = restored_layer.materialize_archived_context()
            mx.eval(restored_archived_k, restored_archived_v)
            self.assertEqual((start, end), (0, 16))

            uninterrupted_logits = model.prefill(continuation, cache)
            restored_logits = model.prefill(continuation, restarted_cache)
            mx.eval(uninterrupted_logits, restored_logits)

            self.assertEqual(cache.layer(0).hot_end, 20)
            self.assertEqual(restarted_cache.layer(0).hot_end, 20)
            self.assertEqual(uninterrupted_logits.shape, (1, 4, config.vocab_size))
            self.assertEqual(restored_logits.shape, (1, 4, config.vocab_size))
            np.testing.assert_allclose(
                np.array(uninterrupted_logits),
                np.array(restored_logits),
                atol=1e-5,
                rtol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()