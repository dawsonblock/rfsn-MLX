from __future__ import annotations

import tempfile
import unittest

import mlx.core as mx

from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.model import RFSNMLX
from tests._helpers import build_config, persist_all_context_to_disk, prompt_tokens, seed_archived_context


class ContextBoundsTest(unittest.TestCase):
    def test_prompt_exactly_at_max_position_embeddings_is_allowed(self) -> None:
        config = build_config(max_position_embeddings=8)
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)

        logits = model.prefill(prompt_tokens(8), cache)
        mx.eval(logits)

        self.assertEqual(logits.shape, (1, 8, config.vocab_size))

    def test_prompt_over_max_position_embeddings_is_rejected(self) -> None:
        config = build_config(max_position_embeddings=8)
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)

        with self.assertRaisesRegex(ValueError, "max_position_embeddings=8"):
            model.prefill(prompt_tokens(9), cache)

    def test_prompt_plus_generation_over_limit_is_rejected(self) -> None:
        config = build_config(max_position_embeddings=8)
        model = RFSNMLX(config)

        with self.assertRaisesRegex(ValueError, "max_position_embeddings=8"):
            model.generate(prompt_tokens(6), max_new_tokens=3)

    def test_restore_continuation_over_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(
                disk_cache_dir=tmpdir,
                session_id="bounded-session",
                max_position_embeddings=12,
            )
            model = RFSNMLX(config)
            cache = RFSNCache(config, batch_size=1)

            seed_archived_context(model, cache, prompt_len=12)
            persist_all_context_to_disk(cache)

            restored_cache = RFSNCache(config, batch_size=1, restore=True)
            restored_cache.restore_from_disk()

            with self.assertRaisesRegex(ValueError, "max_position_embeddings=12"):
                model.prefill(prompt_tokens(1, start=12), restored_cache)


if __name__ == "__main__":
    unittest.main()