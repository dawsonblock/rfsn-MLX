from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from rfsn_v10_5.block_manager import BlockLocation
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode, resolve_dtype
from rfsn_v10_5.model import RFSNMLX


class ExactArchiveCacheTest(unittest.TestCase):
    def _build_config(self, *, disk_cache_dir: str | None = None) -> RFSNConfig:
        return RFSNConfig(
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
            cache_dtype="fp8_e4m3",
            runtime_mode=RuntimeMode.COMPRESSED,
            disk_cache_dir=disk_cache_dir or "./rfsn_disk_cache",
        )

    def _seed_archived_context(self, model: RFSNMLX, cache: RFSNCache, prompt_len: int = 12) -> None:
        prompt = mx.arange(0, prompt_len, dtype=mx.int32)[None, :]
        logits = model.prefill(prompt, cache)
        mx.eval(logits)

    def test_archived_segments_use_exact_model_dtype(self) -> None:
        config = self._build_config()
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)

        self._seed_archived_context(model, cache)

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

    def test_missing_cold_block_is_skipped_without_crashing_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_config(disk_cache_dir=tmpdir)
            model = RFSNMLX(config)
            cache = RFSNCache(config, batch_size=1)

            self._seed_archived_context(model, cache)
            layer_cache = cache.layer(0)
            cold_manifests = [
                manifest
                for manifest in layer_cache.block_manager.iter_blocks(layer_id=0)
                if manifest.residency == BlockLocation.COLD_DISK
            ]
            self.assertTrue(cold_manifests)

            victim = cold_manifests[0]
            assert victim.payload_path is not None
            Path(victim.payload_path).unlink()

            logits = model.decode_step(mx.zeros((1,), dtype=mx.int32), cache, pos=12)
            mx.eval(logits)

            self.assertEqual(logits.shape, (1, config.vocab_size))
            self.assertEqual(victim.residency, BlockLocation.MISSING)
            self.assertFalse(victim.materializable)


if __name__ == "__main__":
    unittest.main()
