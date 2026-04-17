from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np

from rfsn_v10_5.cache import LayerKVCache, RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode, resolve_dtype
from rfsn_v10_5.model import RFSNMLX


class CacheDtypeTest(unittest.TestCase):
    def _build_config(self, runtime_mode: RuntimeMode) -> RFSNConfig:
        return RFSNConfig(
            hidden_dim=32,
            num_heads=2,
            num_kv_heads=2,
            head_dim=16,
            num_layers=1,
            vocab_size=128,
            num_subspaces=4,
            subspace_dim=4,
            hot_capacity=8,
            warm_capacity=16,
            cold_capacity=32,
            block_size_seq=4,
            rvq_max_active=8,
            model_dtype="float16",
            cache_dtype="fp8_e4m3",
            runtime_mode=runtime_mode,
        )

    def _representable_kv(self) -> tuple[mx.array, mx.array]:
        row_a = np.array(
            [
                0.0,
                0.25,
                -0.25,
                0.5,
                -0.5,
                0.75,
                -0.75,
                1.0,
                -1.0,
                1.5,
                -1.5,
                2.0,
                -2.0,
                4.0,
                -4.0,
                8.0,
            ],
            dtype=np.float32,
        )
        row_b = np.array(
            [
                0.0,
                -0.125,
                0.125,
                -0.5,
                0.5,
                -1.0,
                1.0,
                -1.5,
                1.5,
                -2.0,
                2.0,
                -3.0,
                3.0,
                -6.0,
                6.0,
                -12.0,
            ],
            dtype=np.float32,
        )
        base = np.stack([row_a, row_b], axis=0)
        k = np.broadcast_to(base[None, None, :, :], (1, 2, 2, 16)).copy()
        v = np.broadcast_to(base[None, None, ::-1, :], (1, 2, 2, 16)).copy()
        return mx.array(k, dtype=mx.float16), mx.array(v, dtype=mx.float16)

    def _archive_until_cold(self, model: RFSNMLX, cache: RFSNCache, prompt_len: int = 4) -> None:
        prompt = mx.zeros((1, prompt_len), dtype=mx.int32)
        logits = model.prefill(prompt, cache)
        mx.eval(logits)

        token = mx.zeros((1,), dtype=mx.int32)
        pos = prompt_len
        for _ in range(prompt_len * 2 + 1):
            logits = model.decode_step(token, cache, pos)
            mx.eval(logits)
            token = mx.argmax(logits, axis=-1).astype(mx.int32)
            pos += 1
            if cache.layer(0).cold_blocks:
                return

        self.fail("Expected compressed decode to spill at least one block to cold storage")

    def test_fp8_cache_uses_uint8_storage_and_round_trips_representable_values(self) -> None:
        config = self._build_config(RuntimeMode.EXACT)
        cache = LayerKVCache(
            config,
            batch_size=1,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
        )
        k, v = self._representable_kv()

        cache.append_exact(k, v)
        hot_k, hot_v, _, _ = cache.materialize_exact_context()
        mx.eval(hot_k, hot_v)

        self.assertEqual(cache.hot_k.dtype, mx.uint8)
        self.assertEqual(cache.hot_v.dtype, mx.uint8)
        self.assertEqual(np.array(cache.hot_k).dtype.itemsize, 1)
        self.assertLess(np.array(cache.hot_k).dtype.itemsize, np.dtype(np.float16).itemsize)
        np.testing.assert_allclose(np.array(hot_k), np.array(k), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.array(hot_v), np.array(v), atol=0.0, rtol=0.0)

    def test_fp8_cache_runs_through_compressed_decode(self) -> None:
        config = self._build_config(RuntimeMode.COMPRESSED)
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)
        prompt = mx.zeros((1, 8), dtype=mx.int32)

        logits = model.prefill(prompt, cache)
        mx.eval(logits)

        token = mx.zeros((1,), dtype=mx.int32)
        for pos in range(8, 10):
            logits = model.decode_step(token, cache, pos)
            mx.eval(logits)
            token = mx.argmax(logits, axis=-1).astype(mx.int32)

        layer_cache = cache.layer(0)
        self.assertEqual(layer_cache.hot_k.dtype, mx.uint8)
        self.assertTrue(layer_cache.warm_blocks or layer_cache.cold_blocks)

        segment_dtype = resolve_dtype(config.model_dtype)
        segments = layer_cache.get_mixed_attention_segments(model.layers[0].codec)
        self.assertTrue(segments)
        for seg_k, seg_v, _ in segments:
            self.assertEqual(seg_k.dtype, segment_dtype)
            self.assertEqual(seg_v.dtype, segment_dtype)

    def test_fp8_archive_payloads_use_uint8_in_warm_and_cold_tiers(self) -> None:
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
            cold_capacity=16,
            block_size_seq=4,
            rvq_max_active=8,
            model_dtype="float16",
            cache_dtype="fp8_e4m3",
            runtime_mode=RuntimeMode.COMPRESSED,
        )
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)

        self._archive_until_cold(model, cache, prompt_len=4)

        layer_cache = cache.layer(0)
        self.assertTrue(layer_cache.warm_blocks)
        self.assertTrue(layer_cache.cold_blocks)

        for block in layer_cache.warm_blocks + layer_cache.cold_blocks:
            self.assertEqual(block.v_payload.dtype, np.uint8)
            self.assertEqual(block.v_payload.dtype.itemsize, 1)

        segments = layer_cache.get_mixed_attention_segments(model.layers[0].codec)
        self.assertTrue(segments)
        expected_dtype = resolve_dtype(config.model_dtype)
        for _, seg_v, _ in segments:
            self.assertEqual(seg_v.dtype, expected_dtype)

    def test_archived_context_cache_does_not_keep_per_block_reconstructions(self) -> None:
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
            cold_capacity=16,
            block_size_seq=4,
            rvq_max_active=8,
            model_dtype="float16",
            cache_dtype="fp8_e4m3",
            runtime_mode=RuntimeMode.COMPRESSED,
        )
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)

        self._archive_until_cold(model, cache, prompt_len=4)

        layer_cache = cache.layer(0)
        first_segments = layer_cache.get_mixed_attention_segments(model.layers[0].codec)
        second_segments = layer_cache.get_mixed_attention_segments(model.layers[0].codec)

        self.assertTrue(first_segments)
        self.assertIs(layer_cache._archived_k, first_segments[0][0])
        self.assertIs(layer_cache._archived_k, second_segments[0][0])
        for block in layer_cache.warm_blocks + layer_cache.cold_blocks:
            self.assertIsNone(block.reconstructed_k)
            self.assertIsNone(block.reconstructed_v)


if __name__ == "__main__":
    unittest.main()
