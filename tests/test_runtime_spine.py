from __future__ import annotations

import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx

import rfsn_v10_5
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import resolve_dtype
from rfsn_v10_5.layer import RFSNLayerMLX
from rfsn_v10_5.model import RFSNMLX
from tests._helpers import build_config, prompt_tokens


class RuntimeSpineTest(unittest.TestCase):
    def test_single_segment_layer_path_uses_exact_attention(self) -> None:
        config = build_config(session_id="single-segment")
        layer = RFSNLayerMLX(config, layer_idx=0)
        cache = RFSNCache(config, batch_size=1).layer(0)
        x = mx.zeros((1, 1, config.hidden_dim), dtype=resolve_dtype(config.model_dtype))
        attention_output = mx.zeros(
            (1, config.num_heads, 1, config.head_dim),
            dtype=resolve_dtype(config.model_dtype),
        )

        with patch("rfsn_v10_5.layer.exact_attention", return_value=attention_output) as exact_mock, patch(
            "rfsn_v10_5.layer.run_segmented_attention",
            return_value=attention_output,
        ) as segmented_mock:
            out = layer(x, cache, start_pos=0)
            mx.eval(out)

        self.assertEqual(out.shape, (1, 1, config.hidden_dim))
        exact_mock.assert_called_once()
        segmented_mock.assert_not_called()

    def test_multi_segment_layer_path_uses_segmented_attention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(disk_cache_dir=tmpdir, session_id="multi-segment")
            model = RFSNMLX(config)
            cache = RFSNCache(config, batch_size=1)
            prompt = prompt_tokens(8)
            mx.eval(model.prefill(prompt, cache))

            attention_output = mx.zeros(
                (1, config.num_heads, 1, config.head_dim),
                dtype=resolve_dtype(config.model_dtype),
            )
            x = mx.zeros((1, 1, config.hidden_dim), dtype=resolve_dtype(config.model_dtype))

            with patch("rfsn_v10_5.layer.exact_attention", return_value=attention_output) as exact_mock, patch(
                "rfsn_v10_5.layer.run_segmented_attention",
                return_value=attention_output,
            ) as segmented_mock:
                out = model.layers[0](x, cache.layer(0), start_pos=8)
                mx.eval(out)

        self.assertEqual(out.shape, (1, 1, config.hidden_dim))
        segmented_mock.assert_called_once()
        exact_mock.assert_not_called()

    def test_removed_compression_modules_and_symbols_are_absent(self) -> None:
        for module_name in (
            "rfsn_v10_5.codec",
            "rfsn_v10_5.attention_compressed",
            "rfsn_v10_5.fp8",
            "rfsn_v10_5.types",
        ):
            self.assertIsNone(importlib.util.find_spec(module_name))

        self.assertFalse(hasattr(rfsn_v10_5, "HybridKeyCodec"))
        self.assertFalse(hasattr(rfsn_v10_5, "SafetyMode"))


if __name__ == "__main__":
    unittest.main()