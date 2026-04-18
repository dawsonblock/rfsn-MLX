from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfsn_v10_5.hf_config import HFConfigError, load_hf_config


class LoaderAutoConfigTest(unittest.TestCase):
    def _write_config(self, payload: dict[str, object]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        return root

    def test_llama_config_maps_to_valid_rfsn_config(self) -> None:
        root = self._write_config(
            {
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "vocab_size": 128256,
                "max_position_embeddings": 131072,
            }
        )

        config = load_hf_config(root)

        self.assertEqual(config.hidden_dim, 4096)
        self.assertEqual(config.ffn_dim, 14336)
        self.assertEqual(config.num_heads, 32)
        self.assertEqual(config.num_kv_heads, 8)
        self.assertEqual(config.num_layers, 32)
        self.assertEqual(config.rope_base, 500000.0)
        self.assertEqual(config.vocab_size, 128256)
        self.assertEqual(config.max_position_embeddings, 131072)

    def test_mistral_config_maps_to_valid_rfsn_config(self) -> None:
        root = self._write_config(
            {
                "model_type": "mistral",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "vocab_size": 32000,
                "max_position_embeddings": 32768,
            }
        )

        config = load_hf_config(root)

        self.assertEqual(config.hidden_dim, 4096)
        self.assertEqual(config.num_heads, 32)
        self.assertEqual(config.num_kv_heads, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.max_position_embeddings, 32768)

    def test_missing_required_field_raises_human_readable_error(self) -> None:
        root = self._write_config(
            {
                "model_type": "llama",
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
            }
        )

        with self.assertRaisesRegex(HFConfigError, "hidden_size"):
            load_hf_config(root)


if __name__ == "__main__":
    unittest.main()