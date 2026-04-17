from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfsn_v10_5 import loader


class LoaderCheckpointResolutionTest(unittest.TestCase):
    def test_resolve_checkpoint_directory_uses_shard_index(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            shard_two = root / "model-00002-of-00002.safetensors"
            shard_one = root / "model-00001-of-00002.safetensors"
            shard_two.write_bytes(b"")
            shard_one.write_bytes(b"")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": shard_two.name,
                            "model.layers.0.self_attn.q_proj.weight": shard_one.name,
                        }
                    }
                )
            )

            paths = loader._resolve_checkpoint_paths(root)

        self.assertEqual(paths, [shard_one, shard_two])

    def test_resolve_checkpoint_directory_accepts_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"")

            paths = loader._resolve_checkpoint_paths(root)

        self.assertEqual(paths, [checkpoint])