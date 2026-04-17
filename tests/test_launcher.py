from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import mlx.core as mx

from rfsn_v10_5 import launcher


class LauncherTest(unittest.TestCase):
    def test_build_config_accepts_cache_dtype(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "bench",
                "--hidden-dim",
                "128",
                "--num-heads",
                "2",
                "--num-kv-heads",
                "2",
                "--head-dim",
                "64",
                "--num-layers",
                "1",
                "--vocab-size",
                "128",
                "--cache-dtype",
                "fp8_e4m3",
            ]
        )

        config = launcher._build_config(args)

        self.assertEqual(config.cache_dtype, "fp8_e4m3")

    def test_check_command_uses_cache_dtype(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(["check", "--cache-dtype", "fp8_e4m3"])
        seen: dict[str, object] = {}

        class FakeCache:
            def __init__(self, config, batch_size: int) -> None:
                seen["cache_dtype"] = config.cache_dtype
                self.config = config
                self.batch_size = batch_size

        class FakeModel:
            def __init__(self, config) -> None:
                seen["model_cache_dtype"] = config.cache_dtype

            def prefill(self, prompt, cache):
                return mx.zeros((1, 8, 1000), dtype=mx.float32)

            def decode_step(self, token, cache, pos: int):
                return mx.zeros((1, 1000), dtype=mx.float32)

        output = io.StringIO()
        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
            "rfsn_v10_5.launcher.RFSNCache", FakeCache
        ), redirect_stdout(output):
            launcher.cmd_check(args)

        stdout = output.getvalue()
        self.assertEqual(seen["cache_dtype"], "fp8_e4m3")
        self.assertEqual(seen["model_cache_dtype"], "fp8_e4m3")
        self.assertIn("cache_dtype=fp8_e4m3", stdout)
        self.assertIn("[check] PASS", stdout)