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

    def test_generate_command_materializes_sequence_from_generated_steps(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "generate",
                "--prompt-ids",
                "1,2",
                "--max-new-tokens",
                "2",
            ]
        )

        class FakeCache:
            def __init__(self, config, batch_size: int) -> None:
                self.config = config
                self.batch_size = batch_size

        class FakeModel:
            def __init__(self, config) -> None:
                self.config = config

            def generate(
                self,
                prompt_ids,
                max_new_tokens: int,
                cache=None,
                temperature: float = 1.0,
                top_p: float = 1.0,
                top_k: int = 0,
                repetition_penalty: float = 1.0,
            ):
                return [
                    mx.array([3], dtype=mx.int32),
                    mx.array([4], dtype=mx.int32),
                ]

        output = io.StringIO()
        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
            "rfsn_v10_5.launcher.RFSNCache", FakeCache
        ), redirect_stdout(output):
            launcher.cmd_generate(args)

        stdout = output.getvalue()
        self.assertIn("[generate] Generated 2 new tokens", stdout)
        self.assertIn("[generate] Output IDs: [1, 2, 3, 4]", stdout)

    def test_generate_command_uses_tokenizer_for_text_io(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "generate",
                "--prompt",
                "Hello world",
                "--tokenizer",
                "fake-tokenizer",
                "--max-new-tokens",
                "2",
            ]
        )
        seen: dict[str, object] = {}

        class FakeCache:
            def __init__(self, config, batch_size: int) -> None:
                self.config = config
                self.batch_size = batch_size

        class FakeModel:
            def __init__(self, config) -> None:
                self.config = config

            def generate(
                self,
                prompt_ids,
                max_new_tokens: int,
                cache=None,
                temperature: float = 1.0,
                top_p: float = 1.0,
                top_k: int = 0,
                repetition_penalty: float = 1.0,
            ):
                seen["prompt_ids"] = prompt_ids.tolist()
                return [
                    mx.array([7], dtype=mx.int32),
                    mx.array([8], dtype=mx.int32),
                ]

        def fake_load_tokenizer(name_or_path: str):
            seen["tokenizer_name"] = name_or_path
            return object()

        def fake_encode_prompt_text(tokenizer, text: str, *, vocab_size: int, add_special_tokens: bool = True):
            seen["prompt_text"] = text
            seen["vocab_size"] = vocab_size
            return mx.array([[5, 6]], dtype=mx.int32)

        def fake_decode_token_ids(tokenizer, token_ids, *, skip_special_tokens: bool = True):
            as_list = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
            seen.setdefault("decoded_batches", []).append(as_list)
            if as_list == [[5, 6, 7, 8]]:
                return "Hello world extended"
            if as_list == [[7, 8]]:
                return " extended"
            return ""

        output = io.StringIO()
        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
            "rfsn_v10_5.launcher.RFSNCache", FakeCache
        ), patch(
            "rfsn_v10_5.launcher.load_tokenizer", fake_load_tokenizer
        ), patch(
            "rfsn_v10_5.launcher.encode_prompt_text", fake_encode_prompt_text
        ), patch(
            "rfsn_v10_5.launcher.decode_token_ids", fake_decode_token_ids
        ), redirect_stdout(output):
            launcher.cmd_generate(args)

        stdout = output.getvalue()
        self.assertEqual(seen["tokenizer_name"], "fake-tokenizer")
        self.assertEqual(seen["prompt_text"], "Hello world")
        self.assertEqual(seen["prompt_ids"], [[5, 6]])
        self.assertIn("[generate] Output IDs: [5, 6, 7, 8]", stdout)
        self.assertIn("[generate] Output text: Hello world extended", stdout)
        self.assertIn("[generate] New text:  extended", stdout)

    def test_serve_command_uses_checkpoint_and_tokenizer(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "serve",
                "--checkpoint",
                "weights.safetensors",
                "--tokenizer",
                "demo-tokenizer",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
            ]
        )
        seen: dict[str, object] = {}

        def fake_create_app(config, *, checkpoint=None, tokenizer_name_or_path=None, model=None, tokenizer=None):
            seen["checkpoint"] = checkpoint
            seen["tokenizer"] = tokenizer_name_or_path
            seen["vocab_size"] = config.vocab_size
            return object()

        class FakeUvicorn:
            @staticmethod
            def run(app, host: str, port: int) -> None:
                seen["app"] = app
                seen["host"] = host
                seen["port"] = port

        with patch.dict("sys.modules", {"uvicorn": FakeUvicorn}), patch(
            "rfsn_v10_5.api.create_app", fake_create_app
        ):
            launcher.cmd_serve(args)

        self.assertEqual(seen["checkpoint"], "weights.safetensors")
        self.assertEqual(seen["tokenizer"], "demo-tokenizer")
        self.assertEqual(seen["host"], "0.0.0.0")
        self.assertEqual(seen["port"], 9000)