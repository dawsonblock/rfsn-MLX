from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from rfsn_v10_5 import launcher


class LauncherTest(unittest.TestCase):
    def test_build_config_accepts_runtime_mode_and_session_id(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "bench",
                "--runtime-mode",
                "exact",
                "--session-id",
                "cli-session",
            ]
        )

        config = launcher._build_config(args)

        self.assertEqual(config.runtime_mode, launcher.RuntimeMode.EXACT)
        self.assertEqual(config.session_id, "cli-session")

    def test_build_config_accepts_disk_cache_dir_override(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(["bench", "--disk-cache-dir", "/tmp/rfsn-cache"])

        config = launcher._build_config(args)

        self.assertEqual(config.disk_cache_dir, "/tmp/rfsn-cache")

    def test_build_config_accepts_explicit_ffn_dim_and_rope_base(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "bench",
                "--hidden-dim",
                "3072",
                "--num-heads",
                "24",
                "--num-kv-heads",
                "8",
                "--head-dim",
                "128",
                "--num-layers",
                "28",
                "--vocab-size",
                "128256",
                "--ffn-dim",
                "8192",
                "--rope-base",
                "500000",
            ]
        )

        config = launcher._build_config(args)

        self.assertEqual(config.ffn_dim, 8192)
        self.assertEqual(config.rope_base, 500000.0)

    def test_build_config_prefers_hf_autoconfig_from_checkpoint(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(["generate", "--checkpoint", "/tmp/model"])

        hf_config = launcher.RFSNConfig(
            hidden_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            num_layers=32,
            vocab_size=128256,
            rope_base=500000.0,
            ffn_dim=14336,
        )

        with patch("rfsn_v10_5.launcher.load_hf_config", return_value=hf_config):
            config = launcher._build_config(args)

        self.assertEqual(config.hidden_dim, 4096)
        self.assertEqual(config.vocab_size, 128256)

    def test_check_command_uses_runtime_mode_and_session_id(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(["check", "--runtime-mode", "exact", "--session-id", "health-check"])
        seen: dict[str, object] = {}

        class FakeCache:
            def __init__(self, config, batch_size: int, **kwargs) -> None:
                seen["runtime_mode"] = config.runtime_mode
                self.session_id = kwargs.get("session_id", config.session_id or "health-check") or "health-check"
                self.config = config
                self.batch_size = batch_size

        class FakeModel:
            def __init__(self, config) -> None:
                seen["model_runtime_mode"] = config.runtime_mode

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
        self.assertEqual(seen["runtime_mode"], launcher.RuntimeMode.EXACT)
        self.assertEqual(seen["model_runtime_mode"], launcher.RuntimeMode.EXACT)
        self.assertIn("runtime_mode=exact", stdout)
        self.assertIn("session_id=health-check", stdout)
        self.assertIn("[check] PASS", stdout)

    def test_generate_command_materializes_sequence_from_generated_steps(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(["generate", "--prompt-ids", "1,2", "--max-new-tokens", "2"])

        class FakeCache:
            def __init__(self, config, batch_size: int, **kwargs) -> None:
                self.config = config
                self.batch_size = batch_size
                self.session_id = kwargs.get("session_id", config.session_id or "generated-session") or "generated-session"

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
                return [mx.array([3], dtype=mx.int32), mx.array([4], dtype=mx.int32)]

        output = io.StringIO()
        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
            "rfsn_v10_5.launcher.RFSNCache", FakeCache
        ), redirect_stdout(output):
            launcher.cmd_generate(args)

        stdout = output.getvalue()
        self.assertIn("[generate] Session ID:", stdout)
        self.assertIn("[generate] Generated 2 new tokens", stdout)
        self.assertIn("[generate] Output IDs: [1, 2, 3, 4]", stdout)

    def test_generate_command_validates_prompt_ids_against_vocab_size(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "generate",
                "--prompt-ids",
                "1,99",
                "--vocab-size",
                "16",
            ]
        )

        class FakeCache:
            def __init__(self, config, batch_size: int, **kwargs) -> None:
                self.config = config
                self.batch_size = batch_size
                self.session_id = kwargs.get("session_id", config.session_id or "generated-session") or "generated-session"

        class FakeModel:
            def __init__(self, config) -> None:
                self.config = config

        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
            "rfsn_v10_5.launcher.RFSNCache", FakeCache
        ):
            with self.assertRaisesRegex(ValueError, "vocab_size=16"):
                launcher.cmd_generate(args)

    def test_generate_command_restores_persisted_cache_when_requested(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "generate",
                "--prompt-ids",
                "1,2",
                "--restore-cache",
                "--disk-cache-dir",
                "/tmp/rfsn-cache",
                "--session-id",
                "restore-session",
                "--max-new-tokens",
                "2",
            ]
        )
        seen: dict[str, object] = {}

        class FakeCache:
            def __init__(self, config, batch_size: int, **kwargs) -> None:
                seen["disk_cache_dir"] = config.disk_cache_dir
                seen["session_id"] = kwargs.get("session_id", config.session_id)
                self.config = config
                self.batch_size = batch_size
                self.session_id = kwargs.get("session_id", config.session_id or "restore-session") or "restore-session"

            def restore_from_disk(self):
                seen["restored"] = True
                return [[object(), object()]]

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
                return [mx.array([3], dtype=mx.int32), mx.array([4], dtype=mx.int32)]

        output = io.StringIO()
        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
            "rfsn_v10_5.launcher.RFSNCache", FakeCache
        ), redirect_stdout(output):
            launcher.cmd_generate(args)

        stdout = output.getvalue()
        self.assertTrue(seen["restored"])
        self.assertEqual(seen["disk_cache_dir"], "/tmp/rfsn-cache")
        self.assertEqual(seen["session_id"], "restore-session")
        self.assertIn("Restored 2 persisted blocks", stdout)

    def test_generate_command_requires_session_id_for_restore(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "generate",
                "--prompt-ids",
                "1,2",
                "--restore-cache",
                "--max-new-tokens",
                "2",
            ]
        )

        class FakeModel:
            def __init__(self, config) -> None:
                self.config = config

        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel):
            with self.assertRaisesRegex(ValueError, "session_id"):
                launcher.cmd_generate(args)

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
            def __init__(self, config, batch_size: int, **kwargs) -> None:
                self.config = config
                self.batch_size = batch_size
                self.session_id = kwargs.get("session_id", config.session_id or "generated-session") or "generated-session"

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
                return [mx.array([7], dtype=mx.int32), mx.array([8], dtype=mx.int32)]

        def fake_load_tokenizer(name_or_path: str):
            seen["tokenizer_name"] = name_or_path
            return object()

        def fake_encode_prompt_text(tokenizer, text: str, *, vocab_size: int, add_special_tokens: bool = True):
            seen["prompt_text"] = text
            seen["vocab_size"] = vocab_size
            return mx.array([[5, 6]], dtype=mx.int32)

        def fake_decode_token_ids(tokenizer, token_ids, *, skip_special_tokens: bool = True):
            as_list = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
            if as_list == [[5, 6, 7, 8]]:
                return "Hello world extended"
            if as_list == [[7, 8]]:
                return " extended"
            return ""

        output = io.StringIO()
        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
            "rfsn_v10_5.launcher.RFSNCache", FakeCache
        ), patch("rfsn_v10_5.launcher.load_tokenizer", fake_load_tokenizer), patch(
            "rfsn_v10_5.launcher.encode_prompt_text", fake_encode_prompt_text
        ), patch("rfsn_v10_5.launcher.decode_token_ids", fake_decode_token_ids), redirect_stdout(output):
            launcher.cmd_generate(args)

        stdout = output.getvalue()
        self.assertEqual(seen["tokenizer_name"], "fake-tokenizer")
        self.assertEqual(seen["prompt_text"], "Hello world")
        self.assertEqual(seen["prompt_ids"], [[5, 6]])
        self.assertIn("[generate] Output IDs: [5, 6, 7, 8]", stdout)
        self.assertIn("[generate] Output text: Hello world extended", stdout)
        self.assertIn("[generate] New text:  extended", stdout)

    def test_generate_command_uses_messages_json_with_chat_template(self) -> None:
        parser = launcher._make_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            messages_path = f"{tmpdir}/messages.json"
            Path(messages_path).write_text(json.dumps([{"role": "user", "content": "Hello"}]), encoding="utf-8")
            args = parser.parse_args(
                [
                    "generate",
                    "--messages-json",
                    messages_path,
                    "--tokenizer",
                    "fake-tokenizer",
                    "--max-new-tokens",
                    "2",
                ]
            )

            seen: dict[str, object] = {}

            class FakeCache:
                def __init__(self, config, batch_size: int, **kwargs) -> None:
                    self.config = config
                    self.batch_size = batch_size
                    self.layers = []
                    self.session_id = kwargs.get("session_id", config.session_id or "generated-session") or "generated-session"

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
                    return [mx.array([7], dtype=mx.int32), mx.array([8], dtype=mx.int32)]

            def fake_load_tokenizer(name_or_path: str):
                seen["tokenizer_name"] = name_or_path
                return object()

            def fake_encode_messages(tokenizer, messages, *, vocab_size: int, add_generation_prompt: bool = True):
                seen["messages"] = messages
                return mx.array([[9, 10]], dtype=mx.int32)

            output = io.StringIO()
            with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
                "rfsn_v10_5.launcher.RFSNCache", FakeCache
            ), patch("rfsn_v10_5.launcher.load_tokenizer", fake_load_tokenizer), patch(
                "rfsn_v10_5.launcher.encode_messages", fake_encode_messages
            ), patch("rfsn_v10_5.launcher.decode_token_ids", lambda tokenizer, token_ids, **_: "decoded"), redirect_stdout(output):
                launcher.cmd_generate(args)

        self.assertEqual(seen["tokenizer_name"], "fake-tokenizer")
        self.assertEqual(seen["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(seen["prompt_ids"], [[9, 10]])

    def test_serve_command_uses_checkpoint_tokenizer_and_admission_limits(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "serve",
                "--checkpoint",
                "weights.safetensors",
                "--tokenizer",
                "demo-tokenizer",
                "--disk-cache-dir",
                "/tmp/rfsn-cache",
                "--max-concurrent-requests",
                "2",
                "--max-queue-size",
                "4",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
            ]
        )
        seen: dict[str, object] = {}

        def fake_create_app(
            config,
            *,
            checkpoint=None,
            tokenizer_name_or_path=None,
            model=None,
            tokenizer=None,
            max_concurrent_requests=1,
            max_queue_size=0,
        ):
            seen["checkpoint"] = checkpoint
            seen["tokenizer"] = tokenizer_name_or_path
            seen["vocab_size"] = config.vocab_size
            seen["disk_cache_dir"] = config.disk_cache_dir
            seen["max_concurrent_requests"] = max_concurrent_requests
            seen["max_queue_size"] = max_queue_size
            return object()

        class FakeUvicorn:
            @staticmethod
            def run(app, host: str, port: int) -> None:
                seen["app"] = app
                seen["host"] = host
                seen["port"] = port

        with patch.dict("sys.modules", {"uvicorn": FakeUvicorn}), patch("rfsn_v10_5.api.create_app", fake_create_app):
            launcher.cmd_serve(args)

        self.assertEqual(seen["checkpoint"], "weights.safetensors")
        self.assertEqual(seen["tokenizer"], "demo-tokenizer")
        self.assertEqual(seen["disk_cache_dir"], "/tmp/rfsn-cache")
        self.assertEqual(seen["max_concurrent_requests"], 2)
        self.assertEqual(seen["max_queue_size"], 4)
        self.assertEqual(seen["host"], "0.0.0.0")
        self.assertEqual(seen["port"], 9000)

    def test_bench_command_passes_archive_seed_options(self) -> None:
        parser = launcher._make_parser()
        args = parser.parse_args(
            [
                "bench",
                "--seed-prompt-len",
                "32",
                "--archive-seed-steps",
                "6",
            ]
        )
        seen: dict[str, object] = {}

        class FakeCache:
            def __init__(self, config, batch_size: int, **kwargs) -> None:
                self.config = config
                self.batch_size = batch_size
                self.session_id = kwargs.get("session_id", config.session_id or "generated-session") or "generated-session"

        class FakeModel:
            def __init__(self, config) -> None:
                self.config = config

        def fake_bench_prefill(model, cache, *, prompt_len: int, warmup: int, repeats: int):
            seen["prefill_prompt_len"] = prompt_len
            return "prefill-ok"

        def fake_bench_decode(
            model,
            cache,
            *,
            steps: int,
            warmup: int,
            repeats: int,
            seed_prompt_len: int,
            archive_seed_steps: int,
        ):
            seen["steps"] = steps
            seen["seed_prompt_len"] = seed_prompt_len
            seen["archive_seed_steps"] = archive_seed_steps
            return "decode-ok"

        output = io.StringIO()
        with patch("rfsn_v10_5.launcher.RFSNMLX", FakeModel), patch(
            "rfsn_v10_5.launcher.RFSNCache", FakeCache
        ), patch("rfsn_v10_5.bench.bench_prefill", fake_bench_prefill), patch(
            "rfsn_v10_5.bench.bench_decode", fake_bench_decode
        ), redirect_stdout(output):
            launcher.cmd_bench(args)

        stdout = output.getvalue()
        self.assertEqual(seen["prefill_prompt_len"], 256)
        self.assertEqual(seen["steps"], 100)
        self.assertEqual(seen["seed_prompt_len"], 32)
        self.assertEqual(seen["archive_seed_steps"], 6)
        self.assertIn("prefill-ok", stdout)
        self.assertIn("decode-ok", stdout)


if __name__ == "__main__":
    unittest.main()
