from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Callable

import mlx.core as mx

from rfsn_v10_5.bench import bench_decode
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode
from rfsn_v10_5.model import RFSNMLX


class PerformanceSmokeTest(unittest.TestCase):
    def _build_components(self) -> tuple[RFSNMLX, RFSNCache]:
        config = RFSNConfig(
            hidden_dim=64,
            num_heads=4,
            num_kv_heads=4,
            head_dim=16,
            num_layers=1,
            vocab_size=128,
            hot_capacity=8,
            warm_capacity=16,
            cold_capacity=32,
            block_size_seq=4,
            runtime_mode=RuntimeMode.ARCHIVED,
        )
        model = RFSNMLX(config)
        cache = RFSNCache(config, batch_size=1)
        return model, cache

    def _seed_archived_context(self, model: RFSNMLX, cache: RFSNCache) -> mx.array:
        prompt = mx.zeros((1, 8), dtype=mx.int32)
        logits = model.prefill(prompt, cache)
        mx.eval(logits)

        token = mx.zeros((1,), dtype=mx.int32)
        for pos in range(8, 10):
            logits = model.decode_step(token, cache, pos)
            mx.eval(logits)
            token = mx.argmax(logits, axis=-1).astype(mx.int32)

        layer_cache = cache.layer(0)
        self.assertGreater(layer_cache.get_block_stats()["total_blocks"], 0)
        return token

    def _start_capture(self, directory: Path) -> tuple[Path, Callable[[], None]]:
        def _fallback_capture(reason: str) -> tuple[Path, Callable[[], None]]:
            artifact = directory / "decode_step.capture.txt"

            def _write_report() -> None:
                lines = [
                    "capture_mode=fallback",
                    f"reason={reason}",
                    f"has_profiler={hasattr(mx, 'profiler')}",
                    f"has_metal={hasattr(mx, 'metal')}",
                ]
                metal = getattr(mx, "metal", None)
                if metal is not None:
                    if hasattr(metal, "is_available"):
                        lines.append(f"metal_available={metal.is_available()}")
                if hasattr(mx, "get_peak_memory"):
                    lines.append(f"peak_memory={mx.get_peak_memory()}")
                artifact.write_text("\n".join(lines), encoding="utf-8")

            return artifact, _write_report

        profiler = getattr(mx, "profiler", None)
        if profiler is not None:
            start = getattr(profiler, "start", None)
            stop = getattr(profiler, "stop", None)
            if callable(start) and callable(stop):
                artifact = directory / "decode_step.profile"
                try:
                    start(str(artifact))
                except RuntimeError as exc:
                    return _fallback_capture(f"mx.profiler unavailable: {exc}")
                return artifact, stop

        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "start_capture") and hasattr(metal, "stop_capture"):
            artifact = directory / "decode_step.gputrace"
            try:
                metal.start_capture(str(artifact))
            except RuntimeError as exc:
                return _fallback_capture(f"metal capture unavailable: {exc}")
            return artifact, metal.stop_capture

        return _fallback_capture("neither mx.profiler nor mx.metal capture is available")

    def _assert_capture_artifact(self, artifact_path: Path) -> None:
        self.assertTrue(artifact_path.exists())
        if artifact_path.is_dir():
            self.assertTrue(any(artifact_path.iterdir()))
        else:
            self.assertGreater(artifact_path.stat().st_size, 0)

    def test_archived_decode_capture_and_cache_reuse(self) -> None:
        model, cache = self._build_components()
        token = self._seed_archived_context(model, cache)
        layer_cache = cache.layer(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path, stop_capture = self._start_capture(Path(tmpdir))
            try:
                archived_segments_before = layer_cache.get_archived_attention_segments()
                logits = model.decode_step(token, cache, 10)
                mx.eval(logits)

                next_token = mx.argmax(logits, axis=-1).astype(mx.int32)
                logits = model.decode_step(next_token, cache, 11)
                mx.eval(logits)
                archived_segments_after = layer_cache.get_archived_attention_segments()
            finally:
                stop_capture()

            self.assertTrue(archived_segments_before)
            self.assertTrue(archived_segments_after)
            self._assert_capture_artifact(artifact_path)

    def test_bench_decode_archive_seed_runs(self) -> None:
        model, cache = self._build_components()
        result = bench_decode(
            model,
            cache,
            steps=4,
            warmup=1,
            repeats=1,
            seed_prompt_len=8,
            archive_seed_steps=2,
        )
        self.assertGreater(result.mean_ms, 0.0)
        self.assertGreater(result.throughput, 0.0)


if __name__ == "__main__":
    unittest.main()
