"""Benchmarking utilities for the RFSN v10.5 inference engine.

This module provides two functions:

- ``bench_prefill``: measures time-to-first-token (TTFT) for a batch of
  prompt tokens across a range of sequence lengths.
- ``bench_decode``: measures per-token decode throughput (tokens/second)
  for a given number of autoregressive steps.

Both functions return a ``BenchResult`` dataclass with timing statistics.

Usage
-----
::

    import mlx.core as mx
    from rfsn_v10_5 import RFSNMLX, RFSNConfig, RFSNCache
    from rfsn_v10_5.bench import bench_prefill, bench_decode

    config = RFSNConfig(
        hidden_dim=512, num_heads=8, head_dim=64, num_layers=4,
        vocab_size=50000, model_dtype="bfloat16",
        runtime_mode="archived",
    )
    model = RFSNMLX(config)
    cache = RFSNCache(config, batch_size=1)

    result = bench_prefill(model, cache, prompt_len=256, warmup=2, repeats=5)
    print(result)

    result = bench_decode(model, cache, steps=100, warmup=5, repeats=3)
    print(result)

Notes
-----
- Timings are wall-clock times measured with ``time.perf_counter``.
- ``mx.eval`` is called after each forward pass to synchronise the MLX
  graph before stopping the timer.
- Warmup iterations are run before the timed iterations to allow MLX to
  compile and cache the compute graph.
- MLX is Apple Silicon only; this module will raise ``ImportError`` on
  non-Apple platforms unless MLX is installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx

from .config import RFSNConfig
from .cache import RFSNCache
from .model import RFSNMLX


@dataclass
class BenchResult:
    """Timing results from a single benchmark run.

    Attributes
    ----------
    name : str
        Human-readable name of the benchmark.
    mean_ms : float
        Mean wall-clock time per iteration in milliseconds.
    min_ms : float
        Minimum wall-clock time per iteration in milliseconds.
    max_ms : float
        Maximum wall-clock time per iteration in milliseconds.
    throughput : float
        Tokens per second (prompt_len / mean_s for prefill,
        steps / mean_s for decode).
    raw_ms : list[float]
        Per-iteration timings in milliseconds.
    """

    name: str
    mean_ms: float
    min_ms: float
    max_ms: float
    throughput: float
    raw_ms: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.name}: mean={self.mean_ms:.1f}ms  "
            f"min={self.min_ms:.1f}ms  max={self.max_ms:.1f}ms  "
            f"throughput={self.throughput:.1f} tok/s"
        )


def bench_prefill(
    model: RFSNMLX,
    cache: RFSNCache,
    prompt_len: int = 256,
    batch_size: int = 1,
    warmup: int = 2,
    repeats: int = 5,
) -> BenchResult:
    """Benchmark prefill (time-to-first-token).

    Parameters
    ----------
    model : RFSNMLX
    cache : RFSNCache
        A freshly initialised cache (will be reset between iterations).
    prompt_len : int
        Number of prompt tokens to process.
    batch_size : int
        Batch dimension.
    warmup : int
        Number of warmup iterations (not timed).
    repeats : int
        Number of timed iterations.

    Returns
    -------
    BenchResult
    """
    prompt_ids = mx.zeros((batch_size, prompt_len), dtype=mx.int32)

    def _run() -> None:
        # Keep timed iterations focused on model/cache work rather than directory churn.
        cache.reset(clear_persisted=False)
        out = model.prefill(prompt_ids, cache)
        mx.eval(out)

    cache.reset(clear_persisted=True)
    try:
        # Warmup
        for _ in range(warmup):
            _run()

        # Timed iterations
        raw_ms: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _run()
            t1 = time.perf_counter()
            raw_ms.append((t1 - t0) * 1000.0)
    finally:
        cache.reset(clear_persisted=True)

    mean_ms = sum(raw_ms) / len(raw_ms)
    return BenchResult(
        name=f"prefill(len={prompt_len})",
        mean_ms=mean_ms,
        min_ms=min(raw_ms),
        max_ms=max(raw_ms),
        throughput=prompt_len / (mean_ms / 1000.0),
        raw_ms=raw_ms,
    )


def bench_decode(
    model: RFSNMLX,
    cache: RFSNCache,
    steps: int = 100,
    batch_size: int = 1,
    warmup: int = 5,
    repeats: int = 3,
    seed_prompt_len: int = 8,
    archive_seed_steps: int = 0,
) -> BenchResult:
    """Benchmark autoregressive decode throughput.

    Parameters
    ----------
    model : RFSNMLX
    cache : RFSNCache
        Will be seeded with a short prompt before each timed run.
    steps : int
        Number of decode steps to time.
    batch_size : int
        Batch dimension.
    warmup : int
        Number of warmup decode steps before timing.
    repeats : int
        Number of timed runs.
    seed_prompt_len : int
        Length of the seed prompt used to initialise the cache.
    archive_seed_steps : int
        Number of extra decode steps to run after prefill before timing.
        Use this to force hot-tier eviction so the timed decode path
        exercises archived-context attention.

    Returns
    -------
    BenchResult
    """
    seed_ids = mx.zeros((batch_size, seed_prompt_len), dtype=mx.int32)

    def _seed_cache() -> int:
        """Reset cache and run prefill; return next position."""
        cache.reset(clear_persisted=False)
        model.prefill(seed_ids, cache)
        pos = seed_prompt_len
        if archive_seed_steps > 0:
            token_id = mx.zeros((batch_size,), dtype=mx.int32)
            for offset in range(archive_seed_steps):
                out = model.decode_step(token_id, cache, pos + offset)
                mx.eval(out)
                token_id = mx.argmax(out, axis=-1)
            pos += archive_seed_steps
        return pos

    def _run_decode(pos: int) -> None:
        token_id = mx.zeros((batch_size,), dtype=mx.int32)
        for step in range(steps):
            out = model.decode_step(token_id, cache, pos + step)
            mx.eval(out)
            token_id = mx.argmax(out, axis=-1)

    cache.reset(clear_persisted=True)
    try:
        # Warmup
        pos = _seed_cache()
        for _ in range(warmup):
            out = model.decode_step(mx.zeros((batch_size,), dtype=mx.int32), cache, pos)
            mx.eval(out)

        # Timed iterations
        raw_ms: List[float] = []
        for _ in range(repeats):
            pos = _seed_cache()
            t0 = time.perf_counter()
            _run_decode(pos)
            t1 = time.perf_counter()
            raw_ms.append((t1 - t0) * 1000.0)
    finally:
        cache.reset(clear_persisted=True)

    mean_ms = sum(raw_ms) / len(raw_ms)
    return BenchResult(
        name=f"decode(steps={steps})",
        mean_ms=mean_ms,
        min_ms=min(raw_ms),
        max_ms=max(raw_ms),
        throughput=steps / (mean_ms / 1000.0),
        raw_ms=raw_ms,
    )
