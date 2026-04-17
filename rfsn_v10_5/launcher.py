"""Command-line launcher for the RFSN v10.5 inference engine.

This module provides a thin CLI with three sub-commands:

``generate``
    Load a checkpoint and generate text from a prompt.

``bench``
    Run prefill and decode benchmarks and print timing statistics.

``check``
    Instantiate the model and cache, run a short smoke test (prefill +
    decode), and exit 0 on success.

Usage
-----
::

    # Generate text (requires a .safetensors or .npz checkpoint)
    python -m rfsn_v10_5.launcher generate \\
        --checkpoint /path/to/model.safetensors \\
        --hidden-dim 4096 --num-heads 32 --head-dim 128 --num-layers 32 \\
        --num-kv-heads 8 --vocab-size 32000 \\
        --prompt "Once upon a time" \\
        --max-new-tokens 200 --temperature 0.8 --top-p 0.9 --top-k 50

    # Benchmark (no checkpoint needed; uses random weights)
    python -m rfsn_v10_5.launcher bench \\
        --hidden-dim 512 --num-heads 8 --head-dim 64 --num-layers 4 \\
        --prompt-len 256 --decode-steps 100 --cache-dtype fp8_e4m3

    # Smoke test
    python -m rfsn_v10_5.launcher check --cache-dtype fp8_e4m3

Notes
-----
- The launcher is intentionally minimal. For production use, integrate
  ``RFSNMLX`` directly into your application.
- ``tokenizer`` support is not bundled. Pass pre-tokenised integer IDs
  via ``--prompt-ids`` or implement your own tokeniser wrapper.
- MLX is Apple Silicon only; this module will raise ``ImportError`` on
  non-Apple platforms unless MLX is installed.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import mlx.core as mx

from .config import RFSNConfig, RuntimeMode
from .cache import RFSNCache
from .model import RFSNMLX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> RFSNConfig:
    """Construct an ``RFSNConfig`` from parsed CLI arguments."""
    return RFSNConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=getattr(args, "num_kv_heads", 0),
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        hot_capacity=getattr(args, "hot_capacity", 512),
        warm_capacity=getattr(args, "warm_capacity", 2048),
        cold_capacity=getattr(args, "cold_capacity", 8192),
        block_size_seq=getattr(args, "block_size_seq", 64),
        runtime_mode=RuntimeMode.COMPRESSED,
        model_dtype=getattr(args, "model_dtype", "bfloat16"),
        cache_dtype=getattr(args, "cache_dtype", ""),
    )


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    """Load checkpoint and generate text."""
    config = _build_config(args)
    model = RFSNMLX(config)

    if args.checkpoint:
        from .loader import load_hf_weights
        skipped = load_hf_weights(model, args.checkpoint, strict=False)
        if skipped:
            print(f"[loader] Skipped {len(skipped)} keys: {list(skipped.keys())[:5]}")

    cache = RFSNCache(config, batch_size=1)

    if args.prompt_ids:
        prompt_ids = mx.array([list(map(int, args.prompt_ids.split(",")))], dtype=mx.int32)
    else:
        # Fallback: encode prompt as ASCII codepoints (demo only)
        ids = [min(ord(c), config.vocab_size - 1) for c in args.prompt]
        prompt_ids = mx.array([ids], dtype=mx.int32)

    print(f"[generate] Prompt length: {prompt_ids.shape[1]} tokens")
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    if isinstance(generated, list):
        if generated:
            token_ids = mx.concatenate(
                [prompt_ids] + [token[:, None] for token in generated],
                axis=1,
            )
        else:
            token_ids = prompt_ids
    else:
        token_ids = generated

    print(f"[generate] Generated {token_ids.shape[1] - prompt_ids.shape[1]} new tokens")
    print(f"[generate] Output IDs: {token_ids[0].tolist()}")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run prefill and decode benchmarks."""
    from .bench import bench_prefill, bench_decode

    config = _build_config(args)
    model = RFSNMLX(config)
    cache = RFSNCache(config, batch_size=1)

    print(f"[bench] Model: {config.num_layers}L x {config.hidden_dim}d, "
            f"{config.num_heads}H/{config.num_kv_heads}KVH, "
            f"dtype={config.model_dtype}, cache_dtype={config.cache_dtype}")

    prefill_result = bench_prefill(
        model, cache,
        prompt_len=args.prompt_len,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(prefill_result)

    decode_result = bench_decode(
        model, cache,
        steps=args.decode_steps,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(decode_result)


def cmd_check(args: argparse.Namespace) -> None:
    """Run a smoke test: prefill + 5 decode steps."""
    config = RFSNConfig(
        hidden_dim=256, num_heads=4, head_dim=64, num_layers=2,
        vocab_size=1000, hot_capacity=64, warm_capacity=256,
        cold_capacity=1024, block_size_seq=16,
        runtime_mode=RuntimeMode.COMPRESSED,
        model_dtype="float32",
        cache_dtype=getattr(args, "cache_dtype", ""),
    )
    model = RFSNMLX(config)
    cache = RFSNCache(config, batch_size=1)

    print(
        f"[check] Config: dtype={config.model_dtype}, cache_dtype={config.cache_dtype}"
    )

    prompt = mx.zeros((1, 8), dtype=mx.int32)
    out = model.prefill(prompt, cache)
    mx.eval(out)
    print(f"[check] Prefill OK, logits shape: {out.shape}")

    token = mx.zeros((1,), dtype=mx.int32)
    for step in range(5):
        out = model.decode_step(token, cache, pos=8 + step)
        mx.eval(out)
        token = mx.argmax(out, axis=-1)
    print(f"[check] Decode OK, 5 steps completed")
    print("[check] PASS")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rfsn_v10_5.launcher",
        description="RFSN v10.5 inference engine CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared model arguments
    def _add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--hidden-dim", type=int, default=512)
        p.add_argument("--num-heads", type=int, default=8)
        p.add_argument("--num-kv-heads", type=int, default=0,
                       help="KV heads for GQA (0 = same as num-heads)")
        p.add_argument("--head-dim", type=int, default=64)
        p.add_argument("--num-layers", type=int, default=4)
        p.add_argument("--vocab-size", type=int, default=50000)
        p.add_argument("--hot-capacity", type=int, default=512)
        p.add_argument("--warm-capacity", type=int, default=2048)
        p.add_argument("--cold-capacity", type=int, default=8192)
        p.add_argument("--block-size-seq", type=int, default=64)
        p.add_argument("--model-dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"])
        p.add_argument(
            "--cache-dtype",
            type=str,
            default=None,
            choices=["float16", "bfloat16", "float32", "fp8_e4m3"],
            help="Cache storage dtype (defaults to model dtype)",
        )

    # generate sub-command
    gen = sub.add_parser("generate", help="Generate text from a prompt")
    _add_model_args(gen)
    gen.add_argument("--checkpoint", type=str, default=None,
                     help="Path to .safetensors or .npz checkpoint")
    gen.add_argument("--prompt", type=str, default="Hello",
                     help="Text prompt (ASCII codepoints used as token IDs)")
    gen.add_argument("--prompt-ids", type=str, default=None,
                     help="Comma-separated integer token IDs (overrides --prompt)")
    gen.add_argument("--max-new-tokens", type=int, default=50)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.set_defaults(func=cmd_generate)

    # bench sub-command
    bnc = sub.add_parser("bench", help="Run prefill and decode benchmarks")
    _add_model_args(bnc)
    bnc.add_argument("--prompt-len", type=int, default=256)
    bnc.add_argument("--decode-steps", type=int, default=100)
    bnc.add_argument("--warmup", type=int, default=2)
    bnc.add_argument("--repeats", type=int, default=5)
    bnc.set_defaults(func=cmd_bench)

    # check sub-command
    chk = sub.add_parser("check", help="Run a smoke test")
    chk.add_argument(
        "--cache-dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32", "fp8_e4m3"],
        help="Cache storage dtype for the smoke test (defaults to model dtype)",
    )
    chk.set_defaults(func=cmd_check)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = _make_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
