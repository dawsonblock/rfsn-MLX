"""Command-line launcher for the RFSN-MLX V11 runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx

from .cache import RFSNCache
from .config import RFSNConfig, RuntimeMode
from .hf_config import HFConfigError, load_hf_config
from .model import RFSNMLX
from .tokenizer_utils import (
    decode_token_ids,
    encode_messages,
    encode_prompt_text,
    load_tokenizer,
    materialize_generated_sequence,
    prompt_ids_from_list,
)


_MODEL_ARG_DEFAULTS = {
    "hidden_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 0,
    "head_dim": 64,
    "num_layers": 4,
    "vocab_size": 50000,
    "rope_base": 10000.0,
    "ffn_dim": 0,
    "hot_capacity": 512,
    "warm_capacity": 2048,
    "cold_capacity": 8192,
    "block_size_seq": 64,
    "runtime_mode": RuntimeMode.ARCHIVED.value,
    "model_dtype": "bfloat16",
    "disk_cache_dir": "./rfsn_disk_cache",
    "session_id": "",
}


def _build_manual_config(args: argparse.Namespace) -> RFSNConfig:
    return RFSNConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=getattr(args, "num_kv_heads", 0),
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        rope_base=getattr(args, "rope_base", 10000.0),
        ffn_dim=getattr(args, "ffn_dim", 0),
        hot_capacity=getattr(args, "hot_capacity", 512),
        warm_capacity=getattr(args, "warm_capacity", 2048),
        cold_capacity=getattr(args, "cold_capacity", 8192),
        block_size_seq=getattr(args, "block_size_seq", 64),
        runtime_mode=RuntimeMode(getattr(args, "runtime_mode", RuntimeMode.ARCHIVED.value)),
        model_dtype=getattr(args, "model_dtype", "bfloat16"),
        disk_cache_dir=getattr(args, "disk_cache_dir", "./rfsn_disk_cache"),
        session_id=getattr(args, "session_id", ""),
    )


def _build_config(args: argparse.Namespace) -> RFSNConfig:
    checkpoint = getattr(args, "checkpoint", None)
    use_hf_config = checkpoint is not None and not getattr(args, "no_hf_config", False)
    if not use_hf_config:
        return _build_manual_config(args)

    assert checkpoint is not None
    try:
        config = load_hf_config(
            checkpoint,
            hot_capacity=args.hot_capacity,
            warm_capacity=args.warm_capacity,
            cold_capacity=args.cold_capacity,
            block_size_seq=args.block_size_seq,
            runtime_mode=getattr(args, "runtime_mode", RuntimeMode.ARCHIVED.value),
            model_dtype=args.model_dtype,
            disk_cache_dir=args.disk_cache_dir,
            session_id=getattr(args, "session_id", ""),
        )
    except (FileNotFoundError, HFConfigError):
        return _build_manual_config(args)

    config_kwargs = dict(config.__dict__)
    for field, default_value in _MODEL_ARG_DEFAULTS.items():
        if not hasattr(args, field):
            continue
        current_value = getattr(args, field)
        if current_value != default_value:
            config_kwargs[field] = current_value
    return RFSNConfig(**config_kwargs)


def _load_messages_json(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("messages JSON must be a list of message objects")
    if any(not isinstance(message, dict) for message in payload):
        raise ValueError("messages JSON entries must be objects with chat message fields")
    return payload


def _restore_cache_if_requested(cache: RFSNCache, *, enabled: bool) -> int:
    if not enabled:
        return 0
    restored = cache.restore_from_disk()
    return sum(len(layer_manifests) for layer_manifests in restored)


def _aggregate_block_stats(cache: Any) -> dict[str, int]:
    if not hasattr(cache, "layers"):
        return {
            "total_blocks": 0,
            "hot_tokens": 0,
            "warm_ram_blocks": 0,
            "cold_disk_blocks": 0,
            "missing_blocks": 0,
        }

    aggregate = {
        "total_blocks": 0,
        "hot_tokens": 0,
        "warm_ram_blocks": 0,
        "cold_disk_blocks": 0,
        "missing_blocks": 0,
    }
    for layer_cache in cache.layers:
        stats = layer_cache.get_block_stats()
        aggregate["total_blocks"] += stats["total_blocks"]
        aggregate["hot_tokens"] += stats["hot_tokens"]
        aggregate["warm_ram_blocks"] += stats["by_location"]["warm_ram"]["blocks"]
        aggregate["cold_disk_blocks"] += stats["by_location"]["cold_disk"]["blocks"]
        aggregate["missing_blocks"] += stats["by_location"]["missing"]["blocks"]
    return aggregate


def cmd_generate(args: argparse.Namespace) -> None:
    config = _build_config(args)
    model = RFSNMLX(config)

    if args.checkpoint:
        from .loader import load_hf_weights

        skipped = load_hf_weights(model, args.checkpoint, strict=False)
        if skipped:
            print(f"[loader] Skipped {len(skipped)} keys: {list(skipped.keys())[:5]}")

    cache = RFSNCache(config, batch_size=1, restore=args.restore_cache)
    restored_block_count = _restore_cache_if_requested(cache, enabled=args.restore_cache)
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else None

    print(f"[generate] Session ID: {cache.session_id}")

    if args.restore_cache:
        print(
            f"[generate] Restored {restored_block_count} persisted blocks for session '{cache.session_id}' "
            f"from {config.disk_cache_dir}; provided input will be appended as continuation context"
        )

    if args.prompt_ids:
        prompt_ids = prompt_ids_from_list(
            (token_id.strip() for token_id in args.prompt_ids.split(",")),
            vocab_size=config.vocab_size,
        )
    elif args.messages_json:
        if tokenizer is None:
            raise ValueError("Message prompts require --tokenizer with chat-template support")
        prompt_ids = encode_messages(
            tokenizer,
            _load_messages_json(args.messages_json),
            vocab_size=config.vocab_size,
            add_generation_prompt=True,
        )
    elif tokenizer is not None:
        prompt_ids = encode_prompt_text(tokenizer, args.prompt, vocab_size=config.vocab_size)
    else:
        raise ValueError("Text prompts require --tokenizer, or provide --prompt-ids directly")

    print(f"[generate] Prompt length: {prompt_ids.shape[1]} tokens")
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        cache=cache,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    token_ids = materialize_generated_sequence(prompt_ids, generated)

    print(f"[generate] Generated {token_ids.shape[1] - prompt_ids.shape[1]} new tokens")
    print(f"[generate] Output IDs: {token_ids[0].tolist()}")
    block_stats = _aggregate_block_stats(cache)
    print(
        "[generate] Block stats: "
        f"total_blocks={block_stats['total_blocks']} "
        f"warm_ram={block_stats['warm_ram_blocks']} "
        f"cold_disk={block_stats['cold_disk_blocks']} "
        f"missing={block_stats['missing_blocks']} "
        f"hot_tokens={block_stats['hot_tokens']}"
    )
    if tokenizer is not None:
        print(f"[generate] Output text: {decode_token_ids(tokenizer, token_ids)}")
        if token_ids.shape[1] > prompt_ids.shape[1]:
            print(
                "[generate] New text: "
                f"{decode_token_ids(tokenizer, token_ids[:, prompt_ids.shape[1]:])}"
            )


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Serving requires the 'uvicorn' package. Install it with `python -m pip install uvicorn`."
        ) from exc

    from .api import create_app

    config = _build_config(args)
    app = create_app(
        config,
        checkpoint=args.checkpoint,
        tokenizer_name_or_path=args.tokenizer,
        max_concurrent_requests=args.max_concurrent_requests,
        max_queue_size=args.max_queue_size,
    )
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_bench(args: argparse.Namespace) -> None:
    from .bench import bench_decode, bench_prefill

    config = _build_config(args)
    model = RFSNMLX(config)
    cache = RFSNCache(config, batch_size=1)

    print(
        f"[bench] Model: {config.num_layers}L x {config.hidden_dim}d, "
        f"{config.num_heads}H/{config.num_kv_heads}KVH, "
        f"dtype={config.model_dtype}, runtime_mode={config.runtime_mode.value}, session_id={cache.session_id}"
    )

    prefill_result = bench_prefill(
        model,
        cache,
        prompt_len=args.prompt_len,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(prefill_result)

    decode_result = bench_decode(
        model,
        cache,
        steps=args.decode_steps,
        warmup=args.warmup,
        repeats=args.repeats,
        seed_prompt_len=args.seed_prompt_len,
        archive_seed_steps=args.archive_seed_steps,
    )
    print(decode_result)


def cmd_check(args: argparse.Namespace) -> None:
    config = RFSNConfig(
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        num_layers=2,
        vocab_size=1000,
        hot_capacity=64,
        warm_capacity=256,
        cold_capacity=1024,
        block_size_seq=16,
        runtime_mode=RuntimeMode(getattr(args, "runtime_mode", RuntimeMode.ARCHIVED.value)),
        model_dtype="float32",
        disk_cache_dir=getattr(args, "disk_cache_dir", "./rfsn_disk_cache"),
        session_id=getattr(args, "session_id", ""),
    )
    model = RFSNMLX(config)
    cache = RFSNCache(config, batch_size=1)

    print(
        f"[check] Config: dtype={config.model_dtype}, "
        f"runtime_mode={config.runtime_mode.value}, session_id={cache.session_id}"
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
    print("[check] Decode OK, 5 steps completed")
    print("[check] PASS")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rfsn_v10_5.launcher",
        description="RFSN-MLX V11 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--hidden-dim", type=int, default=512)
        p.add_argument("--num-heads", type=int, default=8)
        p.add_argument("--num-kv-heads", type=int, default=0,
                       help="KV heads for GQA (0 = same as num-heads)")
        p.add_argument("--head-dim", type=int, default=64)
        p.add_argument("--num-layers", type=int, default=4)
        p.add_argument("--vocab-size", type=int, default=50000)
        p.add_argument("--rope-base", type=float, default=10000.0)
        p.add_argument("--ffn-dim", type=int, default=0)
        p.add_argument("--hot-capacity", type=int, default=512)
        p.add_argument("--warm-capacity", type=int, default=2048)
        p.add_argument("--cold-capacity", type=int, default=8192)
        p.add_argument("--block-size-seq", type=int, default=64)
        p.add_argument(
            "--runtime-mode",
            type=str,
            default=RuntimeMode.ARCHIVED.value,
            choices=[mode.value for mode in RuntimeMode],
            help="'archived' spills exact sealed prefixes to archived blocks; 'exact' is hot-window only",
        )
        p.add_argument("--disk-cache-dir", type=str, default="./rfsn_disk_cache")
        p.add_argument(
            "--session-id",
            type=str,
            default="",
            help="Explicit cache session identifier. Required for restore-cache; auto-generated otherwise.",
        )
        p.add_argument("--model-dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"])

    gen = sub.add_parser("generate", help="Generate text from a prompt")
    _add_model_args(gen)
    gen.add_argument("--checkpoint", type=str, default=None,
                     help="Path to a checkpoint file, sharded index, or model directory")
    gen.add_argument("--prompt", type=str, default="Hello",
                     help="Text prompt when using --tokenizer")
    gen.add_argument("--messages-json", type=str, default=None,
                     help="Path to a JSON array of chat messages formatted through the tokenizer template")
    gen.add_argument("--prompt-ids", type=str, default=None,
                     help="Comma-separated integer token IDs (overrides text and messages)")
    gen.add_argument("--tokenizer", type=str, default=None,
                     help="HuggingFace tokenizer name or local path for text encode/decode")
    gen.add_argument("--restore-cache", action="store_true",
                     help="Restore persisted KV blocks from --disk-cache-dir before appending the provided prompt")
    gen.add_argument("--no-hf-config", action="store_true",
                     help="Disable automatic config.json loading from the checkpoint path")
    gen.add_argument("--max-new-tokens", type=int, default=50)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.set_defaults(func=cmd_generate)

    bnc = sub.add_parser("bench", help="Run prefill and decode benchmarks")
    _add_model_args(bnc)
    bnc.add_argument("--prompt-len", type=int, default=256)
    bnc.add_argument("--decode-steps", type=int, default=100)
    bnc.add_argument("--warmup", type=int, default=2)
    bnc.add_argument("--repeats", type=int, default=5)
    bnc.add_argument("--seed-prompt-len", type=int, default=8)
    bnc.add_argument("--archive-seed-steps", type=int, default=0)
    bnc.set_defaults(func=cmd_bench)

    chk = sub.add_parser("check", help="Run a smoke test")
    chk.add_argument(
        "--runtime-mode",
        type=str,
        default=RuntimeMode.ARCHIVED.value,
        choices=[mode.value for mode in RuntimeMode],
    )
    chk.add_argument("--disk-cache-dir", type=str, default="./rfsn_disk_cache")
    chk.add_argument("--session-id", type=str, default="")
    chk.set_defaults(func=cmd_check)

    srv = sub.add_parser("serve", help="Run the FastAPI inference wrapper")
    _add_model_args(srv)
    srv.add_argument("--checkpoint", type=str, default=None,
                     help="Path to a checkpoint file, sharded index, or model directory")
    srv.add_argument("--tokenizer", type=str, default=None,
                     help="HuggingFace tokenizer name or local path for text encode/decode")
    srv.add_argument("--no-hf-config", action="store_true",
                     help="Disable automatic config.json loading from the checkpoint path")
    srv.add_argument("--host", type=str, default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--max-concurrent-requests", type=int, default=1)
    srv.add_argument("--max-queue-size", type=int, default=0)
    srv.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _make_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
