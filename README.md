# rfsn-MLX

An MLX-native transformer inference engine for Apple Silicon with tiered KV caching, compressed long-context archival, grouped-query attention, and an FP8 cache-storage path.

> Status: inference backend and benchmark harness. The exact path, compressed archive path, packed FP8 cache mode, tokenizer-backed CLI generation, cold-tier pruning, and a thin FastAPI wrapper are implemented. Continuous batching and custom Metal kernels remain future work.

## Highlights

| Area | What is implemented | Why it matters |
| --- | --- | --- |
| Exact attention | MLX-based exact attention is the semantic reference path | Keeps correctness anchored while optimizations evolve |
| Long-context archive | Hot KV overflow is evicted into warm RAM and cold disk tiers | Lets decode continue beyond device-resident context limits |
| Compression | PQ + RVQ are used to compact archived keys | Reduces archive footprint while keeping reconstruction vectorized |
| FP8 cache mode | `cache_dtype=fp8_e4m3` stores hot and archived values in 1-byte form | Cuts cache memory pressure on runtimes without native float8 |
| GQA support | `num_kv_heads` can be smaller than `num_heads` | Matches modern LLM attention layouts |
| Decode-path optimization | Mixed-context attention operates on cached segments instead of rebuilding full context each step | Removes a major per-token reconstruction cost |
| Benchmarking | Built-in smoke checks plus prefill/decode timing helpers | Makes regressions easy to catch locally |

## Architecture

```mermaid
flowchart LR
    A[Token IDs] --> B[Embedding]
    B --> C[Transformer Layers]
    C --> D[Final Norm]
    D --> E[Weight-tied LM head]
    E --> F[Logits]

    subgraph Layer Step
      C1[Q/K/V projection] --> C2[RoPE]
      C2 --> C3[Cache append / eviction]
      C3 --> C4[Exact or mixed attention]
      C4 --> C5[SwiGLU FFN]
    end
```

### Cache pipeline

```mermaid
flowchart LR
    H[Hot tier on device\nring buffer] -->|oldest prefix evicted| W[Warm tier in RAM\nencoded keys + compact V payload]
    W -->|spill when warm grows| C[Cold tier on disk\nserialized blocks]
    W --> M[Combined archived context cache]
    C --> M
    M --> A[Segmented mixed attention]
    H --> A
```

### Performance-oriented design choices

- RoPE tables are hoisted once per forward call and reused across layers.
- The hot tier is a preallocated ring buffer, so append is constant-time and avoids compaction copies.
- Archived RVQ metadata is stored in fixed-width tensors instead of Python-managed sparse objects.
- Decode uses archived and hot attention segments directly instead of rebuilding one monolithic context tensor every token.
- Archived context reuse is cached at the combined archive level, avoiding duplicate per-block reconstructed tensors staying resident.

## What the repository is for

This project is best thought of as an inference-systems playground with real engineering constraints:

- exact short-context inference
- compressed long-context inference once hot capacity is exceeded
- experimentation with cache storage dtypes separate from model dtypes
- loading HuggingFace-style LLaMA/Mistral checkpoints
- benchmarking prefill and decode behavior on Apple Silicon

It is not currently a full serving stack. A thin HTTP API and tokenizer-backed CLI are included, but there is still no request batching layer, scheduler, distributed runtime, or in-repo checkpoint-specific tokenizer assets.

## Requirements

- macOS on Apple Silicon
- Python 3.10+
- `mlx`
- `numpy`
- `transformers` for tokenizer-backed text generation
- `fastapi` and `uvicorn` for the HTTP wrapper
- optional: `safetensors` if you want to load external checkpoints outside MLX-native formats

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Minimal local runtime for smoke checks and benchmarks:

```bash
python -m pip install -r requirements-core.txt
```

Full local runtime for checkpoint loading, tokenizer-backed text I/O, and HTTP serving:

```bash
python -m pip install -r requirements.txt
```

If you only need random-weight smoke checks and do not plan to load checkpoints, `requirements-core.txt` is enough. If you only plan to drive the engine with `--prompt-ids`, tokenizer-backed text I/O remains optional.

### Apple Silicon notes

- MLX handles Apple Silicon acceleration automatically; there is no CUDA setup path in this repo.
- This project targets the MLX runtime on Apple Silicon, not a guaranteed explicit Neural Engine execution path.
- On an M1, M2, or M3 Mac, the recommended operator flow is `check` -> `bench` -> `generate` or `serve`.

## Quick start

### 1. Smoke-test the engine

```bash
python -m rfsn_v10_5.launcher check
python -m rfsn_v10_5.launcher check --cache-dtype fp8_e4m3
```

Expected shape of the output:

```text
[check] Config: dtype=float32, cache_dtype=fp8_e4m3
[check] Prefill OK, logits shape: (1, 8, 1000)
[check] Decode OK, 5 steps completed
[check] PASS
```

### 2. Run a tiny benchmark

```bash
python -m rfsn_v10_5.launcher bench \
  --hidden-dim 128 \
  --num-heads 2 \
  --num-kv-heads 2 \
  --head-dim 64 \
  --num-layers 1 \
  --vocab-size 128 \
  --prompt-len 8 \
  --decode-steps 8 \
  --warmup 1 \
  --repeats 1 \
  --model-dtype float16 \
  --cache-dtype fp8_e4m3
```

Example output from the tiny validation configuration used in this workspace:

```text
[bench] Model: 1L x 128d, 2H/2KVH, dtype=float16, cache_dtype=fp8_e4m3
prefill(len=8): mean=2.1ms  min=2.1ms  max=2.1ms  throughput=3773.1 tok/s
decode(steps=8): mean=13.6ms  min=13.6ms  max=13.6ms  throughput=587.6 tok/s
```

Treat those numbers as sanity-check output, not a throughput claim for real model sizes.

### 3. Generate from a checkpoint

```bash
python -m rfsn_v10_5.launcher generate \
  --checkpoint /path/to/model.safetensors \
  --tokenizer /path/to/tokenizer-or-hf-id \
  --hidden-dim 4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --num-layers 32 \
  --vocab-size 32000 \
  --prompt "Once upon a time" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 50 \
  --cache-dtype fp8_e4m3
```

Notes:

- `--tokenizer` accepts a HuggingFace tokenizer ID or a local tokenizer path.
- `--prompt-ids` remains available as the low-level escape hatch when you already have token IDs.
- If no tokenizer is supplied, `--prompt` still falls back to ASCII codepoints for demo/debug use only.
- The `bench` subcommand uses random weights; only `generate` needs a checkpoint.

### 4. Run the HTTP API

```bash
python -m rfsn_v10_5.launcher serve \
  --checkpoint /path/to/model.safetensors \
  --tokenizer /path/to/tokenizer-or-hf-id \
  --hidden-dim 4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --num-layers 32 \
  --vocab-size 32000 \
  --host 127.0.0.1 \
  --port 8000
```

Example requests:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Once upon a time","max_new_tokens":32}'

curl -N -X POST http://127.0.0.1:8000/generate/stream \
  -H 'content-type: application/json' \
  -d '{"prompt":"Once upon a time","max_new_tokens":8}'
```

### API schema

`GET /health`

Response body:

```json
{
  "status": "ok",
  "vocab_size": 32000,
  "tokenizer_loaded": true
}
```

`POST /generate`

Request body:

```json
{
  "prompt": "Once upon a time",
  "prompt_ids": [1, 2, 3],
  "max_new_tokens": 32,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 50,
  "repetition_penalty": 1.0
}
```

Use either `prompt` or `prompt_ids`. Response body:

```json
{
  "prompt_token_count": 4,
  "generated_token_count": 32,
  "token_ids": [1, 2, 3, 4, 5],
  "text": "Once upon a time ...",
  "generated_text": "..."
}
```

`POST /generate/stream`

The request body matches `/generate`. The route emits Server-Sent Events with:

- `token`: `{"step": 1, "token_id": 7, "text": "hello"}`
- `complete`: `{"token_ids": [...], "generated_token_count": 32, "text": "...", "generated_text": "..."}`

The streaming route emits Server-Sent Events. It is still single-request-at-a-time inference and does not implement continuous batching.

## Python API

### Minimal prefill + decode loop

```python
import mlx.core as mx

from rfsn_v10_5 import RFSNCache, RFSNConfig, RFSNMLX, RuntimeMode

config = RFSNConfig(
    hidden_dim=512,
    num_heads=8,
    num_kv_heads=8,
    head_dim=64,
    num_layers=4,
    vocab_size=32000,
    runtime_mode=RuntimeMode.COMPRESSED,
    model_dtype="bfloat16",
    cache_dtype="fp8_e4m3",
)

model = RFSNMLX(config)
cache = RFSNCache(config, batch_size=1)

prompt_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
logits = model.prefill(prompt_ids, cache)
mx.eval(logits)

token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
pos = prompt_ids.shape[1]

for _ in range(8):
    logits = model.decode_step(token, cache, pos)
    mx.eval(logits)
    token = mx.argmax(logits, axis=-1).astype(mx.int32)
    pos += 1
```

### Loading HuggingFace-style weights

```python
from rfsn_v10_5 import RFSNConfig, RFSNMLX
from rfsn_v10_5.loader import load_hf_weights

config = RFSNConfig(
    hidden_dim=4096,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    num_layers=32,
    vocab_size=32000,
    model_dtype="bfloat16",
)

model = RFSNMLX(config)
load_hf_weights(model, "/path/to/model.safetensors", strict=False)
```

Supported checkpoint formats:

- `.safetensors`
- `.npz`

The loader remaps common LLaMA/Mistral key names into the local module layout and skips `lm_head.weight` because the LM head is weight-tied to embeddings.

### Benchmark helpers

```python
from rfsn_v10_5.bench import bench_decode, bench_prefill

prefill = bench_prefill(model, cache, prompt_len=256, warmup=2, repeats=5)
decode = bench_decode(model, cache, steps=100, warmup=5, repeats=3)

print(prefill)
print(decode)
```

Use `archive_seed_steps` in `bench_decode(...)` when you explicitly want the timed decode loop to run with archived context already present.

### FastAPI wrapper

```python
from rfsn_v10_5 import RFSNConfig, RuntimeMode, create_app

config = RFSNConfig(
  hidden_dim=512,
  num_heads=8,
  num_kv_heads=8,
  head_dim=64,
  num_layers=4,
  vocab_size=32000,
  runtime_mode=RuntimeMode.COMPRESSED,
)

app = create_app(
  config,
  checkpoint="/path/to/model.safetensors",
  tokenizer_name_or_path="/path/to/tokenizer-or-hf-id",
)
```

## Configuration guide

### Core architectural invariants

`RFSNConfig` validates several constraints up front:

- `hidden_dim == num_heads * head_dim`
- `num_subspaces * subspace_dim == head_dim`
- `hot_capacity <= warm_capacity <= cold_capacity`
- `num_heads % num_kv_heads == 0`
- `block_size_seq > 0`

### Important runtime knobs

| Field | Meaning |
| --- | --- |
| `runtime_mode` | `exact` keeps only the hot cache; `compressed` archives overflow and reuses it during decode |
| `model_dtype` | Weight and activation dtype for the model path |
| `cache_dtype` | Storage dtype for KV cache; defaults to `model_dtype`; accepts `fp8_e4m3` |
| `num_kv_heads` | Enables grouped-query attention when smaller than `num_heads` |
| `hot_capacity` | Max tokens kept device-resident per layer |
| `warm_capacity` | Max archived tokens kept in RAM before spilling blocks onward |
| `cold_capacity` | Maximum retained archived tokens across disk-tier blocks; excess cold blocks are pruned |
| `block_size_seq` | Eviction/compression granularity |
| `rvq_max_active` | Fixed-width budget for archived RVQ residual entries |

### Choosing cache dtypes

- `float16`: good default for low memory use with minimal complexity
- `bfloat16`: safer numeric range on supported workloads
- `float32`: easiest debug mode
- `fp8_e4m3`: one-byte cache storage, implemented as native float8 if MLX exposes it or as a packed `uint8` software fallback otherwise

The FP8 path is a cache-storage optimization, not a full-model FP8 execution path.

## Repository layout

```text
rfsn_v10_5/
  __init__.py
  api.py
  attention_compressed.py
  attention_exact.py
  bench.py
  cache.py
  codec.py
  config.py
  fp8.py
  launcher.py
  layer.py
  loader.py
  model.py
  tokenizer_utils.py
  types.py
tests/
  test_api.py
  test_cache_dtype.py
  test_cold_tier_gc.py
  test_launcher.py
  test_performance.py
  test_tokenizer_utils.py
```

Module roles:

- `config.py`: validated runtime and architecture config
- `cache.py`: hot/warm/cold KV cache, eviction, archive reuse, FP8 storage plumbing
- `api.py`: thin FastAPI wrapper with blocking and SSE-style generation endpoints
- `codec.py`: PQ + RVQ encoding and decode helpers for archived keys
- `attention_exact.py`: exact attention and segmented attention utilities
- `attention_compressed.py`: mixed archived + hot attention path
- `layer.py`: per-layer projections, cache updates, attention routing, SwiGLU FFN
- `model.py`: full model orchestration, prefill, decode, generation
- `loader.py`: checkpoint remapping and loading
- `bench.py`: prefill/decode timing helpers
- `launcher.py`: CLI entry point

## Testing

Run the focused regression suite:

```bash
python -m unittest tests.test_api tests.test_launcher tests.test_tokenizer_utils tests.test_cold_tier_gc tests.test_cache_dtype tests.test_performance
```

What these tests cover:

- FastAPI health, blocking generate, and SSE stream routes
- launcher config plumbing and smoke command behavior
- packed FP8 hot-cache storage and round-trip correctness
- archived warm/cold payload storage in compact `uint8` form
- combined archived-context cache reuse across decode steps
- cold-tier file deletion and retained-context pruning once `cold_capacity` is exceeded
- archived decode performance smoke path, including capture-artifact fallback when the runtime lacks usable profiling hooks

## Current limitations

- Apple Silicon only, because MLX is the execution backend.
- No checkpoint-specific tokenizer assets or chat templates are bundled in-repo.
- Compressed prefill for a single prompt chunk larger than `hot_capacity` is not implemented; large prompts should be chunked before prefill.
- The profiling/capture smoke path falls back to a text artifact on runtimes where `mx.profiler` or Metal capture cannot start.
- `cold_capacity` now prunes the least-recently-used retained cold archive, which means the oldest retained context may be truncated once the disk-tier budget is exceeded.
- This repo focuses on inference; there is no training or fine-tuning pipeline.
- The included FastAPI wrapper is single-request and does not solve continuous batching.

## Where to extend next

- chunked prefill for very long prompts
- checkpoint-aware tokenizer auto-discovery and chat templates
- continuous batching / serving wrapper
- richer profiling once runtime support is available
- deeper kernel-level optimization if MLX graph-level improvements stop being enough

## License and project notes

No license file is included in the current workspace snapshot. If you plan to publish or redistribute the project, add one explicitly.