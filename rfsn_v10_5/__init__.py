"""Top-level package for the exact archived-context runtime.

This module exposes the primary user-facing classes and configuration
objects. Consumers should import from here rather than reaching into
internal modules whenever possible.

The implementation follows a strict separation of concerns:

- `config` defines the model configuration and validates invariants.
- `cache` manages the per-layer exact KV caches, including block-managed
  archival into warm RAM and cold disk tiers.
- `block_manager` defines the V11 page-table metadata and residency
  authority for archived exact KV blocks.
- `storage` provides corruption-safe disk persistence for block payloads
  and manifest rebuild on restart.
- `residency` adds the explicit warm-set and look-ahead prefetch policy
  used by the exact block-managed cache.
- `attention_exact` provides batched exact attention functions built
  atop MLX's fast attention primitive. The V11 runtime always uses this
  exact segmented-attention path.
- `layer` wires together projection, caching, attention and feed-forward
  network logic for a single transformer layer. Exports
  ``build_rope_tables`` for RoPE hoisting at the model level.
- `model` defines the full transformer model and orchestrates the
  per-layer caches during prefill and decode.
- `loader` provides ``load_hf_weights`` for loading HuggingFace
  LLaMA/Mistral checkpoints with automatic key remapping.
- `hf_config` maps supported Hugging Face config.json files into
  `RFSNConfig` without putting config parsing in the runtime hot path.
- `tokenizer_utils` keeps text and message formatting at the application
  boundary while the model stays token-ID based.
- `launcher` contains the CLI for generation, benchmarks, smoke checks,
  and session-scoped cache-persistence controls.
- `api` exposes the single-request FastAPI wrapper with admission
  control and per-request cache restoration.

Pass 5 additions (retained)
----------------------------
- ``load_hf_weights``: HuggingFace safetensors/npz weight loader.
- ``build_rope_tables``: exported for use in custom training loops.
- ``RFSNConfig.model_dtype``: configurable tensor dtype (default bfloat16).

Pass 6 additions
----------------
- ``RFSNConfig.num_kv_heads``: GQA support (default = num_heads for MHA).
- ``bench``: prefill and decode benchmarking module (``bench_prefill``,
  ``bench_decode``).
- ``launcher``: CLI entry point (``python -m rfsn_v10_5.launcher``).
- ``mlx.utils`` import fix in loader.py (``tree_flatten``/``tree_unflatten``).
- ``cold_capacity`` enforcement in cache.py.

Do not import internal modules directly; use the top-level symbols
exposed here instead. See README.md for usage examples.
"""

from .config import RFSNConfig, RuntimeMode  # noqa: F401
from .cache import RFSNCache, LayerKVCache  # noqa: F401
from .layer import RFSNLayerMLX, build_rope_tables  # noqa: F401
from .model import RFSNMLX  # noqa: F401
from .loader import load_hf_weights  # noqa: F401
from .api import create_app  # noqa: F401
from .block_manager import (  # noqa: F401
  BlockId,
  BlockLocation,
  BlockManager,
  BlockManifest,
  BlockSpan,
  PageTable,
)
from .residency import ResidencyManager  # noqa: F401
from .storage import BlockStorage  # noqa: F401
from .hf_config import HFConfigError, load_hf_config, load_hf_config_json  # noqa: F401

__all__ = [
    "RFSNConfig",
    "RuntimeMode",
    "RFSNCache",
    "LayerKVCache",
    "RFSNLayerMLX",
    "build_rope_tables",
    "RFSNMLX",
    "load_hf_weights",
    "create_app",
    "BlockId",
    "BlockLocation",
    "BlockManager",
    "BlockManifest",
    "BlockSpan",
    "PageTable",
    "ResidencyManager",
    "BlockStorage",
    "HFConfigError",
    "load_hf_config",
    "load_hf_config_json",
]
