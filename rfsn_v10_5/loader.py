"""Weight loading utilities for RFSNMLX.

This module provides ``load_hf_weights``, a utility that loads a
HuggingFace LLaMA/Mistral checkpoint (in ``.safetensors`` or ``.npz``
format) and maps the standard HuggingFace weight names to the RFSN
naming convention used by ``RFSNMLX``.

HuggingFace LLaMA key mapping
------------------------------
The standard HuggingFace LLaMA naming convention uses the following
prefixes and suffixes::

  model.embed_tokens.weight
  model.layers.{i}.self_attn.q_proj.weight
  model.layers.{i}.self_attn.k_proj.weight
  model.layers.{i}.self_attn.v_proj.weight
  model.layers.{i}.self_attn.o_proj.weight
  model.layers.{i}.mlp.gate_proj.weight
  model.layers.{i}.mlp.up_proj.weight
  model.layers.{i}.mlp.down_proj.weight
  model.layers.{i}.input_layernorm.weight
  model.layers.{i}.post_attention_layernorm.weight
  model.norm.weight
  lm_head.weight  (skipped: weight-tied to embed_tokens)

RFSN naming convention
-----------------------
::

  embed_tokens.weight
  layers.{i}.q_proj.weight
  layers.{i}.k_proj.weight
  layers.{i}.v_proj.weight
  layers.{i}.o_proj.weight
  layers.{i}.gate_proj.weight
  layers.{i}.up_proj.weight
  layers.{i}.down_proj.weight
  layers.{i}.attn_norm.weight
  layers.{i}.ffn_norm.weight
  norm.weight

Usage
-----
::

    from rfsn_v10_5 import RFSNMLX, RFSNConfig
    from rfsn_v10_5.loader import load_hf_weights

    config = RFSNConfig(
        hidden_dim=4096, num_heads=32, head_dim=128, num_layers=32,
        vocab_size=32000, model_dtype="bfloat16",
        runtime_mode="archived",
    )
    model = RFSNMLX(config)
    load_hf_weights(model, "/path/to/model.safetensors")

Notes
-----
- ``lm_head.weight`` is intentionally skipped because ``RFSNMLX`` uses
  weight-tied embeddings (``_lm_head`` reuses ``embed_tokens.weight``).
- For sharded checkpoints (multiple ``.safetensors`` files), pass each
  shard path in a list to ``load_hf_weights``.
- The function calls ``mx.eval(model.parameters())`` after loading to
  materialise all weights on the device.
- ``safetensors`` is an optional dependency. If it is not installed, the
  loader falls back to ``mx.load`` which supports ``.npz`` and the
  native MLX ``.safetensors`` reader (MLX >= 0.5).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Union

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .model import RFSNMLX


# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------

def _remap_hf_key(hf_key: str) -> str | None:
    """Translate a HuggingFace LLaMA key to an RFSN key.

    Returns ``None`` for keys that should be skipped (e.g. ``lm_head``).
    """
    # Skip lm_head (weight-tied to embed_tokens in RFSNMLX)
    if hf_key.startswith("lm_head"):
        return None

    # Strip leading "model." prefix
    key = hf_key
    if key.startswith("model."):
        key = key[len("model."):]

    # self_attn projections -> flat projection names
    m = re.match(r"^layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(.+)$", key)
    if m:
        return f"layers.{m.group(1)}.{m.group(2)}.{m.group(3)}"

    # mlp projections -> flat projection names
    m = re.match(r"^layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(.+)$", key)
    if m:
        return f"layers.{m.group(1)}.{m.group(2)}.{m.group(3)}"

    # input_layernorm -> attn_norm
    m = re.match(r"^layers\.(\d+)\.input_layernorm\.(.+)$", key)
    if m:
        return f"layers.{m.group(1)}.attn_norm.{m.group(2)}"

    # post_attention_layernorm -> ffn_norm
    m = re.match(r"^layers\.(\d+)\.post_attention_layernorm\.(.+)$", key)
    if m:
        return f"layers.{m.group(1)}.ffn_norm.{m.group(2)}"

    # embed_tokens, norm — pass through unchanged
    return key


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> Dict[str, mx.array]:
    """Load a single checkpoint file into a dict of mx.arrays.

    Supports:
    - ``.safetensors`` via ``mx.load`` (MLX >= 0.5 has native support)
    - ``.npz`` via ``mx.load``
    """
    suffix = path.suffix.lower()
    if suffix in (".safetensors", ".npz"):
        raw = mx.load(str(path))
        # mx.load returns a flat dict of arrays
        return {k: v for k, v in raw.items()}
    raise ValueError(
        f"Unsupported checkpoint format '{suffix}'. "
        "Expected .safetensors or .npz"
    )


def _resolve_checkpoint_index(index_path: Path) -> List[Path]:
    """Resolve a Hugging Face sharded checkpoint index into shard paths."""
    payload = json.loads(index_path.read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"Checkpoint index '{index_path}' does not contain a non-empty weight_map"
        )

    shard_paths = sorted({index_path.parent / shard_name for shard_name in weight_map.values()})
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint index references missing shard files: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return shard_paths


def _resolve_checkpoint_paths(
    paths: Union[str, Path, List[Union[str, Path]]],
) -> List[Path]:
    """Expand checkpoint inputs into concrete shard/file paths.

    Accepted inputs:
    - a single ``.safetensors`` or ``.npz`` file
    - a ``*.safetensors.index.json`` file
    - a directory containing either a sharded index or checkpoint files
    - a list combining any of the above
    """
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    resolved: List[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        if path not in seen:
            seen.add(path)
            resolved.append(path)

    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            index_files = sorted(path.glob("*.safetensors.index.json"))
            if index_files:
                for shard_path in _resolve_checkpoint_index(index_files[0]):
                    _add(shard_path)
                continue

            checkpoint_files = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.npz"))
            if checkpoint_files:
                for checkpoint_path in checkpoint_files:
                    _add(checkpoint_path)
                continue

            raise ValueError(
                f"Checkpoint directory '{path}' does not contain .safetensors, .npz, "
                "or a .safetensors.index.json file"
            )

        if path.name.endswith(".safetensors.index.json"):
            for shard_path in _resolve_checkpoint_index(path):
                _add(shard_path)
            continue

        if path.suffix.lower() in (".safetensors", ".npz"):
            _add(path)
            continue

        raise ValueError(
            f"Unsupported checkpoint input '{path}'. Expected a checkpoint file, "
            "a sharded index file, or a directory containing them."
        )

    return resolved


def load_hf_weights(
    model: RFSNMLX,
    paths: Union[str, Path, List[Union[str, Path]]],
    strict: bool = True,
) -> Dict[str, str]:
    """Load HuggingFace LLaMA/Mistral weights into an ``RFSNMLX`` model.

    Parameters
    ----------
    model : RFSNMLX
        The model to load weights into.
    paths : str | Path | list
        Path(s) to ``.safetensors`` or ``.npz`` checkpoint file(s).
        For sharded checkpoints, pass all shard paths as a list.
    strict : bool
        If True (default), raise a ``KeyError`` when a remapped key is
        not found in the model's parameter tree. Set to False to allow
        partial loading (useful when loading adapter weights).

    Returns
    -------
    skipped : dict[str, str]
        Mapping of original HF key -> reason for skipping.
    """
    paths = _resolve_checkpoint_paths(paths)

    # Resolve target dtype from config
    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    target_dtype = dtype_map[model.config.model_dtype]

    # Collect all weights from all shards
    all_weights: Dict[str, mx.array] = {}
    for p in paths:
        shard = _load_file(p)
        all_weights.update(shard)

    # Remap keys and build the update dict
    rfsn_weights: Dict[str, mx.array] = {}
    skipped: Dict[str, str] = {}

    for hf_key, tensor in all_weights.items():
        rfsn_key = _remap_hf_key(hf_key)
        if rfsn_key is None:
            skipped[hf_key] = "weight-tied (lm_head skipped)"
            continue
        rfsn_weights[rfsn_key] = tensor.astype(target_dtype)

    # Flatten the model parameter tree to validate keys
    flat_params = dict(tree_flatten(model.parameters()))

    if strict:
        missing = set(flat_params.keys()) - set(rfsn_weights.keys())
        unexpected = set(rfsn_weights.keys()) - set(flat_params.keys())
        if missing:
            raise KeyError(
                f"load_hf_weights: {len(missing)} model parameter(s) not found in "
                f"checkpoint: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )
        if unexpected:
            raise KeyError(
                f"load_hf_weights: {len(unexpected)} checkpoint key(s) not found in "
                f"model: {sorted(unexpected)[:10]}{'...' if len(unexpected) > 10 else ''}"
            )

    # Apply weights
    model.update(tree_unflatten(list(rfsn_weights.items())))
    mx.eval(model.parameters())

    return skipped
