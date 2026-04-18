"""Hugging Face ``config.json`` mapping for supported decoder families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import RFSNConfig, RuntimeMode


class HFConfigError(ValueError):
    """Raised when a Hugging Face config cannot be mapped safely."""


def _resolve_config_path(path_or_dir: str | Path) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        candidate = path / "config.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No config.json found in directory '{path}'")

    if path.name == "config.json":
        return path

    sibling = path.parent / "config.json"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        f"Could not resolve config.json from '{path}'. Provide a model directory or config.json path."
    )


def load_hf_config_json(path_or_dir: str | Path) -> dict[str, Any]:
    config_path = _resolve_config_path(path_or_dir)
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise HFConfigError(f"Invalid JSON in '{config_path}': {exc}") from exc


def detect_hf_family(config_json: Mapping[str, Any]) -> str:
    model_type = str(config_json.get("model_type", "")).lower()
    architectures = [str(name).lower() for name in config_json.get("architectures", [])]

    if model_type in {"llama", "mistral"}:
        return model_type
    if any("llama" in name for name in architectures):
        return "llama"
    if any("mistral" in name for name in architectures):
        return "mistral"

    raise HFConfigError(
        "Unsupported Hugging Face architecture. Supported families: LLaMA, Mistral. "
        f"Received model_type='{model_type}' architectures={architectures or '[]'}"
    )


def _require_int(config_json: Mapping[str, Any], key: str) -> int:
    value = config_json.get(key)
    if value is None:
        raise HFConfigError(f"config.json is missing required field '{key}'")
    return int(value)


def _require_float(config_json: Mapping[str, Any], key: str, default: float) -> float:
    value = config_json.get(key, default)
    return float(value)


def _resolve_head_dim(hidden_size: int, num_attention_heads: int, config_json: Mapping[str, Any]) -> int:
    explicit_head_dim = config_json.get("head_dim")
    if explicit_head_dim is not None:
        head_dim = int(explicit_head_dim)
        if hidden_size != num_attention_heads * head_dim:
            raise HFConfigError(
                f"hidden_size={hidden_size} does not match num_attention_heads * head_dim="
                f"{num_attention_heads * head_dim}"
            )
        return head_dim

    if hidden_size % num_attention_heads != 0:
        raise HFConfigError(
            f"hidden_size={hidden_size} is not divisible by num_attention_heads={num_attention_heads}"
        )
    return hidden_size // num_attention_heads


def hf_config_to_rfsn_config(
    config_json: Mapping[str, Any],
    *,
    hot_capacity: int = 512,
    warm_capacity: int = 2048,
    cold_capacity: int = 8192,
    block_size_seq: int = 64,
    runtime_mode: RuntimeMode = RuntimeMode.ARCHIVED,
    model_dtype: str = "bfloat16",
    disk_cache_dir: str = "./rfsn_disk_cache",
    session_id: str = "",
) -> RFSNConfig:
    family = detect_hf_family(config_json)
    hidden_size = _require_int(config_json, "hidden_size")
    num_attention_heads = _require_int(config_json, "num_attention_heads")
    num_key_value_heads = int(config_json.get("num_key_value_heads", num_attention_heads))
    num_hidden_layers = _require_int(config_json, "num_hidden_layers")
    intermediate_size = int(config_json.get("intermediate_size", config_json.get("mlp_dim", 0)))
    if intermediate_size <= 0:
        raise HFConfigError(
            "config.json is missing a positive FFN dimension; expected 'intermediate_size' or 'mlp_dim'"
        )

    head_dim = _resolve_head_dim(hidden_size, num_attention_heads, config_json)
    vocab_size = _require_int(config_json, "vocab_size")
    norm_eps = _require_float(config_json, "rms_norm_eps", 1e-5)
    rope_theta = _require_float(config_json, "rope_theta", 10000.0)
    max_position_embeddings = int(config_json.get("max_position_embeddings", 0))

    if family not in {"llama", "mistral"}:
        raise HFConfigError(f"Unsupported architecture family '{family}'")

    return RFSNConfig(
        hidden_dim=hidden_size,
        num_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        num_layers=num_hidden_layers,
        hot_capacity=hot_capacity,
        warm_capacity=warm_capacity,
        cold_capacity=cold_capacity,
        block_size_seq=block_size_seq,
        runtime_mode=runtime_mode,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        disk_cache_dir=disk_cache_dir,
        session_id=session_id,
        rope_base=rope_theta,
        ffn_dim=intermediate_size,
        norm_eps=norm_eps,
        model_dtype=model_dtype,
    )


def load_hf_config(path_or_dir: str | Path, **kwargs: Any) -> RFSNConfig:
    return hf_config_to_rfsn_config(load_hf_config_json(path_or_dir), **kwargs)
