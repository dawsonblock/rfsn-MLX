from __future__ import annotations

import numpy as np
import mlx.core as mx

from rfsn_v10_5.block_manager import BlockId, BlockLocation, BlockManifest, BlockSpan
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.config import RFSNConfig, RuntimeMode
from rfsn_v10_5.model import RFSNMLX


def build_config(**overrides: object) -> RFSNConfig:
    config_kwargs: dict[str, object] = {
        "hidden_dim": 32,
        "num_heads": 2,
        "num_kv_heads": 2,
        "head_dim": 16,
        "num_layers": 1,
        "vocab_size": 128,
        "hot_capacity": 4,
        "warm_capacity": 4,
        "cold_capacity": 32,
        "block_size_seq": 4,
        "model_dtype": "float16",
        "runtime_mode": RuntimeMode.ARCHIVED,
        "disk_cache_dir": "./rfsn_disk_cache",
        "session_id": "",
        "max_position_embeddings": 0,
    }
    config_kwargs.update(overrides)
    return RFSNConfig(**config_kwargs)


def prompt_tokens(length: int, *, start: int = 0) -> mx.array:
    return mx.arange(start, start + length, dtype=mx.int32)[None, :]


def seed_archived_context(
    model: RFSNMLX,
    cache: RFSNCache,
    *,
    prompt_len: int = 12,
    start: int = 0,
) -> mx.array:
    prompt = prompt_tokens(prompt_len, start=start)
    logits = model.prefill(prompt, cache)
    mx.eval(logits)
    return logits


def persist_all_context_to_disk(cache: RFSNCache) -> None:
    for layer_cache in cache.layers:
        if layer_cache.hot_seq_len > 0:
            layer_cache.evict_for_append(layer_cache.config.hot_capacity)
        for manifest in list(layer_cache.block_manager.iter_blocks(layer_id=layer_cache.layer_id)):
            if manifest.residency == BlockLocation.WARM_RAM:
                layer_cache.demote_manifest_to_cold(manifest)


def build_manifest(
    model_id: str,
    *,
    layer_id: int = 0,
    block_name: str = "blk0",
    start: int = 0,
    end: int = 4,
    dtype: str = "float32",
) -> BlockManifest:
    token_count = end - start
    shape = (1, 1, token_count, 2)
    return BlockManifest(
        block_id=BlockId(model_id, layer_id, block_name),
        span=BlockSpan(start, end),
        codec_version="v11-exact-kv",
        dtype=dtype,
        shape_metadata={"keys": shape, "values": shape},
        residency=BlockLocation.COLD_DISK,
    )


def build_payload(token_count: int, *, dtype: np.dtype = np.float32) -> dict[str, np.ndarray]:
    resolved_dtype = np.dtype(dtype)
    keys = np.arange(token_count * 2, dtype=resolved_dtype).reshape(1, 1, token_count, 2)
    values = (keys + resolved_dtype.type(100.0)).astype(resolved_dtype)
    return {"keys": keys, "values": values}


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": [5, 6]}

    def apply_chat_template(self, messages, *, tokenize: bool = False, add_generation_prompt: bool = False):
        if tokenize:
            return {"input_ids": [9, 10]}
        return "chat-template"

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "|".join(str(token_id) for token_id in token_ids)


class FakeModel:
    def __init__(self, config: RFSNConfig) -> None:
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
        return [mx.array([7], dtype=mx.int32), mx.array([8], dtype=mx.int32)]

    def prefill(self, prompt_ids, cache):
        logits = mx.zeros((1, prompt_ids.shape[1], self.config.vocab_size), dtype=mx.float32)
        logits[:, -1, 7] = 10.0
        return logits

    def decode_step(self, token, cache, pos: int):
        logits = mx.zeros((1, self.config.vocab_size), dtype=mx.float32)
        next_id = 8 if pos == 2 else 9
        logits[:, next_id] = 10.0
        return logits