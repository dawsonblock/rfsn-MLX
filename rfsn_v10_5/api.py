"""Thin FastAPI wrapper for tokenizer-backed text generation.

This module keeps the core engine token-ID based while exposing a small
HTTP surface for one-request-at-a-time inference. It does not implement
continuous batching or multi-tenant scheduling.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .cache import RFSNCache
from .config import RFSNConfig
from .loader import load_hf_weights
from .model import RFSNMLX
from .tokenizer_utils import (
    decode_token_ids,
    encode_prompt_text,
    load_tokenizer,
    materialize_generated_sequence,
    prompt_ids_from_list,
)


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    prompt_ids: Optional[list[int]] = None
    max_new_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1.0)


class RFSNAPIService:
    """Single-request generation wrapper used by the FastAPI app."""

    def __init__(
        self,
        config: RFSNConfig,
        *,
        checkpoint: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        model: Optional[RFSNMLX] = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self.model = model if model is not None else RFSNMLX(config)
        if checkpoint:
            load_hf_weights(self.model, checkpoint, strict=False)
        self.tokenizer = tokenizer if tokenizer is not None else (
            load_tokenizer(tokenizer_name_or_path) if tokenizer_name_or_path else None
        )

    def _prepare_prompt_ids(self, request: GenerateRequest) -> mx.array:
        if request.prompt_ids is not None:
            return prompt_ids_from_list(request.prompt_ids, vocab_size=self.config.vocab_size)
        if request.prompt is not None:
            if self.tokenizer is not None:
                return encode_prompt_text(
                    self.tokenizer,
                    request.prompt,
                    vocab_size=self.config.vocab_size,
                )
            ascii_ids = [min(ord(char), self.config.vocab_size - 1) for char in request.prompt]
            return prompt_ids_from_list(ascii_ids, vocab_size=self.config.vocab_size)
        raise ValueError("Request must provide either 'prompt' or 'prompt_ids'")

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        prompt_ids = self._prepare_prompt_ids(request)
        token_ids = materialize_generated_sequence(
            prompt_ids,
            self.model.generate(
                prompt_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
            ),
        )

        response: dict[str, Any] = {
            "prompt_token_count": prompt_ids.shape[1],
            "generated_token_count": token_ids.shape[1] - prompt_ids.shape[1],
            "token_ids": token_ids[0].tolist(),
        }
        if self.tokenizer is not None:
            response["text"] = decode_token_ids(self.tokenizer, token_ids)
            response["generated_text"] = decode_token_ids(
                self.tokenizer,
                token_ids[:, prompt_ids.shape[1]:],
            ) if token_ids.shape[1] > prompt_ids.shape[1] else ""
        else:
            response["text"] = None
            response["generated_text"] = None
        return response

    def _iter_generated_tokens(self, request: GenerateRequest) -> tuple[mx.array, Iterator[mx.array]]:
        prompt_ids = self._prepare_prompt_ids(request)

        def _token_iterator() -> Iterator[mx.array]:
            cache = RFSNCache(self.config, batch_size=1)
            seen_ids = prompt_ids
            logits = self.model.prefill(prompt_ids, cache)
            mx.eval(logits)

            next_token = RFSNMLX._sample(
                logits[:, -1, :],
                request.temperature,
                request.top_p,
                request.top_k,
                request.repetition_penalty,
                seen_ids,
            )
            yield next_token
            seen_ids = mx.concatenate([seen_ids, next_token[:, None]], axis=1)

            pos = prompt_ids.shape[1]
            for _ in range(request.max_new_tokens - 1):
                logits = self.model.decode_step(next_token, cache, pos)
                mx.eval(logits)
                next_token = RFSNMLX._sample(
                    logits,
                    request.temperature,
                    request.top_p,
                    request.top_k,
                    request.repetition_penalty,
                    seen_ids,
                )
                yield next_token
                seen_ids = mx.concatenate([seen_ids, next_token[:, None]], axis=1)
                pos += 1

        return prompt_ids, _token_iterator()

    def stream_generate(self, request: GenerateRequest) -> Iterator[str]:
        prompt_ids, token_iterator = self._iter_generated_tokens(request)
        generated: list[mx.array] = []

        for step, token in enumerate(token_iterator, start=1):
            generated.append(token)
            token_id = int(token.tolist()[0])
            payload: dict[str, Any] = {
                "step": step,
                "token_id": token_id,
            }
            if self.tokenizer is not None:
                payload["text"] = decode_token_ids(self.tokenizer, [[token_id]])
            yield f"event: token\ndata: {json.dumps(payload)}\n\n"

        token_ids = materialize_generated_sequence(prompt_ids, generated)
        payload = {
            "token_ids": token_ids[0].tolist(),
            "generated_token_count": token_ids.shape[1] - prompt_ids.shape[1],
        }
        if self.tokenizer is not None:
            payload["text"] = decode_token_ids(self.tokenizer, token_ids)
            payload["generated_text"] = decode_token_ids(
                self.tokenizer,
                token_ids[:, prompt_ids.shape[1]:],
            ) if token_ids.shape[1] > prompt_ids.shape[1] else ""
        yield f"event: complete\ndata: {json.dumps(payload)}\n\n"


def create_app(
    config: RFSNConfig,
    *,
    checkpoint: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    model: Optional[RFSNMLX] = None,
    tokenizer: Any | None = None,
) -> FastAPI:
    service = RFSNAPIService(
        config,
        checkpoint=checkpoint,
        tokenizer_name_or_path=tokenizer_name_or_path,
        model=model,
        tokenizer=tokenizer,
    )

    app = FastAPI(title="rfsn-MLX API", version="0.1.0")
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "vocab_size": service.config.vocab_size,
            "tokenizer_loaded": service.tokenizer is not None,
        }

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        try:
            return service.generate(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/generate/stream")
    def generate_stream(request: GenerateRequest) -> StreamingResponse:
        try:
            stream = service.stream_generate(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamingResponse(stream, media_type="text/event-stream")

    return app