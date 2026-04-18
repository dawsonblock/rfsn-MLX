"""Honest FastAPI wrapper for the RFSN-MLX V11 runtime."""

from __future__ import annotations

from contextlib import contextmanager
import json
import threading
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
    encode_messages,
    encode_prompt_text,
    load_tokenizer,
    materialize_generated_sequence,
    prompt_ids_from_list,
)


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[list[dict[str, Any]]] = None
    prompt_ids: Optional[list[int]] = None
    session_id: Optional[str] = None
    restore_cache: bool = False
    max_new_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1.0)


class OverloadedError(RuntimeError):
    """Raised when the server is at capacity."""


class AdmissionController:
    """Bounded concurrency with a bounded wait queue."""

    def __init__(self, *, max_concurrent_requests: int, max_queue_size: int) -> None:
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.max_queue_size = max(0, int(max_queue_size))
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent_requests)
        self._lock = threading.Lock()
        self._active = 0
        self._queued = 0

    @contextmanager
    def admit(self) -> Iterator[None]:
        immediate = self._semaphore.acquire(blocking=False)
        if immediate:
            with self._lock:
                self._active += 1
            try:
                yield
            finally:
                with self._lock:
                    self._active -= 1
                self._semaphore.release()
            return

        with self._lock:
            if self._queued >= self.max_queue_size:
                raise OverloadedError(
                    "Server is at capacity: all worker slots are busy and the request queue is full"
                )
            self._queued += 1

        self._semaphore.acquire()
        with self._lock:
            self._queued -= 1
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_concurrent_requests": self.max_concurrent_requests,
                "max_queue_size": self.max_queue_size,
                "active_requests": self._active,
                "queued_requests": self._queued,
            }


class RFSNAPIService:
    """Single-request generation wrapper used by the FastAPI app."""

    def __init__(
        self,
        config: RFSNConfig,
        *,
        checkpoint: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        max_concurrent_requests: int = 1,
        max_queue_size: int = 0,
    ) -> None:
        self.config = config
        self.model = model if model is not None else RFSNMLX(config)
        if checkpoint:
            load_hf_weights(self.model, checkpoint, strict=False)
        self.tokenizer = tokenizer if tokenizer is not None else (
            load_tokenizer(tokenizer_name_or_path) if tokenizer_name_or_path else None
        )
        self.admission = AdmissionController(
            max_concurrent_requests=max_concurrent_requests,
            max_queue_size=max_queue_size,
        )

    def _build_cache(
        self,
        batch_size: int,
        *,
        restore_cache: bool = False,
        session_id: Optional[str] = None,
    ) -> RFSNCache:
        cache = RFSNCache(
            self.config,
            batch_size=batch_size,
            session_id=session_id,
            restore=restore_cache,
        )
        if restore_cache:
            cache.restore_from_disk()
        return cache

    def _prepare_prompt_ids(self, request: GenerateRequest) -> mx.array:
        if request.prompt_ids is not None:
            return prompt_ids_from_list(request.prompt_ids, vocab_size=self.config.vocab_size)
        if request.messages is not None:
            if self.tokenizer is None:
                raise ValueError("Structured messages require a tokenizer with chat-template support")
            return encode_messages(
                self.tokenizer,
                request.messages,
                vocab_size=self.config.vocab_size,
                add_generation_prompt=True,
            )
        if request.prompt is not None:
            if self.tokenizer is None:
                raise ValueError("Text prompts require a tokenizer, or provide prompt_ids directly")
            return encode_prompt_text(
                self.tokenizer,
                request.prompt,
                vocab_size=self.config.vocab_size,
            )
        raise ValueError("Request must provide one of 'prompt', 'messages', or 'prompt_ids'")

    def _generate_from_prompt_ids(
        self,
        prompt_ids: mx.array,
        request: GenerateRequest,
    ) -> dict[str, Any]:
        cache = self._build_cache(
            prompt_ids.shape[0],
            restore_cache=request.restore_cache,
            session_id=request.session_id,
        )
        token_ids = materialize_generated_sequence(
            prompt_ids,
            self.model.generate(
                prompt_ids,
                max_new_tokens=request.max_new_tokens,
                cache=cache,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
            ),
        )

        response: dict[str, Any] = {
            "session_id": cache.session_id,
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

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        prompt_ids = self._prepare_prompt_ids(request)
        return self._generate_from_prompt_ids(prompt_ids, request)

    def _iter_generated_tokens(
        self,
        prompt_ids: mx.array,
        request: GenerateRequest,
        *,
        cache: RFSNCache,
    ) -> Iterator[mx.array]:

        def _token_iterator() -> Iterator[mx.array]:
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

        return _token_iterator()

    def stream_generate(self, request: GenerateRequest, *, prompt_ids: Optional[mx.array] = None) -> Iterator[str]:
        resolved_prompt_ids = prompt_ids if prompt_ids is not None else self._prepare_prompt_ids(request)
        cache = self._build_cache(
            resolved_prompt_ids.shape[0],
            restore_cache=request.restore_cache,
            session_id=request.session_id,
        )
        token_iterator = self._iter_generated_tokens(resolved_prompt_ids, request, cache=cache)
        generated: list[mx.array] = []

        for step, token in enumerate(token_iterator, start=1):
            generated.append(token)
            token_id = int(token.tolist()[0])
            payload: dict[str, Any] = {"step": step, "token_id": token_id}
            if self.tokenizer is not None:
                payload["text"] = decode_token_ids(self.tokenizer, [[token_id]])
            yield f"event: token\ndata: {json.dumps(payload)}\n\n"

        token_ids = materialize_generated_sequence(resolved_prompt_ids, generated)
        payload = {
            "session_id": cache.session_id,
            "token_ids": token_ids[0].tolist(),
            "generated_token_count": token_ids.shape[1] - resolved_prompt_ids.shape[1],
        }
        if self.tokenizer is not None:
            payload["text"] = decode_token_ids(self.tokenizer, token_ids)
            payload["generated_text"] = decode_token_ids(
                self.tokenizer,
                token_ids[:, resolved_prompt_ids.shape[1]:],
            ) if token_ids.shape[1] > resolved_prompt_ids.shape[1] else ""
        yield f"event: complete\ndata: {json.dumps(payload)}\n\n"


def create_app(
    config: RFSNConfig,
    *,
    checkpoint: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    max_concurrent_requests: int = 1,
    max_queue_size: int = 0,
) -> FastAPI:
    service = RFSNAPIService(
        config,
        checkpoint=checkpoint,
        tokenizer_name_or_path=tokenizer_name_or_path,
        model=model,
        tokenizer=tokenizer,
        max_concurrent_requests=max_concurrent_requests,
        max_queue_size=max_queue_size,
    )

    app = FastAPI(title="rfsn-MLX API", version="0.1.0")
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "vocab_size": service.config.vocab_size,
            "max_position_embeddings": service.config.max_position_embeddings,
            "disk_cache_dir": service.config.disk_cache_dir,
            "default_session_id": service.config.session_id or None,
            "tokenizer_loaded": service.tokenizer is not None,
            "admission_control": service.admission.snapshot(),
        }

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        try:
            with service.admission.admit():
                return service.generate(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OverloadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/generate/stream")
    def generate_stream(request: GenerateRequest) -> StreamingResponse:
        admission_context = None
        prompt_ids = None
        try:
            admission_context = service.admission.admit()
            admission_context.__enter__()
            prompt_ids = service._prepare_prompt_ids(request)
        except ValueError as exc:
            if admission_context is not None:
                admission_context.__exit__(None, None, None)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            if admission_context is not None:
                admission_context.__exit__(None, None, None)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OverloadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            if admission_context is not None:
                admission_context.__exit__(None, None, None)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        def _stream() -> Iterator[str]:
            try:
                yield from service.stream_generate(request, prompt_ids=prompt_ids)
            finally:
                admission_context.__exit__(None, None, None)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app
