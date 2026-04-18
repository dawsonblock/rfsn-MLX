from __future__ import annotations

import unittest

import mlx.core as mx
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from rfsn_v10_5.api import GenerateRequest, create_app
from rfsn_v10_5.config import RFSNConfig


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


class APITest(unittest.TestCase):
    def _build_app(self, **create_app_kwargs):
        config = RFSNConfig(
            hidden_dim=32,
            num_heads=2,
            num_kv_heads=2,
            head_dim=16,
            num_layers=1,
            vocab_size=32,
            num_subspaces=4,
            subspace_dim=4,
        )
        return create_app(
            config,
            model=FakeModel(config),
            tokenizer=FakeTokenizer(),
            **create_app_kwargs,
        )

    def _get_route_endpoint(self, app, path: str, method: str):
        for route in app.routes:
            if isinstance(route, APIRoute) and route.path == path and method.upper() in route.methods:
                return route.endpoint
        raise AssertionError(f"No route found for {method} {path}")

    def test_health_endpoint_reports_ready(self) -> None:
        app = self._build_app()
        health = self._get_route_endpoint(app, "/health", "GET")
        payload = health()

        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["tokenizer_loaded"])
        self.assertEqual(payload["admission_control"]["max_concurrent_requests"], 1)

    def test_generate_endpoint_returns_token_ids_and_text(self) -> None:
        app = self._build_app()
        generate = self._get_route_endpoint(app, "/generate", "POST")
        payload = generate(GenerateRequest(prompt="Hello", max_new_tokens=2))

        self.assertEqual(payload["token_ids"], [5, 6, 7, 8])
        self.assertEqual(payload["text"], "5|6|7|8")
        self.assertEqual(payload["generated_text"], "7|8")

    def test_generate_endpoint_accepts_messages_via_chat_template(self) -> None:
        app = self._build_app()
        generate = self._get_route_endpoint(app, "/generate", "POST")
        payload = generate(
            GenerateRequest(
                messages=[{"role": "user", "content": "Hello"}],
                max_new_tokens=2,
            )
        )

        self.assertEqual(payload["token_ids"], [9, 10, 7, 8])

    def test_generate_stream_endpoint_emits_token_and_complete_events(self) -> None:
        app = self._build_app()
        with TestClient(app) as client:
            with client.stream("POST", "/generate/stream", json={"prompt": "Hello", "max_new_tokens": 2}) as response:
                body = "".join(response.iter_text())
                status_code = response.status_code

        self.assertEqual(status_code, 200)
        self.assertIn("event: token", body)
        self.assertIn('"token_id": 7', body)
        self.assertIn("event: complete", body)
        self.assertIn('"generated_text": "7|8"', body)

    def test_generate_endpoint_rejects_missing_prompt(self) -> None:
        app = self._build_app()
        generate = self._get_route_endpoint(app, "/generate", "POST")

        with self.assertRaises(HTTPException) as exc_info:
            generate(GenerateRequest(max_new_tokens=2))

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("prompt", exc_info.exception.detail)

    def test_generate_endpoint_rejects_overload_cleanly(self) -> None:
        config = RFSNConfig(
            hidden_dim=32,
            num_heads=2,
            num_kv_heads=2,
            head_dim=16,
            num_layers=1,
            vocab_size=32,
            num_subspaces=4,
            subspace_dim=4,
        )
        app = create_app(
            config,
            model=FakeModel(config),
            tokenizer=FakeTokenizer(),
            max_concurrent_requests=1,
            max_queue_size=0,
        )
        generate = self._get_route_endpoint(app, "/generate", "POST")

        admission_context = app.state.service.admission.admit()
        admission_context.__enter__()
        try:
            with self.assertRaises(HTTPException) as exc_info:
                generate(GenerateRequest(prompt="Hello", max_new_tokens=2))
        finally:
            admission_context.__exit__(None, None, None)

        self.assertEqual(exc_info.exception.status_code, 503)
        self.assertIn("queue is full", exc_info.exception.detail)


if __name__ == "__main__":
    unittest.main()
