from __future__ import annotations

import unittest

import mlx.core as mx
from fastapi.testclient import TestClient

from rfsn_v10_5.api import create_app
from rfsn_v10_5.config import RFSNConfig


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": [5, 6]}

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
        return [
            mx.array([7], dtype=mx.int32),
            mx.array([8], dtype=mx.int32),
        ]

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
    def _build_client(self) -> TestClient:
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
        )
        return TestClient(app)

    def test_health_endpoint_reports_ready(self) -> None:
        client = self._build_client()

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["tokenizer_loaded"])

    def test_generate_endpoint_returns_token_ids_and_text(self) -> None:
        client = self._build_client()

        response = client.post("/generate", json={"prompt": "Hello", "max_new_tokens": 2})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["token_ids"], [5, 6, 7, 8])
        self.assertEqual(payload["text"], "5|6|7|8")
        self.assertEqual(payload["generated_text"], "7|8")

    def test_generate_stream_endpoint_emits_token_and_complete_events(self) -> None:
        client = self._build_client()

        with client.stream("POST", "/generate/stream", json={"prompt": "Hello", "max_new_tokens": 2}) as response:
            body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: token", body)
        self.assertIn('"token_id": 7', body)
        self.assertIn("event: complete", body)
        self.assertIn('"generated_text": "7|8"', body)

    def test_generate_endpoint_rejects_missing_prompt(self) -> None:
        client = self._build_client()

        response = client.post("/generate", json={"max_new_tokens": 2})

        self.assertEqual(response.status_code, 400)
        self.assertIn("prompt", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()