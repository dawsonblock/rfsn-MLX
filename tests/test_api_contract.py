from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.routing import APIRoute

from rfsn_v10_5.api import GenerateRequest, create_app
from rfsn_v10_5.cache import RFSNCache
from rfsn_v10_5.model import RFSNMLX
from tests._helpers import FakeModel, FakeTokenizer, build_config, build_manifest, build_payload


class APIContractTest(unittest.TestCase):
    def _build_app(self, **create_app_kwargs):
        config = create_app_kwargs.pop("config", build_config(vocab_size=32))
        model = create_app_kwargs.pop("model", FakeModel(config))
        tokenizer = create_app_kwargs.pop("tokenizer", FakeTokenizer())
        return create_app(
            config,
            model=model,
            tokenizer=tokenizer,
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
        self.assertEqual(payload["disk_cache_dir"], app.state.service.config.disk_cache_dir)
        self.assertIsNone(payload["default_session_id"])
        self.assertEqual(payload["admission_control"]["max_concurrent_requests"], 1)

    def test_generate_endpoint_returns_token_ids_and_text(self) -> None:
        app = self._build_app()
        generate = self._get_route_endpoint(app, "/generate", "POST")
        payload = generate(GenerateRequest(prompt="Hello", max_new_tokens=2))

        self.assertTrue(payload["session_id"])
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

    def test_generate_endpoint_restores_cache_when_requested(self) -> None:
        app = self._build_app()
        generate = self._get_route_endpoint(app, "/generate", "POST")
        seen: dict[str, object] = {}

        class FakeCache:
            def __init__(self, config, batch_size: int, session_id=None, restore: bool = False) -> None:
                seen["batch_size"] = batch_size
                seen["session_id"] = session_id
                seen["restore"] = restore
                self.session_id = session_id or "request-session"

            def restore_from_disk(self):
                seen["restored"] = True
                return [[object()]]

        with patch("rfsn_v10_5.api.RFSNCache", FakeCache):
            payload = generate(
                GenerateRequest(
                    prompt="Hello",
                    max_new_tokens=2,
                    restore_cache=True,
                    session_id="request-session",
                )
            )

        self.assertTrue(seen["restored"])
        self.assertEqual(seen["batch_size"], 1)
        self.assertEqual(seen["session_id"], "request-session")
        self.assertTrue(seen["restore"])
        self.assertEqual(payload["session_id"], "request-session")
        self.assertEqual(payload["token_ids"], [5, 6, 7, 8])

    def test_generate_endpoint_rejects_restore_without_session_id(self) -> None:
        app = self._build_app()
        generate = self._get_route_endpoint(app, "/generate", "POST")

        with self.assertRaises(HTTPException) as exc_info:
            generate(GenerateRequest(prompt="Hello", max_new_tokens=2, restore_cache=True))

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("session_id", exc_info.exception.detail)

    def test_generate_endpoint_rejects_invalid_session_id(self) -> None:
        app = self._build_app()
        generate = self._get_route_endpoint(app, "/generate", "POST")

        with self.assertRaises(HTTPException) as exc_info:
            generate(GenerateRequest(prompt="Hello", max_new_tokens=2, session_id="bad/session"))

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("session_id", exc_info.exception.detail)

    def test_generate_endpoint_surfaces_runtime_bounds_as_bad_request(self) -> None:
        config = build_config(vocab_size=32, max_position_embeddings=2)
        app = self._build_app(config=config, model=RFSNMLX(config), tokenizer=None)
        generate = self._get_route_endpoint(app, "/generate", "POST")

        with self.assertRaises(HTTPException) as exc_info:
            generate(GenerateRequest(prompt_ids=[1, 2], max_new_tokens=1))

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("max_position_embeddings=2", exc_info.exception.detail)

    def test_generate_endpoint_maps_restore_errors_to_http_status_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(vocab_size=32, disk_cache_dir=tmpdir)
            app = self._build_app(config=config, model=RFSNMLX(config), tokenizer=None)
            generate = self._get_route_endpoint(app, "/generate", "POST")

            with self.assertRaises(HTTPException) as missing_exc:
                generate(
                    GenerateRequest(
                        prompt_ids=[1, 2],
                        max_new_tokens=1,
                        restore_cache=True,
                        session_id="unknown-session",
                    )
                )
            self.assertEqual(missing_exc.exception.status_code, 404)
            self.assertIn("unknown-session", missing_exc.exception.detail)

            gap_session = "gap-session"
            seed_cache = RFSNCache(config, batch_size=1, session_id=gap_session)
            layer_cache = seed_cache.layer(0)
            first = build_manifest(seed_cache.model_id, block_name="blk0", start=0, end=4)
            second = build_manifest(seed_cache.model_id, block_name="blk1", start=8, end=12)
            layer_cache.storage.persist_block(first, build_payload(first.token_count))
            layer_cache.storage.persist_block(second, build_payload(second.token_count))

            with self.assertRaises(HTTPException) as gap_exc:
                generate(
                    GenerateRequest(
                        prompt_ids=[1, 2],
                        max_new_tokens=1,
                        restore_cache=True,
                        session_id=gap_session,
                    )
                )
            self.assertEqual(gap_exc.exception.status_code, 409)
            self.assertIn("gap", gap_exc.exception.detail)

    def test_generate_stream_endpoint_emits_token_and_complete_events(self) -> None:
        app = self._build_app()
        generate_stream = self._get_route_endpoint(app, "/generate/stream", "POST")
        request = GenerateRequest(prompt="Hello", max_new_tokens=2)
        response = generate_stream(request)
        body = "".join(app.state.service.stream_generate(request))
        status_code = response.status_code

        self.assertEqual(status_code, 200)
        self.assertEqual(response.media_type, "text/event-stream")
        self.assertIn("event: token", body)
        self.assertIn('"token_id": 7', body)
        self.assertIn("event: complete", body)
        self.assertIn('"session_id":', body)
        self.assertIn('"generated_text": "7|8"', body)

    def test_generate_endpoint_rejects_missing_prompt(self) -> None:
        app = self._build_app()
        generate = self._get_route_endpoint(app, "/generate", "POST")

        with self.assertRaises(HTTPException) as exc_info:
            generate(GenerateRequest(max_new_tokens=2))

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("prompt", exc_info.exception.detail)

    def test_generate_endpoint_rejects_overload_cleanly(self) -> None:
        app = self._build_app(max_concurrent_requests=1, max_queue_size=0)
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