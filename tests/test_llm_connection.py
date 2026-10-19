from __future__ import annotations

import json
import tempfile
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Event, Thread
from urllib.error import HTTPError

from talktracker.core.llm_client import (
    LocalLLMConnection,
    LocalLLMCancelled,
    LocalLLMError,
    LocalLLMHTTPError,
    LocalLLMModel,
    OpenAICompatibleLocalClient,
    _server_error_message,
    normalize_local_server_url,
)
from talktracker.core.llm_config import LocalLLMResponseTokenLimit, LocalLLMSettings, LocalLLMSettingsStore
from talktracker.core.workflow import LocalLLMConnectionManager
from talktracker.main import _llm_operation_error_message, _unexpected_llm_operation_error_message


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    received_model = ""
    received_authorization = ""
    received_body: dict[str, object] = {}
    received_bodies: list[dict[str, object]] = []
    stream_delay_seconds = 0.0
    stream_events: list[dict[str, object]] = []
    stream_event_batches: list[list[dict[str, object]]] = []
    reject_json_mode = False
    stream_started = Event()

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._write_json(
            {
                "object": "list",
                "data": [
                    {"id": "talktracker-test", "owned_by": "local"},
                    {"id": "alpha", "owned_by": "local"},
                    {"id": "talktracker-test", "owned_by": "local"},
                ],
            }
        )

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        type(self).received_model = body["model"]
        type(self).received_authorization = self.headers.get("Authorization", "")
        type(self).received_body = body
        type(self).received_bodies.append(body)
        if type(self).reject_json_mode and "response_format" in body:
            self.send_error(400)
            return
        if body.get("stream") and type(self).stream_delay_seconds:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            type(self).stream_started.set()
            time.sleep(type(self).stream_delay_seconds)
            return
        events = type(self).stream_event_batches.pop(0) if body.get("stream") and type(self).stream_event_batches else type(self).stream_events
        if body.get("stream") and events:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._write_json({"choices": [{"message": {"content": "OK"}}]})

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep test output clean."""

    def _write_json(self, payload: dict[str, object]) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class OpenAICompatibleLocalClientTests(unittest.TestCase):
    def setUp(self) -> None:
        _OpenAICompatibleHandler.received_model = ""
        _OpenAICompatibleHandler.received_authorization = ""
        _OpenAICompatibleHandler.received_body = {}
        _OpenAICompatibleHandler.received_bodies = []
        _OpenAICompatibleHandler.stream_delay_seconds = 0.0
        _OpenAICompatibleHandler.stream_events = []
        _OpenAICompatibleHandler.stream_event_batches = []
        _OpenAICompatibleHandler.reject_json_mode = False
        _OpenAICompatibleHandler.stream_started.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    def test_reads_models_and_runs_a_short_completion(self) -> None:
        connection = LocalLLMConnection.create(self.base_url, "temporary-secret")
        client = OpenAICompatibleLocalClient(connection)

        self.assertEqual([model.id for model in client.list_models()], ["alpha", "talktracker-test"])
        result = client.test_connection("talktracker-test")

        self.assertEqual(result.model, "talktracker-test")
        self.assertEqual(result.response_preview, "OK")
        self.assertEqual(_OpenAICompatibleHandler.received_model, "talktracker-test")
        self.assertEqual(_OpenAICompatibleHandler.received_authorization, "Bearer temporary-secret")

    def test_streamed_completion_accepts_servers_that_return_regular_json(self) -> None:
        content = OpenAICompatibleLocalClient(LocalLLMConnection.create(self.base_url)).create_chat_completion(
            "talktracker-test",
            messages=[{"role": "user", "content": "JSON만 답해"}],
        )

        self.assertEqual(content, "OK")
        self.assertEqual(_OpenAICompatibleHandler.received_model, "talktracker-test")

    def test_json_mode_uses_the_standard_response_format(self) -> None:
        OpenAICompatibleLocalClient(LocalLLMConnection.create(self.base_url)).create_chat_completion(
            "talktracker-test",
            messages=[{"role": "user", "content": "JSON만 답해"}],
            json_mode=True,
        )

        self.assertEqual(_OpenAICompatibleHandler.received_body["response_format"], {"type": "json_object"})

    def test_json_mode_retries_without_response_format_for_older_servers(self) -> None:
        _OpenAICompatibleHandler.reject_json_mode = True

        content = OpenAICompatibleLocalClient(LocalLLMConnection.create(self.base_url)).create_chat_completion(
            "talktracker-test",
            messages=[{"role": "user", "content": "JSON만 답해"}],
            json_mode=True,
        )

        self.assertEqual(content, "OK")
        self.assertNotIn("response_format", _OpenAICompatibleHandler.received_body)

    def test_length_limited_completion_retries_once_with_a_larger_budget(self) -> None:
        _OpenAICompatibleHandler.stream_event_batches = [
            [{"choices": [{"delta": {"content": "{\"segments\": ["}, "finish_reason": "length"}]}]
        ]
        retries: list[tuple[int, int]] = []
        learned_limits: list[int] = []

        content = OpenAICompatibleLocalClient(LocalLLMConnection.create(self.base_url)).create_chat_completion(
            "talktracker-test",
            messages=[{"role": "user", "content": "JSON만 답해"}],
            max_tokens=1_024,
            retry_on_length=True,
            on_length_retry=lambda current, expanded: retries.append((current, expanded)),
            on_length_retry_success=learned_limits.append,
        )

        self.assertEqual(content, "OK")
        self.assertEqual(retries, [(1_024, 4_096)])
        self.assertEqual(learned_limits, [4_096])
        self.assertEqual([body["max_tokens"] for body in _OpenAICompatibleHandler.received_bodies], [1_024, 4_096])

    def test_streamed_completion_closes_when_cancelled(self) -> None:
        _OpenAICompatibleHandler.stream_delay_seconds = 1.5
        cancelled = Event()
        captured: list[Exception] = []
        client = OpenAICompatibleLocalClient(LocalLLMConnection.create(self.base_url))

        def run() -> None:
            try:
                client.create_chat_completion(
                    "talktracker-test",
                    messages=[{"role": "user", "content": "잠시 기다려"}],
                    should_cancel=cancelled.is_set,
                )
            except Exception as error:  # Assert after the worker returns.
                captured.append(error)

        worker = Thread(target=run)
        worker.start()
        self.assertTrue(_OpenAICompatibleHandler.stream_started.wait(timeout=1))
        cancelled.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(captured), 1)
        self.assertIsInstance(captured[0], LocalLLMCancelled)

    def test_streamed_completion_explains_when_reasoning_uses_the_output_limit(self) -> None:
        _OpenAICompatibleHandler.stream_events = [
            {"choices": [{"delta": {"reasoning_content": "응답을 생각 중입니다."}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ]

        with self.assertRaisesRegex(LocalLLMError, "응답 길이 제한"):
            OpenAICompatibleLocalClient(LocalLLMConnection.create(self.base_url)).create_chat_completion(
                "talktracker-test",
                messages=[{"role": "user", "content": "JSON만 답해"}],
            )

    def test_streamed_completion_explains_a_reasoning_only_response(self) -> None:
        _OpenAICompatibleHandler.stream_events = [
            {"choices": [{"delta": {"reasoning_content": "응답을 생각 중입니다."}, "finish_reason": "stop"}]},
        ]

        with self.assertRaisesRegex(LocalLLMError, "추론 내용만"):
            OpenAICompatibleLocalClient(LocalLLMConnection.create(self.base_url)).create_chat_completion(
                "talktracker-test",
                messages=[{"role": "user", "content": "JSON만 답해"}],
            )

    def test_trusted_network_openai_compatible_server_urls_are_accepted(self) -> None:
        self.assertEqual(normalize_local_server_url("http://localhost:8080"), "http://localhost:8080/v1")
        self.assertEqual(normalize_local_server_url("http://[::1]:8080/v1/"), "http://[::1]:8080/v1")
        self.assertEqual(normalize_local_server_url("http://192.168.0.20:8000"), "http://192.168.0.20:8000/v1")
        self.assertEqual(normalize_local_server_url("http://spark-f8df:12300/v1"), "http://spark-f8df:12300/v1")

        for value in ("ftp://spark-f8df:12300", "http://user:pass@spark-f8df:12300", "http://spark-f8df:12300/?api_key=secret"):
            with self.subTest(value=value), self.assertRaises(LocalLLMError):
                normalize_local_server_url(value)

    def test_server_error_does_not_reflect_a_possible_api_key(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:8000/v1/models",
            401,
            "Unauthorized",
            None,
            BytesIO(b'{"error":{"message":"Bearer temporary-secret is invalid"}}'),
        )

        message = _server_error_message(error)

        self.assertIn("인증", message)
        self.assertNotIn("temporary-secret", message)


class LocalLLMSettingsTests(unittest.TestCase):
    def test_unsuccessful_server_is_not_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "local-llm.json"
            manager = LocalLLMConnectionManager(settings_store=LocalLLMSettingsStore(path), client_factory=_FailingClient)

            with self.assertRaises(LocalLLMError):
                manager.verify("http://localhost:11434", "temporary-secret", "unavailable-model")

            self.assertFalse(path.exists())
            self.assertEqual(manager.snapshot().settings.recent_base_urls, ())

    def test_settings_remember_successful_servers_without_api_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "local-llm.json"
            store = LocalLLMSettingsStore(path)
            manager = LocalLLMConnectionManager(settings_store=store, client_factory=_FakeClient)

            result = manager.verify("http://localhost:11434", "temporary-secret", "local-model")
            manager.verify("http://127.0.0.1:8000", "another-secret", "local-model")

            self.assertEqual(result.model, "local-model")
            self.assertTrue(manager.snapshot().connected)
            persisted = path.read_text(encoding="utf-8")
            self.assertNotIn("temporary-secret", persisted)
            self.assertNotIn("another-secret", persisted)
            self.assertEqual(
                json.loads(persisted),
                {
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "local-model",
                    "recent_base_urls": ["http://127.0.0.1:8000/v1", "http://localhost:11434/v1"],
                },
            )
            self.assertEqual(
                store.load(),
                LocalLLMSettings(
                    "http://127.0.0.1:8000/v1",
                    "local-model",
                    ("http://127.0.0.1:8000/v1", "http://localhost:11434/v1"),
                ),
            )

    def test_response_limit_is_remembered_only_for_the_verified_server_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "local-llm.json"
            store = LocalLLMSettingsStore(path)
            manager = LocalLLMConnectionManager(settings_store=store, client_factory=_FakeClient)

            manager.verify("http://localhost:11434", "temporary-secret", "local-model")
            manager.remember_response_token_limit(4_096)

            expected_limit = LocalLLMResponseTokenLimit("http://localhost:11434/v1", "local-model", 4_096)
            self.assertEqual(manager.snapshot().settings.response_token_limits, (expected_limit,))
            self.assertEqual(store.load().response_token_limit_for("http://localhost:11434/v1", "local-model"), 4_096)
            persisted = path.read_text(encoding="utf-8")
            self.assertNotIn("temporary-secret", persisted)
            self.assertIn('"max_tokens": 4096', persisted)

            manager.verify("http://127.0.0.1:8000", "another-secret", "local-model")
            self.assertEqual(
                manager.snapshot().settings.response_token_limit_for("http://127.0.0.1:8000/v1", "local-model"),
                None,
            )
            self.assertEqual(
                manager.snapshot().settings.response_token_limit_for("http://localhost:11434/v1", "local-model"),
                4_096,
            )


class LocalLLMDisplayErrorTests(unittest.TestCase):
    def test_authentication_error_tells_user_how_to_check_a_key(self) -> None:
        message = _llm_operation_error_message("models", LocalLLMHTTPError(401, "인증 오류"))

        self.assertIn("API 키 표시", message)
        self.assertIn("모델 목록", message)

    def test_unexpected_error_has_a_safe_diagnostic_code(self) -> None:
        message = _unexpected_llm_operation_error_message("models", KeyError("api_key"))

        self.assertIn("KeyError", message)
        self.assertNotIn("api_key", message)


class _FakeClient:
    def __init__(self, connection: LocalLLMConnection) -> None:
        self.connection = connection

    def list_models(self) -> list[LocalLLMModel]:
        return [LocalLLMModel("local-model")]

    def test_connection(self, model: str):
        from talktracker.core.llm_client import LocalLLMConnectionTest

        return LocalLLMConnectionTest(model=model, response_preview="OK")


class _FailingClient(_FakeClient):
    def test_connection(self, model: str):
        raise LocalLLMError("연결할 수 없습니다.")


if __name__ == "__main__":
    unittest.main()
