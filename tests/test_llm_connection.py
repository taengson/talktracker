from __future__ import annotations

import json
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError

from talktracker.core.llm_client import (
    LocalLLMConnection,
    LocalLLMError,
    LocalLLMModel,
    OpenAICompatibleLocalClient,
    _server_error_message,
    normalize_local_server_url,
)
from talktracker.core.llm_config import LocalLLMSettings, LocalLLMSettingsStore
from talktracker.core.workflow import LocalLLMConnectionManager


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    received_model = ""
    received_authorization = ""

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

    def test_only_loopback_server_urls_are_accepted(self) -> None:
        self.assertEqual(normalize_local_server_url("http://localhost:8080"), "http://localhost:8080/v1")
        self.assertEqual(normalize_local_server_url("http://[::1]:8080/v1/"), "http://[::1]:8080/v1")

        for value in ("https://example.com", "http://192.168.0.20:8000", "http://user:pass@localhost:8000", "http://localhost:8000/?a=1"):
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
    def test_settings_exclude_api_keys_and_connection_is_runtime_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "local-llm.json"
            store = LocalLLMSettingsStore(path)
            manager = LocalLLMConnectionManager(settings_store=store, client_factory=_FakeClient)

            result = manager.verify("http://localhost:11434", "temporary-secret", "local-model")

            self.assertEqual(result.model, "local-model")
            self.assertTrue(manager.snapshot().connected)
            persisted = path.read_text(encoding="utf-8")
            self.assertNotIn("temporary-secret", persisted)
            self.assertEqual(json.loads(persisted), {"base_url": "http://localhost:11434/v1", "model": "local-model"})
            self.assertEqual(store.load(), LocalLLMSettings("http://localhost:11434/v1", "local-model"))


class _FakeClient:
    def __init__(self, connection: LocalLLMConnection) -> None:
        self.connection = connection

    def list_models(self) -> list[LocalLLMModel]:
        return [LocalLLMModel("local-model")]

    def test_connection(self, model: str):
        from talktracker.core.llm_client import LocalLLMConnectionTest

        return LocalLLMConnectionTest(model=model, response_preview="OK")


if __name__ == "__main__":
    unittest.main()
