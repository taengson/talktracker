"""Small OpenAI-compatible client used by TalkTracker's LLM features.

The project deliberately calls the two standard endpoints directly instead of
depending on a particular server product.  A user may choose a server on the
same PC or a trusted local network; the connection screen makes that data-flow
choice explicit.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Thread
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


class LocalLLMError(RuntimeError):
    """Base class for user-actionable local LLM connection errors."""


class LocalLLMResponseError(LocalLLMError):
    """Raised when a server response is not an OpenAI-compatible payload."""


class LocalLLMHTTPError(LocalLLMError):
    """Raised for a failed request without exposing request credentials."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message or f"로컬 AI 서버가 HTTP {status_code} 오류를 반환했습니다.")


class LocalLLMCancelled(LocalLLMError):
    """Raised when a caller no longer wants the current completion result."""


@dataclass(frozen=True)
class LocalLLMConnection:
    """A validated OpenAI-compatible server address.

    ``api_key`` intentionally lives only on this runtime object.  The settings
    store persists the endpoint and model identifier, never this field.
    """

    base_url: str
    api_key: str = ""

    @classmethod
    def create(cls, base_url: str, api_key: str = "") -> "LocalLLMConnection":
        normalized = normalize_local_server_url(base_url)
        return cls(base_url=normalized, api_key=api_key.strip())

    def endpoint(self, resource: str) -> str:
        resource = resource.lstrip("/")
        return f"{self.base_url}/{resource}"


@dataclass(frozen=True)
class LocalLLMModel:
    """One model reported by ``GET /v1/models``."""

    id: str
    owned_by: str = ""


@dataclass(frozen=True)
class LocalLLMConnectionTest:
    """A successful short generation used to verify a selected model."""

    model: str
    response_preview: str


def normalize_local_server_url(value: str) -> str:
    """Normalize a user-selected OpenAI-compatible URL to an API ``/v1`` root.

    The caller chooses the server.  Credentials, queries, and fragments stay
    out of URLs so they cannot be accidentally persisted in connection history.
    """

    candidate = value.strip()
    if not candidate:
        raise LocalLLMError("로컬 AI 서버 주소를 입력해 주세요.")
    try:
        parsed = urlsplit(candidate)
    except ValueError as error:
        raise LocalLLMError("로컬 AI 서버 주소가 올바르지 않습니다.") from error

    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        raise LocalLLMError("로컬 AI 서버 주소는 http 또는 https 주소여야 합니다.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LocalLLMError("서버 주소에는 계정 정보, 쿼리, 조각 식별자를 넣을 수 없습니다.")

    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class OpenAICompatibleLocalClient:
    """Minimal blocking client for a local OpenAI-compatible server."""

    def __init__(self, connection: LocalLLMConnection, *, timeout_seconds: float = 45.0) -> None:
        self.connection = connection
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> list[LocalLLMModel]:
        payload = self._request_json("GET", "models")
        data = payload.get("data")
        if not isinstance(data, list):
            raise LocalLLMResponseError("모델 목록 응답 형식이 올바르지 않습니다.")

        models: list[LocalLLMModel] = []
        seen: set[str] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            models.append(LocalLLMModel(id=model_id, owned_by=str(item.get("owned_by", "")).strip()))
        return sorted(models, key=lambda model: model.id.lower())

    def test_connection(self, model: str) -> LocalLLMConnectionTest:
        model = model.strip()
        if not model:
            raise LocalLLMError("연결을 확인할 모델 ID를 입력하거나 선택해 주세요.")
        payload = self._request_json(
            "POST",
            "chat/completions",
            body={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a local connection check. Reply with only OK."},
                    {"role": "user", "content": "Reply with OK."},
                ],
                "temperature": 0,
                "max_tokens": 8,
                "stream": False,
            },
            timeout_seconds=max(self.timeout_seconds, 120.0),
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalLLMResponseError("연결 확인 응답에 생성 결과가 없습니다.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LocalLLMResponseError("연결 확인 응답 형식이 올바르지 않습니다.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LocalLLMResponseError("연결 확인 응답에 메시지가 없습니다.")
        content = str(message.get("content", "")).strip()
        # Some reasoning models finish a tiny request without visible content.
        # A structurally valid completion is still proof that the selected model ran.
        return LocalLLMConnectionTest(model=model, response_preview=content[:160])

    def create_chat_completion(
        self,
        model: str,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 1_024,
        should_cancel: Callable[[], bool] | None = None,
    ) -> str:
        """Return a streamed text completion and close it when cancelled.

        OpenAI-compatible servers generally stream Server-Sent Events.  A
        small watcher closes the response socket when cancellation is requested
        so the caller need not wait for a long local generation to finish.
        A normal JSON response remains accepted for small compatibility
        servers that ignore the ``stream`` flag.
        """

        if should_cancel and should_cancel():
            raise LocalLLMCancelled("AI 정리 요청을 취소했습니다.")
        model = model.strip()
        if not model:
            raise LocalLLMError("AI 정리에 사용할 모델을 선택해 주세요.")
        if not messages:
            raise LocalLLMError("AI 정리에 보낼 대화 내용을 찾지 못했습니다.")
        body = {
            "model": model,
            "messages": messages,
            "temperature": max(0, min(2, float(temperature))),
            "max_tokens": max(1, int(max_tokens)),
            "stream": True,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        http_request = Request(self.connection.endpoint("chat/completions"), data=data, method="POST")
        http_request.add_header("Accept", "text/event-stream, application/json")
        http_request.add_header("Content-Type", "application/json")
        if self.connection.api_key:
            http_request.add_header("Authorization", f"Bearer {self.connection.api_key}")

        try:
            response = urlopen(http_request, timeout=max(self.timeout_seconds, 180.0))
        except HTTPError as error:
            raise LocalLLMHTTPError(error.code, _server_error_message(error)) from error
        except (URLError, socket.timeout, TimeoutError, OSError) as error:
            raise LocalLLMError("로컬 AI 서버에 연결할 수 없습니다. 서버 주소와 실행 상태를 확인해 주세요.") from error

        done = Event()

        def close_on_cancel() -> None:
            while not done.is_set():
                if should_cancel and should_cancel():
                    response.close()
                    return
                sleep(0.05)

        watcher = Thread(target=close_on_cancel, name="talktracker-llm-cancel", daemon=True)
        watcher.start()
        chunks: list[str] = []
        fallback_lines: list[bytes] = []
        try:
            while True:
                if should_cancel and should_cancel():
                    raise LocalLLMCancelled("AI 정리 요청을 취소했습니다.")
                try:
                    raw_line = response.readline()
                except (OSError, ValueError, socket.timeout) as error:
                    if should_cancel and should_cancel():
                        raise LocalLLMCancelled("AI 정리 요청을 취소했습니다.") from error
                    raise LocalLLMError("로컬 AI 서버의 생성 응답을 읽지 못했습니다.") from error
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    fallback_lines.append(raw_line)
                    continue
                data_line = line[5:].strip()
                if data_line == "[DONE]":
                    break
                try:
                    event = json.loads(data_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                chunks.extend(_stream_content(event))
        finally:
            done.set()
            response.close()

        if should_cancel and should_cancel():
            raise LocalLLMCancelled("AI 정리 요청을 취소했습니다.")
        content = "".join(chunks).strip()
        if not content and fallback_lines:
            try:
                fallback = json.loads(b"".join(fallback_lines).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise LocalLLMResponseError("AI 정리 응답을 읽을 수 없습니다.") from error
            if isinstance(fallback, dict):
                content = _completion_content(fallback)
        if not content:
            raise LocalLLMResponseError("AI 정리 응답에 읽을 수 있는 내용이 없습니다.")
        return content

    def _request_json(
        self,
        method: str,
        resource: str,
        *,
        body: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = Request(self.connection.endpoint(resource), data=data, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if self.connection.api_key:
            request.add_header("Authorization", f"Bearer {self.connection.api_key}")

        try:
            with urlopen(request, timeout=timeout_seconds or self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as error:
            raise LocalLLMHTTPError(error.code, _server_error_message(error)) from error
        except (URLError, socket.timeout, TimeoutError, OSError) as error:
            raise LocalLLMError("로컬 AI 서버에 연결할 수 없습니다. 서버 주소와 실행 상태를 확인해 주세요.") from error

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LocalLLMResponseError("로컬 AI 서버의 응답을 읽을 수 없습니다.") from error
        if not isinstance(payload, dict):
            raise LocalLLMResponseError("로컬 AI 서버의 응답 형식이 올바르지 않습니다.")
        return payload


def _server_error_message(error: HTTPError) -> str:
    """Translate HTTP failures without displaying arbitrary server text.

    Some local servers echo request fragments in their error payload.  Keeping
    that payload out of our UI guarantees an API key cannot reappear in an
    error message, browser response, or log forwarded by a caller.
    """

    if error.code in {401, 403}:
        return "로컬 AI 서버가 인증을 거부했습니다. API 키와 서버 설정을 확인해 주세요."
    if error.code == 404:
        return "로컬 AI 서버에서 필요한 OpenAI 호환 API 경로를 찾지 못했습니다. 서버 주소와 /v1 설정을 확인해 주세요."
    if error.code >= 500:
        return f"로컬 AI 서버 내부 오류가 발생했습니다 (HTTP {error.code}). 서버 실행 상태를 확인해 주세요."
    return f"로컬 AI 서버가 HTTP {error.code} 오류를 반환했습니다."


def _stream_content(payload: dict[str, Any]) -> list[str]:
    """Read text deltas from the common OpenAI-compatible SSE shape."""

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
    return parts


def _completion_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""
