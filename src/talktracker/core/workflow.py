"""Shared local LLM connection state for GUI and future web controls."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable

from .llm_client import LocalLLMConnection, LocalLLMConnectionTest, LocalLLMModel, OpenAICompatibleLocalClient
from .llm_config import LocalLLMSettings, LocalLLMSettingsStore


ClientFactory = Callable[[LocalLLMConnection], OpenAICompatibleLocalClient]


@dataclass(frozen=True)
class LocalLLMConnectionState:
    """Non-secret state that both application surfaces may safely display."""

    settings: LocalLLMSettings
    connected: bool
    message: str


class LocalLLMConnectionManager:
    """Own local LLM connection state without retaining API keys.

    Network methods are intentionally blocking; GUI and web adapters run them in
    their existing background worker mechanisms.  The manager itself remains
    thread-safe so both adapters can share it in a later phase.
    """

    def __init__(
        self,
        settings_store: LocalLLMSettingsStore | None = None,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._store = settings_store or LocalLLMSettingsStore()
        self._client_factory = client_factory or (lambda connection: OpenAICompatibleLocalClient(connection))
        self._lock = RLock()
        self._settings = self._store.load()
        self._connected = False
        self._message = "로컬 AI 연결 필요" if not self._settings.is_configured else "로컬 AI 연결 확인 필요"

    def snapshot(self) -> LocalLLMConnectionState:
        with self._lock:
            return LocalLLMConnectionState(self._settings, self._connected, self._message)

    def list_models(self, base_url: str, api_key: str = "") -> list[LocalLLMModel]:
        connection = LocalLLMConnection.create(base_url, api_key)
        return self._client_factory(connection).list_models()

    def verify(self, base_url: str, api_key: str, model: str) -> LocalLLMConnectionTest:
        connection = LocalLLMConnection.create(base_url, api_key)
        result = self._client_factory(connection).test_connection(model)
        with self._lock:
            recent_base_urls = self._settings.recent_base_urls
        settings = self._store.save(
            LocalLLMSettings(
                base_url=connection.base_url,
                model=result.model,
                recent_base_urls=recent_base_urls,
            )
        )
        with self._lock:
            self._settings = settings
            self._connected = True
            self._message = f"로컬 AI 연결됨 · {settings.model}"
        return result

    def mark_disconnected(self, message: str = "로컬 AI 연결 확인 필요") -> None:
        """Clear only volatile connectivity; saved endpoint/model remain available."""

        with self._lock:
            self._connected = False
            self._message = message
