"""Persist non-secret local LLM connection preferences."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .llm_client import LocalLLMError, normalize_local_server_url


RECENT_SERVER_LIMIT = 8
MAX_RESPONSE_TOKEN_LIMIT = 16_384


@dataclass(frozen=True)
class LocalLLMResponseTokenLimit:
    """A non-secret output budget learned for one server and model pair."""

    base_url: str
    model: str
    max_tokens: int


@dataclass(frozen=True)
class LocalLLMSettings:
    """The local LLM preferences that may safely be written to disk.

    ``recent_base_urls`` contains only servers which completed a connection
    check.  It makes it convenient to return to a known local server without
    turning the list into a record of failed or mistyped addresses.  API keys
    are deliberately absent and remain memory-only.
    """

    base_url: str = ""
    model: str = ""
    recent_base_urls: tuple[str, ...] = ()
    response_token_limits: tuple[LocalLLMResponseTokenLimit, ...] = ()

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)

    def response_token_limit_for(self, base_url: str, model: str) -> int | None:
        """Return a previously successful output budget for this exact pair."""

        normalized_model = model.strip()
        for item in self.response_token_limits:
            if item.base_url == base_url and item.model == normalized_model:
                return item.max_tokens
        return None


class LocalLLMSettingsStore:
    """Atomic JSON settings store outside the project and model directories."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_settings_path()

    def load(self) -> LocalLLMSettings:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return LocalLLMSettings()
        except OSError:
            return LocalLLMSettings()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return LocalLLMSettings()
        if not isinstance(payload, dict):
            return LocalLLMSettings()
        base_url = str(payload.get("base_url", "")).strip()
        model = str(payload.get("model", "")).strip()
        raw_recent_urls = payload.get("recent_base_urls", ())
        if not isinstance(raw_recent_urls, list):
            raw_recent_urls = ()
        raw_response_token_limits = payload.get("response_token_limits", ())
        if not isinstance(raw_response_token_limits, list):
            raw_response_token_limits = ()
        response_token_limits = tuple(
            LocalLLMResponseTokenLimit(
                base_url=str(item.get("base_url", "")),
                model=str(item.get("model", "")),
                max_tokens=item.get("max_tokens", 0),
            )
            for item in raw_response_token_limits
            if isinstance(item, dict)
        )
        try:
            normalized_base_url = normalize_local_server_url(base_url) if base_url else ""
        except LocalLLMError:
            return LocalLLMSettings()
        return _sanitize_settings(
            LocalLLMSettings(
                base_url=normalized_base_url,
                model=model,
                recent_base_urls=tuple(str(value) for value in raw_recent_urls),
                response_token_limits=response_token_limits,
            )
        )

    def save(self, settings: LocalLLMSettings) -> LocalLLMSettings:
        sanitized = _sanitize_settings(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload: dict[str, object] = {
            "base_url": sanitized.base_url,
            "model": sanitized.model,
            "recent_base_urls": list(sanitized.recent_base_urls),
        }
        if sanitized.response_token_limits:
            payload["response_token_limits"] = [
                {"base_url": item.base_url, "model": item.model, "max_tokens": item.max_tokens}
                for item in sanitized.response_token_limits
            ]
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return sanitized


def _sanitize_settings(settings: LocalLLMSettings) -> LocalLLMSettings:
    """Normalize preferences and keep a small, duplicate-free server history."""

    base_url = normalize_local_server_url(settings.base_url) if settings.base_url else ""
    recent_urls: list[str] = []
    for value in (base_url, *settings.recent_base_urls):
        try:
            normalized = normalize_local_server_url(value) if value else ""
        except LocalLLMError:
            continue
        if normalized and normalized not in recent_urls:
            recent_urls.append(normalized)
        if len(recent_urls) == RECENT_SERVER_LIMIT:
            break
    response_token_limits: list[LocalLLMResponseTokenLimit] = []
    seen_pairs: set[tuple[str, str]] = set()
    for item in settings.response_token_limits:
        try:
            normalized_url = normalize_local_server_url(item.base_url)
            max_tokens = int(item.max_tokens)
        except (LocalLLMError, TypeError, ValueError):
            continue
        model = item.model.strip()
        pair = (normalized_url, model)
        if not model or not 1 <= max_tokens <= MAX_RESPONSE_TOKEN_LIMIT or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        response_token_limits.append(LocalLLMResponseTokenLimit(normalized_url, model, max_tokens))
    return LocalLLMSettings(
        base_url=base_url,
        model=settings.model.strip(),
        recent_base_urls=tuple(recent_urls),
        response_token_limits=tuple(response_token_limits),
    )


def default_settings_path() -> Path:
    """Return a platform-appropriate user settings location."""

    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "TalkTracker" / "local-llm.json"
