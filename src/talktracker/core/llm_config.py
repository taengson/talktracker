"""Persist non-secret local LLM connection preferences."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .llm_client import LocalLLMError, normalize_local_server_url


RECENT_SERVER_LIMIT = 8


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

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)


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
        try:
            normalized_base_url = normalize_local_server_url(base_url) if base_url else ""
        except LocalLLMError:
            return LocalLLMSettings()
        return _sanitize_settings(
            LocalLLMSettings(
                base_url=normalized_base_url,
                model=model,
                recent_base_urls=tuple(str(value) for value in raw_recent_urls),
            )
        )

    def save(self, settings: LocalLLMSettings) -> LocalLLMSettings:
        sanitized = _sanitize_settings(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "base_url": sanitized.base_url,
                    "model": sanitized.model,
                    "recent_base_urls": list(sanitized.recent_base_urls),
                },
                ensure_ascii=False,
                indent=2,
            )
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
    return LocalLLMSettings(base_url=base_url, model=settings.model.strip(), recent_base_urls=tuple(recent_urls))


def default_settings_path() -> Path:
    """Return a platform-appropriate user settings location."""

    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "TalkTracker" / "local-llm.json"
