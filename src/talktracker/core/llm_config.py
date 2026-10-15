"""Persist non-secret local LLM connection preferences."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .llm_client import LocalLLMError, normalize_local_server_url


@dataclass(frozen=True)
class LocalLLMSettings:
    """The only LLM connection fields allowed on disk.

    API keys are deliberately absent.  A user enters one again for a fresh app
    session whenever their local server requires it.
    """

    base_url: str = ""
    model: str = ""

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
        if not base_url:
            return LocalLLMSettings(model=model)
        try:
            return LocalLLMSettings(base_url=normalize_local_server_url(base_url), model=model)
        except LocalLLMError:
            return LocalLLMSettings()

    def save(self, settings: LocalLLMSettings) -> LocalLLMSettings:
        base_url = normalize_local_server_url(settings.base_url) if settings.base_url else ""
        model = settings.model.strip()
        sanitized = LocalLLMSettings(base_url=base_url, model=model)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"base_url": sanitized.base_url, "model": sanitized.model}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return sanitized


def default_settings_path() -> Path:
    """Return a platform-appropriate user settings location."""

    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "TalkTracker" / "local-llm.json"
