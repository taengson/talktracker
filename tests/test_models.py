from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from talktracker.core.models import get_whisper_profile, missing_whisper_runtime_files
from talktracker.main import TalkTrackerApp


class WhisperRuntimeFilesTests(unittest.TestCase):
    def test_small_accepts_the_official_vocabulary_text_layout(self) -> None:
        profile = get_whisper_profile("small")
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
                (model_directory / name).write_text("test", encoding="utf-8")

            self.assertEqual(missing_whisper_runtime_files(model_directory, profile), ())

    def test_small_reports_the_actual_missing_file_for_its_repository(self) -> None:
        profile = get_whisper_profile("small")
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            for name in ("config.json", "model.bin", "tokenizer.json"):
                (model_directory / name).write_text("test", encoding="utf-8")

            self.assertEqual(missing_whisper_runtime_files(model_directory, profile), ("vocabulary.txt",))

    def test_turbo_still_requires_the_modern_json_layout(self) -> None:
        profile = get_whisper_profile("turbo")
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
                (model_directory / name).write_text("test", encoding="utf-8")

            self.assertEqual(
                missing_whisper_runtime_files(model_directory, profile),
                ("preprocessor_config.json", "vocabulary.json"),
            )

    def test_gui_model_status_uses_the_selected_profile(self) -> None:
        """Model-state refresh must not prevent the desktop app from opening."""

        with tempfile.TemporaryDirectory() as directory, patch("talktracker.main.USES_NATIVE_NEMOTRON", False):
            app = TalkTrackerApp.__new__(TalkTrackerApp)
            app.whisper_profile_key = SimpleNamespace(get=lambda: "small")
            app._whisper_model_label = lambda: "빠른 전사 Whisper 전사 · CPU · INT8"
            app._model_locations = lambda: {
                app._whisper_model_label(): (Path(directory) / "whisper-small-ct2", "model.bin"),
                "Nemotron 화자 분리": (Path(directory) / "nemotron", "model.safetensors"),
            }
            app.model_status_button = SimpleNamespace(configure=lambda **_kwargs: None)

            app.refresh_model_status()

            self.assertIn(app._whisper_model_label(), app.missing_models)


if __name__ == "__main__":
    unittest.main()
