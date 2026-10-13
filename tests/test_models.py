from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from talktracker.core.models import get_whisper_profile, missing_whisper_runtime_files


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


if __name__ == "__main__":
    unittest.main()
