from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from talktracker.core.jobs import JobManager


class FakeResult:
    is_partial = False

    def to_markdown(self) -> str:
        return "# TalkTracker 대화 기록\n\n테스트 결과\n"


class FakeEngine:
    def analyze(self, source: Path, *, progress, on_diarization, should_cancel):
        self.source = source
        progress(10, "음성 파일을 정리하고 있습니다.")
        on_diarization([SimpleNamespace(start=0.0, end=2.5, speaker="화자 1")], 2.5)
        progress(80, "음성을 전사하고 있습니다.")
        return FakeResult()


class JobManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.manager = JobManager(
            self.project_root,
            engine_factory=lambda _root, _profile: FakeEngine(),
        )

    def tearDown(self) -> None:
        self.manager.shutdown()
        self.temporary_directory.cleanup()

    def _wait_for_terminal_state(self, job_id: str) -> dict[str, object]:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = self.manager.snapshot(job_id)
            if snapshot["state"] in {"completed", "cancelled", "failed"}:
                return snapshot
            time.sleep(0.02)
        self.fail("analysis worker did not finish in time")

    def test_upload_runs_once_and_writes_markdown_result(self) -> None:
        job = self.manager.create_upload_job("meeting.m4a", "turbo")
        job.source_path.write_bytes(b"not a real recording")

        queued = self.manager.finish_upload(job.id, job.source_path.stat().st_size)
        self.assertEqual(queued["state"], "queued")

        completed = self._wait_for_terminal_state(job.id)
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(completed["progress"], 100)
        self.assertTrue(completed["result_available"])
        self.assertEqual(completed["speaker_segments"][0]["speaker"], "화자 1")
        self.assertIn("테스트 결과", self.manager.result_path(job.id).read_text(encoding="utf-8"))

    def test_cancelled_upload_never_enters_the_queue(self) -> None:
        job = self.manager.create_upload_job("meeting.m4a", "small")
        cancelled = self.manager.cancel(job.id)
        job.source_path.write_bytes(b"recording")

        after_upload = self.manager.finish_upload(job.id, job.source_path.stat().st_size)
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(after_upload["state"], "cancelled")
        self.assertEqual(self.manager.snapshot(job.id)["state"], "cancelled")

    def test_second_job_waits_for_the_first_job_to_finish(self) -> None:
        first_started = Event()
        release_first = Event()
        started_directories: list[str] = []
        self_test = self

        class SerialEngine:
            def analyze(self, source: Path, *, progress, on_diarization, should_cancel):
                started_directories.append(source.parent.name)
                if len(started_directories) == 1:
                    first_started.set()
                    self_test.assertTrue(release_first.wait(timeout=2))
                progress(100, "완료")
                return FakeResult()

        self.manager.shutdown()
        self.manager = JobManager(
            self.project_root,
            engine_factory=lambda _root, _profile: SerialEngine(),
        )
        first = self.manager.create_upload_job("first.m4a", "turbo")
        second = self.manager.create_upload_job("second.m4a", "turbo")
        first.source_path.write_bytes(b"first")
        second.source_path.write_bytes(b"second")
        self.manager.finish_upload(first.id, first.source_path.stat().st_size)
        self.manager.finish_upload(second.id, second.source_path.stat().st_size)

        self.assertTrue(first_started.wait(timeout=2))
        self.assertEqual(self.manager.snapshot(second.id)["state"], "queued")
        release_first.set()

        self.assertEqual(self._wait_for_terminal_state(first.id)["state"], "completed")
        self.assertEqual(self._wait_for_terminal_state(second.id)["state"], "completed")
        self.assertEqual(started_directories, [first.directory.name, second.directory.name])


if __name__ == "__main__":
    unittest.main()
