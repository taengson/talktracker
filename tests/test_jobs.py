from __future__ import annotations

import tempfile
import time
import unittest
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from talktracker.core.jobs import JobManager
from talktracker.core.analysis import AnalysisResult, SpeakerSegment, Utterance
from talktracker.core.llm_client import LocalLLMConnection, LocalLLMConnectionTest, LocalLLMModel
from talktracker.core.llm_config import LocalLLMSettingsStore
from talktracker.core.models import get_whisper_profile
from talktracker.core.workflow import LocalLLMConnectionManager
from talktracker.server import _index_page, _model_download_command, create_app, main as server_main


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


class FakeLocalLLMClient:
    def __init__(self, connection: LocalLLMConnection) -> None:
        self.connection = connection

    def list_models(self) -> list[LocalLLMModel]:
        return [LocalLLMModel(id="talktracker-test", owned_by="local")]

    def test_connection(self, model: str) -> LocalLLMConnectionTest:
        return LocalLLMConnectionTest(model=model, response_preview="OK")

    def create_chat_completion(
        self,
        model: str,
        *,
        messages,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        retry_on_length: bool = False,
        on_length_retry=None,
        on_length_retry_success=None,
        should_cancel=None,
    ) -> str:
        if should_cancel and should_cancel():
            from talktracker.core.llm_client import LocalLLMCancelled

            raise LocalLLMCancelled("AI 정리 요청을 취소했습니다.")
        return json.dumps(
            {
                "segments": [
                    {"segment_id": "s-0001", "refined_text": "다음 주까지 초안을 준비하겠습니다."},
                    {"segment_id": "s-0002", "refined_text": "일정을 확인하겠습니다."},
                ],
                "summary": {
                    "overview": "초안 준비 일정과 확인 작업을 논의했다.",
                    "key_points": ["다음 주까지 초안을 준비한다."],
                    "decisions": [],
                    "action_items": ["일정을 확인한다."],
                    "open_questions": [],
                },
            },
            ensure_ascii=False,
        )


class RefinableFakeEngine:
    def analyze(self, source: Path, *, progress, on_diarization, should_cancel):
        progress(100, "완료")
        on_diarization([SimpleNamespace(start=0.0, end=7.0, speaker="화자 1")], 7.0)
        return AnalysisResult(
            source_name=source.name,
            duration=7.0,
            speaker_segments=[SpeakerSegment(0.0, 7.0, "화자 1")],
            utterances=[
                Utterance(0.0, 3.0, "화자 1", "다음주 까지 초안 준비 할게요"),
                Utterance(3.2, 7.0, "화자 1", "일정 확인 할게요"),
            ],
        )


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


class ServerPageTests(unittest.TestCase):
    def test_windows_web_runner_forwards_server_long_options_unchanged(self) -> None:
        runner = Path(__file__).resolve().parents[1] / "scripts" / "windows" / "Run-Web.ps1"
        content = runner.read_text(encoding="utf-8")

        self.assertNotIn("[CmdletBinding()]", content)
        self.assertIn("$ServerArguments = @()", content)
        self.assertIn("$args.Count", content)
        self.assertIn("'-BindHost'", content)
        self.assertIn("'--host'", content)
        self.assertIn("& $Python @Arguments", content)

    def test_model_preparation_command_matches_the_server_platform(self) -> None:
        profile = get_whisper_profile("small")

        with patch("talktracker.server.sys.platform", "win32"):
            self.assertEqual(
                _model_download_command(profile),
                "powershell -ExecutionPolicy Bypass -File "
                ".\\scripts\\windows\\Download-Models.ps1 -WhisperProfile small -PrepareForLocalUse",
            )
        with patch("talktracker.server.sys.platform", "darwin"):
            self.assertEqual(
                _model_download_command(profile),
                "bash scripts/macos/Download-Models.sh --whisper-profile small",
            )
        with patch("talktracker.server.sys.platform", "linux"):
            self.assertEqual(
                _model_download_command(profile),
                "bash scripts/linux/Download-Models.sh --whisper-profile small",
            )

    def test_embedded_client_script_uses_a_valid_quote_key_for_html_escaping(self) -> None:
        """Keep Python string escaping from corrupting the browser client."""

        page = _index_page()
        self.assertIn("'\"':\"&quot;\"", page)
        self.assertNotIn('"\\\\\"":"&quot;"', page)
        self.assertIn('request("/api/profiles")', page)
        self.assertIn('event.preventDefault()', page)
        self.assertIn('function renderTimeline(segments, durationSeconds)', page)
        self.assertIn('class="speaker-lane"', page)
        self.assertIn('id="server-setup"', page)
        self.assertIn('id="local-ai"', page)
        self.assertIn('/api/local-llm/verify', page)
        self.assertIn('id="llm-api-key"', page)
        self.assertIn('id="llm-url-options"', page)
        self.assertIn('id="ai-refinement"', page)
        self.assertIn('/ai-refinement/cancel', page)
        self.assertNotIn('id="ai-original"', page)
        self.assertIn("원본 전사는 분석 결과로 자동 보존됩니다.", page)
        self.assertIn("AI_RESULT_LABELS", page)
        self.assertIn('form.classList.add("hidden")', page)
        self.assertNotIn('id="token"', page)
        protected_page = _index_page(access_token_required=True)
        self.assertIn('id="token"', protected_page)
        self.assertIn('refreshProtectedData', protected_page)
        network_page = _index_page(local_llm_available=False)
        self.assertNotIn('id="local-ai"', network_page)
        self.assertNotIn('id="llm-api-key"', network_page)
        closed_network_page = _index_page(local_llm_over_network=True)
        self.assertIn('id="local-ai"', closed_network_page)
        self.assertIn("사내 폐쇄망 모드입니다.", closed_network_page)
        self.assertIn("API 키와 AI 요청은 네트워크를 통해 서버로 전달됩니다.", closed_network_page)

    def test_local_llm_api_lists_and_verifies_without_persisting_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings_path = root / "settings" / "local-llm.json"
            llm_manager = LocalLLMConnectionManager(
                settings_store=LocalLLMSettingsStore(settings_path),
                client_factory=FakeLocalLLMClient,
            )
            with TestClient(create_app(project_root=root, llm_connection_manager=llm_manager)) as client:
                self.assertFalse(client.get("/api/local-llm").json()["configured"])
                models = client.post(
                    "/api/local-llm/models",
                    json={"base_url": "http://localhost:11434", "api_key": "temporary-secret"},
                )
                self.assertEqual(models.status_code, 200)
                self.assertEqual(models.json()["models"], [{"id": "talktracker-test", "owned_by": "local"}])

                verified = client.post(
                    "/api/local-llm/verify",
                    json={"base_url": "http://localhost:11434", "api_key": "temporary-secret", "model": "talktracker-test"},
                )
                self.assertEqual(verified.status_code, 200)
                self.assertTrue(verified.json()["connected"])
                self.assertEqual(verified.json()["model"], "talktracker-test")
                self.assertEqual(verified.json()["recent_base_urls"], ["http://localhost:11434/v1"])

                network_server = client.post(
                    "/api/local-llm/models",
                    json={"base_url": "http://spark-f8df:8000", "api_key": "temporary-secret"},
                )
                self.assertEqual(network_server.status_code, 200)
                self.assertEqual(network_server.json()["models"], [{"id": "talktracker-test", "owned_by": "local"}])

                unsafe_url = client.post(
                    "/api/local-llm/models",
                    json={"base_url": "http://user:pass@spark-f8df:8000", "api_key": "temporary-secret"},
                )
                self.assertEqual(unsafe_url.status_code, 422)
                self.assertNotIn("temporary-secret", unsafe_url.text)

                malformed = client.post(
                    "/api/local-llm/models",
                    json={"base_url": 42, "api_key": "temporary-secret"},
                )
                self.assertEqual(malformed.status_code, 422)
                self.assertNotIn("temporary-secret", malformed.text)

            self.assertNotIn("temporary-secret", settings_path.read_text(encoding="utf-8"))

    def test_web_ai_refinement_preserves_separate_results_across_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = JobManager(root, engine_factory=lambda _root, _profile: RefinableFakeEngine())
            settings_path = root / "settings" / "local-llm.json"
            llm_manager = LocalLLMConnectionManager(
                settings_store=LocalLLMSettingsStore(settings_path),
                client_factory=FakeLocalLLMClient,
            )
            try:
                job = manager.create_upload_job("meeting.m4a", "turbo")
                job.source_path.write_bytes(b"recording")
                manager.finish_upload(job.id, job.source_path.stat().st_size)
                deadline = time.monotonic() + 3
                while manager.snapshot(job.id)["state"] not in {"completed", "failed"} and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(manager.snapshot(job.id)["state"], "completed")
                with TestClient(create_app(project_root=root, manager=manager, llm_connection_manager=llm_manager)) as client:
                    verified = client.post(
                        "/api/local-llm/verify",
                        json={"base_url": "http://localhost:11434", "api_key": "temporary-secret", "model": "talktracker-test"},
                    )
                    self.assertEqual(verified.status_code, 200)
                    started = client.post(
                        f"/api/jobs/{job.id}/ai-refinement",
                        json={
                            "selection": {"refined": True, "summary": False},
                            "background": "프로젝트 회의",
                            "speaker_guides": [{"speaker": "화자 1", "display_name": "손태영", "guidance": "프로젝트 매니저"}],
                        },
                    )
                    self.assertEqual(started.status_code, 200)
                    deadline = time.monotonic() + 3
                    status_payload = started.json()
                    while status_payload["active"] and time.monotonic() < deadline:
                        time.sleep(0.02)
                        status_payload = client.get(f"/api/jobs/{job.id}/ai-refinement").json()
                    self.assertEqual(status_payload["state"], "completed")
                    self.assertIn("elapsed_seconds", status_payload)
                    self.assertGreaterEqual(status_payload["elapsed_seconds"], 0)
                    sections = status_payload["result"]["sections"]
                    self.assertIn("original", sections)
                    self.assertIn("다음 주까지 초안을 준비하겠습니다.", sections["refined"])
                    self.assertIn("(손태영)", sections["refined"])
                    self.assertNotIn("summary", sections)
                    self.assertNotIn("프로젝트 매니저", str(status_payload))
                    summary_started = client.post(
                        f"/api/jobs/{job.id}/ai-refinement",
                        json={
                            "selection": {"refined": False, "summary": True},
                            "background": "프로젝트 회의",
                            "speaker_guides": [{"speaker": "화자 1", "display_name": "손태영", "guidance": "프로젝트 매니저"}],
                        },
                    )
                    self.assertEqual(summary_started.status_code, 200)
                    status_payload = summary_started.json()
                    while status_payload["active"] and time.monotonic() < deadline:
                        time.sleep(0.02)
                        status_payload = client.get(f"/api/jobs/{job.id}/ai-refinement").json()
                    self.assertEqual(status_payload["state"], "completed")
                    sections = status_payload["result"]["sections"]
                    self.assertIn("다음 주까지 초안을 준비하겠습니다.", sections["refined"])
                    self.assertIn("초안 준비 일정과 확인 작업을 논의했다.", sections["summary"])
                    original_document = client.get(f"/api/jobs/{job.id}/ai-results/original.md")
                    self.assertEqual(original_document.status_code, 200)
                    self.assertIn("다음주 까지 초안 준비 할게요", original_document.text)
                    refined_document = client.get(f"/api/jobs/{job.id}/ai-results/refined.md")
                    self.assertEqual(refined_document.status_code, 200)
                    self.assertIn("# 다듬은 대화", refined_document.text)
                    summary_document = client.get(f"/api/jobs/{job.id}/ai-results/summary.md")
                    self.assertEqual(summary_document.status_code, 200)
                    self.assertIn("# 요약", summary_document.text)
            finally:
                manager.shutdown()

    def test_server_disables_missing_models_and_rejects_their_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(project_root=Path(directory))) as client:
                page = client.get("/").text
                self.assertIn("분석 요청 수신 불가", page)
                self.assertIn("이 PC는 현재 요청 처리 서버가 아닙니다.", page)
                self.assertIn("파일 업로드와 분석 요청은 받지 않습니다.", page)
                self.assertNotIn("config.json", page)
                self.assertNotIn('id="upload-form"', page)

                profiles = client.get("/api/profiles").json()["profiles"]
                small = next(profile for profile in profiles if profile["key"] == "small")
                self.assertFalse(small["available"])
                self.assertIn("Whisper 모델이 준비되지 않았습니다", small["unavailable_reason"])

                response = client.post(
                    "/api/jobs",
                    data={"profile": "small"},
                    files={"file": ("meeting.m4a", b"recording", "audio/mp4")},
                )
                self.assertEqual(response.status_code, 409)
                self.assertIn("Whisper 모델이 준비되지 않았습니다", response.json()["detail"])

    def test_internal_network_mode_can_start_without_an_access_token(self) -> None:
        uvicorn_run = Mock()
        with (
            patch("talktracker.server.create_app", return_value=object()),
            patch("sys.argv", ["talktracker.server", "--host", "0.0.0.0"]),
            patch.dict("os.environ", {"TALKTRACKER_WEB_TOKEN": ""}, clear=False),
            patch.dict("sys.modules", {"uvicorn": SimpleNamespace(run=uvicorn_run)}),
        ):
            server_main()

        self.assertEqual(uvicorn_run.call_args.kwargs["host"], "0.0.0.0")

    def test_internal_network_option_enables_local_ai_controls(self) -> None:
        uvicorn_run = Mock()
        create_app_mock = Mock(return_value=object())
        with (
            patch("talktracker.server.create_app", create_app_mock),
            patch(
                "sys.argv",
                ["talktracker.server", "--host", "0.0.0.0", "--allow-local-ai-over-network"],
            ),
            patch.dict("os.environ", {"TALKTRACKER_WEB_TOKEN": ""}, clear=False),
            patch.dict("sys.modules", {"uvicorn": SimpleNamespace(run=uvicorn_run)}),
        ):
            server_main()

        self.assertEqual(uvicorn_run.call_args.kwargs["host"], "0.0.0.0")
        self.assertTrue(create_app_mock.call_args.kwargs["local_llm_enabled"])
        self.assertTrue(create_app_mock.call_args.kwargs["local_llm_over_network"])

    def test_internal_network_without_the_option_rejects_local_ai_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(project_root=Path(directory), local_llm_enabled=False)) as client:
                response = client.get("/api/local-llm")

        self.assertEqual(response.status_code, 404)
        self.assertIn("--allow-local-ai-over-network", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
