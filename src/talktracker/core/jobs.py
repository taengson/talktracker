"""Single-worker job queue shared by the TalkTracker web server."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, RLock, Thread
from typing import Any, Protocol

from .models import get_whisper_profile


SUPPORTED_AUDIO_SUFFIXES = frozenset({".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg"})
TERMINAL_STATES = frozenset({"completed", "cancelled", "failed"})


class JobNotFoundError(KeyError):
    """Raised when a client refers to an unknown analysis job."""


class JobStateError(RuntimeError):
    """Raised when an operation is invalid for the job's current state."""


class Analyzer(Protocol):
    """The small part of ``AnalysisEngine`` required by the queue."""

    def analyze(
        self,
        source: Path,
        *,
        progress: Callable[[int, str], None] | None = None,
        on_diarization: Callable[[list[Any], float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Any: ...


EngineFactory = Callable[[Path, str], Analyzer]


def _now() -> str:
    """Return a portable UTC timestamp for API responses and job metadata."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_engine_factory(project_root: Path, whisper_profile: str) -> Analyzer:
    # Import only in the worker. Starting the web service must still work long
    # enough to show a useful health response when a model dependency is missing.
    from .analysis import AnalysisEngine

    return AnalysisEngine(project_root, whisper_profile=whisper_profile)


@dataclass
class AnalysisJob:
    """Mutable server-side state for one uploaded recording."""

    id: str
    source_name: str
    whisper_profile: str
    directory: Path
    source_path: Path
    created_at: str
    updated_at: str
    state: str = "uploading"
    progress: int = 0
    message: str = "파일을 받고 있습니다."
    source_bytes: int = 0
    duration: float | None = None
    speaker_segments: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    result_path: Path | None = None
    is_partial: bool = False
    cancel_requested: bool = False
    revision: int = 0
    cancel_event: Event = field(default_factory=Event, repr=False)


class JobManager:
    """Persist uploads and execute one local analysis job at a time.

    Models are intentionally shared by one worker instead of running multiple
    analyses in parallel. This keeps the CUDA/CPU memory demand predictable on
    the Linux server that hosts TalkTracker.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        engine_factory: EngineFactory | None = None,
        jobs_root: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.jobs_root = (jobs_root or self.project_root / "outputs" / "jobs").resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._engine_factory = engine_factory or _default_engine_factory
        self._jobs: dict[str, AnalysisJob] = {}
        self._queue_order: list[str] = []
        self._queue: Queue[str | None] = Queue()
        self._lock = RLock()
        self._closed = False
        self._worker = Thread(target=self._worker_loop, name="talktracker-analysis-worker", daemon=True)
        self._worker.start()

    def create_upload_job(self, source_name: str, whisper_profile: str) -> AnalysisJob:
        """Reserve an isolated job directory before an upload is written."""

        get_whisper_profile(whisper_profile)
        safe_name = Path(source_name or "recording").name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            accepted = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
            raise ValueError(f"지원하지 않는 파일 형식입니다. 지원 형식: {accepted}")

        job_id = uuid.uuid4().hex
        directory = self.jobs_root / job_id
        directory.mkdir(parents=True, exist_ok=False)
        now = _now()
        job = AnalysisJob(
            id=job_id,
            source_name=safe_name,
            whisper_profile=whisper_profile,
            directory=directory,
            source_path=directory / f"source{suffix}",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist_locked(job)
        return job

    def finish_upload(self, job_id: str, source_bytes: int) -> dict[str, Any]:
        """Mark an uploaded recording ready and place it behind active jobs."""

        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state == "cancelled":
                return self._snapshot_locked(job)
            if job.state != "uploading":
                raise JobStateError("업로드를 완료할 수 없는 작업 상태입니다.")
            if not job.source_path.is_file() or source_bytes <= 0:
                self._transition_locked(job, "failed", "업로드된 음성 파일을 확인하지 못했습니다.")
                job.error = "업로드된 음성 파일을 확인하지 못했습니다."
                self._persist_locked(job)
                return self._snapshot_locked(job)
            job.source_bytes = source_bytes
            self._transition_locked(job, "queued", "분석 대기열에 추가되었습니다.")
            self._queue_order.append(job.id)
            self._persist_locked(job)
            self._queue.put(job.id)
            return self._snapshot_locked(job)

    def fail_upload(self, job_id: str, message: str) -> None:
        """Record a rejected or interrupted upload without starting analysis."""

        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state in TERMINAL_STATES:
                return
            job.error = message
            self._transition_locked(job, "failed", message)
            self._persist_locked(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued job immediately or request safe cancellation in flight."""

        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state in TERMINAL_STATES:
                return self._snapshot_locked(job)
            job.cancel_requested = True
            job.cancel_event.set()
            if job.state in {"uploading", "queued"}:
                self._remove_from_queue_locked(job.id)
                self._transition_locked(job, "cancelled", "분석 요청을 취소했습니다.")
            else:
                self._touch_locked(job, "중단 요청을 처리하고 있습니다.")
            self._persist_locked(job)
            return self._snapshot_locked(job)

    def snapshot(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked(self._require_job_locked(job_id))

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
            return [self._snapshot_locked(job) for job in jobs]

    def result_path(self, job_id: str) -> Path:
        with self._lock:
            job = self._require_job_locked(job_id)
            if not job.result_path or not job.result_path.is_file():
                raise JobStateError("아직 내려받을 결과가 없습니다.")
            return job.result_path

    def delete(self, job_id: str) -> None:
        """Remove a completed, cancelled, or failed job and all of its files."""

        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state not in TERMINAL_STATES:
                raise JobStateError("실행 중인 작업은 삭제할 수 없습니다. 먼저 중단해 주세요.")
            try:
                job.directory.resolve().relative_to(self.jobs_root)
            except ValueError as error:  # Defensive guard for future path changes.
                raise JobStateError("안전하지 않은 작업 폴더는 삭제할 수 없습니다.") from error
            del self._jobs[job_id]
        shutil.rmtree(job.directory, ignore_errors=True)

    def cleanup_expired(self, max_age_days: int) -> int:
        """Remove untracked job folders older than the configured retention window."""

        if max_age_days < 1:
            raise ValueError("보관 기간은 하루 이상이어야 합니다.")
        cutoff = time.time() - max_age_days * 24 * 60 * 60
        removed = 0
        with self._lock:
            active_directories = {job.directory for job in self._jobs.values()}
        for directory in self.jobs_root.iterdir():
            if not directory.is_dir() or directory in active_directories:
                continue
            try:
                if directory.stat().st_mtime >= cutoff:
                    continue
                directory.resolve().relative_to(self.jobs_root)
            except (OSError, ValueError):
                continue
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
        return removed

    def shutdown(self) -> None:
        """Stop accepting queued work when the web service exits."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            self._remove_from_queue_locked(job_id)
            if job.state != "queued":
                return
            if job.cancel_event.is_set():
                self._transition_locked(job, "cancelled", "분석 요청을 취소했습니다.")
                self._persist_locked(job)
                return
            self._transition_locked(job, "running", "로컬 분석을 시작하고 있습니다.")
            self._persist_locked(job)

        try:
            engine = self._engine_factory(self.project_root, job.whisper_profile)
            result = engine.analyze(
                job.source_path,
                progress=lambda value, message: self._record_progress(job_id, value, message),
                on_diarization=lambda segments, duration: self._record_diarization(job_id, segments, duration),
                should_cancel=job.cancel_event.is_set,
            )
            result_path = job.directory / "result.md"
            result_path.write_text(result.to_markdown(), encoding="utf-8")
        except Exception as error:
            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                job.error = str(error).strip() or type(error).__name__
                self._transition_locked(job, "failed", "음성 파일을 처리하지 못했습니다.")
                self._persist_locked(job)
            return

        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.result_path = result_path
            job.is_partial = bool(getattr(result, "is_partial", False))
            if job.is_partial or job.cancel_event.is_set():
                self._transition_locked(job, "cancelled", "중단 시점까지의 결과를 저장했습니다.")
            else:
                self._transition_locked(job, "completed", "대화 기록을 완성했습니다.")
                job.progress = 100
            self._persist_locked(job)

    def _record_progress(self, job_id: str, value: int, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.state != "running":
                return
            job.progress = max(0, min(100, int(value)))
            self._touch_locked(job, message)
            self._persist_locked(job)

    def _record_diarization(self, job_id: str, segments: list[Any], duration: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.state != "running":
                return
            job.duration = float(duration)
            job.speaker_segments = [
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": str(segment.speaker),
                }
                for segment in segments
            ]
            self._touch_locked(job, "화자 구간을 찾았습니다. 음성을 전사하고 있습니다…")
            self._persist_locked(job)

    def _require_job_locked(self, job_id: str) -> AnalysisJob:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise JobNotFoundError(job_id) from error

    def _remove_from_queue_locked(self, job_id: str) -> None:
        try:
            self._queue_order.remove(job_id)
        except ValueError:
            pass

    def _transition_locked(self, job: AnalysisJob, state: str, message: str) -> None:
        job.state = state
        self._touch_locked(job, message)

    def _touch_locked(self, job: AnalysisJob, message: str) -> None:
        job.message = message
        job.updated_at = _now()
        job.revision += 1

    def _snapshot_locked(self, job: AnalysisJob) -> dict[str, Any]:
        queue_position = None
        if job.state == "queued":
            try:
                queue_position = self._queue_order.index(job.id) + 1
            except ValueError:
                queue_position = None
        return {
            "id": job.id,
            "source_name": job.source_name,
            "whisper_profile": job.whisper_profile,
            "state": job.state,
            "progress": job.progress,
            "message": job.message,
            "source_bytes": job.source_bytes,
            "duration_seconds": job.duration,
            "speaker_segments": list(job.speaker_segments),
            "error": job.error,
            "result_available": bool(job.result_path and job.result_path.is_file()),
            "is_partial": job.is_partial,
            "cancel_requested": job.cancel_requested,
            "queue_position": queue_position,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "revision": job.revision,
        }

    def _persist_locked(self, job: AnalysisJob) -> None:
        """Keep lightweight state next to the upload for troubleshooting."""

        metadata_path = job.directory / "job.json"
        temporary_path = job.directory / ".job.json.tmp"
        temporary_path.write_text(
            json.dumps(self._snapshot_locked(job), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)
