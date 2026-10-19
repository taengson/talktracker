"""Shared local LLM connection state for GUI and future web controls."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, RLock, Thread
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable

from .llm_client import LocalLLMCancelled, LocalLLMConnection, LocalLLMConnectionTest, LocalLLMError, LocalLLMModel, OpenAICompatibleLocalClient
from .llm_config import LocalLLMSettings, LocalLLMSettingsStore
from .refinement import AIRefinementRequest, AIRefinementResult, RefinementCancelled, RefinementError, TranscriptRefiner

if TYPE_CHECKING:
    from .analysis import AnalysisResult


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
        self._api_key = ""

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
            # Keep the key only for this running process.  It is deliberately
            # absent from snapshots and the settings file.
            self._api_key = connection.api_key
        return result

    def active_client(self) -> OpenAICompatibleLocalClient:
        """Return a client for the verified, in-memory connection session."""

        with self._lock:
            if not self._connected or not self._settings.is_configured:
                raise LocalLLMError("로컬 AI 연결을 확인한 뒤 AI 정리를 실행해 주세요.")
            connection = LocalLLMConnection(self._settings.base_url, self._api_key)
        return self._client_factory(connection)

    def mark_disconnected(self, message: str = "로컬 AI 연결 확인 필요") -> None:
        """Clear only volatile connectivity; saved endpoint/model remain available."""

        with self._lock:
            self._connected = False
            self._message = message


@dataclass(frozen=True)
class AIRefinementState:
    """A non-secret snapshot of the one active AI refinement task."""

    state: str = "idle"
    message: str = "AI 정리 결과를 만들 수 있습니다."
    started_at: float | None = None
    finished_at: float | None = None
    result: AIRefinementResult | None = None
    error: str | None = None
    owner_id: str | None = None

    @property
    def active(self) -> bool:
        return self.state == "running"


class AIRefinementBusyError(RuntimeError):
    """Raised when a second refinement would overlap the current work."""


class AIRefinementManager:
    """Run one non-destructive local AI refinement task at a time.

    A caller may share ``execution_lock`` with audio analysis.  The task never
    changes the original :class:`AnalysisResult`; it exposes a separate result
    only after the model response has passed validation.
    """

    def __init__(
        self,
        connection_manager: LocalLLMConnectionManager,
        *,
        execution_lock: Lock | None = None,
    ) -> None:
        self._connection_manager = connection_manager
        self._execution_lock = execution_lock or Lock()
        self._lock = RLock()
        self._state = AIRefinementState()
        self._cancel_event: Event | None = None

    def snapshot(self) -> AIRefinementState:
        with self._lock:
            return self._state

    def start(
        self,
        analysis: "AnalysisResult",
        request: AIRefinementRequest,
        *,
        owner_id: str | None = None,
    ) -> AIRefinementState:
        request = request.normalized()
        with self._lock:
            if self._state.active:
                raise AIRefinementBusyError("이미 AI 정리 작업이 진행 중입니다. 완료하거나 중단한 뒤 다시 시도해 주세요.")
            cancel_event = Event()
            self._cancel_event = cancel_event
            self._state = AIRefinementState(
                state="running",
                message="AI 정리 작업을 시작하고 있습니다…",
                started_at=monotonic(),
                owner_id=owner_id,
            )
        Thread(
            target=self._run,
            args=(analysis, request, cancel_event, owner_id),
            name="talktracker-ai-refinement",
            daemon=True,
        ).start()
        return self.snapshot()

    def cancel(self) -> AIRefinementState:
        with self._lock:
            if self._state.active and self._cancel_event:
                self._cancel_event.set()
                self._state = AIRefinementState(
                    state="running",
                    message="AI 정리 중단 요청을 처리하고 있습니다…",
                    started_at=self._state.started_at,
                    owner_id=self._state.owner_id,
                )
            return self._state

    def _run(
        self,
        analysis: "AnalysisResult",
        request: AIRefinementRequest,
        cancel_event: Event,
        owner_id: str | None,
    ) -> None:
        acquired = False
        try:
            if request.selection.needs_ai:
                acquired = self._execution_lock.acquire(blocking=False)
                if not acquired:
                    raise AIRefinementBusyError("다른 음성 분석 또는 AI 정리 작업이 끝난 뒤 다시 시도해 주세요.")
                client = self._connection_manager.active_client()
            else:
                client = None
            refiner = TranscriptRefiner(client)
            result = refiner.refine(
                analysis,
                request,
                model=self._connection_manager.snapshot().settings.model,
                progress=self._update_progress,
                should_cancel=cancel_event.is_set,
            )
        except (RefinementCancelled, LocalLLMCancelled):
            self._finish("cancelled", "AI 정리를 중단했습니다. 원본 전사문은 그대로 유지됩니다.", owner_id=owner_id)
        except (AIRefinementBusyError, LocalLLMError, RefinementError) as error:
            self._finish("failed", str(error), error=str(error), owner_id=owner_id)
        except Exception as error:  # Avoid reflecting an unexpected object or sensitive request data.
            self._finish(
                "failed",
                f"AI 정리 중 처리하지 못한 오류가 발생했습니다 (오류 코드: {type(error).__name__}).",
                error=type(error).__name__,
                owner_id=owner_id,
            )
        else:
            self._finish("completed", "AI 정리 결과를 만들었습니다.", result=result, owner_id=owner_id)
        finally:
            if acquired:
                self._execution_lock.release()

    def _update_progress(self, message: str) -> None:
        with self._lock:
            if not self._state.active:
                return
            self._state = AIRefinementState(
                state="running",
                message=message,
                started_at=self._state.started_at,
                owner_id=self._state.owner_id,
            )

    def _finish(
        self,
        state: str,
        message: str,
        *,
        result: AIRefinementResult | None = None,
        error: str | None = None,
        owner_id: str | None,
    ) -> None:
        with self._lock:
            self._state = AIRefinementState(
                state=state,
                message=message,
                started_at=self._state.started_at,
                finished_at=monotonic(),
                result=result,
                error=error,
                owner_id=owner_id,
            )
