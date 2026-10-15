"""Local audio-processing pipeline for TalkTracker."""

from .analysis import AnalysisEngine, AnalysisResult, ProcessingError
from .jobs import JobManager, JobNotFoundError, JobStateError
from .llm_client import (
    LocalLLMConnection,
    LocalLLMConnectionTest,
    LocalLLMError,
    LocalLLMHTTPError,
    LocalLLMModel,
    LocalLLMResponseError,
    OpenAICompatibleLocalClient,
    normalize_local_server_url,
)
from .llm_config import LocalLLMSettings, LocalLLMSettingsStore
from .models import (
    WHISPER_PROFILES,
    CudaRuntimeStatus,
    WhisperProfile,
    WhisperRuntime,
    cuda_available,
    get_cuda_runtime_status,
    get_cpu_runtime,
    get_whisper_profile,
    get_whisper_runtime,
)
from .workflow import LocalLLMConnectionManager, LocalLLMConnectionState

__all__ = [
    "AnalysisEngine",
    "AnalysisResult",
    "ProcessingError",
    "JobManager",
    "JobNotFoundError",
    "JobStateError",
    "LocalLLMConnection",
    "LocalLLMConnectionTest",
    "LocalLLMConnectionManager",
    "LocalLLMConnectionState",
    "LocalLLMError",
    "LocalLLMHTTPError",
    "LocalLLMModel",
    "LocalLLMResponseError",
    "LocalLLMSettings",
    "LocalLLMSettingsStore",
    "OpenAICompatibleLocalClient",
    "WHISPER_PROFILES",
    "CudaRuntimeStatus",
    "WhisperProfile",
    "WhisperRuntime",
    "cuda_available",
    "get_cuda_runtime_status",
    "get_cpu_runtime",
    "get_whisper_profile",
    "get_whisper_runtime",
    "normalize_local_server_url",
]
