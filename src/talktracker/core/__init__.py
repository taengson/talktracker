"""Local audio-processing pipeline for TalkTracker."""

from .analysis import AnalysisEngine, AnalysisResult, ProcessingError
from .jobs import JobManager, JobNotFoundError, JobStateError
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

__all__ = [
    "AnalysisEngine",
    "AnalysisResult",
    "ProcessingError",
    "JobManager",
    "JobNotFoundError",
    "JobStateError",
    "WHISPER_PROFILES",
    "CudaRuntimeStatus",
    "WhisperProfile",
    "WhisperRuntime",
    "cuda_available",
    "get_cuda_runtime_status",
    "get_cpu_runtime",
    "get_whisper_profile",
    "get_whisper_runtime",
]
