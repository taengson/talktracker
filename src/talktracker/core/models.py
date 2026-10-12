"""Whisper profiles that TalkTracker can prepare and run locally."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WhisperProfile:
    """One user-facing transcription quality option."""

    key: str
    label: str
    short_label: str
    description: str
    repository: str
    source_folder: str
    cpu_runtime_folder: str | None
    cuda_runtime_folder: str

    @property
    def requires_gpu(self) -> bool:
        """Whether this profile has no supported CPU runtime."""

        return self.cpu_runtime_folder is None


@dataclass(frozen=True)
class WhisperRuntime:
    """A converted model and execution settings for one available backend."""

    folder: str
    device: str
    compute_type: str
    label: str

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"


WHISPER_PROFILES: tuple[WhisperProfile, ...] = (
    WhisperProfile(
        key="small",
        label="빠른 전사 · Whisper Small",
        short_label="빠른 전사",
        description="긴 녹음을 빠르게 훑어볼 때 알맞습니다.",
        repository="openai/whisper-small",
        source_folder="whisper-small",
        cpu_runtime_folder="whisper-small-ct2",
        cuda_runtime_folder="whisper-small-ct2-cuda-fp16",
    ),
    WhisperProfile(
        key="turbo",
        label="권장 · Whisper Large-v3 Turbo",
        short_label="권장",
        description="한국어 품질과 처리 시간의 균형을 맞춘 기본 선택입니다.",
        repository="openai/whisper-large-v3-turbo",
        source_folder="whisper-large-v3-turbo",
        cpu_runtime_folder="whisper-large-v3-turbo-ct2",
        cuda_runtime_folder="whisper-large-v3-turbo-ct2-cuda-fp16",
    ),
    WhisperProfile(
        key="large-v3",
        label="최고 정확도 · Whisper Large-v3 (GPU 필요)",
        short_label="최고 정확도",
        description="호환 GPU가 있는 PC에서 가장 높은 품질을 우선할 때 사용합니다.",
        repository="openai/whisper-large-v3",
        source_folder="whisper-large-v3",
        cpu_runtime_folder=None,
        # Retain the existing GPU model location for users who already prepared it.
        cuda_runtime_folder="whisper-large-v3-ct2",
    ),
)

_PROFILES_BY_KEY = {profile.key: profile for profile in WHISPER_PROFILES}


def get_whisper_profile(key: str) -> WhisperProfile:
    """Return a known profile, with a user-actionable error for invalid values."""

    try:
        return _PROFILES_BY_KEY[key]
    except KeyError as error:
        available = ", ".join(profile.key for profile in WHISPER_PROFILES)
        raise ValueError(f"알 수 없는 Whisper 프로필입니다: {key} (선택 가능: {available})") from error


def cuda_available() -> bool:
    """Return whether faster-whisper can actually run on the first CUDA GPU.

    PyTorch can be installed in CPU-only form while CTranslate2 has usable CUDA
    support (or vice versa), so this deliberately probes CTranslate2 itself.
    """

    try:
        import ctranslate2

        return bool(
            ctranslate2.get_cuda_device_count()
            and "float16" in ctranslate2.get_supported_compute_types("cuda")
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        return False


def get_cpu_runtime(profile: WhisperProfile) -> WhisperRuntime | None:
    """Return the CPU fallback runtime for a profile, when one is supported."""

    if not profile.cpu_runtime_folder:
        return None
    return WhisperRuntime(
        folder=profile.cpu_runtime_folder,
        device="cpu",
        compute_type="int8",
        label="CPU · INT8",
    )


def get_whisper_runtime(profile: WhisperProfile, *, use_cuda: bool | None = None) -> WhisperRuntime | None:
    """Prefer CUDA FP16 when CTranslate2 can use it, otherwise use CPU INT8."""

    if use_cuda is None:
        use_cuda = cuda_available()
    if use_cuda:
        return WhisperRuntime(
            folder=profile.cuda_runtime_folder,
            device="cuda",
            compute_type="float16",
            label="CUDA GPU · FP16",
        )
    return get_cpu_runtime(profile)
