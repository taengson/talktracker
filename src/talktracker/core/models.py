"""Whisper profiles that TalkTracker can prepare and run locally."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


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
    runtime_file_sets: tuple[tuple[str, ...], ...]

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


@dataclass(frozen=True)
class CudaRuntimeStatus:
    """The CTranslate2 CUDA capability seen by this running application."""

    available: bool
    message: str


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
        # Systran's ready-to-run Small checkpoint predates the newer
        # preprocessor_config.json/vocabulary.json layout.
        runtime_file_sets=(
            ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
            ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.json"),
        ),
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
        runtime_file_sets=(("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.json"),),
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
        runtime_file_sets=(("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.json"),),
    ),
)

_PROFILES_BY_KEY = {profile.key: profile for profile in WHISPER_PROFILES}

def missing_whisper_runtime_files(model_directory: Path, profile: WhisperProfile) -> tuple[str, ...]:
    """Return missing files for the first compatible CTranslate2 file layout.

    The upstream Small checkpoint uses ``vocabulary.txt`` and intentionally
    has no preprocessor configuration file, while newer checkpoints use the
    JSON layout.  A model is ready when it satisfies any supported layout.
    """

    missing_sets = [
        tuple(name for name in required_files if not (model_directory / name).is_file())
        for required_files in profile.runtime_file_sets
    ]
    if any(not missing for missing in missing_sets):
        return ()
    return min(missing_sets, key=lambda missing: (len(missing), missing))


def get_whisper_profile(key: str) -> WhisperProfile:
    """Return a known profile, with a user-actionable error for invalid values."""

    try:
        return _PROFILES_BY_KEY[key]
    except KeyError as error:
        available = ", ".join(profile.key for profile in WHISPER_PROFILES)
        raise ValueError(f"알 수 없는 Whisper 프로필입니다: {key} (선택 가능: {available})") from error


def get_cuda_runtime_status() -> CudaRuntimeStatus:
    """Describe whether faster-whisper can use CUDA in this process.

    PyTorch can be installed in CPU-only form while CTranslate2 has usable CUDA
    support (or vice versa), so this deliberately probes CTranslate2 itself.
    """

    try:
        import ctranslate2
    except (ImportError, OSError, RuntimeError) as error:
        details = str(error).strip()
        message = "CTranslate2 패키지를 불러오지 못했습니다."
        if details:
            message = f"{message} {details}"
        return CudaRuntimeStatus(False, message)

    try:
        compute_types = ctranslate2.get_supported_compute_types("cuda")
    except (OSError, RuntimeError, ValueError) as error:
        details = str(error).strip() or type(error).__name__
        if "not compiled with CUDA support" in details:
            return CudaRuntimeStatus(
                False,
                "설치된 CTranslate2가 CUDA 없이 빌드된 CPU 전용판입니다. "
                "GPU 환경에서는 현재 운영체제의 Install-Dependencies 스크립트를 다시 실행해 주세요.",
            )
        return CudaRuntimeStatus(False, f"CTranslate2가 CUDA를 초기화하지 못했습니다: {details}")

    try:
        device_count = ctranslate2.get_cuda_device_count()
    except (OSError, RuntimeError, ValueError) as error:
        details = str(error).strip() or type(error).__name__
        return CudaRuntimeStatus(False, f"CTranslate2가 CUDA GPU를 확인하지 못했습니다: {details}")

    if not device_count:
        return CudaRuntimeStatus(
            False,
            "CTranslate2가 사용할 CUDA GPU를 찾지 못했습니다. NVIDIA 드라이버와 GPU 전달 설정을 확인해 주세요.",
        )

    if "float16" not in compute_types:
        return CudaRuntimeStatus(
            False,
            "감지된 CUDA GPU가 Whisper FP16 실행을 지원하지 않습니다.",
        )
    return CudaRuntimeStatus(True, f"CUDA GPU {device_count}개를 감지했습니다.")


def cuda_available() -> bool:
    """Return whether faster-whisper can actually run on the first CUDA GPU."""

    return get_cuda_runtime_status().available


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
        # Linux downloads the upstream faster-whisper CTranslate2 checkpoint
        # directly. The same model.bin supports CPU INT8 and CUDA FP16, so a
        # second converted GPU folder is neither needed nor produced there.
        folder = profile.cuda_runtime_folder
        if sys.platform.startswith("linux") and profile.cpu_runtime_folder:
            folder = profile.cpu_runtime_folder
        return WhisperRuntime(
            folder=folder,
            device="cuda",
            compute_type="float16",
            label="CUDA GPU · FP16",
        )
    return get_cpu_runtime(profile)
