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
    runtime_folder: str
    requires_gpu: bool = False
    compute_type: str = "int8"


WHISPER_PROFILES: tuple[WhisperProfile, ...] = (
    WhisperProfile(
        key="small",
        label="빠른 전사 · Whisper Small",
        short_label="빠른 전사",
        description="긴 녹음을 빠르게 훑어볼 때 알맞습니다.",
        repository="openai/whisper-small",
        source_folder="whisper-small",
        runtime_folder="whisper-small-ct2",
    ),
    WhisperProfile(
        key="turbo",
        label="권장 · Whisper Large-v3 Turbo",
        short_label="권장",
        description="한국어 품질과 처리 시간의 균형을 맞춘 기본 선택입니다.",
        repository="openai/whisper-large-v3-turbo",
        source_folder="whisper-large-v3-turbo",
        runtime_folder="whisper-large-v3-turbo-ct2",
    ),
    WhisperProfile(
        key="large-v3",
        label="최고 정확도 · Whisper Large-v3 (GPU 필요)",
        short_label="최고 정확도",
        description="호환 GPU가 있는 PC에서 가장 높은 품질을 우선할 때 사용합니다.",
        repository="openai/whisper-large-v3",
        source_folder="whisper-large-v3",
        runtime_folder="whisper-large-v3-ct2",
        requires_gpu=True,
        compute_type="float16",
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
