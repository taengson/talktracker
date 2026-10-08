"""Sequential local processing: audio preparation, diarization, transcription, merge."""

from __future__ import annotations

import gc
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

from .models import WhisperProfile, get_whisper_profile


SAMPLE_RATE = 16_000
ProgressCallback = Callable[[int, str], None]
DiarizationCallback = Callable[[list["SpeakerSegment"], float], None]
CancelCallback = Callable[[], bool]


class ProcessingError(RuntimeError):
    """Raised when a user-actionable local processing step fails."""


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TranscriptChunk:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class AnalysisResult:
    source_name: str
    duration: float
    speaker_segments: list[SpeakerSegment]
    utterances: list[Utterance]
    whisper_profile: str = ""
    is_partial: bool = False

    @property
    def speakers(self) -> list[str]:
        return list(dict.fromkeys(segment.speaker for segment in self.speaker_segments))

    def to_markdown(self) -> str:
        lines = [
            "# TalkTracker 대화 기록",
            "",
            f"- 원본 파일: {self.source_name}",
            f"- 길이: {_format_time(self.duration)}",
            f"- 감지된 화자: {len(self.speakers)}명",
        ]
        if self.whisper_profile:
            lines.append(f"- 전사 모델: {self.whisper_profile}")
        lines.extend(
            [
                "- 상태: 중단 시점까지의 결과" if self.is_partial else "- 상태: 완료",
                "",
                "## 대화 내용",
                "",
            ]
        )
        for utterance in self.utterances:
            lines.extend(
                [
                    f"### {utterance.speaker} · {_format_time(utterance.start)}–{_format_time(utterance.end)}",
                    utterance.text,
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"


class AnalysisEngine:
    """Run the two project-local models one after another.

    Nemotron is deliberately unloaded before Whisper starts. This costs a little
    load time but keeps peak memory use lower and avoids competing CPU work.
    """

    def __init__(self, project_root: Path, whisper_profile: str = "turbo") -> None:
        self.project_root = project_root
        self.whisper_profile: WhisperProfile = get_whisper_profile(whisper_profile)
        self.whisper_dir = project_root / "models" / self.whisper_profile.runtime_folder
        self.nemotron_dir = project_root / "models" / "nemotron-3-diarization"

    def analyze(
        self,
        source: Path,
        progress: ProgressCallback | None = None,
        on_diarization: DiarizationCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AnalysisResult:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise ProcessingError("선택한 음성 파일을 찾지 못했습니다.")
        self._ensure_models()

        speaker_segments: list[SpeakerSegment] = []
        transcript_chunks: list[TranscriptChunk] = []
        duration = 0.0
        with tempfile.TemporaryDirectory(prefix="talktracker-") as workspace:
            normalized = Path(workspace) / "input-16khz-mono.wav"
            self._notify(progress, 5, "음성 파일을 처리할 준비를 하고 있습니다…")
            self._normalise_audio(source, normalized)
            audio, duration = self._read_normalized_wav(normalized)
            if self._cancelled(should_cancel):
                return self._partial_result(source, duration, speaker_segments, transcript_chunks)

            self._notify(progress, 15, "화자 구간을 분석하고 있습니다…")
            speaker_segments = self._diarize(audio)
            if on_diarization:
                on_diarization(speaker_segments, duration)
            self._release_memory()
            if self._cancelled(should_cancel):
                return self._partial_result(source, duration, speaker_segments, transcript_chunks)

            self._notify(progress, 52, "음성을 전사하고 있습니다…")
            transcript_chunks, cancelled = self._transcribe(normalized, should_cancel)
            self._release_memory()
            if cancelled or self._cancelled(should_cancel):
                return self._partial_result(source, duration, speaker_segments, transcript_chunks)

        self._notify(progress, 91, "화자와 대화 내용을 시간 기준으로 정리하고 있습니다…")
        utterances = self._merge_speakers_and_text(speaker_segments, transcript_chunks)
        self._notify(progress, 100, "대화 기록을 완성했습니다.")
        return AnalysisResult(
            source_name=source.name,
            duration=duration,
            speaker_segments=speaker_segments,
            utterances=utterances,
            whisper_profile=self.whisper_profile.label,
        )

    def _ensure_models(self) -> None:
        missing = [
            name
            for name, directory, weight_file in (
                (f"{self.whisper_profile.short_label} Whisper 모델", self.whisper_dir, "model.bin"),
                ("Nemotron", self.nemotron_dir, "model.safetensors"),
            )
            if not (directory / weight_file).is_file()
        ]
        if missing:
            raise ProcessingError(f"로컬 모델이 준비되지 않았습니다: {', '.join(missing)}")

    def _normalise_audio(self, source: Path, destination: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ProcessingError("FFmpeg를 찾지 못했습니다. FFmpeg를 설치한 뒤 다시 시도해 주세요.")

        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0 or not destination.is_file():
            details = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
            message = details[-1] if details else "알 수 없는 FFmpeg 오류"
            raise ProcessingError(f"음성 파일을 처리하지 못했습니다. {message}")

    def _read_normalized_wav(self, audio_file: Path) -> tuple[np.ndarray, float]:
        try:
            with wave.open(str(audio_file), "rb") as reader:
                channels = reader.getnchannels()
                sample_width = reader.getsampwidth()
                sample_rate = reader.getframerate()
                frame_count = reader.getnframes()
                frames = reader.readframes(frame_count)
        except (OSError, wave.Error) as error:
            raise ProcessingError(f"정규화된 음성 파일을 읽지 못했습니다. {error}") from error

        if channels != 1 or sample_width != 2 or sample_rate != SAMPLE_RATE:
            raise ProcessingError("음성 파일 정규화 결과가 예상한 16 kHz 모노 PCM 형식이 아닙니다.")
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            raise ProcessingError("음성 파일에 분석할 소리가 없습니다.")
        return audio, frame_count / SAMPLE_RATE

    def _diarize(self, audio: np.ndarray) -> list[SpeakerSegment]:
        processor = AutoProcessor.from_pretrained(str(self.nemotron_dir), local_files_only=True)
        model = AutoModelForAudioFrameClassification.from_pretrained(
            str(self.nemotron_dir),
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()

        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits
        raw_segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]

        segments: list[SpeakerSegment] = []
        for segment in raw_segments:
            start = float(segment["Start"])
            end = float(segment["End"])
            if end <= start:
                continue
            speaker_index = segment.get("Speaker", 0)
            try:
                speaker = f"화자 {int(speaker_index) + 1}"
            except (TypeError, ValueError):
                speaker = f"화자 {speaker_index}"
            segments.append(SpeakerSegment(start=start, end=end, speaker=speaker))
        return segments

    def _transcribe(
        self,
        normalized_audio: Path,
        should_cancel: CancelCallback | None,
    ) -> tuple[list[TranscriptChunk], bool]:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as error:
            raise ProcessingError("faster-whisper가 준비되지 않았습니다. 패키지 준비 스크립트를 다시 실행해 주세요.") from error

        model_options: dict[str, Any] = {
            "device": "cuda" if self.whisper_profile.requires_gpu else "cpu",
            "compute_type": self.whisper_profile.compute_type,
        }
        if not self.whisper_profile.requires_gpu:
            model_options.update(cpu_threads=6, num_workers=1)
        try:
            model = WhisperModel(str(self.whisper_dir), **model_options)
        except RuntimeError as error:
            if self.whisper_profile.requires_gpu:
                raise ProcessingError(
                    "Whisper Large-v3는 호환 CUDA GPU가 필요합니다. "
                    "현재 PC에서는 ‘권장’ 또는 ‘빠른 전사’를 선택해 주세요."
                ) from error
            raise
        segments, _info = model.transcribe(
            str(normalized_audio),
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        chunks: list[TranscriptChunk] = []
        for segment in segments:
            if self._cancelled(should_cancel):
                return chunks, True
            if segment.words:
                for word in segment.words:
                    text = word.word.strip()
                    if not text:
                        continue
                    start = float(word.start or segment.start)
                    end = float(word.end if word.end is not None else start)
                    chunks.append(TranscriptChunk(start=start, end=max(start, end), text=text))
                continue
            text = segment.text.strip()
            if text:
                chunks.append(TranscriptChunk(start=float(segment.start), end=float(segment.end), text=text))
        return chunks, False

    def _merge_speakers_and_text(
        self,
        speaker_segments: list[SpeakerSegment],
        transcript_chunks: list[TranscriptChunk],
    ) -> list[Utterance]:
        utterances: list[Utterance] = []
        for chunk in transcript_chunks:
            speaker = self._speaker_for_chunk(chunk, speaker_segments)
            if utterances and utterances[-1].speaker == speaker and chunk.start - utterances[-1].end <= 1.0:
                previous = utterances[-1]
                utterances[-1] = Utterance(
                    start=previous.start,
                    end=max(previous.end, chunk.end),
                    speaker=speaker,
                    text=_join_text(previous.text, chunk.text),
                )
            else:
                utterances.append(Utterance(chunk.start, chunk.end, speaker, chunk.text))
        return utterances

    @staticmethod
    def _speaker_for_chunk(chunk: TranscriptChunk, speaker_segments: list[SpeakerSegment]) -> str:
        best_speaker = "알 수 없음"
        best_overlap = 0.0
        for segment in speaker_segments:
            overlap = max(0.0, min(chunk.end, segment.end) - max(chunk.start, segment.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = segment.speaker
        if best_overlap > 0:
            return best_speaker

        midpoint = (chunk.start + chunk.end) / 2
        for segment in speaker_segments:
            if segment.start <= midpoint <= segment.end:
                return segment.speaker
        return best_speaker

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _notify(progress: ProgressCallback | None, value: int, message: str) -> None:
        if progress:
            progress(value, message)

    @staticmethod
    def _cancelled(should_cancel: CancelCallback | None) -> bool:
        return bool(should_cancel and should_cancel())

    def _partial_result(
        self,
        source: Path,
        duration: float,
        speaker_segments: list[SpeakerSegment],
        transcript_chunks: list[TranscriptChunk],
    ) -> AnalysisResult:
        return AnalysisResult(
            source_name=source.name,
            duration=duration,
            speaker_segments=speaker_segments,
            utterances=self._merge_speakers_and_text(speaker_segments, transcript_chunks),
            whisper_profile=self.whisper_profile.label,
            is_partial=True,
        )


def _join_text(first: str, second: str) -> str:
    if not first:
        return second
    if not second:
        return first
    return f"{first} {second}".strip()


def _format_time(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{minutes:02}:{seconds:02}"
