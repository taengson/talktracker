"""Sequential local processing: audio preparation, diarization, transcription, merge."""

from __future__ import annotations

import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Nemotron's Python integration requires a very recent PyTorch/Transformers
# pair and has proven fragile across local platforms.  macOS and Linux use
# NVIDIA's native NeMo-Speech.cpp runtime with the official GGUF checkpoint.
# Windows retains the Python integration used by its existing scripts.
IS_NATIVE_DIARIZATION = sys.platform == "darwin" or sys.platform.startswith("linux")
if not IS_NATIVE_DIARIZATION:
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

from .models import (
    WhisperProfile,
    WhisperRuntime,
    get_cpu_runtime,
    get_whisper_profile,
    get_whisper_runtime,
    missing_whisper_runtime_files,
)


SAMPLE_RATE = 16_000
DEFAULT_TRANSCRIPTION_LANGUAGE = "ko"
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
        self.whisper_runtime: WhisperRuntime | None = get_whisper_runtime(self.whisper_profile)
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
            speaker_segments = self._diarize(normalized, audio, should_cancel)
            if on_diarization:
                on_diarization(speaker_segments, duration)
            self._release_memory()
            if self._cancelled(should_cancel):
                return self._partial_result(source, duration, speaker_segments, transcript_chunks)

            self._notify(progress, 52, "음성을 전사하고 있습니다…")
            transcript_chunks, cancelled = self._transcribe(normalized, duration, should_cancel, progress)
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
            whisper_profile=self._whisper_result_label(),
        )

    def _ensure_models(self) -> None:
        if not self.whisper_runtime:
            raise ProcessingError(
                "Whisper Large-v3는 호환 CUDA GPU가 필요합니다. "
                "현재 PC에서는 ‘권장’ 또는 ‘빠른 전사’를 선택해 주세요."
            )
        whisper_directory = self.project_root / "models" / self.whisper_runtime.folder
        missing = []
        missing_whisper_files = missing_whisper_runtime_files(whisper_directory, self.whisper_profile)
        if missing_whisper_files:
            missing.append(
                f"{self.whisper_profile.short_label} Whisper 모델 ({', '.join(missing_whisper_files)} 없음)"
            )
        nemotron_weight = "Nemotron-3-Diarization.q8_0.gguf" if IS_NATIVE_DIARIZATION else "model.safetensors"
        if not (self.nemotron_dir / nemotron_weight).is_file():
            missing.append("Nemotron")
        if missing:
            raise ProcessingError(f"로컬 모델이 준비되지 않았습니다: {', '.join(missing)}")
        self._validate_whisper_model_config()
        if IS_NATIVE_DIARIZATION and not self._nemo_speech_binary():
            raise ProcessingError(
                "화자 분리 실행기가 준비되지 않았습니다. "
                "현재 운영체제의 Install-Dependencies 스크립트를 다시 실행해 주세요."
            )

    def _validate_whisper_model_config(self) -> None:
        """Fail early when a partial CTranslate2 model download is detected.

        CTranslate2 reads these values lazily while decoding.  Without this
        preflight check, an incomplete config can look like a CUDA failure only
        after the lengthy speaker-separation step has already completed.
        """

        assert self.whisper_runtime is not None
        config_path = self.project_root / "models" / self.whisper_runtime.folder / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProcessingError(
                f"Whisper 모델 구성 파일을 읽지 못했습니다: {config_path.name}. "
                "로컬 모델 관리에서 선택 모델을 다시 준비해 주세요."
            ) from error

        required_keys = ("lang_ids", "suppress_ids", "suppress_ids_begin")
        invalid_keys = [key for key in required_keys if not isinstance(config.get(key), list)]
        if invalid_keys:
            joined_keys = ", ".join(invalid_keys)
            raise ProcessingError(
                "Whisper 모델 구성 파일이 완전하지 않습니다 "
                f"({joined_keys}). 로컬 모델 관리에서 선택 모델을 다시 준비해 주세요."
            )

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

    def _diarize(
        self,
        normalized_audio: Path,
        audio: np.ndarray,
        should_cancel: CancelCallback | None = None,
    ) -> list[SpeakerSegment]:
        if IS_NATIVE_DIARIZATION:
            return self._diarize_with_nemo_speech(normalized_audio, should_cancel)
        return self._diarize_with_transformers(audio)

    def _diarize_with_transformers(self, audio: np.ndarray) -> list[SpeakerSegment]:
        try:
            processor = AutoProcessor.from_pretrained(str(self.nemotron_dir), local_files_only=True)
            model = AutoModelForAudioFrameClassification.from_pretrained(
                str(self.nemotron_dir),
                local_files_only=True,
                torch_dtype=torch.float32,
            )
        except ImportError as error:
            if "librosa" in str(error).lower():
                raise ProcessingError(
                    "화자 분리에 필요한 librosa 패키지가 없습니다. "
                    "패키지 준비 스크립트를 다시 실행해 주세요."
                ) from error
            raise
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

    def _nemo_speech_binary(self) -> Path | None:
        """Find the project-local NeMo-Speech.cpp CLI on supported systems.

        The installer keeps this native dependency under `.tools` rather than
        modifying the user's shell profile.  An environment variable and PATH
        lookup are retained for advanced users who manage the CLI themselves.
        """

        configured = os.environ.get("TALKTRACKER_NEMO_SPEECH")
        candidates = [
            Path(configured).expanduser() if configured else None,
            self.project_root / ".tools" / "nemo-speech" / "bin" / "nemo-speech",
        ]
        path_binary = shutil.which("nemo-speech")
        if path_binary:
            candidates.append(Path(path_binary))
        for candidate in candidates:
            if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _diarize_with_nemo_speech(
        self,
        normalized_audio: Path,
        should_cancel: CancelCallback | None = None,
    ) -> list[SpeakerSegment]:
        """Run the native GGUF Nemotron runtime and read its RTTM result."""

        executable = self._nemo_speech_binary()
        model = self.nemotron_dir / "Nemotron-3-Diarization.q8_0.gguf"
        if not executable:
            raise ProcessingError(
                "화자 분리 실행기를 찾지 못했습니다. "
                "현재 운영체제의 Install-Dependencies 스크립트를 다시 실행해 주세요."
            )
        if not model.is_file():
            raise ProcessingError(
                "Nemotron GGUF 화자 분리 모델이 없습니다. "
                "로컬 모델 관리에서 모델 다운로드를 다시 실행해 주세요."
            )

        rttm_output = normalized_audio.with_suffix(".rttm")
        # A temporary workspace is normally fresh, but do not let a leftover
        # RTTM file make the native CLI reject an otherwise valid retry.
        rttm_output.unlink(missing_ok=True)
        command = [
            str(executable),
            "diarize",
            str(normalized_audio),
            "--model",
            str(model),
            "--format",
            "rttm",
            "--output",
            str(rttm_output),
        ]
        try:
            # NeMo-Speech can take a while on an Intel Mac or a CPU-only Linux
            # host.  A new process session lets a cancel request stop the CLI
            # and any workers it launched instead of waiting for inference to
            # reach a natural checkpoint.
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as output:
                process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                while process.poll() is None:
                    if self._cancelled(should_cancel):
                        self._terminate_process_group(process)
                        return []
                    time.sleep(0.1)
                return_code = process.returncode
                output.seek(0)
                command_output = output.read()
        except OSError as error:
            raise ProcessingError(f"네이티브 화자 분리 실행기를 시작하지 못했습니다. {error}") from error

        if return_code != 0 or not rttm_output.is_file():
            details = [line.strip() for line in command_output.splitlines() if line.strip()]
            last_line = details[-1] if details else "실행 결과를 받지 못했습니다."
            raise ProcessingError(f"Nemotron 화자 분리를 완료하지 못했습니다. {last_line}")

        return self._parse_rttm(rttm_output)

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        """Stop a cancellable native worker without leaving child processes."""

        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                return
            process.wait()

    @staticmethod
    def _parse_rttm(rttm_file: Path) -> list[SpeakerSegment]:
        """Translate standard RTTM speaker turns into the application's labels."""

        segments: list[SpeakerSegment] = []
        speaker_labels: dict[str, str] = {}
        try:
            lines = rttm_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            raise ProcessingError(f"화자 분리 결과 파일을 읽지 못했습니다. {error}") from error

        for line in lines:
            columns = line.split()
            if len(columns) < 8 or columns[0].upper() != "SPEAKER":
                continue
            try:
                start = float(columns[3])
                duration = float(columns[4])
            except ValueError:
                continue
            end = start + duration
            if end <= start:
                continue
            raw_speaker = columns[7]
            speaker = speaker_labels.setdefault(raw_speaker, f"화자 {len(speaker_labels) + 1}")
            segments.append(SpeakerSegment(start=start, end=end, speaker=speaker))
        return segments

    def _transcribe(
        self,
        normalized_audio: Path,
        duration: float,
        should_cancel: CancelCallback | None,
        progress: ProgressCallback | None,
    ) -> tuple[list[TranscriptChunk], bool]:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as error:
            raise ProcessingError("faster-whisper가 준비되지 않았습니다. 패키지 준비 스크립트를 다시 실행해 주세요.") from error

        if not self.whisper_runtime:
            raise ProcessingError("전사 실행 환경을 선택하지 못했습니다.")

        try:
            model = self._load_whisper_model(WhisperModel, self.whisper_runtime)
        except (OSError, RuntimeError, ValueError) as error:
            model = self._load_cpu_fallback(WhisperModel, error, progress)

        try:
            return self._collect_transcript(
                model,
                normalized_audio,
                duration,
                should_cancel,
                progress,
                word_timestamps=self._uses_word_timestamps(),
            )
        except (OSError, RuntimeError, ValueError) as error:
            # Large-v3 deliberately has no CPU fallback.  More importantly,
            # a decoding failure is not evidence that CUDA failed to start:
            # the former can occur minutes after a successful GPU load.
            if self.whisper_profile.requires_gpu:
                self._raise_transcription_failure(error, normalized_audio, word_timestamps=self._uses_word_timestamps())
            cpu_model = self._load_cpu_fallback(WhisperModel, error, progress)
            try:
                return self._collect_transcript(
                    cpu_model,
                    normalized_audio,
                    duration,
                    should_cancel,
                    progress,
                    word_timestamps=self._uses_word_timestamps(),
                )
            except (OSError, RuntimeError, ValueError) as cpu_error:
                self._raise_transcription_failure(
                    cpu_error,
                    normalized_audio,
                    word_timestamps=self._uses_word_timestamps(),
                )

    def _uses_word_timestamps(self) -> bool:
        """Return whether to request Whisper's optional word alignment pass.

        The app always has Whisper segment timings, which are sufficient to
        match a transcript to Nemotron speaker turns.  On some Linux CUDA
        builds, the Large-v3 cross-attention word-alignment call fails inside
        CTranslate2 after decoding.  Skip only that optional pass for Large-v3
        so the GPU-only accuracy profile remains usable.
        """

        return self.whisper_profile.key != "large-v3"

    def _collect_transcript(
        self,
        model: Any,
        normalized_audio: Path,
        duration: float,
        should_cancel: CancelCallback | None,
        progress: ProgressCallback | None,
        *,
        word_timestamps: bool,
    ) -> tuple[list[TranscriptChunk], bool]:
        # Automatic language detection in the current CTranslate2 Large-v3
        # runtime can fail with a JSON type error on some CUDA environments.
        # TalkTracker is Korean-first, so use Korean by default and retain an
        # environment override for another known recording language. An empty
        # value deliberately restores upstream automatic detection.
        language = os.environ.get("TALKTRACKER_LANGUAGE", DEFAULT_TRANSCRIPTION_LANGUAGE).strip() or None
        options: dict[str, Any] = {
            "beam_size": 5,
            "word_timestamps": word_timestamps,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if language:
            options["language"] = language
        segments, _info = model.transcribe(str(normalized_audio), **options)
        chunks: list[TranscriptChunk] = []
        last_progress = 52
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
            else:
                text = segment.text.strip()
                if text:
                    chunks.append(TranscriptChunk(start=float(segment.start), end=float(segment.end), text=text))
            if duration > 0:
                fraction = min(1.0, max(0.0, float(segment.end) / duration))
                current_progress = min(90, 52 + int(fraction * 38))
                if current_progress > last_progress:
                    last_progress = current_progress
                    self._notify(
                        progress,
                        current_progress,
                        f"음성을 전사하고 있습니다… {_format_time(float(segment.end))} / {_format_time(duration)}",
                    )
        return chunks, False

    def _raise_transcription_failure(
        self,
        error: Exception,
        normalized_audio: Path,
        *,
        word_timestamps: bool,
    ) -> None:
        """Report a decoder error without incorrectly blaming CUDA setup."""

        diagnostic_path = self._write_transcription_diagnostic(
            error,
            normalized_audio,
            word_timestamps=word_timestamps,
            traceback_text=traceback.format_exc(),
        )
        runtime_label = self.whisper_runtime.label if self.whisper_runtime else "알 수 없는 실행 환경"
        message = (
            f"{self.whisper_profile.short_label} Whisper 전사 중 오류가 발생했습니다.\n"
            f"CUDA 시작 오류로 단정할 수 없습니다. 실제 전사 처리 중 발생한 오류입니다.\n\n"
            f"실행 환경: {runtime_label}\n"
            f"세부 오류: {error}"
        )
        if diagnostic_path:
            message = f"{message}\n\n진단 로그: {diagnostic_path}"
        raise ProcessingError(message) from error

    def _write_transcription_diagnostic(
        self,
        error: Exception,
        normalized_audio: Path,
        *,
        word_timestamps: bool,
        traceback_text: str,
    ) -> Path | None:
        """Persist the runtime facts needed to reproduce a local decoder error."""

        try:
            assert self.whisper_runtime is not None
            model_dir = self.project_root / "models" / self.whisper_runtime.folder
            config_path = model_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_summary = {
                key: type(config.get(key)).__name__
                for key in ("lang_ids", "suppress_ids", "suppress_ids_begin", "alignment_heads")
            }

            import ctranslate2
            import faster_whisper

            diagnostics_dir = self.project_root / "outputs" / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            destination = diagnostics_dir / f"whisper-{self.whisper_profile.key}-{timestamp}.log"
            details = [
                "TalkTracker Whisper diagnostic",
                f"created_at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
                f"platform: {sys.platform}",
                f"profile: {self.whisper_profile.key}",
                f"runtime: {self.whisper_runtime.label}",
                f"model_dir: {model_dir}",
                f"model_bin_bytes: {(model_dir / 'model.bin').stat().st_size}",
                f"normalized_audio_bytes: {normalized_audio.stat().st_size}",
                f"language: {os.environ.get('TALKTRACKER_LANGUAGE', DEFAULT_TRANSCRIPTION_LANGUAGE).strip() or 'auto'}",
                f"word_timestamps: {word_timestamps}",
                "vad_filter: True",
                "condition_on_previous_text: False",
                f"ctranslate2_version: {ctranslate2.__version__}",
                f"faster_whisper_version: {getattr(faster_whisper, '__version__', 'unknown')}",
                f"config_value_types: {json.dumps(config_summary, ensure_ascii=False, sort_keys=True)}",
                f"error_type: {type(error).__name__}",
                f"error: {error}",
                "",
                "traceback:",
                traceback_text.rstrip(),
                "",
            ]
            destination.write_text("\n".join(details), encoding="utf-8")
            return destination
        except (OSError, json.JSONDecodeError, ImportError, AssertionError):
            return None

    def _load_whisper_model(self, whisper_model: Any, runtime: WhisperRuntime) -> Any:
        model_options: dict[str, Any] = {
            "device": runtime.device,
            "compute_type": runtime.compute_type,
        }
        if not runtime.is_cuda:
            model_options.update(cpu_threads=6, num_workers=1)
        return whisper_model(str(self.project_root / "models" / runtime.folder), **model_options)

    def _load_cpu_fallback(
        self,
        whisper_model: Any,
        gpu_error: Exception,
        progress: ProgressCallback | None,
    ) -> Any:
        """Retry on CPU when CUDA cannot load the model, such as after a VRAM error."""

        if not self.whisper_runtime or not self.whisper_runtime.is_cuda:
            raise gpu_error

        cpu_runtime = get_cpu_runtime(self.whisper_profile)
        if not cpu_runtime:
            details = str(gpu_error).strip()
            if sys.platform.startswith("linux"):
                message = (
                    "Whisper Large-v3의 CUDA GPU 실행을 시작하지 못했습니다. "
                    "scripts/linux/Install-Dependencies.sh를 다시 실행해 "
                    "CUDA 12/cuDNN 9 런타임을 준비해 주세요."
                )
            else:
                message = "Whisper Large-v3의 CUDA GPU 실행을 시작하지 못했습니다. GPU 드라이버와 CUDA 환경을 확인해 주세요."
            if details:
                message = f"{message}\n\n세부 오류: {details}"
            raise ProcessingError(
                message
            ) from gpu_error

        cpu_model = self.project_root / "models" / cpu_runtime.folder / "model.bin"
        if not cpu_model.is_file():
            raise ProcessingError(
                "CUDA GPU 전사를 시작하지 못했고 CPU 대체 모델도 준비되지 않았습니다. "
                "로컬 모델 관리에서 선택 모델을 다시 준비해 주세요."
            ) from gpu_error

        self._notify(progress, 54, "CUDA GPU를 사용할 수 없어 CPU 전사로 전환하고 있습니다…")
        try:
            model = self._load_whisper_model(whisper_model, cpu_runtime)
        except (OSError, RuntimeError, ValueError) as cpu_error:
            raise ProcessingError("CPU 대체 전사 모델도 시작하지 못했습니다.") from cpu_error
        self.whisper_runtime = cpu_runtime
        return model

    def _whisper_result_label(self) -> str:
        if not self.whisper_runtime:
            return self.whisper_profile.label
        return f"{self.whisper_profile.label} · {self.whisper_runtime.label}"

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
        best_speaker = "화자 알 수 없음"
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
        if not IS_NATIVE_DIARIZATION and torch.cuda.is_available():
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
            whisper_profile=self._whisper_result_label(),
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
