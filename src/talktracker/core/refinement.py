"""Safe, short-transcript refinement through an OpenAI-compatible LLM.

The original transcription stays immutable.  A model only returns replacement
text keyed by an internal segment id, so speaker names, timing, and ordering
continue to come from :mod:`analysis` rather than an LLM response.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .llm_client import LocalLLMError, OpenAICompatibleLocalClient

if TYPE_CHECKING:
    from .analysis import AnalysisResult, Utterance


PROMPT_VERSION = "2026-10-19.1"
ProgressCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]
LengthRetrySuccessCallback = Callable[[int], None]


class RefinementError(LocalLLMError):
    """Raised when an LLM response cannot safely become a TalkTracker result."""


class RefinementCancelled(RefinementError):
    """Raised after the user asks the active refinement task to stop."""


@dataclass(frozen=True)
class ResultSelection:
    """The output sections explicitly requested by a user."""

    original: bool = False
    refined: bool = False
    summary: bool = False

    def validate(self) -> None:
        if not (self.original or self.refined or self.summary):
            raise RefinementError("결과로 받을 항목을 하나 이상 선택해 주세요.")
        if self.original and self.needs_ai:
            raise RefinementError("원본 전사 메시지는 다듬은 대화 또는 요약과 함께 선택할 수 없습니다.")

    @property
    def needs_ai(self) -> bool:
        return self.refined or self.summary


@dataclass(frozen=True)
class SpeakerGuide:
    """Optional display name and context for one detected speaker."""

    speaker: str
    guidance: str = ""
    display_name: str = ""


@dataclass(frozen=True)
class AIRefinementRequest:
    selection: ResultSelection
    background: str = ""
    speaker_guides: tuple[SpeakerGuide, ...] = ()

    def normalized(self) -> "AIRefinementRequest":
        self.selection.validate()
        guides: list[SpeakerGuide] = []
        for guide in self.speaker_guides:
            speaker = guide.speaker.strip()
            guidance = guide.guidance.strip()
            display_name = guide.display_name.strip()
            if speaker and (guidance or display_name):
                guides.append(SpeakerGuide(speaker=speaker, guidance=guidance, display_name=display_name))
        return AIRefinementRequest(
            selection=self.selection,
            background=self.background.strip(),
            speaker_guides=tuple(guides),
        )


@dataclass(frozen=True)
class ConversationSummary:
    overview: str
    key_points: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    action_items: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()


@dataclass(frozen=True)
class AIRefinementResult:
    """Non-destructive output of one short-transcript LLM request."""

    selection: ResultSelection
    refined_text_by_segment_id: dict[str, str] = field(default_factory=dict)
    speaker_display_names: dict[str, str] = field(default_factory=dict)
    summary: ConversationSummary | None = None
    warnings: tuple[str, ...] = ()
    prompt_version: str = PROMPT_VERSION
    model: str = ""


def merge_ai_refinement_results(
    previous: AIRefinementResult | None,
    latest: AIRefinementResult,
) -> AIRefinementResult:
    """Keep prior AI documents while replacing only newly requested ones.

    A user can create a refined conversation first and request a summary in a
    later pass.  Those are separate documents for the same immutable analysis,
    so a summary-only pass must not discard the already validated refinement.
    """

    if previous is None:
        return latest
    refined_requested = previous.selection.refined or latest.selection.refined
    summary_requested = previous.selection.summary or latest.selection.summary
    refined_text = latest.refined_text_by_segment_id if latest.selection.refined else previous.refined_text_by_segment_id
    summary = latest.summary if latest.selection.summary else previous.summary
    warnings = tuple(dict.fromkeys((*previous.warnings, *latest.warnings)))
    return AIRefinementResult(
        selection=ResultSelection(refined=refined_requested, summary=summary_requested),
        refined_text_by_segment_id=refined_text,
        speaker_display_names={**previous.speaker_display_names, **latest.speaker_display_names},
        summary=summary,
        warnings=warnings,
        prompt_version=latest.prompt_version,
        model=latest.model or previous.model,
    )


@dataclass(frozen=True)
class _Segment:
    id: str
    utterance: "Utterance"


class TranscriptRefiner:
    """Build one JSON-only request for a short, already transcribed recording."""

    def __init__(self, client: OpenAICompatibleLocalClient | None) -> None:
        self.client = client

    def refine(
        self,
        analysis: "AnalysisResult",
        request: AIRefinementRequest,
        *,
        model: str,
        response_token_limit: int | None = None,
        on_length_retry_success: LengthRetrySuccessCallback | None = None,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AIRefinementResult:
        normalized = request.normalized()
        if not normalized.selection.needs_ai:
            return AIRefinementResult(selection=normalized.selection)
        if not analysis.utterances:
            raise RefinementError("AI로 정리할 전사문이 없습니다. 음성 분석 결과를 먼저 확인해 주세요.")
        if not self.client:
            raise RefinementError("로컬 AI 연결을 확인한 뒤 다시 시도해 주세요.")
        self._raise_if_cancelled(should_cancel)

        segments = tuple(_Segment(f"s-{index:04d}", utterance) for index, utterance in enumerate(analysis.utterances, start=1))
        estimated_limit = max(1_024, min(8_192, 512 + sum(len(segment.utterance.text) for segment in segments) * 2))
        remembered_limit = max(0, int(response_token_limit or 0))
        requested_limit = max(estimated_limit, remembered_limit)
        if progress:
            if remembered_limit:
                progress("이전에 확인한 응답 길이 설정으로 AI 정리를 시작하고 있습니다…")
            else:
                progress("AI가 짧은 전사 전체를 읽고 정리하고 있습니다…")
        successful_retry_limits: list[int] = []
        response = self.client.create_chat_completion(
            model,
            messages=self._messages(segments, normalized),
            temperature=0,
            max_tokens=requested_limit,
            json_mode=True,
            retry_on_length=True,
            on_length_retry=(
                lambda _current_limit, _retry_limit: progress(
                    "AI 응답 길이 한도에 도달해 더 큰 설정으로 다시 정리하고 있습니다. 완료되면 이 연결에 기억합니다…"
                )
                if progress
                else None
            ),
            on_length_retry_success=successful_retry_limits.append,
            should_cancel=should_cancel,
        )
        self._raise_if_cancelled(should_cancel)
        payload = _parse_response_json(response)
        warnings: list[str] = []
        refined: dict[str, str] = {}
        if normalized.selection.refined:
            refined, refinement_warnings = _validated_refinements(payload, segments)
            warnings.extend(refinement_warnings)
        summary = _validated_summary(payload) if normalized.selection.summary else None
        if progress:
            progress("AI 정리 결과를 검증하고 있습니다…")
        result = AIRefinementResult(
            selection=normalized.selection,
            refined_text_by_segment_id=refined,
            speaker_display_names={
                guide.speaker: guide.display_name for guide in normalized.speaker_guides if guide.display_name
            },
            summary=summary,
            warnings=tuple(warnings),
            model=model.strip(),
        )
        if successful_retry_limits and on_length_retry_success:
            on_length_retry_success(successful_retry_limits[-1])
        return result

    @staticmethod
    def _messages(segments: tuple[_Segment, ...], request: AIRefinementRequest) -> list[dict[str, str]]:
        system = (
            "You refine a Korean conversation transcript. Use only the provided transcript and optional context. "
            "Do not invent facts, speakers, times, names, decisions, or action items. "
            "Use a supplied display_name as the speaker label in any generated wording about that speaker. "
            "Speaker guidance is only context and must not be output as a separate field. "
            "Keep each segment's meaning. Return JSON only, with no markdown fence or explanation. "
            "When refinement is requested, return every requested segment_id exactly once with a non-empty refined_text. "
            "When summary is requested, return a summary object with overview (string), key_points, decisions, "
            "action_items, and open_questions (arrays of strings). Use empty arrays when the transcript has no evidence."
        )
        content = {
            "prompt_version": PROMPT_VERSION,
            "requested_output": {
                "refined_conversation": request.selection.refined,
                "summary": request.selection.summary,
            },
            "conversation_background": request.background,
            "speaker_guides": [
                {"speaker": guide.speaker, "display_name": guide.display_name, "guidance": guide.guidance}
                for guide in request.speaker_guides
            ],
            "segments": [
                {"segment_id": segment.id, "speaker": segment.utterance.speaker, "text": segment.utterance.text}
                for segment in segments
            ],
            "response_shape": {
                "segments": [{"segment_id": "s-0001", "refined_text": "..."}],
                "summary": {
                    "overview": "...",
                    "key_points": ["..."],
                    "decisions": ["..."],
                    "action_items": ["..."],
                    "open_questions": ["..."],
                },
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        ]

    @staticmethod
    def _raise_if_cancelled(should_cancel: CancelCallback | None) -> None:
        if should_cancel and should_cancel():
            raise RefinementCancelled("AI 정리 요청을 취소했습니다.")


def render_original_conversation(analysis: "AnalysisResult") -> str:
    return _render_conversation(analysis.utterances, {})


def render_refined_conversation(analysis: "AnalysisResult", result: AIRefinementResult) -> str:
    return _render_conversation(analysis.utterances, result.refined_text_by_segment_id, result.speaker_display_names)


def render_summary(summary: ConversationSummary) -> str:
    sections = [
        ("개요", (summary.overview,) if summary.overview else ()),
        ("핵심 내용", summary.key_points),
        ("결정 사항", summary.decisions),
        ("할 일", summary.action_items),
        ("확인이 필요한 내용", summary.open_questions),
    ]
    lines: list[str] = []
    for title, values in sections:
        lines.append(f"[{title}]")
        if values:
            lines.extend(f"- {value}" for value in values)
        else:
            lines.append("- 전사문에서 확인된 내용이 없습니다.")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_selected_markdown(analysis: "AnalysisResult", result: AIRefinementResult) -> str:
    """Render exactly the requested, non-secret output sections for saving."""

    lines = [
        "# TalkTracker 대화 기록",
        "",
        f"- 원본 파일: {analysis.source_name}",
        f"- 길이: {_format_time(analysis.duration)}",
        "- 상태: AI 정리 완료",
    ]
    if result.model:
        lines.append(f"- AI 모델: {result.model}")
        lines.append(f"- 프롬프트 버전: {result.prompt_version}")
    for title, body in (
        ("원본 전사 메시지", render_original_conversation(analysis) if result.selection.original else ""),
        ("다듬은 대화", render_refined_conversation(analysis, result) if result.selection.refined else ""),
        ("요약", render_summary(result.summary) if result.selection.summary and result.summary else ""),
    ):
        if body:
            lines.extend(["", f"## {title}", "", body])
    if result.warnings:
        lines.extend(["", "## AI 처리 참고", "", *(f"- {warning}" for warning in result.warnings)])
    return "\n".join(lines).rstrip() + "\n"


def render_result_documents(analysis: "AnalysisResult", result: AIRefinementResult | None = None) -> dict[str, str]:
    """Create one independent Markdown document for each available result."""

    documents = {"original": analysis.to_markdown()}
    if not result:
        return documents

    metadata = [f"- 원본 파일: {analysis.source_name}", f"- 길이: {_format_time(analysis.duration)}"]
    if result.model:
        metadata.extend((f"- AI 모델: {result.model}", f"- 프롬프트 버전: {result.prompt_version}"))
    warnings = ["", "## AI 처리 참고", "", *(f"- {warning}" for warning in result.warnings)] if result.warnings else []
    if result.selection.refined:
        documents["refined"] = "\n".join(
            ["# 다듬은 대화", "", *metadata, "", "## 대화 내용", "", render_refined_conversation(analysis, result), *warnings]
        ).rstrip() + "\n"
    if result.selection.summary and result.summary:
        documents["summary"] = "\n".join(
            ["# 요약", "", *metadata, "", "## 요약 내용", "", render_summary(result.summary), *warnings]
        ).rstrip() + "\n"
    return documents


def _render_conversation(
    utterances: list["Utterance"], replacements: dict[str, str], speaker_display_names: dict[str, str] | None = None
) -> str:
    lines: list[str] = []
    speaker_display_names = speaker_display_names or {}
    for index, utterance in enumerate(utterances, start=1):
        text = replacements.get(f"s-{index:04d}", utterance.text)
        speaker = speaker_display_names.get(utterance.speaker, utterance.speaker)
        lines.extend(
            [
                f"({speaker})  {_format_time(utterance.start)} – {_format_time(utterance.end)}",
                text,
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _format_time(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def _parse_response_json(response: str) -> dict[str, Any]:
    candidate = response.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        candidate = candidate[first_newline + 1 :] if first_newline >= 0 else ""
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise RefinementError("AI 응답에서 필요한 JSON 형식을 찾지 못했습니다. 원본 전사문은 그대로 유지됩니다.")
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as error:
        raise RefinementError("AI 응답의 JSON 형식이 올바르지 않습니다. 원본 전사문은 그대로 유지됩니다.") from error
    if not isinstance(payload, dict):
        raise RefinementError("AI 응답의 JSON 형식이 올바르지 않습니다. 원본 전사문은 그대로 유지됩니다.")
    return payload


def _validated_refinements(payload: dict[str, Any], segments: tuple[_Segment, ...]) -> tuple[dict[str, str], list[str]]:
    response_segments = payload.get("segments")
    if not isinstance(response_segments, list):
        return {}, ["AI가 다듬은 대화의 구간 목록을 반환하지 않아 원본 전사문을 유지했습니다."]
    expected = {segment.id for segment in segments}
    valid: dict[str, str] = {}
    invalid_count = 0
    for item in response_segments:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        segment_id = item.get("segment_id")
        text = item.get("refined_text")
        if (
            not isinstance(segment_id, str)
            or segment_id not in expected
            or segment_id in valid
            or not isinstance(text, str)
            or not text.strip()
            or len(text.strip()) > 20_000
        ):
            invalid_count += 1
            continue
        valid[segment_id] = text.strip()
    missing = len(expected - set(valid))
    warnings: list[str] = []
    if invalid_count or missing:
        warnings.append(
            f"AI 응답에서 확인할 수 없는 구간 {invalid_count + missing}개는 원본 전사문으로 유지했습니다."
        )
    return valid, warnings


def _validated_summary(payload: dict[str, Any]) -> ConversationSummary:
    raw = payload.get("summary")
    if not isinstance(raw, dict):
        raise RefinementError("AI 응답에 요약 정보가 없습니다. 원본 전사문은 그대로 유지됩니다.")
    overview = raw.get("overview", "")
    if not isinstance(overview, str) or len(overview.strip()) > 8_000:
        raise RefinementError("AI 요약의 개요 형식이 올바르지 않습니다. 원본 전사문은 그대로 유지됩니다.")

    def values(name: str) -> tuple[str, ...]:
        value = raw.get(name, [])
        if not isinstance(value, list) or len(value) > 100:
            raise RefinementError("AI 요약의 목록 형식이 올바르지 않습니다. 원본 전사문은 그대로 유지됩니다.")
        cleaned = tuple(item.strip() for item in value if isinstance(item, str) and item.strip() and len(item.strip()) <= 4_000)
        if len(cleaned) != len(value):
            raise RefinementError("AI 요약에 올바르지 않은 항목이 있습니다. 원본 전사문은 그대로 유지됩니다.")
        return cleaned

    return ConversationSummary(
        overview=overview.strip(),
        key_points=values("key_points"),
        decisions=values("decisions"),
        action_items=values("action_items"),
        open_questions=values("open_questions"),
    )
