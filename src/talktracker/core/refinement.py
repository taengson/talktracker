"""Transcript refinement through an OpenAI-compatible LLM.

The original transcription always stays immutable.  Production refinement
uses Agent Chat's attachment policy: write the transcript as a temporary
document, select the whole document or relevant 16 KiB units, then show the
model's free-form Markdown response as a separate result document.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from .attachment_content import (
    AttachmentChunk,
    MAX_ATTACHMENT_CONTENT_BYTES,
    MAX_ATTACHMENT_FILE_BYTES,
    create_attachment_chunks,
    select_attachment_content,
)
from .llm_client import LocalLLMError, LocalLLMResponseLengthError, OpenAICompatibleLocalClient

if TYPE_CHECKING:
    from .analysis import AnalysisResult, Utterance


PROMPT_VERSION = "2026-10-27.1"
ProgressCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]
LengthRetrySuccessCallback = Callable[[int], None]
ReasoningEffortFallbackCallback = Callable[[], None]

# A user chooses an effort for each result-making request. ``default`` is
# deliberately represented as ``None`` at the API boundary so an
# OpenAI-compatible server can retain its own configured default.
REASONING_EFFORT_DEFAULT = "default"
REASONING_EFFORT_CHOICES = (REASONING_EFFORT_DEFAULT, "medium", "low", "none")

# Keep each request small enough that a local model does not spend most of its
# context window reasoning about one very long transcript.  We prefer utterance
# boundaries, but a diarizer can group a long reading into one utterance.  Such
# an utterance is split internally at sentence/word boundaries without changing
# the original utterance, its timing, or its speaker in the rendered result.
_TRANSCRIPT_CHUNK_PLANS = ((20, 2_400), (15, 1_800), (10, 1_200), (5, 600), (3, 360), (1, 180))
_MAX_ATOMIC_SEGMENT_CHARACTERS = 160
_MAX_SUMMARY_GROUP_CHARACTERS = 4_000
_MAX_CHUNK_RESPONSE_TOKENS = 4_096


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
    reasoning_effort: str = REASONING_EFFORT_DEFAULT

    def normalized(self) -> "AIRefinementRequest":
        self.selection.validate()
        if not isinstance(self.reasoning_effort, str):
            raise RefinementError("추론 강도는 기본, medium, low, none 중에서 선택해 주세요.")
        reasoning_effort = self.reasoning_effort.strip().lower()
        if reasoning_effort not in REASONING_EFFORT_CHOICES:
            raise RefinementError("추론 강도는 기본, medium, low, none 중에서 선택해 주세요.")
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
            reasoning_effort=reasoning_effort,
        )

    @property
    def requested_reasoning_effort(self) -> str | None:
        """Return the optional field sent to the local model server."""

        return None if self.reasoning_effort == REASONING_EFFORT_DEFAULT else self.reasoning_effort


@dataclass(frozen=True)
class ConversationSummary:
    overview: str
    key_points: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    action_items: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()


@dataclass(frozen=True)
class AIRefinementResult:
    """Non-destructive output of one transcript LLM request."""

    selection: ResultSelection
    refined_text_by_segment_id: dict[str, str] = field(default_factory=dict)
    refined_document: str = ""
    speaker_display_names: dict[str, str] = field(default_factory=dict)
    summary: ConversationSummary | None = None
    summary_document: str = ""
    attachment_selection_summary: str = ""
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
    refined_document = latest.refined_document if latest.selection.refined else previous.refined_document
    summary = latest.summary if latest.selection.summary else previous.summary
    summary_document = latest.summary_document if latest.selection.summary else previous.summary_document
    attachment_selection_summary = latest.attachment_selection_summary or previous.attachment_selection_summary
    warnings = tuple(dict.fromkeys((*previous.warnings, *latest.warnings)))
    return AIRefinementResult(
        selection=ResultSelection(refined=refined_requested, summary=summary_requested),
        refined_text_by_segment_id=refined_text,
        refined_document=refined_document,
        speaker_display_names={**previous.speaker_display_names, **latest.speaker_display_names},
        summary=summary,
        summary_document=summary_document,
        attachment_selection_summary=attachment_selection_summary,
        warnings=warnings,
        prompt_version=latest.prompt_version,
        model=latest.model or previous.model,
    )


@dataclass(frozen=True)
class _Segment:
    id: str
    source_id: str
    utterance: "Utterance"
    text: str


class TranscriptRefiner:
    """Refine an immutable transcript without ever overwriting its source."""

    def __init__(self, client: OpenAICompatibleLocalClient | None, *, document_first: bool = True) -> None:
        self.client = client
        self.document_first = document_first

    def refine(
        self,
        analysis: "AnalysisResult",
        request: AIRefinementRequest,
        *,
        model: str,
        prior_refined_document: str = "",
        response_token_limit: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_effort_fallback_levels: tuple[str, ...] = (),
        on_length_retry_success: LengthRetrySuccessCallback | None = None,
        on_reasoning_effort_fallback: ReasoningEffortFallbackCallback | None = None,
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

        original_document = render_original_conversation(analysis)
        if self.document_first or (normalized.selection.summary and prior_refined_document.strip()):
            # Keep the generated transcript in a real, short-lived Markdown
            # file while it is treated as a chat attachment.  The context
            # manager removes both the file and its private directory after a
            # normal result, a model error, or cancellation.
            with _temporary_transcript_document(original_document) as document_path:
                if document_path.stat().st_size > MAX_ATTACHMENT_FILE_BYTES:
                    raise RefinementError(
                        "전사문이 Agent Chat의 첨부 파일 한도(5MiB)를 넘었습니다. "
                        "원본 전사문은 보존되어 있으니 파일을 나눈 뒤 다시 시도해 주세요."
                    )
                attachment_source = document_path.read_text(encoding="utf-8")
                return self._refine_document_first(
                    normalized,
                    attachment_name=_transcript_attachment_name(analysis.source_name),
                    attachment_source=attachment_source,
                    attachment_chunks=create_attachment_chunks(attachment_source),
                    prior_refined_document=prior_refined_document.strip(),
                    model=model,
                    reasoning_effort=normalized.requested_reasoning_effort,
                    progress=progress,
                    should_cancel=should_cancel,
                )

        segments = _segments_for_utterances(analysis.utterances)
        remembered_limit = min(_MAX_CHUNK_RESPONSE_TOKENS, max(0, int(response_token_limit or 0)))
        if progress:
            if remembered_limit:
                progress("이전에 확인한 응답 길이 설정으로 AI 정리를 시작하고 있습니다…")
            elif reasoning_effort_fallback_levels:
                progress("중간 추론 강도로 AI 정리를 시작하고 있습니다. 응답 길이 한도에 도달하면 낮은 강도로 다시 시도합니다…")
            elif reasoning_effort:
                progress("중간 추론 강도로 AI 정리를 시작하고 있습니다…")
            else:
                progress("AI가 전사 내용을 묶음으로 나누어 정리하고 있습니다…")
        successful_retry_limits: list[int] = []
        warnings: list[str] = []
        refined: dict[str, str] = {}
        if normalized.selection.refined:
            refined, refinement_warnings = self._refine_transcript_batches(
                segments,
                normalized,
                model=model,
                remembered_limit=remembered_limit,
                reasoning_effort=reasoning_effort,
                reasoning_effort_fallback_levels=reasoning_effort_fallback_levels,
                successful_retry_limits=successful_retry_limits,
                on_reasoning_effort_fallback=on_reasoning_effort_fallback,
                progress=progress,
                should_cancel=should_cancel,
            )
            warnings.extend(refinement_warnings)
        summary = None
        if normalized.selection.summary:
            summary = self._summarize_transcript_batches(
                segments,
                normalized,
                model=model,
                remembered_limit=remembered_limit,
                reasoning_effort=reasoning_effort,
                reasoning_effort_fallback_levels=reasoning_effort_fallback_levels,
                successful_retry_limits=successful_retry_limits,
                on_reasoning_effort_fallback=on_reasoning_effort_fallback,
                progress=progress,
                should_cancel=should_cancel,
            )
        if progress:
            progress("AI 정리 결과를 검증하고 있습니다…")
        result = AIRefinementResult(
            selection=normalized.selection,
            refined_text_by_segment_id=_merge_refined_fragments(segments, refined),
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

    def _refine_document_first(
        self,
        request: AIRefinementRequest,
        *,
        attachment_name: str,
        attachment_source: str,
        attachment_chunks: tuple[AttachmentChunk, ...],
        prior_refined_document: str,
        model: str,
        reasoning_effort: str | None,
        progress: ProgressCallback | None,
        should_cancel: CancelCallback | None,
    ) -> AIRefinementResult:
        """Create free-form result documents through the Agent Chat flow."""

        if progress:
            source_label = (
                "다듬은 대화"
                if request.selection.summary and not request.selection.refined and prior_refined_document.strip()
                else "전사문"
            )
            progress(f"AI에 {source_label}을 첨부 문서로 전달하고 있습니다…")

        selection_summaries: list[str] = []

        def messages_for(mode: str, source: str, source_name: str) -> list[dict[str, str]]:
            query = self._document_user_request(request, mode=mode)
            chunks = attachment_chunks if source == attachment_source else create_attachment_chunks(source)
            selection = select_attachment_content(
                source,
                chunks,
                query,
                MAX_ATTACHMENT_CONTENT_BYTES,
            )
            if selection.selection_summary:
                selection_summaries.append(selection.selection_summary)
            return self._document_messages(
                source_name,
                selection.content,
                query,
            )

        refined_document = ""
        if request.selection.refined:
            if progress:
                progress(_document_progress_message("AI가 전사문을 다듬은 대화 문서로 만들고 있습니다", reasoning_effort))
            refined_document = self._request_document(
                model,
                messages_for("refine", attachment_source, attachment_name),
                reasoning_effort=reasoning_effort,
                should_cancel=should_cancel,
            )

        summary_document = ""
        if request.selection.summary:
            current_refined_document = refined_document.strip()
            existing_refined_document = prior_refined_document.strip()
            use_refined_source = bool(current_refined_document or existing_refined_document)
            summary_source = current_refined_document or existing_refined_document or attachment_source
            if len(summary_source.encode("utf-8")) > MAX_ATTACHMENT_FILE_BYTES:
                raise RefinementError(
                    "요약에 사용할 다듬은 대화가 첨부 문서 한도(5MiB)를 넘었습니다. "
                    "다듬은 대화의 길이를 줄인 뒤 다시 시도해 주세요."
                )
            summary_name = _refined_attachment_name(attachment_name) if use_refined_source else attachment_name
            if progress:
                source_label = "다듬은 대화" if use_refined_source else "전사문"
                progress(_document_progress_message(f"AI가 {source_label}을 요약 문서로 만들고 있습니다", reasoning_effort))
            summary_document = self._request_document(
                model,
                messages_for("summary", summary_source, summary_name),
                reasoning_effort=reasoning_effort,
                should_cancel=should_cancel,
            )

        if progress:
            progress("AI가 만든 문서를 저장하고 있습니다…")
        return AIRefinementResult(
            selection=request.selection,
            refined_document=refined_document,
            speaker_display_names={
                guide.speaker: guide.display_name for guide in request.speaker_guides if guide.display_name
            },
            summary_document=summary_document,
            attachment_selection_summary=" · ".join(dict.fromkeys(selection_summaries)),
            model=model.strip(),
        )

    @staticmethod
    def _document_user_request(request: AIRefinementRequest, *, mode: str) -> str:
        """Build the text used as Agent Chat's user request for this file."""

        speaker_guides = "\n".join(
            "- {}{}{}".format(
                guide.speaker,
                f" → {guide.display_name}" if guide.display_name else "",
                f" ({guide.guidance})" if guide.guidance else "",
            )
            for guide in request.speaker_guides
        ) or "- 없음"
        if mode == "refine":
            task = (
                "전사문을 읽기 쉬운 한국어 대화 기록으로 다듬으세요. 맞춤법·띄어쓰기·문장부호를 고치되 "
                "발언의 뜻, 화자 순서, 시각 정보는 바꾸지 마세요. 화자 표기와 시각 표기는 가능한 한 유지하세요."
            )
            output = "다듬은 대화 본문만 Markdown으로 반환하세요. 설명, JSON, 코드 블록은 쓰지 마세요."
        else:
            task = "제공된 대화 기록을 사실에 근거해 한국어 회의 요약으로 정리하세요. 내용에 없는 사실은 만들지 마세요."
            output = (
                "Markdown으로 반환하세요. `## 개요`, `## 핵심 내용`, `## 결정 사항`, `## 할 일`, "
                "`## 확인이 필요한 내용` 제목을 사용하고, 근거가 없는 항목은 '제공된 대화에서 확인된 내용이 없습니다.'라고 적으세요."
            )
        return "\n\n".join(
            (
                task,
                output,
                f"[대화 목적·배경]\n{request.background or '제공되지 않음'}",
                f"[화자 안내]\n{speaker_guides}",
            )
        )

    @staticmethod
    def _document_messages(attachment_name: str, attachment_content: str, user_request: str) -> list[dict[str, str]]:
        """Use Agent Chat's attachment wrapper verbatim in one user message."""

        return [
            {
                "role": "user",
                "content": (
                    f"[첨부 파일: {attachment_name}]\n{attachment_content}\n[첨부 파일 끝]"
                    f"\n\n[사용자 요청]\n{user_request}"
                ),
            }
        ]

    def _request_document(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        reasoning_effort: str | None,
        should_cancel: CancelCallback | None,
    ) -> str:
        assert self.client is not None

        response = self.client.create_chat_completion(
            model,
            messages=messages,
            temperature=0,
            # Agent Chat leaves output length to the selected model/server and
            # does not automatically retry with a different inference setup.
            max_tokens=None,
            json_mode=False,
            reasoning_effort=reasoning_effort,
            reasoning_effort_retry_levels=(),
            retry_on_length=False,
            # A person explicitly chose this level. Do not quietly retry at
            # the server default (which may be much slower, e.g. Qwen xhigh)
            # when a server rejects the optional field.
            allow_reasoning_effort_fallback=False,
            should_cancel=should_cancel,
        )
        self._raise_if_cancelled(should_cancel)
        document = response.strip()
        if not document:
            raise RefinementError("AI가 비어 있는 문서를 반환했습니다. 원본 전사문은 그대로 유지됩니다.")
        return document

    @staticmethod
    def _messages(
        segments: tuple[_Segment, ...], request: AIRefinementRequest, *, mode: str
    ) -> list[dict[str, str]]:
        system = (
            "You refine a Korean conversation transcript. Use only the provided transcript and optional context. "
            "Do not invent facts, speakers, times, names, decisions, or action items. "
            "Use a supplied display_name as the speaker label in any generated wording about that speaker. "
            "Speaker guidance is only context and must not be output as a separate field. "
            "Return JSON only, with no markdown fence or explanation. "
        )
        if mode == "refine":
            system += "Keep each segment's meaning. Return every requested segment_id exactly once with a non-empty refined_text."
        else:
            system += (
                "Return a summary object with overview (string), key_points, decisions, action_items, and open_questions "
                "(arrays of strings). Use empty arrays when the transcript has no evidence."
            )
        content = {
            "prompt_version": PROMPT_VERSION,
            "requested_output": {
                "refined_conversation": mode == "refine",
                "summary": mode == "summary",
            },
            "conversation_background": request.background,
            "speaker_guides": [
                {"speaker": guide.speaker, "display_name": guide.display_name, "guidance": guide.guidance}
                for guide in request.speaker_guides
            ],
            "segments": [
                {"segment_id": segment.id, "speaker": segment.utterance.speaker, "text": segment.text}
                for segment in segments
            ],
            "response_shape": {
                "summary": {
                    "overview": "...",
                    "key_points": ["..."],
                    "decisions": ["..."],
                    "action_items": ["..."],
                    "open_questions": ["..."],
                },
            },
        }
        if mode == "refine":
            content["response_shape"] = {"segments": [{"segment_id": "s-0001", "refined_text": "..."}]}
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        ]

    def _refine_transcript_batches(
        self,
        segments: tuple[_Segment, ...],
        request: AIRefinementRequest,
        *,
        model: str,
        remembered_limit: int,
        reasoning_effort: str | None,
        reasoning_effort_fallback_levels: tuple[str, ...],
        successful_retry_limits: list[int],
        on_reasoning_effort_fallback: ReasoningEffortFallbackCallback | None,
        progress: ProgressCallback | None,
        should_cancel: CancelCallback | None,
    ) -> tuple[dict[str, str], list[str]]:
        batches = list(_segment_chunks(segments, _TRANSCRIPT_CHUNK_PLANS[0]))
        plan_index = 0
        batch_index = 0
        refined: dict[str, str] = {}
        warnings: list[str] = []
        while batch_index < len(batches):
            chunk = batches[batch_index]
            # Repartitioning comes before changing the model's generation
            # settings.  It preserves the normal medium-reasoning path and
            # makes the next request genuinely smaller instead of repeating
            # the same oversized work with a different model option.
            repartitioned = _lower_unfinished_transcript_batches(batches, batch_index, plan_index)
            if progress:
                progress(f"다듬은 대화 {batch_index + 1}/{len(batches)} 묶음을 AI로 정리하고 있습니다…")
            try:
                payload = self._request_json(
                    model,
                    self._messages(chunk, request, mode="refine"),
                    response_token_limit=_chunk_response_limit(chunk, remembered_limit),
                    reasoning_effort=reasoning_effort,
                    reasoning_effort_fallback_levels=reasoning_effort_fallback_levels,
                    allow_length_retry=repartitioned is None,
                    successful_retry_limits=successful_retry_limits,
                    on_reasoning_effort_fallback=on_reasoning_effort_fallback,
                    progress=progress,
                    should_cancel=should_cancel,
                )
            except LocalLLMResponseLengthError as error:
                if repartitioned is None:
                    raise _unrecoverable_transcript_batch_error(error) from error
                plan_index, replacement = repartitioned
                batches[batch_index:] = replacement
                if progress:
                    progress(
                        f"응답 길이 한도에 도달했습니다. 완료한 {batch_index}개 묶음은 유지하고, "
                        f"남은 전사를 더 작게 나누어 다시 정리하고 있습니다…"
                    )
                continue
            chunk_refined, chunk_warnings = _validated_refinements(payload, chunk)
            refined.update(chunk_refined)
            warnings.extend(chunk_warnings)
            batch_index += 1
        return refined, warnings

    def _summarize_transcript_batches(
        self,
        segments: tuple[_Segment, ...],
        request: AIRefinementRequest,
        *,
        model: str,
        remembered_limit: int,
        reasoning_effort: str | None,
        reasoning_effort_fallback_levels: tuple[str, ...],
        successful_retry_limits: list[int],
        on_reasoning_effort_fallback: ReasoningEffortFallbackCallback | None,
        progress: ProgressCallback | None,
        should_cancel: CancelCallback | None,
    ) -> ConversationSummary:
        batches = list(_segment_chunks(segments, _TRANSCRIPT_CHUNK_PLANS[0]))
        plan_index = 0
        batch_index = 0
        summaries: list[ConversationSummary] = []
        while batch_index < len(batches):
            chunk = batches[batch_index]
            repartitioned = _lower_unfinished_transcript_batches(batches, batch_index, plan_index)
            if progress:
                progress(f"요약을 위해 전사 {batch_index + 1}/{len(batches)} 묶음을 읽고 있습니다…")
            try:
                payload = self._request_json(
                    model,
                    self._messages(chunk, request, mode="summary"),
                    response_token_limit=_summary_response_limit(chunk, remembered_limit),
                    reasoning_effort=reasoning_effort,
                    reasoning_effort_fallback_levels=reasoning_effort_fallback_levels,
                    allow_length_retry=repartitioned is None,
                    successful_retry_limits=successful_retry_limits,
                    on_reasoning_effort_fallback=on_reasoning_effort_fallback,
                    progress=progress,
                    should_cancel=should_cancel,
                )
            except LocalLLMResponseLengthError as error:
                if repartitioned is None:
                    raise _unrecoverable_transcript_batch_error(error) from error
                plan_index, replacement = repartitioned
                batches[batch_index:] = replacement
                if progress:
                    progress(
                        f"응답 길이 한도에 도달했습니다. 만든 {batch_index}개 부분 요약은 유지하고, "
                        f"남은 전사를 더 작게 나누어 다시 요약하고 있습니다…"
                    )
                continue
            summaries.append(_validated_summary(payload))
            batch_index += 1
        return self._combine_summaries(
            summaries,
            model=model,
            remembered_limit=remembered_limit,
            reasoning_effort=reasoning_effort,
            reasoning_effort_fallback_levels=reasoning_effort_fallback_levels,
            successful_retry_limits=successful_retry_limits,
            on_reasoning_effort_fallback=on_reasoning_effort_fallback,
            progress=progress,
            should_cancel=should_cancel,
        )

    def _combine_summaries(
        self,
        summaries: list[ConversationSummary],
        *,
        model: str,
        remembered_limit: int,
        reasoning_effort: str | None,
        reasoning_effort_fallback_levels: tuple[str, ...],
        successful_retry_limits: list[int],
        on_reasoning_effort_fallback: ReasoningEffortFallbackCallback | None,
        progress: ProgressCallback | None,
        should_cancel: CancelCallback | None,
    ) -> ConversationSummary:
        while len(summaries) > 1:
            groups = _summary_groups(tuple(summaries))
            # A malformed-but-valid model response can contain an unusually
            # verbose partial summary.  Do not loop forever on singleton
            # groups; reduce neighbouring summaries in that exceptional case.
            if len(groups) == len(summaries):
                groups = tuple(tuple(summaries[index : index + 2]) for index in range(0, len(summaries), 2))
            next_summaries: list[ConversationSummary] = []
            for index, group in enumerate(groups, start=1):
                if progress:
                    progress(f"부분 요약 {index}/{len(groups)} 묶음을 합치고 있습니다…")
                payload = self._request_json(
                    model,
                    self._summary_reduction_messages(group),
                    response_token_limit=_summary_reduction_limit(group, remembered_limit),
                    reasoning_effort=reasoning_effort,
                    reasoning_effort_fallback_levels=reasoning_effort_fallback_levels,
                    # A final map/reduce group has no transcript boundary to
                    # repartition, so use the bounded model-setting retries.
                    allow_length_retry=True,
                    successful_retry_limits=successful_retry_limits,
                    on_reasoning_effort_fallback=on_reasoning_effort_fallback,
                    progress=progress,
                    should_cancel=should_cancel,
                )
                next_summaries.append(_validated_summary(payload))
            summaries = next_summaries
        return summaries[0]

    def _request_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        response_token_limit: int,
        reasoning_effort: str | None,
        reasoning_effort_fallback_levels: tuple[str, ...],
        allow_length_retry: bool,
        successful_retry_limits: list[int],
        on_reasoning_effort_fallback: ReasoningEffortFallbackCallback | None,
        progress: ProgressCallback | None,
        should_cancel: CancelCallback | None,
    ) -> dict[str, Any]:
        assert self.client is not None

        def effort_fallback() -> None:
            if on_reasoning_effort_fallback:
                on_reasoning_effort_fallback()
            if progress:
                progress("서버가 추론 강도 설정을 지원하지 않아 기본 설정으로 계속 정리하고 있습니다…")

        def effort_lowered(effort: str) -> None:
            if progress:
                label = "중간" if effort == "medium" else "낮은"
                progress(f"응답 길이 한도에 도달해 추론 강도를 {label} 수준으로 낮추고 같은 묶음을 다시 정리하고 있습니다…")

        response = self.client.create_chat_completion(
            model,
            messages=messages,
            temperature=0,
            max_tokens=response_token_limit,
            json_mode=True,
            reasoning_effort=reasoning_effort,
            reasoning_effort_retry_levels=(reasoning_effort_fallback_levels if allow_length_retry else ()),
            retry_on_length=allow_length_retry,
            on_length_retry=(
                lambda _current_limit, _retry_limit: progress(
                    "AI 응답 길이 한도에 도달해 이 묶음만 더 큰 설정으로 다시 정리하고 있습니다…"
                )
                if progress
                else None
            ),
            on_length_retry_success=successful_retry_limits.append,
            on_reasoning_effort_fallback=effort_fallback if reasoning_effort else None,
            on_reasoning_effort_lowered=effort_lowered if reasoning_effort_fallback_levels else None,
            should_cancel=should_cancel,
        )
        self._raise_if_cancelled(should_cancel)
        return _parse_response_json(response)

    @staticmethod
    def _summary_reduction_messages(summaries: tuple[ConversationSummary, ...]) -> list[dict[str, str]]:
        system = (
            "You combine partial Korean conversation summaries. Use only the supplied summaries. "
            "Do not invent facts, decisions, action items, names, or open questions. "
            "Return JSON only, with no markdown fence or explanation. Return a summary object with overview (string), "
            "key_points, decisions, action_items, and open_questions (arrays of strings). Use empty arrays when there is no evidence."
        )
        content = {
            "prompt_version": PROMPT_VERSION,
            "partial_summaries": [_summary_as_payload(summary) for summary in summaries],
            "response_shape": {
                "summary": {
                    "overview": "...",
                    "key_points": ["..."],
                    "decisions": ["..."],
                    "action_items": ["..."],
                    "open_questions": ["..."],
                }
            },
        }
        return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]

    @staticmethod
    def _raise_if_cancelled(should_cancel: CancelCallback | None) -> None:
        if should_cancel and should_cancel():
            raise RefinementCancelled("AI 정리 요청을 취소했습니다.")


@contextmanager
def _temporary_transcript_document(contents: str) -> Iterator[Path]:
    """Materialize one transcript attachment and always remove it afterwards."""

    with TemporaryDirectory(prefix="talktracker-transcript-") as directory:
        path = Path(directory) / "transcript.md"
        path.write_text(contents, encoding="utf-8")
        yield path


def _transcript_attachment_name(source_name: str) -> str:
    """Give the temporary Markdown file the stable name users would expect."""

    stem = Path(source_name).stem.strip() or "전사문"
    return f"{stem}-전사문.md"


def _refined_attachment_name(transcript_attachment_name: str) -> str:
    """Label a refined transcript attachment without exposing a file path."""

    stem = Path(transcript_attachment_name).stem
    if stem.endswith("-전사문"):
        stem = stem[: -len("-전사문")]
    return f"{stem}-다듬은대화.md"


def _document_progress_message(message: str, reasoning_effort: str | None) -> str:
    """Make the selected request setting visible while a model is working."""

    suffix = f" · 추론 강도 {reasoning_effort}" if reasoning_effort else " · 추론 강도 기본(서버 설정)"
    return f"{message}{suffix}…"


def _segments_for_utterances(utterances: list["Utterance"]) -> tuple[_Segment, ...]:
    """Create request-only segments while retaining each source utterance id.

    Speaker diarization sometimes produces one very long utterance for a
    reading or monologue.  Treating that utterance as indivisible defeats all
    later batch plans.  These request segments are deliberately internal: the
    final rendered transcript still has exactly the original utterance timing
    and speaker rows.
    """

    segments: list[_Segment] = []
    for index, utterance in enumerate(utterances, start=1):
        source_id = f"s-{index:04d}"
        fragments = _split_utterance_for_request(utterance.text)
        multiple_fragments = len(fragments) > 1
        for fragment_index, text in enumerate(fragments, start=1):
            segment_id = f"{source_id}-p{fragment_index:02d}" if multiple_fragments else source_id
            segments.append(_Segment(segment_id, source_id, utterance, text))
    return tuple(segments)


def _split_utterance_for_request(text: str) -> tuple[str, ...]:
    """Prefer sentence boundaries and use whitespace only as a last resort."""

    value = text.strip()
    if len(value) <= _MAX_ATOMIC_SEGMENT_CHARACTERS:
        return (value,)

    sentences: list[str] = []
    current: list[str] = []
    for character in value:
        current.append(character)
        if character in ".?!。！？\n":
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    trailing = "".join(current).strip()
    if trailing:
        sentences.append(trailing)
    if not sentences:
        sentences = [value]

    bounded_sentences: list[str] = []
    for sentence in sentences:
        remaining = sentence
        while len(remaining) > _MAX_ATOMIC_SEGMENT_CHARACTERS:
            boundary = remaining.rfind(" ", 0, _MAX_ATOMIC_SEGMENT_CHARACTERS + 1)
            if boundary <= 0:
                boundary = _MAX_ATOMIC_SEGMENT_CHARACTERS
            bounded_sentences.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].strip()
        if remaining:
            bounded_sentences.append(remaining)

    fragments: list[str] = []
    current_fragment = ""
    for sentence in bounded_sentences:
        candidate = f"{current_fragment} {sentence}".strip() if current_fragment else sentence
        if current_fragment and len(candidate) > _MAX_ATOMIC_SEGMENT_CHARACTERS:
            fragments.append(current_fragment)
            current_fragment = sentence
        else:
            current_fragment = candidate
    if current_fragment:
        fragments.append(current_fragment)
    return tuple(fragments) or (value,)


def _merge_refined_fragments(segments: tuple[_Segment, ...], replacements: dict[str, str]) -> dict[str, str]:
    """Fold request-only fragments back into their immutable source utterance."""

    by_source: dict[str, list[_Segment]] = {}
    for segment in segments:
        by_source.setdefault(segment.source_id, []).append(segment)

    merged: dict[str, str] = {}
    for source_id, fragments in by_source.items():
        if not any(fragment.id in replacements for fragment in fragments):
            continue
        merged[source_id] = " ".join(replacements.get(fragment.id, fragment.text) for fragment in fragments).strip()
    return merged


def _segment_chunks(
    segments: tuple[_Segment, ...], plan: tuple[int, int]
) -> tuple[tuple[_Segment, ...], ...]:
    """Split at utterance boundaries while keeping every segment exactly once."""

    maximum_segments, maximum_characters = plan
    chunks: list[tuple[_Segment, ...]] = []
    current: list[_Segment] = []
    current_size = 0
    for segment in segments:
        segment_size = max(1, len(segment.text))
        if current and (len(current) >= maximum_segments or current_size + segment_size > maximum_characters):
            chunks.append(tuple(current))
            current, current_size = [], 0
        current.append(segment)
        current_size += segment_size
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def _lower_unfinished_transcript_batches(
    batches: list[tuple[_Segment, ...]], batch_index: int, plan_index: int
) -> tuple[int, list[tuple[_Segment, ...]]] | None:
    """Repartition only the failed batch and its unfinished successors.

    This mirrors the review-batch recovery approach: completed batches remain
    untouched, while a failure makes all remaining work smaller in one
    direction.  A batch that contains one indivisible utterance cannot be
    safely cut in the middle, so it is reported instead of looping forever.
    """

    failed_batch = batches[batch_index]
    remaining_segments = tuple(segment for batch in batches[batch_index:] for segment in batch)
    for candidate_index in range(plan_index + 1, len(_TRANSCRIPT_CHUNK_PLANS)):
        candidate_plan = _TRANSCRIPT_CHUNK_PLANS[candidate_index]
        if len(_segment_chunks(failed_batch, candidate_plan)) <= 1:
            continue
        return candidate_index, list(_segment_chunks(remaining_segments, candidate_plan))
    return None


def _unrecoverable_transcript_batch_error(error: LocalLLMResponseLengthError) -> RefinementError:
    return RefinementError(
        "AI 응답 길이 한도에 계속 도달했습니다. 이 전사 묶음은 문장·공백 경계를 더 작게 나눌 수 없습니다. "
        "로컬 AI 서버의 응답 길이 설정을 확인해 주세요."
    )


def _chunk_response_limit(chunks: tuple[_Segment, ...], remembered_limit: int = 0) -> int:
    characters = sum(len(segment.text) for segment in chunks)
    estimated = max(1_024, min(2_048, 512 + characters))
    return min(_MAX_CHUNK_RESPONSE_TOKENS, max(estimated, remembered_limit))


def _summary_response_limit(chunks: tuple[_Segment, ...], remembered_limit: int = 0) -> int:
    characters = sum(len(segment.text) for segment in chunks)
    estimated = max(1_024, min(1_536, 384 + characters // 2))
    return min(_MAX_CHUNK_RESPONSE_TOKENS, max(estimated, remembered_limit))


def _summary_reduction_limit(summaries: tuple[ConversationSummary, ...], remembered_limit: int = 0) -> int:
    characters = sum(len(json.dumps(_summary_as_payload(summary), ensure_ascii=False)) for summary in summaries)
    estimated = max(512, min(1_536, 384 + characters // 2))
    return min(_MAX_CHUNK_RESPONSE_TOKENS, max(estimated, remembered_limit))


def _summary_as_payload(summary: ConversationSummary) -> dict[str, object]:
    return {
        "overview": summary.overview,
        "key_points": list(summary.key_points),
        "decisions": list(summary.decisions),
        "action_items": list(summary.action_items),
        "open_questions": list(summary.open_questions),
    }


def _summary_groups(summaries: tuple[ConversationSummary, ...]) -> tuple[tuple[ConversationSummary, ...], ...]:
    """Group short partial summaries for recursive map/reduce summarisation."""

    groups: list[tuple[ConversationSummary, ...]] = []
    current: list[ConversationSummary] = []
    current_size = 0
    for summary in summaries:
        summary_size = max(1, len(json.dumps(_summary_as_payload(summary), ensure_ascii=False)))
        if current and current_size + summary_size > _MAX_SUMMARY_GROUP_CHARACTERS:
            groups.append(tuple(current))
            current, current_size = [], 0
        current.append(summary)
        current_size += summary_size
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def render_original_conversation(analysis: "AnalysisResult") -> str:
    return _render_conversation(analysis.utterances, {})


def render_refined_conversation(analysis: "AnalysisResult", result: AIRefinementResult) -> str:
    if result.refined_document.strip():
        return result.refined_document.strip()
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


def render_result_summary(result: AIRefinementResult) -> str:
    """Render either a document-first summary or the large-input fallback."""

    if result.summary_document.strip():
        return result.summary_document.strip()
    return render_summary(result.summary) if result.summary else ""


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
        ("요약", render_result_summary(result) if result.selection.summary else ""),
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
    if result.selection.summary and render_result_summary(result):
        documents["summary"] = "\n".join(
            ["# 요약", "", *metadata, "", "## 요약 내용", "", render_result_summary(result), *warnings]
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
        raise RefinementError(
            "AI가 다듬은 대화의 구간 목록을 반환하지 않았습니다. "
            "결과를 적용하지 않았으니 다시 시도해 주세요."
        )
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
    if not valid:
        raise RefinementError(
            "AI가 적용할 수 있는 다듬은 대화 구간을 반환하지 않았습니다. "
            "결과를 적용하지 않았으니 다시 시도해 주세요."
        )
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
