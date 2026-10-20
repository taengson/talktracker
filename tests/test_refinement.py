from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace

from talktracker.core.analysis import AnalysisResult, Utterance
from talktracker.core.attachment_content import MAX_ATTACHMENT_FILE_BYTES
from talktracker.core.llm_client import LocalLLMResponseLengthError
from talktracker.core.llm_config import LocalLLMReasoningEffort, LocalLLMSettings
from talktracker.core.refinement import (
    AIRefinementRequest,
    AIRefinementResult,
    ConversationSummary,
    RefinementError,
    ResultSelection,
    SpeakerGuide,
    TranscriptRefiner,
    merge_ai_refinement_results,
    render_original_conversation,
    render_refined_conversation,
    render_result_documents,
    render_result_summary,
    render_summary,
    _temporary_transcript_document,
)
from talktracker.core.workflow import AIRefinementManager


class _FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.request: dict[str, object] | None = None

    def create_chat_completion(
        self,
        model: str,
        *,
        messages,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
        reasoning_effort_retry_levels: tuple[str, ...] = (),
        retry_on_length: bool = False,
        on_length_retry=None,
        on_length_retry_success=None,
        on_reasoning_effort_fallback=None,
        on_reasoning_effort_lowered=None,
        should_cancel=None,
    ) -> str:
        self.request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
            "reasoning_effort": reasoning_effort,
            "reasoning_effort_retry_levels": reasoning_effort_retry_levels,
            "retry_on_length": retry_on_length,
            "on_length_retry_success": on_length_retry_success,
        }
        if should_cancel and should_cancel():
            raise AssertionError("cancelled requests should not call the client")
        return json.dumps(self.response, ensure_ascii=False)


class _ReasoningConnectionManager:
    """Minimal verified connection used to exercise the worker's request policy."""

    def __init__(self, client: _FakeClient, *, auto_lower_on_length: bool) -> None:
        self.client = client
        self.settings = LocalLLMSettings(
            base_url="http://localhost:11434/v1",
            model="Qwen-test",
            reasoning_efforts=(
                LocalLLMReasoningEffort(
                    "http://localhost:11434/v1",
                    "Qwen-test",
                    ("medium", "low"),
                    auto_lower_on_length,
                ),
            ),
        )

    def snapshot(self):
        return SimpleNamespace(settings=self.settings)

    def active_client(self) -> _FakeClient:
        return self.client

    def remember_response_token_limit(self, *args, **kwargs) -> None:
        return None

    def disable_reasoning_effort(self, *args, **kwargs) -> None:
        return None


class _ChunkingFakeClient:
    """Returns a schema-correct result for whichever bounded request it sees."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create_chat_completion(self, model: str, *, messages, **kwargs) -> str:
        payload = json.loads(messages[1]["content"])
        self.requests.append({"model": model, "payload": payload, **kwargs})
        if payload.get("requested_output", {}).get("refined_conversation"):
            return json.dumps(
                {
                    "segments": [
                        {"segment_id": segment["segment_id"], "refined_text": f"다듬은 {segment['text']}"}
                        for segment in payload["segments"]
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "summary": {
                    "overview": "전사 묶음을 요약했습니다.",
                    "key_points": ["핵심 내용을 확인했습니다."],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                }
            },
            ensure_ascii=False,
        )


class _DocumentFakeClient:
    """Returns ordinary Markdown, like a user-facing chat model."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create_chat_completion(self, model: str, *, messages, **kwargs) -> str:
        self.requests.append({"model": model, "messages": messages, **kwargs})
        if "회의 요약" in messages[0]["content"]:
            return "## 개요\n\n초안 준비와 일정 확인을 논의했습니다.\n\n## 핵심 내용\n\n- 다음 주까지 초안을 준비합니다."
        return "(손태영)  00:01 – 00:05\n네, 다음 주까지 초안을 준비할게요.\n\n(화자 2)  00:05 – 00:09\n좋습니다. 일정을 확인해 주세요."


class _RepartitioningFakeClient:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.retry_on_length: list[bool] = []
        self.failed_once = False

    def create_chat_completion(self, model: str, *, messages, **kwargs) -> str:
        payload = json.loads(messages[1]["content"])
        segments = payload.get("segments", [])
        self.batch_sizes.append(len(segments))
        self.retry_on_length.append(bool(kwargs.get("retry_on_length")))
        if not self.failed_once:
            self.failed_once = True
            raise LocalLLMResponseLengthError("AI 응답 길이 한도에 도달했습니다.")
        return json.dumps(
            {
                "segments": [
                    {"segment_id": segment["segment_id"], "refined_text": f"다듬은 {segment['text']}"}
                    for segment in segments
                ]
            },
            ensure_ascii=False,
        )


class _RepartitionAfterSuccessFakeClient(_RepartitioningFakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def create_chat_completion(self, model: str, *, messages, **kwargs) -> str:
        self.calls += 1
        payload = json.loads(messages[1]["content"])
        segments = payload.get("segments", [])
        self.batch_sizes.append(len(segments))
        if self.calls == 2:
            raise LocalLLMResponseLengthError("AI 응답 길이 한도에 도달했습니다.")
        return json.dumps(
            {
                "segments": [
                    {"segment_id": segment["segment_id"], "refined_text": f"다듬은 {segment['text']}"}
                    for segment in segments
                ]
            },
            ensure_ascii=False,
        )


class _SummaryRepartitioningFakeClient:
    def __init__(self) -> None:
        self.transcript_batch_sizes: list[int] = []
        self.failed_once = False

    def create_chat_completion(self, model: str, *, messages, **kwargs) -> str:
        payload = json.loads(messages[1]["content"])
        if "segments" in payload:
            self.transcript_batch_sizes.append(len(payload["segments"]))
            if not self.failed_once:
                self.failed_once = True
                raise LocalLLMResponseLengthError("AI 응답 길이 한도에 도달했습니다.")
        return json.dumps(
            {
                "summary": {
                    "overview": "발화 묶음을 요약했습니다.",
                    "key_points": [],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                }
            },
            ensure_ascii=False,
        )


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        source_name="meeting.m4a",
        duration=16.2,
        speaker_segments=[],
        utterances=[
            Utterance(start=1.2, end=4.8, speaker="화자 1", text="네 다음주 까지 초안 준비 할게요"),
            Utterance(start=5.1, end=8.7, speaker="화자 2", text="좋습니다 일정 확인 해주세요"),
        ],
        whisper_profile="빠른 전사",
    )


class TranscriptRefinerTests(unittest.TestCase):
    def test_normal_transcript_uses_a_plain_document_request_and_keeps_markdown_output(self) -> None:
        client = _DocumentFakeClient()
        request = AIRefinementRequest(
            selection=ResultSelection(refined=True, summary=True),
            background="주간 프로젝트 회의",
            speaker_guides=(SpeakerGuide("화자 1", display_name="손태영"),),
        )

        result = TranscriptRefiner(client).refine(_analysis(), request, model="local-model")

        self.assertEqual(len(client.requests), 2)
        self.assertTrue(all(request["json_mode"] is False for request in client.requests))
        self.assertTrue(all(request["max_tokens"] is None for request in client.requests))
        self.assertTrue(all(request["retry_on_length"] is False for request in client.requests))
        self.assertTrue(all(request["reasoning_effort_retry_levels"] == () for request in client.requests))
        self.assertTrue(all(len(request["messages"]) == 1 for request in client.requests))
        self.assertTrue(all("[첨부 파일:" in request["messages"][0]["content"] for request in client.requests))
        self.assertTrue(all("[첨부 파일 끝]" in request["messages"][0]["content"] for request in client.requests))
        self.assertTrue(all("[사용자 요청]" in request["messages"][0]["content"] for request in client.requests))
        self.assertTrue(all("네 다음주 까지 초안 준비 할게요" in request["messages"][0]["content"] for request in client.requests))
        self.assertEqual(result.refined_text_by_segment_id, {})
        self.assertIn("다음 주까지 초안을 준비할게요.", render_refined_conversation(_analysis(), result))
        self.assertIn("초안 준비와 일정 확인", render_result_summary(result))
        documents = render_result_documents(_analysis(), result)
        self.assertIn("## 개요", documents["summary"])
        self.assertIn("(손태영)", documents["refined"])
        self.assertIn("네 다음주 까지 초안 준비 할게요", render_original_conversation(_analysis()))

    def test_temporary_transcript_document_is_removed_after_success_and_error(self) -> None:
        with _temporary_transcript_document("임시 전사") as path:
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_text(encoding="utf-8"), "임시 전사")
        self.assertFalse(path.exists())

        failed_path = None
        with self.assertRaisesRegex(RuntimeError, "중단"):
            with _temporary_transcript_document("중단된 전사") as path:
                failed_path = path
                raise RuntimeError("중단")
        assert failed_path is not None
        self.assertFalse(failed_path.exists())

    def test_agent_chat_file_limit_is_checked_before_a_model_request(self) -> None:
        client = _DocumentFakeClient()
        analysis = AnalysisResult(
            source_name="too-large.m4a",
            duration=1,
            speaker_segments=[],
            utterances=[Utterance(0, 1, "화자 1", "가" * (MAX_ATTACHMENT_FILE_BYTES // 3))],
            whisper_profile="빠른 전사",
        )

        with self.assertRaisesRegex(RefinementError, "5MiB"):
            TranscriptRefiner(client).refine(
                analysis,
                AIRefinementRequest(selection=ResultSelection(summary=True)),
                model="local-model",
            )
        self.assertEqual(client.requests, [])

    def test_18_kib_korean_transcript_uses_the_document_request_path(self) -> None:
        client = _DocumentFakeClient()
        analysis = AnalysisResult(
            source_name="18kb.m4a",
            duration=60,
            speaker_segments=[],
            utterances=[Utterance(0, 60, "화자 1", "가" * 6_000)],
            whisper_profile="빠른 전사",
        )

        result = TranscriptRefiner(client).refine(
            analysis,
            AIRefinementRequest(selection=ResultSelection(summary=True)),
            model="local-model",
        )

        self.assertEqual(len(client.requests), 1)
        self.assertFalse(client.requests[0]["json_mode"])
        self.assertIn("가" * 6_000, client.requests[0]["messages"][0]["content"])
        self.assertIn("초안 준비와 일정 확인", render_result_summary(result))

    def test_refinement_keeps_original_speaker_time_and_order(self) -> None:
        client = _FakeClient(
            {
                "segments": [
                    {"segment_id": "s-0001", "refined_text": "네, 다음 주까지 초안을 준비할게요."},
                    {"segment_id": "s-0002", "refined_text": "좋습니다. 일정을 확인해 주세요."},
                ],
                "summary": {
                    "overview": "초안 준비 일정을 확인했다.",
                    "key_points": ["다음 주까지 초안을 준비한다."],
                    "decisions": [],
                    "action_items": ["화자 1이 초안을 준비한다."],
                    "open_questions": [],
                },
            }
        )
        request = AIRefinementRequest(
            selection=ResultSelection(refined=True, summary=True),
            background="주간 프로젝트 회의",
            speaker_guides=(SpeakerGuide("화자 1", guidance="프로젝트 매니저", display_name="손태영"),),
        )

        result = TranscriptRefiner(client, document_first=False).refine(_analysis(), request, model="local-model")

        original = render_original_conversation(_analysis())
        refined = render_refined_conversation(_analysis(), result)
        self.assertIn("(화자 1)  00:01 – 00:05", original)
        self.assertIn("(손태영)  00:01 – 00:05", refined)
        self.assertIn("네, 다음 주까지 초안을 준비할게요.", refined)
        self.assertNotIn("프로젝트 매니저", refined)
        self.assertIsNotNone(result.summary)
        assert result.summary is not None
        self.assertIn("초안 준비 일정을 확인했다.", render_summary(result.summary))
        self.assertEqual(client.request["model"], "local-model")
        self.assertTrue(client.request["json_mode"])
        # A two-segment transcript can still be repartitioned into individual
        # requests, so it must do that before adjusting model settings.
        self.assertFalse(client.request["retry_on_length"])
        payload = json.loads(client.request["messages"][1]["content"])
        self.assertEqual(
            payload["speaker_guides"],
            [{"speaker": "화자 1", "display_name": "손태영", "guidance": "프로젝트 매니저"}],
        )
        documents = render_result_documents(_analysis(), result)
        self.assertEqual(set(documents), {"original", "refined", "summary"})
        self.assertIn("# 다듬은 대화", documents["refined"])
        self.assertNotIn("# 원본 전사 메시지", documents["refined"])
        self.assertIn("(손태영)", documents["refined"])

    def test_refinement_uses_the_successfully_learned_limit_for_the_same_connection(self) -> None:
        client = _FakeClient(
            {
                "segments": [
                    {"segment_id": "s-0001", "refined_text": "첫 문장을 다듬었습니다."},
                    {"segment_id": "s-0002", "refined_text": "둘째 문장을 다듬었습니다."},
                ]
            }
        )
        progress: list[str] = []

        TranscriptRefiner(client, document_first=False).refine(
            _analysis(),
            AIRefinementRequest(selection=ResultSelection(refined=True)),
            model="local-model",
            response_token_limit=4_096,
            progress=progress.append,
        )

        self.assertEqual(client.request["max_tokens"], 4_096)
        self.assertIn("이전에 확인한 응답 길이 설정", progress[0])

    def test_automatic_reasoning_starts_at_medium_effort(self) -> None:
        client = _FakeClient(
            {
                "segments": [
                    {"segment_id": "s-0001", "refined_text": "첫 문장을 다듬었습니다."},
                    {"segment_id": "s-0002", "refined_text": "둘째 문장을 다듬었습니다."},
                ]
            }
        )
        progress: list[str] = []

        TranscriptRefiner(client, document_first=False).refine(
            _analysis(),
            AIRefinementRequest(selection=ResultSelection(refined=True)),
            model="Qwen3.8-27B",
            reasoning_effort="medium",
            reasoning_effort_fallback_levels=("low",),
            progress=progress.append,
        )

        self.assertEqual(client.request["reasoning_effort"], "medium")
        # The automatic fallback is deliberately held back until a request
        # cannot be repartitioned any further.
        self.assertEqual(client.request["reasoning_effort_retry_levels"], ())
        self.assertIn("중간 추론 강도", progress[0])

    def test_attachment_policy_uses_auto_reasoning_even_for_a_verified_reasoning_model(self) -> None:
        client = _FakeClient(
            {
                "segments": [
                    {"segment_id": "s-0001", "refined_text": "첫 문장을 다듬었습니다."},
                    {"segment_id": "s-0002", "refined_text": "둘째 문장을 다듬었습니다."},
                ]
            }
        )
        manager = AIRefinementManager(_ReasoningConnectionManager(client, auto_lower_on_length=False))
        manager.start(_analysis(), AIRefinementRequest(selection=ResultSelection(refined=True)))
        deadline = time.monotonic() + 1
        while manager.snapshot().active and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(manager.snapshot().state, "completed")
        assert client.request is not None
        self.assertIsNone(client.request["reasoning_effort"])
        self.assertEqual(client.request["reasoning_effort_retry_levels"], ())

    def test_long_transcript_is_chunked_and_its_results_are_merged(self) -> None:
        client = _ChunkingFakeClient()
        analysis = AnalysisResult(
            source_name="long.m4a",
            duration=30,
            speaker_segments=[],
            utterances=[
                Utterance(float(index), float(index + 1), "화자 1", f"발화 {index} " + ("가" * 800))
                for index in range(4)
            ],
            whisper_profile="빠른 전사",
        )
        progress: list[str] = []

        result = TranscriptRefiner(client, document_first=False).refine(
            analysis,
            AIRefinementRequest(selection=ResultSelection(refined=True, summary=True)),
            model="local-model",
            progress=progress.append,
        )

        refined_requests = [
            item for item in client.requests if item["payload"].get("requested_output", {}).get("refined_conversation")
        ]
        summary_requests = [
            item for item in client.requests if item["payload"].get("requested_output", {}).get("summary")
        ]
        reduction_requests = [item for item in client.requests if "partial_summaries" in item["payload"]]
        self.assertEqual(len(refined_requests), 2)
        self.assertEqual(len(summary_requests), 2)
        self.assertEqual(len(reduction_requests), 1)
        self.assertEqual(set(result.refined_text_by_segment_id), {"s-0001", "s-0002", "s-0003", "s-0004"})
        self.assertIsNotNone(result.summary)
        self.assertTrue(any("다듬은 대화 1/2" in message for message in progress))
        self.assertTrue(any("부분 요약 1/1" in message for message in progress))

    def test_long_single_utterance_is_repartitioned_at_sentence_boundaries(self) -> None:
        client = _RepartitioningFakeClient()
        long_reading = " ".join(("가" * 150 + ".", "나" * 150 + ".", "다" * 150 + "."))
        analysis = AnalysisResult(
            source_name="reading.m4a",
            duration=30,
            speaker_segments=[],
            utterances=[Utterance(0, 30, "화자 1", long_reading)],
            whisper_profile="빠른 전사",
        )

        result = TranscriptRefiner(client, document_first=False).refine(
            analysis,
            AIRefinementRequest(selection=ResultSelection(refined=True)),
            model="local-model",
        )

        # The long diarization utterance becomes three request fragments.  On
        # a length failure, only those unfinished fragments are divided again.
        self.assertEqual(client.batch_sizes, [3, 2, 1])
        self.assertEqual(client.retry_on_length, [False, False, True])
        self.assertEqual(set(result.refined_text_by_segment_id), {"s-0001"})
        rendered = render_refined_conversation(analysis, result)
        self.assertEqual(rendered.count("(화자 1)"), 1)
        self.assertIn("다듬은", rendered)

    def test_length_failure_repartitions_only_unfinished_utterance_batches(self) -> None:
        client = _RepartitioningFakeClient()
        analysis = AnalysisResult(
            source_name="retry.m4a",
            duration=16,
            speaker_segments=[],
            utterances=[Utterance(float(index), float(index + 1), "화자 1", f"짧은 발화 {index}") for index in range(8)],
            whisper_profile="빠른 전사",
        )
        progress: list[str] = []

        result = TranscriptRefiner(client, document_first=False).refine(
            analysis,
            AIRefinementRequest(selection=ResultSelection(refined=True)),
            model="local-model",
            progress=progress.append,
        )

        self.assertEqual(client.batch_sizes, [8, 5, 3])
        self.assertEqual(set(result.refined_text_by_segment_id), {f"s-{index:04d}" for index in range(1, 9)})
        self.assertTrue(any("완료한 0개 묶음은 유지" in message for message in progress))

    def test_length_failure_keeps_completed_batches_before_repartitioning(self) -> None:
        client = _RepartitionAfterSuccessFakeClient()
        analysis = AnalysisResult(
            source_name="resume.m4a",
            duration=60,
            speaker_segments=[],
            utterances=[Utterance(float(index), float(index + 1), "화자 1", f"발화 {index}") for index in range(30)],
            whisper_profile="빠른 전사",
        )
        progress: list[str] = []

        result = TranscriptRefiner(client, document_first=False).refine(
            analysis,
            AIRefinementRequest(selection=ResultSelection(refined=True)),
            model="local-model",
            progress=progress.append,
        )

        self.assertEqual(client.batch_sizes, [20, 10, 5, 5])
        self.assertEqual(set(result.refined_text_by_segment_id), {f"s-{index:04d}" for index in range(1, 31)})
        self.assertTrue(any("완료한 1개 묶음은 유지" in message for message in progress))

    def test_summary_repartitions_unfinished_transcript_batches(self) -> None:
        client = _SummaryRepartitioningFakeClient()
        analysis = AnalysisResult(
            source_name="summary-retry.m4a",
            duration=16,
            speaker_segments=[],
            utterances=[Utterance(float(index), float(index + 1), "화자 1", f"짧은 발화 {index}") for index in range(8)],
            whisper_profile="빠른 전사",
        )

        result = TranscriptRefiner(client, document_first=False).refine(
            analysis,
            AIRefinementRequest(selection=ResultSelection(summary=True)),
            model="local-model",
        )

        self.assertEqual(client.transcript_batch_sizes, [8, 5, 3])
        self.assertIsNotNone(result.summary)

    def test_summary_created_later_keeps_the_existing_refined_conversation(self) -> None:
        refined = AIRefinementResult(
            selection=ResultSelection(refined=True),
            refined_text_by_segment_id={
                "s-0001": "네, 다음 주까지 초안을 준비할게요.",
                "s-0002": "좋습니다. 일정을 확인해 주세요.",
            },
            model="local-model",
        )
        summary = AIRefinementResult(
            selection=ResultSelection(summary=True),
            summary=ConversationSummary(overview="초안 준비 일정을 논의했다."),
            model="local-model",
        )

        merged = merge_ai_refinement_results(refined, summary)

        self.assertTrue(merged.selection.refined)
        self.assertTrue(merged.selection.summary)
        self.assertEqual(merged.refined_text_by_segment_id, refined.refined_text_by_segment_id)
        self.assertEqual(merged.summary, summary.summary)
        self.assertEqual(set(render_result_documents(_analysis(), merged)), {"original", "refined", "summary"})

    def test_invalid_or_missing_segments_fall_back_to_original_text(self) -> None:
        client = _FakeClient(
            {
                "segments": [
                    {"segment_id": "s-0001", "refined_text": "정상적으로 다듬은 첫 문장입니다."},
                    {"segment_id": "unknown", "refined_text": "이 문장은 사용하면 안 됩니다."},
                ]
            }
        )

        result = TranscriptRefiner(client, document_first=False).refine(
            _analysis(),
            AIRefinementRequest(selection=ResultSelection(refined=True)),
            model="local-model",
        )

        rendered = render_refined_conversation(_analysis(), result)
        self.assertIn("정상적으로 다듬은 첫 문장입니다.", rendered)
        self.assertIn("좋습니다 일정 확인 해주세요", rendered)
        self.assertNotIn("이 문장은 사용하면 안 됩니다.", rendered)
        self.assertTrue(result.warnings)

    def test_refinement_without_any_valid_segment_fails_without_applying_speaker_names(self) -> None:
        client = _FakeClient({"segments": [{"segment_id": "unknown", "refined_text": "사용하면 안 됩니다."}]})

        with self.assertRaisesRegex(RefinementError, "적용할 수 있는 다듬은 대화 구간"):
            TranscriptRefiner(client, document_first=False).refine(
                _analysis(),
                AIRefinementRequest(
                    selection=ResultSelection(refined=True),
                    speaker_guides=(SpeakerGuide("화자 1", display_name="손태영"),),
                ),
                model="local-model",
            )

    def test_original_only_does_not_require_a_model_connection(self) -> None:
        result = TranscriptRefiner(None).refine(
            _analysis(),
            AIRefinementRequest(selection=ResultSelection(original=True)),
            model="",
        )

        self.assertTrue(result.selection.original)
        self.assertFalse(result.selection.needs_ai)
        self.assertEqual(result.refined_text_by_segment_id, {})

    def test_original_cannot_be_combined_with_ai_results(self) -> None:
        with self.assertRaisesRegex(RefinementError, "함께 선택"):
            AIRefinementRequest(selection=ResultSelection(original=True, refined=True)).normalized()


if __name__ == "__main__":
    unittest.main()
