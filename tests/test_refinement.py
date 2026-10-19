from __future__ import annotations

import json
import unittest

from talktracker.core.analysis import AnalysisResult, Utterance
from talktracker.core.refinement import (
    AIRefinementRequest,
    ResultSelection,
    SpeakerGuide,
    TranscriptRefiner,
    render_original_conversation,
    render_refined_conversation,
    render_summary,
)


class _FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.request: dict[str, object] | None = None

    def create_chat_completion(self, model: str, *, messages, temperature: float, max_tokens: int, should_cancel=None) -> str:
        self.request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if should_cancel and should_cancel():
            raise AssertionError("cancelled requests should not call the client")
        return json.dumps(self.response, ensure_ascii=False)


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
            selection=ResultSelection(original=True, refined=True, summary=True),
            background="주간 프로젝트 회의",
            speaker_guides=(SpeakerGuide("화자 1", "프로젝트 매니저"),),
        )

        result = TranscriptRefiner(client).refine(_analysis(), request, model="local-model")

        original = render_original_conversation(_analysis())
        refined = render_refined_conversation(_analysis(), result)
        self.assertIn("(화자 1)  00:01 – 00:05", original)
        self.assertIn("(화자 1)  00:01 – 00:05", refined)
        self.assertIn("네, 다음 주까지 초안을 준비할게요.", refined)
        self.assertNotIn("프로젝트 매니저", refined)
        self.assertIsNotNone(result.summary)
        assert result.summary is not None
        self.assertIn("초안 준비 일정을 확인했다.", render_summary(result.summary))
        self.assertEqual(client.request["model"], "local-model")
        payload = json.loads(client.request["messages"][1]["content"])
        self.assertEqual(payload["speaker_guides"], [{"speaker": "화자 1", "guidance": "프로젝트 매니저"}])

    def test_invalid_or_missing_segments_fall_back_to_original_text(self) -> None:
        client = _FakeClient(
            {
                "segments": [
                    {"segment_id": "s-0001", "refined_text": "정상적으로 다듬은 첫 문장입니다."},
                    {"segment_id": "unknown", "refined_text": "이 문장은 사용하면 안 됩니다."},
                ]
            }
        )

        result = TranscriptRefiner(client).refine(
            _analysis(),
            AIRefinementRequest(selection=ResultSelection(original=True, refined=True)),
            model="local-model",
        )

        rendered = render_refined_conversation(_analysis(), result)
        self.assertIn("정상적으로 다듬은 첫 문장입니다.", rendered)
        self.assertIn("좋습니다 일정 확인 해주세요", rendered)
        self.assertNotIn("이 문장은 사용하면 안 됩니다.", rendered)
        self.assertTrue(result.warnings)

    def test_original_only_does_not_require_a_model_connection(self) -> None:
        result = TranscriptRefiner(None).refine(
            _analysis(),
            AIRefinementRequest(selection=ResultSelection(original=True)),
            model="",
        )

        self.assertTrue(result.selection.original)
        self.assertFalse(result.selection.needs_ai)
        self.assertEqual(result.refined_text_by_segment_id, {})


if __name__ == "__main__":
    unittest.main()
