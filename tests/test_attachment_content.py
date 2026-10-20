from __future__ import annotations

import unittest

from talktracker.core.attachment_content import (
    ATTACHMENT_EXCERPT_MARKER,
    MAX_ATTACHMENT_CONTENT_BYTES,
    create_attachment_chunks,
    select_attachment_content,
    text_byte_size,
)


class AttachmentContentTests(unittest.TestCase):
    def test_small_document_is_sent_whole(self) -> None:
        source = "# 회의 전사\n\n(화자 1) 00:01\n초안을 다음 주까지 준비합니다."

        selection = select_attachment_content(source, create_attachment_chunks(source), "이 파일을 요약해 주세요")

        self.assertEqual(selection.content, source)
        self.assertFalse(selection.truncated)
        self.assertEqual(selection.selection_summary, "")

    def test_long_document_selects_relevant_units_with_context_anchors(self) -> None:
        sections = ["# 시작\n\n" + ("일반 논의입니다. " * 4_000)]
        sections.extend(f"# 일반 논의 {index}\n\n" + ("다른 안건입니다. " * 4_000) for index in range(1, 5))
        sections.append("# 초안 일정\n\n" + ("초안 일정을 다음 주에 확정합니다. " * 4_000))
        sections.append("# 마무리\n\n" + ("후속 확인이 필요합니다. " * 4_000))
        source = "\n\n".join(sections)

        selection = select_attachment_content(source, create_attachment_chunks(source), "초안 일정 요약")

        self.assertTrue(selection.truncated)
        self.assertLessEqual(text_byte_size(selection.content), MAX_ATTACHMENT_CONTENT_BYTES)
        self.assertIn("[문서 전체", selection.content)
        self.assertIn("초안 일정", selection.content)
        self.assertRegex(selection.selection_summary, r"^문서 \d+개 단위 중 \d+개 전달$")

    def test_long_document_without_query_match_uses_agent_chat_excerpt_marker(self) -> None:
        source = "가" * (MAX_ATTACHMENT_CONTENT_BYTES // 3 + 1)

        selection = select_attachment_content(source, create_attachment_chunks(source), "zzzzzz")

        self.assertTrue(selection.truncated)
        self.assertLessEqual(text_byte_size(selection.content), MAX_ATTACHMENT_CONTENT_BYTES)
        self.assertIn(ATTACHMENT_EXCERPT_MARKER, selection.content)
        self.assertEqual(selection.selection_summary, "앞·뒤 발췌본 전달")


if __name__ == "__main__":
    unittest.main()
