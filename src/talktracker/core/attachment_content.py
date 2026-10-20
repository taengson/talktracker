"""Agent Chat-compatible text attachment selection.

This is a direct Python port of Agent Chat's current attachment-content
policy.  It keeps a document whole when it fits the model-content budget and,
when it does not, sends the most relevant 16 KiB document units together with
small leading/trailing anchors.  The policy is deliberately independent from
the model so a local model receives one ordinary chat message, not an internal
segment-ID protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


CHUNK_TARGET_BYTES = 16 * 1024
SELECTION_OVERHEAD_BYTES = 8 * 1024
MAX_ATTACHMENT_FILE_BYTES = 5 * 1024 * 1024
MAX_ATTACHMENT_TOTAL_FILE_BYTES = 12 * 1024 * 1024
MAX_ATTACHMENT_CONTENT_BYTES = 240 * 1024
MAX_ATTACHMENT_TOTAL_CONTENT_BYTES = 512 * 1024
ATTACHMENT_EXCERPT_MARKER = "\n\n[문서가 길어 앞부분과 뒷부분만 모델에 전달했습니다.]\n\n"

_QUERY_STOP_WORDS = {
    "이", "그", "저", "것", "내용", "문서", "파일", "관련", "대해", "대한", "해서", "해주세요", "해줘",
    "알려줘", "알려주세요", "설명해줘", "설명해주세요", "정리해줘", "정리해주세요", "요약해줘", "요약해주세요",
    "the", "and", "for", "with", "from", "about", "this", "that", "please", "document", "file",
}
_PARTICLE_SUFFIX = re.compile(r"(은|는|이|가|을|를|와|과|에|의|으로|로|에서|에게|한테|부터|까지|보다|처럼|도|만)$")
_TOKEN_PATTERN = re.compile(r"[^\W_]{2,}", re.UNICODE)
_SENTENCE_BOUNDARIES = re.compile(r"\n\s*\n|\n|[.!?。！？](?:\s|$)")
_NUMBERED_HEADING = re.compile(r"^(?:제\s*\d+\s*(?:편|부|장|절|조)|\d+(?:\.\d+){0,4}[.)]?)\s+\S+")


@dataclass(frozen=True)
class AttachmentChunk:
    label: str
    content: str


@dataclass(frozen=True)
class AttachmentSelection:
    content: str
    truncated: bool
    selection_summary: str = ""


def text_byte_size(content: str) -> int:
    return len(content.encode("utf-8"))


def _decode_leading_utf8(content: bytes, length: int) -> str:
    for trim in range(4):
        try:
            return content[: max(0, length - trim)].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            pass
    return content[:length].decode("utf-8", errors="replace")


def _decode_trailing_utf8(content: bytes, length: int) -> str:
    for trim in range(4):
        try:
            return content[max(0, len(content) - length + trim) :].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            pass
    return content[max(0, len(content) - length) :].decode("utf-8", errors="replace")


def _heading_label(line: str) -> str | None:
    value = line.strip()
    pdf_page = re.match(r"^\[PDF\s+(\d+)쪽\]$", value)
    if pdf_page:
        return f"PDF {pdf_page.group(1)}쪽"
    markdown = re.match(r"^#{1,6}\s+(.+)$", value)
    if markdown:
        return markdown.group(1).strip()
    return value if _NUMBERED_HEADING.match(value) else None


def _split_oversized_block(content: str, max_bytes: int) -> list[str]:
    pieces: list[str] = []
    remaining = content.strip()
    while text_byte_size(remaining) > max_bytes:
        leading = _decode_leading_utf8(remaining.encode("utf-8"), max_bytes)
        threshold = int(len(leading) * 0.45)
        boundaries = [match.end() for match in _SENTENCE_BOUNDARIES.finditer(leading) if match.end() >= threshold]
        boundary = boundaries[-1] if boundaries else len(leading)
        piece = leading[:boundary].rstrip()
        if not piece:
            break
        pieces.append(piece)
        remaining = remaining[boundary:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _document_blocks(content: str) -> list[AttachmentChunk]:
    blocks: list[AttachmentChunk] = []
    current_label = "문서 내용"
    for paragraph in re.split(r"\n{2,}", content.strip()):
        lines = paragraph.split("\n")
        segment: list[str] = []

        def push_segment() -> None:
            value = "\n".join(segment).strip()
            if value:
                blocks.append(AttachmentChunk(label=current_label, content=value))

        for line in lines:
            label = _heading_label(line)
            if label and any(value.strip() for value in segment):
                push_segment()
                segment = []
            if label:
                current_label = label
            segment.append(line)
        push_segment()
    return blocks


def create_attachment_chunks(content: str) -> tuple[AttachmentChunk, ...]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ()
    chunks: list[AttachmentChunk] = []
    current: AttachmentChunk | None = None
    for block in _document_blocks(normalized):
        pieces = _split_oversized_block(block.content, CHUNK_TARGET_BYTES)
        for index, piece in enumerate(pieces, start=1):
            label = f"{block.label} · 계속 {index}" if len(pieces) > 1 else block.label
            content_with_separator = f"{current.content}\n\n{piece}" if current else piece
            if current and (current.label != label or text_byte_size(content_with_separator) > CHUNK_TARGET_BYTES):
                chunks.append(current)
                current = None
            current = AttachmentChunk(label=label, content=piece if current is None else content_with_separator)
    if current and current.content:
        chunks.append(current)
    return tuple(chunks)


def _fallback_excerpt(content: str, content_budget: int) -> AttachmentSelection:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    encoded = normalized.encode("utf-8")
    if len(encoded) <= content_budget:
        return AttachmentSelection(content=normalized, truncated=False)
    available = content_budget - text_byte_size(ATTACHMENT_EXCERPT_MARKER)
    leading = int(available * 0.7)
    trailing = available - leading
    return AttachmentSelection(
        content=f"{_decode_leading_utf8(encoded, leading)}{ATTACHMENT_EXCERPT_MARKER}{_decode_trailing_utf8(encoded, trailing)}",
        truncated=True,
        selection_summary="앞·뒤 발췌본 전달",
    )


def _query_terms(query: str) -> tuple[str, ...]:
    terms: set[str] = set()
    for token in _TOKEN_PATTERN.findall(query.lower()):
        if token not in _QUERY_STOP_WORDS:
            terms.add(token)
        without_particle = _PARTICLE_SUFFIX.sub("", token)
        if len(without_particle) >= 2 and without_particle not in _QUERY_STOP_WORDS:
            terms.add(without_particle)
    return tuple(terms)


def _relevance_score(chunk: AttachmentChunk, terms: tuple[str, ...]) -> int:
    label, content = chunk.label.lower(), chunk.content.lower()
    score = 0
    for term in terms:
        if term in label:
            score += 8
        score += min(content.count(term), 3) * 2
    return score


def _display_label(label: str) -> str:
    trimmed = " ".join(label.split())
    return f"{trimmed[:77]}…" if len(trimmed) > 80 else trimmed


def _selected_chunk_size(chunk: AttachmentChunk, index: int, total: int) -> int:
    return text_byte_size(f"[문서 단위 {index + 1}/{total}: {_display_label(chunk.label)}]\n{chunk.content}\n\n")


def _render_selected_chunks(
    chunks: tuple[AttachmentChunk, ...], selected: dict[int, int], content_budget: int
) -> tuple[str, int] | None:
    indexes = sorted(selected)
    while indexes:
        header = f"[문서 전체 {len(chunks)}개 단위 중 {len(indexes)}개를 질문 관련성과 앞뒤 문맥에 따라 전달합니다.]"
        content = header + "\n\n" + "\n\n".join(
            f"[문서 단위 {index + 1}/{len(chunks)}: {_display_label(chunks[index].label)}]\n{chunks[index].content}"
            for index in indexes
        )
        if text_byte_size(content) <= content_budget:
            return content, len(indexes)
        remove = min(indexes, key=lambda index: selected[index])
        indexes.remove(remove)
    return None


def select_attachment_content(
    source: str,
    chunks: tuple[AttachmentChunk, ...],
    query: str,
    content_budget: int = MAX_ATTACHMENT_CONTENT_BYTES,
) -> AttachmentSelection:
    """Select text exactly as Agent Chat does before one model request."""

    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    if text_byte_size(normalized) <= content_budget:
        return AttachmentSelection(content=normalized, truncated=False)

    ranked = sorted(
        ((index, _relevance_score(chunk, _query_terms(query))) for index, chunk in enumerate(chunks)),
        key=lambda item: (-item[1], item[0]),
    )
    ranked = [(index, score) for index, score in ranked if score > 0]
    if not ranked:
        return _fallback_excerpt(normalized, content_budget)

    available = max(0, content_budget - SELECTION_OVERHEAD_BYTES)
    anchor_budget = int(available * 0.3)
    selected: dict[int, int] = {}
    used_size = 0

    def add(index: int, priority: int, limit: int) -> bool:
        nonlocal used_size
        if index < 0 or index >= len(chunks) or index in selected:
            return False
        size = _selected_chunk_size(chunks[index], index, len(chunks))
        if used_size + size > limit:
            return False
        selected[index] = priority
        used_size += size
        return True

    left, right, use_leading = 0, len(chunks) - 1, True
    while left <= right and used_size < anchor_budget:
        index = left if use_leading else right
        if use_leading:
            left += 1
        else:
            right -= 1
        add(index, 1, anchor_budget)
        use_leading = not use_leading
    for index, _score in ranked:
        add(index, 3, available)
    for index, _score in ranked:
        add(index - 1, 2, available)
        add(index + 1, 2, available)

    rendered = _render_selected_chunks(chunks, selected, content_budget)
    if not rendered:
        return _fallback_excerpt(normalized, content_budget)
    content, selected_count = rendered
    return AttachmentSelection(
        content=content,
        truncated=True,
        selection_summary=f"문서 {len(chunks)}개 단위 중 {selected_count}개 전달",
    )
