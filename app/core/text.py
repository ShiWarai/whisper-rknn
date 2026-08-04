"""Склейка текстов чанков без зависимости от RKNN."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:
    from app.core.types import TranscriptSegment

# Типичный watermark из субтитров (часто единственный «текст» в коротких роликах).
_WATERMARK_RE = re.compile(r"(?iu)субтитры\s+создавал\s+Dima\s*Torzok")


def clean_transcript_text(text: str) -> str:
    """Убрать известные watermark-фразы из распознанного текста."""
    if not text:
        return ""
    out = _WATERMARK_RE.sub("", text)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s*([.,!?;:])\s*", r"\1 ", out)
    out = re.sub(r"([.,!?;:])\s*\1+", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out.strip(".,!?;: «»\"'()[]")


def clean_transcript_segments(
    segments: Optional[Sequence["TranscriptSegment"]],
) -> Optional[List["TranscriptSegment"]]:
    if not segments:
        return None
    from app.core.types import TranscriptSegment

    cleaned: List[TranscriptSegment] = []
    for seg in segments:
        text = clean_transcript_text(seg.text)
        if text:
            cleaned.append(
                TranscriptSegment(start=seg.start, end=seg.end, text=text)
            )
    return cleaned or None


def redistribute_text_to_spans(
    text: str,
    spans: Sequence["TranscriptSegment"],
) -> List["TranscriptSegment"]:
    """Разложить качественный текст по готовым timing-сегментам (по длительности)."""
    from app.core.types import TranscriptSegment

    words = text.split()
    if not spans or not words:
        return []

    durations = [max(0.02, float(s.end) - float(s.start)) for s in spans]
    total = sum(durations) or float(len(spans))
    counts: List[int] = []
    allocated = 0
    for i, dur in enumerate(durations):
        if i == len(durations) - 1:
            counts.append(len(words) - allocated)
        else:
            n = int(round(len(words) * dur / total))
            n = max(0, min(n, len(words) - allocated))
            counts.append(n)
            allocated += n

    diff = len(words) - sum(counts)
    idx = len(counts) - 1
    while diff != 0 and counts:
        step = 1 if diff > 0 else -1
        if counts[idx] + step >= 0:
            counts[idx] += step
            diff -= step
        idx = (idx - 1) % len(counts)

    out: List[TranscriptSegment] = []
    cursor = 0
    for span, count in zip(spans, counts, strict=True):
        part = " ".join(words[cursor : cursor + count]).strip()
        cursor += count
        if part:
            out.append(
                TranscriptSegment(start=span.start, end=span.end, text=part)
            )
    return out


def stitch_transcripts(parts: List[str]) -> str:
    """Склеить тексты чанков; убрать дубли overlap (длиннейшее совпадение слов на стыке)."""
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    out = cleaned[0]
    for nxt in cleaned[1:]:
        out = _merge_overlap_text(out, nxt)
    return out.strip()


def _merge_overlap_text(left: str, right: str) -> str:
    """Добавить ``right`` к ``left``, убрав длиннейшую общую границу слов."""
    lw = left.split()
    rw = right.split()
    if not lw:
        return right
    if not rw:
        return left

    max_k = min(len(lw), len(rw), 48)

    def _norm(w: str) -> str:
        return w.lower().strip(".,!?;:«»\"'()[]")

    best = 0
    for k in range(max_k, 0, -1):
        if [_norm(x) for x in lw[-k:]] == [_norm(x) for x in rw[:k]]:
            best = k
            break
    if best:
        return " ".join(lw + rw[best:])
    return left + " " + right
