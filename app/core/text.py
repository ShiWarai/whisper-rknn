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
