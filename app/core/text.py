"""Склейка текстов чанков без зависимости от RKNN."""

from __future__ import annotations

from typing import List


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
