"""Тесты очистки и склейки текста."""

from __future__ import annotations

from app.core.text import clean_transcript_segments, clean_transcript_text, stitch_transcripts
from app.core.types import TranscriptSegment


def test_clean_transcript_text_removes_dimatorzok_watermark():
    assert clean_transcript_text("Субтитры создавал DimaTorzok") == ""
    assert clean_transcript_text("субтитры создавал dima torzok") == ""
    assert (
        clean_transcript_text("Привет. Субтитры создавал DimaTorzok. Пока.")
        == "Привет. Пока"
    )


def test_clean_transcript_segments_drops_empty_segments():
    segments = clean_transcript_segments(
        [
            TranscriptSegment(start=0.0, end=1.0, text="Субтитры создавал DimaTorzok"),
            TranscriptSegment(start=1.0, end=2.0, text="живой текст"),
        ]
    )
    assert segments == [TranscriptSegment(start=1.0, end=2.0, text="живой текст")]


def test_stitch_transcripts_dedupes_overlap():
    left = "привет проверка звука я хочу понять насколько"
    right = "я хочу понять насколько тяжело склеить сообщения"
    assert stitch_transcripts([left, right]) == (
        "привет проверка звука я хочу понять насколько тяжело склеить сообщения"
    )
