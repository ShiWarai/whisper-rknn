"""Unit tests for app.speech_cut (no RKNN)."""

from __future__ import annotations

import numpy as np

from app.speech_cut import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    ChunkSpan,
    find_cut_sample,
    iter_voice_aware_spans,
)


def _probs_from_mask(mask: list[bool]) -> np.ndarray:
    return np.array([0.9 if speech else 0.1 for speech in mask], dtype=np.float32)


def test_find_cut_sample_prefers_long_gap():
    # Gap near the 30s boundary (frames 27-29 of 30-frame window)
    mask = [True] * 27 + [False] * 3 + [True] * 10
    probs = _probs_from_mask(mask)
    cut, reason = find_cut_sample(
        probs,
        start_sample=0,
        max_samples=30 * WINDOW_SAMPLES,
        search_back_samples=10 * WINDOW_SAMPLES,
        threshold=0.5,
        min_gap_frames=3,
    )
    assert cut > 0
    assert cut <= 30 * WINDOW_SAMPLES
    assert "vad_gap" in reason


def test_iter_voice_aware_spans_tail():
    n_samples = int(35 * SAMPLE_RATE)
    audio = np.zeros(n_samples, dtype=np.float32)
    probs = np.full(n_samples // WINDOW_SAMPLES + 1, 0.9, dtype=np.float32)
    spans = iter_voice_aware_spans(
        audio,
        probs,
        max_sec=30.0,
        search_back_sec=3.0,
        threshold=0.5,
        min_gap_ms=250.0,
    )
    assert len(spans) >= 2
    assert spans[0].start == 0
    assert spans[-1].end == n_samples
    assert spans[-1].reason == "tail"


def test_chunk_span_duration():
    span = ChunkSpan(16000, 48000, "test")
    assert span.duration_s == 2.0
