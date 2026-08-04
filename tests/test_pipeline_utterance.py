"""Unit-тесты run_utterance_pipeline с mock transport."""

from __future__ import annotations

import asyncio

import numpy as np

from app.core.model_config import ModelProfile
from app.core.types import DecodeResult, TranscriptSegment
from app.pipeline.utterance import run_utterance_pipeline


class _MockTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    async def encode_then_decode(
        self,
        mel,
        *,
        chunk_id: int,
        time_offset_sec: float,
        task,
        language,
        collect_timings: bool = False,
    ) -> DecodeResult:
        del mel, task, language, collect_timings
        self.calls.append((chunk_id, time_offset_sec))
        return DecodeResult(text=f"chunk{chunk_id}")


def test_run_utterance_pipeline_single_window(monkeypatch):
    profile = ModelProfile.from_profile("tiny")
    transport = _MockTransport()
    samples = np.zeros(16000, dtype=np.float32)

    async def _no_vad(*args, **kwargs):
        raise AssertionError("VAD should not run for short audio")

    monkeypatch.setattr("app.pipeline.chunks.plan_voice_aware_chunks", _no_vad)

    result = asyncio.run(
        run_utterance_pipeline(
            samples,
            transport,
            profile,
            task="transcribe",
            language="ru",
        )
    )
    assert result.text == "chunk0"
    assert transport.calls == [(0, 0.0)]


def test_run_utterance_pipeline_timestamps_uses_fine_vad_spans(monkeypatch):
    from app.speech_cut import ChunkSpan

    profile = ModelProfile.from_profile("tiny")
    transport = _MockTransport()
    samples = np.zeros(16000, dtype=np.float32)

    async def _no_vad(*args, **kwargs):
        raise AssertionError("VAD should not run for short audio")

    monkeypatch.setattr("app.pipeline.chunks.plan_voice_aware_chunks", _no_vad)

    class _FakeVad:
        def speech_probs(self, audio):
            return np.ones(max(1, int(audio.shape[0]) // 512), dtype=np.float32)

    monkeypatch.setattr("app.pipeline.utterance.get_vad_session", lambda: _FakeVad())
    monkeypatch.setattr(
        "app.pipeline.utterance.segment_spans_from_probs",
        lambda samples, probs, **kwargs: [
            ChunkSpan(0, int(samples.shape[0]), "fine")
        ],
    )

    result = asyncio.run(
        run_utterance_pipeline(
            samples,
            transport,
            profile,
            task="transcribe",
            language="ru",
            timestamps=True,
        )
    )
    assert transport.calls == [(0, 0.0)]
    assert result.text == "chunk0"
    assert result.segments is not None
    assert result.segments == [
        TranscriptSegment(start=0.0, end=1.0, text="chunk0"),
    ]
