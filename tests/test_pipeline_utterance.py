"""Unit-тесты run_utterance_pipeline с mock transport."""

from __future__ import annotations

import asyncio

import numpy as np

from app.core.model_config import ModelProfile
from app.core.types import DecodeResult, TranscriptSegment
from app.pipeline.utterance import run_utterance_pipeline


class _MockTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float, bool]] = []

    async def encode_then_decode(
        self,
        mel,
        *,
        chunk_id: int,
        time_offset_sec: float,
        task,
        language,
        timestamps: bool,
        collect_timings: bool = False,
    ) -> DecodeResult:
        del mel, task, language, collect_timings
        self.calls.append((chunk_id, time_offset_sec, timestamps))
        if timestamps:
            return DecodeResult(
                text="hello",
                segments=[TranscriptSegment(start=0.0, end=1.0, text="hello")],
            )
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
    assert transport.calls == [(0, 0.0, False)]


def test_run_utterance_pipeline_timestamps_single_decode(monkeypatch):
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
            timestamps=True,
        )
    )
    assert transport.calls == [(0, 0.0, True)]
    assert result.text == "hello"
    assert result.segments is not None
    assert result.segments[0].text == "hello"
