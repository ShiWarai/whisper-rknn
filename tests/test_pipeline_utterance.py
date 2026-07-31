"""Unit-тесты run_utterance_pipeline с mock transport."""

from __future__ import annotations

import asyncio

import numpy as np

from app.core.model_config import ModelProfile
from app.core.types import DecodeResult
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
        timestamps: bool,
        collect_timings: bool = False,
    ) -> DecodeResult:
        del mel, task, language, timestamps, collect_timings
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
