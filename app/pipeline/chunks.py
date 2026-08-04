"""Планирование чанков и mel для высказывания."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.audio_features import compute_features
from app.core.model_config import ModelProfile
from app.core.types import DecodeTimings
from app.core.window import whisper_window_samples
from app.speech_cut import ChunkSpan, VadCutTimings, chunk_audio_views, plan_voice_aware_chunks


@dataclass(frozen=True)
class UtterancePlan:
    spans: List[ChunkSpan]
    vad_timings: VadCutTimings
    use_full_audio: bool
    # Silero probs на кадр (для мелкой нарезки таймингов без повторного VAD).
    probs: Optional[np.ndarray] = None


def plan_utterance_chunks(
    samples: np.ndarray,
    profile: ModelProfile,
) -> UtterancePlan:
    """VAD или одно окно; при одном span — декодировать всё высказывание целиком."""
    sample_rate = 16000
    chunk_samples = whisper_window_samples(sample_rate, profile.mel_time_frames)
    n = int(samples.shape[0])

    if n <= chunk_samples:
        return UtterancePlan(
            spans=[ChunkSpan(0, n, reason="single_window")],
            vad_timings=VadCutTimings(),
            use_full_audio=True,
            probs=None,
        )

    spans, probs, vad_timings = plan_voice_aware_chunks(samples)
    if len(spans) == 1:
        return UtterancePlan(
            spans=[ChunkSpan(0, n, reason="single_vad")],
            vad_timings=vad_timings,
            use_full_audio=True,
            probs=probs,
        )
    return UtterancePlan(
        spans=spans,
        vad_timings=vad_timings,
        use_full_audio=False,
        probs=probs,
    )


def utterance_mels(
    samples: np.ndarray,
    plan: UtterancePlan,
    profile: ModelProfile,
    *,
    timings: Optional[DecodeTimings] = None,
) -> List[np.ndarray]:
    """Mel-спектрограммы для каждого span плана."""
    if plan.use_full_audio and len(plan.spans) == 1:
        chunks = [samples]
    else:
        chunks = chunk_audio_views(samples, plan.spans)

    mel_t0 = time.perf_counter()
    mels = [
        compute_features(
            chunk,
            n_mels=profile.n_mels,
            target_frames=profile.mel_time_frames,
        )
        for chunk in chunks
    ]
    if timings is not None:
        timings.mel_ms = (time.perf_counter() - mel_t0) * 1000.0
    return mels
