"""Основной цикл utterance: VAD → mel → encode+decode → stitch."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Callable, List, Optional

import numpy as np

from app.core.model_config import ModelProfile
from app.core.text import stitch_transcripts
from app.core.types import DecodeResult, DecodeTimings, TaskType, TranscriptSegment
from app.pipeline.chunk_transport import ChunkTransport
from app.pipeline.chunks import plan_utterance_chunks, utterance_mels

_SAMPLE_RATE = 16000


def _merge_timings(
    timings: Optional[DecodeTimings],
    result: DecodeResult,
) -> None:
    if timings is None or result.timings is None:
        return
    timings.encoder_ms += result.timings.encoder_ms
    timings.decoder_ms += result.timings.decoder_ms
    timings.tokens += result.timings.tokens
    timings.decoder_calls += result.timings.decoder_calls
    timings.truncated = timings.truncated or result.truncated


def _finalize_timings(
    timings: Optional[DecodeTimings],
    *,
    wall_t0: float,
    n_samples: int,
    chunks: int,
    truncated: bool,
) -> None:
    if timings is None:
        return
    timings.chunks = chunks
    timings.truncated = truncated
    timings.wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    duration_s = n_samples / float(_SAMPLE_RATE)
    if duration_s > 0:
        timings.rtf = (timings.wall_ms / 1000.0) / duration_s


async def run_utterance_pipeline(
    samples: np.ndarray,
    transport: ChunkTransport,
    profile: ModelProfile,
    *,
    task: TaskType = "transcribe",
    language: Optional[str] = None,
    timestamps: bool = False,
    on_chunk: Optional[Callable[[DecodeResult], None]] = None,
    collect_timings: bool = False,
) -> DecodeResult:
    wall_t0 = time.perf_counter()
    timings = DecodeTimings(decoder_backend="onnx") if collect_timings else None

    plan = plan_utterance_chunks(samples, profile)
    if timings is not None:
        timings.vad_ms = plan.vad_timings.vad_ms
        timings.cut_ms = plan.vad_timings.cut_ms
        timings.chunks = len(plan.spans)

    mels = utterance_mels(samples, plan, profile, timings=timings)

    if len(plan.spans) == 1:
        result = await transport.encode_then_decode(
            mels[0],
            chunk_id=0,
            time_offset_sec=0.0,
            task=task,
            language=language,
            timestamps=timestamps,
            collect_timings=collect_timings,
        )
        if timings is not None and result.timings is not None:
            timings.encoder_ms = result.timings.encoder_ms
            timings.decoder_ms = result.timings.decoder_ms
            timings.tokens = result.timings.tokens
            timings.decoder_calls = result.timings.decoder_calls
            timings.truncated = result.truncated
        elif timings is not None:
            timings.truncated = result.truncated
        _finalize_timings(
            timings,
            wall_t0=wall_t0,
            n_samples=int(samples.shape[0]),
            chunks=1,
            truncated=bool(result.truncated),
        )
        if timings is not None:
            result.timings = timings
        if on_chunk is not None and result.text:
            on_chunk(result)
        return result

    async def _process_chunk(chunk_id: int, span, mel: np.ndarray) -> DecodeResult:
        return await transport.encode_then_decode(
            mel,
            chunk_id=chunk_id,
            time_offset_sec=span.start / float(_SAMPLE_RATE),
            task=task,
            language=language,
            timestamps=timestamps,
            collect_timings=collect_timings,
        )

    tasks = [
        asyncio.create_task(_process_chunk(i, span, mel))
        for i, (span, mel) in enumerate(zip(plan.spans, mels, strict=True))
    ]
    results = await asyncio.gather(*tasks)

    parts: List[str] = []
    all_segments: List[TranscriptSegment] = []
    any_truncated = False

    for result in results:
        any_truncated = any_truncated or result.truncated
        if on_chunk is not None and result.text:
            on_chunk(result)
        _merge_timings(timings, result)
        if timestamps and result.segments is not None:
            all_segments.extend(result.segments)
            if result.text:
                parts.append(result.text)
        elif result.text:
            parts.append(result.text)

    if timestamps:
        text = stitch_transcripts(parts)
        segments: Optional[List[TranscriptSegment]] = all_segments or None
    else:
        text = stitch_transcripts(parts)
        segments = None

    _finalize_timings(
        timings,
        wall_t0=wall_t0,
        n_samples=int(samples.shape[0]),
        chunks=len(plan.spans),
        truncated=any_truncated,
    )
    return DecodeResult(
        text=text,
        segments=segments,
        timings=timings,
        truncated=any_truncated,
    )


async def utterance_stream(
    samples: np.ndarray,
    transport: ChunkTransport,
    profile: ModelProfile,
    *,
    task: TaskType = "transcribe",
    language: Optional[str] = None,
) -> AsyncIterator[DecodeResult]:
    """Тот же цикл, что run_utterance_pipeline, но с emit_chunks_in_order для SSE."""
    from app.pipeline.stream import emit_chunks_in_order

    plan = plan_utterance_chunks(samples, profile)
    mels = utterance_mels(samples, plan, profile)

    if len(plan.spans) == 1:
        result = await transport.encode_then_decode(
            mels[0],
            chunk_id=0,
            time_offset_sec=0.0,
            task=task,
            language=language,
            timestamps=False,
        )
        if result.text:
            yield result
        return

    async def _chunk_results():
        async def _process(chunk_id: int, span, mel: np.ndarray) -> tuple[int, DecodeResult]:
            result = await transport.encode_then_decode(
                mel,
                chunk_id=chunk_id,
                time_offset_sec=span.start / float(_SAMPLE_RATE),
                task=task,
                language=language,
                timestamps=False,
            )
            return chunk_id, result

        tasks = [
            asyncio.create_task(_process(chunk_id, span, mel))
            for chunk_id, (span, mel) in enumerate(zip(plan.spans, mels, strict=True))
        ]
        for finished in asyncio.as_completed(tasks):
            yield await finished

    async for result in emit_chunks_in_order(_chunk_results()):
        yield result
