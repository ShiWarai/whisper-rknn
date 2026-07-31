"""Основной цикл utterance: VAD → mel → encode+decode → stitch."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Callable, List, Optional

import numpy as np

from app.core.model_config import ModelProfile
from app.core.text import stitch_transcripts
from app.core.types import DecodeResult, DecodeTimings, TaskType, TranscriptSegment
from app.pipeline.chunks import plan_utterance_chunks, utterance_mels
from app.pipeline.transport import ChunkTransport


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
    sample_rate = 16000

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
            timings.wall_ms = (time.perf_counter() - wall_t0) * 1000.0
            duration_s = samples.shape[0] / float(sample_rate)
            if duration_s > 0:
                timings.rtf = (timings.wall_ms / 1000.0) / duration_s
            result.timings = timings
        elif timings is not None:
            timings.truncated = result.truncated
            timings.wall_ms = (time.perf_counter() - wall_t0) * 1000.0
            duration_s = samples.shape[0] / float(sample_rate)
            if duration_s > 0:
                timings.rtf = (timings.wall_ms / 1000.0) / duration_s
            result.timings = timings
        if on_chunk is not None and result.text:
            on_chunk(result)
        return result

    async def _process_chunk(chunk_id: int, span, mel: np.ndarray) -> DecodeResult:
        return await transport.encode_then_decode(
            mel,
            chunk_id=chunk_id,
            time_offset_sec=span.start / float(sample_rate),
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
        if timings is not None and result.timings is not None:
            timings.encoder_ms += result.timings.encoder_ms
            timings.decoder_ms += result.timings.decoder_ms
            timings.tokens += result.timings.tokens
            timings.decoder_calls += result.timings.decoder_calls

        response_text = result.text
        if timestamps and result.segments:
            all_segments.extend(result.segments)
            if response_text:
                parts.append(response_text)
        elif response_text:
            parts.append(response_text)

    if timestamps:
        text = " ".join(segment.text for segment in all_segments).strip() or stitch_transcripts(
            parts
        )
        segments: Optional[List[TranscriptSegment]] = all_segments
    else:
        text = stitch_transcripts(parts)
        segments = None

    if timings is not None:
        timings.truncated = any_truncated
        timings.wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        duration_s = samples.shape[0] / float(sample_rate)
        if duration_s > 0:
            timings.rtf = (timings.wall_ms / 1000.0) / duration_s

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
    sample_rate = 16000

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
                time_offset_sec=span.start / float(sample_rate),
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
