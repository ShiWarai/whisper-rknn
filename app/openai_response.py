"""Форматирование ответов транскрипции, совместимых с OpenAI (hwdsl2/docker-whisper)."""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterable, List, Literal, Optional

from app.core.text import stitch_transcripts
from app.core.types import DecodeResult, TaskType, TranscriptSegment
from app.runtime.backend import AsrBackend

ResponseFormat = Literal["json", "text", "verbose_json", "srt", "vtt"]


def _fmt_ts(seconds: float, fmt: str) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    sep = "," if fmt == "srt" else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: List[TranscriptSegment]) -> str:
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(
            f"{i}\n"
            f"{_fmt_ts(seg.start, 'srt')} --> {_fmt_ts(seg.end, 'srt')}\n"
            f"{seg.text.strip()}\n"
        )
    return "\n".join(lines)


def to_vtt(segments: List[TranscriptSegment]) -> str:
    lines = ["WEBVTT\n"]
    for seg in segments:
        lines.append(
            f"{_fmt_ts(seg.start, 'vtt')} --> {_fmt_ts(seg.end, 'vtt')}\n"
            f"{seg.text.strip()}\n"
        )
    return "\n".join(lines)


def verbose_json_payload(
    result: DecodeResult,
    *,
    task: TaskType,
    language: Optional[str],
    duration: float,
) -> dict:
    segments = result.segments or []
    seg_list = [
        {
            "id": idx,
            "seek": 0,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 0.0,
            "no_speech_prob": 0.0,
        }
        for idx, seg in enumerate(segments)
    ]
    return {
        "task": task,
        "language": language or "unknown",
        "duration": round(duration, 3),
        "text": result.text,
        "segments": seg_list,
    }


def timings_payload(result: DecodeResult) -> Optional[dict]:
    if result.timings is None:
        return None
    return result.timings.to_dict()


def format_transcription_response(
    result: DecodeResult,
    *,
    response_format: ResponseFormat,
    task: TaskType,
    language: Optional[str],
    duration: float,
) -> tuple[bytes | str, str]:
    """Вернуть (body, media_type)."""
    if response_format == "text":
        return result.text, "text/plain"

    if response_format == "srt":
        segments = result.segments or []
        return to_srt(segments), "text/plain"

    if response_format == "vtt":
        segments = result.segments or []
        return to_vtt(segments), "text/plain"

    if response_format == "verbose_json":
        payload = verbose_json_payload(
            result, task=task, language=language, duration=duration
        )
        stage_timings = timings_payload(result)
        if stage_timings is not None:
            payload["timings"] = stage_timings
        return json.dumps(payload, ensure_ascii=False), "application/json"

    return json.dumps({"text": result.text}, ensure_ascii=False), "application/json"


def sse_delta_frame(delta: str) -> str:
    payload = json.dumps({"type": "transcript.text.delta", "delta": delta})
    return f"data: {payload}\n\n"


def sse_done_frame(full_text: str) -> str:
    payload = json.dumps({"type": "transcript.text.done", "text": full_text})
    return f"data: {payload}\n\n"


def sse_error_frame(message: str) -> str:
    payload = json.dumps(
        {"error": {"type": "transcription_error", "message": message}}
    )
    return f"data: {payload}\n\n"


def stream_sse_frames(chunk_texts: Iterable[str], full_text: str) -> List[str]:
    """Собрать SSE-кадры в стиле OpenAI из текста по чанкам."""
    frames: List[str] = []
    first = True
    for part in chunk_texts:
        text = part.strip()
        if not text:
            continue
        delta = text if first else " " + text
        first = False
        frames.append(sse_delta_frame(delta))
    frames.append(sse_done_frame(full_text))
    frames.append("data: [DONE]\n\n")
    return frames


async def stream_transcription_sse(
    backend: AsrBackend,
    samples,
    *,
    task: TaskType,
    language: Optional[str],
) -> AsyncIterator[str]:
    """Единый SSE-поток для local и distributed (reorder в pipeline)."""
    chunk_texts: List[str] = []
    first = True
    async for chunk_result in backend.decode_utterance_stream(
        samples,
        task=task,
        language=language,
    ):
        text = chunk_result.text.strip()
        if not text:
            continue
        chunk_texts.append(chunk_result.text)
        delta = text if first else " " + text
        first = False
        yield sse_delta_frame(delta)
    full_text = stitch_transcripts(chunk_texts)
    yield sse_done_frame(full_text)
    yield "data: [DONE]\n\n"


def merge_chunk_results(chunk_results: List[DecodeResult]) -> DecodeResult:
    parts = [r.text for r in chunk_results if r.text]
    text = stitch_transcripts(parts)
    all_segments: List[TranscriptSegment] = []
    for r in chunk_results:
        if r.segments:
            all_segments.extend(r.segments)
    return DecodeResult(
        text=text,
        segments=all_segments or None,
    )
