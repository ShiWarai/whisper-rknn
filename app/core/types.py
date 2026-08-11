"""Общие типы ASR-пайплайна (без зависимости от RKNN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

TaskType = Literal["transcribe", "translate"]


@dataclass
class TranscriptSegment:
    """Временной отрезок транскрипта (секунды от начала высказывания)."""

    start: float
    end: float
    text: str


@dataclass
class DecodeTimings:
    """Тайминги этапов на одно высказывание (миллисекунды, если не указано иное)."""

    audio_ms: float = 0.0
    mel_ms: float = 0.0
    encoder_ms: float = 0.0
    decoder_ms: float = 0.0
    tokens: int = 0
    decoder_calls: int = 0
    chunks: int = 1
    wall_ms: float = 0.0
    rtf: float = 0.0
    decoder_backend: str = "rknn"
    truncated: bool = False
    vad_ms: float = 0.0
    cut_ms: float = 0.0
    encode_queue_wait_ms: float = 0.0
    encoder_wall_ms: float = 0.0
    parallel_workers: int = 0

    def to_dict(self) -> dict:
        return {
            "audio_ms": round(self.audio_ms, 2),
            "mel_ms": round(self.mel_ms, 2),
            "encoder_ms": round(self.encoder_ms, 2),
            "encoder_wall_ms": round(self.encoder_wall_ms, 2),
            "decoder_ms": round(self.decoder_ms, 2),
            "tokens": self.tokens,
            "decoder_calls": self.decoder_calls,
            "chunks": self.chunks,
            "wall_ms": round(self.wall_ms, 2),
            "rtf": round(self.rtf, 4),
            "decoder_backend": self.decoder_backend,
            "truncated": self.truncated,
            "vad_ms": round(self.vad_ms, 2),
            "cut_ms": round(self.cut_ms, 2),
            "encode_queue_wait_ms": round(self.encode_queue_wait_ms, 2),
            "parallel_workers": self.parallel_workers,
        }

    def merge(self, other: "DecodeTimings") -> None:
        self.audio_ms += other.audio_ms
        self.mel_ms += other.mel_ms
        self.encoder_ms += other.encoder_ms
        self.encoder_wall_ms += other.encoder_wall_ms
        self.decoder_ms += other.decoder_ms
        self.tokens += other.tokens
        self.decoder_calls += other.decoder_calls
        self.chunks += other.chunks
        self.wall_ms += other.wall_ms
        self.truncated = self.truncated or other.truncated
        self.vad_ms += other.vad_ms
        self.cut_ms += other.cut_ms
        self.encode_queue_wait_ms += other.encode_queue_wait_ms
        if other.parallel_workers:
            self.parallel_workers = other.parallel_workers
        if other.decoder_backend and other.decoder_backend != self.decoder_backend:
            self.decoder_backend = other.decoder_backend


@dataclass
class DecodeResult:
    text: str
    segments: Optional[List[TranscriptSegment]] = None
    timings: Optional[DecodeTimings] = None
    truncated: bool = False
