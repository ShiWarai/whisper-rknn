"""Протокол транспорта чанка (без зависимости от gRPC)."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from app.core.types import DecodeResult, TaskType


@runtime_checkable
class ChunkTransport(Protocol):
    async def encode_then_decode(
        self,
        mel: np.ndarray,
        *,
        chunk_id: int,
        time_offset_sec: float,
        task: TaskType,
        language: Optional[str],
        timestamps: bool,
        collect_timings: bool = False,
    ) -> DecodeResult: ...
