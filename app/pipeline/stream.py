"""SSE reorder-буфер для чанков out-of-order."""

from __future__ import annotations

from typing import AsyncIterator

from app.core.types import DecodeResult


async def emit_chunks_in_order(
    async_iter,
) -> AsyncIterator[DecodeResult]:
    """
  Буферизовать (chunk_id, DecodeResult) и эмитить по возрастанию chunk_id.
  Пустые слоты не блокируют следующие chunk_id.
  """
    pending: dict[int, DecodeResult] = {}
    next_to_emit = 0

    async for chunk_id, result in async_iter:
        pending[chunk_id] = result
        while next_to_emit in pending:
            slot = pending.pop(next_to_emit)
            next_to_emit += 1
            if slot.text:
                yield slot

    while next_to_emit in pending:
        slot = pending.pop(next_to_emit)
        next_to_emit += 1
        if slot.text:
            yield slot
