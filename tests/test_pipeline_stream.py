"""Unit-тесты reorder-буфера SSE."""

from __future__ import annotations

import asyncio

from app.core.types import DecodeResult
from app.pipeline.stream import emit_chunks_in_order


def test_emit_chunks_in_order_skips_empty_chunk_zero():
    async def _chunks():
        yield 1, DecodeResult(text="second")
        yield 0, DecodeResult(text="")

    out = asyncio.run(_collect(emit_chunks_in_order(_chunks())))
    assert out == ["second"]


def test_emit_chunks_in_order_preserves_sequence():
    async def _chunks():
        yield 2, DecodeResult(text="c")
        yield 0, DecodeResult(text="a")
        yield 1, DecodeResult(text="b")

    out = asyncio.run(_collect(emit_chunks_in_order(_chunks())))
    assert out == ["a", "b", "c"]


async def _collect(aiter):
    return [r.text async for r in aiter]
