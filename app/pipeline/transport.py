"""Транспорт encode→decode на чанк (local in-process или gRPC)."""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

import numpy as np

from app.core.grpc_client import DecodeClientPool, EncodeClientPool
from app.core.grpc_gen.whisper_rknn.v1 import worker_pb2
from app.core.model_config import ModelProfile
from app.core.tensor_codec import ndarray_to_tensor
from app.core.types import DecodeResult, DecodeTimings, TaskType, TranscriptSegment
from app.pipeline.chunk_transport import ChunkTransport

__all__ = [
    "ChunkTransport",
    "GrpcChunkTransport",
    "LocalChunkTransport",
    "decode_response_to_result",
]


def decode_response_to_result(
    response: worker_pb2.DecodeResponse,
    *,
    encode_ms: float = 0.0,
    collect_timings: bool = False,
) -> DecodeResult:
    segments = None
    if response.segments:
        segments = [
            TranscriptSegment(
                start=segment.start_sec,
                end=segment.end_sec,
                text=segment.text,
            )
            for segment in response.segments
        ]
    timings = None
    if collect_timings:
        timings = DecodeTimings(
            decoder_backend="onnx",
            encoder_ms=encode_ms,
            decoder_ms=response.decode_ms,
        )
    return DecodeResult(
        text=response.text,
        segments=segments,
        timings=timings,
        truncated=response.truncated,
    )


class LocalChunkTransport:
    """In-process: run_encoder → decode_from_cross_kv."""

    def __init__(self, model, id2token: dict) -> None:
        self._model = model
        self._id2token = id2token

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
    ) -> DecodeResult:
        del chunk_id

        def _run() -> DecodeResult:
            import time

            from app.decode import decode_from_cross_kv

            enc_t0 = time.perf_counter()
            cross_kv = self._model.run_encoder(mel)
            encode_ms = (time.perf_counter() - enc_t0) * 1000.0
            try:
                result = decode_from_cross_kv(
                    self._model,
                    self._id2token,
                    cross_kv,
                    verbose=False,
                    timestamps=timestamps,
                    time_offset=time_offset_sec,
                    task=task,
                    language=language,
                    collect_timings=collect_timings,
                )
            finally:
                del cross_kv
            if collect_timings and result.timings is not None:
                result.timings.encoder_ms = encode_ms
            return result

        return await asyncio.to_thread(_run)


class GrpcChunkTransport:
    """gRPC EncodeThenDecode: encode воркер сам пересылает cross_kv в decode."""

    def __init__(
        self,
        profile: ModelProfile,
        encode_pool: EncodeClientPool,
        decode_pool: DecodeClientPool,
    ) -> None:
        self._profile = profile
        self._encode_pool = encode_pool
        self._decode_pool = decode_pool

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
    ) -> DecodeResult:
        job_id = uuid.uuid4().hex
        decode_ep = await self._decode_pool.acquire()
        try:
            request = worker_pb2.EncodeThenDecodeRequest(
                job_id=job_id,
                chunk_id=chunk_id,
                mel=ndarray_to_tensor(mel),
                model_profile=self._profile.size_key,
                decode_target=decode_ep.target,
                language=language or "",
                task=task,
                timestamps=timestamps,
                time_offset_sec=time_offset_sec,
            )
            response = await self._encode_pool.encode_then_decode(request)
            return decode_response_to_result(
                response,
                encode_ms=response.encode_ms,
                collect_timings=collect_timings,
            )
        finally:
            await self._decode_pool.release(decode_ep)
