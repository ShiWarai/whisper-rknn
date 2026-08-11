"""ASR-бэкенды: тонкие обёртки над единым pipeline."""

from __future__ import annotations

import os
from typing import AsyncIterator, Optional, Protocol, runtime_checkable

import numpy as np

from app.core.grpc_client import DecodeClientPool, EncodeClientPool, expand_targets, parse_targets
from app.core.model_config import ModelProfile
from app.core.types import DecodeResult, TaskType
from app.pipeline.transport import ChunkTransport, GrpcChunkTransport, LocalChunkTransport
from app.pipeline.utterance import run_utterance_pipeline, utterance_stream


@runtime_checkable
class AsrBackend(Protocol):
    @property
    def profile(self) -> ModelProfile: ...

    @property
    def english_only(self) -> bool: ...

    async def decode_utterance(
        self,
        samples: np.ndarray,
        *,
        timestamps: bool = False,
        task: TaskType = "transcribe",
        language: Optional[str] = None,
        collect_timings: bool = False,
    ) -> DecodeResult: ...

    async def decode_utterance_stream(
        self,
        samples: np.ndarray,
        *,
        task: TaskType = "transcribe",
        language: Optional[str] = None,
    ) -> AsyncIterator[DecodeResult]: ...

    async def shutdown(self) -> None: ...


class _PipelineBackend:
    def __init__(self, profile: ModelProfile, transport: ChunkTransport) -> None:
        self._profile = profile
        self._transport = transport

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def english_only(self) -> bool:
        return self._profile.english_only

    async def decode_utterance(
        self,
        samples: np.ndarray,
        *,
        timestamps: bool = False,
        task: TaskType = "transcribe",
        language: Optional[str] = None,
        collect_timings: bool = False,
    ) -> DecodeResult:
        return await run_utterance_pipeline(
            samples,
            self._transport,
            self._profile,
            task=task,
            language=language,
            timestamps=timestamps,
            collect_timings=collect_timings,
        )

    async def decode_utterance_stream(
        self,
        samples: np.ndarray,
        *,
        task: TaskType = "transcribe",
        language: Optional[str] = None,
    ) -> AsyncIterator[DecodeResult]:
        async for chunk in utterance_stream(
            samples,
            self._transport,
            self._profile,
            task=task,
            language=language,
        ):
            yield chunk

    async def shutdown(self) -> None:
        return None


class LocalBackend(_PipelineBackend):
    """RKNN in-process."""

    def __init__(self, model, id2token: dict, profile: ModelProfile) -> None:
        super().__init__(profile, LocalChunkTransport(model, id2token))
        self._model = model

    @property
    def model(self):
        return self._model

    async def shutdown(self) -> None:
        import asyncio

        model = self._model
        if model is not None:
            await asyncio.to_thread(model.release)


class GrpcBackend(_PipelineBackend):
    """Distributed encode→decode через gRPC."""

    def __init__(
        self,
        profile: ModelProfile,
        encode_pool: EncodeClientPool,
        decode_pool: DecodeClientPool,
    ) -> None:
        super().__init__(
            profile,
            GrpcChunkTransport(profile, encode_pool, decode_pool),
        )
        self._encode_pool = encode_pool
        self._decode_pool = decode_pool

    @classmethod
    async def create(cls) -> "GrpcBackend":
        encode_targets = expand_targets(
            parse_targets("ENCODER_TARGETS", "encoder-0:50051")
        )
        decode_targets = expand_targets(
            parse_targets("DECODER_TARGETS", "decoder-0:50052")
        )
        if not encode_targets or not decode_targets:
            raise RuntimeError(
                "ENCODER_TARGETS and DECODER_TARGETS must be set for distributed runtime"
            )

        profile = ModelProfile.from_profile()
        encode_pool = EncodeClientPool(encode_targets)
        decode_pool = DecodeClientPool(decode_targets)
        await encode_pool.connect()
        await decode_pool.connect()
        return cls(profile, encode_pool, decode_pool)

    async def shutdown(self) -> None:
        if self._encode_pool is not None:
            await self._encode_pool.close()
        if self._decode_pool is not None:
            await self._decode_pool.close()


def runtime_mode() -> str:
    return os.environ.get("WHISPER_RUNTIME", "local").strip().lower()
