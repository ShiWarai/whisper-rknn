"""Сессия ONNX-декодера для decode-воркера."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.core.model_config import ModelProfile
from app.core.types import TaskType
from app.decode import build_sot_sequence
from app.onnx_decoder import OnnxDecoder, resolve_decoder_backend


class DecodeSession:
    """CPU ONNX-декодер без энкодера."""

    def __init__(
        self,
        decoder_path: str,
        *,
        decoder_backend: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> None:
        backend, resolved_decoder = resolve_decoder_backend(
            backend=decoder_backend,
            decoder_path=decoder_path,
        )
        if backend != "onnx":
            raise RuntimeError("decoder worker requires WHISPER_DECODER_BACKEND=onnx")
        self.profile = ModelProfile.from_profile(profile)
        self.decoder_backend = backend
        self.decoder_path = resolved_decoder
        self._onnx_decoder = OnnxDecoder(
            resolved_decoder,
            n_text_layer=self.profile.n_text_layer,
        )

    @property
    def size_key(self) -> str:
        return self.profile.size_key

    @property
    def english_only(self) -> bool:
        return self.profile.english_only

    @property
    def eot(self) -> int:
        return self.profile.eot

    @property
    def n_text_layer(self) -> int:
        return self.profile.n_text_layer

    @property
    def n_text_ctx(self) -> int:
        return self.profile.n_text_ctx

    @property
    def n_text_state(self) -> int:
        return self.profile.n_text_state

    @property
    def notimestamps_id(self) -> Optional[int]:
        return self.profile.notimestamps_id

    @property
    def timestamp_begin(self) -> Optional[int]:
        return self.profile.timestamp_begin

    def sot_sequence_for(
        self,
        *,
        task: TaskType = "transcribe",
        language: Optional[str] = None,
    ) -> List[int]:
        return build_sot_sequence(
            size_key=self.size_key,
            english_only=self.english_only,
            task=task,
            language=language,
            notimestamps_id=self.notimestamps_id,
        )

    def get_self_cache(self) -> List[np.ndarray]:
        self_cache: List[np.ndarray] = []
        batch_size = 1
        for _ in range(self.n_text_layer):
            k = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state),
                dtype=np.float32,
            )
            v = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state),
                dtype=np.float32,
            )
            self_cache.extend([k, v])
        return self_cache

    def run_decoder(self, tokens, self_kv, cross_kv, offset, mask):
        return self._onnx_decoder.run(tokens, self_kv, cross_kv, offset, mask)

    def release(self) -> None:
        if self._onnx_decoder is not None:
            self._onnx_decoder.release()
            self._onnx_decoder = None

    @property
    def ready(self) -> bool:
        return self._onnx_decoder is not None
