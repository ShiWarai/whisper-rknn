"""Сессия RKNN-энкодера для encoder-воркера: пул на все dedicated NPU cores."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.core.model_config import ModelProfile
from app.decode import apply_librknnrt_from_optional_path
from app.encode_pool import EncoderPool, resolve_encoder_worker_count


class EncodeSession:
    """RKNN encoder pool — по одному воркеру на ядро NPU, общие веса через rknn_dup_context."""

    def __init__(self, encoder_path: str) -> None:
        apply_librknnrt_from_optional_path(None)
        self.profile = ModelProfile.from_encoder_path(encoder_path)
        self.encoder_path = encoder_path
        n_workers = resolve_encoder_worker_count(encoder_path)
        if n_workers < 1:
            raise RuntimeError("encoder worker: no NPU workers fit (RAM/NPU probe)")
        self._pool = EncoderPool(
            encoder_path,
            n_workers=n_workers,
            verbose=True,
        )
        self.encoder_workers = self._pool.n_workers

    @property
    def ready(self) -> bool:
        return self._pool is not None and self._pool.n_workers > 0

    def encode(self, mel: np.ndarray) -> Sequence[np.ndarray]:
        future = self._pool.submit(0, mel)
        result = future.result()
        return result.cross_kv

    def release(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
