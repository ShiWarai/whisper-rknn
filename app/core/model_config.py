"""Гиперпараметры модели для gateway и воркеров (без RKNN)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


def _infer_model_size_key(encoder_path: str, profile: Optional[str]) -> str:
    """
    Размер модели: WHISPER_MODEL_PROFILE / WHISPER_VARIANT, иначе подстрока в пути
    (tiny, base, small, medium, turbo).
    """
    p = (profile or "").strip().lower()
    if p:
        if p in ("tiny", "base", "small", "medium", "turbo"):
            return p
        raise ValueError(
            f"WHISPER_MODEL_PROFILE must be one of tiny,base,small,medium,turbo; got {profile!r}"
        )
    enc = encoder_path.lower()
    for k in ("tiny", "base", "small", "medium", "turbo"):
        if k in enc:
            return k
    raise ValueError(
        "Cannot infer model size from encoder path (expected tiny/base/small/medium/turbo in the "
        f"path, or set WHISPER_MODEL_PROFILE): {encoder_path!r}"
    )


def _english_only_flag(encoder_path: str, size_key: str) -> bool:
    raw = os.environ.get("WHISPER_ENGLISH_ONLY", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    if encoder_path and ".en" in encoder_path.lower():
        return True
    profile = (
        os.environ.get("WHISPER_MODEL_PROFILE")
        or os.environ.get("WHISPER_VARIANT")
        or ""
    ).lower()
    return ".en" in profile or profile.endswith("-en")


def _profile_tuple(
    size_key: str,
    *,
    english_only: bool,
) -> Tuple[str, bool, int, int, int, int, int, int, Optional[int], Optional[int]]:
    if english_only:
        notimestamps_id = 50362
        timestamp_begin = 50363
        eot = 50256
        n_mels, mel_time_frames = 80, 3000
    elif size_key == "turbo":
        notimestamps_id = 50364
        timestamp_begin = 50365
        eot = 50257
        n_mels, mel_time_frames = 128, 3000
    else:
        notimestamps_id = 50363
        timestamp_begin = 50364
        eot = 50257
        n_mels, mel_time_frames = 80, 3000

    if size_key == "tiny":
        n_text_layer, n_text_ctx, n_text_state = 4, 448, 384
    elif size_key == "base":
        n_text_layer, n_text_ctx, n_text_state = 6, 448, 512
    elif size_key == "small":
        n_text_layer, n_text_ctx, n_text_state = 12, 448, 768
    elif size_key == "medium":
        n_text_layer, n_text_ctx, n_text_state = 24, 448, 1024
    elif size_key == "turbo":
        n_text_layer, n_text_ctx, n_text_state = 4, 448, 1280
    else:
        raise ValueError(f"Unsupported model size: {size_key!r}")

    return (
        size_key,
        english_only,
        eot,
        n_text_layer,
        n_text_ctx,
        n_text_state,
        n_mels,
        mel_time_frames,
        notimestamps_id,
        timestamp_begin,
    )


def model_config_from_encoder_path(
    encoder_path: str,
    profile: Optional[str] = None,
):
    """Гиперпараметры декодера + форма mel для данного пути encoder .rknn."""
    prof_env = os.environ.get("WHISPER_MODEL_PROFILE") or os.environ.get("WHISPER_VARIANT")
    size_key = _infer_model_size_key(
        encoder_path,
        profile if profile is not None else prof_env,
    )
    english_only = _english_only_flag(encoder_path, size_key)
    return _profile_tuple(size_key, english_only=english_only)


@dataclass(frozen=True)
class ModelProfile:
    size_key: str
    english_only: bool
    eot: int
    n_text_layer: int
    n_text_ctx: int
    n_text_state: int
    n_mels: int
    mel_time_frames: int
    notimestamps_id: Optional[int]
    timestamp_begin: Optional[int]

    @classmethod
    def from_encoder_path(
        cls,
        encoder_path: str,
        profile: Optional[str] = None,
    ) -> "ModelProfile":
        return cls(*model_config_from_encoder_path(encoder_path, profile=profile))

    @classmethod
    def from_profile(cls, profile: Optional[str] = None) -> "ModelProfile":
        """Профиль только по env (gateway/decode без encoder.rknn на диске)."""
        prof_env = (
            profile
            or os.environ.get("WHISPER_MODEL_PROFILE")
            or os.environ.get("WHISPER_VARIANT")
        )
        encoder_path = os.environ.get("WHISPER_ENCODER", "")
        size_key = _infer_model_size_key(encoder_path, prof_env)
        english_only = _english_only_flag(encoder_path, size_key)
        return cls(*_profile_tuple(size_key, english_only=english_only))
