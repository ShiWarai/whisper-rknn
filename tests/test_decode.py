"""Unit tests for pure helpers in app.decode."""

from __future__ import annotations

import numpy as np
import pytest

from app.decode import (
    causal_mask_1d,
    iter_audio_chunks,
    model_config_from_encoder_path,
    resample_linear,
    stitch_transcripts,
    whisper_window_samples,
)


def test_resample_linear_same_rate():
    samples = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    out = resample_linear(samples, 16000, 16000)
    np.testing.assert_array_equal(out, samples)


def test_resample_linear_downsample():
    samples = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    out = resample_linear(samples, 16000, 8000)
    assert out.dtype == np.float32
    assert len(out) == 2


def test_causal_mask_1d():
    mask = causal_mask_1d(2, 5)
    assert mask.tolist() == [0, 0, 1, 1, 1]


def test_model_config_base_from_path(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL_PROFILE", raising=False)
    monkeypatch.delenv("WHISPER_VARIANT", raising=False)
    monkeypatch.setenv("WHISPER_LANGUAGE", "en")
    cfg = model_config_from_encoder_path("/models/base-encoder.rknn")
    sot, eot, n_layer, n_ctx, n_state, n_mels, mel_frames = cfg
    assert sot == [50258, 50259, 50359, 50363]
    assert eot == 50257
    assert n_layer == 6
    assert n_ctx == 448
    assert n_state == 512
    assert n_mels == 80
    assert mel_frames == 3000


def test_model_config_turbo_profile(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL_PROFILE", "turbo")
    monkeypatch.setenv("WHISPER_LANGUAGE", "ru")
    cfg = model_config_from_encoder_path("/models/encoder.rknn")
    sot, eot, n_layer, _, n_state, n_mels, _ = cfg
    assert sot == [50258, 50263, 50360, 50364]  # ru + turbo specials
    assert eot == 50257
    assert n_layer == 4
    assert n_state == 1280
    assert n_mels == 128


def test_model_config_invalid_profile():
    with pytest.raises(ValueError, match="WHISPER_MODEL_PROFILE"):
        model_config_from_encoder_path("/models/encoder.rknn", profile="xlarge")


def test_whisper_window_samples_30s():
    assert whisper_window_samples(16000, 3000) == 480_000


def test_iter_audio_chunks_short_single():
    samples = np.zeros(1000, dtype=np.float32)
    chunks = iter_audio_chunks(samples, 480_000)
    assert len(chunks) == 1
    assert len(chunks[0]) == 1000


def test_iter_audio_chunks_splits_long():
    # 65 s @ 16 kHz, no overlap → 3 windows (30 + 30 + 5)
    samples = np.arange(65 * 16000, dtype=np.float32)
    chunks = iter_audio_chunks(samples, 480_000, overlap_samples=0)
    assert len(chunks) == 3
    assert len(chunks[0]) == 480_000
    assert len(chunks[1]) == 480_000
    assert len(chunks[2]) == 5 * 16000
    joined = np.concatenate(chunks)
    np.testing.assert_array_equal(joined, samples)


def test_iter_audio_chunks_overlap_keeps_window_size():
    # 47.7 s, window 30 s, overlap 5 s → hop 25 s → two windows, each ≤ 30 s
    n = int(47.7 * 16000)
    samples = np.arange(n, dtype=np.float32)
    window = 480_000
    overlap = 5 * 16000
    chunks = iter_audio_chunks(samples, window, overlap_samples=overlap)
    assert len(chunks) == 2
    assert len(chunks[0]) == window
    assert len(chunks[1]) == n - (window - overlap)
    assert all(len(c) <= window for c in chunks)
    # overlap region matches
    hop = window - overlap
    np.testing.assert_array_equal(chunks[0][hop:], chunks[1][:overlap])


def test_stitch_transcripts_dedupes_overlap():
    left = "привет проверка звука я хочу понять насколько"
    right = "я хочу понять насколько тяжело склеить сообщения"
    assert stitch_transcripts([left, right]) == (
        "привет проверка звука я хочу понять насколько тяжело склеить сообщения"
    )
