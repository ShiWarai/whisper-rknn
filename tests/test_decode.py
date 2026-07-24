"""Unit tests for pure helpers in app.decode."""

from __future__ import annotations

import numpy as np
import pytest

from app.decode import (
    causal_mask_1d,
    model_config_from_encoder_path,
    resample_linear,
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


def test_model_config_base_from_path():
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
    cfg = model_config_from_encoder_path("/models/encoder.rknn")
    _, _, n_layer, _, n_state, n_mels, _ = cfg
    assert n_layer == 4
    assert n_state == 1280
    assert n_mels == 128


def test_model_config_invalid_profile():
    with pytest.raises(ValueError, match="WHISPER_MODEL_PROFILE"):
        model_config_from_encoder_path("/models/encoder.rknn", profile="xlarge")
