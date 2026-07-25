"""Unit tests for pure helpers in app.decode and app.audio_features."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio_features import N_SAMPLES, compute_features, pad_or_trim
from app.decode import (
    _HAS_AV,
    causal_mask_1d,
    iter_audio_chunks,
    load_audio_16k_mono,
    model_config_from_encoder_path,
    resample_linear,
    stitch_transcripts,
    whisper_window_samples,
)
from app.whisper_languages import language_token_id


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


def test_pad_or_trim_short_pads():
    x = np.zeros(1000, dtype=np.float32)
    out = pad_or_trim(x)
    assert out.shape == (N_SAMPLES,)
    assert out[1000:].sum() == 0


def test_pad_or_trim_long_trims():
    x = np.ones(N_SAMPLES + 500, dtype=np.float32)
    out = pad_or_trim(x)
    assert out.shape == (N_SAMPLES,)


def test_compute_features_turbo_shape():
    samples = np.random.randn(16000 * 5).astype(np.float32)
    mel = compute_features(samples, n_mels=128, target_frames=3000)
    assert mel.shape == (1, 128, 3000)
    assert mel.dtype == np.float32


def test_compute_features_base_shape():
    samples = np.random.randn(16000 * 5).astype(np.float32)
    mel = compute_features(samples, n_mels=80, target_frames=3000)
    assert mel.shape == (1, 80, 3000)
    assert mel.dtype == np.float32


def test_language_token_id_ru_en():
    assert language_token_id("ru") == 50263
    assert language_token_id("en") == 50259


def _write_test_wav(path: str, sample_rate: int = 48000, duration_s: float = 0.1) -> None:
    import soundfile as sf

    n = max(1, int(round(sample_rate * duration_s)))
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float64)
    samples = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(path, samples, sample_rate)


def test_load_audio_16k_mono_from_wav_path(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_test_wav(str(wav), sample_rate=48000, duration_s=0.1)
    out = load_audio_16k_mono(str(wav))
    assert out.dtype == np.float32
    assert out.ndim == 1
    expected_len = max(1, int(round(0.1 * 16000)))
    assert abs(len(out) - expected_len) <= 2


@pytest.mark.skipif(not _HAS_AV, reason="PyAV not installed")
def test_load_audio_16k_mono_from_bytes(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_test_wav(str(wav), sample_rate=16000, duration_s=0.05)
    out = load_audio_16k_mono(wav.read_bytes(), format_hint=".wav")
    assert out.dtype == np.float32
    assert len(out) > 0
