"""Тесты чистых хелперов app.decode и app.audio_features."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio_features import (
    N_SAMPLES,
    _stft_power,
    _stft_power_loop,
    compute_features,
    pad_or_trim,
)
from app.decode import (
    _HAS_AV,
    _merge_short_tail_spans,
    _next_chunk_start,
    build_sot_sequence,
    causal_mask_1d,
    iter_audio_chunk_spans,
    iter_audio_chunks,
    load_audio_16k_mono,
    model_config_from_encoder_path,
    resample_linear,
    resolve_decode_token_limit,
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
    size_key, english_only, eot, n_layer, n_ctx, n_state, n_mels, mel_frames, no_ts, ts_begin = cfg
    assert size_key == "base"
    assert english_only is False
    assert eot == 50257
    assert n_layer == 6
    assert n_ctx == 448
    assert n_state == 512
    assert n_mels == 80
    assert mel_frames == 3000
    assert no_ts == 50363
    assert ts_begin == 50364
    sot = build_sot_sequence(
        size_key=size_key,
        english_only=english_only,
        task="transcribe",
        language="en",
        notimestamps_id=no_ts,
    )
    assert sot == [50258, 50259, 50359, 50363]


def test_model_config_turbo_profile(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL_PROFILE", "turbo")
    monkeypatch.setenv("WHISPER_LANGUAGE", "ru")
    cfg = model_config_from_encoder_path("/models/encoder.rknn")
    size_key, english_only, eot, n_layer, _, n_state, n_mels, _, no_ts, ts_begin = cfg
    assert size_key == "turbo"
    assert english_only is False
    assert eot == 50257
    assert n_layer == 4
    assert n_state == 1280
    assert n_mels == 128
    assert no_ts == 50364
    assert ts_begin == 50365
    sot = build_sot_sequence(
        size_key=size_key,
        english_only=english_only,
        task="transcribe",
        language="ru",
        notimestamps_id=no_ts,
    )
    assert sot == [50258, 50263, 50360, 50364]


def test_build_sot_sequence_translate_turbo(monkeypatch):
    monkeypatch.setenv("WHISPER_LANGUAGE", "ru")
    sot = build_sot_sequence(
        size_key="turbo",
        english_only=False,
        task="translate",
        language="ru",
        notimestamps_id=50364,
    )
    assert sot == [50258, 50263, 50359, 50364]


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
    # 47.7 s, window 30 s, overlap 2 s → hop 28 s → two windows, each ≤ 30 s
    n = int(47.7 * 16000)
    samples = np.arange(n, dtype=np.float32)
    window = 480_000
    overlap = 2 * 16000
    chunks = iter_audio_chunks(samples, window, overlap_samples=overlap)
    assert len(chunks) == 2
    assert len(chunks[0]) == window
    assert len(chunks[1]) == n - (window - overlap)
    assert all(len(c) <= window for c in chunks)
    hop = window - overlap
    np.testing.assert_array_equal(chunks[0][hop:], chunks[1][:overlap])


def test_resolve_decode_token_limit_defaults_to_ctx(monkeypatch):
    monkeypatch.delenv("WHISPER_MAX_DECODE_TOKENS", raising=False)
    assert resolve_decode_token_limit(448) == 448
    monkeypatch.setenv("WHISPER_MAX_DECODE_TOKENS", "auto")
    assert resolve_decode_token_limit(448) == 448
    monkeypatch.setenv("WHISPER_MAX_DECODE_TOKENS", "0")
    assert resolve_decode_token_limit(448) == 448
    monkeypatch.setenv("WHISPER_MAX_DECODE_TOKENS", "100")
    assert resolve_decode_token_limit(448) == 100
    monkeypatch.setenv("WHISPER_MAX_DECODE_TOKENS", "9999")
    assert resolve_decode_token_limit(448) == 448


def test_next_chunk_start_pulls_back_on_truncate():
    # Достаточно длинно, чтобы end-snap не сработал; truncate → переслушать последние 10 с.
    sr = 16000
    chunk = 480_000
    hop = 28 * sr
    overlap = 2 * sr
    n = 90 * sr
    nxt = _next_chunk_start(
        start=0,
        end=chunk,
        n=n,
        hop=hop,
        overlap_samples=overlap,
        chunk_samples=chunk,
        sample_rate=sr,
        truncated=True,
    )
    assert nxt == chunk - 10 * sr
    nxt_ok = _next_chunk_start(
        start=0,
        end=chunk,
        n=n,
        hop=hop,
        overlap_samples=overlap,
        chunk_samples=chunk,
        sample_rate=sr,
        truncated=False,
    )
    assert nxt_ok == hop


def test_merge_short_tail_replaces_tiny_final_window():
    window = 480_000
    n = 35 * 16000 + 9000  # последний span ~7.9 с при overlap=2
    samples = np.arange(n, dtype=np.float32)
    overlap = 2 * 16000
    spans = iter_audio_chunk_spans(samples, window, overlap_samples=overlap)
    merged = _merge_short_tail_spans(
        spans, samples, window, min_tail_samples=8 * 16000
    )
    assert len(merged) == len(spans)
    final_start, final_chunk = merged[-1]
    assert final_start == max(0, n - window)
    assert len(final_chunk) == n - final_start
    assert len(final_chunk) >= 8 * 16000 or final_start == 0


def test_stft_power_matches_loop_reference():
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(16000 * 3).astype(np.float32)
    ref = _stft_power_loop(audio)
    fast = _stft_power(audio)
    assert ref.shape == fast.shape
    np.testing.assert_allclose(ref, fast, rtol=0, atol=1e-5)

    clip = rng.standard_normal(8000).astype(np.float32)
    ref2 = _stft_power_loop(clip)
    fast2 = _stft_power(clip)
    np.testing.assert_allclose(ref2, fast2, rtol=0, atol=1e-5)


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
