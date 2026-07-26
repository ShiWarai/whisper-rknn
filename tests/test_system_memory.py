"""Unit tests for RAM forecast helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.system_memory import (
    BYTES_PER_FLOAT32,
    MODEL_HEADROOM,
    PCM_EXPANSION_FACTOR,
    REQUEST_OVERHEAD_BYTES,
    estimate_model_ram_bytes,
    estimate_pcm_bytes,
    estimate_request_ram_bytes,
    file_size_bytes,
    has_enough_ram,
    max_audio_seconds,
    read_mem_available_bytes,
)


def test_read_mem_available_bytes_positive_on_linux():
    avail = read_mem_available_bytes()
    if avail is not None:
        assert avail > 0


def test_has_enough_ram_refuses_when_tight():
    avail = read_mem_available_bytes()
    if avail is None:
        pytest.skip("MemAvailable unreadable")
    ok, reason = has_enough_ram(avail + 1, context="model RSS")
    assert ok is False
    assert "insufficient RAM" in reason
    assert "model RSS" in reason


def test_has_enough_ram_allows_when_unknown(monkeypatch):
    monkeypatch.setattr("app.system_memory.read_mem_available_bytes", lambda: None)
    ok, reason = has_enough_ram(10**12)
    assert ok is True
    assert reason == ""


def test_file_size_bytes_missing(tmp_path):
    assert file_size_bytes(tmp_path / "missing.rknn") is None


def test_estimate_model_ram_bytes_with_mock_sizes():
    with (
        patch("app.system_memory.file_size_bytes", side_effect=[1_000_000, 2_000_000]),
    ):
        need = estimate_model_ram_bytes("encoder.rknn", "decoder.rknn")
    expected = int(1_000_000 * MODEL_HEADROOM) + int(2_000_000 * MODEL_HEADROOM)
    assert need == expected


def test_estimate_pcm_bytes_capped_by_max_seconds(monkeypatch):
    monkeypatch.setenv("WHISPER_MAX_AUDIO_SECONDS", "10")
    assert max_audio_seconds() == 10
    huge_upload = 50 * 1024 * 1024
    pcm = estimate_pcm_bytes(huge_upload)
    assert pcm == 10 * 16_000 * BYTES_PER_FLOAT32


def test_estimate_pcm_bytes_from_upload_when_smaller(monkeypatch):
    monkeypatch.setenv("WHISPER_MAX_AUDIO_SECONDS", "600")
    upload = 100_000
    assert estimate_pcm_bytes(upload) == upload * PCM_EXPANSION_FACTOR


def test_estimate_request_ram_bytes_includes_components():
    need = estimate_request_ram_bytes(
        upload_bytes=1_000_000,
        n_mels=80,
        n_text_layer=4,
        n_text_ctx=448,
        n_text_state=384,
    )
    pcm = estimate_pcm_bytes(1_000_000)
    mel = 80 * 3000 * BYTES_PER_FLOAT32
    kv = 4 * 2 * 448 * 384 * BYTES_PER_FLOAT32
    assert need == 1_000_000 + pcm + mel + kv + REQUEST_OVERHEAD_BYTES
