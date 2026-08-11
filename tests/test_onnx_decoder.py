"""Тесты выбора backend ONNX decoder."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.onnx_decoder import resolve_decoder_backend


def test_resolve_decoder_backend_auto_prefers_onnx(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "decoder.onnx").write_bytes(b"onnx")
    (models / "decoder.rknn").write_bytes(b"rknn")

    backend, path = resolve_decoder_backend(
        backend="auto",
        models_dir=str(models),
    )
    assert backend == "onnx"
    assert path.endswith("decoder.onnx")


def test_resolve_decoder_backend_rknn_only(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "decoder.rknn").write_bytes(b"rknn")

    backend, path = resolve_decoder_backend(
        backend="auto",
        models_dir=str(models),
    )
    assert backend == "rknn"
    assert path.endswith("decoder.rknn")


def test_resolve_decoder_backend_onnx_missing_raises(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_decoder_backend(backend="onnx", models_dir=str(models))
