"""Тесты worker_runtime и фильтра NPU cores для encoder pool."""

from __future__ import annotations

import pytest

from app.encode_pool import allowed_npu_core_masks, core_mask_for_worker
from app.worker_runtime import parse_cpu_affinity, resolve_onnx_intra_op_threads


def test_parse_cpu_affinity_list():
    assert parse_cpu_affinity("4,5,6") == frozenset({4, 5, 6})


def test_parse_cpu_affinity_range():
    assert parse_cpu_affinity("4-7") == frozenset({4, 5, 6, 7})


def test_parse_cpu_affinity_mixed():
    assert parse_cpu_affinity("0,2-4") == frozenset({0, 2, 3, 4})


def test_allowed_npu_core_masks_respects_env(monkeypatch):
    monkeypatch.setattr(
        "app.encode_pool.dedicated_npu_core_masks",
        lambda: (1, 2, 4),
    )
    monkeypatch.setenv("WHISPER_NPU_CORE_MASK", "0_1")
    allowed = allowed_npu_core_masks()
    assert allowed == (1, 2)
    assert core_mask_for_worker(0) == 1
    assert core_mask_for_worker(1) == 2
    with pytest.raises(ValueError):
        core_mask_for_worker(2)


def test_resolve_onnx_threads_explicit(monkeypatch):
    monkeypatch.setenv("WHISPER_ONNX_INTRA_OP_THREADS", "2")
    assert resolve_onnx_intra_op_threads() == 2


def test_resolve_onnx_threads_auto(monkeypatch):
    monkeypatch.delenv("WHISPER_ONNX_INTRA_OP_THREADS", raising=False)
    assert resolve_onnx_intra_op_threads() >= 1
