"""Тесты привязки ядер encoder pool и полей таймингов."""

from __future__ import annotations

import pytest

from app.encode_pool import allowed_npu_core_masks, core_mask_for_worker, core_mask_name, dedicated_npu_core_masks


def test_dedicated_masks_are_single_cores():
    masks = dedicated_npu_core_masks()
    assert len(masks) >= 1
    for mask in masks:
        assert mask > 0
        assert (mask & (mask - 1)) == 0  # степень двойки


def test_core_mask_for_worker_never_auto():
    masks = allowed_npu_core_masks()
    for i in range(len(masks)):
        mask = core_mask_for_worker(i, n_workers=len(masks))
        assert mask == masks[i]
        assert "AUTO" not in core_mask_name(mask)


def test_core_mask_for_worker_two_workers_uses_core0_and_core1():
    """Регрессия: при N=2 последний воркер не должен получать AUTO."""
    masks = allowed_npu_core_masks()
    if len(masks) < 2:
        pytest.skip("need at least 2 dedicated NPU cores in mock/runtime")
    assert core_mask_for_worker(0, 2) == masks[0]
    assert core_mask_for_worker(1, 2) == masks[1]


def test_core_mask_out_of_range():
    masks = allowed_npu_core_masks()
    with pytest.raises(ValueError):
        core_mask_for_worker(len(masks), n_workers=len(masks) + 1)
