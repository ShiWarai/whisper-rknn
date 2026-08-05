"""Тесты хелперов RKNN encoder с общими весами."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rknn_share import drop_rknn_model_bytes, load_shared_encoder_sessions


def test_drop_rknn_model_bytes_clears_payload():
    session = MagicMock()
    session.rknn_data = b"big-model"
    drop_rknn_model_bytes(session)
    assert session.rknn_data is None


def test_drop_rknn_model_bytes_noop_when_missing():
    drop_rknn_model_bytes(None)
    session = MagicMock(spec=[])
    drop_rknn_model_bytes(session)


def test_load_shared_encoder_sessions_dups_after_base():
    base = MagicMock(name="base")
    d1 = MagicMock(name="dup1")
    d2 = MagicMock(name="dup2")
    init = MagicMock(return_value=base)

    with (
        patch("app.rknn_share.drop_rknn_model_bytes") as drop,
        patch("app.rknn_share.dup_rknn_lite", side_effect=[d1, d2]) as dup,
    ):
        sessions = load_shared_encoder_sessions(
            "encoder.rknn",
            [1, 2, 4],
            init_model=init,
        )

    assert sessions == [base, d1, d2]
    init.assert_called_once_with("encoder.rknn", core_mask=1)
    drop.assert_called_once_with(base)
    assert dup.call_args_list[0].args == (base, 2)
    assert dup.call_args_list[1].args == (base, 4)


def test_load_shared_encoder_sessions_releases_on_dup_failure():
    base = MagicMock(name="base")
    init = MagicMock(return_value=base)

    with (
        patch("app.rknn_share.drop_rknn_model_bytes"),
        patch("app.rknn_share.dup_rknn_lite", side_effect=RuntimeError("dup failed")),
    ):
        with pytest.raises(RuntimeError, match="dup failed"):
            load_shared_encoder_sessions(
                "encoder.rknn",
                [1, 2],
                init_model=init,
            )

    base.release.assert_called_once()
