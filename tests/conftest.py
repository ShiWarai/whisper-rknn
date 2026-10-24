"""Общие pytest-фикстуры: mock rknnlite до импорта app."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

if "rknnlite" not in sys.modules:
    rknnlite = MagicMock()
    rknnlite.api = MagicMock()
    rknnlite.api.RKNNLite = MagicMock()
    rknnlite.api.RKNNLite.NPU_CORE_0 = 1
    rknnlite.api.RKNNLite.NPU_CORE_1 = 2
    rknnlite.api.RKNNLite.NPU_CORE_2 = 4
    rknnlite.api.RKNNLite.NPU_CORE_0_1 = 3
    rknnlite.api.RKNNLite.NPU_CORE_0_1_2 = 7
    rknnlite.api.RKNNLite.NPU_CORE_ALL = 65535
    rknnlite.api.RKNNLite.NPU_CORE_AUTO = 0
    sys.modules["rknnlite"] = rknnlite
    sys.modules["rknnlite.api"] = rknnlite.api
    sys.modules["rknnlite.api.rknn_runtime"] = MagicMock()


@pytest.fixture(autouse=True)
def _turbo_profile_env(monkeypatch):
    """Профиль тестов/runtime по умолчанию как в проде: large-v3-turbo."""
    monkeypatch.setenv("WHISPER_MODEL_PROFILE", "turbo")
    monkeypatch.setenv("WHISPER_LANGUAGE", "ru")
