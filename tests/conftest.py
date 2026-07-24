"""Shared pytest fixtures: mock rknnlite before app imports."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

if "rknnlite" not in sys.modules:
    rknnlite = MagicMock()
    rknnlite.api = MagicMock()
    rknnlite.api.RKNNLite = MagicMock()
    rknnlite.api.RKNNLite.NPU_CORE_0 = 1
    sys.modules["rknnlite"] = rknnlite
    sys.modules["rknnlite.api"] = rknnlite.api
    sys.modules["rknnlite.api.rknn_runtime"] = MagicMock()
