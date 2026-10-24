"""Точка входа gateway (совместимость): WHISPER_RUNTIME=distributed + api_server."""

from __future__ import annotations

import os

from app.api_server import main


def _run() -> None:
    os.environ.setdefault("WHISPER_RUNTIME", "distributed")
    main()


if __name__ == "__main__":
    _run()
