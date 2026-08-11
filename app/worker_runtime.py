"""Runtime-параметры encoder/decoder воркеров (не gateway: VAD там без pin)."""

from __future__ import annotations

import os
from typing import FrozenSet, Optional


def parse_cpu_affinity(spec: str) -> FrozenSet[int]:
    """
    Разобрать ``WHISPER_CPU_AFFINITY``: ``4,5,6`` или ``4-7`` (допускается смесь).
    """
    cpus: set[int] = set()
    for part in spec.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start_s, end_s = piece.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
            if start > end:
                start, end = end, start
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(piece))
    if not cpus:
        raise ValueError(f"WHISPER_CPU_AFFINITY is empty or invalid: {spec!r}")
    return frozenset(cpus)


def apply_worker_cpu_affinity() -> Optional[FrozenSet[int]]:
    """
    Привязать текущий процесс к CPU из ``WHISPER_CPU_AFFINITY``.

    Возвращает набор CPU или ``None``, если переменная не задана.
    """
    raw = os.environ.get("WHISPER_CPU_AFFINITY", "").strip()
    if not raw:
        return None
    cpus = parse_cpu_affinity(raw)
    os.sched_setaffinity(0, cpus)
    return cpus


def visible_cpu_count() -> int:
    """Число CPU, доступных процессу (с учётом affinity/cpuset)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        return max(1, int(os.cpu_count() or 1))


def resolve_onnx_intra_op_threads() -> int:
    """
    Потоки ONNX Runtime для decoder.

    ``WHISPER_ONNX_INTRA_OP_THREADS``: ``0``/``auto`` = все видимые CPU процесса;
    положительное число — явный потолок.
    """
    raw = os.environ.get("WHISPER_ONNX_INTRA_OP_THREADS", "0").strip().lower()
    if raw in ("", "0", "auto"):
        return max(1, visible_cpu_count())
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            f"Invalid WHISPER_ONNX_INTRA_OP_THREADS={raw!r}; use 0, auto, or a positive int"
        ) from exc
