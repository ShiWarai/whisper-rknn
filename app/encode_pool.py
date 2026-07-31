"""Parallel RKNN encoder workers with per-core pinning (video-descriptor pattern)."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:  # pragma: no cover
    RKNNLite = None  # type: ignore[misc, assignment]


def _init_model(*args, **kwargs):
    from app.decode import init_model

    return init_model(*args, **kwargs)


@dataclass(frozen=True)
class EncodeJobResult:
    chunk_id: int
    cross_kv: Sequence[np.ndarray]
    encode_ms: float
    queue_wait_ms: float
    worker_id: int
    submitted_at: float
    started_at: float
    finished_at: float


def dedicated_npu_core_masks() -> Tuple[int, ...]:
    """
    Single-core NPU masks available in the installed rknnlite.

    Discovers ``NPU_CORE_0``, ``NPU_CORE_1``, … (power-of-two masks only),
    so RK3588 yields 3 today and future SoCs with more cores are picked up
    automatically when the runtime exposes them.
    """
    if RKNNLite is None:
        raise RuntimeError("rknnlite is not installed")
    masks: List[int] = []
    for i in range(0, 32):
        name = f"NPU_CORE_{i}"
        if not hasattr(RKNNLite, name):
            if i >= 3:
                break
            continue
        val = int(getattr(RKNNLite, name))
        # Skip AUTO(0) and multi-core OR-masks (not powers of two).
        if val > 0 and (val & (val - 1)) == 0:
            masks.append(val)
    if not masks:
        raise RuntimeError("rknnlite exposes no dedicated NPU_CORE_N masks")
    return tuple(masks)


def default_max_encoder_workers() -> int:
    """Ceiling for auto/forced workers: env override or discovered core count."""
    raw = os.environ.get("WHISPER_ENCODER_MAX_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, len(dedicated_npu_core_masks()))


def core_mask_for_worker(index: int, n_workers: Optional[int] = None) -> int:
    """
    Pin worker *index* to its own dedicated NPU core (never AUTO).

    ``n_workers`` is accepted for call-site compatibility; each worker still
    gets ``dedicated_masks[index]``.
    """
    del n_workers  # reserved / API compat
    masks = dedicated_npu_core_masks()
    if index < 0 or index >= len(masks):
        raise ValueError(
            f"encoder worker index {index} needs a dedicated core; "
            f"only {len(masks)} available: {masks}"
        )
    return masks[index]


def core_mask_name(mask: int) -> str:
    if RKNNLite is None:
        return str(mask)
    for i in range(0, 32):
        name = f"NPU_CORE_{i}"
        if hasattr(RKNNLite, name) and int(getattr(RKNNLite, name)) == mask:
            return name
    if hasattr(RKNNLite, "NPU_CORE_AUTO") and int(RKNNLite.NPU_CORE_AUTO) == mask:
        return "NPU_CORE_AUTO"
    return str(mask)


def _release_list(sessions: List) -> None:
    # Duplicated contexts share weights with sessions[0]; release dups first.
    for session in reversed(sessions):
        try:
            session.release()
        except Exception:
            pass


def _probe_npu_capacity_worker(
    encoder_path: str,
    n_workers: int,
    target_platform: str,
    n_mels: int,
    mel_time_frames: int,
) -> None:
    """
    Child process: load N encoders (shared weights) + one forward. Exit 0 if OK.

    Runs in a subprocess so an RKNN native abort cannot kill the API process.
    """
    import sys

    from app.rknn_share import load_shared_encoder_sessions

    sessions: List = []
    ok = False
    try:
        mel = np.zeros((1, int(n_mels), int(mel_time_frames)), dtype=np.float32)
        masks = [core_mask_for_worker(i, n_workers) for i in range(n_workers)]
        sessions = load_shared_encoder_sessions(
            encoder_path,
            masks,
            init_model=lambda path, core_mask: _init_model(
                path,
                target_platform=target_platform,
                core_mask=core_mask,
            ),
            verbose=False,
        )
        out = sessions[0].inference(inputs=[np.ascontiguousarray(mel)])
        if out is None:
            raise RuntimeError("warmup returned None")
        ok = True
    except Exception as exc:
        print(f"encoder_pool probe[{n_workers}]: {exc}", flush=True)
    finally:
        _release_list(sessions)
    sys.exit(0 if ok else 1)


def probe_npu_worker_count(
    encoder_path: str,
    *,
    max_workers: Optional[int] = None,
    target_platform: str = "rk3588",
    n_mels: int = 128,
    mel_time_frames: int = 3000,
    verbose: bool = True,
    join_timeout_s: float = 180.0,
) -> int:
    """Largest N in 1..max that survives load+warmup in a subprocess."""
    if max_workers is None:
        max_workers = default_max_encoder_workers()
    core_limit = len(dedicated_npu_core_masks())
    capped = max(1, min(int(max_workers), core_limit))
    ctx = mp.get_context("spawn")
    for try_n in range(capped, 0, -1):
        if verbose:
            print(f"encoder_pool: NPU probe {try_n} worker(s)…", flush=True)
        proc = ctx.Process(
            target=_probe_npu_capacity_worker,
            args=(encoder_path, try_n, target_platform, n_mels, mel_time_frames),
            name=f"encoder-probe-{try_n}",
        )
        proc.start()
        proc.join(timeout=join_timeout_s)
        if proc.is_alive():
            if verbose:
                print(f"encoder_pool: NPU probe {try_n} timed out — killing", flush=True)
            proc.kill()
            proc.join(timeout=15.0)
            time.sleep(0.5)
            continue
        if proc.exitcode == 0:
            if verbose:
                print(f"encoder_pool: NPU probe OK for {try_n} worker(s)", flush=True)
            return try_n
        if verbose:
            print(
                f"encoder_pool: NPU probe failed for {try_n} "
                f"(exit={proc.exitcode})",
                flush=True,
            )
        time.sleep(0.5)
    return 0


class EncoderPool:
    """Thread pool of RKNN encoder sessions with shared weights, one core each."""

    def __init__(
        self,
        encoder_path: str,
        *,
        n_workers: Optional[int] = None,
        target_platform: str = "rk3588",
        verbose: bool = True,
        n_mels: int = 128,
        mel_time_frames: int = 3000,
    ):
        if n_workers is None:
            n_workers = default_max_encoder_workers()
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        core_limit = len(dedicated_npu_core_masks())
        if n_workers > core_limit:
            raise ValueError(
                f"n_workers={n_workers} exceeds dedicated NPU cores ({core_limit})"
            )
        if RKNNLite is None:
            raise RuntimeError("rknnlite is not installed")
        self._encoder_path = encoder_path
        self._n_workers = 0
        self._queue: queue.Queue = queue.Queue()
        self._shutdown = threading.Event()
        self._workers: List[threading.Thread] = []
        self._sessions: List = []
        self._verbose = verbose

        fit = probe_npu_worker_count(
            encoder_path,
            max_workers=n_workers,
            target_platform=target_platform,
            n_mels=n_mels,
            mel_time_frames=mel_time_frames,
            verbose=verbose,
        )
        if fit < 1:
            raise RuntimeError(
                f"encoder_pool: NPU cannot fit any encoder worker for {encoder_path}"
            )
        if verbose and fit < n_workers:
            print(
                f"encoder_pool: reduced workers {n_workers} -> {fit} (NPU capacity)",
                flush=True,
            )

        self._sessions = self._load_sessions(fit, target_platform=target_platform)
        self._start_threads()
        self._n_workers = fit
        if verbose:
            print(f"encoder_pool: ready with {fit} worker(s)", flush=True)

    def _release_sessions(self, sessions: Optional[List] = None) -> None:
        _release_list(sessions if sessions is not None else self._sessions)
        if sessions is None:
            self._sessions.clear()

    def _teardown_sessions(self) -> None:
        self._release_sessions()
        self._workers.clear()

    def _load_sessions(self, n_workers: int, *, target_platform: str) -> List:
        """Load one RKNN model, dup contexts for the rest; threads after success."""
        from app.rknn_share import load_shared_encoder_sessions

        masks = [core_mask_for_worker(i, n_workers) for i in range(n_workers)]
        if self._verbose:
            for i, mask in enumerate(masks):
                kind = "load" if i == 0 else "dup"
                print(
                    f"encoder_pool worker {i}: {core_mask_name(mask)} ({mask}) [{kind}]",
                    flush=True,
                )
        return load_shared_encoder_sessions(
            self._encoder_path,
            masks,
            init_model=lambda path, core_mask: _init_model(
                path,
                target_platform=target_platform,
                core_mask=core_mask,
            ),
            verbose=self._verbose,
        )

    def _start_threads(self) -> None:
        self._workers.clear()
        for i, session in enumerate(self._sessions):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(i, session),
                name=f"encoder-worker-{i}",
                daemon=True,
            )
            thread.start()
            self._workers.append(thread)

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def _worker_loop(self, worker_id: int, session) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break
            chunk_id, mel, future, submitted_at = item
            started_at = time.perf_counter()
            queue_wait_ms = (started_at - submitted_at) * 1000.0
            try:
                arr = np.ascontiguousarray(np.asarray(mel), dtype=np.float32)
                cross_kv = session.inference(inputs=[arr])
                if cross_kv is None:
                    raise RuntimeError("encoder inference returned None")
                finished_at = time.perf_counter()
                future.set_result(
                    EncodeJobResult(
                        chunk_id=chunk_id,
                        cross_kv=cross_kv,
                        encode_ms=(finished_at - started_at) * 1000.0,
                        queue_wait_ms=queue_wait_ms,
                        worker_id=worker_id,
                        submitted_at=submitted_at,
                        started_at=started_at,
                        finished_at=finished_at,
                    )
                )
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self._queue.task_done()

    def submit(self, chunk_id: int, mel: np.ndarray) -> Future:
        future: Future = Future()
        submitted_at = time.perf_counter()
        self._queue.put((chunk_id, mel, future, submitted_at))
        return future

    def shutdown(self) -> None:
        self._shutdown.set()
        for _ in self._workers:
            self._queue.put(None)
        for thread in self._workers:
            thread.join(timeout=30.0)
        self._teardown_sessions()


def resolve_encoder_worker_count(
    encoder_path: str,
    *,
    requested: Optional[int] = None,
    decoder_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    credit_bytes: int = 0,
) -> int:
    """
    Resolve WHISPER_ENCODER_WORKERS.

    ``0`` / unset = auto (largest N that fits MemAvailable, capped by dedicated
    NPU core count / ``WHISPER_ENCODER_MAX_WORKERS``). EncoderPool then probes
    N→…→1 in a subprocess. Explicit N forces that probe ceiling.
    Returns 0 when auto finds nothing fits in RAM.
    """
    from app.system_memory import pick_encoder_worker_count

    if max_workers is None:
        max_workers = default_max_encoder_workers()
    core_limit = len(dedicated_npu_core_masks())
    max_workers = max(1, min(int(max_workers), core_limit))

    if requested is None:
        raw = os.environ.get("WHISPER_ENCODER_WORKERS", "0").strip()
        try:
            requested = int(raw)
        except ValueError:
            requested = 0
    if requested < 0:
        requested = 0
    if requested > max_workers:
        requested = max_workers
    if requested == 0:
        return pick_encoder_worker_count(
            encoder_path,
            decoder_path=decoder_path,
            max_workers=max_workers,
            credit_bytes=credit_bytes,
        )
    return requested
