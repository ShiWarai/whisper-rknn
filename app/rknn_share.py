"""Общие веса RKNN между encoder-воркерами через ``rknn_dup_context``.

``RKNNLite`` всегда делает полный ``load_rknn`` + ``build_graph`` на сессию.
C API умеет дублировать контекст: веса общие, runtime / IO / внутренние
буферы — на воркера. Паттерн из Rockchip SDK и community
(Immich, rknn_dup_context gists).
"""

from __future__ import annotations

import ctypes
import gc
from typing import Any, Optional

try:
    from rknnlite.api import RKNNLite
    from rknnlite.api.rknn_runtime import RKNNRuntime
except ImportError:  # pragma: no cover
    RKNNLite = None  # type: ignore[misc, assignment]
    RKNNRuntime = None  # type: ignore[misc, assignment]

# На aarch64 rknn_context pointer-sized (значения больше 32 бит).
_RKNN_CONTEXT = ctypes.c_uint64

_RUNTIME_COPY_ATTRS = (
    "graph_is_built",
    "is_dynamic_shape",
    "support_dynamic",
    "npu_model",
    "rknn2precompile",
    "eval_memory",
    "perf_debug",
    "target_soc",
)

_lib: Optional[ctypes.CDLL] = None


def _resolve_librknn_path(base_runtime: Any = None) -> str:
    if base_runtime is not None:
        getter = getattr(base_runtime, "_get_rknn_api_lib_path", None)
        if callable(getter):
            try:
                path = getter()
                if path:
                    return str(path)
            except Exception:
                pass
    return "/usr/lib/librknnrt.so"


def _load_lib(base_runtime: Any = None) -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    path = _resolve_librknn_path(base_runtime)
    lib = ctypes.CDLL(path)
    lib.rknn_dup_context.argtypes = [
        ctypes.POINTER(_RKNN_CONTEXT),
        ctypes.POINTER(_RKNN_CONTEXT),
    ]
    lib.rknn_dup_context.restype = ctypes.c_int32
    lib.rknn_set_core_mask.argtypes = [_RKNN_CONTEXT, ctypes.c_uint32]
    lib.rknn_set_core_mask.restype = ctypes.c_int32
    _lib = lib
    return lib


def drop_rknn_model_bytes(session: Any) -> None:
    """
    Free the Python ``rknn_data`` copy kept after ``load_rknn``.

    RKNNLite reads the whole ``.rknn`` into RAM and keeps it even after
    ``init_runtime`` has copied weights into DMA/NPU memory. Clearing it
    roughly halves host RSS for large encoders.
    """
    if session is None:
        return
    data = getattr(session, "rknn_data", None)
    if data is None:
        return
    session.rknn_data = None
    del data
    gc.collect()


def dup_rknn_lite(base: Any, core_mask: int) -> Any:
    """
    Duplicate ``base`` RKNNLite context with shared weights, pin to ``core_mask``.

    Returns a new ``RKNNLite`` whose ``inference`` / ``release`` work like a
    normal session. Caller must ``release()`` duplicates before the base.
    """
    if RKNNLite is None or RKNNRuntime is None:
        raise RuntimeError("rknnlite is not installed")
    if base is None or getattr(base, "rknn_runtime", None) is None:
        raise ValueError("base RKNNLite session is not initialized")
    rt = base.rknn_runtime
    ctx_val = getattr(rt, "context", None)
    if ctx_val is None:
        raise RuntimeError("base RKNN runtime has no context")

    lib = _load_lib(rt)
    ctx_in = _RKNN_CONTEXT(int(ctx_val))
    ctx_out = _RKNN_CONTEXT(0)
    ret = int(lib.rknn_dup_context(ctypes.byref(ctx_in), ctypes.byref(ctx_out)))
    if ret != 0:
        raise RuntimeError(f"rknn_dup_context failed (ret={ret})")
    ret = int(lib.rknn_set_core_mask(ctx_out, int(core_mask)))
    if ret != 0:
        # По возможности уничтожить осиротевший dup-контекст.
        try:
            lib.rknn_destroy.argtypes = [_RKNN_CONTEXT]
            lib.rknn_destroy.restype = ctypes.c_int32
            lib.rknn_destroy(ctx_out)
        except Exception:
            pass
        raise RuntimeError(
            f"rknn_set_core_mask failed for dup (ret={ret}, mask={core_mask})"
        )

    clone = RKNNLite(verbose=False)
    clone.rknn_data = None
    clone.rknn_runtime = RKNNRuntime(
        root_dir=base.root_dir,
        target=None,
        device_id=None,
        async_mode=False,
        core_mask=core_mask,
    )
    clone.rknn_runtime.context = int(ctx_out.value)
    for attr in _RUNTIME_COPY_ATTRS:
        if hasattr(rt, attr):
            try:
                setattr(clone.rknn_runtime, attr, getattr(rt, attr))
            except Exception:
                pass
    return clone


def load_shared_encoder_sessions(
    encoder_path: str,
    core_masks: list[int],
    *,
    init_model,
    verbose: bool = False,
    log=print,
) -> list:
    """
    Загрузить один encoder и ``rknn_dup_context`` для остальных (маска на воркер).

    ``init_model(path, core_mask=...)`` должен вернуть инициализированный RKNNLite.
    """
    if not core_masks:
        raise ValueError("core_masks must be non-empty")
    sessions: list = []
    try:
        base = init_model(encoder_path, core_mask=core_masks[0])
        drop_rknn_model_bytes(base)
        sessions.append(base)
        for i, mask in enumerate(core_masks[1:], start=1):
            if verbose:
                log(f"encoder_pool worker {i}: dup context (mask={mask})", flush=True)
            sessions.append(dup_rknn_lite(base, mask))
        return sessions
    except Exception:
        # Сначала дубли, потом базовая сессия.
        for session in reversed(sessions):
            try:
                session.release()
            except Exception:
                pass
        raise
