"""CPU ONNX Runtime backend for Whisper decoder (RKNN encoder stays on NPU)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional at import time
    ort = None  # type: ignore[assignment]


class OnnxDecoder:
    """Run sherpa-exported Whisper decoder via ONNX Runtime (CPU)."""

    def __init__(self, model_path: str, *, n_text_layer: int):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed; add onnxruntime to requirements.txt"
            )
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ONNX decoder not found: {model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = max(1, int(__import__("os").cpu_count() or 1))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._n_text_layer = n_text_layer
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def release(self) -> None:
        self._session = None  # type: ignore[assignment]

    def _ordered_inputs(
        self,
        tokens: np.ndarray,
        self_kv: Sequence[np.ndarray],
        cross_kv: Sequence[np.ndarray],
        offset: np.ndarray,
        mask: np.ndarray,
    ) -> dict:
        if len(self_kv) != self._n_text_layer * 2:
            raise ValueError(
                f"expected {self._n_text_layer * 2} self_kv tensors, got {len(self_kv)}"
            )
        if len(cross_kv) != self._n_text_layer * 2:
            raise ValueError(
                f"expected {self._n_text_layer * 2} cross_kv tensors, got {len(cross_kv)}"
            )

        tokens = np.ascontiguousarray(tokens, dtype=np.int32)
        offset = np.ascontiguousarray(offset, dtype=np.int32)
        mask = np.ascontiguousarray(mask, dtype=np.int32)

        by_suffix = {
            "tokens": tokens,
            "offset": offset,
            "mask": mask,
        }
        for i in range(self._n_text_layer):
            by_suffix[f"self_k_{i}"] = np.ascontiguousarray(self_kv[i * 2], dtype=np.float32)
            by_suffix[f"self_v_{i}"] = np.ascontiguousarray(
                self_kv[i * 2 + 1], dtype=np.float32
            )
            by_suffix[f"cross_k_{i}"] = np.ascontiguousarray(
                cross_kv[i * 2], dtype=np.float32
            )
            by_suffix[f"cross_v_{i}"] = np.ascontiguousarray(
                cross_kv[i * 2 + 1], dtype=np.float32
            )

        feed: dict = {}
        for name in self._input_names:
            key = name
            if "-" in name:
                key = name.split("-", 1)[1]
            if key not in by_suffix:
                raise KeyError(
                    f"cannot map ONNX input {name!r} (suffix {key!r}); "
                    f"known: {sorted(by_suffix)}"
                )
            feed[name] = by_suffix[key]
        return feed

    def run(
        self,
        tokens: np.ndarray,
        self_kv: Sequence[np.ndarray],
        cross_kv: Sequence[np.ndarray],
        offset: np.ndarray,
        mask: np.ndarray,
    ) -> List[np.ndarray]:
        feed = self._ordered_inputs(tokens, self_kv, cross_kv, offset, mask)
        outputs = self._session.run(self._output_names, feed)
        return list(outputs)


def resolve_decoder_backend(
    backend: Optional[str] = None,
    decoder_path: Optional[str] = None,
    models_dir: Optional[str] = None,
) -> tuple[str, str]:
    """
    Resolve decoder backend and path.

    Returns ``(backend, path)`` where backend is ``onnx`` or ``rknn``.
    """
    import os

    raw = (backend or os.environ.get("WHISPER_DECODER_BACKEND", "auto")).strip().lower()
    dec = (decoder_path or os.environ.get("WHISPER_DECODER", "")).strip()

    models = Path(models_dir or os.environ.get("WHISPER_MODELS_DIR", "")).expanduser()
    onnx_candidates: List[Path] = []
    rknn_candidates: List[Path] = []

    if dec:
        p = Path(dec).expanduser()
        if p.suffix.lower() == ".onnx":
            onnx_candidates.append(p)
        elif p.suffix.lower() == ".rknn":
            rknn_candidates.append(p)
        else:
            onnx_candidates.append(p.with_suffix(".onnx"))
            rknn_candidates.append(p)

    if models.is_dir():
        onnx_candidates.append(models / "decoder.onnx")
        rknn_candidates.append(models / "decoder.rknn")

    onnx_path = next((p for p in onnx_candidates if p.is_file()), None)
    rknn_path = next((p for p in rknn_candidates if p.is_file()), None)

    if raw == "onnx":
        if onnx_path is None:
            raise FileNotFoundError("WHISPER_DECODER_BACKEND=onnx but decoder.onnx not found")
        return "onnx", str(onnx_path.resolve())
    if raw == "rknn":
        if rknn_path is None:
            raise FileNotFoundError("WHISPER_DECODER_BACKEND=rknn but decoder.rknn not found")
        return "rknn", str(rknn_path.resolve())

    # auto
    if onnx_path is not None:
        return "onnx", str(onnx_path.resolve())
    if rknn_path is not None:
        return "rknn", str(rknn_path.resolve())
    raise FileNotFoundError("No decoder.onnx or decoder.rknn found")
