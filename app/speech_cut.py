"""Нарезка аудио по паузам речи в RAM через Silero VAD (ONNX)."""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64
DEFAULT_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)


@dataclass(frozen=True)
class ChunkSpan:
    """Полуинтервал сэмплов ``[start, end)`` в 16 кГц mono PCM."""

    start: int
    end: int
    reason: str = ""

    @property
    def duration_s(self) -> float:
        return (self.end - self.start) / float(SAMPLE_RATE)


@dataclass
class VadCutTimings:
    vad_ms: float = 0.0
    cut_ms: float = 0.0


def _install_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_vad_model_path() -> Path:
    env = os.environ.get("WHISPER_VAD_MODEL", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    models = os.environ.get("WHISPER_MODELS_DIR", "").strip()
    if models:
        candidate = Path(models) / "silero_vad.onnx"
        if candidate.is_file():
            return candidate.resolve()
    return _install_root() / ".cache" / "silero_vad.onnx"


def ensure_vad_model(path: Optional[Path] = None) -> Path:
    target = path or resolve_vad_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 100_000:
        return target
    url = os.environ.get("WHISPER_VAD_MODEL_URL", DEFAULT_MODEL_URL).strip()
    urllib.request.urlretrieve(url, target)
    return target


_vad_session: Optional[SileroVadOnnx] = None


def preload_vad(*, model_path: Optional[Path] = None) -> SileroVadOnnx:
    """Загрузить Silero VAD ONNX один раз (при старте API). Идемпотентно."""
    global _vad_session
    if _vad_session is None:
        path = ensure_vad_model(model_path)
        _vad_session = SileroVadOnnx(path)
    return _vad_session


def get_vad_session() -> SileroVadOnnx:
    """Вернуть предзагруженную VAD-сессию; при пропуске preload — загрузить при первом вызове."""
    return preload_vad()


def release_vad() -> None:
    global _vad_session
    _vad_session = None


class SileroVadOnnx:
    """Минимальная обёртка Silero VAD ONNX (16 кГц, кадры по 512 сэмплов)."""

    def __init__(self, model_path: Path):
        if ort is None:
            raise RuntimeError("onnxruntime is required for Silero VAD")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        if chunk.shape[0] != WINDOW_SAMPLES:
            raise ValueError(f"need {WINDOW_SAMPLES} samples, got {chunk.shape[0]}")
        x = chunk.astype(np.float32, copy=False).reshape(1, WINDOW_SAMPLES)
        x = np.concatenate([self._context, x], axis=1)
        out, state = self._session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        self._state = state
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(out.reshape(-1)[0])

    def speech_probs(self, audio: np.ndarray) -> np.ndarray:
        self.reset()
        n = int(audio.shape[0])
        pad = (WINDOW_SAMPLES - (n % WINDOW_SAMPLES)) % WINDOW_SAMPLES
        if pad:
            audio = np.pad(audio, (0, pad))
        probs = [self.prob(audio[i : i + WINDOW_SAMPLES]) for i in range(0, audio.shape[0], WINDOW_SAMPLES)]
        return np.asarray(probs, dtype=np.float32)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def vad_config_from_env() -> dict:
    return {
        "max_sec": _env_float("WHISPER_MAX_CHUNK_SECONDS", 30.0),
        "search_back_sec": _env_float("WHISPER_VAD_SEARCH_BACK_SEC", 3.0),
        "threshold": _env_float("WHISPER_VAD_THRESHOLD", 0.5),
        "min_gap_ms": _env_float("WHISPER_VAD_MIN_GAP_MS", 250.0),
        "pad_sec": _env_float("WHISPER_VAD_CHUNK_PAD_SEC", 0.0),
    }


def find_cut_sample(
    probs: np.ndarray,
    *,
    start_sample: int,
    max_samples: int,
    search_back_samples: int,
    threshold: float,
    min_gap_frames: int,
) -> Tuple[int, str]:
    """Выбрать cut <= start+max, предпочитая самый длинный промежуток без речи в окне поиска."""
    total_samples = len(probs) * WINDOW_SAMPLES
    hard_end = min(start_sample + max_samples, total_samples)
    if hard_end <= start_sample:
        return start_sample, "empty"

    search_from = max(start_sample, hard_end - search_back_samples)
    f0 = search_from // WINDOW_SAMPLES
    f1 = max(f0 + 1, (hard_end + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES)
    f1 = min(f1, len(probs))

    best: Optional[Tuple[int, int]] = None
    gap_start: Optional[int] = None
    for fi in range(f0, f1):
        if probs[fi] < threshold:
            if gap_start is None:
                gap_start = fi
        elif gap_start is not None:
            gap_len = fi - gap_start
            if gap_len >= min_gap_frames:
                mid = gap_start + gap_len // 2
                if best is None or gap_len > best[0]:
                    best = (gap_len, mid)
            gap_start = None
    if gap_start is not None:
        gap_len = f1 - gap_start
        if gap_len >= min_gap_frames:
            mid = gap_start + gap_len // 2
            if best is None or gap_len > best[0]:
                best = (gap_len, mid)

    if best is not None:
        cut = min(hard_end, max(start_sample + 1, best[1] * WINDOW_SAMPLES))
        gap_s = best[0] * WINDOW_SAMPLES / SAMPLE_RATE
        return cut, f"vad_gap frames={best[0]} ({gap_s:.2f}s)"
    return hard_end, "fallback_hard_limit"


def iter_voice_aware_spans(
    audio: np.ndarray,
    probs: np.ndarray,
    *,
    max_sec: float,
    search_back_sec: float,
    threshold: float,
    min_gap_ms: float,
) -> List[ChunkSpan]:
    max_samples = int(round(max_sec * SAMPLE_RATE))
    search_back = int(round(search_back_sec * SAMPLE_RATE))
    min_gap_frames = max(1, int(round((min_gap_ms / 1000.0) * SAMPLE_RATE / WINDOW_SAMPLES)))
    spans: List[ChunkSpan] = []
    start = 0
    n = int(audio.shape[0])
    while start < n:
        remaining = n - start
        if remaining <= max_samples:
            spans.append(ChunkSpan(start, n, "tail"))
            break
        cut, reason = find_cut_sample(
            probs,
            start_sample=start,
            max_samples=max_samples,
            search_back_samples=search_back,
            threshold=threshold,
            min_gap_frames=min_gap_frames,
        )
        if cut <= start:
            cut = min(n, start + max_samples)
            reason = "forced_progress"
        spans.append(ChunkSpan(start, cut, reason))
        start = cut
    return spans


def plan_voice_aware_chunks(
    audio: np.ndarray,
    *,
    vad: Optional[SileroVadOnnx] = None,
    model_path: Optional[Path] = None,
    max_sec: Optional[float] = None,
    search_back_sec: Optional[float] = None,
    threshold: Optional[float] = None,
    min_gap_ms: Optional[float] = None,
) -> Tuple[List[ChunkSpan], np.ndarray, VadCutTimings]:
    """VAD + планирование span'ов для аудио уже в RAM."""
    cfg = vad_config_from_env()
    max_sec = max_sec if max_sec is not None else cfg["max_sec"]
    search_back_sec = search_back_sec if search_back_sec is not None else cfg["search_back_sec"]
    threshold = threshold if threshold is not None else cfg["threshold"]
    min_gap_ms = min_gap_ms if min_gap_ms is not None else cfg["min_gap_ms"]

    if vad is None:
        vad = get_vad_session()

    t0 = time.perf_counter()
    probs = vad.speech_probs(audio)
    vad_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    spans = iter_voice_aware_spans(
        audio,
        probs,
        max_sec=max_sec,
        search_back_sec=search_back_sec,
        threshold=threshold,
        min_gap_ms=min_gap_ms,
    )
    cut_ms = (time.perf_counter() - t1) * 1000.0
    return spans, probs, VadCutTimings(vad_ms=vad_ms, cut_ms=cut_ms)


def chunk_audio_views(audio: np.ndarray, spans: List[ChunkSpan]) -> List[np.ndarray]:
    """Вернуть view в RAM (без копии) для каждого span."""
    n = int(audio.shape[0])
    out: List[np.ndarray] = []
    for span in spans:
        end = min(span.end, n)
        start = max(0, span.start)
        out.append(audio[start:end])
    return out
