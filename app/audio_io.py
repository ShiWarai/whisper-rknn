"""Декодирование аудио в 16 kHz mono float32 (без RKNN)."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

try:
    import av

    _HAS_AV = True
except ImportError:
    av = None  # type: ignore[assignment]
    _HAS_AV = False

AudioSource = Union[str, Path, bytes, bytearray, BinaryIO]

_AV_FORMAT_BY_SUFFIX = {
    ".ogg": "ogg",
    ".opus": "ogg",
    ".mp3": "mp3",
    ".m4a": "mov",
    ".aac": "aac",
    ".webm": "webm",
    ".mp4": "mov",
    ".mkv": "matroska",
    ".wav": "wav",
    ".flac": "flac",
}


def _install_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples
    duration = len(samples) / float(src_sr)
    n_out = max(1, int(round(duration * dst_sr)))
    x_new = np.linspace(0.0, len(samples) - 1, num=n_out, dtype=np.float64)
    return np.interp(
        x_new,
        np.arange(len(samples), dtype=np.float64),
        samples.astype(np.float64),
    ).astype(np.float32)


def load_audio_wav(path: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.ascontiguousarray(data[:, 0])
    return mono, int(sample_rate)


def _av_format_from_hint(format_hint: Optional[str]) -> Optional[str]:
    if not format_hint:
        return None
    hint = format_hint.strip().lower()
    if hint == "bin":
        return None
    if not hint.startswith("."):
        hint = f".{hint}"
    return _AV_FORMAT_BY_SUFFIX.get(hint)


def _resolve_input_path(source: AudioSource) -> str:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {source}")
    return str(path)


def _resampled_chunks_to_mono_f32(resampler, frame) -> List[np.ndarray]:
    chunks: List[np.ndarray] = []
    for resampled in resampler.resample(frame):
        arr = resampled.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
        chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))
    return chunks


def _decode_container_to_16k_mono(container) -> np.ndarray:
    if not container.streams.audio:
        raise RuntimeError("No audio stream in container")

    resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks: List[np.ndarray] = []
    for frame in container.decode(audio=0):
        chunks.extend(_resampled_chunks_to_mono_f32(resampler, frame))
    chunks.extend(_resampled_chunks_to_mono_f32(resampler, None))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def _load_via_pyav(source: AudioSource, format_hint: Optional[str]) -> np.ndarray:
    if not _HAS_AV:
        raise RuntimeError("PyAV is not installed")

    av_format = _av_format_from_hint(format_hint)
    open_kwargs = {"format": av_format} if av_format else {}

    if isinstance(source, (bytes, bytearray)):
        with av.open(io.BytesIO(source), **open_kwargs) as container:
            return _decode_container_to_16k_mono(container)

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio not found: {source}")
        hint = format_hint or path.suffix
        av_format = _av_format_from_hint(hint)
        open_kwargs = {"format": av_format} if av_format else {}
        with av.open(str(path), **open_kwargs) as container:
            return _decode_container_to_16k_mono(container)

    if hasattr(source, "read"):
        with av.open(source, **open_kwargs) as container:
            return _decode_container_to_16k_mono(container)

    raise TypeError(f"Unsupported audio source type: {type(source)!r}")


def _materialize_source_to_path(
    source: AudioSource,
    cache_dir: Optional[Path],
    format_hint: Optional[str],
) -> Tuple[str, Optional[Path]]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio not found: {source}")
        return str(path), None

    cache = cache_dir or (_install_root() / ".cache")
    cache.mkdir(parents=True, exist_ok=True)
    suffix = ".bin"
    if format_hint:
        suffix = format_hint if format_hint.startswith(".") else f".{format_hint}"

    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="whisper_in_", dir=cache)
    os.close(fd)
    tmp_path = Path(tmp)

    if isinstance(source, (bytes, bytearray)):
        tmp_path.write_bytes(source)
    elif hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode()
        tmp_path.write_bytes(data)
        if hasattr(source, "seek"):
            source.seek(0)
    else:
        tmp_path.unlink(missing_ok=True)
        raise TypeError(f"Unsupported audio source type: {type(source)!r}")

    return str(tmp_path), tmp_path


def _load_via_soundfile_16k(path: str) -> np.ndarray:
    mono, sr = load_audio_wav(path)
    if sr != 16000:
        mono = resample_linear(mono, sr, 16000)
    return mono


def _load_via_ffmpeg_pipe(path: str) -> np.ndarray:
    ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        stderr = r.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed:\n{stderr}")
    if not r.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def load_audio_16k_mono(
    source: AudioSource,
    *,
    format_hint: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Decode audio to 16 kHz mono float32 in RAM.

    Primary path: PyAV (libav API, in-process).
    Fallbacks: soundfile (WAV/FLAC), CLI ffmpeg f32le pipe (last resort).
    """
    errors: List[str] = []

    if _HAS_AV:
        try:
            return _load_via_pyav(source, format_hint)
        except Exception as exc:
            errors.append(f"PyAV: {exc}")

    temp_path: Optional[Path] = None
    try:
        if isinstance(source, (str, Path)):
            path = _resolve_input_path(source)
        else:
            path, temp_path = _materialize_source_to_path(
                source, cache_dir, format_hint
            )
            if _HAS_AV:
                try:
                    return _load_via_pyav(path, format_hint)
                except Exception as exc:
                    errors.append(f"PyAV(file): {exc}")

        suffix = Path(path).suffix.lower()
        if suffix in (".wav", ".flac"):
            try:
                return _load_via_soundfile_16k(path)
            except Exception as exc:
                errors.append(f"soundfile: {exc}")

        try:
            return _load_via_ffmpeg_pipe(path)
        except Exception as exc:
            errors.append(f"ffmpeg: {exc}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    detail = "; ".join(errors) if errors else "no decoder available"
    raise RuntimeError(f"Cannot decode audio ({detail})")


def prepare_audio_16k_mono(
    input_path: str, cache_dir: Optional[Path] = None
) -> np.ndarray:
    """Алиас для совместимости с :func:`load_audio_16k_mono` (без временного WAV)."""
    return load_audio_16k_mono(input_path, cache_dir=cache_dir)
