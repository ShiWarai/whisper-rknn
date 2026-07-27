#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
# Ядро декодирования Whisper RKNN (RK3588); вызывается из app.api_server.

"""
Whisper RKNN: fbank -> encoder/decoder RKNN -> text.
Аудио: PyAV (libav in-process) -> 16 kHz mono float32 в RAM.
Fallback: soundfile (WAV/FLAC), CLI ffmpeg (f32le pipe).
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from app.audio_features import compute_features
from app.whisper_languages import language_token_id

TaskType = Literal["transcribe", "translate"]


@dataclass
class TranscriptSegment:
    """Timed transcript span (seconds from start of utterance)."""

    start: float
    end: float
    text: str


@dataclass
class DecodeResult:
    text: str
    segments: Optional[List[TranscriptSegment]] = None

try:
    import av

    _HAS_AV = True
except ImportError:
    av = None  # type: ignore[assignment]
    _HAS_AV = False

try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("Install rknn_toolkit_lite2 (см. Docker-сборку и каталог third_party/*.whl).")
    raise


def _install_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_librknnrt_path(cli_path: Optional[str]) -> Optional[Path]:
    """Путь к librknnrt для патча rknnlite. None = искать в системе (обычно /usr/lib)."""
    if cli_path:
        p = Path(cli_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"librknnrt path is not a file: {cli_path}")
        return p.resolve()

    env = os.environ.get("LIBRKNNRT_SO") or os.environ.get("SHERPA_RKNNRT_SO")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
    return None


def apply_rknnrt_path_override(so_path: Path) -> None:
    import rknnlite.api.rknn_runtime as rr

    resolved = str(so_path)
    _orig = rr.RKNNRuntime._get_rknn_api_lib_path

    def _patched(self):
        if os.path.isfile(resolved):
            return resolved
        return _orig(self)

    rr.RKNNRuntime._get_rknn_api_lib_path = _patched


def causal_mask_1d(n: int, L: int):
    mask = np.ones((L,), dtype=np.int32)
    if n > 0:
        mask[:n] = 0
    return mask


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
    """Backward-compatible alias for :func:`load_audio_16k_mono` (no temp WAV)."""
    return load_audio_16k_mono(input_path, cache_dir=cache_dir)


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def resolve_npu_core_mask(name: Optional[str] = None) -> int:
    """
    Map WHISPER_NPU_CORE_MASK to RKNNLite constants.
    Default: NPU_CORE_0_1_2 (all three cores on RK3588).
    """
    raw = (name if name is not None else os.environ.get("WHISPER_NPU_CORE_MASK", "0_1_2")).strip()
    aliases = {
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "0_1": "NPU_CORE_0_1",
        "0_1_2": "NPU_CORE_0_1_2",
        "all": "NPU_CORE_ALL",
        "auto": "NPU_CORE_AUTO",
        "NPU_CORE_0": "NPU_CORE_0",
        "NPU_CORE_1": "NPU_CORE_1",
        "NPU_CORE_2": "NPU_CORE_2",
        "NPU_CORE_0_1": "NPU_CORE_0_1",
        "NPU_CORE_0_1_2": "NPU_CORE_0_1_2",
        "NPU_CORE_ALL": "NPU_CORE_ALL",
        "NPU_CORE_AUTO": "NPU_CORE_AUTO",
    }
    key = aliases.get(raw, aliases.get(raw.upper()))
    if key is None:
        raise ValueError(
            f"Unknown WHISPER_NPU_CORE_MASK={raw!r}; use 0, 0_1, 0_1_2, all, auto"
        )
    return int(getattr(RKNNLite, key))


def init_model(filename, target_platform="rk3588", core_mask: Optional[int] = None):
    if not Path(filename).is_file():
        raise FileNotFoundError(f"{filename} does not exist")

    if core_mask is None:
        core_mask = resolve_npu_core_mask()

    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(path=filename)
    if ret != 0:
        raise RuntimeError(f"Load model {filename} failed!")

    ret = rknn_lite.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(
            f"Failed to init rknn runtime for {filename} (core_mask={core_mask})"
        )
    return rknn_lite


def _default_language() -> str:
    return os.environ.get("WHISPER_LANGUAGE", "ru").strip() or "ru"


def _task_token_id(size_key: str, task: TaskType) -> int:
    if size_key == "turbo":
        return 50359 if task == "translate" else 50360
    return 50358 if task == "translate" else 50359


def build_sot_sequence(
    *,
    size_key: str,
    english_only: bool,
    task: TaskType = "transcribe",
    language: Optional[str] = None,
    timestamps: bool = False,
    notimestamps_id: Optional[int] = None,
) -> List[int]:
    """Build Whisper decoder prompt tokens for a single request."""
    if english_only:
        if task == "translate":
            raise ValueError(
                "Translation is not supported with English-only models; "
                "use a multilingual model."
            )
        seq = [50257, notimestamps_id]
    elif size_key == "turbo":
        lang_id = _language_token_id(language or _default_language())
        task_id = _task_token_id(size_key, task)
        seq = [50258, lang_id, task_id, notimestamps_id]
    else:
        lang_id = _language_token_id(language or _default_language())
        task_id = _task_token_id(size_key, task)
        seq = [50258, lang_id, task_id, notimestamps_id]

    if (
        timestamps
        and notimestamps_id is not None
        and seq
        and seq[-1] == notimestamps_id
    ):
        return seq[:-1]
    return seq


class RKNNModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        size_key: str,
        english_only: bool,
        eot: int,
        n_text_layer: int,
        n_text_ctx: int,
        n_text_state: int,
        n_mels: int = 80,
        mel_time_frames: int = 3000,
        notimestamps_id: Optional[int] = None,
        timestamp_begin: Optional[int] = None,
        target_platform="rk3588",
        verbose: bool = True,
    ):
        self.size_key = size_key
        self.english_only = english_only
        self.eot = eot
        self.n_text_layer = n_text_layer
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state
        self.n_mels = n_mels
        self.mel_time_frames = mel_time_frames
        self.notimestamps_id = notimestamps_id
        self.timestamp_begin = timestamp_begin

        core_mask = resolve_npu_core_mask()
        if verbose:
            print("model_size", self.size_key, "english_only", self.english_only)
            print("eot", self.eot)
            print("timestamp_begin", self.timestamp_begin)
            print("npu_core_mask", core_mask, os.environ.get("WHISPER_NPU_CORE_MASK", "0_1_2"))

        self.encoder = init_model(encoder, core_mask=core_mask)
        self.decoder = init_model(decoder, core_mask=core_mask)

    def sot_sequence_for(
        self,
        timestamps: bool,
        *,
        task: TaskType = "transcribe",
        language: Optional[str] = None,
    ) -> List[int]:
        """Prompt tokens for the current request."""
        return build_sot_sequence(
            size_key=self.size_key,
            english_only=self.english_only,
            task=task,
            language=language,
            timestamps=timestamps,
            notimestamps_id=self.notimestamps_id,
        )

    def release(self):
        self.encoder.release()
        self.decoder.release()

    def run_encoder(self, x):
        arr = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
        return self.encoder.inference(inputs=[arr])

    def get_self_cache(self) -> List[np.ndarray]:
        self_cache = []
        batch_size = 1
        for i in range(self.n_text_layer):
            k = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state), dtype=np.float32
            )
            v = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state), dtype=np.float32
            )
            self_cache.extend([k, v])
        return self_cache

    def run_decoder(self, tokens: np.ndarray, self_kv, cross_kv, offset, mask):
        return self.decoder.inference(
            inputs=[tokens] + self_kv + cross_kv + [offset, mask]
        )


def whisper_window_samples(sample_rate: int = 16000, mel_time_frames: int = 3000) -> int:
    """Samples per model window (~30 s for mel_time_frames=3000 @ 100 frames/s)."""
    seconds = float(mel_time_frames) / 100.0
    return max(1, int(round(seconds * sample_rate)))


def iter_audio_chunk_spans(
    samples: np.ndarray,
    chunk_samples: int,
    overlap_samples: int = 0,
) -> List[Tuple[int, np.ndarray]]:
    """
    Sliding windows of length ``chunk_samples`` (RKNN mel window, e.g. 30 s / 3000 frames).

    Returns ``(start_sample, chunk)`` so absolute timestamps can be recovered.
    ``overlap_samples`` advances the window by ``chunk_samples - overlap``.
    The last window may be shorter (feature pad / pad_or_trim fills the rest).
    """
    n = int(samples.shape[0])
    if n <= 0:
        return [(0, samples)]
    if n <= chunk_samples:
        return [(0, samples)]

    overlap_samples = int(max(0, min(overlap_samples, chunk_samples - 1)))
    hop = chunk_samples - overlap_samples
    spans: List[Tuple[int, np.ndarray]] = []
    start = 0
    while start < n:
        end = min(start + chunk_samples, n)
        spans.append((start, samples[start:end]))
        if end >= n:
            break
        start += hop
        # Snap final window to the end so a tiny tail is not a near-duplicate hop.
        if start < n and start + chunk_samples >= n and n - start < hop:
            final_start = max(0, n - chunk_samples)
            if final_start > start:
                start = final_start
    return spans


def iter_audio_chunks(
    samples: np.ndarray,
    chunk_samples: int,
    overlap_samples: int = 0,
) -> List[np.ndarray]:
    """Sliding windows of length ``chunk_samples`` (see :func:`iter_audio_chunk_spans`)."""
    return [chunk for _, chunk in iter_audio_chunk_spans(samples, chunk_samples, overlap_samples)]


def stitch_transcripts(parts: List[str]) -> str:
    """Join chunk texts; drop duplicated overlap (longest word suffix/prefix match)."""
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    out = cleaned[0]
    for nxt in cleaned[1:]:
        out = _merge_overlap_text(out, nxt)
    return out.strip()


def _merge_overlap_text(left: str, right: str) -> str:
    """Append ``right`` to ``left``, removing the longest shared word boundary."""
    lw = left.split()
    rw = right.split()
    if not lw:
        return right
    if not rw:
        return left

    max_k = min(len(lw), len(rw), 48)

    def _norm(w: str) -> str:
        return w.lower().strip(".,!?;:«»\"'()[]")

    best = 0
    for k in range(max_k, 0, -1):
        if [_norm(x) for x in lw[-k:]] == [_norm(x) for x in rw[:k]]:
            best = k
            break
    if best:
        return " ".join(lw + rw[best:])
    return left + " " + right


def _tokens_to_text(ans: List[int], id2token: dict) -> str:
    pieces = []
    for i in ans:
        if i in id2token:
            pieces.append(base64.b64decode(id2token[i]))
    return b"".join(pieces).decode().strip()


def parse_timestamp_tokens(
    token_ids: List[int],
    id2token: dict,
    timestamp_begin: int,
    *,
    time_offset: float = 0.0,
) -> Tuple[str, List[TranscriptSegment]]:
    """
    Split Whisper decoder ids into segment spans.

    Expected pattern: ``<|t0|> text… <|t1|>`` (optionally consecutive timestamps
    for the next start). Id ``timestamp_begin + k`` → ``k * 0.02`` seconds.
    """

    def is_ts(tid: int) -> bool:
        return tid >= timestamp_begin

    def ts_sec(tid: int) -> float:
        return (tid - timestamp_begin) * 0.02

    segments: List[TranscriptSegment] = []
    i = 0
    n = len(token_ids)
    while i < n:
        if not is_ts(token_ids[i]):
            i += 1
            continue
        start = ts_sec(token_ids[i])
        i += 1
        content: List[int] = []
        while i < n and not is_ts(token_ids[i]):
            content.append(token_ids[i])
            i += 1
        if i >= n:
            text = _tokens_to_text(content, id2token)
            if text:
                abs_start = round(start + time_offset, 3)
                segments.append(
                    TranscriptSegment(start=abs_start, end=abs_start, text=text)
                )
            break
        end = ts_sec(token_ids[i])
        text = _tokens_to_text(content, id2token)
        if text:
            segments.append(
                TranscriptSegment(
                    start=round(start + time_offset, 3),
                    end=round(end + time_offset, 3),
                    text=text,
                )
            )
        # Consecutive timestamp after end is the next segment start — keep it.
        if i + 1 < n and is_ts(token_ids[i + 1]):
            i += 1
        else:
            i += 1

    full = " ".join(seg.text for seg in segments).strip()
    if not full:
        text_ids = [t for t in token_ids if t < timestamp_begin]
        full = _tokens_to_text(text_ids, id2token)
    return full, segments


def decode_samples(
    model: RKNNModel,
    id2token: dict,
    samples: np.ndarray,
    verbose: bool = True,
    *,
    timestamps: bool = False,
    time_offset: float = 0.0,
    task: TaskType = "transcribe",
    language: Optional[str] = None,
) -> DecodeResult:
    """Decode one window of 16 kHz mono float samples (≤ model window; shorter is padded)."""
    features = compute_features(
        samples,
        n_mels=model.n_mels,
        target_frames=model.mel_time_frames,
    )
    if verbose:
        print(features.shape)
    cross_kv = model.run_encoder(features)
    self_kv = model.get_self_cache()

    sot = model.sot_sequence_for(timestamps, task=task, language=language)
    offset = np.array([0], dtype=np.int32)
    out = None
    for t in sot:
        token = np.array([[t]], dtype=np.int32)
        mask = causal_mask_1d(offset.item(), model.n_text_ctx)
        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )
        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]
        offset += 1

    assert out is not None
    logits = np.asarray(out[0][0, 0], dtype=np.float32).copy()
    # Force first emitted token to be a timestamp when timestamps are requested.
    if timestamps and model.timestamp_begin is not None:
        logits[: model.timestamp_begin] = -1e9
    idx = int(logits.argmax())
    ans: List[int] = []
    max_ngram_repeats = int(os.environ.get("WHISPER_MAX_NGRAM_REPEAT", "6"))
    max_tokens = int(
        os.environ.get("WHISPER_MAX_DECODE_TOKENS", "150" if timestamps else "100")
    )

    def _repeating_tail(tokens: List[int]) -> bool:
        """Stop if the same 1–4 token pattern repeats at the end."""
        for n in (1, 2, 3, 4):
            need = n * max_ngram_repeats
            if len(tokens) < need:
                continue
            pattern = tokens[-n:]
            ok = True
            for r in range(1, max_ngram_repeats):
                start = len(tokens) - (r + 1) * n
                if tokens[start : start + n] != pattern:
                    ok = False
                    break
            if ok:
                return True
        return False

    while idx != model.eot and offset.item() < max_tokens:
        ans.append(int(idx))
        if _repeating_tail(ans):
            for n in (1, 2, 3, 4):
                need = n * max_ngram_repeats
                if len(ans) >= need and ans[-n:] * max_ngram_repeats == ans[-need:]:
                    ans = ans[: len(ans) - n * (max_ngram_repeats - 1)]
                    break
            break
        token = np.array([[idx]], dtype=np.int32)
        mask = causal_mask_1d(offset.item(), model.n_text_ctx)
        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )
        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]
        offset += 1
        idx = int(out[0][0, 0].argmax())

    if verbose:
        print(ans)

    segments: Optional[List[TranscriptSegment]] = None
    if timestamps and model.timestamp_begin is not None:
        text, segments = parse_timestamp_tokens(
            ans, id2token, model.timestamp_begin, time_offset=time_offset
        )
    else:
        text_ids = ans
        if model.timestamp_begin is not None:
            text_ids = [t for t in ans if t < model.timestamp_begin]
        text = _tokens_to_text(text_ids, id2token)

    if verbose:
        print(text)
    return DecodeResult(text=text, segments=segments)


def _utterance_chunk_spans(model: RKNNModel, samples: np.ndarray) -> Tuple[List[Tuple[int, np.ndarray]], int, int]:
    """Return chunk spans plus hop/overlap sample counts for timestamp trimming."""
    sample_rate = 16000
    chunk_samples = whisper_window_samples(sample_rate, model.mel_time_frames)
    env_sec = os.environ.get("WHISPER_CHUNK_SECONDS", "").strip()
    if env_sec:
        try:
            sec = float(env_sec)
            if sec > 0:
                chunk_samples = min(
                    chunk_samples, max(1, int(round(sec * sample_rate)))
                )
        except ValueError:
            pass

    overlap_samples = 0
    env_ov = os.environ.get("WHISPER_CHUNK_OVERLAP_SECONDS", "5").strip()
    if env_ov:
        try:
            ov_sec = float(env_ov)
            if ov_sec > 0:
                overlap_samples = min(
                    chunk_samples - 1,
                    max(0, int(round(ov_sec * sample_rate))),
                )
        except ValueError:
            pass

    spans = iter_audio_chunk_spans(
        samples, chunk_samples, overlap_samples=overlap_samples
    )
    hop = chunk_samples - overlap_samples if overlap_samples else chunk_samples
    return spans, hop, sample_rate


def decode_utterance(
    model: RKNNModel,
    id2token: dict,
    audio: Union[str, np.ndarray],
    verbose: bool = True,
    *,
    timestamps: bool = False,
    task: TaskType = "transcribe",
    language: Optional[str] = None,
    on_chunk: Optional[Callable[[DecodeResult], None]] = None,
) -> DecodeResult:
    """
    Full utterance: if longer than the model window (~30 s / 3000 mel), split into
    sliding chunks of that window size (optional overlap), decode each, stitch text.

    ``audio`` may be 16 kHz mono float32 samples or a path/bytes source for decoding.
    When ``timestamps`` is True, returns segment spans for LLM/video alignment.
    """
    if isinstance(audio, np.ndarray):
        samples = np.ascontiguousarray(audio, dtype=np.float32)
    else:
        samples = load_audio_16k_mono(audio)

    spans, hop, sample_rate = _utterance_chunk_spans(model, samples)
    if verbose or len(spans) > 1:
        dur_s = len(samples) / float(sample_rate)
        print(
            f"audio_chunks={len(spans)} duration_s={dur_s:.2f} "
            f"window_s={spans[0][1].shape[0] / sample_rate if spans else 0:.2f} "
            f"timestamps={timestamps} task={task}"
        )

    parts: List[str] = []
    all_segments: List[TranscriptSegment] = []
    for i, (start_sample, chunk) in enumerate(spans):
        if verbose and len(spans) > 1:
            print(
                f"chunk {i + 1}/{len(spans)} samples={len(chunk)} start={start_sample}"
            )
        time_offset = start_sample / float(sample_rate)
        result = decode_samples(
            model,
            id2token,
            chunk,
            verbose=verbose,
            timestamps=timestamps,
            time_offset=time_offset,
            task=task,
            language=language,
        )
        if on_chunk is not None and result.text:
            on_chunk(result)
        if timestamps and result.segments is not None:
            segs = result.segments
            if i < len(spans) - 1:
                boundary = (start_sample + hop) / float(sample_rate)
                segs = [s for s in segs if s.start < boundary]
            all_segments.extend(segs)
            if result.text:
                parts.append(result.text)
        elif result.text:
            parts.append(result.text)

    if timestamps:
        text = " ".join(s.text for s in all_segments).strip() or stitch_transcripts(
            parts
        )
        segments: Optional[List[TranscriptSegment]] = all_segments
    else:
        text = stitch_transcripts(parts)
        segments = None

    if verbose and len(spans) > 1:
        print(text)
    return DecodeResult(text=text, segments=segments)


def decode_utterance_stream(
    model: RKNNModel,
    id2token: dict,
    audio: Union[str, np.ndarray],
    verbose: bool = True,
    *,
    timestamps: bool = False,
    task: TaskType = "transcribe",
    language: Optional[str] = None,
) -> Iterator[DecodeResult]:
    """Yield per-chunk decode results as each RKNN window completes."""
    if isinstance(audio, np.ndarray):
        samples = np.ascontiguousarray(audio, dtype=np.float32)
    else:
        samples = load_audio_16k_mono(audio)

    spans, hop, sample_rate = _utterance_chunk_spans(model, samples)
    for i, (start_sample, chunk) in enumerate(spans):
        time_offset = start_sample / float(sample_rate)
        result = decode_samples(
            model,
            id2token,
            chunk,
            verbose=verbose,
            timestamps=timestamps,
            time_offset=time_offset,
            task=task,
            language=language,
        )
        if result.text:
            yield result


def _infer_model_size_key(encoder_path: str, profile: Optional[str]) -> str:
    """
    Размер модели: WHISPER_MODEL_PROFILE / WHISPER_VARIANT, иначе подстрока в пути
    (tiny, base, small, medium, turbo).
    """
    p = (profile or "").strip().lower()
    if p:
        if p in ("tiny", "base", "small", "medium", "turbo"):
            return p
        raise ValueError(
            f"WHISPER_MODEL_PROFILE must be one of tiny,base,small,medium,turbo; got {profile!r}"
        )
    enc = encoder_path.lower()
    for k in ("tiny", "base", "small", "medium", "turbo"):
        if k in enc:
            return k
    raise ValueError(
        "Cannot infer model size from encoder path (expected tiny/base/small/medium/turbo in the "
        f"path, or set WHISPER_MODEL_PROFILE): {encoder_path!r}"
    )


def _language_token_id(lang: str) -> int:
    """Whisper multilingual language token (e.g. ru -> 50263)."""
    return language_token_id(lang)


def model_config_from_encoder_path(
    encoder_path: str,
    profile: Optional[str] = None,
):
    """Return decoder hyperparams + mel shape for a given encoder .rknn path."""
    prof_env = os.environ.get("WHISPER_MODEL_PROFILE") or os.environ.get("WHISPER_VARIANT")
    size_key = _infer_model_size_key(encoder_path, profile if profile is not None else prof_env)

    enc = encoder_path
    english_only = ".en" in enc
    if english_only:
        notimestamps_id = 50362
        timestamp_begin = 50363
        eot = 50256
        n_mels, mel_time_frames = 80, 3000
    elif size_key == "turbo":
        notimestamps_id = 50364
        timestamp_begin = 50365
        eot = 50257
        n_mels, mel_time_frames = 128, 3000
    else:
        notimestamps_id = 50363
        timestamp_begin = 50364
        eot = 50257
        n_mels, mel_time_frames = 80, 3000

    if size_key == "tiny":
        n_text_layer, n_text_ctx, n_text_state = 4, 448, 384
    elif size_key == "base":
        n_text_layer, n_text_ctx, n_text_state = 6, 448, 512
    elif size_key == "small":
        n_text_layer, n_text_ctx, n_text_state = 12, 448, 768
    elif size_key == "medium":
        n_text_layer, n_text_ctx, n_text_state = 24, 448, 1024
    elif size_key == "turbo":
        n_text_layer, n_text_ctx, n_text_state = 4, 448, 1280
    else:
        raise ValueError(f"Unsupported model size: {size_key!r}")

    return (
        size_key,
        english_only,
        eot,
        n_text_layer,
        n_text_ctx,
        n_text_state,
        n_mels,
        mel_time_frames,
        notimestamps_id,
        timestamp_begin,
    )


def apply_librknnrt_from_optional_path(cli_path: Optional[str]) -> None:
    lib_path = resolve_librknnrt_path(cli_path)
    if lib_path is not None:
        apply_rknnrt_path_override(lib_path)
