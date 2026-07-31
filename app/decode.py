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
import os
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from app import audio_io as _audio_io
from app.audio_features import compute_features
from app.audio_io import (  # noqa: F401
    AudioSource,
    load_audio_16k_mono,
    prepare_audio_16k_mono,
    resample_linear,
)
from app.core.model_config import model_config_from_encoder_path  # noqa: F401
from app.core.text import stitch_transcripts
from app.core.types import DecodeResult, DecodeTimings, TaskType, TranscriptSegment
from app.core.window import whisper_window_samples
from app.onnx_decoder import OnnxDecoder, resolve_decoder_backend
from app.whisper_languages import language_token_id

_HAS_AV = _audio_io._HAS_AV


def _get_rknnlite():
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise ImportError(
            "Install rknn_toolkit_lite2 (см. Docker-сборку и каталог third_party/*.whl)."
        ) from exc
    return RKNNLite


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


# Re-export для тестов и CLI (реализация в app.audio_io).


def load_audio_wav(path: str) -> Tuple[np.ndarray, int]:
    from app.audio_io import load_audio_wav as _load

    return _load(path)


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def resolve_npu_core_mask(name: Optional[str] = None) -> int:
    """
    Сопоставить WHISPER_NPU_CORE_MASK с константами RKNNLite.
    По умолчанию: NPU_CORE_0_1_2 (все три ядра на RK3588).
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
    RKNNLite = _get_rknnlite()
    return int(getattr(RKNNLite, key))


def init_model(filename, target_platform="rk3588", core_mask: Optional[int] = None):
    if not Path(filename).is_file():
        raise FileNotFoundError(f"{filename} does not exist")

    if core_mask is None:
        core_mask = resolve_npu_core_mask()

    from app.rknn_share import drop_rknn_model_bytes

    RKNNLite = _get_rknnlite()
    rknn_lite = RKNNLite(verbose=False)
    try:
        ret = rknn_lite.load_rknn(path=filename)
        if ret != 0:
            raise RuntimeError(f"Load model {filename} failed!")

        ret = rknn_lite.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(
                f"Failed to init rknn runtime for {filename} (core_mask={core_mask})"
            )
        # RKNNLite держит полную Python-копию .rknn после DMA upload.
        drop_rknn_model_bytes(rknn_lite)
    except Exception:
        try:
            rknn_lite.release()
        except Exception:
            pass
        raise
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
    """Собрать prompt-токены декодера Whisper для одного запроса."""
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
        decoder_backend: Optional[str] = None,
        encoder_workers: Optional[int] = None,
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

        backend, decoder_path = resolve_decoder_backend(
            backend=decoder_backend,
            decoder_path=decoder,
        )
        self.decoder_backend = backend
        self.decoder_path = decoder_path
        self.encoder_path = encoder

        from app.encode_pool import EncoderPool, resolve_encoder_worker_count

        self._encoder_pool: Optional[EncoderPool] = None
        self.encoder = None
        self.encoder_workers = 1

        if parallel_encode_enabled():
            n_workers = resolve_encoder_worker_count(
                encoder,
                requested=encoder_workers,
                decoder_path=decoder_path,
            )
            if n_workers < 1:
                raise RuntimeError(
                    "encoder_pool: insufficient MemAvailable for even 1 encoder worker "
                    "(set WHISPER_ENCODER_WORKERS=1 to force, or free RAM)"
                )
            if verbose:
                print(f"encoder_pool: MemAvailable pick={n_workers}, probing NPU…")
            self._encoder_pool = EncoderPool(
                encoder,
                n_workers=n_workers,
                target_platform=target_platform,
                verbose=verbose,
                n_mels=n_mels,
                mel_time_frames=mel_time_frames,
            )
            self.encoder_workers = self._encoder_pool.n_workers
        else:
            core_mask = resolve_npu_core_mask()
            if verbose:
                print("model_size", self.size_key, "english_only", self.english_only)
                print("eot", self.eot)
                print("timestamp_begin", self.timestamp_begin)
                print("npu_core_mask", core_mask, os.environ.get("WHISPER_NPU_CORE_MASK", "0_1_2"))
            self.encoder = init_model(encoder, core_mask=core_mask)

        if verbose:
            print("decoder_backend", self.decoder_backend, self.decoder_path)
        self._onnx_decoder: Optional[OnnxDecoder] = None
        self.decoder = None
        if self.decoder_backend == "onnx":
            self._onnx_decoder = OnnxDecoder(decoder_path, n_text_layer=n_text_layer)
        else:
            dec_mask = resolve_npu_core_mask()
            self.decoder = init_model(decoder_path, core_mask=dec_mask)

    def sot_sequence_for(
        self,
        timestamps: bool,
        *,
        task: TaskType = "transcribe",
        language: Optional[str] = None,
    ) -> List[int]:
        """Prompt-токены для текущего запроса."""
        return build_sot_sequence(
            size_key=self.size_key,
            english_only=self.english_only,
            task=task,
            language=language,
            timestamps=timestamps,
            notimestamps_id=self.notimestamps_id,
        )

    def release(self):
        if self._encoder_pool is not None:
            self._encoder_pool.shutdown()
            self._encoder_pool = None
        if self.encoder is not None:
            self.encoder.release()
            self.encoder = None
        if self.decoder is not None:
            self.decoder.release()
        if self._onnx_decoder is not None:
            self._onnx_decoder.release()

    def run_encoder(self, x):
        arr = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
        if self._encoder_pool is not None:
            future = self._encoder_pool.submit(0, arr)
            return future.result().cross_kv
        assert self.encoder is not None
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
        if self._onnx_decoder is not None:
            return self._onnx_decoder.run(tokens, self_kv, cross_kv, offset, mask)
        assert self.decoder is not None
        return self.decoder.inference(
            inputs=[tokens] + self_kv + cross_kv + [offset, mask]
        )


def parallel_encode_enabled() -> bool:
    return os.environ.get("WHISPER_PARALLEL_ENCODE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _max_chunk_samples(model: RKNNModel, sample_rate: int = 16000) -> int:
    chunk_samples = whisper_window_samples(sample_rate, model.mel_time_frames)
    raw = os.environ.get("WHISPER_MAX_CHUNK_SECONDS", "").strip()
    if not raw:
        raw = os.environ.get("WHISPER_CHUNK_SECONDS", "").strip()
    if raw:
        try:
            sec = float(raw)
            if sec > 0:
                chunk_samples = min(chunk_samples, max(1, int(round(sec * sample_rate))))
        except ValueError:
            pass
    return chunk_samples


def _min_tail_samples(sample_rate: int = 16000) -> int:
    raw = os.environ.get("WHISPER_MIN_TAIL_SECONDS", "8").strip()
    try:
        sec = float(raw)
    except ValueError:
        sec = 8.0
    if sec <= 0:
        return 0
    return max(0, int(round(sec * sample_rate)))


def _merge_short_tail_spans(
    spans: List[Tuple[int, np.ndarray]],
    samples: np.ndarray,
    chunk_samples: int,
    *,
    min_tail_samples: int,
) -> List[Tuple[int, np.ndarray]]:
    """Заменить крошечный хвостовой кусок одним финальным полноразмерным окном."""
    if min_tail_samples <= 0 or len(spans) <= 1:
        return spans
    last_start, last_chunk = spans[-1]
    if len(last_chunk) >= min_tail_samples:
        return spans
    n = int(samples.shape[0])
    final_start = max(0, n - chunk_samples)
    return spans[:-1] + [(final_start, samples[final_start:n])]


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
        # Прижать финальное окно к концу, чтобы крошечный хвост не дублировал hop.
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
    """Скользящие окна длины ``chunk_samples`` (см. :func:`iter_audio_chunk_spans`)."""
    return [chunk for _, chunk in iter_audio_chunk_spans(samples, chunk_samples, overlap_samples)]


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
        # Подряд идущий timestamp после end — начало следующего сегмента, оставляем.
        if i + 1 < n and is_ts(token_ids[i + 1]):
            i += 1
        else:
            i += 1

    full = " ".join(seg.text for seg in segments).strip()
    if not full:
        text_ids = [t for t in token_ids if t < timestamp_begin]
        full = _tokens_to_text(text_ids, id2token)
    return full, segments


_LOGIT_MASK = -1e9
# OpenAI default max_initial_timestamp=1.0s, precision=0.02s → index 50.
_DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX = 50


def _logsumexp(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    if not np.isfinite(vmax):
        return vmax
    shifted = values - vmax
    return float(vmax + np.log(np.sum(np.exp(shifted))))


def apply_timestamp_rules(
    logits: np.ndarray,
    sampled: Sequence[int],
    *,
    eot: int,
    timestamp_begin: int,
    notimestamps_id: Optional[int],
    max_initial_timestamp_index: Optional[int] = _DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX,
) -> np.ndarray:
    """
    Numpy-порт OpenAI ``ApplyTimestampRules`` для одного greedy-шага.

    ``sampled`` — токены, уже сгенерированные после SOT (аналог ``tokens[sample_begin:]``).
  """
    out = np.asarray(logits, dtype=np.float32).copy()
    seq = list(sampled)

    if notimestamps_id is not None:
        out[notimestamps_id] = _LOGIT_MASK

    last_was_timestamp = len(seq) >= 1 and seq[-1] >= timestamp_begin
    penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin

    if last_was_timestamp:
        if penultimate_was_timestamp:
            out[timestamp_begin:] = _LOGIT_MASK
        else:
            out[:eot] = _LOGIT_MASK

    timestamps = [t for t in seq if t >= timestamp_begin]
    if timestamps:
        if last_was_timestamp and not penultimate_was_timestamp:
            timestamp_last = timestamps[-1]
        else:
            timestamp_last = timestamps[-1] + 1
        out[timestamp_begin:timestamp_last] = _LOGIT_MASK

    if len(seq) == 0:
        out[:timestamp_begin] = _LOGIT_MASK
        if max_initial_timestamp_index is not None:
            last_allowed = timestamp_begin + max_initial_timestamp_index
            out[last_allowed + 1 :] = _LOGIT_MASK

    # log_softmax по уже замаскированным logits
    shifted = out - np.max(out)
    logprobs = shifted - _logsumexp(shifted)
    timestamp_logprob = _logsumexp(logprobs[timestamp_begin:])
    max_text_token_logprob = float(np.max(logprobs[:timestamp_begin]))
    if timestamp_logprob > max_text_token_logprob:
        out[:timestamp_begin] = _LOGIT_MASK

    return out


def _select_next_token(
    logits: np.ndarray,
    *,
    timestamps: bool,
    sampled: Sequence[int],
    model: RKNNModel,
) -> int:
    arr = np.asarray(logits, dtype=np.float32).copy()
    if timestamps and model.timestamp_begin is not None:
        arr = apply_timestamp_rules(
            arr,
            sampled,
            eot=model.eot,
            timestamp_begin=model.timestamp_begin,
            notimestamps_id=model.notimestamps_id,
        )
    return int(arr.argmax())


def decode_from_cross_kv(
    model: RKNNModel,
    id2token: dict,
    cross_kv,
    *,
    verbose: bool = True,
    timestamps: bool = False,
    time_offset: float = 0.0,
    task: TaskType = "transcribe",
    language: Optional[str] = None,
    collect_timings: bool = False,
    wall_t0: Optional[float] = None,
) -> DecodeResult:
    """Авторегрессионный decode по готовому cross-attention KV энкодера."""
    timings = DecodeTimings(decoder_backend=model.decoder_backend) if collect_timings else None
    if wall_t0 is None:
        wall_t0 = time.perf_counter()

    self_kv = model.get_self_cache()
    sot = model.sot_sequence_for(timestamps, task=task, language=language)
    offset = np.array([0], dtype=np.int32)
    out = None
    decoder_calls = 0
    for t in sot:
        token = np.array([[t]], dtype=np.int32)
        mask = causal_mask_1d(offset.item(), model.n_text_ctx)
        dec_t0 = time.perf_counter()
        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )
        if timings is not None:
            timings.decoder_ms += (time.perf_counter() - dec_t0) * 1000.0
        decoder_calls += 1
        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]
        offset += 1

    assert out is not None
    idx = _select_next_token(
        out[0][0, 0],
        timestamps=timestamps,
        sampled=[],
        model=model,
    )
    ans: List[int] = []
    max_ngram_repeats = int(os.environ.get("WHISPER_MAX_NGRAM_REPEAT", "6"))
    stop_at = resolve_decode_token_limit(model.n_text_ctx)
    stopped_on_repeat = False

    def _repeating_tail(tokens: List[int]) -> bool:
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

    while idx != model.eot and offset.item() < stop_at:
        ans.append(int(idx))
        if _repeating_tail(ans):
            stopped_on_repeat = True
            for n in (1, 2, 3, 4):
                need = n * max_ngram_repeats
                if len(ans) >= need and ans[-n:] * max_ngram_repeats == ans[-need:]:
                    ans = ans[: len(ans) - n * (max_ngram_repeats - 1)]
                    break
            break
        token = np.array([[idx]], dtype=np.int32)
        mask = causal_mask_1d(offset.item(), model.n_text_ctx)
        dec_t0 = time.perf_counter()
        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )
        if timings is not None:
            timings.decoder_ms += (time.perf_counter() - dec_t0) * 1000.0
        decoder_calls += 1
        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]
        offset += 1
        idx = _select_next_token(
            out[0][0, 0],
            timestamps=timestamps,
            sampled=ans,
            model=model,
        )

    truncated = idx != model.eot and not stopped_on_repeat and offset.item() >= stop_at
    if timings is not None:
        timings.tokens = len(ans)
        timings.decoder_calls = decoder_calls
        timings.wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        timings.truncated = truncated
    if truncated and verbose:
        print(
            f"decode truncated: offset={offset.item()} stop_at={stop_at} "
            f"n_text_ctx={model.n_text_ctx} (no EOT)"
        )

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
    return DecodeResult(text=text, segments=segments, timings=timings, truncated=truncated)


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
    collect_timings: bool = False,
) -> DecodeResult:
    """Декодировать одно окно 16 кГц mono float (≤ окно модели; короче — с паддингом)."""
    timings = DecodeTimings(decoder_backend=model.decoder_backend) if collect_timings else None
    wall_t0 = time.perf_counter()

    mel_t0 = time.perf_counter()
    features = compute_features(
        samples,
        n_mels=model.n_mels,
        target_frames=model.mel_time_frames,
    )
    if timings is not None:
        timings.mel_ms = (time.perf_counter() - mel_t0) * 1000.0

    if verbose:
        print(features.shape)

    enc_t0 = time.perf_counter()
    cross_kv = model.run_encoder(features)
    if timings is not None:
        timings.encoder_ms = (time.perf_counter() - enc_t0) * 1000.0

    decoded = decode_from_cross_kv(
        model,
        id2token,
        cross_kv,
        verbose=verbose,
        timestamps=timestamps,
        time_offset=time_offset,
        task=task,
        language=language,
        collect_timings=collect_timings,
        wall_t0=wall_t0,
    )
    if timings is not None and decoded.timings is not None:
        timings.decoder_ms = decoded.timings.decoder_ms
        timings.tokens = decoded.timings.tokens
        timings.decoder_calls = decoded.timings.decoder_calls
        timings.truncated = decoded.timings.truncated
        timings.wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    return DecodeResult(
        text=decoded.text,
        segments=decoded.segments,
        timings=timings,
        truncated=decoded.truncated,
    )


def resolve_decode_token_limit(n_text_ctx: int) -> int:
    """
    Hard stop for autoregressive decode.

    Default ``0`` / ``auto`` → model context ``n_text_ctx`` (stop on EOT or full KV).
    A positive ``WHISPER_MAX_DECODE_TOKENS`` is an optional soft cap (clamped to ctx).
    """
    raw = os.environ.get("WHISPER_MAX_DECODE_TOKENS", "0").strip().lower()
    if raw in ("", "0", "auto", "ctx", "full"):
        return int(n_text_ctx)
    try:
        n = int(raw)
    except ValueError:
        return int(n_text_ctx)
    if n <= 0:
        return int(n_text_ctx)
    return min(n, int(n_text_ctx))


def _truncate_retry_samples(sample_rate: int = 16000) -> int:
    """Доп. overlap, если окно остановилось без EOT (переслушать хвост)."""
    raw = os.environ.get("WHISPER_TRUNCATE_RETRY_SECONDS", "10").strip()
    try:
        sec = float(raw)
    except ValueError:
        sec = 10.0
    if sec <= 0:
        return 0
    return max(0, int(round(sec * sample_rate)))


def _utterance_window_params(model: RKNNModel) -> Tuple[int, int, int]:
    """Вернуть ``(sample_rate, chunk_samples, overlap_samples)``."""
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
    env_ov = os.environ.get("WHISPER_CHUNK_OVERLAP_SECONDS", "2").strip()
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
    return sample_rate, chunk_samples, overlap_samples


def _next_chunk_start(
    *,
    start: int,
    end: int,
    n: int,
    hop: int,
    overlap_samples: int,
    chunk_samples: int,
    sample_rate: int,
    truncated: bool,
    segments: Optional[List[TranscriptSegment]],
    timestamps: bool,
) -> int:
    """
    Advance sliding window. If decode stopped without EOT, pull next start back
    so the unfinished tail is re-decoded (Whisper-style seek without full VAD).
    With timestamps, seek from the last segment end when available.
    """
    if end >= n:
        return n

    if timestamps and segments:
        last_end = max((s.end for s in segments), default=None)
        if last_end is not None and last_end > 0:
            seek = int(round(last_end * sample_rate))
            lookback = max(overlap_samples, int(0.5 * sample_rate))
            nxt = max(start + max(hop // 4, 1), seek - lookback)
            return _maybe_merge_short_tail(nxt, n, chunk_samples, sample_rate)

    if truncated:
        retry = _truncate_retry_samples(sample_rate)
        pullback = max(overlap_samples, retry)
        advance = max(hop // 2, 1)
        nxt = max(start + advance, end - pullback)
    else:
        nxt = start + hop

    return _maybe_merge_short_tail(nxt, n, chunk_samples, sample_rate)


def _maybe_merge_short_tail(
    nxt: int, n: int, chunk_samples: int, sample_rate: int
) -> int:
    """Если хвост после ``nxt`` крошечный — прижать к финальному полному окну."""
    if nxt >= n:
        return n
    remaining = n - nxt
    min_tail = _min_tail_samples(sample_rate)
    if 0 < remaining < min_tail:
        return max(0, n - chunk_samples)
    return nxt


def _utterance_chunk_spans(model: RKNNModel, samples: np.ndarray) -> Tuple[List[Tuple[int, np.ndarray]], int, int]:
    """Вернуть span'ы чанков и hop/overlap в сэмплах для обрезки таймкодов."""
    sample_rate, chunk_samples, overlap_samples = _utterance_window_params(model)
    spans = iter_audio_chunk_spans(
        samples, chunk_samples, overlap_samples=overlap_samples
    )
    spans = _merge_short_tail_spans(
        spans,
        samples,
        chunk_samples,
        min_tail_samples=_min_tail_samples(sample_rate),
    )
    hop = chunk_samples - overlap_samples if overlap_samples else chunk_samples
    return spans, hop, sample_rate


def decode_utterance_parallel(
    model: RKNNModel,
    id2token: dict,
    samples: np.ndarray,
    verbose: bool = True,
    *,
    timestamps: bool = False,
    task: TaskType = "transcribe",
    language: Optional[str] = None,
    on_chunk: Optional[Callable[[DecodeResult], None]] = None,
    collect_timings: bool = False,
    wall_t0: Optional[float] = None,
) -> DecodeResult:
    """
    Span'ы по VAD в RAM + параллельный NPU encode (encoder pool) + последовательный CPU decode.
    """
    from app.speech_cut import chunk_audio_views, plan_voice_aware_chunks

    if wall_t0 is None:
        wall_t0 = time.perf_counter()
    sample_rate = 16000
    timings = DecodeTimings(decoder_backend=model.decoder_backend) if collect_timings else None
    if timings is not None:
        timings.parallel_workers = model.encoder_workers

    spans, _probs, vad_timings = plan_voice_aware_chunks(samples)
    if timings is not None:
        timings.vad_ms = vad_timings.vad_ms
        timings.cut_ms = vad_timings.cut_ms

    if verbose:
        dur_s = samples.shape[0] / float(sample_rate)
        print(
            f"parallel vad: duration_s={dur_s:.2f} chunks={len(spans)} "
            f"workers={model.encoder_workers} vad_ms={vad_timings.vad_ms:.1f}"
        )

    if len(spans) == 1:
        return decode_samples(
            model,
            id2token,
            samples,
            verbose=verbose,
            timestamps=timestamps,
            task=task,
            language=language,
            collect_timings=collect_timings,
        )

    chunk_audio = chunk_audio_views(samples, spans)
    mels: List[np.ndarray] = []
    mel_t0 = time.perf_counter()
    for i, chunk in enumerate(chunk_audio):
        if verbose:
            span = spans[i]
            print(
                f"chunk {i + 1}/{len(spans)} samples={len(chunk)} "
                f"start={span.start} ({span.start / sample_rate:.2f}s) reason={span.reason}"
            )
        mels.append(
            compute_features(
                chunk,
                n_mels=model.n_mels,
                target_frames=model.mel_time_frames,
            )
        )
    if timings is not None:
        timings.mel_ms = (time.perf_counter() - mel_t0) * 1000.0

    encode_futures = []
    if model._encoder_pool is not None:
        for i, mel in enumerate(mels):
            encode_futures.append(model._encoder_pool.submit(i, mel))
    else:
        encode_futures = None

    parts: List[str] = []
    all_segments: List[TranscriptSegment] = []
    any_truncated = False
    enc_started_at: List[float] = []
    enc_finished_at: List[float] = []

    for i, span in enumerate(spans):
        if encode_futures is not None:
            enc_result = encode_futures[i].result()
            cross_kv = enc_result.cross_kv
            if timings is not None:
                # Реальное ожидание в очереди (submit → старт воркера), не блок future.result().
                timings.encode_queue_wait_ms += enc_result.queue_wait_ms
                # Сумма чистого времени NPU inference по чанкам.
                timings.encoder_ms += enc_result.encode_ms
                enc_started_at.append(enc_result.started_at)
                enc_finished_at.append(enc_result.finished_at)
        else:
            enc_t0 = time.perf_counter()
            cross_kv = model.run_encoder(mels[i])
            if timings is not None:
                elapsed = (time.perf_counter() - enc_t0) * 1000.0
                timings.encoder_ms += elapsed
                timings.encoder_wall_ms += elapsed

        time_offset = span.start / float(sample_rate)
        result = decode_from_cross_kv(
            model,
            id2token,
            cross_kv,
            verbose=verbose,
            timestamps=timestamps,
            time_offset=time_offset,
            task=task,
            language=language,
            collect_timings=collect_timings,
        )
        any_truncated = any_truncated or result.truncated
        if timings is not None and result.timings is not None:
            timings.decoder_ms += result.timings.decoder_ms
            timings.tokens += result.timings.tokens
            timings.decoder_calls += result.timings.decoder_calls
            timings.truncated = timings.truncated or result.timings.truncated

        if on_chunk is not None and result.text:
            on_chunk(result)

        if timestamps and result.segments is not None:
            all_segments.extend(result.segments)
            if result.text:
                parts.append(result.text)
        elif result.text:
            parts.append(result.text)

    if timings is not None:
        timings.chunks = len(spans)
        timings.truncated = timings.truncated or any_truncated
        if enc_started_at and enc_finished_at:
            # Время на стене параллельной волны encode (первый старт → последний финиш).
            timings.encoder_wall_ms = (
                max(enc_finished_at) - min(enc_started_at)
            ) * 1000.0

    if timestamps:
        text = " ".join(s.text for s in all_segments).strip() or stitch_transcripts(parts)
        segments: Optional[List[TranscriptSegment]] = all_segments
    else:
        text = stitch_transcripts(parts)
        segments = None

    if verbose:
        print(text)

    if timings is not None:
        timings.wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        duration_s = samples.shape[0] / float(sample_rate)
        if duration_s > 0:
            timings.rtf = (timings.wall_ms / 1000.0) / duration_s

    return DecodeResult(
        text=text,
        segments=segments,
        timings=timings,
        truncated=any_truncated,
    )


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
    collect_timings: bool = False,
) -> DecodeResult:
    """
    Full utterance: if longer than the model window (~30 s / 3000 mel), split into
    sliding chunks of that window size (optional overlap), decode each, stitch text.

    Decode stops on EOT (or model ``n_text_ctx``). If a non-final window exits
    without EOT, the next window starts earlier so the unfinished tail is re-heard.

    ``audio`` may be 16 kHz mono float32 samples or a path/bytes source for decoding.
    When ``timestamps`` is True, returns segment spans for LLM/video alignment.
    """
    wall_t0 = time.perf_counter()
    timings = DecodeTimings(decoder_backend=model.decoder_backend) if collect_timings else None

    audio_t0 = time.perf_counter()
    if isinstance(audio, np.ndarray):
        samples = np.ascontiguousarray(audio, dtype=np.float32)
    else:
        samples = load_audio_16k_mono(audio)
    if timings is not None:
        timings.audio_ms = (time.perf_counter() - audio_t0) * 1000.0

    sample_rate = 16000
    chunk_samples = _max_chunk_samples(model, sample_rate)
    n = int(samples.shape[0])

    if parallel_encode_enabled() and n > chunk_samples:
        return decode_utterance_parallel(
            model,
            id2token,
            samples,
            verbose=verbose,
            timestamps=timestamps,
            task=task,
            language=language,
            on_chunk=on_chunk,
            collect_timings=collect_timings,
            wall_t0=wall_t0,
        )

    sample_rate, chunk_samples, overlap_samples = _utterance_window_params(model)
    hop = chunk_samples - overlap_samples if overlap_samples else chunk_samples
    min_tail = _min_tail_samples(sample_rate)
    n = int(samples.shape[0])

    if verbose or n > chunk_samples:
        dur_s = n / float(sample_rate)
        print(
            f"audio_duration_s={dur_s:.2f} window_s={chunk_samples / sample_rate:.2f} "
            f"overlap_s={overlap_samples / sample_rate:.2f} "
            f"timestamps={timestamps} task={task}"
        )

    parts: List[str] = []
    all_segments: List[TranscriptSegment] = []
    start = 0
    chunk_i = 0
    any_truncated = False
    max_chunks = max(1, (n // max(hop // 2, 1)) + 8)

    while start < n and chunk_i < max_chunks:
        remaining = n - start
        if remaining <= chunk_samples:
            if chunk_i > 0 and remaining < min_tail:
                start = max(0, n - chunk_samples)
            end = n
        else:
            end = start + chunk_samples

        chunk = samples[start:end]
        if verbose and (n > chunk_samples or chunk_i > 0):
            print(
                f"chunk {chunk_i + 1} samples={len(chunk)} "
                f"start={start} ({start / sample_rate:.2f}s)"
            )
        time_offset = start / float(sample_rate)
        result = decode_samples(
            model,
            id2token,
            chunk,
            verbose=verbose,
            timestamps=timestamps,
            time_offset=time_offset,
            task=task,
            language=language,
            collect_timings=collect_timings,
        )
        chunk_truncated = result.truncated
        any_truncated = any_truncated or chunk_truncated
        if timings is not None and result.timings is not None:
            chunk_timings = result.timings
            timings.mel_ms += chunk_timings.mel_ms
            timings.encoder_ms += chunk_timings.encoder_ms
            timings.decoder_ms += chunk_timings.decoder_ms
            timings.tokens += chunk_timings.tokens
            timings.decoder_calls += chunk_timings.decoder_calls
            timings.truncated = timings.truncated or chunk_timings.truncated
        elif timings is not None and chunk_truncated:
            timings.truncated = True

        if on_chunk is not None and result.text:
            on_chunk(result)

        next_start = _next_chunk_start(
            start=start,
            end=end,
            n=n,
            hop=hop,
            overlap_samples=overlap_samples,
            chunk_samples=chunk_samples,
            sample_rate=sample_rate,
            truncated=chunk_truncated,
            segments=result.segments,
            timestamps=timestamps,
        )

        if timestamps and result.segments is not None:
            segs = result.segments
            if next_start < n:
                boundary = next_start / float(sample_rate)
                segs = [s for s in segs if s.start < boundary]
            all_segments.extend(segs)
            if result.text:
                parts.append(result.text)
        elif result.text:
            parts.append(result.text)

        if chunk_truncated and next_start < n and verbose:
            print(
                f"truncate-retry: next_start={next_start} "
                f"({next_start / sample_rate:.2f}s)"
            )

        if next_start <= start and end >= n:
            break
        if next_start <= start:
            next_start = min(n, start + max(hop // 2, 1))
        start = next_start
        chunk_i += 1

    if timings is not None:
        timings.chunks = chunk_i
        timings.truncated = timings.truncated or any_truncated

    if timestamps:
        text = " ".join(s.text for s in all_segments).strip() or stitch_transcripts(
            parts
        )
        segments: Optional[List[TranscriptSegment]] = all_segments
    else:
        text = stitch_transcripts(parts)
        segments = None

    if verbose and chunk_i > 1:
        print(text)

    if timings is not None:
        timings.wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        duration_s = n / float(sample_rate)
        if duration_s > 0:
            timings.rtf = (timings.wall_ms / 1000.0) / duration_s

    return DecodeResult(
        text=text,
        segments=segments,
        timings=timings,
        truncated=any_truncated,
    )


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
    """Отдавать decode-результаты по чанкам по мере готовности каждого окна RKNN."""
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


def _language_token_id(lang: str) -> int:
    """Мультиязычный language token Whisper (напр. ru -> 50263)."""
    return language_token_id(lang)


def apply_librknnrt_from_optional_path(cli_path: Optional[str]) -> None:
    lib_path = resolve_librknnrt_path(cli_path)
    if lib_path is not None:
        apply_rknnrt_path_override(lib_path)
