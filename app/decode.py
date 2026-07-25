#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
# Ядро декодирования Whisper RKNN (RK3588); вызывается из app.api_server.

"""
Whisper RKNN: fbank -> encoder/decoder RKNN -> text.
Аудио через ffmpeg в 16 kHz mono WAV; без ffmpeg - форматы, которые читает soundfile.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.audio_features import compute_features
from app.whisper_languages import language_token_id

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


def prepare_audio_16k_mono(
    input_path: str, cache_dir: Path
) -> Tuple[str, Optional[Path]]:
    """
    Returns (wav_path, temp_path_or_none). If temp_path is set, caller should delete unless keep_temp.
    """
    src = Path(input_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Audio not found: {input_path}")

    ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if ffmpeg:
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="whisper_rknn_", dir=cache_dir)
        os.close(fd)
        tmp_path = Path(tmp)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(tmp_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")
        return str(tmp_path), tmp_path

    try:
        mono, sr = load_audio_wav(str(src))
    except Exception as e:
        raise RuntimeError(
            f"Cannot read {src} ({e}). "
            "Install ffmpeg for mp3/m4a/… or use WAV/FLAC readable by soundfile."
        ) from e

    if sr != 16000:
        mono = resample_linear(mono, sr, 16000)
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="whisper_rknn_sf_", dir=cache_dir)
    os.close(fd)
    tmp_path = Path(tmp)
    sf.write(str(tmp_path), mono, 16000, subtype="PCM_16")
    return str(tmp_path), tmp_path


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


class RKNNModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        sot_sequence: List[int],
        eot: int,
        n_text_layer: int,
        n_text_ctx: int,
        n_text_state: int,
        n_mels: int = 80,
        mel_time_frames: int = 3000,
        target_platform="rk3588",
        verbose: bool = True,
    ):
        self.sot_sequence = sot_sequence
        self.eot = eot
        self.n_text_layer = n_text_layer
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state
        self.n_mels = n_mels
        self.mel_time_frames = mel_time_frames

        core_mask = resolve_npu_core_mask()
        if verbose:
            print("sot_sequence", self.sot_sequence)
            print("eot", self.eot)
            print("npu_core_mask", core_mask, os.environ.get("WHISPER_NPU_CORE_MASK", "0_1_2"))

        self.encoder = init_model(encoder, core_mask=core_mask)
        self.decoder = init_model(decoder, core_mask=core_mask)

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


def iter_audio_chunks(
    samples: np.ndarray,
    chunk_samples: int,
    overlap_samples: int = 0,
) -> List[np.ndarray]:
    """
    Sliding windows of length ``chunk_samples`` (RKNN mel window, e.g. 30 s / 3000 frames).

    ``overlap_samples`` advances the window by ``chunk_samples - overlap`` so the
    model input stays ≤3000 frames while neighbouring chunks share audio.
    The last window may be shorter (feature pad / pad_or_trim fills the rest).
    """
    n = int(samples.shape[0])
    if n <= 0:
        return [samples]
    if n <= chunk_samples:
        return [samples]

    overlap_samples = int(max(0, min(overlap_samples, chunk_samples - 1)))
    hop = chunk_samples - overlap_samples
    chunks: List[np.ndarray] = []
    start = 0
    while start < n:
        end = min(start + chunk_samples, n)
        chunks.append(samples[start:end])
        if end >= n:
            break
        start += hop
        # Snap final window to the end so a tiny tail is not a near-duplicate hop.
        if start < n and start + chunk_samples >= n and n - start < hop:
            final_start = max(0, n - chunk_samples)
            if final_start > start:
                start = final_start
            # else next loop takes start:n (short tail) — fine
    return chunks


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


def decode_samples(
    model: RKNNModel,
    id2token: dict,
    samples: np.ndarray,
    verbose: bool = True,
) -> str:
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

    offset = np.array([0], dtype=np.int32)
    out = None
    for t in model.sot_sequence:
        token = np.array([[t]], dtype=np.int32)
        mask = causal_mask_1d(offset.item(), model.n_text_ctx)
        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )
        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]
        offset += 1

    assert out is not None
    idx = out[0][0, 0].argmax()
    ans: List[int] = []
    max_ngram_repeats = int(os.environ.get("WHISPER_MAX_NGRAM_REPEAT", "6"))

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

    while idx != model.eot and offset.item() < 100:
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
        idx = out[0][0, 0].argmax()

    if verbose:
        print(ans)
    text = _tokens_to_text(ans, id2token)
    if verbose:
        print(text)
    return text


def decode_utterance(
    model: RKNNModel, id2token: dict, wav_16k_path: str, verbose: bool = True
) -> str:
    """
    Full utterance: if longer than the model window (~30 s / 3000 mel), split into
    sliding chunks of that window size (optional overlap), decode each, stitch text.
    """
    samples, sample_rate = load_audio_wav(wav_16k_path)
    if sample_rate != 16000:
        samples = resample_linear(samples, sample_rate, 16000)
        sample_rate = 16000

    # Hard cap: never feed more samples than the static RKNN window (3000 mel ≈ 30 s).
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

    # Overlap must stay inside the window so each encode still sees ≤3000 frames.
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

    chunks = iter_audio_chunks(
        samples, chunk_samples, overlap_samples=overlap_samples
    )
    if verbose or len(chunks) > 1:
        dur_s = len(samples) / float(sample_rate)
        print(
            f"audio_chunks={len(chunks)} duration_s={dur_s:.2f} "
            f"window_s={chunk_samples / sample_rate:.2f} "
            f"overlap_s={overlap_samples / sample_rate:.2f}"
        )

    parts: List[str] = []
    for i, chunk in enumerate(chunks):
        if verbose and len(chunks) > 1:
            print(f"chunk {i + 1}/{len(chunks)} samples={len(chunk)}")
        part = decode_samples(model, id2token, chunk, verbose=verbose)
        if part:
            parts.append(part)

    text = stitch_transcripts(parts)
    if verbose and len(chunks) > 1:
        print(text)
    return text


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
    lang = os.environ.get("WHISPER_LANGUAGE", "ru")

    enc = encoder_path
    if ".en" in enc:
        sot_sequence = [50257, 50362]
        eot = 50256
        n_mels, mel_time_frames = 80, 3000
    elif size_key == "turbo":
        # large-v3 / turbo: vocab with 100 languages → transcribe=50360, notimestamps=50364
        lang_id = _language_token_id(lang)
        sot_sequence = [50258, lang_id, 50360, 50364]
        eot = 50257
        n_mels, mel_time_frames = 128, 3000
    else:
        # tiny..medium multilingual (99 langs): transcribe=50359, notimestamps=50363
        lang_id = _language_token_id(lang)
        sot_sequence = [50258, lang_id, 50359, 50363]
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
        sot_sequence,
        eot,
        n_text_layer,
        n_text_ctx,
        n_text_state,
        n_mels,
        mel_time_frames,
    )


def apply_librknnrt_from_optional_path(cli_path: Optional[str]) -> None:
    lib_path = resolve_librknnrt_path(cli_path)
    if lib_path is not None:
        apply_rknnrt_path_override(lib_path)
