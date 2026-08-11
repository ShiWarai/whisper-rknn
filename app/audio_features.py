"""Whisper mel без PyTorch / openai-whisper."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import kaldi_native_fbank as knf
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480_000 samples / 30 s window


@lru_cache(maxsize=2)
def _mel_filters(n_mels: int) -> np.ndarray:
    if n_mels not in (80, 128):
        raise ValueError(f"Unsupported n_mels: {n_mels}")
    path = Path(__file__).resolve().parent / "assets" / "mel_filters.npz"
    with np.load(path, allow_pickle=False) as f:
        filters = f[f"mel_{n_mels}"].astype(np.float32)
    # STFT Whisper отбрасывает бин Найквиста; оставляем (n_mels, n_fft // 2).
    return filters[:, : N_FFT // 2]


def pad_or_trim(
    array: np.ndarray, length: int = N_SAMPLES, *, axis: int = -1
) -> np.ndarray:
    """Дополнить или обрезать аудио до ``length`` сэмплов (по умолчанию 30 с @ 16 кГц)."""
    array = np.asarray(array, dtype=np.float32)
    if array.shape[axis] > length:
        sl = [slice(None)] * array.ndim
        sl[axis] = slice(0, length)
        return array[tuple(sl)].copy()
    if array.shape[axis] < length:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, length - array.shape[axis])
        return np.pad(array, pad_width, mode="constant")
    return array


def _stft_power_loop(audio: np.ndarray) -> np.ndarray:
    """Эталонный STFT (цикл); оставлен для регрессионных тестов."""
    window = np.hanning(N_FFT).astype(np.float32)
    pad = N_FFT // 2
    audio = np.pad(audio.astype(np.float32, copy=False), (pad, pad), mode="reflect")
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    n_bins = N_FFT // 2
    power = np.empty((n_bins, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = audio[start : start + N_FFT] * window
        spectrum = np.fft.rfft(frame, n=N_FFT)
        power[:, i] = np.abs(spectrum[:-1]) ** 2
    return power


def _stft_power(audio: np.ndarray) -> np.ndarray:
    """Спектр мощности как у ``torch.stft`` + ``[..., :-1]`` (форма: n_fft//2 × T)."""
    window = np.hanning(N_FFT).astype(np.float32)
    pad = N_FFT // 2
    audio = np.pad(audio.astype(np.float32, copy=False), (pad, pad), mode="reflect")
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    n_bins = N_FFT // 2
    if n_frames <= 0:
        return np.zeros((n_bins, 0), dtype=np.float32)

    shape = (n_frames, N_FFT)
    strides = (HOP_LENGTH * audio.strides[0], audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    frames = frames * window
    spectrum = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = (np.abs(spectrum[:, :-1]) ** 2).T.astype(np.float32)
    return power


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """
    Log-mel features as in openai-whisper ``audio.log_mel_spectrogram``.
    Returns float32 array of shape (n_mels, n_frames).
    """
    audio = np.asarray(audio, dtype=np.float32)
    magnitudes = _stft_power(audio)
    mel_spec = _mel_filters(n_mels) @ magnitudes
    log_spec = np.maximum(mel_spec, 1e-10)
    log_spec = np.log10(log_spec)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def _normalize_whisper_mel(mel: np.ndarray) -> np.ndarray:
    """Путь sherpa-onnx / knf: та же log-компрессия, что у Whisper."""
    mel = np.maximum(mel.astype(np.float32, copy=False), 1e-10)
    mel = np.log10(mel)
    mel = np.maximum(mel, mel.max() - 8.0)
    return ((mel + 4.0) / 4.0).astype(np.float32)


def _pad_mel_frames(mel: np.ndarray, target_frames: int) -> np.ndarray:
    """Дополнить/обрезать ось времени до ``target_frames`` (mel: T × n_mels)."""
    mel = np.pad(mel, ((0, 1500), (0, 0)), mode="constant")
    if mel.shape[0] > target_frames:
        mel = mel[: target_frames - 50]
        mel = np.pad(mel, ((0, 50), (0, 0)), mode="constant")
    elif mel.shape[0] < target_frames:
        mel = np.pad(mel, ((0, target_frames - mel.shape[0]), (0, 0)), mode="constant")
    return mel


def _features_from_knf(samples: np.ndarray, n_mels: int) -> np.ndarray:
    frames = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = n_mels
    fbank = knf.OnlineWhisperFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, samples)
    fbank.input_finished()
    for i in range(fbank.num_frames_ready):
        frames.append(fbank.get_frame(i))
    mel = np.stack(frames, axis=0).astype(np.float32)
    return _normalize_whisper_mel(mel)


def compute_features(
    samples: Union[np.ndarray, list],
    n_mels: int = 80,
    target_frames: int = 3000,
) -> np.ndarray:
    """
    Encoder input float32 [1, n_mels, target_frames].

  turbo (128 mels): pad_or_trim + log_mel_spectrogram (Whisper large-v3 path).
  tiny..medium (80 mels): knf fbank + sherpa padding to 3000 frames.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if n_mels == 128:
        audio = pad_or_trim(samples)
        mel = log_mel_spectrogram(audio, n_mels=128)
        # Энкодер ожидает ровно target_frames (3000 для 30 с @ 100 fps).
        if mel.shape[1] > target_frames:
            mel = mel[:, :target_frames]
        elif mel.shape[1] < target_frames:
            mel = np.pad(
                mel, ((0, 0), (0, target_frames - mel.shape[1])), mode="constant"
            )
        return mel[np.newaxis, ...].astype(np.float32)

    mel = _features_from_knf(samples, n_mels)
    mel = _pad_mel_frames(mel, target_frames)
    return np.ascontiguousarray(mel.T[np.newaxis, ...], dtype=np.float32)
