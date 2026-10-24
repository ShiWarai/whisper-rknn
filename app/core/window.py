"""Параметры окон mel/аудио без зависимости от RKNN."""

from __future__ import annotations


def whisper_window_samples(sample_rate: int = 16000, mel_time_frames: int = 3000) -> int:
    """Сэмплов на окно модели (~30 с при mel_time_frames=3000 @ 100 frames/s)."""
    seconds = float(mel_time_frames) / 100.0
    return max(1, int(round(seconds * sample_rate)))
