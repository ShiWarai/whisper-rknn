"""Conservative RAM forecasts from MemAvailable (see video-descriptor-rkllm system_memory)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

PathLike = Union[str, Path]

MODEL_HEADROOM = 1.15
PCM_EXPANSION_FACTOR = 20
SAMPLE_RATE_HZ = 16_000
BYTES_PER_FLOAT32 = 4
MEL_TIME_FRAMES = 3000
REQUEST_OVERHEAD_BYTES = 128 * 1024 * 1024  # PyAV / numpy / Python headroom
DEFAULT_MAX_AUDIO_SECONDS = 600


def max_audio_seconds() -> int:
    raw = os.environ.get("WHISPER_MAX_AUDIO_SECONDS", str(DEFAULT_MAX_AUDIO_SECONDS)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_MAX_AUDIO_SECONDS
    return max(1, value)


def read_mem_available_bytes() -> Optional[int]:
    """MemAvailable from /proc/meminfo in bytes, or None if unreadable."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                kb = int(parts[1])
                return kb * 1024
    except OSError:
        return None
    return None


def file_size_bytes(path: PathLike) -> Optional[int]:
    try:
        size = Path(path).stat().st_size
    except OSError:
        return None
    if size <= 0:
        return None
    return size


def estimate_model_ram_bytes(encoder_path: PathLike, decoder_path: PathLike) -> int:
    """Conservative RSS estimate for encoder + decoder RKNN contexts."""
    total = 0
    for path in (encoder_path, decoder_path):
        size = file_size_bytes(path)
        if size is not None:
            total += int(size * MODEL_HEADROOM)
    return total


def estimate_pcm_bytes(upload_bytes: int, *, max_seconds: Optional[int] = None) -> int:
    """Worst-case float32 PCM after decode (compressed audio expands in RAM)."""
    if upload_bytes <= 0:
        return 0
    cap_seconds = max_audio_seconds() if max_seconds is None else max(1, max_seconds)
    from_upload = upload_bytes * PCM_EXPANSION_FACTOR
    from_duration = cap_seconds * SAMPLE_RATE_HZ * BYTES_PER_FLOAT32
    return min(from_upload, from_duration)


def estimate_request_ram_bytes(
    upload_bytes: int,
    n_mels: int,
    n_text_layer: int,
    n_text_ctx: int,
    n_text_state: int,
    *,
    max_seconds: Optional[int] = None,
) -> int:
    """Peak host RAM for one /v1/audio/transcriptions request (intentionally high)."""
    pcm_bytes = estimate_pcm_bytes(upload_bytes, max_seconds=max_seconds)
    mel_bytes = n_mels * MEL_TIME_FRAMES * BYTES_PER_FLOAT32
    kv_bytes = n_text_layer * 2 * n_text_ctx * n_text_state * BYTES_PER_FLOAT32
    return upload_bytes + pcm_bytes + mel_bytes + kv_bytes + REQUEST_OVERHEAD_BYTES


def format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.0f}"


def has_enough_ram(required_bytes: int, *, context: str = "request peak") -> Tuple[bool, str]:
    """Return (ok, reason). Unknown MemAvailable → allow (same as VLM)."""
    avail = read_mem_available_bytes()
    if avail is None:
        return True, ""
    if avail >= required_bytes:
        return True, ""
    reason = (
        f"insufficient RAM: need ~{format_mib(required_bytes)} MiB "
        f"(estimated {context}), MemAvailable ~{format_mib(avail)} MiB"
    )
    return False, reason


def estimate_encoder_worker_ram_bytes(encoder_path: PathLike) -> int:
    """RSS estimate for one RKNN encoder context (file × headroom)."""
    size = file_size_bytes(encoder_path)
    if size is None:
        return 400 * 1024 * 1024
    return int(size * MODEL_HEADROOM)


def estimate_encoder_pool_ram_bytes(
    encoder_path: PathLike,
    decoder_path: Optional[PathLike] = None,
    n_workers: int = 1,
) -> int:
    """
    Host RSS estimate for decoder + N encoder RKNN contexts.

    Mirrors video-descriptor ``estimateModelRamBytes`` (N × vision × 1.15 + llm).
    """
    n = max(1, n_workers)
    total = estimate_encoder_worker_ram_bytes(encoder_path) * n
    if decoder_path is not None:
        dec = file_size_bytes(decoder_path)
        if dec is not None:
            total += int(dec * MODEL_HEADROOM)
    return total


def pick_encoder_worker_count(
    encoder_path: PathLike,
    *,
    decoder_path: Optional[PathLike] = None,
    max_workers: int = 3,
    credit_bytes: int = 0,
) -> int:
    """
    Largest encoder worker count that fits MemAvailable (video-descriptor
    ``pickVisionWorkerCount``): try max → … → 1.

    Returns 0 if even one worker does not fit. Unknown MemAvailable → try
    ``max_workers`` (EncoderPool still probes down on NPU MALLOC_FAIL).
    """
    capped = max(1, int(max_workers))
    avail = read_mem_available_bytes()
    if avail is None:
        return capped
    avail_bytes = avail + max(0, credit_bytes)
    for n in range(capped, 0, -1):
        need = estimate_encoder_pool_ram_bytes(encoder_path, decoder_path, n)
        if avail_bytes >= need:
            return n
    return 0


def model_ram_check_message(required_bytes: int) -> str:
    avail = read_mem_available_bytes()
    if avail is None:
        return f"memory: estimated model RSS ~{format_mib(required_bytes)} MiB"
    return (
        f"memory: estimated model RSS ~{format_mib(required_bytes)} MiB, "
        f"MemAvailable ~{format_mib(avail)} MiB"
    )
