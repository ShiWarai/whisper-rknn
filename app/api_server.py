#!/usr/bin/env python3
"""HTTP API: POST /transcribe (multipart file) -> JSON { text, elapsed_s }."""

from __future__ import annotations

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.auth import require_api_key
from app.decode import (
    RKNNModel,
    _install_root,
    apply_librknnrt_from_optional_path,
    decode_utterance,
    load_audio_16k_mono,
    load_tokens,
    model_config_from_encoder_path,
    resolve_librknnrt_path,
)


def _restore_std_log_levels() -> None:
    """rknnlite modifies logging._nameToLevel; restore std names for uvicorn."""
    import logging

    logging._nameToLevel.update(
        {
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }
    )


_restore_std_log_levels()


class TranscribeResponse(BaseModel):
    text: str
    elapsed_s: float


_encoder_path = os.environ.get("WHISPER_ENCODER", "")
_decoder_path = os.environ.get("WHISPER_DECODER", "")
_tokens_path = os.environ.get("WHISPER_TOKENS", "")
_host = os.environ.get("HOST", "0.0.0.0")
_port = int(os.environ.get("PORT", "8080"))
_max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "25"))

_model: Optional[RKNNModel] = None
_id2token: Optional[dict] = None
_infer_lock = asyncio.Lock()


def _load_model_sync() -> None:
    global _model, _id2token
    models_dir = os.environ.get("WHISPER_MODELS_DIR", "").strip()
    if models_dir:
        print("model bundle:", Path(models_dir).name)
    else:
        print("model bundle: (set WHISPER_MODELS_DIR to host path for a clear folder name in logs)")

    apply_librknnrt_from_optional_path(os.environ.get("LIBRKNNRT_SO"))

    lib = resolve_librknnrt_path(None)
    if lib is not None:
        print("librknnrt (override):", lib)
    else:
        print("librknnrt: system paths (e.g. /usr/lib)")

    for name, path in (
        ("WHISPER_ENCODER", _encoder_path),
        ("WHISPER_DECODER", _decoder_path),
        ("WHISPER_TOKENS", _tokens_path),
    ):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"{name} must point to an existing file: {path!r}")

    (
        sot_sequence,
        eot,
        n_text_layer,
        n_text_ctx,
        n_text_state,
        n_mels,
        mel_time_frames,
    ) = model_config_from_encoder_path(_encoder_path)
    _id2token = load_tokens(_tokens_path)
    _model = RKNNModel(
        encoder=_encoder_path,
        decoder=_decoder_path,
        sot_sequence=sot_sequence,
        eot=eot,
        n_text_layer=n_text_layer,
        n_text_ctx=n_text_ctx,
        n_text_state=n_text_state,
        n_mels=n_mels,
        mel_time_frames=mel_time_frames,
        verbose=False,
    )
    from app.auth import auth_enabled

    if auth_enabled():
        print("api auth: enabled (Bearer WHISPER_API_KEY / OPENAI_API_KEY)")
    else:
        print("api auth: disabled (set WHISPER_API_KEY to require Authorization header)")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_model_sync)
    yield
    global _model
    if _model is not None:
        await loop.run_in_executor(None, _model.release)
        _model = None


app = FastAPI(title="Whisper RKNN", lifespan=lifespan)


@app.get("/health")
async def health():
    ok = _model is not None and _id2token is not None
    return {"status": "ok" if ok else "loading"}


@app.post("/transcribe", response_model=TranscribeResponse, dependencies=[Depends(require_api_key)])
async def transcribe(file: UploadFile = File(...)):
    if _model is None or _id2token is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    body = await file.read()
    max_bytes = _max_upload_mb * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {_max_upload_mb} MB)",
        )

    suffix = Path(file.filename or "audio").suffix or ".bin"
    if suffix.lower() not in (
        ".wav",
        ".flac",
        ".ogg",
        ".opus",
        ".mp3",
        ".m4a",
        ".aac",
        ".webm",
        ".mp4",
        ".bin",
    ):
        suffix = ".bin"

    cache_dir = _install_root() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    async with _infer_lock:
        loop = asyncio.get_running_loop()

        def _run():
            t0 = time.time()
            samples = load_audio_16k_mono(
                io.BytesIO(body),
                format_hint=suffix,
                cache_dir=cache_dir,
            )
            text = decode_utterance(_model, _id2token, samples, verbose=False)
            elapsed = time.time() - t0
            return text, elapsed

        try:
            text, elapsed = await loop.run_in_executor(None, _run)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return TranscribeResponse(text=text, elapsed_s=round(elapsed, 3))


def main():
    uvicorn.run(
        "app.api_server:app",
        host=_host,
        port=_port,
        reload=False,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
