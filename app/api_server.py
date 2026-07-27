#!/usr/bin/env python3
"""OpenAI-compatible Whisper RKNN HTTP API (hwdsl2/docker-whisper contract)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from app.auth import require_api_key
from app.decode import (
    RKNNModel,
    TaskType,
    _install_root,
    apply_librknnrt_from_optional_path,
    decode_utterance,
    decode_utterance_stream,
    load_audio_16k_mono,
    load_tokens,
    model_config_from_encoder_path,
    resolve_decoder_backend,
    resolve_librknnrt_path,
    stitch_transcripts,
)
from app.openai_response import format_transcription_response, stream_sse_frames
from app.system_memory import (
    estimate_model_ram_bytes,
    estimate_request_ram_bytes,
    has_enough_ram,
    model_ram_check_message,
)

logger = logging.getLogger("whisper_rknn_api")

_VALID_FORMATS = frozenset({"json", "text", "verbose_json", "srt", "vtt"})
_AUDIO_SUFFIXES = frozenset(
    {".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".webm", ".mp4", ".bin"}
)


def _restore_std_log_levels() -> None:
    """rknnlite modifies logging._nameToLevel; restore std names for uvicorn."""
    import logging as _logging

    _logging._nameToLevel.update(
        {
            "CRITICAL": _logging.CRITICAL,
            "FATAL": _logging.CRITICAL,
            "ERROR": _logging.ERROR,
            "WARN": _logging.WARNING,
            "WARNING": _logging.WARNING,
            "INFO": _logging.INFO,
            "DEBUG": _logging.DEBUG,
            "NOTSET": _logging.NOTSET,
        }
    )


_restore_std_log_levels()

_encoder_path = os.environ.get("WHISPER_ENCODER", "")
_decoder_path = os.environ.get("WHISPER_DECODER", "")
_decoder_backend = os.environ.get("WHISPER_DECODER_BACKEND", "auto")
_tokens_path = os.environ.get("WHISPER_TOKENS", "")
_host = os.environ.get("HOST", "0.0.0.0")
_port = int(os.environ.get("PORT", "8080"))
_max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "25"))

_model: Optional[RKNNModel] = None
_id2token: Optional[dict] = None
_model_name: str = "whisper-1"
_infer_lock = asyncio.Lock()


def _active_model_name() -> str:
    profile = (
        os.environ.get("WHISPER_MODEL_PROFILE")
        or os.environ.get("WHISPER_VARIANT")
        or ""
    ).strip()
    if profile:
        return profile
    models_dir = os.environ.get("WHISPER_MODELS_DIR", "").strip()
    if models_dir:
        return Path(models_dir).name
    return "whisper-1"


def _resolve_language(request_language: Optional[str]) -> Optional[str]:
    if request_language and request_language.strip().lower() not in ("", "auto"):
        return request_language.strip().lower()
    env_lang = os.environ.get("WHISPER_LANGUAGE", "ru").strip()
    if env_lang.lower() in ("", "auto"):
        return None
    return env_lang.lower()


def _needs_timestamps(response_format: str) -> bool:
    return response_format in ("verbose_json", "srt", "vtt")


async def _timestamp_granularities(request: Request) -> List[str]:
    form = await request.form()
    values = form.getlist("timestamp_granularities[]")
    if not values:
        return ["segment"]
    return values


def _validate_temperature(temperature: float) -> None:
    if not 0 <= temperature <= 1:
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 1.")


def _validate_timestamp_granularities(granularities: List[str]) -> None:
    if "word" in granularities:
        raise HTTPException(
            status_code=400,
            detail="Word-level timestamps are not supported; use segment only.",
        )
    allowed = {"segment"}
    if any(g not in allowed for g in granularities):
        raise HTTPException(
            status_code=400,
            detail="timestamp_granularities[] supports only 'segment'.",
        )


def _load_model_sync() -> None:
    global _model, _id2token, _model_name
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
        ("WHISPER_TOKENS", _tokens_path),
    ):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"{name} must point to an existing file: {path!r}")

    backend, resolved_decoder = resolve_decoder_backend(
        backend=_decoder_backend,
        decoder_path=_decoder_path or None,
        models_dir=os.environ.get("WHISPER_MODELS_DIR"),
    )
    print(f"decoder: backend={backend} path={resolved_decoder}")

    model_ram_need = estimate_model_ram_bytes(_encoder_path, resolved_decoder)
    ok, reason = has_enough_ram(model_ram_need, context="model RSS")
    print(model_ram_check_message(model_ram_need))
    if not ok:
        raise RuntimeError(reason)

    (
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
    ) = model_config_from_encoder_path(_encoder_path)
    _id2token = load_tokens(_tokens_path)
    _model_name = _active_model_name()
    _model = RKNNModel(
        encoder=_encoder_path,
        decoder=resolved_decoder,
        size_key=size_key,
        english_only=english_only,
        eot=eot,
        n_text_layer=n_text_layer,
        n_text_ctx=n_text_ctx,
        n_text_state=n_text_state,
        n_mels=n_mels,
        mel_time_frames=mel_time_frames,
        notimestamps_id=notimestamps_id,
        timestamp_begin=timestamp_begin,
        verbose=False,
        decoder_backend=backend,
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


app = FastAPI(
    title="Whisper RKNN",
    description="OpenAI-compatible speech-to-text API powered by RKNN (RK3588 NPU).",
    lifespan=lifespan,
)


@app.get("/health", include_in_schema=False)
async def health():
    ok = _model is not None and _id2token is not None
    if ok:
        return {"status": "ok", "model": _model_name}
    return {"status": "loading"}


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name,
                "object": "model",
                "created": 0,
                "owned_by": "whisper-rknn",
            }
        ],
    }


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    body = await file.read()
    max_bytes = _max_upload_mb * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {_max_upload_mb} MB)",
        )
    suffix = Path(file.filename or "audio").suffix or ".bin"
    if suffix.lower() not in _AUDIO_SUFFIXES:
        suffix = ".bin"
    return body, suffix


def _check_request_ram(body_len: int) -> None:
    if _model is None:
        return
    request_ram_need = estimate_request_ram_bytes(
        body_len,
        _model.n_mels,
        _model.n_text_layer,
        _model.n_text_ctx,
        _model.n_text_state,
    )
    ok, reason = has_enough_ram(request_ram_need)
    if not ok:
        raise HTTPException(status_code=507, detail=reason)


async def _handle_audio(
    task: TaskType,
    file: UploadFile,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    temperature: float,
    stream: Optional[str],
    timestamp_granularities: List[str],
) -> Response:
    if _model is None or _id2token is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    if task == "translate" and _model.english_only:
        raise HTTPException(
            status_code=400,
            detail=(
                "Translation is not supported with English-only models. "
                "Use a multilingual model."
            ),
        )

    _validate_temperature(temperature)
    _validate_timestamp_granularities(timestamp_granularities)

    if response_format not in _VALID_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid response_format '{response_format}'. "
                f"Must be one of: {', '.join(sorted(_VALID_FORMATS))}"
            ),
        )

    stream_flag = stream is not None and stream.strip().lower() == "true"
    if stream_flag and response_format not in ("json", "text"):
        raise HTTPException(
            status_code=400,
            detail="response_format is ignored when stream=true; use json or text.",
        )

    if prompt:
        logger.info("prompt parameter is accepted but not applied by RKNN decoder")

    body, suffix = await _read_upload(file)
    _check_request_ram(len(body))

    resolved_lang = _resolve_language(language)
    timestamps = _needs_timestamps(response_format) if not stream_flag else False
    cache_dir = _install_root() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    async with _infer_lock:
        loop = asyncio.get_running_loop()

        if stream_flag:
            def _stream_chunks():
                samples = load_audio_16k_mono(
                    io.BytesIO(body),
                    format_hint=suffix,
                    cache_dir=cache_dir,
                )
                chunk_texts: List[str] = []
                for chunk_result in decode_utterance_stream(
                    _model,
                    _id2token,
                    samples,
                    verbose=False,
                    timestamps=False,
                    task=task,
                    language=resolved_lang,
                ):
                    chunk_texts.append(chunk_result.text)
                full_text = stitch_transcripts(chunk_texts)
                return chunk_texts, full_text

            try:
                chunk_texts, full_text = await loop.run_in_executor(
                    None, _stream_chunks
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except FileNotFoundError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            frames = stream_sse_frames(chunk_texts, full_text)

            async def _event_stream():
                for frame in frames:
                    yield frame

            return StreamingResponse(
                _event_stream(),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                },
            )

        def _run():
            samples = load_audio_16k_mono(
                io.BytesIO(body),
                format_hint=suffix,
                cache_dir=cache_dir,
            )
            duration = len(samples) / 16000.0
            result = decode_utterance(
                _model,
                _id2token,
                samples,
                verbose=False,
                timestamps=timestamps,
                task=task,
                language=resolved_lang,
                collect_timings=True,
            )
            if result.timings is not None:
                logger.info("decode timings: %s", result.timings.to_dict())
            return result, duration

        try:
            result, duration = await loop.run_in_executor(None, _run)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    body_out, media_type = format_transcription_response(
        result,
        response_format=response_format,  # type: ignore[arg-type]
        task=task,
        language=resolved_lang,
        duration=duration,
    )
    if media_type == "text/plain":
        return PlainTextResponse(str(body_out), media_type=media_type)
    if media_type == "application/json":
        return JSONResponse(content=json.loads(body_out))
    return Response(content=body_out, media_type=media_type)


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_api_key)])
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    stream: Optional[str] = Form(default=None),
):
    timestamp_granularities = await _timestamp_granularities(request)
    return await _handle_audio(
        task="transcribe",
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        stream=stream,
        timestamp_granularities=timestamp_granularities,
    )


@app.post("/v1/audio/translations", dependencies=[Depends(require_api_key)])
async def translate(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    stream: Optional[str] = Form(default=None),
):
    return await _handle_audio(
        task="translate",
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        stream=stream,
        timestamp_granularities=["segment"],
    )


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
