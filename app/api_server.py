#!/usr/bin/env python3
"""HTTP API Whisper RKNN, совместимый с OpenAI (контракт hwdsl2/docker-whisper)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from app.audio_io import load_audio_16k_mono
from app.auth import require_api_key
from app.core.model_config import ModelProfile, model_config_from_encoder_path
from app.core.types import TaskType
from app.decode import (
    RKNNModel,
    apply_librknnrt_from_optional_path,
    load_tokens,
    parallel_encode_enabled,
    resolve_decoder_backend,
    resolve_librknnrt_path,
)
from app.encode_pool import resolve_encoder_worker_count
from app.openai_response import format_transcription_response, stream_transcription_sse
from app.runtime.backend import AsrBackend, GrpcBackend, LocalBackend, runtime_mode
from app.system_memory import (
    estimate_encoder_pool_ram_bytes,
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
    """rknnlite меняет logging._nameToLevel; восстанавливаем имена для uvicorn."""
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

_max_inflight_jobs = max(1, int(os.environ.get("MAX_INFLIGHT_JOBS", "2")))

_backend: Optional[AsrBackend] = None
_model: Optional[RKNNModel] = None
_id2token: Optional[dict] = None
_model_name: str = "whisper-1"
_infer_lock = asyncio.Lock()
_job_semaphore = asyncio.Semaphore(_max_inflight_jobs)
_runtime = runtime_mode()


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
    from app.worker_runtime import apply_worker_cpu_affinity

    pinned = apply_worker_cpu_affinity()
    if pinned is not None:
        print(f"worker cpu affinity: {sorted(pinned)}")
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

    n_workers = 1
    if parallel_encode_enabled():
        n_workers = resolve_encoder_worker_count(
            _encoder_path,
            decoder_path=resolved_decoder,
        )
        if n_workers < 1:
            need1 = estimate_encoder_pool_ram_bytes(_encoder_path, resolved_decoder, 1)
            ok1, reason1 = has_enough_ram(need1, context="model RSS (1 encoder)")
            raise RuntimeError(
                reason1
                or f"insufficient RAM for encoder pool (need ~{need1 // (1024*1024)} MiB)"
            )
        model_ram_need = estimate_encoder_pool_ram_bytes(
            _encoder_path, resolved_decoder, n_workers
        )
        print(
            f"encoder_pool: MemAvailable pick={n_workers} "
            f"(~{model_ram_need // (1024 * 1024)} MiB RSS est.; NPU may reduce)"
        )
    else:
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
        encoder_workers=n_workers if parallel_encode_enabled() else None,
    )
    if parallel_encode_enabled():
        from app.speech_cut import preload_vad, resolve_vad_model_path

        t0 = time.perf_counter()
        preload_vad()
        vad_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"encoder_pool: {_model.encoder_workers} worker(s) "
            f"[shared weights]; silero_vad ready "
            f"({resolve_vad_model_path()}, {vad_ms:.0f} ms)"
        )
    _model_name = _active_model_name()


def _install_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _check_request_ram(body_len: int) -> None:
    if _backend is None:
        return
    profile = _backend.profile
    request_ram_need = estimate_request_ram_bytes(
        body_len,
        profile.n_mels,
        profile.n_text_layer,
        profile.n_text_ctx,
        profile.n_text_state,
    )
    ok, reason = has_enough_ram(request_ram_need)
    if not ok:
        raise HTTPException(status_code=507, detail=reason)


async def _startup() -> None:
    global _backend, _model, _id2token, _model_name
    loop = asyncio.get_running_loop()

    if _runtime == "distributed":
        _backend = await GrpcBackend.create()
        _model_name = _active_model_name()
        await loop.run_in_executor(None, _preload_vad)
        print(
            f"api: distributed runtime profile={_backend.profile.size_key} "
            f"model={_model_name}"
        )
    else:
        await loop.run_in_executor(None, _load_model_sync)
        profile = ModelProfile.from_encoder_path(_encoder_path)
        assert _model is not None and _id2token is not None
        _backend = LocalBackend(_model, _id2token, profile)
        print(f"api: local runtime profile={profile.size_key} model={_model_name}")

    _log_auth_state()


def _preload_vad() -> None:
    from app.speech_cut import preload_vad

    preload_vad()


def _log_auth_state() -> None:
    from app.auth import auth_enabled

    if auth_enabled():
        print("api auth: enabled (Bearer WHISPER_API_KEY / OPENAI_API_KEY)")
    else:
        print("api auth: disabled (set WHISPER_API_KEY to require Authorization header)")


async def _shutdown() -> None:
    global _backend, _model, _id2token
    if _backend is not None:
        await _backend.shutdown()
        _backend = None
    _model = None
    _id2token = None
    from app.speech_cut import release_vad

    release_vad()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _startup()
    yield
    await _shutdown()


app = FastAPI(
    title="Whisper RKNN",
    description="OpenAI-compatible speech-to-text API powered by RKNN (RK3588 NPU).",
    lifespan=lifespan,
)


@app.get("/health", include_in_schema=False)
async def health():
    ok = _backend is not None
    if ok:
        payload = {"status": "ok", "model": _model_name}
        if _runtime == "distributed":
            payload["runtime"] = "distributed"
        return payload
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
    if _backend is None:
        raise HTTPException(status_code=503, detail="API not ready")

    if task == "translate" and _backend.english_only:
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

    # Decode outside the infer lock: PyAV does not need NPU, and dropping `body`
    # right after PCM avoids holding upload+PCM together through the whole job.
    try:
        samples = load_audio_16k_mono(
            io.BytesIO(body),
            format_hint=suffix,
            cache_dir=cache_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        del body

    duration = len(samples) / 16000.0
    lock = _job_semaphore if _runtime == "distributed" else _infer_lock

    if stream_flag:
        # Hold the lock for the whole SSE lifetime so local NPU stays single-flight
        # (previously StreamingResponse returned inside `async with` and released early).
        pcm = samples

        async def _sse_under_lock():
            async with lock:
                async for frame in stream_transcription_sse(
                    _backend,
                    pcm,
                    task=task,
                    language=resolved_lang,
                ):
                    yield frame

        return StreamingResponse(
            _sse_under_lock(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    async with lock:
        try:
            result = await _backend.decode_utterance(
                samples,
                timestamps=timestamps,
                task=task,
                language=resolved_lang,
                collect_timings=True,
            )
            if result.timings is not None:
                logger.info("decode timings: %s", result.timings.to_dict())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
