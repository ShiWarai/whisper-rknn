"""gRPC-сервер decode-воркера."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent import futures

import grpc

from app.core.grpc_gen.whisper_rknn.v1 import worker_pb2, worker_pb2_grpc
from app.core.tensor_codec import tensors_to_ndarrays
from app.decode import TaskType, decode_from_cross_kv, load_tokens
from app.decode_worker.session import DecodeSession
from app.system_memory import format_mib, has_enough_ram, read_mem_available_bytes

logger = logging.getLogger("whisper_decoder_worker")

_CROSS_KV_HEADROOM_BYTES = 96 * 1024 * 1024


class DecodeServicer(worker_pb2_grpc.DecodeServiceServicer):
    def __init__(self, session: DecodeSession, id2token: dict) -> None:
        self._session = session
        self._id2token = id2token
        self._inflight = 0
        self._lock = threading.Lock()

    def _mem_available_mib(self) -> float:
        avail = read_mem_available_bytes()
        if avail is None:
            return -1.0
        return avail / (1024 * 1024)

    def Health(self, request: worker_pb2.HealthRequest, context) -> worker_pb2.HealthResponse:
        del request, context
        with self._lock:
            inflight = self._inflight
        return worker_pb2.HealthResponse(
            ready=self._session.ready,
            inflight=inflight,
            mem_available_mib=self._mem_available_mib(),
            npu_core="",
            model_profile=self._session.size_key,
            service="decoder",
        )

    def Decode(self, request: worker_pb2.DecodeRequest, context) -> worker_pb2.DecodeResponse:
        ok, reason = has_enough_ram(_CROSS_KV_HEADROOM_BYTES, context="decode cross_kv")
        if not ok:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, reason)

        with self._lock:
            self._inflight += 1
        try:
            cross_kv = tensors_to_ndarrays(request.cross_kv)
            task: TaskType = "translate" if request.task == "translate" else "transcribe"
            language = request.language or None
            t0 = time.perf_counter()
            result = decode_from_cross_kv(
                self._session,
                self._id2token,
                cross_kv,
                verbose=False,
                timestamps=request.timestamps,
                time_offset=request.time_offset_sec,
                task=task,
                language=language,
            )
            decode_ms = (time.perf_counter() - t0) * 1000.0
            segments = [
                worker_pb2.Segment(
                    start_sec=segment.start,
                    end_sec=segment.end,
                    text=segment.text,
                )
                for segment in (result.segments or [])
            ]
            return worker_pb2.DecodeResponse(
                job_id=request.job_id,
                chunk_id=request.chunk_id,
                text=result.text,
                segments=segments,
                truncated=result.truncated,
                decode_ms=decode_ms,
            )
        except Exception as exc:
            logger.exception("decode failed job=%s chunk=%s", request.job_id, request.chunk_id)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        finally:
            with self._lock:
                self._inflight -= 1


def serve() -> None:
    from app.worker_runtime import apply_worker_cpu_affinity

    pinned = apply_worker_cpu_affinity()
    if pinned is not None:
        logger.info("worker cpu affinity: %s", sorted(pinned))

    decoder_path = os.environ.get("WHISPER_DECODER", "/models/decoder.onnx")
    tokens_path = os.environ.get("WHISPER_TOKENS", "/models/tokens.txt")
    host = os.environ.get("GRPC_HOST", "0.0.0.0")
    port = int(os.environ.get("GRPC_PORT", "50052"))
    max_workers = int(os.environ.get("GRPC_MAX_WORKERS", "2"))

    session = DecodeSession(
        decoder_path,
        decoder_backend=os.environ.get("WHISPER_DECODER_BACKEND", "onnx"),
        profile=os.environ.get("WHISPER_MODEL_PROFILE"),
    )
    from app.worker_runtime import resolve_onnx_intra_op_threads

    onnx_threads = resolve_onnx_intra_op_threads()
    id2token = load_tokens(tokens_path)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    worker_pb2_grpc.add_DecodeServiceServicer_to_server(
        DecodeServicer(session, id2token),
        server,
    )
    bind = f"{host}:{port}"
    server.add_insecure_port(bind)
    server.start()
    avail = read_mem_available_bytes()
    mem_msg = f"~{format_mib(avail)} MiB" if avail is not None else "unknown"
    logger.info(
        "decoder worker listening on %s profile=%s decoder=%s onnx_threads=%s MemAvailable=%s",
        bind,
        session.size_key,
        session.decoder_path,
        onnx_threads,
        mem_msg,
    )
    try:
        server.wait_for_termination()
    finally:
        session.release()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    serve()


if __name__ == "__main__":
    main()
