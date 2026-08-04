"""gRPC-сервер encoder-воркера."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent import futures
from typing import Dict

import grpc

from app.core.grpc_client import grpc_channel_options
from app.core.grpc_gen.whisper_rknn.v1 import worker_pb2, worker_pb2_grpc
from app.core.tensor_codec import ndarray_to_tensor, tensor_to_ndarray
from app.encode_worker.session import EncodeSession
from app.system_memory import format_mib, has_enough_ram, read_mem_available_bytes

logger = logging.getLogger("whisper_encoder_worker")

_CROSS_KV_HEADROOM_BYTES = 96 * 1024 * 1024
_decode_stubs: Dict[str, worker_pb2_grpc.DecodeServiceStub] = {}
_decode_lock = threading.Lock()


def _decode_stub(target: str) -> worker_pb2_grpc.DecodeServiceStub:
    with _decode_lock:
        stub = _decode_stubs.get(target)
        if stub is None:
            channel = grpc.insecure_channel(target, options=grpc_channel_options())
            stub = worker_pb2_grpc.DecodeServiceStub(channel)
            _decode_stubs[target] = stub
        return stub


class EncodeServicer(worker_pb2_grpc.EncodeServiceServicer):
    def __init__(self, session: EncodeSession) -> None:
        self._session = session
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
            npu_core=str(self._session.encoder_workers),
            model_profile=self._session.profile.size_key,
            service="encoder",
        )

    def Encode(self, request: worker_pb2.EncodeRequest, context) -> worker_pb2.EncodeResponse:
        ok, reason = has_enough_ram(_CROSS_KV_HEADROOM_BYTES, context="encode cross_kv")
        if not ok:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, reason)

        with self._lock:
            self._inflight += 1
        try:
            from app.core.tensor_codec import tensor_to_ndarray

            mel = tensor_to_ndarray(request.mel)
            t0 = time.perf_counter()
            cross_kv = self._session.encode(mel)
            encode_ms = (time.perf_counter() - t0) * 1000.0
            tensors = [ndarray_to_tensor(arr) for arr in cross_kv]
            return worker_pb2.EncodeResponse(
                job_id=request.job_id,
                chunk_id=request.chunk_id,
                cross_kv=tensors,
                encode_ms=encode_ms,
            )
        except Exception as exc:
            logger.exception("encode failed job=%s chunk=%s", request.job_id, request.chunk_id)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        finally:
            with self._lock:
                self._inflight -= 1

    def EncodeThenDecode(
        self, request: worker_pb2.EncodeThenDecodeRequest, context
    ) -> worker_pb2.DecodeResponse:
        ok, reason = has_enough_ram(_CROSS_KV_HEADROOM_BYTES, context="encode cross_kv")
        if not ok:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, reason)
        if not request.decode_target:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "decode_target is required")

        with self._lock:
            self._inflight += 1
        try:
            mel = tensor_to_ndarray(request.mel)
            t0 = time.perf_counter()
            cross_kv = self._session.encode(mel)
            encode_ms = (time.perf_counter() - t0) * 1000.0
            tensors = [ndarray_to_tensor(arr) for arr in cross_kv]
            dec_request = worker_pb2.DecodeRequest(
                job_id=request.job_id,
                chunk_id=request.chunk_id,
                cross_kv=tensors,
                language=request.language,
                task=request.task,
                timestamps=False,
                time_offset_sec=request.time_offset_sec,
                model_profile=request.model_profile,
            )
            dec_response = _decode_stub(request.decode_target).Decode(dec_request)
            return worker_pb2.DecodeResponse(
                job_id=dec_response.job_id,
                chunk_id=dec_response.chunk_id,
                text=dec_response.text,
                segments=dec_response.segments,
                truncated=dec_response.truncated,
                decode_ms=dec_response.decode_ms,
                encode_ms=encode_ms,
            )
        except Exception as exc:
            logger.exception(
                "encode_then_decode failed job=%s chunk=%s target=%s",
                request.job_id,
                request.chunk_id,
                request.decode_target,
            )
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        finally:
            with self._lock:
                self._inflight -= 1


def serve() -> None:
    from app.worker_runtime import apply_worker_cpu_affinity

    pinned = apply_worker_cpu_affinity()
    if pinned is not None:
        logger.info("worker cpu affinity: %s", sorted(pinned))

    encoder_path = os.environ.get("WHISPER_ENCODER", "/models/encoder.rknn")
    host = os.environ.get("GRPC_HOST", "0.0.0.0")
    port = int(os.environ.get("GRPC_PORT", "50051"))
    max_workers = int(os.environ.get("GRPC_MAX_WORKERS", "4"))

    session = EncodeSession(encoder_path)
    from app.encode_pool import allowed_npu_core_masks

    npu_masks = allowed_npu_core_masks()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    worker_pb2_grpc.add_EncodeServiceServicer_to_server(EncodeServicer(session), server)
    bind = f"{host}:{port}"
    server.add_insecure_port(bind)
    server.start()
    avail = read_mem_available_bytes()
    mem_msg = f"~{format_mib(avail)} MiB" if avail is not None else "unknown"
    logger.info(
        "encoder worker listening on %s profile=%s npu_workers=%s npu_cores=%s MemAvailable=%s",
        bind,
        session.profile.size_key,
        session.encoder_workers,
        [hex(m) for m in npu_masks],
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
