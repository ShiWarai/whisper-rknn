"""Пулы gRPC-клиентов с выбором воркера по наименьшему inflight."""

from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass
from typing import List, Optional, Sequence

import grpc

from app.core.grpc_gen.whisper_rknn.v1 import worker_pb2, worker_pb2_grpc

DEFAULT_MAX_MESSAGE = 256 * 1024 * 1024


def grpc_channel_options() -> list:
    return [
        ("grpc.max_send_message_length", DEFAULT_MAX_MESSAGE),
        ("grpc.max_receive_message_length", DEFAULT_MAX_MESSAGE),
    ]


def parse_targets(env_name: str, default: str = "") -> List[str]:
    raw = os.environ.get(env_name, default).strip()
    if not raw:
        return []
    return [target.strip() for target in raw.split(",") if target.strip()]


def expand_targets(targets: Sequence[str]) -> List[str]:
    """Разрешить hostname (в т.ч. headless k8s Service) в список host:port."""
    expanded: List[str] = []
    for target in targets:
        host, sep, port = target.rpartition(":")
        if not sep:
            host, port = target, "50051"
        try:
            port_num = int(port)
        except ValueError:
            expanded.append(target)
            continue
        try:
            infos = socket.getaddrinfo(
                host,
                port_num,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror:
            expanded.append(target)
            continue
        for info in infos:
            address = info[4][0]
            expanded.append(f"{address}:{port}")
    if not expanded:
        return list(targets)
    return list(dict.fromkeys(expanded))


@dataclass
class _Endpoint:
    target: str
    inflight: int = 0
    channel: Optional[grpc.aio.Channel] = None
    encode_stub: Optional[worker_pb2_grpc.EncodeServiceStub] = None
    decode_stub: Optional[worker_pb2_grpc.DecodeServiceStub] = None

    def acquire(self) -> None:
        self.inflight += 1

    def release(self) -> None:
        self.inflight = max(0, self.inflight - 1)


class _BasePool:
    def __init__(self, targets: Sequence[str]):
        if not targets:
            raise ValueError("at least one gRPC target is required")
        self._endpoints = [_Endpoint(target=target) for target in targets]
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        for endpoint in self._endpoints:
            endpoint.channel = grpc.aio.insecure_channel(
                endpoint.target,
                options=grpc_channel_options(),
            )

    async def close(self) -> None:
        for endpoint in self._endpoints:
            if endpoint.channel is not None:
                await endpoint.channel.close()
                endpoint.channel = None

    async def _pick(self) -> _Endpoint:
        async with self._lock:
            return min(self._endpoints, key=lambda endpoint: endpoint.inflight)


class EncodeClientPool(_BasePool):
    async def connect(self) -> None:
        await super().connect()
        for endpoint in self._endpoints:
            assert endpoint.channel is not None
            endpoint.encode_stub = worker_pb2_grpc.EncodeServiceStub(endpoint.channel)

    async def encode(self, request: worker_pb2.EncodeRequest) -> worker_pb2.EncodeResponse:
        endpoint = await self._pick()
        endpoint.acquire()
        try:
            assert endpoint.encode_stub is not None
            return await endpoint.encode_stub.Encode(request)
        finally:
            endpoint.release()

    async def encode_then_decode(
        self, request: worker_pb2.EncodeThenDecodeRequest
    ) -> worker_pb2.DecodeResponse:
        endpoint = await self._pick()
        endpoint.acquire()
        try:
            assert endpoint.encode_stub is not None
            return await endpoint.encode_stub.EncodeThenDecode(request)
        finally:
            endpoint.release()

    async def health(self) -> List[worker_pb2.HealthResponse]:
        responses: List[worker_pb2.HealthResponse] = []
        for endpoint in self._endpoints:
            assert endpoint.encode_stub is not None
            responses.append(await endpoint.encode_stub.Health(worker_pb2.HealthRequest()))
        return responses


class DecodeClientPool(_BasePool):
    async def connect(self) -> None:
        await super().connect()
        for endpoint in self._endpoints:
            assert endpoint.channel is not None
            endpoint.decode_stub = worker_pb2_grpc.DecodeServiceStub(endpoint.channel)

    async def decode(self, request: worker_pb2.DecodeRequest) -> worker_pb2.DecodeResponse:
        endpoint = await self._pick()
        endpoint.acquire()
        try:
            assert endpoint.decode_stub is not None
            return await endpoint.decode_stub.Decode(request)
        finally:
            endpoint.release()

    async def acquire(self) -> _Endpoint:
        endpoint = await self._pick()
        endpoint.acquire()
        return endpoint

    async def release(self, endpoint: _Endpoint) -> None:
        endpoint.release()

    async def health(self) -> List[worker_pb2.HealthResponse]:
        responses: List[worker_pb2.HealthResponse] = []
        for endpoint in self._endpoints:
            assert endpoint.decode_stub is not None
            responses.append(await endpoint.decode_stub.Health(worker_pb2.HealthRequest()))
        return responses
