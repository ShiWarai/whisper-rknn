"""Сериализация NumPy ↔ protobuf Tensor для gRPC."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from app.core.grpc_gen.whisper_rknn.v1 import worker_pb2


def ndarray_to_tensor(arr: np.ndarray, *, dtype: str = "float32") -> worker_pb2.Tensor:
    """Сериализовать ndarray в little-endian raw bytes."""
    contiguous = np.ascontiguousarray(arr, dtype=np.dtype(dtype))
    return worker_pb2.Tensor(
        shape=[int(x) for x in contiguous.shape],
        dtype=str(contiguous.dtype),
        data=contiguous.tobytes(order="C"),
    )


def tensor_to_ndarray(tensor: worker_pb2.Tensor) -> np.ndarray:
    """Десериализовать protobuf Tensor в ndarray."""
    if not tensor.data:
        raise ValueError("tensor data is empty")
    dtype = np.dtype(tensor.dtype or "float32")
    arr = np.frombuffer(tensor.data, dtype=dtype)
    shape = tuple(int(x) for x in tensor.shape)
    if shape:
        arr = arr.reshape(shape)
    return np.ascontiguousarray(arr)


def ndarrays_to_tensors(arrays: Sequence[np.ndarray]) -> List[worker_pb2.Tensor]:
    return [ndarray_to_tensor(arr) for arr in arrays]


def tensors_to_ndarrays(tensors: Iterable[worker_pb2.Tensor]) -> List[np.ndarray]:
    return [tensor_to_ndarray(tensor) for tensor in tensors]
