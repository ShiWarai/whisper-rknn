"""Тесты roundtrip tensor codec."""

from __future__ import annotations

import numpy as np

from app.core.tensor_codec import (
    ndarray_to_tensor,
    ndarrays_to_tensors,
    tensor_to_ndarray,
    tensors_to_ndarrays,
)


def test_tensor_roundtrip_mel_shape():
    mel = np.random.randn(1, 128, 3000).astype(np.float32)
    restored = tensor_to_ndarray(ndarray_to_tensor(mel))
    np.testing.assert_array_equal(restored, mel)


def test_tensor_roundtrip_cross_kv_list():
    arrays = [
        np.random.randn(1, 1500, 1280).astype(np.float32),
        np.random.randn(1, 1500, 1280).astype(np.float32),
    ]
    restored = tensors_to_ndarrays(ndarrays_to_tensors(arrays))
    assert len(restored) == 2
    for original, decoded in zip(arrays, restored, strict=True):
        np.testing.assert_array_equal(decoded, original)


def test_tensor_empty_shape_rejected():
  import pytest

  from app.core.grpc_gen.whisper_rknn.v1 import worker_pb2

  tensor = worker_pb2.Tensor(shape=[], dtype="float32", data=b"")
  with pytest.raises(ValueError, match="empty"):
      tensor_to_ndarray(tensor)
