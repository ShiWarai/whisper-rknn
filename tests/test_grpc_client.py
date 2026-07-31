"""Тесты развёртывания gRPC-таргетов."""

from app.core.grpc_client import expand_targets


def test_expand_targets_passthrough_hostport():
    targets = expand_targets(["127.0.0.1:50051"])
    assert "127.0.0.1:50051" in targets
