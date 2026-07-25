"""API tests with mocked RKNN model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth import reload_api_keys


def _fake_load_model_sync():
    import app.api_server as srv

    srv._model = MagicMock(name="RKNNModel")
    srv._id2token = {1: "aGVsbG8="}


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("WHISPER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reload_api_keys()


@pytest.fixture
def client():
    with patch("app.api_server._load_model_sync", _fake_load_model_sync):
        from app.api_server import app

        with TestClient(app) as test_client:
            yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_returns_text(client):
    with (
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch("app.api_server.decode_utterance", return_value="привет"),
    ):
        response = client.post(
            "/transcribe",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "привет"
    assert "elapsed_s" in body
    assert isinstance(body["elapsed_s"], float)


def test_transcribe_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("WHISPER_API_KEY", "sk-test-secret")
    reload_api_keys()

    response = client.post(
        "/transcribe",
        files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["error"]["code"] == "invalid_api_key"


def test_transcribe_accepts_bearer_key(client, monkeypatch):
    monkeypatch.setenv("WHISPER_API_KEY", "sk-test-secret")
    reload_api_keys()

    with (
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch("app.api_server.decode_utterance", return_value="привет"),
    ):
        response = client.post(
            "/transcribe",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            headers={"Authorization": "Bearer sk-test-secret"},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "привет"


def test_transcribe_model_not_ready():
    with patch("app.api_server._load_model_sync"):
        import app.api_server as srv
        from app.api_server import app

        srv._model = None
        srv._id2token = None

        with TestClient(app) as client:
            response = client.post(
                "/transcribe",
                files={"file": ("voice.ogg", b"x", "audio/ogg")},
            )

    assert response.status_code == 503
