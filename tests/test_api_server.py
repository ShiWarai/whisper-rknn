"""API tests with mocked RKNN model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth import reload_api_keys


def _fake_load_model_sync():
    import app.api_server as srv

    model = MagicMock(name="RKNNModel")
    model.n_mels = 80
    model.n_text_layer = 4
    model.n_text_ctx = 448
    model.n_text_state = 384
    srv._model = model
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
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch(
            "app.api_server.decode_utterance",
            return_value=__import__("app.decode", fromlist=["DecodeResult"]).DecodeResult(
                text="привет"
            ),
        ),
    ):
        response = client.post(
            "/transcribe",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "привет"
    assert body.get("segments") is None
    assert "elapsed_s" in body
    assert isinstance(body["elapsed_s"], float)


def test_transcribe_with_timestamps(client):
    from app.decode import DecodeResult, TranscriptSegment

    result = DecodeResult(
        text="привет мир",
        segments=[
            TranscriptSegment(start=0.0, end=1.2, text="привет"),
            TranscriptSegment(start=1.2, end=2.0, text="мир"),
        ],
    )
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch("app.api_server.decode_utterance", return_value=result) as mock_decode,
    ):
        response = client.post(
            "/transcribe",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={"timestamps": "true"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "привет мир"
    assert body["segments"] == [
        {"start": 0.0, "end": 1.2, "text": "привет"},
        {"start": 1.2, "end": 2.0, "text": "мир"},
    ]
    assert mock_decode.call_args.kwargs.get("timestamps") is True


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
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch(
            "app.api_server.decode_utterance",
            return_value=__import__("app.decode", fromlist=["DecodeResult"]).DecodeResult(
                text="привет"
            ),
        ),
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


def test_transcribe_insufficient_ram_returns_507(client):
    with patch(
        "app.api_server.has_enough_ram",
        return_value=(False, "insufficient RAM: need ~512 MiB (estimated request peak)"),
    ):
        response = client.post(
            "/transcribe",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
        )

    assert response.status_code == 507
    assert "insufficient RAM" in response.json()["detail"]
