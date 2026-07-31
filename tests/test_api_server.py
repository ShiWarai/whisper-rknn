"""API tests with mocked RKNN model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth import reload_api_keys
from app.decode import DecodeResult, TranscriptSegment


def _fake_load_model_sync():
    import app.api_server as srv

    model = MagicMock(name="RKNNModel")
    model.n_mels = 128
    model.n_text_layer = 4
    model.n_text_ctx = 448
    model.n_text_state = 1280
    model.english_only = False
    srv._model = model
    srv._id2token = {1: "aGVsbG8="}
    srv._model_name = "turbo"


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
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "turbo"


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "turbo"


def test_transcriptions_json(client):
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch(
            "app.api_server.decode_utterance",
            return_value=DecodeResult(text="привет"),
        ),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={"model": "whisper-1"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "привет"}


def test_transcriptions_verbose_json(client):
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
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={
                "model": "whisper-1",
                "response_format": "verbose_json",
                "language": "ru",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "привет мир"
    assert body["task"] == "transcribe"
    assert len(body["segments"]) == 2
    assert mock_decode.call_args.kwargs.get("timestamps") is True


def test_transcriptions_text_format(client):
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch(
            "app.api_server.decode_utterance",
            return_value=DecodeResult(text="hello"),
        ),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={"response_format": "text"},
        )

    assert response.status_code == 200
    assert response.text == "hello"


def test_transcriptions_srt(client):
    result = DecodeResult(
        text="hello",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="hello")],
    )
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch("app.api_server.decode_utterance", return_value=result),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={"response_format": "srt"},
        )

    assert response.status_code == 200
    assert "WEBVTT" not in response.text
    assert "-->" in response.text


def test_transcriptions_stream_sse(client):
    chunks = [
        DecodeResult(text="привет"),
        DecodeResult(text="мир"),
    ]

    def _fake_stream(*_args, **_kwargs):
        yield from chunks

    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch("app.api_server.decode_utterance_stream", side_effect=_fake_stream),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={"stream": "true"},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    body = response.text
    assert "transcript.text.delta" in body
    assert "transcript.text.done" in body
    assert "[DONE]" in body


def test_translations(client):
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
        patch(
            "app.api_server.decode_utterance",
            return_value=DecodeResult(text="hello"),
        ) as mock_decode,
    ):
        response = client.post(
            "/v1/audio/translations",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={"model": "whisper-1"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "hello"}
    assert mock_decode.call_args.kwargs.get("task") == "translate"


def test_word_timestamps_rejected(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
        data={"timestamp_granularities[]": "word"},
    )
    assert response.status_code == 400
    assert "word" in response.json()["detail"].lower()


def test_transcriptions_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("WHISPER_API_KEY", "sk-test-secret")
    reload_api_keys()

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["error"]["code"] == "invalid_api_key"


def test_transcriptions_accepts_bearer_key(client, monkeypatch):
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
            return_value=DecodeResult(text="привет"),
        ),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            headers={"Authorization": "Bearer sk-test-secret"},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "привет"


def test_transcriptions_model_not_ready():
    with patch("app.api_server._load_model_sync"):
        import app.api_server as srv
        from app.api_server import app

        srv._model = None
        srv._id2token = None

        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("voice.ogg", b"x", "audio/ogg")},
            )

    assert response.status_code == 503


def test_transcriptions_insufficient_ram_returns_507(client):
    with patch(
        "app.api_server.has_enough_ram",
        return_value=(False, "insufficient RAM: need ~512 MiB (estimated request peak)"),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
        )

    assert response.status_code == 507
    assert "insufficient RAM" in response.json()["detail"]


def test_transcriptions_english_only_translate_rejected(client):
    import app.api_server as srv

    srv._model.english_only = True
    response = client.post(
        "/v1/audio/translations",
        files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
    )
    assert response.status_code == 400
    assert "English-only" in response.json()["detail"]
