"""Тесты API с замоканным ASR-бэкендом."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth import reload_api_keys
from app.core.types import DecodeResult, TranscriptSegment


def _make_backend(*, english_only: bool = False) -> MagicMock:
    backend = MagicMock(name="AsrBackend")
    backend.english_only = english_only
    backend.profile = MagicMock(
        n_mels=128,
        n_text_layer=4,
        n_text_ctx=448,
        n_text_state=1280,
    )
    backend.decode_utterance = AsyncMock(return_value=DecodeResult(text="привет"))
    backend.decode_utterance_stream = AsyncMock()
    backend.shutdown = AsyncMock()
    return backend


async def _fake_startup() -> None:
    import app.api_server as srv

    srv._backend = _make_backend()
    srv._model_name = "turbo"


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("WHISPER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reload_api_keys()


@pytest.fixture
def client():
    with patch("app.api_server._startup", _fake_startup):
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
    import app.api_server as srv

    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={"model": "whisper-1"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "привет"}
    srv._backend.decode_utterance.assert_awaited_once()


def test_transcriptions_verbose_json(client):
    import app.api_server as srv

    result = DecodeResult(
        text="привет мир",
        segments=[
            TranscriptSegment(start=0.0, end=1.2, text="привет"),
            TranscriptSegment(start=1.2, end=2.0, text="мир"),
        ],
    )
    srv._backend.decode_utterance = AsyncMock(return_value=result)
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
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
    assert srv._backend.decode_utterance.await_args.kwargs.get("timestamps") is True


def test_transcriptions_text_format(client):
    import app.api_server as srv

    srv._backend.decode_utterance = AsyncMock(return_value=DecodeResult(text="hello"))
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
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
    import app.api_server as srv

    result = DecodeResult(
        text="hello",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="hello")],
    )
    srv._backend.decode_utterance = AsyncMock(return_value=result)
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
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
    import app.api_server as srv

    chunks = [
        DecodeResult(text="привет"),
        DecodeResult(text="мир"),
    ]

    async def _fake_stream(*_args, **_kwargs):
        for chunk in chunks:
            yield chunk

    srv._backend.decode_utterance_stream = _fake_stream
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
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
    import app.api_server as srv

    srv._backend.decode_utterance = AsyncMock(return_value=DecodeResult(text="hello"))
    with (
        patch("app.api_server.has_enough_ram", return_value=(True, "")),
        patch(
            "app.api_server.load_audio_16k_mono",
            return_value=__import__("numpy").zeros(16000, dtype="float32"),
        ),
    ):
        response = client.post(
            "/v1/audio/translations",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            data={"model": "whisper-1"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "hello"}
    assert srv._backend.decode_utterance.await_args.kwargs.get("task") == "translate"


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
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
            headers={"Authorization": "Bearer sk-test-secret"},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "привет"


def test_transcriptions_model_not_ready():
    async def _noop_startup() -> None:
        return None

    with patch("app.api_server._startup", _noop_startup):
        import app.api_server as srv
        from app.api_server import app

        srv._backend = None

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

    srv._backend.english_only = True
    response = client.post(
        "/v1/audio/translations",
        files={"file": ("voice.ogg", b"fake-audio", "audio/ogg")},
    )
    assert response.status_code == 400
    assert "English-only" in response.json()["detail"]
