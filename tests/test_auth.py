"""Tests for OpenAI-style API key auth."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.auth import (
    auth_enabled,
    extract_bearer_token,
    reload_api_keys,
    verify_api_key,
)


@pytest.fixture(autouse=True)
def _clear_keys(monkeypatch):
    monkeypatch.delenv("WHISPER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reload_api_keys()


def test_auth_disabled_by_default():
    assert not auth_enabled()
    verify_api_key(None)


def test_whisper_api_key_enables_auth(monkeypatch):
    monkeypatch.setenv("WHISPER_API_KEY", "sk-one")
    reload_api_keys()
    assert auth_enabled()


def test_openai_api_key_alias(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    reload_api_keys()
    assert auth_enabled()
    verify_api_key("Bearer sk-openai")


def test_whisper_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("WHISPER_API_KEY", "sk-whisper")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    reload_api_keys()
    verify_api_key("Bearer sk-whisper")
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key("Bearer sk-openai")
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "invalid_api_key"


def test_multiple_keys(monkeypatch):
    monkeypatch.setenv("WHISPER_API_KEY", "sk-one, sk-two")
    reload_api_keys()
    verify_api_key("Bearer sk-two")


def test_extract_bearer_token():
    assert extract_bearer_token("Bearer abc") == "abc"
    assert extract_bearer_token("bearer abc") == "abc"
    assert extract_bearer_token("Token abc") is None
    assert extract_bearer_token(None) is None
