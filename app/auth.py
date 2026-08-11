"""Аутентификация по API-ключу в стиле OpenAI (Bearer token)."""

from __future__ import annotations

import os
import secrets
from typing import FrozenSet, Optional

from fastapi import Header, HTTPException

_API_KEYS: Optional[FrozenSet[str]] = None


def parse_api_keys_from_env() -> FrozenSet[str]:
    raw = os.environ.get("WHISPER_API_KEY", "").strip()
    if not raw:
        raw = os.environ.get("OPENAI_API_KEY", "").strip()
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def configured_api_keys() -> FrozenSet[str]:
    global _API_KEYS
    if _API_KEYS is None:
        _API_KEYS = parse_api_keys_from_env()
    return _API_KEYS


def reload_api_keys() -> FrozenSet[str]:
    """Перечитать ключи из env (тесты / перезагрузка конфига)."""
    global _API_KEYS
    _API_KEYS = parse_api_keys_from_env()
    return _API_KEYS


def auth_enabled() -> bool:
    return bool(configured_api_keys())


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _auth_error(message: str, *, code: str = "invalid_api_key") -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
    )


def verify_api_key(authorization: Optional[str]) -> None:
    keys = configured_api_keys()
    if not keys:
        return

    token = extract_bearer_token(authorization)
    if token is None:
        raise _auth_error(
            "You didn't provide an API key. You need to provide your API key in an "
            "Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY)."
        )

    if not any(secrets.compare_digest(token, key) for key in keys):
        raise _auth_error("Incorrect API key provided: your_key")


async def require_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> None:
    verify_api_key(authorization)
