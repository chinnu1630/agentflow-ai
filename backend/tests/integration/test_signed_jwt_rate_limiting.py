"""End-to-end JWT authorization and Redis rate-limit API tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from jose import jwt

from app.api.dependencies.rate_limit import (
    build_identity_rate_limit_key,
)
from app.core.config import get_settings
from app.main import app

TEST_ISSUER = "https://identity.example.com/"
TEST_AUDIENCE = "agentflow-api"
TEST_HMAC_SECRET = "signed-jwt-rate-limit-secret"  # noqa: S105
TEST_SUBJECT = "release-manager-123"
TEST_EMAIL = "release.manager@example.com"


@pytest.fixture(autouse=True)
def configure_signed_jwt_rate_limiting(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Enable real JWT verification and deterministic Redis rate limiting."""
    private_key = rsa.generate_private_key(
        public_exponent=65_537,
        key_size=2_048,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_JWT_ISSUER", TEST_ISSUER)
    monkeypatch.setenv("AUTH_JWT_AUDIENCE", TEST_AUDIENCE)
    monkeypatch.setenv("AUTH_JWT_PUBLIC_KEY", public_pem)
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv(
        "RATE_LIMIT_KEY_HMAC_SECRET",
        TEST_HMAC_SECRET,
    )
    get_settings.cache_clear()

    app.dependency_overrides.clear()
    app.state.redis_client = None
    app.state.test_private_key = private_pem

    yield

    app.dependency_overrides.clear()
    app.state.redis_client = None

    if hasattr(app.state, "test_private_key"):
        del app.state.test_private_key

    get_settings.cache_clear()


def _build_access_token(*, scopes: str) -> str:
    """Create one valid externally issued RS256 bearer token."""
    now = datetime.now(UTC)

    return cast(
        str,
        jwt.encode(
            {
                "sub": TEST_SUBJECT,
                "email": TEST_EMAIL,
                "roles": ["release_manager"],
                "scope": scopes,
                "iss": TEST_ISSUER,
                "aud": TEST_AUDIENCE,
                "iat": now,
                "exp": now + timedelta(minutes=5),
            },
            app.state.test_private_key,
            algorithm="RS256",
        ),
    )


def _denying_redis_client() -> Mock:
    """Return a Redis-compatible client with an exhausted bucket."""
    redis_client = Mock()
    redis_client.eval = AsyncMock(
        return_value=[0, "0.0", 5_000],
    )
    return redis_client


def test_valid_signed_jwt_is_rate_limited_by_verified_subject() -> None:
    """A verified subject should map to its pseudonymous Redis bucket."""
    redis_client = _denying_redis_client()
    app.state.redis_client = redis_client
    token = _build_access_token(scopes="agent:query")

    response = TestClient(
        app,
        raise_server_exceptions=False,
    ).post(
        "/api/v1/agent/query-plan",
        json={
            "query": "What are the biggest release risks this week?",
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    redis_client.eval.assert_awaited_once()
    evaluated_key = redis_client.eval.await_args.args[2]
    expected_key = build_identity_rate_limit_key(
        subject=TEST_SUBJECT,
        policy_name="standard",
        hmac_secret=TEST_HMAC_SECRET,
    )

    assert evaluated_key == expected_key
    assert TEST_SUBJECT not in evaluated_key
    assert TEST_EMAIL not in evaluated_key


def test_missing_agent_scope_is_rejected_before_redis_evaluation() -> None:
    """Unauthorized callers must not consume distributed rate-limit state."""
    redis_client = _denying_redis_client()
    app.state.redis_client = redis_client
    token = _build_access_token(scopes="release:read")

    response = TestClient(
        app,
        raise_server_exceptions=False,
    ).post(
        "/api/v1/agent/query-plan",
        json={
            "query": "What are the biggest release risks this week?",
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "AUTHORIZATION_FAILED"
    redis_client.eval.assert_not_awaited()
