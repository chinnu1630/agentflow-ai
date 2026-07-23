"""Tests for FastAPI authentication and scope-authorization dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from app.api.dependencies.security import (
    get_current_principal,
    require_scopes,
)
from app.api.exception_handlers import register_exception_handlers
from app.core.config import get_settings
from app.core.security import AuthenticatedPrincipal
from app.middleware.request_context import RequestContextMiddleware


@pytest.fixture(autouse=True)
def clear_cached_settings() -> Iterator[None]:
    """Prevent environment-backed settings from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def rsa_key_pair() -> tuple[str, str]:
    """Create one ephemeral RSA key pair for bearer-token tests."""
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

    return private_pem, public_pem


AuthenticatedPrincipalDependency = Annotated[
    AuthenticatedPrincipal,
    Depends(get_current_principal),
]
ReleaseApprovalPrincipalDependency = Annotated[
    AuthenticatedPrincipal,
    Depends(require_scopes("release:approve")),
]


def _create_test_app() -> FastAPI:
    """Create a protected test application using production dependencies."""
    test_app = FastAPI()
    test_app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(test_app)

    @test_app.get("/authenticated")
    async def authenticated_route(
        principal: AuthenticatedPrincipalDependency,
    ) -> dict[str, object]:
        """Return safe principal metadata after authentication."""
        return {
            "subject": principal.subject,
            "email": principal.email,
            "roles": sorted(principal.roles),
            "scopes": sorted(principal.scopes),
        }

    @test_app.post("/approve")
    async def approval_route(
        principal: ReleaseApprovalPrincipalDependency,
    ) -> dict[str, str]:
        """Represent an operation requiring release approval permission."""
        return {"actor": principal.audit_identity}

    return test_app


def _configure_enabled_authentication(
    monkeypatch: pytest.MonkeyPatch,
    public_key: str,
) -> None:
    """Configure deterministic JWT verification for one test."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "AUTH_JWT_ISSUER",
        "https://identity.example.com/",
    )
    monkeypatch.setenv("AUTH_JWT_AUDIENCE", "agentflow-api")
    monkeypatch.setenv("AUTH_JWT_PUBLIC_KEY", public_key)
    get_settings.cache_clear()


def _build_token(
    private_key: str,
    *,
    scopes: str,
) -> str:
    """Create a valid externally issued RS256 access token."""
    now = datetime.now(UTC)

    return cast(str, jwt.encode(
        {
            "sub": "user-123",
            "email": "manager@example.com",
            "roles": ["release_manager"],
            "scope": scopes,
            "iss": "https://identity.example.com/",
            "aud": "agentflow-api",
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        private_key,
        algorithm="RS256",
    ))


def test_disabled_local_authentication_returns_development_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local development should remain usable without bearer tokens."""
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()

    response = TestClient(_create_test_app()).get("/authenticated")

    assert response.status_code == 200
    assert response.json() == {
        "subject": "local-development",
        "email": None,
        "roles": ["local_admin"],
        "scopes": ["*"],
    }


def test_enabled_authentication_rejects_missing_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
    rsa_key_pair: tuple[str, str],
) -> None:
    """Protected routes must return 401 when credentials are absent."""
    _, public_key = rsa_key_pair
    _configure_enabled_authentication(monkeypatch, public_key)

    response = TestClient(_create_test_app()).get("/authenticated")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"


def test_enabled_authentication_returns_verified_principal(
    monkeypatch: pytest.MonkeyPatch,
    rsa_key_pair: tuple[str, str],
) -> None:
    """A valid bearer token should populate the trusted principal."""
    private_key, public_key = rsa_key_pair
    _configure_enabled_authentication(monkeypatch, public_key)
    token = _build_token(
        private_key,
        scopes="release:read release:approve",
    )

    response = TestClient(_create_test_app()).get(
        "/authenticated",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "subject": "user-123",
        "email": "manager@example.com",
        "roles": ["release_manager"],
        "scopes": ["release:approve", "release:read"],
    }


def test_scope_dependency_rejects_insufficient_permissions(
    monkeypatch: pytest.MonkeyPatch,
    rsa_key_pair: tuple[str, str],
) -> None:
    """Authenticated callers without the required scope must receive 403."""
    private_key, public_key = rsa_key_pair
    _configure_enabled_authentication(monkeypatch, public_key)
    token = _build_token(private_key, scopes="release:read")

    response = TestClient(_create_test_app()).post(
        "/approve",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "AUTHORIZATION_FAILED"


def test_scope_dependency_allows_required_permission(
    monkeypatch: pytest.MonkeyPatch,
    rsa_key_pair: tuple[str, str],
) -> None:
    """A caller with the required scope should reach the protected route."""
    private_key, public_key = rsa_key_pair
    _configure_enabled_authentication(monkeypatch, public_key)
    token = _build_token(private_key, scopes="release:approve")

    response = TestClient(_create_test_app()).post(
        "/approve",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {"actor": "manager@example.com"}


def test_local_wildcard_scope_allows_protected_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit local-development principal may use protected routes."""
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()

    response = TestClient(_create_test_app()).post("/approve")

    assert response.status_code == 200
    assert response.json() == {"actor": "local-development"}
