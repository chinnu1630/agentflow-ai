"""Tests for AgentFlow JWT authentication and trusted principals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt
from pydantic import SecretStr

from app.core.security import (
    AuthenticatedPrincipal,
    AuthenticationError,
    JWTAuthenticator,
)


@pytest.fixture
def rsa_key_pair() -> tuple[str, str]:
    """Create one ephemeral RSA private/public key pair for JWT tests."""
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


def _build_authenticator(public_key: str) -> JWTAuthenticator:
    """Build a JWT authenticator with deterministic test configuration."""
    return JWTAuthenticator(
        public_key=SecretStr(public_key),
        algorithm="RS256",
        issuer="https://identity.example.com/",
        audience="agentflow-api",
    )


def _build_token(
    private_key: str,
    *,
    issuer: str = "https://identity.example.com/",
    audience: str = "agentflow-api",
    subject: str = "user-123",
    expires_delta: timedelta = timedelta(minutes=5),
) -> str:
    """Create a signed RS256 access token for one test scenario."""
    now = datetime.now(UTC)

    return cast(str, jwt.encode(
        {
            "sub": subject,
            "email": "manager@example.com",
            "roles": ["release_manager", "viewer", "release_manager"],
            "scope": "release:read release:approve release:read",
            "iss": issuer,
            "aud": audience,
            "iat": now,
            "exp": now + expires_delta,
        },
        private_key,
        algorithm="RS256",
    ))


def test_authenticator_returns_normalized_trusted_principal(
    rsa_key_pair: tuple[str, str],
) -> None:
    """A valid JWT should produce an immutable normalized principal."""
    private_key, public_key = rsa_key_pair
    authenticator = _build_authenticator(public_key)

    principal = authenticator.authenticate(
        _build_token(private_key),
        run_id="run-security-001",
    )

    assert principal == AuthenticatedPrincipal(
        subject="user-123",
        email="manager@example.com",
        roles=frozenset({"release_manager", "viewer"}),
        scopes=frozenset({"release:read", "release:approve"}),
    )
    assert principal.audit_identity == "manager@example.com"


@pytest.mark.parametrize(
    ("issuer", "audience"),
    [
        ("https://untrusted.example.com/", "agentflow-api"),
        ("https://identity.example.com/", "different-api"),
    ],
)
def test_authenticator_rejects_untrusted_issuer_or_audience(
    rsa_key_pair: tuple[str, str],
    issuer: str,
    audience: str,
) -> None:
    """JWT issuer and audience must match the trusted configuration."""
    private_key, public_key = rsa_key_pair
    authenticator = _build_authenticator(public_key)

    token = _build_token(
        private_key,
        issuer=issuer,
        audience=audience,
    )

    with pytest.raises(
        AuthenticationError,
        match="Authentication credentials are invalid.",
    ):
        authenticator.authenticate(token, run_id="run-security-002")


def test_authenticator_rejects_expired_token(
    rsa_key_pair: tuple[str, str],
) -> None:
    """Expired access tokens must fail closed."""
    private_key, public_key = rsa_key_pair
    authenticator = _build_authenticator(public_key)

    token = _build_token(
        private_key,
        expires_delta=timedelta(seconds=-1),
    )

    with pytest.raises(AuthenticationError):
        authenticator.authenticate(token, run_id="run-security-003")


def test_authenticator_rejects_missing_subject(
    rsa_key_pair: tuple[str, str],
) -> None:
    """A token without a usable subject cannot establish identity."""
    private_key, public_key = rsa_key_pair
    authenticator = _build_authenticator(public_key)

    token = _build_token(private_key, subject="   ")

    with pytest.raises(AuthenticationError):
        authenticator.authenticate(token, run_id="run-security-004")


def test_authenticator_rejects_token_signed_by_unknown_key(
    rsa_key_pair: tuple[str, str],
) -> None:
    """A token signed by another key must not be trusted."""
    _, public_key = rsa_key_pair
    other_key = rsa.generate_private_key(
        public_exponent=65_537,
        key_size=2_048,
    )
    other_private_key = other_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    authenticator = _build_authenticator(public_key)

    token = _build_token(other_private_key)

    with pytest.raises(AuthenticationError):
        authenticator.authenticate(token, run_id="run-security-005")


def test_authenticator_rejects_blank_token(
    rsa_key_pair: tuple[str, str],
) -> None:
    """Blank bearer tokens must fail before JWT decoding."""
    _, public_key = rsa_key_pair
    authenticator = _build_authenticator(public_key)

    with pytest.raises(AuthenticationError):
        authenticator.authenticate("   ", run_id="run-security-006")
