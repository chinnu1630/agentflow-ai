"""Authentication primitives for AgentFlow API security.

AgentFlow trusts identities issued by an external identity provider. This
module verifies signed JWT access tokens and converts validated claims into an
immutable principal that downstream authorization dependencies can trust.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


class AuthenticationError(AppError):
    """Raised when an API request cannot establish a trusted identity."""

    def __init__(self) -> None:
        """Create a safe authentication failure without exposing token details."""
        super().__init__(
            message="Authentication credentials are invalid.",
            error_code="AUTHENTICATION_FAILED",
            status_code=401,
        )


class AuthenticatedPrincipal(BaseModel):
    """Immutable identity derived only from a successfully verified JWT."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    subject: str = Field(min_length=1, max_length=255)
    email: str | None = Field(default=None, min_length=3, max_length=320)
    roles: frozenset[str] = Field(default_factory=frozenset)
    scopes: frozenset[str] = Field(default_factory=frozenset)

    @property
    def audit_identity(self) -> str:
        """Return the safest stable identity for audit records."""
        return self.email or self.subject


class _VerifiedJWTClaims(BaseModel):
    """Validated subset of claims required after signature verification."""

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    sub: str = Field(min_length=1, max_length=255)
    email: str | None = Field(default=None, min_length=3, max_length=320)
    roles: list[str] | str | None = None
    scope: list[str] | str | None = None
    scopes: list[str] | str | None = None


class JWTAuthenticator:
    """Verify externally issued RS256 JWTs and build trusted principals."""

    def __init__(
        self,
        *,
        public_key: SecretStr,
        algorithm: Literal["RS256"],
        issuer: str,
        audience: str,
    ) -> None:
        """Create an authenticator from trusted identity-provider settings.

        Args:
            public_key: PEM public key used only for signature verification.
            algorithm: Allowed asymmetric JWT signing algorithm.
            issuer: Exact trusted issuer claim.
            audience: Exact AgentFlow API audience claim.

        Raises:
            ValueError: If any verification setting is blank.
        """
        public_key_value = public_key.get_secret_value().strip()
        issuer_value = issuer.strip()
        audience_value = audience.strip()

        if not public_key_value:
            raise ValueError("JWT public key must not be blank.")

        if not issuer_value:
            raise ValueError("JWT issuer must not be blank.")

        if not audience_value:
            raise ValueError("JWT audience must not be blank.")

        self._public_key = public_key_value
        self._algorithm = algorithm
        self._issuer = issuer_value
        self._audience = audience_value

    def authenticate(
        self,
        token: str,
        *,
        run_id: str,
    ) -> AuthenticatedPrincipal:
        """Verify one JWT and return its normalized trusted principal.

        Args:
            token: Raw bearer token extracted from the Authorization header.
            run_id: Safe request correlation identifier for structured logs.

        Returns:
            Immutable authenticated principal.

        Raises:
            AuthenticationError: If signature, claims, expiry, or structure
                validation fails.
        """
        normalized_token = token.strip()

        if not normalized_token:
            self._log_failure(run_id=run_id, reason="blank_token")
            raise AuthenticationError()

        try:
            raw_claims: dict[str, Any] = jwt.decode(
                normalized_token,
                self._public_key,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                audience=self._audience,
                options={
                    "require_exp": True,
                    "require_iat": True,
                    "require_iss": True,
                    "require_aud": True,
                    "require_sub": True,
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )

            claims = _VerifiedJWTClaims.model_validate(raw_claims)

            principal = AuthenticatedPrincipal(
                subject=claims.sub,
                email=claims.email,
                roles=self._normalize_claim_values(claims.roles),
                scopes=self._normalize_claim_values(
                    claims.scope,
                    claims.scopes,
                ),
            )

        except ExpiredSignatureError as exc:
            self._log_failure(run_id=run_id, reason="expired_token")
            raise AuthenticationError() from exc

        except (JWTError, ValidationError, TypeError, ValueError) as exc:
            self._log_failure(run_id=run_id, reason="invalid_token")
            raise AuthenticationError() from exc

        logger.info(
            "jwt_authentication_succeeded",
            extra={
                "run_id": run_id,
                "role_count": len(principal.roles),
                "scope_count": len(principal.scopes),
                "email_present": principal.email is not None,
            },
        )

        return principal

    @staticmethod
    def _normalize_claim_values(
        *claim_values: list[str] | str | None,
    ) -> frozenset[str]:
        """Normalize role or scope claims into unique non-blank values."""
        normalized_values: set[str] = set()

        for claim_value in claim_values:
            if claim_value is None:
                continue

            values: Iterable[str]

            if isinstance(claim_value, str):
                values = claim_value.split()
            else:
                values = claim_value

            for value in values:
                normalized_value = value.strip()

                if normalized_value:
                    normalized_values.add(normalized_value)

        return frozenset(normalized_values)

    @staticmethod
    def _log_failure(*, run_id: str, reason: str) -> None:
        """Log a safe authentication failure category without sensitive data."""
        logger.warning(
            "jwt_authentication_failed",
            extra={
                "run_id": run_id,
                "failure_reason": reason,
            },
        )
