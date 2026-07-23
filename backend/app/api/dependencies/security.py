"""FastAPI authentication and authorization dependencies."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.core.security import (
    AuthenticatedPrincipal,
    AuthenticationError,
    JWTAuthenticator,
)

logger = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


class AuthorizationError(AppError):
    """Raised when an authenticated principal lacks required permissions."""

    def __init__(self) -> None:
        """Create a safe authorization failure response."""
        super().__init__(
            message="The authenticated principal is not authorized.",
            error_code="AUTHORIZATION_FAILED",
            status_code=403,
        )


BearerCredentialsDependency = Annotated[
    HTTPAuthorizationCredentials | None,
    Depends(_bearer_scheme),
]


async def get_current_principal(
    request: Request,
    credentials: BearerCredentialsDependency,
) -> AuthenticatedPrincipal:
    """Return the trusted principal for the current API request.

    Local and test environments may use the explicit development principal
    only when authentication is disabled by validated application settings.
    When authentication is enabled, a valid externally issued RS256 bearer
    token is required.

    Args:
        request: Current FastAPI request containing the correlation ID.
        credentials: Optional bearer credentials parsed by FastAPI.

    Returns:
        Verified immutable principal.

    Raises:
        AuthenticationError: If credentials are missing or invalid.
    """
    settings = get_settings()
    run_id = str(
        getattr(request.state, "run_id", "unknown-run-id")
    )

    if not settings.auth_enabled:
        principal = AuthenticatedPrincipal(
            subject="local-development",
            roles=frozenset({"local_admin"}),
            scopes=frozenset({"*"}),
        )

        logger.info(
            "local_authentication_bypass_used",
            extra={
                "run_id": run_id,
                "environment": settings.environment,
            },
        )

        return principal

    if credentials is None or credentials.scheme.lower() != "bearer":
        logger.warning(
            "bearer_credentials_missing",
            extra={"run_id": run_id},
        )
        raise AuthenticationError()

    if (
        settings.auth_jwt_public_key is None
        or settings.auth_jwt_issuer is None
        or settings.auth_jwt_audience is None
    ):
        logger.error(
            "authentication_configuration_unavailable",
            extra={"run_id": run_id},
        )
        raise AuthenticationError()

    authenticator = JWTAuthenticator(
        public_key=settings.auth_jwt_public_key,
        algorithm=settings.auth_jwt_algorithm,
        issuer=settings.auth_jwt_issuer,
        audience=settings.auth_jwt_audience,
    )

    return authenticator.authenticate(
        credentials.credentials,
        run_id=run_id,
    )


PrincipalDependency = Annotated[
    AuthenticatedPrincipal,
    Depends(get_current_principal),
]

AuthorizationDependency = Callable[
    [Request, AuthenticatedPrincipal],
    Coroutine[Any, Any, AuthenticatedPrincipal],
]


def require_scopes(
    *required_scopes: str,
) -> AuthorizationDependency:
    """Create a dependency requiring every supplied authorization scope.

    Authorization is evaluated using set containment. For ``r`` required
    scopes and ``s`` principal scopes, average-case complexity is O(r) time
    after the principal scope set has been created, with O(r) temporary space.

    Args:
        required_scopes: Permissions that must all be present.

    Returns:
        Async FastAPI dependency enforcing the requested scopes.

    Raises:
        ValueError: If no scopes are supplied or a scope is blank.
    """
    normalized_scopes = frozenset(
        scope.strip() for scope in required_scopes if scope.strip()
    )

    if not normalized_scopes:
        raise ValueError(
            "At least one non-blank authorization scope is required."
        )

    async def authorize(
        request: Request,
        principal: PrincipalDependency,
    ) -> AuthenticatedPrincipal:
        """Authorize one verified principal against required scopes."""
        run_id = str(
            getattr(request.state, "run_id", "unknown-run-id")
        )

        if "*" in principal.scopes:
            return principal

        missing_scopes = normalized_scopes.difference(
            principal.scopes
        )

        if missing_scopes:
            logger.warning(
                "scope_authorization_failed",
                extra={
                    "run_id": run_id,
                    "required_scope_count": len(normalized_scopes),
                    "missing_scope_count": len(missing_scopes),
                },
            )
            raise AuthorizationError()

        logger.info(
            "scope_authorization_succeeded",
            extra={
                "run_id": run_id,
                "required_scope_count": len(normalized_scopes),
            },
        )

        return principal

    return authorize
