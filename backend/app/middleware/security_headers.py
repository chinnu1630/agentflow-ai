"""HTTP response security-header middleware for AgentFlow AI."""

from __future__ import annotations

from fastapi import Request, Response
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach browser hardening headers to every HTTP response."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Return the downstream response with safe security headers."""
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )

        return response
