"""Bound incoming HTTP request-body sizes before business execution."""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.schemas.error import ErrorDetail, ErrorResponse


class RequestBodyTooLargeError(Exception):
    """Raised when a streamed request body exceeds its configured limit."""


class RequestBodyLimitMiddleware:
    """Reject request bodies larger than a validated byte limit."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
    ) -> None:
        """Initialize the middleware with a positive request-body limit."""
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")

        self._app = app
        self._max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Enforce declared and streamed HTTP request-body sizes."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")

        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > self._max_body_bytes
        ):
            await self._send_too_large_response(scope, receive, send)
            return

        received_bytes = 0

        async def limited_receive() -> Message:
            """Count streamed body bytes before forwarding each message."""
            nonlocal received_bytes

            message = await receive()

            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))

                if received_bytes > self._max_body_bytes:
                    raise RequestBodyTooLargeError

            return message

        try:
            await self._app(scope, limited_receive, send)
        except RequestBodyTooLargeError:
            await self._send_too_large_response(scope, receive, send)

    async def _send_too_large_response(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Return AgentFlow's safe JSON error contract for oversized bodies."""
        state = scope.get("state", {})
        run_id = str(state.get("run_id", "unknown"))

        payload = ErrorResponse(
            error=ErrorDetail(
                code="REQUEST_TOO_LARGE",
                message="Request body exceeds the configured size limit.",
            ),
            run_id=run_id,
        )

        response = JSONResponse(
            status_code=413,
            content=payload.model_dump(),
        )
        await response(scope, receive, send)
