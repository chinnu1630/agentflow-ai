import time
from uuid import UUID, uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.logging import get_logger


logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request context such as run_id to every incoming API request."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Create a run_id, attach it to request state, and log request lifecycle."""
        run_id = _resolve_run_id(request.headers.get("X-Run-ID"))
        request.state.run_id = run_id
        request.state.request_id = run_id

        start_time = time.perf_counter()

        logger.info(
            "request_started",
            extra={
                "run_id": run_id,
                "method": request.method,
                "path": request.url.path,
            },
        )

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed",
                extra={
                    "run_id": run_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        response.headers["X-Run-ID"] = run_id

        logger.info(
            "request_completed",
            extra={
                "run_id": run_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        return response


def _resolve_run_id(incoming_run_id: str | None) -> str:
    """Return a valid incoming run_id or generate a new UUID run_id."""
    if incoming_run_id is None:
        return str(uuid4())

    try:
        UUID(incoming_run_id)
    except ValueError:
        return str(uuid4())

    return incoming_run_id