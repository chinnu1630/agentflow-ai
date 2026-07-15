from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.schemas.error import ErrorDetail, ErrorResponse

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all application exception handlers."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle expected application errors with a safe JSON response."""
    run_id = _get_run_id(request)

    logger.warning(
        "application_error",
        extra={
            "run_id": run_id,
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "path": request.url.path,
        },
    )

    response = ErrorResponse(
        error=ErrorDetail(
            code=exc.error_code,
            message=exc.message,
        ),
        run_id=run_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle standard HTTP errors with AgentFlow's error format."""
    run_id = _get_run_id(request)

    response = ErrorResponse(
        error=ErrorDetail(
            code="HTTP_ERROR",
            message=str(exc.detail),
        ),
        run_id=run_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle request validation errors safely."""
    run_id = _get_run_id(request)

    logger.warning(
        "validation_error",
        extra={
            "run_id": run_id,
            "path": request.url.path,
            "errors": exc.errors(),
        },
    )

    response = ErrorResponse(
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
        ),
        run_id=run_id,
    )

    return JSONResponse(
        status_code=422,
        content=response.model_dump(),
    )


def _get_run_id(request: Request) -> str:
    """Return request run_id if available."""
    return str(getattr(request.state, "run_id", "unknown"))