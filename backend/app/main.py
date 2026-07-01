from fastapi import FastAPI

from app.api.exception_handlers import register_exception_handlers
from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.middleware.request_context import RequestContextMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    setup_logging()

    fastapi_app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Backend API for enterprise release risk automation.",
    )

    fastapi_app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(fastapi_app)

    fastapi_app.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )

    return fastapi_app


app = create_app()