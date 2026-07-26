from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import Lifespan

from app.api.exception_handlers import register_exception_handlers
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.core.redis_client import (
    close_redis_client,
    create_redis_client,
    ping_redis_client,
)
from app.middleware.request_body_limit import RequestBodyLimitMiddleware
from app.middleware.request_context import RequestContextMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.observability.metrics import configure_metrics
from app.observability.tracing import configure_tracing

logger = get_logger(__name__)

def create_application_lifespan(
    settings: Settings,
) -> Lifespan[FastAPI]:
    """Create the FastAPI lifespan handler for shared infrastructure.

    Args:
        settings: Validated application configuration captured at app creation.

    Returns:
        Async lifespan context managing the process-wide Redis client.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize and close the shared Redis client safely."""
        redis_client: Redis[str] | None = None
        app.state.redis_client = None

        if settings.rate_limit_enabled:
            try:
                redis_client = create_redis_client(settings)
                await ping_redis_client(redis_client)
                app.state.redis_client = redis_client

                logger.info("redis_lifecycle_startup_succeeded")

            except RedisError as exc:
                logger.error(
                    "redis_lifecycle_startup_failed",
                    extra={
                        "error_type": type(exc).__name__,
                    },
                )

        try:
            yield
        finally:
            app.state.redis_client = None

            if redis_client is not None:
                try:
                    await close_redis_client(redis_client)
                except RedisError as exc:
                    logger.error(
                        "redis_lifecycle_shutdown_failed",
                        extra={
                            "error_type": type(exc).__name__,
                        },
                    )

    return lifespan


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    setup_logging()

    fastapi_app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Backend API for enterprise release risk automation.",
        lifespan=create_application_lifespan(settings),
    )
    fastapi_app.state.redis_client = None

    configure_tracing(
        fastapi_app,
        enabled=settings.otel_enabled,
        service_name=settings.otel_service_name,
        environment=settings.environment,
        app_version=settings.app_version,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        sample_ratio=settings.otel_sample_ratio,
    )

    configure_metrics(
        enabled=settings.otel_metrics_enabled,
        service_name=settings.otel_service_name,
        environment=settings.environment,
        app_version=settings.app_version,
        otlp_endpoint=settings.otel_metrics_exporter_otlp_endpoint,
        export_interval_milliseconds=(
            settings.otel_metrics_export_interval_milliseconds
        ),
    )

    fastapi_app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(settings.trusted_hosts),
        www_redirect=False,
    )
    fastapi_app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=settings.max_request_body_bytes,
    )
    fastapi_app.add_middleware(SecurityHeadersMiddleware)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Run-ID"],
        expose_headers=["X-Run-ID"],
        allow_credentials=False,
    )
    fastapi_app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(fastapi_app)

    fastapi_app.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )

    return fastapi_app


app = create_app()