from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.middleware.request_body_limit import RequestBodyLimitMiddleware
from app.middleware.request_context import RequestContextMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.observability.tracing import configure_tracing


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    setup_logging()

    fastapi_app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Backend API for enterprise release risk automation.",
    )

    configure_tracing(
        fastapi_app,
        enabled=settings.otel_enabled,
        service_name=settings.otel_service_name,
        environment=settings.environment,
        app_version=settings.app_version,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        sample_ratio=settings.otel_sample_ratio,
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