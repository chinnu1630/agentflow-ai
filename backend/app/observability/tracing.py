"""OpenTelemetry tracing setup for AgentFlow AI.

This module owns tracing configuration so FastAPI startup code stays small and
business services can create safe domain spans without leaking sensitive data.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, Tracer

from app.core.logging import get_logger

logger = get_logger(__name__)

_SAFE_ATTRIBUTE_TYPES = (str, bool, int, float)
_TRACING_CONFIGURED = False


def configure_tracing(
    app: FastAPI,
    *,
    enabled: bool,
    service_name: str,
    environment: str,
    app_version: str,
    otlp_endpoint: str | None,
    sample_ratio: float,
) -> None:
    """Configure OpenTelemetry tracing for the FastAPI application.

    Args:
        app: FastAPI application instance to instrument.
        enabled: Whether tracing should be enabled.
        service_name: Logical service name shown in trace backends.
        environment: Runtime environment such as local, dev, staging, or prod.
        app_version: Application version attached to trace resources.
        otlp_endpoint: Optional OTLP HTTP endpoint for exporting spans.
        sample_ratio: Trace sampling ratio between 0.0 and 1.0.

    The function is intentionally safe for local development and tests. When
    tracing is disabled, it does nothing. When enabled without an OTLP endpoint,
    spans are created in-process but not exported.
    """
    global _TRACING_CONFIGURED

    if not enabled:
        logger.info(
            "otel_tracing_disabled",
            extra={
                "service_name": service_name,
                "environment": environment,
            },
        )
        return

    if _TRACING_CONFIGURED:
        logger.info(
            "otel_tracing_already_configured",
            extra={
                "service_name": service_name,
                "environment": environment,
            },
        )
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": app_version,
            "deployment.environment": environment,
        }
    )

    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
    )

    if otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=otlp_endpoint),
            )
        )

    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()

    _TRACING_CONFIGURED = True

    logger.info(
        "otel_tracing_configured",
        extra={
            "service_name": service_name,
            "environment": environment,
            "otlp_exporter_enabled": otlp_endpoint is not None,
            "sample_ratio": sample_ratio,
        },
    )


def get_tracer(name: str) -> Tracer:
    """Return an OpenTelemetry tracer for manual business instrumentation."""
    return trace.get_tracer(name)


@contextmanager
def start_business_span(
    span_name: str,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Start a safe manual span for AgentFlow business workflow steps.

    Args:
        span_name: Stable domain span name, for example release_risk.workflow.
        attributes: Safe span attributes. Raw PR text, Jira descriptions,
            knowledge chunk content, Slack payloads, and secrets must never be
            passed here.

    Yields:
        The active OpenTelemetry span.
    """
    tracer = get_tracer("agentflow.business")

    with tracer.start_as_current_span(span_name) as span:
        set_safe_span_attributes(span, attributes or {})
        yield span


def set_safe_span_attributes(
    span: Span,
    attributes: Mapping[str, Any],
) -> None:
    """Attach only safe scalar attributes to a span.

    OpenTelemetry span attributes must be simple scalar values or arrays. For
    this project we intentionally allow only safe scalar metadata so sensitive
    enterprise content does not leak into observability tooling.
    """
    for key, value in attributes.items():
        if value is None:
            continue

        if not isinstance(value, _SAFE_ATTRIBUTE_TYPES):
            logger.warning(
                "otel_span_attribute_dropped",
                extra={
                    "attribute_key": key,
                    "attribute_type": type(value).__name__,
                },
            )
            continue

        span.set_attribute(key, value)
