"""Safe OpenTelemetry metrics for AgentFlow AI."""

from __future__ import annotations

import re
from enum import StrEnum

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.metrics import Counter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource

from app.core.logging import get_logger

logger = get_logger(__name__)

_ALLOWED_POLICIES = frozenset({"standard", "expensive"})
_SAFE_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_METRICS_CONFIGURED = False

_meter = metrics.get_meter("agentflow.rate_limit")

_RATE_LIMIT_REQUEST_COUNTER: Counter = _meter.create_counter(
    name="agentflow.rate_limit.requests",
    unit="{request}",
    description=(
        "Number of rate-limit evaluations grouped by policy and outcome."
    ),
)

_RATE_LIMIT_REDIS_ERROR_COUNTER: Counter = _meter.create_counter(
    name="agentflow.rate_limit.redis_errors",
    unit="{error}",
    description=(
        "Number of Redis or protocol failures during rate-limit evaluation."
    ),
)


def configure_metrics(
    *,
    enabled: bool,
    service_name: str,
    environment: str,
    app_version: str,
    otlp_endpoint: str | None,
    export_interval_milliseconds: int,
) -> None:
    """Configure periodic OpenTelemetry metrics export.

    Args:
        enabled: Whether metrics export is enabled.
        service_name: Logical service name attached to metric resources.
        environment: Runtime environment such as local or production.
        app_version: Application version attached to metric resources.
        otlp_endpoint: Explicit OTLP HTTP metrics endpoint.
        export_interval_milliseconds: Interval between metric exports.

    Raises:
        ValueError: If metrics are enabled without an exporter endpoint.
    """
    global _METRICS_CONFIGURED

    if not enabled:
        logger.info(
            "otel_metrics_disabled",
            extra={
                "service_name": service_name,
                "environment": environment,
            },
        )
        return

    if _METRICS_CONFIGURED:
        logger.info(
            "otel_metrics_already_configured",
            extra={
                "service_name": service_name,
                "environment": environment,
            },
        )
        return

    if otlp_endpoint is None or not otlp_endpoint.strip():
        raise ValueError(
            "OpenTelemetry metrics require an OTLP endpoint."
        )

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": app_version,
            "deployment.environment": environment,
        }
    )
    exporter = OTLPMetricExporter(
        endpoint=otlp_endpoint,
    )
    metric_reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=export_interval_milliseconds,
    )
    provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader],
    )

    metrics.set_meter_provider(provider)
    _METRICS_CONFIGURED = True

    logger.info(
        "otel_metrics_configured",
        extra={
            "service_name": service_name,
            "environment": environment,
            "export_interval_milliseconds": (
                export_interval_milliseconds
            ),
        },
    )


class RateLimitMetricOutcome(StrEnum):
    """Bounded outcomes for rate-limit request metrics."""

    ALLOWED = "allowed"
    DENIED = "denied"
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"
    DISABLED = "disabled"


def record_rate_limit_outcome(
    *,
    policy: str,
    outcome: RateLimitMetricOutcome,
) -> None:
    """Increment the rate-limit request counter using safe dimensions.

    Args:
        policy: Validated endpoint policy class.
        outcome: Bounded request outcome.

    Raises:
        ValueError: If the policy is not a supported bounded value.
    """
    normalized_policy = _validate_policy(policy)

    _RATE_LIMIT_REQUEST_COUNTER.add(
        1,
        {
            "rate_limit.policy": normalized_policy,
            "rate_limit.outcome": outcome.value,
        },
    )


def record_rate_limit_redis_error(
    *,
    policy: str,
    error_type: str,
) -> None:
    """Increment the Redis-error counter without sensitive values.

    Args:
        policy: Validated endpoint policy class.
        error_type: Safe exception class name such as ``ConnectionError``.

    Raises:
        ValueError: If either metric dimension is unsafe or unbounded.
    """
    normalized_policy = _validate_policy(policy)
    normalized_error_type = error_type.strip()

    if not _SAFE_ERROR_TYPE_PATTERN.fullmatch(normalized_error_type):
        raise ValueError(
            "Rate-limit metric error type must be a safe class name."
        )

    _RATE_LIMIT_REDIS_ERROR_COUNTER.add(
        1,
        {
            "rate_limit.policy": normalized_policy,
            "rate_limit.error_type": normalized_error_type,
        },
    )


def _validate_policy(policy: str) -> str:
    """Return one supported low-cardinality policy value."""
    normalized_policy = policy.strip()

    if normalized_policy not in _ALLOWED_POLICIES:
        raise ValueError(
            "Rate-limit metric policy must be standard or expensive."
        )

    return normalized_policy
