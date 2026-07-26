"""Tests for safe OpenTelemetry rate-limit metrics."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from app.observability.metrics import (
    RateLimitMetricOutcome,
    record_rate_limit_outcome,
    record_rate_limit_redis_error,
)


@pytest.mark.parametrize(
    "outcome",
    list(RateLimitMetricOutcome),
)
def test_record_rate_limit_outcome_uses_bounded_safe_attributes(
    outcome: RateLimitMetricOutcome,
) -> None:
    """Every request outcome should use only bounded metric dimensions."""
    counter = Mock()

    with patch(
        "app.observability.metrics._RATE_LIMIT_REQUEST_COUNTER",
        counter,
    ):
        record_rate_limit_outcome(
            policy="expensive",
            outcome=outcome,
        )

    counter.add.assert_called_once_with(
        1,
        {
            "rate_limit.policy": "expensive",
            "rate_limit.outcome": outcome.value,
        },
    )

    serialized_call = repr(counter.add.call_args)

    assert "subject" not in serialized_call
    assert "email" not in serialized_call
    assert "redis://" not in serialized_call
    assert "agentflow:rate-limit:" not in serialized_call


def test_record_rate_limit_redis_error_uses_safe_error_class() -> None:
    """Redis failures should expose only policy and exception class."""
    counter = Mock()

    with patch(
        "app.observability.metrics._RATE_LIMIT_REDIS_ERROR_COUNTER",
        counter,
    ):
        record_rate_limit_redis_error(
            policy="standard",
            error_type="ConnectionError",
        )

    counter.add.assert_called_once_with(
        1,
        {
            "rate_limit.policy": "standard",
            "rate_limit.error_type": "ConnectionError",
        },
    )


@pytest.mark.parametrize(
    ("policy", "error_type"),
    [
        ("", "ConnectionError"),
        ("standard", ""),
        ("unknown", "ConnectionError"),
        ("standard", "redis://secret@example.com"),
    ],
)
def test_record_rate_limit_redis_error_rejects_unsafe_dimensions(
    policy: str,
    error_type: str,
) -> None:
    """Metric dimensions must remain validated and low cardinality."""
    with pytest.raises(ValueError):
        record_rate_limit_redis_error(
            policy=policy,
            error_type=error_type,
        )


def test_configure_metrics_disabled_does_not_initialize_provider() -> None:
    """Disabled metrics should not create or register SDK components."""
    from app.observability.metrics import configure_metrics

    with (
        patch(
            "app.observability.metrics.OTLPMetricExporter",
        ) as exporter_class,
        patch(
            "app.observability.metrics.metrics.set_meter_provider",
        ) as set_provider,
    ):
        configure_metrics(
            enabled=False,
            service_name="agentflow-ai-backend",
            environment="test",
            app_version="0.1.0",
            otlp_endpoint=None,
            export_interval_milliseconds=60_000,
        )

    exporter_class.assert_not_called()
    set_provider.assert_not_called()


def test_configure_metrics_registers_periodic_otlp_provider() -> None:
    """Enabled metrics should use a periodic OTLP HTTP exporter."""
    from app.observability import metrics as metrics_module

    exporter = Mock()
    reader = Mock()
    provider = Mock()

    with (
        patch.object(
            metrics_module,
            "_METRICS_CONFIGURED",
            False,
        ),
        patch(
            "app.observability.metrics.OTLPMetricExporter",
            return_value=exporter,
        ) as exporter_class,
        patch(
            "app.observability.metrics.PeriodicExportingMetricReader",
            return_value=reader,
        ) as reader_class,
        patch(
            "app.observability.metrics.MeterProvider",
            return_value=provider,
        ) as provider_class,
        patch(
            "app.observability.metrics.metrics.set_meter_provider",
        ) as set_provider,
    ):
        metrics_module.configure_metrics(
            enabled=True,
            service_name="agentflow-ai-backend",
            environment="production",
            app_version="1.2.3",
            otlp_endpoint="http://otel-collector:4318/v1/metrics",
            export_interval_milliseconds=30_000,
        )

    exporter_class.assert_called_once_with(
        endpoint="http://otel-collector:4318/v1/metrics",
    )
    reader_class.assert_called_once_with(
        exporter,
        export_interval_millis=30_000,
    )

    provider_kwargs = provider_class.call_args.kwargs
    assert provider_kwargs["metric_readers"] == [reader]
    assert provider_kwargs["resource"].attributes[
        "service.name"
    ] == "agentflow-ai-backend"
    assert provider_kwargs["resource"].attributes[
        "service.version"
    ] == "1.2.3"
    assert provider_kwargs["resource"].attributes[
        "deployment.environment"
    ] == "production"

    set_provider.assert_called_once_with(provider)
