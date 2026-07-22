"""Tests for AgentFlow OpenTelemetry tracing setup."""

from __future__ import annotations

from fastapi import FastAPI
from opentelemetry.trace import Status, StatusCode

from app.observability.tracing import (
    configure_tracing,
    record_business_span_failure,
    set_safe_span_attributes,
)


class FakeSpan:
    """Small fake span used to verify safe attribute filtering."""

    def __init__(self) -> None:
        """Create an empty fake span."""
        self.attributes: dict[str, str | bool | int | float] = {}
        self.status: Status | None = None

    def set_attribute(self, key: str, value: str | bool | int | float) -> None:
        """Store a span attribute like an OpenTelemetry span would."""
        self.attributes[key] = value

    def set_status(self, status: Status) -> None:
        """Store the safe span status assigned by tracing helpers."""
        self.status = status


def test_configure_tracing_disabled_does_not_fail() -> None:
    """Tracing setup should be safe when disabled for local tests."""
    app = FastAPI()

    configure_tracing(
        app,
        enabled=False,
        service_name="agentflow-ai-backend",
        environment="test",
        app_version="0.1.0",
        otlp_endpoint=None,
        sample_ratio=1.0,
    )


def test_set_safe_span_attributes_keeps_only_safe_scalars() -> None:
    """Only safe scalar metadata should be attached to spans."""
    span = FakeSpan()

    set_safe_span_attributes(
        span,  # type: ignore[arg-type]
        {
            "release_run_id": "123",
            "approval_required": True,
            "risk_count": 4,
            "score": 0.87,
            "none_value": None,
            "unsafe_payload": {"jira_description": "do not export"},
            "unsafe_list": ["chunk text"],
        },
    )

    assert span.attributes == {
        "release_run_id": "123",
        "approval_required": True,
        "risk_count": 4,
        "score": 0.87,
    }



def test_record_business_span_failure_uses_safe_metadata_only() -> None:
    """Failure tracing should expose type and stage, never exception messages."""
    span = FakeSpan()
    exception = RuntimeError("internal Jira description must not be exported")

    record_business_span_failure(
        span,  # type: ignore[arg-type]
        failure_stage="dynamic_synthesis",
        exception=exception,
        execution_status="partial",
    )

    assert span.attributes == {
        "failure_stage": "dynamic_synthesis",
        "exception_type": "RuntimeError",
        "execution_status": "partial",
    }
    assert span.status is not None
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "AgentFlow business operation failed."
    assert "internal Jira description" not in str(span.attributes)
    assert "internal Jira description" not in str(span.status)
