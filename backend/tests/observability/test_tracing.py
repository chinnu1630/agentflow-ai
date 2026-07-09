"""Tests for AgentFlow OpenTelemetry tracing setup."""

from __future__ import annotations

from fastapi import FastAPI

from app.observability.tracing import configure_tracing, set_safe_span_attributes


class FakeSpan:
    """Small fake span used to verify safe attribute filtering."""

    def __init__(self) -> None:
        """Create an empty fake span."""
        self.attributes: dict[str, str | bool | int | float] = {}

    def set_attribute(self, key: str, value: str | bool | int | float) -> None:
        """Store a span attribute like an OpenTelemetry span would."""
        self.attributes[key] = value


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
