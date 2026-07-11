"""Tests for request context propagation into OpenTelemetry spans."""

from __future__ import annotations

from pathlib import Path


def test_request_context_middleware_attaches_run_id_to_current_span() -> None:
    """Request middleware should attach AgentFlow correlation IDs to spans."""
    source = Path("app/middleware/request_context.py").read_text()

    assert "trace.get_current_span()" in source
    assert "set_safe_span_attributes(" in source
    assert '"agentflow.run_id": run_id' in source
    assert '"agentflow.request_id": run_id' in source
    assert '"http.request.method": request.method' in source
    assert '"http.route.path": request.url.path' in source
