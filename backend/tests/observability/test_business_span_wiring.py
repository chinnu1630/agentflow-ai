"""Tests that key AgentFlow business workflow boundaries are traced."""

from __future__ import annotations

from pathlib import Path


def test_preferred_release_risks_endpoint_has_business_span() -> None:
    """The preferred /risks endpoint should create a domain-level span."""
    route_source = Path("app/api/routes/release_runs.py").read_text()

    assert '"release_run.risks_endpoint"' in route_source
    assert '"release_run_id": str(release_run_id)' in route_source
    assert '"run_id": request_id' in route_source
    assert (
        '"route": "/api/v1/release-runs/{release_run_id}/risks"'
        in route_source
    )


def test_release_risk_workflow_has_safe_business_span() -> None:
    """The LangGraph workflow entry point should create a safe domain span."""
    service_source = Path("app/services/release_run_service.py").read_text()

    assert '"release_risk.workflow"' in service_source
    assert '"release_run_id": str(release_run_id)' in service_source
    assert '"run_id": self._request_id' in service_source
    assert '"manager_query_present": manager_query != ""' in service_source
    assert '"requested_by_present": requested_by is not None' in service_source
    assert '"manager_query": manager_query' in service_source
