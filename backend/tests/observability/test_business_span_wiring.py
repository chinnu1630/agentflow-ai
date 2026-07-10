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


def test_post_workflow_helpers_have_child_business_spans() -> None:
    """Post-workflow scoring, approval, and snapshot helpers should be traced."""
    route_source = Path("app/api/routes/release_runs.py").read_text()

    assert '"risk.scoring_audit"' in route_source
    assert '"approval.ensure_pending"' in route_source
    assert '"snapshot.persist"' in route_source

    assert (
        '"github_risk_count": _count_collection_risks(response.github)'
        in route_source
    )
    assert (
        '"jira_risk_count": _count_collection_risks(response.jira)'
        in route_source
    )
    assert (
        '"total_risk_count": _count_collection_risks(response.github) + _count_collection_risks(response.jira)'
        in route_source
    )
    assert '"approval_required": response.approval_required is True' in route_source
    assert (
        '"overall_severity": _safe_enum_value(response.release_summary.overall_severity)'
        in route_source
    )


def test_risk_count_helper_is_defensive() -> None:
    """Risk-count tracing helper should not crash on unknown response shapes."""
    route_source = Path("app/api/routes/release_runs.py").read_text()

    assert "def _count_collection_risks(collection: object) -> int:" in route_source
    assert 'for attribute_name in ("risks", "risk_signals", "signals"):' in route_source
    assert "return 0" in route_source
