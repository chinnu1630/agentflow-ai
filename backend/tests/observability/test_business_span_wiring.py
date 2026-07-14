"""Tests that key AgentFlow business workflow boundaries are traced."""

from __future__ import annotations

from pathlib import Path


def test_preferred_release_risks_endpoint_has_business_span() -> None:
    """The preferred /risks endpoint should create a domain-level span."""
    route_source = Path("app/api/routes/release_runs.py").read_text()

    assert '"release_run.risks_endpoint"' in route_source
    assert '"release_run_id": str(release_run_id)' in route_source
    assert '"run_id": request_id' in route_source
    assert '"route": "/api/v1/release-runs/{release_run_id}/risks"' in route_source


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

    finalizer_source = Path("app/services/release_risk_execution_finalizer.py").read_text()
    normalized_source = " ".join(finalizer_source.split())

    assert '"risk.scoring_audit"' in finalizer_source
    assert '"approval.ensure_pending"' in finalizer_source
    assert '"snapshot.persist"' in finalizer_source

    assert "github_risk_count = _count_collection_risks(response.github)" in normalized_source
    assert "jira_risk_count = _count_collection_risks(response.jira)" in normalized_source
    assert '"total_risk_count": (github_risk_count + jira_risk_count)' in normalized_source
    assert '"approval_required": response.approval_required is True' in normalized_source
    assert '"overall_severity": _safe_enum_value(' in normalized_source
    assert "response.release_summary.overall_severity" in normalized_source


def test_risk_count_helper_is_defensive() -> None:
    """Risk-count tracing helper should not crash on unknown response shapes."""

    finalizer_source = Path("app/services/release_risk_execution_finalizer.py").read_text()

    assert "def _count_collection_risks(collection: object) -> int:" in finalizer_source
    assert 'for attribute_name in ("risks", "risk_signals", "signals"):' in finalizer_source
    assert "return 0" in finalizer_source


def test_langgraph_approval_decision_node_has_business_span() -> None:
    """HITL approval decision node should expose safe tracing metadata."""
    node_source = Path("app/workflows/release_risk_service_nodes.py").read_text()

    assert '"approval.decision"' in node_source
    assert '"release_run_id": str(running_state.release_run_id)' in node_source
    assert '"run_id": running_state.run_id' in node_source
    assert '"release_summary_present": running_state.release_summary is not None' in node_source
    assert '"risk_score_present": running_state.risk_score is not None' in node_source
    assert '"approval.required", decision.approval_required' in node_source
    assert '"approval.reason_present"' in node_source


def test_knowledge_retrieval_node_has_business_span() -> None:
    """Knowledge retrieval node should be traced without exposing document text."""
    node_source = Path("app/workflows/release_risk_service_nodes.py").read_text()

    assert '"knowledge.retrieve"' in node_source
    assert '"release_run_id": str(validated_state.release_run_id)' in node_source
    assert '"run_id": validated_state.run_id' in node_source
    assert '"query_present": bool(validated_state.manager_query)' in node_source
    assert "chunk.content" not in node_source
    assert "document_text" not in node_source


def test_slack_alert_route_has_business_span() -> None:
    """Manual approved Slack alert route should be traced safely."""
    route_source = Path("app/api/routes/release_runs.py").read_text()

    assert '"slack.release_alert.route"' in route_source
    assert '"release_run_id": str(release_run_id)' in route_source
    assert '"run_id": request_id_for_span' in route_source
    assert '"route": "/api/v1/release-runs/{release_run_id}/slack-alert"' in route_source
    assert "SLACK_BOT_TOKEN" in route_source
    assert "slack_bot_token" in route_source
    assert '"slack_bot_token"' not in route_source


def test_slack_duplicate_check_has_business_span() -> None:
    """Slack duplicate-send protection should be traced safely."""
    route_source = Path("app/api/routes/release_runs.py").read_text()

    assert '"slack.release_alert.duplicate_check"' in route_source
    assert '"release_run_id": str(release_run_id)' in route_source
    assert '"run_id": request_id' in route_source
    assert '"slack.duplicate_found", duplicate_found' in route_source
    assert "SLACK_BOT_TOKEN" in route_source
    assert '"slack_bot_token"' not in route_source
