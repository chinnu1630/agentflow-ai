"""Tests for release-risk scoring workflow node."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.workflows.release_risk_service_nodes import create_score_release_risk_node
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
)


def _signal(*, category: str, severity: str, score: float) -> dict[str, Any]:
    """Build a minimal workflow risk signal."""
    return {
        "source_type": "github_pull_request",
        "source_id": "PR-1",
        "rule_id": f"rule_{category}",
        "category": category,
        "severity": severity,
        "score": score,
        "title": f"{category} detected",
        "description": f"{category} detected",
        "evidence": {},
    }


def test_score_release_risk_node_adds_features_and_score_to_state() -> None:
    """Scoring node should store feature vector and risk score in workflow state."""
    node = create_score_release_risk_node()
    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-scoring-node-001",
        github={
            "status": "success",
            "risk_results": [
                {
                    "signals": [
                        _signal(
                            category="ci_failure",
                            severity="critical",
                            score=0.95,
                        )
                    ]
                }
            ],
        },
        jira={
            "status": "success",
            "signals": [
                {
                    **_signal(
                        category="open_critical_bug",
                        severity="critical",
                        score=0.9,
                    ),
                    "source_type": "jira_issue",
                }
            ],
        },
        knowledge_status="completed",
        knowledge_results=[{"title": "Payment Runbook", "score": 5.0}],
    )

    result = node(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.stage == ReleaseRiskWorkflowStage.SCORING_RELEASE_RISK
    assert "score_release_risk" in final_state.completed_nodes
    assert final_state.risk_features is not None
    assert final_state.risk_score is not None

    assert final_state.risk_features["feature_version"] == (
        "release_risk_features_v1"
    )
    assert final_state.risk_features["total_risk_count"] == 2
    assert final_state.risk_features["critical_risk_count"] == 2
    assert final_state.risk_features["knowledge_result_count"] == 1

    assert final_state.risk_score["scoring_version"] == (
        "rule_based_release_risk_v1"
    )
    assert final_state.risk_score["risk_level"] == "critical"
    assert final_state.risk_score["recommended_action"] == "block_release"
    assert final_state.risk_score["score"] >= 0.85
