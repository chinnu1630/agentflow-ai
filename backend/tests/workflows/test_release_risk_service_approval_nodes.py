"""Tests for HITL approval requirement workflow node."""

from __future__ import annotations

from uuid import uuid4

from app.workflows.release_risk_service_nodes import (
    create_determine_approval_requirement_node,
)
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
)


def test_determine_approval_requirement_node_requires_approval_for_critical() -> None:
    """Critical/block-release score should require approval."""
    node = create_determine_approval_requirement_node()
    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-approval-node-001",
        risk_score={
            "scoring_version": "rule_based_release_risk_v1",
            "feature_version": "release_risk_features_v1",
            "score": 0.9,
            "risk_level": "critical",
            "recommended_action": "block_release",
            "reasons": ["Critical risk detected."],
            "component_scores": {},
        },
    )

    result = node(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.stage == ReleaseRiskWorkflowStage.DETERMINING_APPROVAL_REQUIREMENT
    assert "determine_approval_requirement" in final_state.completed_nodes
    assert final_state.approval_required is True
    assert final_state.approval_reason == (
        "Release is blocked by deterministic risk scoring."
    )
    assert final_state.approval_policy_version == "hitl_policy_v1"


def test_determine_approval_requirement_node_does_not_require_low_proceed() -> None:
    """Low/proceed score should not require approval in MVP policy."""
    node = create_determine_approval_requirement_node()
    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-approval-node-002",
        risk_score={
            "scoring_version": "rule_based_release_risk_v1",
            "feature_version": "release_risk_features_v1",
            "score": 0.1,
            "risk_level": "low",
            "recommended_action": "proceed",
            "reasons": ["No material risk detected."],
            "component_scores": {},
        },
    )

    result = node(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.approval_required is False
    assert final_state.approval_reason is None
    assert final_state.approval_policy_version == "hitl_policy_v1"
