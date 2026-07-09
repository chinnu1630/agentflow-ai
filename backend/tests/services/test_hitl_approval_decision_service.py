"""Tests for deterministic HITL approval decisions."""

from __future__ import annotations

from app.services.hitl_approval_decision_service import HITLApprovalDecisionService


def test_approval_required_when_release_is_blocked() -> None:
    """Blocked release recommendation should require manager approval."""
    service = HITLApprovalDecisionService()

    decision = service.determine_approval(
        {
            "risk_level": "critical",
            "recommended_action": "block_release",
        },
        run_id="test-run-1",
    )

    assert decision.approval_policy_version == "hitl_policy_v1"
    assert decision.approval_required is True
    assert decision.approval_reason == (
        "Release is blocked by deterministic risk scoring."
    )


def test_approval_required_when_data_is_partial() -> None:
    """Partial data review should require approval before proceeding."""
    service = HITLApprovalDecisionService()

    decision = service.determine_approval(
        {
            "risk_level": "low",
            "recommended_action": "partial_data_review",
        }
    )

    assert decision.approval_required is True
    assert decision.approval_reason == (
        "Release analysis used degraded or partial data."
    )


def test_approval_required_for_high_risk_even_when_action_is_review_required() -> None:
    """High risk should require manager approval."""
    service = HITLApprovalDecisionService()

    decision = service.determine_approval(
        {
            "risk_level": "high",
            "recommended_action": "review_required",
        }
    )

    assert decision.approval_required is True
    assert decision.approval_reason == (
        "High release risk requires manager approval."
    )


def test_approval_not_required_for_low_proceed() -> None:
    """Low risk proceed decision should not require approval in MVP policy."""
    service = HITLApprovalDecisionService()

    decision = service.determine_approval(
        {
            "risk_level": "low",
            "recommended_action": "proceed",
        }
    )

    assert decision.approval_required is False
    assert decision.approval_reason is None


def test_approval_required_when_score_is_missing() -> None:
    """Missing score should fail safe and require approval."""
    service = HITLApprovalDecisionService()

    decision = service.determine_approval(None)

    assert decision.approval_required is True
    assert decision.approval_reason == (
        "Release risk score is unavailable or incomplete."
    )
