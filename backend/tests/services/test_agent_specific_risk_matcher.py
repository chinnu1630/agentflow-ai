"""Unit tests for matching a follow-up query to one persisted risk."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.schemas.agent_query import (
    AgentEntityReferences,
    AgentIntent,
    AgentQueryPlan,
    ResponseDepth,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.agent_specific_risk_matcher import (
    AgentSpecificRiskMatcher,
    AgentSpecificRiskNotFoundError,
)
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


def build_plan(
    *,
    pull_request_numbers: list[int] | None = None,
    jira_issue_keys: list[str] | None = None,
) -> AgentQueryPlan:
    """Build an executable specific-risk query plan."""

    return AgentQueryPlan(
        intent=AgentIntent.EXPLAIN_SPECIFIC_RISK,
        response_depth=ResponseDepth.DEEP,
        confidence=1.0,
        entity_references=AgentEntityReferences(
            pull_request_numbers=pull_request_numbers or [],
            jira_issue_keys=jira_issue_keys or [],
        ),
        requires_current_snapshot=True,
        routing_reason_code="test_specific_risk_explanation",
    )


def build_release_risk_response() -> ReleaseRunRiskResponse:
    """Build a snapshot containing one GitHub risk and one Jira risk."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    payload["release_summary"]["top_risks"].append(
        {
            "source": "jira",
            "source_type": "jira_issue",
            "source_id": "PAY-102",
            "source_url": "https://jira.example/browse/PAY-102",
            "severity": "critical",
            "score": 0.95,
            "title": "Payment release blocker",
            "reason": "A critical payment production bug remains open.",
            "evidence": {
                "priority": "P1",
                "status": "Blocked",
            },
        }
    )

    return ReleaseRunRiskResponse.model_validate(payload)


def test_matches_pull_request_reference() -> None:
    """An explicit PR number should match the corresponding GitHub risk."""

    matcher = AgentSpecificRiskMatcher(request_id="request-123")

    risk = matcher.match(
        query="Why is PR 1 dangerous?",
        plan=build_plan(pull_request_numbers=[1]),
        release_risk=build_release_risk_response(),
    )

    assert risk.source_type == "github_pull_request"
    assert risk.source_id == "1"


def test_matches_jira_issue_reference() -> None:
    """An explicit Jira key should match the corresponding Jira risk."""

    matcher = AgentSpecificRiskMatcher(request_id="request-123")

    risk = matcher.match(
        query="What evidence supports PAY-102?",
        plan=build_plan(jira_issue_keys=["PAY-102"]),
        release_risk=build_release_risk_response(),
    )

    assert risk.source_type == "jira_issue"
    assert risk.source_id == "PAY-102"


def test_matches_ordinal_risk_reference() -> None:
    """An ordinal reference should use the persisted ranked-risk order."""

    matcher = AgentSpecificRiskMatcher(request_id="request-123")

    risk = matcher.match(
        query="Explain the first risk.",
        plan=build_plan(),
        release_risk=build_release_risk_response(),
    )

    assert risk.source_id == "1"
    assert risk.title == "Payment API has failing CI"


def test_matches_risk_using_title_keywords() -> None:
    """Meaningful query terms should match risk title and reason text."""

    matcher = AgentSpecificRiskMatcher(request_id="request-123")

    risk = matcher.match(
        query="Explain the payment blocker.",
        plan=build_plan(),
        release_risk=build_release_risk_response(),
    )

    assert risk.source_id == "PAY-102"
    assert risk.title == "Payment release blocker"


def test_raises_when_no_persisted_risk_matches() -> None:
    """A query with no matching persisted risk should fail explicitly."""

    matcher = AgentSpecificRiskMatcher(request_id="request-123")

    with pytest.raises(
        AgentSpecificRiskNotFoundError,
        match="No persisted risk matched the query.",
    ):
        matcher.match(
            query="Explain the authentication timeout.",
            plan=build_plan(),
            release_risk=build_release_risk_response(),
        )


def test_matches_prefixed_pull_request_source_id() -> None:
    """PR 42 should match a persisted GitHub source ID formatted as PR-42."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    payload["release_summary"]["top_risks"][0]["source_id"] = "PR-42"

    release_risk = ReleaseRunRiskResponse.model_validate(payload)
    matcher = AgentSpecificRiskMatcher(request_id="request-123")

    risk = matcher.match(
        query="Why is this dangerous?",
        plan=build_plan(pull_request_numbers=[42]),
        release_risk=release_risk,
    )

    assert risk.source_type == "github_pull_request"
    assert risk.source_id == "PR-42"
