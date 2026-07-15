"""Unit tests for resolving one Jira ticket from persisted risk context."""

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
from app.services.agent_jira_ticket_resolver import (
    AgentJiraTicketNotFoundError,
    AgentJiraTicketResolver,
)
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


def build_plan(
    *,
    jira_issue_keys: list[str],
) -> AgentQueryPlan:
    """Build a Jira ticket question plan."""

    return AgentQueryPlan(
        intent=AgentIntent.JIRA_TICKET_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=1.0,
        entity_references=AgentEntityReferences(
            jira_issue_keys=jira_issue_keys,
        ),
        requires_current_snapshot=True,
        routing_reason_code="test_jira_ticket_question",
    )


def build_release_risk_response() -> ReleaseRunRiskResponse:
    """Build a persisted snapshot containing PAY-102 details."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )

    payload["jira"]["issues"] = [
        {
            "issue_key": "PAY-102",
            "title": "Payment release blocker",
            "issue_url": "https://jira.example/browse/PAY-102",
            "signals": [
                {
                    "source_type": "jira_issue",
                    "source_id": "PAY-102",
                    "source_url": "https://jira.example/browse/PAY-102",
                    "rule_id": "jira_release_blocker",
                    "category": "release_blocker_issue",
                    "severity": "critical",
                    "score": 0.95,
                    "title": "Jira issue is marked as a release blocker",
                    "description": "The issue explicitly blocks the release.",
                    "evidence": {
                        "status": "blocked",
                        "priority": "P1",
                        "is_blocking_release": True,
                    },
                }
            ],
        }
    ]
    payload["jira"]["total_issues_analyzed"] = 1
    payload["jira"]["total_signals"] = 1
    payload["jira"]["signals"] = payload["jira"]["issues"][0]["signals"]

    return ReleaseRunRiskResponse.model_validate(payload)


def test_resolves_jira_ticket_using_extracted_key() -> None:
    """An extracted Jira key should resolve the persisted issue."""

    resolver = AgentJiraTicketResolver(request_id="request-123")

    issue = resolver.resolve(
        plan=build_plan(jira_issue_keys=["PAY-102"]),
        release_risk=build_release_risk_response(),
    )

    assert issue.issue_key == "PAY-102"
    assert issue.title == "Payment release blocker"
    assert len(issue.signals) == 1


def test_matches_jira_key_case_insensitively() -> None:
    """Persisted Jira keys should be matched case-insensitively."""

    resolver = AgentJiraTicketResolver(request_id="request-123")

    issue = resolver.resolve(
        plan=build_plan(jira_issue_keys=["pay-102"]),
        release_risk=build_release_risk_response(),
    )

    assert issue.issue_key == "PAY-102"


def test_raises_when_jira_key_is_missing() -> None:
    """A Jira ticket question must identify one issue key."""

    resolver = AgentJiraTicketResolver(request_id="request-123")

    with pytest.raises(
        AgentJiraTicketNotFoundError,
        match="No Jira issue key was provided.",
    ):
        resolver.resolve(
            plan=build_plan(jira_issue_keys=[]),
            release_risk=build_release_risk_response(),
        )


def test_raises_when_persisted_ticket_does_not_exist() -> None:
    """An unknown Jira key should fail without rerunning Jira."""

    resolver = AgentJiraTicketResolver(request_id="request-123")

    with pytest.raises(
        AgentJiraTicketNotFoundError,
        match="No persisted Jira issue matched the query.",
    ):
        resolver.resolve(
            plan=build_plan(jira_issue_keys=["PAY-999"]),
            release_risk=build_release_risk_response(),
        )
