"""Unit tests for filtering risks from a trusted persisted snapshot."""

from __future__ import annotations

from uuid import uuid4

from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryFilters,
    AgentQueryPlan,
    ResponseDepth,
    RiskSourceFilter,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.agent_risk_filter import AgentRiskFilter
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


def build_plan(
    *,
    sources: list[RiskSourceFilter] | None = None,
    severities: list[str] | None = None,
    blockers_only: bool = False,
    open_items_only: bool = False,
) -> AgentQueryPlan:
    """Build a valid risk-filter query plan."""

    return AgentQueryPlan(
        intent=AgentIntent.FILTER_RISKS,
        response_depth=ResponseDepth.STANDARD,
        confidence=1.0,
        filters=AgentQueryFilters(
            sources=sources or [],
            severities=severities or [],
            blockers_only=blockers_only,
            open_items_only=open_items_only,
        ),
        requires_current_snapshot=True,
        routing_reason_code="test_risk_filter",
    )


def build_release_risk_response() -> ReleaseRunRiskResponse:
    """Build a snapshot containing GitHub and Jira risks."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )

    payload["release_summary"]["top_risks"].extend(
        [
            {
                "source": "jira",
                "source_type": "jira_issue",
                "source_id": "PAY-102",
                "source_url": "https://jira.example/browse/PAY-102",
                "severity": "critical",
                "score": 0.95,
                "title": "Payment release blocker",
                "reason": "The payment issue blocks the current release.",
                "evidence": {
                    "status": "blocked",
                    "is_blocking_release": True,
                    "priority": "P1",
                },
            },
            {
                "source": "jira",
                "source_type": "jira_issue",
                "source_id": "PAY-103",
                "source_url": "https://jira.example/browse/PAY-103",
                "severity": "high",
                "score": 0.80,
                "title": "Completed payment incident follow-up",
                "reason": "The issue was previously high risk but is now completed.",
                "evidence": {
                    "status": "done",
                    "priority": "P1",
                },
            },
        ]
    )

    return ReleaseRunRiskResponse.model_validate(payload)


def test_filters_github_risks_only() -> None:
    """A GitHub source filter should exclude Jira risks."""

    risk_filter = AgentRiskFilter(request_id="request-123")

    risks = risk_filter.filter(
        plan=build_plan(sources=[RiskSourceFilter.GITHUB]),
        release_risk=build_release_risk_response(),
    )

    assert len(risks) == 1
    assert risks[0].source == "github"
    assert risks[0].source_id == "1"


def test_filters_jira_blockers_only() -> None:
    """A Jira blocker filter should use trusted persisted evidence."""

    risk_filter = AgentRiskFilter(request_id="request-123")

    risks = risk_filter.filter(
        plan=build_plan(
            sources=[RiskSourceFilter.JIRA],
            blockers_only=True,
        ),
        release_risk=build_release_risk_response(),
    )

    assert len(risks) == 1
    assert risks[0].source_id == "PAY-102"


def test_filters_by_severity() -> None:
    """Severity filters should match normalized persisted severity values."""

    risk_filter = AgentRiskFilter(request_id="request-123")

    risks = risk_filter.filter(
        plan=build_plan(severities=["critical"]),
        release_risk=build_release_risk_response(),
    )

    assert len(risks) == 1
    assert risks[0].source_id == "PAY-102"
    assert risks[0].severity.value == "critical"


def test_filters_open_items_only() -> None:
    """Open-only filtering should exclude completed Jira issues."""

    risk_filter = AgentRiskFilter(request_id="request-123")

    risks = risk_filter.filter(
        plan=build_plan(open_items_only=True),
        release_risk=build_release_risk_response(),
    )

    source_ids = {risk.source_id for risk in risks}

    assert "1" in source_ids
    assert "PAY-102" in source_ids
    assert "PAY-103" not in source_ids


def test_combines_all_requested_filters() -> None:
    """All configured filters should be applied using AND semantics."""

    risk_filter = AgentRiskFilter(request_id="request-123")

    risks = risk_filter.filter(
        plan=build_plan(
            sources=[RiskSourceFilter.JIRA],
            severities=["critical"],
            blockers_only=True,
            open_items_only=True,
        ),
        release_risk=build_release_risk_response(),
    )

    assert [risk.source_id for risk in risks] == ["PAY-102"]


def test_returns_empty_list_when_no_risks_match() -> None:
    """No matching risks should return an empty result instead of failing."""

    risk_filter = AgentRiskFilter(request_id="request-123")

    risks = risk_filter.filter(
        plan=build_plan(
            sources=[RiskSourceFilter.GITHUB],
            severities=["critical"],
        ),
        release_risk=build_release_risk_response(),
    )

    assert risks == []
