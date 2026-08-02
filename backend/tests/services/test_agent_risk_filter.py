"""Unit tests for filtering risks from a trusted persisted snapshot."""

from __future__ import annotations

from datetime import UTC, datetime
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
    evaluated_at = datetime.now(UTC).isoformat()

    payload["github"]["risk_results"] = [
        {
            "source_type": "github_pull_request",
            "source_id": "1",
            "source_url": "https://github.example/pr/1",
            "pull_request_number": 1,
            "total_score": 0.78,
            "max_severity": "high",
            "signals": [
                {
                    "source_type": "github_pull_request",
                    "source_id": "1",
                    "source_url": "https://github.example/pr/1",
                    "rule_id": "github_ci_failure",
                    "category": "ci_failure",
                    "severity": "high",
                    "score": 0.78,
                    "title": "Payment API has failing CI",
                    "description": (
                        "CI failed on a release-critical service."
                    ),
                    "evidence": {
                        "ci_status": "failed",
                    },
                }
            ],
            "evaluated_at": evaluated_at,
        }
    ]

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


def test_filters_github_blockers_from_trusted_signal_category() -> None:
    """GitHub blocker filtering should use persisted rule categories."""

    risks = AgentRiskFilter(request_id="request-123").filter(
        plan=build_plan(
            sources=[RiskSourceFilter.GITHUB],
            blockers_only=True,
        ),
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


def test_blocker_filter_deduplicates_signals_from_same_source() -> None:
    """Blocker answers should contain one ranked item per underlying ticket."""

    release_risk = build_release_risk_response()
    duplicate = release_risk.release_summary.top_risks[-2].model_copy(
        update={
            "severity": "high",
            "score": 0.80,
            "title": "Another blocker rule matched",
            "reason": "A second rule matched the same Jira blocker.",
        }
    )
    release_risk.release_summary.top_risks.append(duplicate)

    risks = AgentRiskFilter(request_id="request-123").filter(
        plan=build_plan(blockers_only=True),
        release_risk=release_risk,
    )

    assert [risk.source_id for risk in risks] == [
        "1",
        "PAY-102",
    ]
    assert sum(
        risk.source_id == "PAY-102"
        for risk in risks
    ) == 1
    assert next(
        risk
        for risk in risks
        if risk.source_id == "PAY-102"
    ).title == "Payment release blocker"


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
