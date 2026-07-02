"""Tests for Jira issue risk rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.schemas.jira import (
    JiraIssue,
    JiraIssuePriority,
    JiraIssueStatus,
    JiraIssueType,
)
from app.services.jira_risk_rules import (
    JiraRiskCategory,
    JiraRiskRuleEngine,
    JiraRiskSeverity,
)


def _build_issue(
    *,
    issue_key: str = "PAY-102",
    title: str = "Payment checkout fails during release validation",
    issue_type: JiraIssueType = JiraIssueType.BUG,
    status: JiraIssueStatus = JiraIssueStatus.IN_PROGRESS,
    priority: JiraIssuePriority = JiraIssuePriority.P1,
    assignee: str | None = "engineer@example.com",
    affected_services: list[str] | None = None,
    due_at: datetime | None = None,
    is_blocking_release: bool = False,
) -> JiraIssue:
    """Build a deterministic JiraIssue for risk rule tests."""
    return JiraIssue(
        issue_key=issue_key,
        title=title,
        description="Checkout API returns 500 for some payment requests.",
        issue_type=issue_type,
        status=status,
        priority=priority,
        assignee=assignee,
        reporter="qa@example.com",
        labels=["payments"],
        components=["payment-service"],
        affected_services=affected_services or [],
        issue_url=f"https://jira.example.com/browse/{issue_key}",
        created_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        due_at=due_at,
        is_blocking_release=is_blocking_release,
    )


def test_jira_risk_rules_detect_open_high_priority_bug() -> None:
    """Jira rules should detect open P1 bugs as high release risks."""
    engine = JiraRiskRuleEngine()

    result = engine.evaluate_issue(
        _build_issue(),
        run_id="release-run-test",
        evaluated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )

    assert result.issue_key == "PAY-102"
    assert result.total_score > 0
    assert result.max_severity == JiraRiskSeverity.HIGH
    assert any(
        signal.category == JiraRiskCategory.OPEN_CRITICAL_BUG
        for signal in result.signals
    )


def test_jira_risk_rules_detect_release_blocker_issue() -> None:
    """Jira rules should detect issues explicitly marked release blocking."""
    engine = JiraRiskRuleEngine()

    result = engine.evaluate_issue(
        _build_issue(is_blocking_release=True),
        run_id="release-run-test",
        evaluated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )

    assert result.max_severity == JiraRiskSeverity.CRITICAL
    assert any(
        signal.category == JiraRiskCategory.RELEASE_BLOCKER_ISSUE
        for signal in result.signals
    )


def test_jira_risk_rules_detect_blocked_issue() -> None:
    """Jira rules should detect blocked Jira issues."""
    engine = JiraRiskRuleEngine()

    result = engine.evaluate_issue(
        _build_issue(status=JiraIssueStatus.BLOCKED),
        run_id="release-run-test",
        evaluated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )

    assert any(
        signal.category == JiraRiskCategory.BLOCKED_JIRA_ISSUE
        for signal in result.signals
    )


def test_jira_risk_rules_detect_unassigned_high_priority_issue() -> None:
    """Jira rules should detect unassigned high-priority Jira issues."""
    engine = JiraRiskRuleEngine()

    result = engine.evaluate_issue(
        _build_issue(assignee=None),
        run_id="release-run-test",
        evaluated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )

    assert any(
        signal.category == JiraRiskCategory.UNASSIGNED_HIGH_PRIORITY_ISSUE
        for signal in result.signals
    )


def test_jira_risk_rules_detect_due_soon_issue() -> None:
    """Jira rules should detect open issues due soon."""
    engine = JiraRiskRuleEngine()
    evaluated_at = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    result = engine.evaluate_issue(
        _build_issue(due_at=evaluated_at + timedelta(days=1)),
        run_id="release-run-test",
        evaluated_at=evaluated_at,
    )

    assert any(
        signal.category == JiraRiskCategory.DUE_SOON_ISSUE
        for signal in result.signals
    )


def test_jira_risk_rules_detect_critical_service_issue() -> None:
    """Jira rules should detect issues affecting configured critical services."""
    engine = JiraRiskRuleEngine()

    result = engine.evaluate_issue(
        _build_issue(affected_services=["payment-service"]),
        run_id="release-run-test",
        evaluated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )

    assert any(
        signal.category == JiraRiskCategory.CRITICAL_SERVICE_ISSUE
        for signal in result.signals
    )


def test_jira_risk_rules_ignore_done_issue() -> None:
    """Jira rules should not flag completed issues."""
    engine = JiraRiskRuleEngine()

    result = engine.evaluate_issue(
        _build_issue(
            status=JiraIssueStatus.DONE,
            is_blocking_release=False,
            affected_services=["payment-service"],
        ),
        run_id="release-run-test",
        evaluated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )

    assert result.signals == []
    assert result.total_score == 0.0
    assert result.max_severity is None


def test_jira_risk_rules_evaluate_many_issues() -> None:
    """Jira rules should evaluate many issues in input order."""
    engine = JiraRiskRuleEngine()

    results = engine.evaluate_issues(
        [
            _build_issue(issue_key="PAY-102"),
            _build_issue(issue_key="PAY-103", status=JiraIssueStatus.DONE),
        ],
        run_id="release-run-test",
        evaluated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )

    assert [result.issue_key for result in results] == ["PAY-102", "PAY-103"]
    assert len(results[0].signals) > 0
    assert results[1].signals == []