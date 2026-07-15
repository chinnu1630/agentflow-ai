"""Tests for Jira risk collection service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.integrations.jira_client import JiraClientError
from app.schemas.jira import (
    JiraIssue,
    JiraIssuePriority,
    JiraIssueStatus,
    JiraIssueType,
)
from app.services.jira_risk_collector import (
    JiraRiskCollectionStatus,
    JiraRiskCollector,
)


class FakeSuccessfulJiraClient:
    """Fake Jira client that returns configured issues."""

    def __init__(self, issues: list[JiraIssue]) -> None:
        """Initialize fake client with issues."""

        self._issues = issues
        self.was_called = False

    async def search_release_risk_issues(
        self,
        *,
        run_id: str,
        max_results: int = 50,
    ) -> list[JiraIssue]:
        """Return fake open Jira issues."""

        self.was_called = True
        return self._issues


class FakeFailingJiraClient:
    """Fake Jira client that simulates Jira API failure."""

    async def search_release_risk_issues(
        self,
        *,
        run_id: str,
        max_results: int = 50,
    ) -> list[JiraIssue]:
        """Raise a Jira client error."""

        raise JiraClientError("Jira API unavailable")


def _build_jira_issue(
    *,
    issue_key: str = "PAY-102",
    title: str = "Payment authorization failures before release",
    issue_type: JiraIssueType = JiraIssueType.BUG,
    status: JiraIssueStatus = JiraIssueStatus.BLOCKED,
    priority: JiraIssuePriority = JiraIssuePriority.P0,
    assignee: str | None = None,
    is_blocking_release: bool = True,
) -> JiraIssue:
    """Build a valid Jira issue for collector tests."""

    now = datetime.now(UTC)

    return JiraIssue(
        issue_key=issue_key,
        title=title,
        description="Payment authorization intermittently fails during checkout.",
        issue_type=issue_type,
        status=status,
        priority=priority,
        assignee=assignee,
        reporter="qa@example.com",
        labels=["release-blocker", "payments"],
        components=["payments"],
        affected_services=["payment-service"],
        issue_url=f"https://jira.example.com/browse/{issue_key}",
        created_at=now - timedelta(days=2),
        updated_at=now,
        due_at=now + timedelta(days=1),
        is_blocking_release=is_blocking_release,
        linked_pull_request_urls=[],
    )


@pytest.mark.anyio
async def test_collect_returns_jira_risk_signals_for_open_issues() -> None:
    """Collector should fetch Jira issues and flatten rule-engine signals."""

    issue = _build_jira_issue()
    fake_client = FakeSuccessfulJiraClient([issue])
    collector = JiraRiskCollector(jira_client=fake_client)

    result = await collector.collect(run_id="test-run-1")

    assert fake_client.was_called is True
    assert result.status == JiraRiskCollectionStatus.SUCCESS
    assert result.total_issues_analyzed == 1
    assert len(result.issue_results) == 1
    assert result.total_signals > 0
    assert all(signal.source_type == "jira_issue" for signal in result.signals)
    assert all(signal.source_id == "PAY-102" for signal in result.signals)
    assert result.error_message is None
    assert result.duration_ms >= 0


@pytest.mark.anyio
async def test_collect_returns_success_with_no_signals_when_no_issues_exist() -> None:
    """Collector should succeed with empty results when Jira has no open issues."""

    fake_client = FakeSuccessfulJiraClient([])
    collector = JiraRiskCollector(jira_client=fake_client)

    result = await collector.collect(run_id="test-run-empty")

    assert fake_client.was_called is True
    assert result.status == JiraRiskCollectionStatus.SUCCESS
    assert result.total_issues_analyzed == 0
    assert result.total_signals == 0
    assert result.issues == []
    assert result.issue_results == []
    assert result.signals == []
    assert result.error_message is None


@pytest.mark.anyio
async def test_collect_degrades_gracefully_when_jira_client_fails() -> None:
    """Collector should return failed Jira result instead of crashing release analysis."""

    collector = JiraRiskCollector(jira_client=FakeFailingJiraClient())

    result = await collector.collect(run_id="test-run-failure")

    assert result.status == JiraRiskCollectionStatus.FAILED
    assert result.total_issues_analyzed == 0
    assert result.total_signals == 0
    assert result.issues == []
    assert result.issue_results == []
    assert result.signals == []
    assert result.error_message is not None
    assert "Jira risk collection failed" in result.error_message
    assert result.duration_ms >= 0
