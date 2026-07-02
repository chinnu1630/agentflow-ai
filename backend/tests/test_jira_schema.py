"""Tests for Jira schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.jira import (
    JiraIssue,
    JiraIssuePriority,
    JiraIssueStatus,
    JiraIssueType,
)


def test_jira_issue_accepts_valid_payload() -> None:
    """JiraIssue should validate a valid normalized Jira issue."""
    issue = JiraIssue(
        issue_key="PAY-102",
        title="Payment checkout fails during release validation",
        description="Checkout API returns 500 for some payment requests.",
        issue_type=JiraIssueType.BUG,
        status=JiraIssueStatus.IN_PROGRESS,
        priority=JiraIssuePriority.P1,
        assignee="engineer@example.com",
        reporter="qa@example.com",
        labels=["payments", "release-risk"],
        components=["checkout-api"],
        affected_services=["payment-service"],
        issue_url="https://jira.example.com/browse/PAY-102",
        created_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        due_at=datetime(2026, 7, 3, 17, 0, tzinfo=UTC),
        is_blocking_release=True,
        linked_pull_request_urls=[
            "https://github.com/acme/acme-backend-services/pull/42",
        ],
    )

    assert issue.issue_key == "PAY-102"
    assert issue.issue_type == JiraIssueType.BUG
    assert issue.status == JiraIssueStatus.IN_PROGRESS
    assert issue.priority == JiraIssuePriority.P1
    assert issue.is_blocking_release is True
    assert issue.affected_services == ["payment-service"]


def test_jira_issue_rejects_invalid_issue_key() -> None:
    """JiraIssue should reject issue keys that do not match Jira format."""
    with pytest.raises(ValidationError):
        JiraIssue(
            issue_key="bad-key",
            title="Payment checkout fails during release validation",
            issue_type=JiraIssueType.BUG,
            status=JiraIssueStatus.IN_PROGRESS,
            priority=JiraIssuePriority.P1,
            issue_url="https://jira.example.com/browse/PAY-102",
            created_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
            updated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        )


def test_jira_issue_deduplicates_list_fields() -> None:
    """JiraIssue should clean duplicate and empty list values."""
    issue = JiraIssue(
        issue_key="PAY-103",
        title="Payment retry queue has delayed processing",
        issue_type=JiraIssueType.BUG,
        status=JiraIssueStatus.BLOCKED,
        priority=JiraIssuePriority.P2,
        labels=["payments", " payments ", "", "release-risk", "payments"],
        components=["queue", "queue", " retry-worker "],
        affected_services=["payment-service", "", "payment-service"],
        linked_pull_request_urls=[
            "https://github.com/acme/acme-backend-services/pull/43",
            "https://github.com/acme/acme-backend-services/pull/43",
        ],
        issue_url="https://jira.example.com/browse/PAY-103",
        created_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )

    assert issue.labels == ["payments", "release-risk"]
    assert issue.components == ["queue", "retry-worker"]
    assert issue.affected_services == ["payment-service"]
    assert issue.linked_pull_request_urls == [
        "https://github.com/acme/acme-backend-services/pull/43",
    ]


def test_jira_issue_rejects_short_title() -> None:
    """JiraIssue should reject unclear short titles."""
    with pytest.raises(ValidationError):
        JiraIssue(
            issue_key="PAY-104",
            title="Bug",
            issue_type=JiraIssueType.BUG,
            status=JiraIssueStatus.IN_PROGRESS,
            priority=JiraIssuePriority.P1,
            issue_url="https://jira.example.com/browse/PAY-104",
            created_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
            updated_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        )