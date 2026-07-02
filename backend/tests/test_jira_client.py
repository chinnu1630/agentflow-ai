"""Tests for Jira API client normalization."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr

from app.integrations.jira_client import (
    JiraClient,
    JiraClientConfig,
    JiraClientError,
)
from app.schemas.jira import JiraIssuePriority, JiraIssueStatus, JiraIssueType


def _build_jira_config() -> JiraClientConfig:
    """Build deterministic Jira config for tests."""
    return JiraClientConfig(
        base_url="https://jira.example.com",
        email="manager@example.com",
        api_token=SecretStr("fake-token"),
        project_key="PAY",
    )


def test_jira_client_parses_raw_issue_payload() -> None:
    """JiraClient should normalize one raw Jira issue into JiraIssue."""
    client = JiraClient(config=_build_jira_config())

    issue = client._parse_issue(
        raw_issue={
            "key": "PAY-102",
            "fields": {
                "summary": "Payment checkout fails during release validation",
                "description": "Checkout API returns 500.",
                "issuetype": {"name": "Bug"},
                "status": {"name": "In Progress"},
                "priority": {"name": "High"},
                "assignee": {"emailAddress": "engineer@example.com"},
                "reporter": {"displayName": "QA Engineer"},
                "labels": ["payments", "release-risk"],
                "components": [{"name": "payment-service"}],
                "created": "2026-07-01T10:00:00+00:00",
                "updated": "2026-07-01T12:00:00+00:00",
                "duedate": "2026-07-03",
            },
        }
    )

    assert issue.issue_key == "PAY-102"
    assert issue.title == "Payment checkout fails during release validation"
    assert issue.issue_type == JiraIssueType.BUG
    assert issue.status == JiraIssueStatus.IN_PROGRESS
    assert issue.priority == JiraIssuePriority.P1
    assert issue.assignee == "engineer@example.com"
    assert issue.reporter == "QA Engineer"
    assert issue.labels == ["payments", "release-risk"]
    assert issue.components == ["payment-service"]
    assert issue.affected_services == ["payment-service"]
    assert issue.issue_url == "https://jira.example.com/browse/PAY-102"
    assert issue.created_at == datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    assert issue.updated_at == datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    assert issue.due_at == datetime(2026, 7, 3, 0, 0, tzinfo=UTC)
    assert issue.is_blocking_release is True


@pytest.mark.anyio
async def test_jira_client_searches_release_risk_issues() -> None:
    """JiraClient should call Jira search API and return normalized issues."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/api/3/search"

        return httpx.Response(
            status_code=200,
            json={
                "issues": [
                    {
                        "key": "PAY-102",
                        "fields": {
                            "summary": "Payment checkout fails during release validation",
                            "description": "Checkout API returns 500.",
                            "issuetype": {"name": "Bug"},
                            "status": {"name": "Blocked"},
                            "priority": {"name": "Highest"},
                            "assignee": {"emailAddress": "engineer@example.com"},
                            "reporter": {"displayName": "QA Engineer"},
                            "labels": ["release-blocker"],
                            "components": [{"name": "payment-service"}],
                            "created": "2026-07-01T10:00:00+00:00",
                            "updated": "2026-07-01T12:00:00+00:00",
                            "duedate": None,
                        },
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://jira.example.com",
    ) as http_client:
        async with JiraClient(
            config=_build_jira_config(),
            http_client=http_client,
        ) as client:
            issues = await client.search_release_risk_issues(
                run_id="release-run-test",
            )

    assert len(issues) == 1
    assert issues[0].issue_key == "PAY-102"
    assert issues[0].priority == JiraIssuePriority.P0
    assert issues[0].status == JiraIssueStatus.BLOCKED
    assert issues[0].is_blocking_release is True


@pytest.mark.anyio
async def test_jira_client_raises_error_after_retries() -> None:
    """JiraClient should raise JiraClientError after retryable failures."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=503, json={"error": "unavailable"})

    config = JiraClientConfig(
        base_url="https://jira.example.com",
        email="manager@example.com",
        api_token=SecretStr("fake-token"),
        project_key="PAY",
        max_retries=2,
        retry_base_delay_seconds=0.01,
    )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://jira.example.com",
    ) as http_client:
        async with JiraClient(config=config, http_client=http_client) as client:
            with pytest.raises(JiraClientError, match="Failed to query Jira"):
                await client.search_release_risk_issues(
                    run_id="release-run-test",
                )


def test_jira_client_maps_unknown_values_to_safe_defaults() -> None:
    """JiraClient should map unknown Jira values to safe normalized defaults."""
    client = JiraClient(config=_build_jira_config())

    issue = client._parse_issue(
        raw_issue={
            "key": "PAY-105",
            "fields": {
                "summary": "Unknown Jira metadata should not break parsing",
                "issuetype": {"name": "Custom Work Item"},
                "status": {"name": "Waiting"},
                "priority": {"name": "Unknown"},
                "labels": [],
                "components": [],
                "created": "2026-07-01T10:00:00+00:00",
                "updated": "2026-07-01T12:00:00+00:00",
            },
        }
    )

    assert issue.issue_type == JiraIssueType.TASK
    assert issue.status == JiraIssueStatus.TO_DO
    assert issue.priority == JiraIssuePriority.P4