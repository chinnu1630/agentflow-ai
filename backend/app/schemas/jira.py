"""Schemas for Jira issue data used by AgentFlow AI."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JiraIssueType(StrEnum):
    """Supported Jira issue types for release-risk analysis."""

    BUG = "bug"
    INCIDENT = "incident"
    STORY = "story"
    TASK = "task"
    EPIC = "epic"


class JiraIssueStatus(StrEnum):
    """Normalized Jira workflow statuses for release-risk analysis."""

    TO_DO = "to_do"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    BLOCKED = "blocked"
    DONE = "done"


class JiraIssuePriority(StrEnum):
    """Normalized Jira priority levels used for release-risk scoring."""

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"


class JiraIssue(BaseModel):
    """Normalized Jira issue used by Jira risk rules and collectors.

    This schema intentionally stores only the fields AgentFlow needs for
    release-risk analysis. Raw Jira API payloads should be converted into this
    internal model before risk rules run.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    issue_key: str = Field(
        min_length=3,
        max_length=32,
        pattern=r"^[A-Z][A-Z0-9]+-\d+$",
    )
    title: str = Field(min_length=5, max_length=300)
    description: str | None = Field(default=None, max_length=5_000)
    issue_type: JiraIssueType
    status: JiraIssueStatus
    priority: JiraIssuePriority
    assignee: str | None = Field(default=None, max_length=255)
    reporter: str | None = Field(default=None, max_length=255)
    labels: list[str] = Field(default_factory=list, max_length=30)
    components: list[str] = Field(default_factory=list, max_length=20)
    affected_services: list[str] = Field(default_factory=list, max_length=20)
    issue_url: str = Field(min_length=10, max_length=2_048)
    created_at: datetime
    updated_at: datetime
    due_at: datetime | None = None
    is_blocking_release: bool = False
    linked_pull_request_urls: list[str] = Field(default_factory=list, max_length=20)

    @field_validator(
        "labels",
        "components",
        "affected_services",
        "linked_pull_request_urls",
    )
    @classmethod
    def deduplicate_non_empty_strings(cls, values: list[str]) -> list[str]:
        """Remove duplicate and empty strings while preserving input order."""
        cleaned_values: list[str] = []
        seen_values: set[str] = set()

        for value in values:
            normalized_value = value.strip()

            if not normalized_value or normalized_value in seen_values:
                continue

            cleaned_values.append(normalized_value)
            seen_values.add(normalized_value)

        return cleaned_values