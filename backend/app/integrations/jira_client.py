"""Async Jira API client for release-risk issue collection."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.schemas.jira import (
    JiraIssue,
    JiraIssuePriority,
    JiraIssueStatus,
    JiraIssueType,
)

logger = logging.getLogger(__name__)


class JiraClientError(RuntimeError):
    """Raised when Jira issue collection fails."""


class JiraClientConfig(BaseModel):
    """Configuration required to connect to Jira Cloud or Jira Server."""

    model_config = ConfigDict(str_strip_whitespace=True)

    base_url: str = Field(min_length=10, max_length=2_048)
    email: str = Field(min_length=3, max_length=255)
    api_token: SecretStr
    project_key: str = Field(min_length=2, max_length=32)
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=60.0)
    max_retries: int = Field(default=3, ge=1, le=5)
    retry_base_delay_seconds: float = Field(default=0.25, gt=0.0, le=5.0)


class JiraClient:
    """Async client for searching Jira issues relevant to release risk."""

    def __init__(
        self,
        *,
        config: JiraClientConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Jira client.

        Args:
            config: Jira API connection configuration.
            http_client: Optional injected HTTP client for tests.
        """
        self._config = config
        self._http_client = http_client
        self._owns_http_client = http_client is None

    async def __aenter__(self) -> JiraClient:
        """Enter async context manager."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self._config.base_url.rstrip("/"),
                timeout=self._config.timeout_seconds,
                auth=(
                    self._config.email,
                    self._config.api_token.get_secret_value(),
                ),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close owned HTTP client when leaving async context manager."""
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()

    async def search_release_risk_issues(
        self,
        *,
        run_id: str,
        max_results: int = 50,
    ) -> list[JiraIssue]:
        """Search Jira for open issues relevant to release-risk analysis.

        Args:
            run_id: Release run correlation ID for structured logs.
            max_results: Maximum number of Jira issues to request.

        Returns:
            Normalized Jira issues.

        Raises:
            JiraClientError: If Jira cannot be queried successfully.
        """
        if self._http_client is None:
            raise JiraClientError("JiraClient must be used as an async context manager.")

        jql = (
            f'project = "{self._config.project_key}" '
            "AND statusCategory != Done "
            "AND priority in (Highest, High, Medium) "
            "ORDER BY priority DESC, updated DESC"
        )

        payload = {
            "jql": jql,
            "maxResults": max_results,
            "fields": [
                "summary",
                "description",
                "issuetype",
                "status",
                "priority",
                "assignee",
                "reporter",
                "labels",
                "components",
                "created",
                "updated",
                "duedate",
            ],
        }

        logger.info(
            "jira_issue_search_started",
            extra={
                "run_id": run_id,
                "project_key": self._config.project_key,
                "max_results": max_results,
            },
        )

        response_payload = await self._post_with_retries(
            run_id=run_id,
            url="/rest/api/3/search",
            json_payload=payload,
        )

        raw_issues = response_payload.get("issues", [])

        if not isinstance(raw_issues, list):
            raise JiraClientError("Jira search response did not contain an issues list.")

        issues = [
            self._parse_issue(raw_issue=raw_issue)
            for raw_issue in raw_issues
            if isinstance(raw_issue, dict)
        ]

        logger.info(
            "jira_issue_search_completed",
            extra={
                "run_id": run_id,
                "project_key": self._config.project_key,
                "issue_count": len(issues),
            },
        )

        return issues

    async def _post_with_retries(
        self,
        *,
        run_id: str,
        url: str,
        json_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST to Jira with exponential backoff retry behavior."""
        if self._http_client is None:
            raise JiraClientError("Jira HTTP client is not initialized.")

        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = await self._http_client.post(url, json=json_payload)

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise JiraClientError(
                        f"Jira transient error: status_code={response.status_code}"
                    )

                response.raise_for_status()

                data = response.json()

                if not isinstance(data, dict):
                    raise JiraClientError("Jira response body was not a JSON object.")

                return data

            except (httpx.HTTPError, JiraClientError) as error:
                last_error = error

                logger.warning(
                    "jira_request_attempt_failed",
                    extra={
                        "run_id": run_id,
                        "attempt": attempt,
                        "max_retries": self._config.max_retries,
                        "error_type": type(error).__name__,
                    },
                )

                if attempt == self._config.max_retries:
                    break

                await asyncio.sleep(
                    self._config.retry_base_delay_seconds * (2 ** (attempt - 1))
                )

        raise JiraClientError("Failed to query Jira after retries.") from last_error

    def _parse_issue(self, *, raw_issue: dict[str, Any]) -> JiraIssue:
        """Normalize one raw Jira issue payload into a JiraIssue."""
        fields = raw_issue.get("fields", {})

        if not isinstance(fields, dict):
            raise JiraClientError("Jira issue fields payload was invalid.")

        issue_key = str(raw_issue.get("key", "")).strip()
        issue_url = f"{self._config.base_url.rstrip('/')}/browse/{issue_key}"

        return JiraIssue(
            issue_key=issue_key,
            title=self._get_string(fields.get("summary")),
            description=self._get_optional_description(fields.get("description")),
            issue_type=self._map_issue_type(fields.get("issuetype")),
            status=self._map_status(fields.get("status")),
            priority=self._map_priority(fields.get("priority")),
            assignee=self._get_user_email(fields.get("assignee")),
            reporter=self._get_user_email(fields.get("reporter")),
            labels=self._get_string_list(fields.get("labels")),
            components=self._get_component_names(fields.get("components")),
            affected_services=self._get_component_names(fields.get("components")),
            issue_url=issue_url,
            created_at=self._parse_datetime(fields.get("created")),
            updated_at=self._parse_datetime(fields.get("updated")),
            due_at=self._parse_optional_date(fields.get("duedate")),
            is_blocking_release=self._is_release_blocking(fields=fields),
            linked_pull_request_urls=[],
        )

    @staticmethod
    def _get_string(value: Any) -> str:
        """Convert a raw Jira value to a string."""
        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def _get_string_list(value: Any) -> list[str]:
        """Extract a clean string list from a raw Jira value."""
        if not isinstance(value, list):
            return []

        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _get_component_names(value: Any) -> list[str]:
        """Extract Jira component names from raw component objects."""
        if not isinstance(value, list):
            return []

        component_names: list[str] = []

        for item in value:
            if not isinstance(item, dict):
                continue

            name = item.get("name")

            if isinstance(name, str) and name.strip():
                component_names.append(name.strip())

        return component_names

    @staticmethod
    def _get_user_email(value: Any) -> str | None:
        """Extract user email or display name from a Jira user object."""
        if not isinstance(value, dict):
            return None

        email = value.get("emailAddress")
        display_name = value.get("displayName")

        if isinstance(email, str) and email.strip():
            return email.strip()

        if isinstance(display_name, str) and display_name.strip():
            return display_name.strip()

        return None

    @staticmethod
    def _get_optional_description(value: Any) -> str | None:
        """Convert Jira description into a safe optional string.

        Jira Cloud descriptions can be Atlassian Document Format objects.
        For now, we keep this conservative and avoid logging or deeply parsing
        potentially sensitive rich-text content.
        """
        if value is None:
            return None

        if isinstance(value, str):
            return value.strip() or None

        return None

    @staticmethod
    def _map_issue_type(value: Any) -> JiraIssueType:
        """Map Jira issue type names into normalized issue types."""
        if not isinstance(value, dict):
            return JiraIssueType.TASK

        raw_name = str(value.get("name", "")).strip().lower()

        if raw_name == "bug":
            return JiraIssueType.BUG

        if raw_name == "incident":
            return JiraIssueType.INCIDENT

        if raw_name == "story":
            return JiraIssueType.STORY

        if raw_name == "epic":
            return JiraIssueType.EPIC

        return JiraIssueType.TASK

    @staticmethod
    def _map_status(value: Any) -> JiraIssueStatus:
        """Map Jira workflow status into normalized release-risk status."""
        if not isinstance(value, dict):
            return JiraIssueStatus.TO_DO

        raw_name = str(value.get("name", "")).strip().lower()

        if raw_name in {"in progress", "development", "implementing"}:
            return JiraIssueStatus.IN_PROGRESS

        if raw_name in {"in review", "code review", "review"}:
            return JiraIssueStatus.IN_REVIEW

        if raw_name in {"blocked", "blocker"}:
            return JiraIssueStatus.BLOCKED

        if raw_name in {"done", "closed", "resolved"}:
            return JiraIssueStatus.DONE

        return JiraIssueStatus.TO_DO

    @staticmethod
    def _map_priority(value: Any) -> JiraIssuePriority:
        """Map Jira priority names into normalized priority values."""
        if not isinstance(value, dict):
            return JiraIssuePriority.P3

        raw_name = str(value.get("name", "")).strip().lower()

        if raw_name in {"highest", "blocker", "critical", "p0"}:
            return JiraIssuePriority.P0

        if raw_name in {"high", "major", "p1"}:
            return JiraIssuePriority.P1

        if raw_name in {"medium", "p2"}:
            return JiraIssuePriority.P2

        if raw_name in {"low", "minor", "p3"}:
            return JiraIssuePriority.P3

        return JiraIssuePriority.P4

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        """Parse Jira datetime strings into timezone-aware datetimes."""
        if not isinstance(value, str) or not value.strip():
            return datetime.now(UTC)

        normalized_value = value.replace("Z", "+00:00")

        return datetime.fromisoformat(normalized_value)

    @staticmethod
    def _parse_optional_date(value: Any) -> datetime | None:
        """Parse Jira due date value into a timezone-aware datetime."""
        if not isinstance(value, str) or not value.strip():
            return None

        return datetime.fromisoformat(value).replace(tzinfo=UTC)

    @staticmethod
    def _is_release_blocking(*, fields: dict[str, Any]) -> bool:
        """Infer whether a Jira issue is explicitly release blocking."""
        labels = JiraClient._get_string_list(fields.get("labels"))
        normalized_labels = {label.lower() for label in labels}

        return bool({"release-blocker", "block-release", "release-risk"} & normalized_labels)