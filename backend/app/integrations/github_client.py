import asyncio
from json import JSONDecodeError
from typing import Any

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from app.core.logging import get_logger
from app.schemas.github import (
    GitHubCIStatus,
    GitHubPullRequest,
    GitHubPullRequestState,
    GitHubRepositoryConfig,
    GitHubReviewState,
)

logger = get_logger(__name__)


class GitHubClientConfig(BaseModel):
    """Configuration for the GitHub API client."""

    repository: GitHubRepositoryConfig
    token: SecretStr | None = None
    api_base_url: str = Field(
        default="https://api.github.com",
        min_length=1,
        max_length=500,
    )
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    max_retries: int = Field(default=3, ge=1, le=5)
    retry_backoff_seconds: float = Field(default=0.25, gt=0, le=5)


class GitHubClientError(RuntimeError):
    """Raised when GitHub API operations fail."""


class GitHubClient:
    """Async client for reading GitHub pull request data."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        config: GitHubClientConfig,
        request_id: str,
    ) -> None:
        """Initialize the GitHub client."""
        self._http_client = http_client
        self._config = config
        self._request_id = request_id

    async def list_open_pull_requests(self) -> list[GitHubPullRequest]:
        """Fetch open pull requests from GitHub."""
        repository = self._config.repository

        logger.info(
            "github_pull_requests_fetch_started",
            extra={
                "request_id": self._request_id,
                "owner": repository.owner,
                "repo": repository.repo,
            },
        )

        payload = await self._request_json(
            method="GET",
            path=f"/repos/{repository.owner}/{repository.repo}/pulls",
            params={
                "state": "open",
                "base": repository.default_branch,
                "per_page": "100",
            },
        )

        if not isinstance(payload, list):
            raise GitHubClientError("GitHub pull request response must be a list.")

        try:
            pull_requests = [
                self._normalize_pull_request(raw_pull_request)
                for raw_pull_request in payload
            ]
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            logger.exception(
                "github_pull_requests_validation_failed",
                extra={
                    "request_id": self._request_id,
                    "owner": repository.owner,
                    "repo": repository.repo,
                },
            )
            raise GitHubClientError(
                "GitHub pull request response validation failed."
            ) from exc

        logger.info(
            "github_pull_requests_fetch_completed",
            extra={
                "request_id": self._request_id,
                "owner": repository.owner,
                "repo": repository.repo,
                "count": len(pull_requests),
            },
        )

        return pull_requests

    async def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
    ) -> object:
        """Send a GitHub API request with exponential backoff retry."""
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = await self._http_client.request(
                    method=method,
                    url=f"{self._config.api_base_url}{path}",
                    headers=self._build_headers(),
                    params=params,
                    timeout=self._config.request_timeout_seconds,
                )

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        message="GitHub API server error.",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()

                try:
                    return response.json()
                except JSONDecodeError as exc:
                    raise GitHubClientError(
                        "GitHub API returned invalid JSON."
                    ) from exc

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc

                should_retry = self._should_retry(
                    error=exc,
                    attempt=attempt,
                )

                if not should_retry:
                    self._log_github_failure(exc)
                    raise GitHubClientError("GitHub API request failed.") from exc

                delay_seconds = self._calculate_backoff_delay(attempt)

                logger.warning(
                    "github_api_request_retrying",
                    extra={
                        "request_id": self._request_id,
                        "attempt": attempt,
                        "delay_seconds": delay_seconds,
                    },
                )

                await asyncio.sleep(delay_seconds)

        raise GitHubClientError("GitHub API request failed after retries.") from last_error

    def _build_headers(self) -> dict[str, str]:
        """Build GitHub API request headers without logging secrets."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        if self._config.token is not None:
            headers["Authorization"] = (
                f"Bearer {self._config.token.get_secret_value()}"
            )

        return headers

    def _should_retry(
        self,
        error: Exception,
        attempt: int,
    ) -> bool:
        """Return whether a failed GitHub request should be retried."""
        if attempt >= self._config.max_retries:
            return False

        if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
            return True

        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code >= 500

        return False

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay in seconds."""
        return float(
            self._config.retry_backoff_seconds * (2 ** (attempt - 1))
        )

    def _log_github_failure(self, error: Exception) -> None:
        """Log GitHub API failure without exposing secrets."""
        status_code: int | None = None

        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code

        logger.exception(
            "github_api_request_failed",
            extra={
                "request_id": self._request_id,
                "status_code": status_code,
            },
        )

    @staticmethod
    def _normalize_pull_request(
        raw_pull_request: dict[str, Any],
    ) -> GitHubPullRequest:
        """Normalize one raw GitHub pull request payload."""
        user = raw_pull_request.get("user") or {}
        head = raw_pull_request.get("head") or {}
        base = raw_pull_request.get("base") or {}

        labels = [
            label["name"]
            for label in raw_pull_request.get("labels", [])
            if isinstance(label, dict) and "name" in label
        ]

        state = (
            GitHubPullRequestState.MERGED
            if raw_pull_request.get("merged_at")
            else GitHubPullRequestState(raw_pull_request.get("state", "open"))
        )

        return GitHubPullRequest(
            number=raw_pull_request["number"],
            title=raw_pull_request["title"],
            author=user.get("login", "unknown"),
            url=raw_pull_request.get("html_url") or raw_pull_request["url"],
            head_branch=head.get("ref", "unknown"),
            base_branch=base.get("ref", "unknown"),
            state=state,
            is_draft=raw_pull_request.get("draft", False),
            created_at=raw_pull_request["created_at"],
            updated_at=raw_pull_request["updated_at"],
            changed_files=raw_pull_request.get("changed_files", 0),
            additions=raw_pull_request.get("additions", 0),
            deletions=raw_pull_request.get("deletions", 0),
            ci_status=GitHubCIStatus.UNKNOWN,
            review_state=GitHubReviewState.UNKNOWN,
            labels=labels,
        )
