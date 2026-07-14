import httpx
import pytest

from app.integrations.github_client import (
    GitHubClient,
    GitHubClientConfig,
    GitHubClientError,
)
from app.schemas.github import GitHubCIStatus, GitHubRepositoryConfig


def _github_pr_payload() -> dict[str, object]:
    """Return a fake GitHub pull request payload."""
    return {
        "number": 42,
        "title": "Fix payment retry timeout",
        "user": {"login": "engineer1"},
        "html_url": "https://github.com/acme/backend-services/pull/42",
        "url": "https://api.github.com/repos/acme/backend-services/pulls/42",
        "head": {"ref": "fix/payment-timeout"},
        "base": {"ref": "main"},
        "state": "open",
        "draft": False,
        "created_at": "2026-07-01T10:00:00Z",
        "updated_at": "2026-07-01T12:00:00Z",
        "changed_files": 4,
        "additions": 120,
        "deletions": 30,
        "labels": [{"name": "payments"}, {"name": "release-risk"}],
    }


def _github_config() -> GitHubClientConfig:
    """Return GitHub client config for tests."""
    return GitHubClientConfig(
        repository=GitHubRepositoryConfig(
            owner="acme",
            repo="backend-services",
            default_branch="main",
        ),
        token="test-token",  # noqa: S106 - fake credential used only for testing
        api_base_url="https://api.github.test",
        retry_backoff_seconds=0.001,
    )


@pytest.mark.anyio
async def test_list_open_pull_requests_returns_normalized_pull_requests() -> None:
    """GitHub client should return normalized pull request schemas."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/acme/backend-services/pulls"
        assert request.url.params["state"] == "open"
        assert request.url.params["base"] == "main"
        assert request.headers["Authorization"] == "Bearer test-token"

        return httpx.Response(
            status_code=200,
            json=[_github_pr_payload()],
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = GitHubClient(
            http_client=http_client,
            config=_github_config(),
            request_id="test-request-id",
        )

        pull_requests = await client.list_open_pull_requests()

    assert len(pull_requests) == 1

    pull_request = pull_requests[0]

    assert pull_request.number == 42
    assert pull_request.title == "Fix payment retry timeout"
    assert pull_request.author == "engineer1"
    assert pull_request.head_branch == "fix/payment-timeout"
    assert pull_request.base_branch == "main"
    assert pull_request.changed_files == 4
    assert pull_request.total_code_changes == 150
    assert pull_request.ci_status == GitHubCIStatus.UNKNOWN
    assert pull_request.labels == ["payments", "release-risk"]


@pytest.mark.anyio
async def test_list_open_pull_requests_retries_transient_server_error() -> None:
    """GitHub client should retry temporary 5xx failures."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1

        if attempts["count"] == 1:
            return httpx.Response(
                status_code=500,
                json={"message": "temporary server error"},
            )

        return httpx.Response(
            status_code=200,
            json=[_github_pr_payload()],
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = GitHubClient(
            http_client=http_client,
            config=_github_config(),
            request_id="test-request-id",
        )

        pull_requests = await client.list_open_pull_requests()

    assert attempts["count"] == 2
    assert len(pull_requests) == 1


@pytest.mark.anyio
async def test_list_open_pull_requests_raises_error_for_auth_failure() -> None:
    """GitHub client should not retry authentication failures."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={"message": "bad credentials"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = GitHubClient(
            http_client=http_client,
            config=_github_config(),
            request_id="test-request-id",
        )

        with pytest.raises(GitHubClientError, match="GitHub API request failed."):
            await client.list_open_pull_requests()


@pytest.mark.anyio
async def test_list_open_pull_requests_raises_error_for_invalid_payload() -> None:
    """GitHub client should reject invalid GitHub response shapes."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"unexpected": "object"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = GitHubClient(
            http_client=http_client,
            config=_github_config(),
            request_id="test-request-id",
        )

        with pytest.raises(
            GitHubClientError,
            match="GitHub pull request response must be a list.",
        ):
            await client.list_open_pull_requests()