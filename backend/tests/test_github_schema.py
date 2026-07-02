from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.github import (
    GitHubCIStatus,
    GitHubPullRequest,
    GitHubPullRequestState,
    GitHubRepositoryConfig,
    GitHubReviewState,
)


def test_github_repository_config_accepts_valid_values() -> None:
    """Repository config should validate GitHub owner and repo names."""
    config = GitHubRepositoryConfig(
        owner="acme",
        repo="backend-services",
    )

    assert config.owner == "acme"
    assert config.repo == "backend-services"
    assert config.default_branch == "main"


def test_github_pull_request_accepts_valid_values() -> None:
    """GitHub pull request schema should accept normalized PR data."""
    pull_request = GitHubPullRequest(
        number=42,
        title="Fix payment retry timeout",
        author="engineer1",
        url="https://github.com/acme/backend-services/pull/42",
        head_branch="fix/payment-timeout",
        base_branch="main",
        state=GitHubPullRequestState.OPEN,
        is_draft=False,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        changed_files=4,
        additions=120,
        deletions=30,
        ci_status=GitHubCIStatus.FAILURE,
        review_state=GitHubReviewState.REVIEW_REQUIRED,
        labels=["payments", "release-risk"],
    )

    assert pull_request.number == 42
    assert pull_request.ci_status == GitHubCIStatus.FAILURE
    assert pull_request.review_state == GitHubReviewState.REVIEW_REQUIRED
    assert pull_request.total_code_changes == 150


def test_github_pull_request_rejects_invalid_number() -> None:
    """GitHub pull request number must be positive."""
    with pytest.raises(ValidationError):
        GitHubPullRequest(
            number=0,
            title="Invalid PR",
            author="engineer1",
            url="https://github.com/acme/backend-services/pull/0",
            head_branch="bugfix",
            base_branch="main",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            changed_files=1,
            additions=1,
            deletions=0,
        )


def test_github_pull_request_rejects_negative_file_count() -> None:
    """Changed file count cannot be negative."""
    with pytest.raises(ValidationError):
        GitHubPullRequest(
            number=1,
            title="Invalid changed files",
            author="engineer1",
            url="https://github.com/acme/backend-services/pull/1",
            head_branch="bugfix",
            base_branch="main",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            changed_files=-1,
            additions=1,
            deletions=0,
        )