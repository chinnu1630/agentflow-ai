from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class GitHubPullRequestState(StrEnum):
    """Allowed GitHub pull request states."""

    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class GitHubCIStatus(StrEnum):
    """Normalized CI status for a GitHub pull request."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    PENDING = "pending"
    UNKNOWN = "unknown"


class GitHubReviewState(StrEnum):
    """Normalized review state for a GitHub pull request."""

    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REVIEW_REQUIRED = "review_required"
    COMMENTED = "commented"
    UNKNOWN = "unknown"


class GitHubRepositoryConfig(BaseModel):
    """GitHub repository configuration used by the integration client."""

    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    default_branch: str = Field(default="main", min_length=1, max_length=100)


class GitHubPullRequest(BaseModel):
    """Normalized GitHub pull request data used by AgentFlow AI.

    We do not pass raw GitHub API dictionaries through the system.
    This schema gives the EngOps Agent a stable, validated contract.
    """

    number: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=500)
    author: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=1, max_length=1000)

    head_branch: str = Field(min_length=1, max_length=255)
    base_branch: str = Field(min_length=1, max_length=255)

    state: GitHubPullRequestState = GitHubPullRequestState.OPEN
    is_draft: bool = False

    created_at: datetime
    updated_at: datetime

    changed_files: int = Field(ge=0)
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)

    ci_status: GitHubCIStatus = GitHubCIStatus.UNKNOWN
    review_state: GitHubReviewState = GitHubReviewState.UNKNOWN
    labels: list[str] = Field(default_factory=list)

    @property
    def total_code_changes(self) -> int:
        """Return total changed lines in the pull request."""
        return self.additions + self.deletions