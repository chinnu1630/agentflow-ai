"""Risk response schemas for AgentFlow AI APIs.

These schemas define the public API contract for release-risk results.
They are intentionally separate from service-layer models so the API shape
can remain stable even if internal workflow logic changes later.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RiskSeverityResponse(StrEnum):
    """API severity level for a detected risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategoryResponse(StrEnum):
    """API category for a detected risk signal."""

    CI_FAILURE = "ci_failure"
    CI_PENDING = "ci_pending"
    REVIEW_BLOCKED = "review_blocked"
    REVIEW_MISSING = "review_missing"
    STALE_PULL_REQUEST = "stale_pull_request"
    LARGE_CHANGESET = "large_changeset"
    DRAFT_PULL_REQUEST = "draft_pull_request"
    MISSING_JIRA_LINK = "missing_jira_link"
    CRITICAL_FILE_CHANGE = "critical_file_change"


class RiskCollectionStatusResponse(StrEnum):
    """API status for a risk collection operation."""

    SUCCESS = "success"
    DEGRADED = "degraded"


class RiskSignalResponse(BaseModel):
    """API response schema for one explainable risk signal."""

    model_config = ConfigDict(from_attributes=True)

    source_type: Literal["github_pull_request"]
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    rule_id: str = Field(min_length=1)
    category: RiskCategoryResponse
    severity: RiskSeverityResponse
    score: float = Field(ge=0.0, le=1.0)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence: dict[str, str | int | float | bool] = Field(default_factory=dict)


class PullRequestRiskResponse(BaseModel):
    """API response schema for risk evaluation of one GitHub pull request."""

    model_config = ConfigDict(from_attributes=True)

    source_type: Literal["github_pull_request"]
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    pull_request_number: int = Field(ge=1)
    total_score: float = Field(ge=0.0, le=1.0)
    max_severity: RiskSeverityResponse | None = None
    signals: list[RiskSignalResponse] = Field(default_factory=list)
    evaluated_at: datetime


class GitHubRiskCollectionResponse(BaseModel):
    """API response schema for collected GitHub release risks."""

    model_config = ConfigDict(from_attributes=True)

    source: Literal["github"]
    status: RiskCollectionStatusResponse
    pull_request_count: int = Field(ge=0)
    risk_result_count: int = Field(ge=0)
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    risk_results: list[PullRequestRiskResponse] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    collected_at: datetime
    duration_ms: float = Field(ge=0.0)


class ReleaseRunSummaryResponse(BaseModel):
    """API response schema for release-run metadata inside risk responses."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: str
    query: str
    requested_by: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None


class ReleaseRunRiskResponse(BaseModel):
    """API response schema for release-run risk collection result."""

    model_config = ConfigDict(from_attributes=True)

    release_run: ReleaseRunSummaryResponse
    github: GitHubRiskCollectionResponse