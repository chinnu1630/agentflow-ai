"""Risk response schemas for AgentFlow AI APIs.

These schemas define the public API contract for release-risk results.
They are intentionally separate from service-layer models so the API shape
can remain stable even if internal workflow logic changes later.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
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
    OPEN_CRITICAL_BUG = "open_critical_bug"
    BLOCKED_JIRA_ISSUE = "blocked_jira_issue"
    RELEASE_BLOCKER_ISSUE = "release_blocker_issue"
    UNASSIGNED_HIGH_PRIORITY_ISSUE = "unassigned_high_priority_issue"
    DUE_SOON_ISSUE = "due_soon_issue"
    CRITICAL_SERVICE_ISSUE = "critical_service_issue"


class RiskCollectionStatusResponse(StrEnum):
    """API status for a risk collection operation."""

    SUCCESS = "success"
    DEGRADED = "degraded"


class RiskSummaryActionResponse(StrEnum):
    """API recommended action from deterministic risk summary."""

    PROCEED = "proceed"
    REVIEW_REQUIRED = "review_required"
    BLOCK_RELEASE = "block_release"
    PARTIAL_DATA_REVIEW = "partial_data_review"


class RiskSignalResponse(BaseModel):
    """API response schema for one explainable risk signal."""

    model_config = ConfigDict(from_attributes=True)

    source_type: Literal["github_pull_request", "jira_issue"]
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

class JiraIssueRiskResponse(BaseModel):
    """API response model for one Jira issue risk evaluation."""

    issue_key: str
    title: str
    issue_url: str | None = None
    signals: list[RiskSignalResponse] = Field(default_factory=list)


class JiraRiskCollectionResponse(BaseModel):
    """API response model for Jira risk collection results."""

    status: RiskCollectionStatusResponse
    total_issues_analyzed: int = Field(ge=0)
    total_signals: int = Field(ge=0)
    issues: list[JiraIssueRiskResponse] = Field(default_factory=list)
    signals: list[RiskSignalResponse] = Field(default_factory=list)
    error_message: str | None = None
    duration_ms: float = Field(ge=0)


class RiskSummaryItemResponse(BaseModel):
    """API response schema for one prioritized risk summary item."""

    model_config = ConfigDict(from_attributes=True)

    source_type: Literal["github_pull_request", "jira_issue"]
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    severity: RiskSeverityResponse
    score: float = Field(ge=0.0, le=1.0)
    title: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: dict[str, str | int | float | bool] = Field(default_factory=dict)


class GitHubRiskSummaryResponse(BaseModel):
    """API response schema for deterministic GitHub risk summary."""

    model_config = ConfigDict(from_attributes=True)

    source: Literal["github"]
    collection_status: RiskCollectionStatusResponse
    overall_severity: RiskSeverityResponse
    recommended_action: RiskSummaryActionResponse
    pull_request_count: int = Field(ge=0)
    risky_pull_request_count: int = Field(ge=0)
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    top_risks: list[RiskSummaryItemResponse] = Field(default_factory=list)
    summary_text: str = Field(min_length=1)
    generated_at: datetime


class JiraRiskSummaryResponse(BaseModel):
    """API response model for Jira risk summary."""

    model_config = ConfigDict(from_attributes=True)

    source: Literal["jira"] = "jira"
    collection_status: RiskCollectionStatusResponse
    overall_severity: RiskSeverityResponse
    recommended_action: RiskSummaryActionResponse
    issue_count: int = Field(ge=0)
    risky_issue_count: int = Field(ge=0)
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    top_risks: list[RiskSummaryItemResponse] = Field(default_factory=list)
    summary_text: str
    generated_at: datetime

class ReleaseRiskSummaryItemResponse(BaseModel):
    """API response model for one combined release-risk item."""

    model_config = ConfigDict(from_attributes=True)

    source: Literal["github", "jira"]
    source_type: Literal["github_pull_request", "jira_issue"]
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    severity: RiskSeverityResponse
    score: float = Field(ge=0.0, le=1.0)
    title: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReleaseRiskSourceSummaryResponse(BaseModel):
    """API response model for one source summary inside release summary."""

    model_config = ConfigDict(from_attributes=True)

    source: Literal["github", "jira"]
    overall_severity: RiskSeverityResponse
    recommended_action: str
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    summary_text: str = Field(min_length=1)


class ReleaseRiskSummaryResponse(BaseModel):
    """API response model for combined release risk summary."""

    model_config = ConfigDict(from_attributes=True)

    source: Literal["release"] = "release"
    overall_severity: RiskSeverityResponse
    recommended_action: RiskSummaryActionResponse
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    source_summary_count: int = Field(ge=0)
    top_risks: list[ReleaseRiskSummaryItemResponse] = Field(default_factory=list)
    source_summaries: list[ReleaseRiskSourceSummaryResponse] = Field(
        default_factory=list
    )
    summary_text: str = Field(min_length=1)
    generated_at: datetime


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




class KnowledgeContextResultResponse(BaseModel):
    """API response schema for one retrieved engineering knowledge chunk.

    The retrieval layer is still evolving. This response model keeps the
    public contract stable while allowing future retrieval metadata such as
    BM25 score, vector score, reranker score, document source, or section name.
    """

    model_config = ConfigDict(extra="allow")

    document_id: UUID | None = None
    chunk_id: UUID | None = None
    source_type: str | None = None
    title: str | None = None
    content: str | None = None
    score: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReleaseRunRiskResponse(BaseModel):
    """API response model for release run risk analysis."""

    model_config = ConfigDict(from_attributes=True)

    release_run: ReleaseRunSummaryResponse
    github: GitHubRiskCollectionResponse
    github_summary: GitHubRiskSummaryResponse
    jira: JiraRiskCollectionResponse
    jira_summary: JiraRiskSummaryResponse
    release_summary: ReleaseRiskSummaryResponse
    knowledge_query: str | None = None
    knowledge_status: str | None = None
    knowledge_results: list[KnowledgeContextResultResponse] = Field(
        default_factory=list,
    )
    knowledge_error: str | None = None
