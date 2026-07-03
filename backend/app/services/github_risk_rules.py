"""Rule-based GitHub pull request risk detection for AgentFlow AI.

This module converts GitHubPullRequest objects into deterministic risk signals.
These signals are used before ML/XGBoost so the platform has explainable,
auditable baseline risk detection.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.github import GitHubPullRequest

logger = logging.getLogger(__name__)

EvidenceValue = str | int | float | bool


class RiskSeverity(StrEnum):
    """Severity level for a detected release risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(StrEnum):
    """Business category for a detected risk signal."""

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


class RiskRuleEngineConfig(BaseModel):
    """Configuration thresholds for GitHub pull request risk rules."""

    model_config = ConfigDict(frozen=True)

    stale_pr_days: int = Field(default=5, ge=1)
    critical_stale_pr_days: int = Field(default=14, ge=1)
    large_changed_files_threshold: int = Field(default=20, ge=1)
    critical_changed_files_threshold: int = Field(default=50, ge=1)
    large_lines_changed_threshold: int = Field(default=500, ge=1)
    critical_lines_changed_threshold: int = Field(default=1500, ge=1)
    jira_key_pattern: str = r"\b[A-Z][A-Z0-9]+-\d+\b"
    critical_path_patterns: tuple[str, ...] = (
        r"(^|/)(payment|payments|billing)(/|\.|_|-|$)",
        r"(^|/)(auth|authentication|authorization|security)(/|\.|_|-|$)",
        r"(^|/)(migration|migrations|database|db)(/|\.|_|-|$)",
        r"(^|/)(infra|terraform|k8s|helm|docker)(/|\.|_|-|$)",
    )


class RiskSignal(BaseModel):
    """Single explainable risk signal generated from a release-risk source."""

    model_config = ConfigDict(frozen=True)

    source_type: Literal["github_pull_request", "jira_issue"] = "github_pull_request"
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    rule_id: str = Field(min_length=1)
    category: RiskCategory
    severity: RiskSeverity
    score: float = Field(ge=0.0, le=1.0)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence: dict[str, EvidenceValue] = Field(default_factory=dict)


class PullRequestRiskResult(BaseModel):
    """Risk evaluation result for a single GitHub pull request."""

    model_config = ConfigDict(frozen=True)

    source_type: Literal["github_pull_request"] = "github_pull_request"
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    pull_request_number: int = Field(ge=1)
    total_score: float = Field(ge=0.0, le=1.0)
    max_severity: RiskSeverity | None = None
    signals: list[RiskSignal] = Field(default_factory=list)
    evaluated_at: datetime


class RiskRuleEngine:
    """Evaluate GitHub pull requests using deterministic release-risk rules."""

    def __init__(self, config: RiskRuleEngineConfig | None = None) -> None:
        """Initialize the rule engine with validated threshold configuration."""
        self._config = config or RiskRuleEngineConfig()
        self._jira_key_regex = re.compile(self._config.jira_key_pattern)
        self._critical_path_regexes = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in self._config.critical_path_patterns
        )

    def evaluate_pull_requests(
        self,
        pull_requests: Sequence[GitHubPullRequest],
        *,
        run_id: str,
    ) -> list[PullRequestRiskResult]:
        """Evaluate many pull requests and return risk results in input order."""
        logger.info(
            "risk_rules.evaluate_pull_requests.started",
            extra={
                "run_id": run_id,
                "pull_request_count": len(pull_requests),
            },
        )

        results = [
            self.evaluate_pull_request(pull_request, run_id=run_id)
            for pull_request in pull_requests
        ]

        logger.info(
            "risk_rules.evaluate_pull_requests.completed",
            extra={
                "run_id": run_id,
                "pull_request_count": len(results),
                "risk_signal_count": sum(len(result.signals) for result in results),
            },
        )

        return results

    def evaluate_pull_request(
        self,
        pull_request: GitHubPullRequest,
        *,
        run_id: str,
    ) -> PullRequestRiskResult:
        """Evaluate one GitHub pull request and return explainable risk signals."""
        pull_request_number = self._get_int_field(pull_request, "number", default=0)
        source_id = f"PR-{pull_request_number}"
        source_url = self._get_optional_str_field(pull_request, "url")

        logger.info(
            "risk_rules.evaluate_pull_request.started",
            extra={
                "run_id": run_id,
                "pull_request_number": pull_request_number,
            },
        )

        signals = [
            signal
            for signal in (
                self._evaluate_ci_status(pull_request, source_id, source_url),
                self._evaluate_review_status(pull_request, source_id, source_url),
                self._evaluate_staleness(pull_request, source_id, source_url),
                self._evaluate_changeset_size(pull_request, source_id, source_url),
                self._evaluate_draft_status(pull_request, source_id, source_url),
                self._evaluate_jira_link(pull_request, source_id, source_url),
                self._evaluate_critical_file_changes(
                    pull_request,
                    source_id,
                    source_url,
                ),
            )
            if signal is not None
        ]

        result = PullRequestRiskResult(
            source_id=source_id,
            source_url=source_url,
            pull_request_number=pull_request_number,
            total_score=self._calculate_total_score(signals),
            max_severity=self._get_max_severity(signals),
            signals=signals,
            evaluated_at=datetime.now(UTC),
        )

        logger.info(
            "risk_rules.evaluate_pull_request.completed",
            extra={
                "run_id": run_id,
                "pull_request_number": pull_request_number,
                "risk_signal_count": len(signals),
                "total_score": result.total_score,
                "max_severity": result.max_severity.value
                if result.max_severity is not None
                else None,
            },
        )

        return result

    def _evaluate_ci_status(
        self,
        pull_request: GitHubPullRequest,
        source_id: str,
        source_url: str | None,
    ) -> RiskSignal | None:
        ci_status = self._normalize_field(pull_request, "ci_status")

        if ci_status in {"failure", "failed", "error", "cancelled"}:
            return RiskSignal(
                source_id=source_id,
                source_url=source_url,
                rule_id="github_ci_failure",
                category=RiskCategory.CI_FAILURE,
                severity=RiskSeverity.HIGH,
                score=0.85,
                title="Pull request has failing CI",
                description=(
                    "The pull request has a failing or errored CI status. "
                    "Deploying this change may introduce broken code into release."
                ),
                evidence={"ci_status": ci_status},
            )

        if ci_status in {"pending", "queued", "in_progress", "unknown", "missing"}:
            return RiskSignal(
                source_id=source_id,
                source_url=source_url,
                rule_id="github_ci_not_green",
                category=RiskCategory.CI_PENDING,
                severity=RiskSeverity.MEDIUM,
                score=0.45,
                title="Pull request CI is not confirmed green",
                description=(
                    "The pull request CI status is pending, unknown, or missing. "
                    "Release confidence is lower until automated checks pass."
                ),
                evidence={"ci_status": ci_status},
            )

        return None

    def _evaluate_review_status(
        self,
        pull_request: GitHubPullRequest,
        source_id: str,
        source_url: str | None,
    ) -> RiskSignal | None:
        review_state = self._normalize_field(pull_request, "review_state")

        if review_state in {"changes_requested", "blocked", "rejected"}:
            return RiskSignal(
                source_id=source_id,
                source_url=source_url,
                rule_id="github_review_blocked",
                category=RiskCategory.REVIEW_BLOCKED,
                severity=RiskSeverity.HIGH,
                score=0.75,
                title="Pull request has blocking review feedback",
                description=(
                    "A reviewer has requested changes or blocked the pull request. "
                    "This should not be treated as release-ready."
                ),
                evidence={"review_state": review_state},
            )

        if review_state in {
            "pending",
            "review_required",
            "no_review",
            "no_reviews",
            "missing",
            "unknown",
        }:
            return RiskSignal(
                source_id=source_id,
                source_url=source_url,
                rule_id="github_review_missing",
                category=RiskCategory.REVIEW_MISSING,
                severity=RiskSeverity.MEDIUM,
                score=0.50,
                title="Pull request is missing approval",
                description=(
                    "The pull request does not have a confirmed approval. "
                    "Human review is required before release confidence is high."
                ),
                evidence={"review_state": review_state},
            )

        return None

    def _evaluate_staleness(
        self,
        pull_request: GitHubPullRequest,
        source_id: str,
        source_url: str | None,
    ) -> RiskSignal | None:
        created_at = self._get_datetime_field(pull_request, "created_at")
        if created_at is None:
            return None

        age_days = (datetime.now(UTC) - created_at).days

        if age_days >= self._config.critical_stale_pr_days:
            return RiskSignal(
                source_id=source_id,
                source_url=source_url,
                rule_id="github_pr_critically_stale",
                category=RiskCategory.STALE_PULL_REQUEST,
                severity=RiskSeverity.HIGH,
                score=0.70,
                title="Pull request is critically stale",
                description=(
                    "The pull request has been open for a long time. "
                    "It may be outdated, difficult to merge, or hiding unresolved risk."
                ),
                evidence={"age_days": age_days},
            )

        if age_days >= self._config.stale_pr_days:
            return RiskSignal(
                source_id=source_id,
                source_url=source_url,
                rule_id="github_pr_stale",
                category=RiskCategory.STALE_PULL_REQUEST,
                severity=RiskSeverity.MEDIUM,
                score=0.45,
                title="Pull request is stale",
                description=(
                    "The pull request has been open longer than the allowed threshold. "
                    "Stale changes increase release uncertainty."
                ),
                evidence={"age_days": age_days},
            )

        return None

    def _evaluate_changeset_size(
        self,
        pull_request: GitHubPullRequest,
        source_id: str,
        source_url: str | None,
    ) -> RiskSignal | None:
        changed_files = self._get_int_field(pull_request, "changed_files", default=0)
        additions = self._get_int_field(pull_request, "additions", default=0)
        deletions = self._get_int_field(pull_request, "deletions", default=0)
        lines_changed = additions + deletions

        if (
            changed_files >= self._config.critical_changed_files_threshold
            or lines_changed >= self._config.critical_lines_changed_threshold
        ):
            return RiskSignal(
                source_id=source_id,
                source_url=source_url,
                rule_id="github_critical_changeset",
                category=RiskCategory.LARGE_CHANGESET,
                severity=RiskSeverity.HIGH,
                score=0.70,
                title="Pull request has a very large changeset",
                description=(
                    "The pull request changes many files or lines. "
                    "Large changes are harder to review and increase regression risk."
                ),
                evidence={
                    "changed_files": changed_files,
                    "additions": additions,
                    "deletions": deletions,
                    "lines_changed": lines_changed,
                },
            )

        if (
            changed_files >= self._config.large_changed_files_threshold
            or lines_changed >= self._config.large_lines_changed_threshold
        ):
            return RiskSignal(
                source_id=source_id,
                source_url=source_url,
                rule_id="github_large_changeset",
                category=RiskCategory.LARGE_CHANGESET,
                severity=RiskSeverity.MEDIUM,
                score=0.50,
                title="Pull request has a large changeset",
                description=(
                    "The pull request is larger than normal. "
                    "This may require extra reviewer attention before release."
                ),
                evidence={
                    "changed_files": changed_files,
                    "additions": additions,
                    "deletions": deletions,
                    "lines_changed": lines_changed,
                },
            )

        return None

    def _evaluate_draft_status(
        self,
        pull_request: GitHubPullRequest,
        source_id: str,
        source_url: str | None,
    ) -> RiskSignal | None:
        is_draft = self._get_bool_field(pull_request, "is_draft", default=False)

        if not is_draft:
            return None

        return RiskSignal(
            source_id=source_id,
            source_url=source_url,
            rule_id="github_pr_draft",
            category=RiskCategory.DRAFT_PULL_REQUEST,
            severity=RiskSeverity.LOW,
            score=0.25,
            title="Pull request is still in draft",
            description=(
                "The pull request is marked as draft and should not be considered "
                "release-ready until the author marks it ready for review."
            ),
            evidence={"is_draft": is_draft},
        )

    def _evaluate_jira_link(
        self,
        pull_request: GitHubPullRequest,
        source_id: str,
        source_url: str | None,
    ) -> RiskSignal | None:
        if self._has_jira_key(pull_request):
            return None

        return RiskSignal(
            source_id=source_id,
            source_url=source_url,
            rule_id="github_missing_jira_link",
            category=RiskCategory.MISSING_JIRA_LINK,
            severity=RiskSeverity.MEDIUM,
            score=0.40,
            title="Pull request is missing Jira traceability",
            description=(
                "The pull request title or branch does not include a Jira ticket key. "
                "This weakens auditability and makes release approval harder."
            ),
            evidence={"jira_key_found": False},
        )

    def _evaluate_critical_file_changes(
        self,
        pull_request: GitHubPullRequest,
        source_id: str,
        source_url: str | None,
    ) -> RiskSignal | None:
        file_paths = self._get_file_paths(pull_request)
        if not file_paths:
            return None

        critical_paths = self._find_critical_paths(file_paths)
        if not critical_paths:
            return None

        return RiskSignal(
            source_id=source_id,
            source_url=source_url,
            rule_id="github_critical_file_change",
            category=RiskCategory.CRITICAL_FILE_CHANGE,
            severity=RiskSeverity.HIGH,
            score=0.65,
            title="Pull request changes critical system files",
            description=(
                "The pull request touches payment, auth, database, security, "
                "or infrastructure-related files. These areas require extra review."
            ),
            evidence={
                "critical_file_count": len(critical_paths),
                "critical_file_examples": ", ".join(critical_paths[:5]),
            },
        )

    def _has_jira_key(self, pull_request: GitHubPullRequest) -> bool:
        linked_jira_keys = getattr(pull_request, "linked_jira_keys", None)
        if isinstance(linked_jira_keys, list) and len(linked_jira_keys) > 0:
            return True

        searchable_text = " ".join(
            value
            for value in (
                self._get_optional_str_field(pull_request, "title"),
                self._get_optional_str_field(pull_request, "head_branch"),
                self._get_optional_str_field(pull_request, "base_branch"),
                self._get_optional_str_field(pull_request, "body"),
            )
            if value
        )

        return self._jira_key_regex.search(searchable_text) is not None

    def _find_critical_paths(self, file_paths: Sequence[str]) -> list[str]:
        return [
            file_path
            for file_path in file_paths
            if any(pattern.search(file_path) for pattern in self._critical_path_regexes)
        ]

    def _get_file_paths(self, pull_request: GitHubPullRequest) -> list[str]:
        for field_name in ("changed_file_paths", "file_paths"):
            value = getattr(pull_request, field_name, None)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, str)]

        return []

    def _calculate_total_score(self, signals: Sequence[RiskSignal]) -> float:
        if not signals:
            return 0.0

        safe_probability = 1.0
        for signal in signals:
            safe_probability *= 1.0 - signal.score

        combined_score = 1.0 - safe_probability
        return round(min(combined_score, 1.0), 4)

    def _get_max_severity(
        self,
        signals: Sequence[RiskSignal],
    ) -> RiskSeverity | None:
        if not signals:
            return None

        severity_rank = {
            RiskSeverity.LOW: 1,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.HIGH: 3,
            RiskSeverity.CRITICAL: 4,
        }

        return max(signals, key=lambda signal: severity_rank[signal.severity]).severity

    def _normalize_field(self, pull_request: GitHubPullRequest, field_name: str) -> str:
        value = getattr(pull_request, field_name, None)

        if value is None:
            return "unknown"

        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return str(enum_value).lower()

        return str(value).lower()

    def _get_optional_str_field(
        self,
        pull_request: GitHubPullRequest,
        field_name: str,
    ) -> str | None:
        value = getattr(pull_request, field_name, None)
        if value is None:
            return None

        return str(value)

    def _get_int_field(
        self,
        pull_request: GitHubPullRequest,
        field_name: str,
        *,
        default: int,
    ) -> int:
        value = getattr(pull_request, field_name, default)
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value

        return default

    def _get_bool_field(
        self,
        pull_request: GitHubPullRequest,
        field_name: str,
        *,
        default: bool,
    ) -> bool:
        value = getattr(pull_request, field_name, default)
        if isinstance(value, bool):
            return value

        return default

    def _get_datetime_field(
        self,
        pull_request: GitHubPullRequest,
        field_name: str,
    ) -> datetime | None:
        value = getattr(pull_request, field_name, None)
        if not isinstance(value, datetime):
            return None

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)

        return value.astimezone(UTC)