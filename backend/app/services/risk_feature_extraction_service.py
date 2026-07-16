"""Feature extraction for deterministic AgentFlow AI release-risk scoring.

This module converts GitHub, Jira, and Knowledge Agent outputs into a stable
numeric feature vector. The vector is intentionally model-agnostic so it can be
used by the first rule-based scorer now and by an optional trained ML model
later when real labeled release outcome data exists.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class ReleaseRiskFeatureExtractionRequest(BaseModel):
    """Input payload for release-risk feature extraction.

    The request accepts dictionary-shaped outputs from the current LangGraph
    workflow/API response. Keeping this boundary dictionary-based makes the
    feature extractor stable even if internal service objects change later.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    github: dict[str, Any] | None = None
    jira: dict[str, Any] | None = None
    release_summary: dict[str, Any] | None = None
    knowledge_status: str | None = None
    knowledge_results: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_error: str | None = None


class ReleaseRiskFeatureVector(BaseModel):
    """Stable numeric feature vector for release-risk scoring.

    Feature versioning is important because future ML training data must know
    which exact feature contract produced each training row.
    """

    model_config = ConfigDict(frozen=True)

    feature_version: Literal["release_risk_features_v1"] = "release_risk_features_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    total_risk_count: int = Field(ge=0)
    github_risk_count: int = Field(ge=0)
    jira_risk_count: int = Field(ge=0)

    critical_risk_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    medium_risk_count: int = Field(ge=0)
    low_risk_count: int = Field(ge=0)

    ci_failure_count: int = Field(ge=0)
    ci_pending_count: int = Field(ge=0)
    review_blocked_count: int = Field(ge=0)
    review_missing_count: int = Field(ge=0)
    stale_pr_count: int = Field(ge=0)
    large_changeset_count: int = Field(ge=0)
    draft_pull_request_count: int = Field(ge=0)
    missing_jira_link_count: int = Field(ge=0)
    critical_file_change_count: int = Field(ge=0)

    open_critical_bug_count: int = Field(ge=0)
    blocked_jira_issue_count: int = Field(ge=0)
    release_blocker_issue_count: int = Field(ge=0)
    unassigned_high_priority_issue_count: int = Field(ge=0)
    due_soon_issue_count: int = Field(ge=0)
    critical_service_issue_count: int = Field(ge=0)

    knowledge_result_count: int = Field(ge=0)
    knowledge_no_results: bool
    knowledge_failed: bool

    github_degraded: bool
    jira_degraded: bool

    max_rule_score: float = Field(ge=0.0, le=1.0)
    average_rule_score: float = Field(ge=0.0, le=1.0)


class RiskFeatureExtractionService:
    """Convert release-risk workflow outputs into numeric scoring features."""

    _CATEGORY_TO_FEATURE_FIELD: Mapping[str, str] = {
        "ci_failure": "ci_failure_count",
        "ci_pending": "ci_pending_count",
        "review_blocked": "review_blocked_count",
        "review_missing": "review_missing_count",
        "stale_pull_request": "stale_pr_count",
        "large_changeset": "large_changeset_count",
        "draft_pull_request": "draft_pull_request_count",
        "missing_jira_link": "missing_jira_link_count",
        "critical_file_change": "critical_file_change_count",
        "open_critical_bug": "open_critical_bug_count",
        "blocked_jira_issue": "blocked_jira_issue_count",
        "release_blocker_issue": "release_blocker_issue_count",
        "unassigned_high_priority_issue": "unassigned_high_priority_issue_count",
        "due_soon_issue": "due_soon_issue_count",
        "critical_service_issue": "critical_service_issue_count",
    }

    _SEVERITY_TO_FEATURE_FIELD: Mapping[str, str] = {
        "critical": "critical_risk_count",
        "high": "high_risk_count",
        "medium": "medium_risk_count",
        "low": "low_risk_count",
    }

    def extract_features(
        self,
        request: ReleaseRiskFeatureExtractionRequest,
        *,
        run_id: str | None = None,
    ) -> ReleaseRiskFeatureVector:
        """Extract deterministic release-risk features.

        Args:
            request: GitHub, Jira, release summary, and Knowledge Agent outputs.
            run_id: Optional workflow run identifier used only for safe logs.

        Returns:
            Stable feature vector used by the next scoring layer.

        This method is intentionally synchronous because it performs only
        in-memory CPU work and does not call databases, external APIs, or LLMs.
        """
        started_at = time.perf_counter()

        github_signals = self._extract_github_signals(request.github)
        jira_signals = self._extract_jira_signals(request.jira)
        all_signals = [*github_signals, *jira_signals]

        category_counts = self._count_by_field(
            signals=all_signals,
            source_key="category",
            output_fields=self._CATEGORY_TO_FEATURE_FIELD.values(),
            value_to_output_field=self._CATEGORY_TO_FEATURE_FIELD,
        )
        severity_counts = self._count_by_field(
            signals=all_signals,
            source_key="severity",
            output_fields=self._SEVERITY_TO_FEATURE_FIELD.values(),
            value_to_output_field=self._SEVERITY_TO_FEATURE_FIELD,
        )

        rule_scores = [
            score
            for signal in all_signals
            if (score := self._normalize_rule_score(signal.get("score"))) is not None
        ]

        knowledge_status = self._normalize_text(request.knowledge_status)
        knowledge_result_count = len(request.knowledge_results)
        knowledge_failed = knowledge_status == "failed" or bool(request.knowledge_error)
        knowledge_no_results = (
            knowledge_status == "no_results"
            or (
                knowledge_status == "completed"
                and knowledge_result_count == 0
                and not knowledge_failed
            )
        )

        feature_vector = ReleaseRiskFeatureVector(
            total_risk_count=len(all_signals),
            github_risk_count=len(github_signals),
            jira_risk_count=len(jira_signals),
            critical_risk_count=severity_counts["critical_risk_count"],
            high_risk_count=severity_counts["high_risk_count"],
            medium_risk_count=severity_counts["medium_risk_count"],
            low_risk_count=severity_counts["low_risk_count"],
            ci_failure_count=category_counts["ci_failure_count"],
            ci_pending_count=category_counts["ci_pending_count"],
            review_blocked_count=category_counts["review_blocked_count"],
            review_missing_count=category_counts["review_missing_count"],
            stale_pr_count=category_counts["stale_pr_count"],
            large_changeset_count=category_counts["large_changeset_count"],
            draft_pull_request_count=category_counts["draft_pull_request_count"],
            missing_jira_link_count=category_counts["missing_jira_link_count"],
            critical_file_change_count=category_counts["critical_file_change_count"],
            open_critical_bug_count=category_counts["open_critical_bug_count"],
            blocked_jira_issue_count=category_counts["blocked_jira_issue_count"],
            release_blocker_issue_count=category_counts["release_blocker_issue_count"],
            unassigned_high_priority_issue_count=category_counts[
                "unassigned_high_priority_issue_count"
            ],
            due_soon_issue_count=category_counts["due_soon_issue_count"],
            critical_service_issue_count=category_counts[
                "critical_service_issue_count"
            ],
            knowledge_result_count=knowledge_result_count,
            knowledge_no_results=knowledge_no_results,
            knowledge_failed=knowledge_failed,
            github_degraded=self._is_degraded(request.github),
            jira_degraded=self._is_degraded(request.jira),
            max_rule_score=max(rule_scores, default=0.0),
            average_rule_score=(
                round(sum(rule_scores) / len(rule_scores), 4)
                if rule_scores
                else 0.0
            ),
        )

        logger.info(
            "release_risk_features_extracted",
            run_id=run_id,
            feature_version=feature_vector.feature_version,
            total_risk_count=feature_vector.total_risk_count,
            github_risk_count=feature_vector.github_risk_count,
            jira_risk_count=feature_vector.jira_risk_count,
            knowledge_result_count=feature_vector.knowledge_result_count,
            knowledge_failed=feature_vector.knowledge_failed,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )

        return feature_vector

    def extract_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        run_id: str | None = None,
    ) -> ReleaseRiskFeatureVector:
        """Extract features from the full release-risk API/workflow payload.

        Args:
            payload: Dictionary containing github, jira, release_summary, and
                Knowledge Agent fields.
            run_id: Optional workflow run identifier used only for safe logs.

        Returns:
            Stable feature vector for scoring.
        """
        request = ReleaseRiskFeatureExtractionRequest(
            github=self._optional_dict(payload.get("github")),
            jira=self._optional_dict(payload.get("jira")),
            release_summary=self._optional_dict(payload.get("release_summary")),
            knowledge_status=self._optional_string(payload.get("knowledge_status")),
            knowledge_results=self._list_of_dicts(payload.get("knowledge_results")),
            knowledge_error=self._optional_string(payload.get("knowledge_error")),
        )

        return self.extract_features(request, run_id=run_id)

    @classmethod
    def _extract_github_signals(
        cls,
        github_payload: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Extract GitHub risk signals from github.risk_results[*].signals."""
        github = cls._optional_dict(github_payload)
        risk_results = cls._list_of_dicts(github.get("risk_results"))

        signals: list[dict[str, Any]] = []
        for risk_result in risk_results:
            signals.extend(cls._list_of_dicts(risk_result.get("signals")))

        return signals

    @classmethod
    def _extract_jira_signals(
        cls,
        jira_payload: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Extract Jira risk signals without double-counting issue-level signals.

        The current Jira API response can expose top-level `signals` and each
        issue can also contain `signals`. Prefer top-level signals when present
        because it represents the normalized collection output.
        """
        jira = cls._optional_dict(jira_payload)
        top_level_signals = cls._list_of_dicts(jira.get("signals"))

        if top_level_signals:
            return top_level_signals

        issue_signals: list[dict[str, Any]] = []
        for issue in cls._list_of_dicts(jira.get("issues")):
            issue_signals.extend(cls._list_of_dicts(issue.get("signals")))

        return issue_signals

    @classmethod
    def _count_by_field(
        cls,
        *,
        signals: list[dict[str, Any]],
        source_key: str,
        output_fields: Iterable[str],
        value_to_output_field: Mapping[str, str],
    ) -> dict[str, int]:
        """Count normalized signal values into named feature fields."""
        counts = {str(field): 0 for field in output_fields}

        for signal in signals:
            normalized_value = cls._normalize_text(signal.get(source_key))
            output_field = value_to_output_field.get(normalized_value)

            if output_field is not None:
                counts[output_field] += 1

        return counts

    @staticmethod
    def _normalize_rule_score(value: object) -> float | None:
        """Return a bounded 0.0-1.0 rule score or None for invalid input."""
        if isinstance(value, bool):
            return None

        if not isinstance(value, int | float):
            return None

        return min(1.0, max(0.0, float(value)))

    @classmethod
    def _is_degraded(cls, payload: Mapping[str, Any] | None) -> bool:
        """Return whether a source collection payload has degraded status."""
        data = cls._optional_dict(payload)
        return cls._normalize_text(data.get("status")) == "degraded"

    @staticmethod
    def _normalize_text(value: object) -> str:
        """Normalize enum-like or string values into lowercase text."""
        if value is None:
            return ""

        enum_value = getattr(value, "value", None)
        raw_value = enum_value if enum_value is not None else value

        return str(raw_value).strip().lower()

    @staticmethod
    def _optional_string(value: object) -> str | None:
        """Convert a value into a non-empty optional string."""
        if value is None:
            return None

        stripped_value = str(value).strip()
        return stripped_value or None

    @staticmethod
    def _optional_dict(value: object) -> dict[str, Any]:
        """Convert dictionary-like/Pydantic objects into a plain dictionary."""
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        if hasattr(value, "model_dump"):
            dumped = value.model_dump(mode="python")
            if isinstance(dumped, dict):
                return dumped

        return {}

    @classmethod
    def _list_of_dicts(cls, value: object) -> list[dict[str, Any]]:
        """Return only dictionary-like items from a list-like value."""
        if not isinstance(value, list):
            return []

        return [
            item_dict
            for item in value
            if (item_dict := cls._optional_dict(item))
        ]
