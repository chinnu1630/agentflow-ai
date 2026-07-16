"""Find the most similar release from trusted persisted risk snapshots."""

from __future__ import annotations

import logging
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.risk import ReleaseRunRiskResponse

logger = logging.getLogger(__name__)


class SimilarReleaseMatch(BaseModel):
    """Validated result returned by similar-release matching."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    release_risk: ReleaseRunRiskResponse
    similarity_score: float = Field(ge=0.0, le=1.0)


class AgentSimilarReleaseMatcher:
    """Rank historical releases using deterministic risk-feature distance."""

    _FEATURE_NAMES: Final[tuple[str, ...]] = (
        "total_risk_count",
        "github_risk_count",
        "jira_risk_count",
        "critical_risk_count",
        "high_risk_count",
        "ci_failure_count",
        "blocked_jira_issue_count",
        "release_blocker_issue_count",
        "max_rule_score",
        "average_rule_score",
    )

    def __init__(self, request_id: str) -> None:
        """Initialize the matcher.

        Args:
            request_id: Request identifier included in structured logs.
        """
        self._request_id = request_id

    def match(
        self,
        *,
        current_release_risk: ReleaseRunRiskResponse,
        historical_release_risks: list[ReleaseRunRiskResponse],
    ) -> SimilarReleaseMatch | None:
        """Return the historical release with the closest feature vector.

        Similarity uses normalized Manhattan distance across deterministic
        release-risk features. A score of 1.0 means the selected features are
        identical, while 0.0 represents maximum normalized distance.

        Args:
            current_release_risk: Trusted current persisted snapshot.
            historical_release_risks: Trusted previous persisted snapshots.

        Returns:
            Highest-ranked historical release, or ``None`` when history is empty.
        """
        if not historical_release_risks:
            logger.info(
                "agent_similar_release_not_found",
                extra={
                    "run_id": self._request_id,
                    "release_run_id": str(
                        current_release_risk.release_run.id
                    ),
                    "historical_release_count": 0,
                },
            )
            return None

        best_release: ReleaseRunRiskResponse | None = None
        best_score = -1.0

        for historical_release in historical_release_risks:
            similarity_score = self._calculate_similarity(
                current_release_risk,
                historical_release,
            )

            if similarity_score > best_score:
                best_score = similarity_score
                best_release = historical_release

        if best_release is None:
            return None

        logger.info(
            "agent_similar_release_matched",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(current_release_risk.release_run.id),
                "matched_release_run_id": str(best_release.release_run.id),
                "historical_release_count": len(
                    historical_release_risks
                ),
                "similarity_score": round(best_score, 4),
            },
        )

        return SimilarReleaseMatch(
            release_risk=best_release,
            similarity_score=best_score,
        )

    def _calculate_similarity(
        self,
        current: ReleaseRunRiskResponse,
        historical: ReleaseRunRiskResponse,
    ) -> float:
        """Calculate normalized similarity between two release snapshots."""
        if current.risk_features is None or historical.risk_features is None:
            return self._calculate_summary_similarity(current, historical)

        normalized_distances: list[float] = []

        for feature_name in self._FEATURE_NAMES:
            current_value = float(
                getattr(current.risk_features, feature_name)
            )
            historical_value = float(
                getattr(historical.risk_features, feature_name)
            )
            denominator = max(
                abs(current_value),
                abs(historical_value),
                1.0,
            )
            normalized_distances.append(
                abs(current_value - historical_value) / denominator
            )

        average_distance = sum(normalized_distances) / len(
            normalized_distances
        )
        return max(0.0, min(1.0, 1.0 - average_distance))

    @staticmethod
    def _calculate_summary_similarity(
        current: ReleaseRunRiskResponse,
        historical: ReleaseRunRiskResponse,
    ) -> float:
        """Fallback similarity using persisted summary attributes."""
        score = 0.0

        if (
            current.release_summary.overall_severity
            == historical.release_summary.overall_severity
        ):
            score += 0.5

        current_sources = {
            risk.source_type
            for risk in current.release_summary.top_risks
        }
        historical_sources = {
            risk.source_type
            for risk in historical.release_summary.top_risks
        }

        if current_sources or historical_sources:
            union = current_sources | historical_sources
            score += 0.5 * (
                len(current_sources & historical_sources) / len(union)
            )

        return max(0.0, min(1.0, score))
