"""Compose persisted historical release-risk responses."""

from __future__ import annotations

import logging
from typing import Protocol

from app.schemas.agent_query import AgentCitation, AgentQueryPlan, AgentQueryResponse
from app.schemas.risk import ReleaseRunRiskResponse

logger = logging.getLogger(__name__)


class SimilarReleaseMatchProtocol(Protocol):
    """Fields required to compose a similar-release response."""

    release_risk: ReleaseRunRiskResponse
    similarity_score: float


class AgentHistoricalResponseComposerMixin:
    """Compose deterministic historical release-risk responses."""

    _request_id: str

    @staticmethod
    def _build_citations(
        release_risk: ReleaseRunRiskResponse,
    ) -> list[AgentCitation]:
        """Build citations supplied by the concrete response composer."""
        raise NotImplementedError

    def compose_historical_risks(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        historical_release_risks: list[ReleaseRunRiskResponse],
    ) -> AgentQueryResponse:
        """Compose persisted risk history from previous release runs.

        Args:
            plan: Validated historical-risk query plan.
            release_risk: Trusted current release-risk snapshot.
            historical_release_risks: Trusted previous release snapshots.

        Returns:
            Historical summary with citations from previous persisted risks.
        """
        historical_count = len(historical_release_risks)
        release_label = "release" if historical_count == 1 else "releases"

        if not historical_release_risks:
            answer = "No previous releases with persisted risk history were found."
            citations: list[AgentCitation] = []
        else:
            history_lines: list[str] = []
            citations = []

            for index, historical_risk in enumerate(
                historical_release_risks,
                start=1,
            ):
                severity = (
                    historical_risk.release_summary.overall_severity.value
                    .replace("_", " ")
                )
                risk_score_text = (
                    f"{round(historical_risk.risk_score.score * 100)}% risk score"
                    if historical_risk.risk_score is not None
                    else "no persisted risk score"
                )
                top_risk_titles = [
                    risk.title
                    for risk in historical_risk.release_summary.top_risks[:3]
                ]
                top_risks_text = (
                    ", ".join(top_risk_titles)
                    if top_risk_titles
                    else "no ranked top risks"
                )

                history_lines.append(
                    f"{index}. {historical_risk.release_run.run_id}: "
                    f"{severity} severity, {risk_score_text}. "
                    f"Top risks: {top_risks_text}."
                )
                for citation in self._build_citations(historical_risk):
                    citations.append(
                        citation.model_copy(
                            update={
                                "title": (
                                    f"[{historical_risk.release_run.run_id}] "
                                    f"{citation.title}"
                                ),
                            }
                        )
                    )

            answer = (
                f"Found {historical_count} previous {release_label} "
                "with persisted risk history. "
                + " ".join(history_lines)
            )

        logger.info(
            "agent_historical_risk_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "historical_release_count": historical_count,
                "citation_count": len(citations),
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=citations,
            approval_required=release_risk.approval_required is True,
        )

    def compose_previous_release_comparison(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        previous_release_risk: ReleaseRunRiskResponse | None,
    ) -> AgentQueryResponse:
        """Compare the current persisted risk snapshot with the previous release.

        Args:
            plan: Validated previous-release comparison plan.
            release_risk: Trusted current release-risk snapshot.
            previous_release_risk: Immediately previous persisted release snapshot.

        Returns:
            Deterministic comparison with release-qualified citations.
        """
        if previous_release_risk is None:
            answer = (
                "No previous release with persisted risk history was found "
                "for comparison."
            )
            citations: list[AgentCitation] = []
        else:
            current_severity = (
                release_risk.release_summary.overall_severity.value
            )
            previous_severity = (
                previous_release_risk.release_summary.overall_severity.value
            )
            severity_rank = {
                "low": 1,
                "medium": 2,
                "high": 3,
                "critical": 4,
            }

            if severity_rank[current_severity] > severity_rank[previous_severity]:
                severity_change = (
                    f"severity increased from {previous_severity} "
                    f"to {current_severity}"
                )
            elif severity_rank[current_severity] < severity_rank[previous_severity]:
                severity_change = (
                    f"severity decreased from {previous_severity} "
                    f"to {current_severity}"
                )
            else:
                severity_change = f"severity remained {current_severity}"

            current_score = (
                release_risk.risk_score.score
                if release_risk.risk_score is not None
                else None
            )
            previous_score = (
                previous_release_risk.risk_score.score
                if previous_release_risk.risk_score is not None
                else None
            )

            if current_score is not None and previous_score is not None:
                score_delta = round((current_score - previous_score) * 100)

                if score_delta > 0:
                    score_change = (
                        f"risk score increased by {score_delta} "
                        "percentage points"
                    )
                elif score_delta < 0:
                    score_change = (
                        f"risk score decreased by {abs(score_delta)} "
                        "percentage points"
                    )
                else:
                    score_change = "risk score did not change"
            else:
                score_change = "risk score comparison is unavailable"

            current_signal_count = (
                release_risk.release_summary.total_signal_count
            )
            previous_signal_count = (
                previous_release_risk.release_summary.total_signal_count
            )

            if current_signal_count > previous_signal_count:
                signal_change = (
                    f"signal count increased from {previous_signal_count} "
                    f"to {current_signal_count}"
                )
            elif current_signal_count < previous_signal_count:
                signal_change = (
                    f"signal count decreased from {previous_signal_count} "
                    f"to {current_signal_count}"
                )
            else:
                signal_change = (
                    f"signal count remained {current_signal_count}"
                )

            previous_titles = {
                risk.title
                for risk in previous_release_risk.release_summary.top_risks
            }
            new_top_risk_titles = [
                risk.title
                for risk in release_risk.release_summary.top_risks
                if risk.title not in previous_titles
            ]
            new_risks_text = (
                ", ".join(new_top_risk_titles[:3])
                if new_top_risk_titles
                else "no newly ranked top risks"
            )

            answer = (
                f"Compared with "
                f"{previous_release_risk.release_run.run_id}, "
                f"{severity_change}; {score_change}; and {signal_change}. "
                f"New top risks: {new_risks_text}."
            )

            citations = []

            for compared_risk in (
                previous_release_risk,
                release_risk,
            ):
                for citation in self._build_citations(compared_risk):
                    citations.append(
                        citation.model_copy(
                            update={
                                "title": (
                                    f"[{compared_risk.release_run.run_id}] "
                                    f"{citation.title}"
                                ),
                            }
                        )
                    )

        logger.info(
            "agent_previous_release_comparison_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "previous_release_run_id": (
                    str(previous_release_risk.release_run.id)
                    if previous_release_risk is not None
                    else None
                ),
                "intent": plan.intent.value,
                "previous_release_found": previous_release_risk is not None,
                "citation_count": len(citations),
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=citations,
            approval_required=release_risk.approval_required is True,
        )

    def compose_similar_release(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        similar_release_match: SimilarReleaseMatchProtocol | None,
    ) -> AgentQueryResponse:
        """Compose the closest persisted historical release match.

        Args:
            plan: Validated similar-release query plan.
            release_risk: Trusted current release-risk snapshot.
            similar_release_match: Highest-ranked historical match, if present.

        Returns:
            Similar-release response with historical evidence citations.
        """
        if similar_release_match is None:
            answer = (
                "No previous releases with persisted risk history were found "
                "for similarity matching."
            )
            citations: list[AgentCitation] = []
            matched_release_run_id = None
            similarity_score = None
        else:
            matched_release = similar_release_match.release_risk
            similarity_percentage = round(
                similar_release_match.similarity_score * 100
            )
            severity = (
                matched_release.release_summary.overall_severity.value
                .replace("_", " ")
            )
            top_risk_titles = [
                risk.title
                for risk in matched_release.release_summary.top_risks[:3]
            ]
            top_risks_text = (
                ", ".join(top_risk_titles)
                if top_risk_titles
                else "no ranked top risks"
            )

            answer = (
                f"The most similar persisted release was "
                f"{matched_release.release_run.run_id} with "
                f"{similarity_percentage}% similarity. "
                f"It had {severity} severity. "
                f"Top risks: {top_risks_text}."
            )

            citations = [
                citation.model_copy(
                    update={
                        "title": (
                            f"[{matched_release.release_run.run_id}] "
                            f"{citation.title}"
                        )
                    }
                )
                for citation in self._build_citations(matched_release)
            ]
            matched_release_run_id = str(
                matched_release.release_run.id
            )
            similarity_score = similar_release_match.similarity_score

        logger.info(
            "agent_similar_release_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "matched_release_run_id": matched_release_run_id,
                "intent": plan.intent.value,
                "similar_release_found": similar_release_match is not None,
                "similarity_score": similarity_score,
                "citation_count": len(citations),
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=citations,
            approval_required=release_risk.approval_required is True,
        )
