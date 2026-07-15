"""Compose manager-friendly conversational AgentFlow responses."""

from __future__ import annotations

import logging

from app.schemas.agent_query import (
    AgentCitation,
    AgentQueryPlan,
    AgentQueryResponse,
    ResponseDepth,
)
from app.schemas.risk import (
    ReleaseRiskSummaryItemResponse,
    ReleaseRunRiskResponse,
)

logger = logging.getLogger(__name__)


class AgentResponseComposer:
    """Create deterministic conversational answers from trusted risk data."""

    def __init__(self, request_id: str) -> None:
        """Initialize the response composer.

        Args:
            request_id: Request identifier used for structured logging.
        """

        self._request_id = request_id

    def compose(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> AgentQueryResponse:
        """Compose a conversational response with evidence citations.

        Args:
            plan: Validated natural-language query plan.
            release_risk: Trusted release-risk workflow response.

        Returns:
            Manager-friendly response containing the answer and citations.
        """

        citations = self._build_citations(release_risk)
        answer = self._build_answer(
            plan=plan,
            release_risk=release_risk,
        )

        logger.info(
            "agent_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "response_depth": plan.response_depth.value,
                "citation_count": len(citations),
                "approval_required": (release_risk.approval_required is True),
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=citations,
            approval_required=release_risk.approval_required is True,
        )

    def compose_specific_risk(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        selected_risk: ReleaseRiskSummaryItemResponse,
    ) -> AgentQueryResponse:
        """Compose a focused explanation for one persisted release risk.

        Args:
            plan: Validated specific-risk query plan.
            release_risk: Trusted persisted release-risk snapshot.
            selected_risk: Risk selected from the persisted ranked risks.

        Returns:
            Focused response containing only the selected risk citation.
        """

        severity = selected_risk.severity.value.replace("_", " ")
        score_percentage = round(selected_risk.score * 100)

        answer = (
            f"{selected_risk.title} is a {severity} severity risk with a "
            f"{score_percentage}% item score. Reason: {selected_risk.reason}"
        )

        if selected_risk.evidence:
            evidence_items = [
                f"{key.replace('_', ' ')}: {value}"
                for key, value in sorted(selected_risk.evidence.items())
            ]
            answer += f" Evidence: {'; '.join(evidence_items)}."

        if release_risk.approval_required is True:
            answer += (
                " Human approval is required before any downstream "
                "release notification or Slack action."
            )

        citation = AgentCitation(
            source=selected_risk.source,
            source_type=selected_risk.source_type,
            source_id=selected_risk.source_id,
            title=selected_risk.title,
            source_url=selected_risk.source_url,
        )

        logger.info(
            "agent_specific_risk_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "source_type": selected_risk.source_type,
                "source_id": selected_risk.source_id,
                "citation_count": 1,
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[citation],
            approval_required=release_risk.approval_required is True,
        )

    def _build_answer(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> str:
        """Build the answer at the requested response depth."""

        summary = release_risk.release_summary
        severity = summary.overall_severity.value.replace("_", " ")
        action = summary.recommended_action.value.replace("_", " ")

        opening = f"The release risk is {severity}. Recommended action: {action}."

        if plan.response_depth is ResponseDepth.BRIEF:
            return opening

        metrics = (
            f" The analysis found {summary.total_signal_count} risk signals, "
            f"including {summary.high_risk_count} high-severity signals."
        )

        score_text = ""

        if release_risk.risk_score is not None:
            score_percentage = round(release_risk.risk_score.score * 100)
            score_text = f" The deterministic risk score is {score_percentage}%."

        top_risks = summary.top_risks
        risk_limit = 5 if plan.response_depth is ResponseDepth.DEEP else 3

        risk_lines = [
            (f"{index}. {risk.title}: {risk.reason} ({risk.severity.value} severity).")
            for index, risk in enumerate(
                top_risks[:risk_limit],
                start=1,
            )
        ]

        risks_text = ""

        if risk_lines:
            risks_text = " Top risks: " + " ".join(risk_lines)
        else:
            risks_text = " No individual GitHub or Jira risk item was ranked in the final summary."

        approval_text = ""

        if release_risk.approval_required is True:
            approval_text = (
                " Human approval is required before any downstream "
                "release notification or Slack action."
            )

        if plan.response_depth is ResponseDepth.DEEP:
            source_text = (
                f" GitHub status: {release_risk.github.status.value}. "
                f"Jira status: {release_risk.jira.status.value}. "
                f"Knowledge retrieval status: "
                f"{release_risk.knowledge_status or 'not available'}."
            )
        else:
            source_text = ""

        return opening + metrics + score_text + risks_text + source_text + approval_text

    @staticmethod
    def _build_citations(
        release_risk: ReleaseRunRiskResponse,
    ) -> list[AgentCitation]:
        """Build deduplicated citations from trusted workflow evidence."""

        citations: list[AgentCitation] = []
        seen: set[tuple[str, str]] = set()

        for risk in release_risk.release_summary.top_risks:
            key = (risk.source_type, risk.source_id)

            if key in seen:
                continue

            seen.add(key)
            citations.append(
                AgentCitation(
                    source=risk.source,
                    source_type=risk.source_type,
                    source_id=risk.source_id,
                    title=risk.title,
                    source_url=risk.source_url,
                )
            )

        for knowledge_result in release_risk.knowledge_results:
            source_id = str(
                knowledge_result.chunk_id
                or knowledge_result.document_id
                or knowledge_result.title
                or "knowledge-result"
            )
            source_type = knowledge_result.source_type or "engineering_document"
            key = (source_type, source_id)

            if key in seen:
                continue

            seen.add(key)
            citations.append(
                AgentCitation(
                    source="knowledge",
                    source_type=source_type,
                    source_id=source_id,
                    title=(knowledge_result.title or "Engineering knowledge result"),
                )
            )

        return citations
