"""Compose grounded answers from retrieved engineering documents."""

from __future__ import annotations

import logging
import re

from app.schemas.agent_query import (
    AgentCitation,
    AgentQueryPlan,
    AgentQueryResponse,
    ResponseDepth,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalResponse,
    EngineeringDocumentRetrievalResult,
)

logger = logging.getLogger(__name__)


class AgentKnowledgeResponseComposerMixin:
    """Compose deterministic answers using trusted document chunks."""

    _request_id: str

    def compose_knowledge_document(
        self,
        *,
        plan: AgentQueryPlan,
        retrieval: EngineeringDocumentRetrievalResponse,
    ) -> AgentQueryResponse:
        """Return a grounded engineering-document answer with citations."""
        selected_results = self._select_results(
            plan=plan,
            retrieval=retrieval,
        )
        citations = self._build_knowledge_citations(selected_results)
        answer = self._build_knowledge_answer(selected_results)

        logger.info(
            "agent_knowledge_response_composed",
            extra={
                "run_id": self._request_id,
                "intent": plan.intent.value,
                "response_depth": plan.response_depth.value,
                "retrieval_result_count": len(retrieval.results),
                "citation_count": len(citations),
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=None,
            citations=citations,
            approval_required=False,
        )

    @staticmethod
    def _select_results(
        *,
        plan: AgentQueryPlan,
        retrieval: EngineeringDocumentRetrievalResponse,
    ) -> list[EngineeringDocumentRetrievalResult]:
        """Select only the chunks used to construct the final answer."""
        result_limit = {
            ResponseDepth.BRIEF: 1,
            ResponseDepth.STANDARD: 3,
            ResponseDepth.DEEP: 5,
            ResponseDepth.ACTION_CONFIRMATION: 1,
        }[plan.response_depth]

        return retrieval.results[:result_limit]

    def _build_knowledge_answer(
        self,
        selected_results: list[EngineeringDocumentRetrievalResult],
    ) -> str:
        """Build an extractive answer from selected trusted chunks."""
        if not selected_results:
            return (
                "No relevant engineering-document evidence was found for this "
                "question."
            )

        evidence_lines = [
            (
                f"{index}. {result.title} "
                f"(chunk {result.chunk_index}): "
                f"{self._build_excerpt(result)}"
            )
            for index, result in enumerate(selected_results, start=1)
        ]

        return (
            "Based on the retrieved engineering documents:\n\n"
            + "\n\n".join(evidence_lines)
        )

    @staticmethod
    def _build_excerpt(
        result: EngineeringDocumentRetrievalResult,
        *,
        maximum_characters: int = 600,
    ) -> str:
        """Normalize and safely truncate one retrieved chunk."""
        normalized_content = re.sub(r"\s+", " ", result.content).strip()

        if len(normalized_content) <= maximum_characters:
            return normalized_content

        truncated_content = normalized_content[:maximum_characters]
        final_space = truncated_content.rfind(" ")

        if final_space > 0:
            truncated_content = truncated_content[:final_space]

        return f"{truncated_content}..."

    @staticmethod
    def _build_knowledge_citations(
        selected_results: list[EngineeringDocumentRetrievalResult],
    ) -> list[AgentCitation]:
        """Create one trusted citation for each chunk used in the answer."""
        return [
            AgentCitation(
                source="knowledge",
                source_type=result.source_type.value,
                source_id=str(result.chunk_id),
                title=result.title,
                source_url=result.source_uri,
            )
            for result in selected_results
        ]
