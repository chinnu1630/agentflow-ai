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

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

_OPERATIONAL_KNOWLEDGE_PATTERNS = (
    re.compile(
        r"\b(recover|recovery|rollback|restore|remediation|mitigation|triage)\b"
    ),
    re.compile(r"\b(what|which|how)\b.*\b(steps?|procedure|runbook|playbook)\b"),
)

_OPERATIONAL_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "api",
        "application",
        "are",
        "document",
        "documented",
        "documents",
        "does",
        "engineering",
        "for",
        "from",
        "how",
        "mitigation",
        "payment",
        "playbook",
        "procedure",
        "recovery",
        "recover",
        "remediation",
        "runbook",
        "say",
        "says",
        "service",
        "step",
        "steps",
        "system",
        "the",
        "this",
        "to",
        "triage",
        "what",
        "which",
        "with",
    }
)


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
                "selected_result_count": len(selected_results),
                "omitted_result_count": max(
                    0,
                    len(retrieval.results) - len(selected_results),
                ),
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

    @classmethod
    def _select_results(
        cls,
        *,
        plan: AgentQueryPlan,
        retrieval: EngineeringDocumentRetrievalResponse,
    ) -> list[EngineeringDocumentRetrievalResult]:
        """Select grounded chunks while filtering unrelated operational evidence."""
        result_limit = {
            ResponseDepth.BRIEF: 1,
            ResponseDepth.STANDARD: 3,
            ResponseDepth.DEEP: 5,
            ResponseDepth.ACTION_CONFIRMATION: 1,
        }[plan.response_depth]

        limited_results = retrieval.results[:result_limit]

        if len(limited_results) <= 1:
            return limited_results

        normalized_query = " ".join(retrieval.query.casefold().split())

        if not any(
            pattern.search(normalized_query) is not None
            for pattern in _OPERATIONAL_KNOWLEDGE_PATTERNS
        ):
            return limited_results

        anchor_terms = cls._extract_operational_anchor_terms(
            query=retrieval.query,
            top_result=limited_results[0],
        )

        if not anchor_terms:
            return limited_results

        selected_results = [limited_results[0]]

        for result in limited_results[1:]:
            content_terms = cls._tokenize_operational_text(result.content)

            if anchor_terms & content_terms:
                selected_results.append(result)

        return selected_results

    @classmethod
    def _extract_operational_anchor_terms(
        cls,
        *,
        query: str,
        top_result: EngineeringDocumentRetrievalResult,
    ) -> set[str]:
        """Return specific query terms grounded in the highest-ranked chunk."""
        query_terms = {
            cls._normalize_operational_term(token)
            for token in _TOKEN_PATTERN.findall(query.casefold())
            if token.casefold() not in _OPERATIONAL_QUERY_STOPWORDS
        }
        query_terms.discard("")

        if not query_terms:
            return set()

        top_result_terms = cls._tokenize_operational_text(top_result.content)
        return query_terms & top_result_terms

    @classmethod
    def _tokenize_operational_text(cls, value: str) -> set[str]:
        """Return normalized terms used for bounded operational matching."""
        return {
            normalized
            for token in _TOKEN_PATTERN.findall(value.casefold())
            if (normalized := cls._normalize_operational_term(token))
        }

    @staticmethod
    def _normalize_operational_term(token: str) -> str:
        """Normalize simple plural terms without applying broad stemming."""
        normalized_token = token.casefold()

        if (
            len(normalized_token) > 4
            and normalized_token.endswith("s")
            and normalized_token not in {"redis", "status"}
        ):
            return normalized_token[:-1]

        return normalized_token

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
