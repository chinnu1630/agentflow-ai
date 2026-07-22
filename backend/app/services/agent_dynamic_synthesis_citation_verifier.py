"""Verify dynamic synthesis citations against trusted tool evidence."""

from __future__ import annotations

from app.schemas.agent_dynamic_synthesis import AgentDynamicAnswer
from app.schemas.agent_execution_result import AgentExecutionResult


class AgentDynamicSynthesisCitationVerificationError(ValueError):
    """Raised when a dynamic answer cites untrusted evidence."""


class AgentDynamicSynthesisCitationVerifier:
    """Validate Claude citations against executed AgentFlow tool evidence."""

    def verify(
        self,
        *,
        answer: AgentDynamicAnswer,
        execution_result: AgentExecutionResult,
    ) -> AgentDynamicAnswer:
        """Return the answer only when every citation is trusted.

        Complexity:
            Time: O(e + c), where e is tool evidence and c is citations.
            Space: O(e) for the trusted citation allowlist.
        """
        trusted_citations = {
            (evidence.source_type, evidence.source_id)
            for result in execution_result.tool_results
            for evidence in result.evidence
        }

        for citation in answer.citations:
            citation_key = (
                citation.source_type,
                citation.source_id,
            )

            if citation_key not in trusted_citations:
                raise AgentDynamicSynthesisCitationVerificationError(
                    "Dynamic synthesis contains an unverified citation."
                )

        return answer
