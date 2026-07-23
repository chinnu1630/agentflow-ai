"""Verify dynamic synthesis citations against trusted tool evidence."""

from __future__ import annotations

from app.schemas.agent_dynamic_synthesis import AgentDynamicAnswer
from app.schemas.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from app.schemas.agent_tool import AgentToolExecutionStatus
from app.services.agent_dynamic_synthesis_evidence_index import (
    AgentDynamicSynthesisEvidenceIndex,
)


class AgentDynamicSynthesisCitationVerificationError(ValueError):
    """Raised when a dynamic answer violates synthesis grounding rules."""


class AgentDynamicSynthesisCitationVerifier:
    """Validate synthesized answers against executed AgentFlow evidence."""

    def __init__(
        self,
        *,
        evidence_index: AgentDynamicSynthesisEvidenceIndex | None = None,
    ) -> None:
        """Initialize the verifier with the trusted evidence index."""
        self._evidence_index = (
            evidence_index or AgentDynamicSynthesisEvidenceIndex()
        )

    def verify(
        self,
        *,
        answer: AgentDynamicAnswer,
        execution_result: AgentExecutionResult,
    ) -> AgentDynamicAnswer:
        """Return the answer only when every grounding rule is satisfied.

        Complexity:
            Time: O(e + c + t + j), where e is evidence, c is citations,
            t is tool results, and j is traversed bounded JSON output.
            Space: O(e + t + j) for trusted evidence and degraded step IDs.
        """
        trusted_evidence = self._evidence_index.build(execution_result)
        trusted_citations = set(trusted_evidence)

        if trusted_citations and not answer.citations:
            raise AgentDynamicSynthesisCitationVerificationError(
                "Evidence-backed dynamic synthesis must include at least "
                "one verified citation."
            )

        for citation in answer.citations:
            citation_key = (
                citation.source_type,
                citation.source_id,
            )

            if citation_key not in trusted_citations:
                raise AgentDynamicSynthesisCitationVerificationError(
                    "Dynamic synthesis contains an unverified citation."
                )

        if (
            execution_result.status
            in {
                AgentExecutionStatus.PARTIAL,
                AgentExecutionStatus.FAILED,
            }
            and not answer.requires_human_review
        ):
            raise AgentDynamicSynthesisCitationVerificationError(
                "Partial or failed dynamic execution must require human review."
            )

        expected_degraded_steps = {
            result.step_id
            for result in execution_result.tool_results
            if result.status is not AgentToolExecutionStatus.SUCCESS
        }
        reported_degraded_steps = set(answer.degraded_steps)

        if reported_degraded_steps != expected_degraded_steps:
            raise AgentDynamicSynthesisCitationVerificationError(
                "Dynamic synthesis degraded steps must exactly match "
                "degraded execution steps."
            )

        return answer
