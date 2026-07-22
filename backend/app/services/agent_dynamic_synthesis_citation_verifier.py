"""Verify dynamic synthesis citations against trusted tool evidence."""

from __future__ import annotations

from app.schemas.agent_dynamic_synthesis import AgentDynamicAnswer
from app.schemas.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from app.schemas.agent_tool import AgentToolExecutionStatus


class AgentDynamicSynthesisCitationVerificationError(ValueError):
    """Raised when a dynamic answer violates synthesis grounding rules."""


class AgentDynamicSynthesisCitationVerifier:
    """Validate synthesized answers against executed AgentFlow evidence."""

    def verify(
        self,
        *,
        answer: AgentDynamicAnswer,
        execution_result: AgentExecutionResult,
    ) -> AgentDynamicAnswer:
        """Return the answer only when every grounding rule is satisfied.

        Complexity:
            Time: O(e + c + t), where e is evidence, c is citations, and
            t is executed tool results.
            Space: O(e + t) for trusted citations and degraded step IDs.
        """
        trusted_citations = {
            (evidence.source_type, evidence.source_id)
            for result in execution_result.tool_results
            for evidence in result.evidence
        }

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
