"""Coordinate evidence-grounded synthesis for dynamic agent results."""

from __future__ import annotations

from typing import Protocol

import structlog

from app.integrations.anthropic_dynamic_synthesis_client import (
    ClaudeDynamicSynthesisResult,
)
from app.observability.tracing import (
    record_business_span_failure,
    set_safe_span_attributes,
    start_business_span,
)
from app.schemas.agent_execution_result import AgentExecutionResult
from app.schemas.agent_query import AgentQueryPlan, AgentQueryRequest
from app.services.agent_dynamic_synthesis_citation_verifier import (
    AgentDynamicSynthesisCitationVerificationError,
    AgentDynamicSynthesisCitationVerifier,
)
from app.services.agent_dynamic_synthesis_prompt import (
    AgentDynamicSynthesisPromptBuilder,
)

logger = structlog.get_logger(__name__)


class DynamicSynthesisClientProtocol(Protocol):
    """Claude capability required for dynamic answer synthesis."""

    async def synthesize_dynamic_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeDynamicSynthesisResult:
        """Produce one validated structured dynamic answer."""

        ...


class AgentDynamicSynthesisService:
    """Build, execute, and verify dynamic answer synthesis."""

    def __init__(
        self,
        *,
        client: DynamicSynthesisClientProtocol,
        request_id: str,
        prompt_builder: AgentDynamicSynthesisPromptBuilder | None = None,
        citation_verifier: AgentDynamicSynthesisCitationVerifier | None = None,
    ) -> None:
        """Initialize the dynamic synthesis service."""
        self._client = client
        self._request_id = request_id
        self._prompt_builder = (
            prompt_builder or AgentDynamicSynthesisPromptBuilder()
        )
        self._citation_verifier = (
            citation_verifier or AgentDynamicSynthesisCitationVerifier()
        )

    async def synthesize(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
        execution_result: AgentExecutionResult,
    ) -> ClaudeDynamicSynthesisResult:
        """Generate and verify one evidence-grounded manager answer."""
        prompt = self._prompt_builder.build(
            request=request,
            query_plan=query_plan,
            execution_result=execution_result,
        )

        with start_business_span(
            "agent.dynamic_synthesis",
            {
                "run_id": self._request_id,
                "intent": query_plan.intent.value,
                "execution_id": str(execution_result.execution_id),
                "execution_status": execution_result.status.value,
                "step_count": len(execution_result.tool_results),
                "prompt_version": prompt.prompt_version,
            },
        ) as span:
            try:
                result = await self._client.synthesize_dynamic_answer(
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    prompt_version=prompt.prompt_version,
                )
            except Exception as exc:
                record_business_span_failure(
                    span,
                    failure_stage="dynamic_synthesis",
                    exception=exc,
                    execution_status=execution_result.status.value,
                )
                raise

            set_safe_span_attributes(
                span,
                {
                    "model_name": result.model,
                    "input_token_count": result.input_tokens,
                    "output_token_count": result.output_tokens,
                },
            )

            try:
                verified_answer = self._citation_verifier.verify(
                    answer=result.answer,
                    execution_result=execution_result,
                )
            except AgentDynamicSynthesisCitationVerificationError as exc:
                record_business_span_failure(
                    span,
                    failure_stage="grounding_verification",
                    exception=exc,
                    execution_status=execution_result.status.value,
                )
                raise

            set_safe_span_attributes(
                span,
                {
                    "citation_count": len(verified_answer.citations),
                    "requires_human_review": (
                        verified_answer.requires_human_review
                    ),
                },
            )

            verified_result = result.model_copy(
                update={"answer": verified_answer}
            )

        logger.info(
            "agent_dynamic_synthesis_completed",
            run_id=self._request_id,
            intent=query_plan.intent.value,
            execution_id=str(execution_result.execution_id),
            execution_status=execution_result.status.value,
            prompt_version=verified_result.prompt_version,
            model=verified_result.model,
            input_tokens=verified_result.input_tokens,
            output_tokens=verified_result.output_tokens,
            citation_count=len(verified_answer.citations),
            requires_human_review=verified_answer.requires_human_review,
        )

        return verified_result
