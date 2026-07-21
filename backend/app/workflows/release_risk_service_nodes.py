"""Service-backed workflow nodes for AgentFlow AI release-risk orchestration.

These nodes adapt existing application services into LangGraph-compatible
workflow nodes.

Current scope:
- Call ReleaseRunService.collect_release_risks()
- Optionally retrieve Knowledge Agent context from stored engineering documents
- Store existing service output in ReleaseRiskState
- Preserve graceful failure behavior when optional knowledge retrieval fails

Future scope:
- Split GitHub, Jira, RAG, ML, synthesis, HITL, and Slack into separate nodes
- Add OpenTelemetry spans per node
- Add retry boundaries around external API calls in service dependencies
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

import structlog
from pydantic import BaseModel

from app.integrations.anthropic_client import (
    AnthropicClientError,
    ClaudeSynthesisResult,
)
from app.observability.tracing import start_business_span
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalRequest,
)
from app.services.hitl_approval_decision_service import HITLApprovalDecisionService
from app.services.release_risk_response_mapper import (
    to_release_run_risk_response,
)
from app.services.release_risk_synthesis_citation_verifier import (
    ReleaseRiskSynthesisCitationVerifier,
)
from app.services.release_risk_synthesis_prompt import (
    ReleaseRiskSynthesisPromptBuilder,
)
from app.services.risk_feature_extraction_service import RiskFeatureExtractionService
from app.services.rule_based_risk_scoring_service import RuleBasedRiskScoringService
from app.workflows.release_risk_graph import (
    AsyncWorkflowNode,
    WorkflowStateInput,
    WorkflowStateUpdate,
)
from app.workflows.release_risk_state import (
    KnowledgeRetrievalStatus,
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
    RiskSynthesisStatus,
)

logger = structlog.get_logger(__name__)


class ReleaseRiskCollectionService(Protocol):
    """Service contract required by the release-risk collection node."""

    async def collect_release_risks(self, release_run_id: UUID) -> object | None:
        """Collect release risks for a release run."""


class KnowledgeRetrievalService(Protocol):
    """Service contract required by the Knowledge Agent retrieval node."""

    async def retrieve_relevant_chunks(
        self,
        retrieval_request: EngineeringDocumentRetrievalRequest,
        *,
        run_id: str | None = None,
    ) -> object:
        """Retrieve relevant engineering document chunks."""


class RiskSynthesisService(Protocol):
    """Service contract required by the Claude synthesis workflow node."""

    async def synthesize_release_risk(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeSynthesisResult:
        """Produce a validated structured release-risk synthesis."""


def _validate_state_input(state: WorkflowStateInput) -> ReleaseRiskState:
    """Convert raw LangGraph state into the validated workflow state model."""
    if isinstance(state, ReleaseRiskState):
        return state

    return ReleaseRiskState.model_validate(state)


def _serialize_service_result(result: object) -> dict[str, Any]:
    """Convert a service result into a plain dictionary.

    The existing service may return a Pydantic model or a plain dictionary.
    This helper keeps the workflow node independent from the exact service
    response implementation.
    """
    if isinstance(result, BaseModel):
        return result.model_dump(mode="python")

    if isinstance(result, dict):
        return result

    raise TypeError(
        "collect_release_risks() must return a Pydantic model or dictionary"
    )


def _extract_optional_mapping(
    payload: dict[str, Any],
    key: str,
) -> dict[str, Any] | None:
    """Extract an optional dictionary value from a service payload."""
    value = payload.get(key)

    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")

    raise TypeError(f"{key} must be a dictionary, Pydantic model, or None")


def _collect_text_values(value: object) -> list[str]:
    """Collect safe short text values from nested workflow payloads.

    This intentionally avoids logging or exposing raw document content. It only
    builds an internal retrieval query from existing risk titles, categories,
    descriptions, and summary metadata.
    """
    if value is None:
        return []

    if isinstance(value, str):
        stripped_value = value.strip()
        return [stripped_value] if stripped_value else []

    if isinstance(value, dict):
        collected: list[str] = []

        preferred_keys = {
            "title",
            "summary",
            "description",
            "category",
            "severity",
            "recommended_action",
            "overall_status",
            "risk_level",
        }

        for key, nested_value in value.items():
            if key in preferred_keys:
                collected.extend(_collect_text_values(nested_value))

            elif isinstance(nested_value, dict | list):
                collected.extend(_collect_text_values(nested_value))

        return collected

    if isinstance(value, list):
        list_values: list[str] = []

        for item in value:
            list_values.extend(_collect_text_values(item))

        return list_values

    return []


def _build_knowledge_query(state: ReleaseRiskState) -> str:
    """Build a deterministic Knowledge Agent retrieval query from workflow state."""
    query_parts: list[str] = [state.manager_query]

    query_parts.extend(
        _collect_text_values(
            {
                "github_summary": state.github_summary,
                "jira_summary": state.jira_summary,
                "release_summary": state.release_summary,
                "github": state.github,
                "jira": state.jira,
            }
        )
    )

    query = " ".join(query_parts)
    normalized_query = " ".join(query.split())

    return normalized_query[:1_000]


def _serialize_knowledge_result(result: object) -> dict[str, Any]:
    """Serialize a Knowledge retrieval result into a plain dictionary."""
    if isinstance(result, BaseModel):
        return result.model_dump(mode="python")

    if isinstance(result, dict):
        return result

    raise TypeError(
        "retrieve_relevant_chunks() must return a Pydantic model or dictionary"
    )


def create_collect_release_risks_node(
    service: ReleaseRiskCollectionService,
) -> Callable[[WorkflowStateInput], object]:
    """Create an async LangGraph node backed by ReleaseRunService.

    Args:
        service: Service exposing collect_release_risks(release_run_id).

    Returns:
        Async LangGraph-compatible node function.

    The returned node:
    - validates incoming workflow state
    - calls the existing service
    - stores release_run, github, jira, and summary outputs in state
    - returns a plain dictionary update for LangGraph
    """

    async def collect_release_risks_node(
        state: WorkflowStateInput,
    ) -> WorkflowStateUpdate:
        """Collect release risks through the existing service layer."""
        validated_state = _validate_state_input(state)

        running_state = validated_state.mark_running(
            ReleaseRiskWorkflowStage.BUILDING_RELEASE_SUMMARY
        )

        service_result = await service.collect_release_risks(
            running_state.release_run_id
        )

        if service_result is None:
            failed_state = running_state.add_error(
                source="release_run_service",
                message="Release run was not found.",
                recoverable=False,
                details={"release_run_id": str(running_state.release_run_id)},
            )

            return failed_state.model_dump(mode="python")

        payload = _serialize_service_result(service_result)

        updated_state = running_state.model_copy(
            update={
                "release_run": _extract_optional_mapping(payload, "release_run"),
                "github": _extract_optional_mapping(payload, "github"),
                "github_summary": _extract_optional_mapping(
                    payload,
                    "github_summary",
                ),
                "jira": _extract_optional_mapping(payload, "jira"),
                "jira_summary": _extract_optional_mapping(
                    payload,
                    "jira_summary",
                ),
                "release_summary": _extract_optional_mapping(
                    payload,
                    "release_summary",
                ),
            }
        ).add_completed_node("collect_release_risks")

        return updated_state.model_dump(mode="python")

    return collect_release_risks_node


def create_retrieve_knowledge_context_node(
    service: KnowledgeRetrievalService,
) -> Callable[[WorkflowStateInput], object]:
    """Create an async LangGraph node for Knowledge Agent retrieval.

    The returned node is intentionally recoverable. If document retrieval fails,
    the workflow records a safe error and continues with GitHub/Jira evidence.
    """

    async def retrieve_knowledge_context_node(
        state: WorkflowStateInput,
    ) -> WorkflowStateUpdate:
        """Retrieve internal engineering document context for release risks."""
        validated_state = _validate_state_input(state)

        with start_business_span(
            "knowledge.retrieve",
            {
                "release_run_id": str(validated_state.release_run_id),
                "run_id": validated_state.run_id,
                "query_present": bool(validated_state.manager_query),
            },
        ):
            running_state = validated_state.mark_running(
                ReleaseRiskWorkflowStage.RETRIEVING_KNOWLEDGE_CONTEXT
            )

            knowledge_query = _build_knowledge_query(running_state)

            try:
                retrieval_result = await service.retrieve_relevant_chunks(
                    EngineeringDocumentRetrievalRequest(
                        query=knowledge_query,
                        top_k=5,
                        document_limit=100,
                    ),
                    run_id=running_state.run_id,
                )
                payload = _serialize_knowledge_result(retrieval_result)
                results = payload.get("results", [])

                if not isinstance(results, list):
                    raise TypeError("knowledge retrieval results must be a list")

                knowledge_status = (
                    KnowledgeRetrievalStatus.COMPLETED
                    if results
                    else KnowledgeRetrievalStatus.NO_RESULTS
                )

                logger.info(
                    "knowledge_retrieval_node_completed",
                    run_id=running_state.run_id,
                    release_run_id=str(running_state.release_run_id),
                    knowledge_status=knowledge_status.value,
                    result_count=len(results),
                    query_length=len(knowledge_query),
                )

                updated_state = running_state.model_copy(
                    update={
                        "knowledge_query": knowledge_query,
                        "knowledge_results": results,
                        "knowledge_status": knowledge_status,
                        "knowledge_error": None,
                    }
                ).add_completed_node("retrieve_knowledge_context")

                return updated_state.model_dump(mode="python")

            except (TypeError, ValueError) as exc:
                logger.warning(
                    "knowledge_retrieval_node_failed",
                    run_id=running_state.run_id,
                    release_run_id=str(running_state.release_run_id),
                    error_type=exc.__class__.__name__,
                    query_length=len(knowledge_query),
                )

                failed_state = running_state.model_copy(
                    update={
                        "knowledge_query": knowledge_query,
                        "knowledge_results": [],
                        "knowledge_status": KnowledgeRetrievalStatus.FAILED,
                        "knowledge_error": "Knowledge retrieval failed.",
                    }
                ).add_error(
                    source="knowledge_retrieval",
                    message="Knowledge retrieval failed.",
                    recoverable=True,
                    details={"error_type": exc.__class__.__name__},
                )

                return failed_state.model_dump(mode="python")

    return retrieve_knowledge_context_node


def create_score_release_risk_node(
    feature_extraction_service: RiskFeatureExtractionService | None = None,
    risk_scoring_service: RuleBasedRiskScoringService | None = None,
) -> Callable[[WorkflowStateInput], object]:
    """Create a LangGraph node for deterministic release-risk scoring.

    The node converts current workflow state into a stable feature vector,
    scores the release using deterministic rules, stores both outputs in
    ReleaseRiskState, and logs only safe metadata.
    """

    feature_service = feature_extraction_service or RiskFeatureExtractionService()
    scoring_service = risk_scoring_service or RuleBasedRiskScoringService()

    def score_release_risk_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
        """Extract features and score release risk from current workflow state."""
        validated_state = _validate_state_input(state)
        running_state = validated_state.mark_running(
            ReleaseRiskWorkflowStage.SCORING_RELEASE_RISK
        )

        try:
            payload = running_state.model_dump(mode="python")

            risk_features = feature_service.extract_from_payload(
                payload,
                run_id=running_state.run_id,
            )
            risk_score = scoring_service.score_release(
                risk_features,
                run_id=running_state.run_id,
            )

            logger.info(
                "release_risk_scoring_node_completed",
                run_id=running_state.run_id,
                release_run_id=str(running_state.release_run_id),
                feature_version=risk_features.feature_version,
                scoring_version=risk_score.scoring_version,
                score=risk_score.score,
                risk_level=risk_score.risk_level.value,
                recommended_action=risk_score.recommended_action.value,
                total_risk_count=risk_features.total_risk_count,
                knowledge_failed=risk_features.knowledge_failed,
            )

            updated_state = running_state.model_copy(
                update={
                    "risk_features": risk_features.model_dump(mode="python"),
                    "risk_score": risk_score.model_dump(mode="python"),
                }
            ).add_completed_node("score_release_risk")

            return updated_state.model_dump(mode="python")

        except (TypeError, ValueError) as exc:
            logger.warning(
                "release_risk_scoring_node_failed",
                run_id=running_state.run_id,
                release_run_id=str(running_state.release_run_id),
                error_type=exc.__class__.__name__,
            )

            failed_state = running_state.add_error(
                source="release_risk_scoring",
                message="Release-risk scoring failed.",
                recoverable=False,
                details={"error_type": exc.__class__.__name__},
            )

            return failed_state.model_dump(mode="python")

    return score_release_risk_node


def create_synthesize_release_risk_node(
    synthesis_service: RiskSynthesisService,
    prompt_builder: ReleaseRiskSynthesisPromptBuilder | None = None,
    citation_verifier: ReleaseRiskSynthesisCitationVerifier | None = None,
) -> AsyncWorkflowNode:
    """Create an async LangGraph node for Claude risk synthesis.

    Claude receives only validated and bounded AgentFlow evidence. Failures are
    recoverable because deterministic scoring remains available as a fallback.
    """

    builder = prompt_builder or ReleaseRiskSynthesisPromptBuilder()
    verifier = citation_verifier or ReleaseRiskSynthesisCitationVerifier()

    async def synthesize_release_risk_node(
        state: WorkflowStateInput,
    ) -> WorkflowStateUpdate:
        """Generate and persist one structured Claude risk report."""
        validated_state = _validate_state_input(state)
        running_state = validated_state.mark_running(
            ReleaseRiskWorkflowStage.SYNTHESIZING_RELEASE_RISK
        )

        try:
            release_risk = to_release_run_risk_response(
                running_state.model_dump(mode="python")
            )
            prompt = builder.build(release_risk)

            with start_business_span(
                "llm.release_risk_synthesis",
                {
                    "release_run_id": str(running_state.release_run_id),
                    "run_id": running_state.run_id,
                    "prompt_version": prompt.prompt_version,
                    "risk_count": prompt.risk_count,
                    "knowledge_result_count": prompt.knowledge_result_count,
                    "degraded_source_count": prompt.degraded_source_count,
                },
            ) as span:
                result = await synthesis_service.synthesize_release_risk(
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    prompt_version=prompt.prompt_version,
                )
                verified_report = verifier.verify(
                    report=result.report,
                    release_risk=release_risk,
                )

                span.set_attribute("llm.model", result.model)
                span.set_attribute("llm.input_tokens", result.input_tokens)
                span.set_attribute("llm.output_tokens", result.output_tokens)
                span.set_attribute(
                    "llm.recommendation",
                    verified_report.recommendation.value,
                )

            logger.info(
                "release_risk_synthesis_node_completed",
                run_id=running_state.run_id,
                release_run_id=str(running_state.release_run_id),
                prompt_version=result.prompt_version,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                duration_ms=result.duration_ms,
                recommendation=verified_report.recommendation.value,
                risk_count=len(verified_report.risks),
            )

            updated_state = running_state.model_copy(
                update={
                    "synthesis_status": RiskSynthesisStatus.COMPLETED,
                    "synthesis_report": verified_report.model_dump(mode="python"),
                    "synthesis_prompt_version": result.prompt_version,
                    "synthesis_model": result.model,
                    "synthesis_input_tokens": result.input_tokens,
                    "synthesis_output_tokens": result.output_tokens,
                    "synthesis_duration_ms": result.duration_ms,
                    "synthesis_error": None,
                }
            ).add_completed_node("synthesize_release_risk")

            return updated_state.model_dump(mode="python")

        except (AnthropicClientError, TypeError, ValueError) as exc:
            logger.warning(
                "release_risk_synthesis_node_failed",
                run_id=running_state.run_id,
                release_run_id=str(running_state.release_run_id),
                error_type=type(exc).__name__,
            )

            failed_state = running_state.model_copy(
                update={
                    "synthesis_status": RiskSynthesisStatus.FAILED,
                    "synthesis_report": None,
                    "synthesis_error": "Claude risk synthesis failed.",
                }
            ).add_error(
                source="release_risk_synthesis",
                message=(
                    "Claude risk synthesis failed; deterministic risk "
                    "assessment remains available."
                ),
                recoverable=True,
                details={"error_type": type(exc).__name__},
            )

            return failed_state.model_dump(mode="python")

    return synthesize_release_risk_node


def create_determine_approval_requirement_node(
    approval_decision_service: HITLApprovalDecisionService | None = None,
) -> Callable[[WorkflowStateInput], object]:
    """Create a LangGraph node that determines HITL approval requirement."""

    decision_service = approval_decision_service or HITLApprovalDecisionService()

    def determine_approval_requirement_node(
        state: WorkflowStateInput,
    ) -> WorkflowStateUpdate:
        """Determine whether this release requires human approval."""
        validated_state = _validate_state_input(state)
        running_state = validated_state.mark_running(
            ReleaseRiskWorkflowStage.DETERMINING_APPROVAL_REQUIREMENT
        )

        with start_business_span(
            "approval.decision",
            {
                "release_run_id": str(running_state.release_run_id),
                "run_id": running_state.run_id,
                "release_summary_present": running_state.release_summary is not None,
                "risk_score_present": running_state.risk_score is not None,
            },
        ) as span:
            decision = decision_service.determine_approval(
                running_state.risk_score,
                run_id=running_state.run_id,
            )
            span.set_attribute("approval.required", decision.approval_required)
            span.set_attribute("approval.policy_version", decision.approval_policy_version)
            span.set_attribute("approval.reason_present", decision.approval_reason is not None)

            logger.info(
                "approval_requirement_node_completed",
                run_id=running_state.run_id,
                release_run_id=str(running_state.release_run_id),
                approval_policy_version=decision.approval_policy_version,
                approval_required=decision.approval_required,
                approval_reason_present=decision.approval_reason is not None,
            )

            updated_state = running_state.model_copy(
                update={
                    "approval_required": decision.approval_required,
                    "approval_reason": decision.approval_reason,
                    "approval_policy_version": decision.approval_policy_version,
                }
            ).add_completed_node("determine_approval_requirement")

            return updated_state.model_dump(mode="python")

    return determine_approval_requirement_node
