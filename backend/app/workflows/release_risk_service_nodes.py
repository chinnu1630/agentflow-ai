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

from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalRequest,
)
from app.workflows.release_risk_graph import (
    WorkflowStateInput,
    WorkflowStateUpdate,
)
from app.workflows.release_risk_state import (
    KnowledgeRetrievalStatus,
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
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
        collected: list[str] = []

        for item in value:
            collected.extend(_collect_text_values(item))

        return collected

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
