"""Service-backed workflow nodes for AgentFlow AI release-risk orchestration.

These nodes adapt existing application services into LangGraph-compatible
workflow nodes.

Current scope:
- Call ReleaseRunService.collect_release_risks()
- Store existing service output in ReleaseRiskState
- Preserve graceful failure behavior when a release run is missing

Future scope:
- Split GitHub, Jira, RAG, ML, synthesis, HITL, and Slack into separate nodes
- Add OpenTelemetry spans per node
- Add retry boundaries around external API calls in service dependencies
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel

from app.workflows.release_risk_graph import (
    WorkflowStateInput,
    WorkflowStateUpdate,
)
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
)


class ReleaseRiskCollectionService(Protocol):
    """Service contract required by the release-risk collection node."""

    async def collect_release_risks(self, release_run_id: UUID) -> object | None:
        """Collect release risks for a release run."""


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