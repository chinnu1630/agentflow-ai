"""Service-backed LangGraph workflow for AgentFlow AI release-risk collection.

This graph connects LangGraph orchestration to the existing ReleaseRunService.

Current scope:
- Start workflow
- Call ReleaseRunService.collect_release_risks()
- Route to completion when collection succeeds
- Stop as failed when release run is missing

Future scope:
- Split GitHub/Jira collection into separate parallel nodes
- Add hybrid RAG node
- Add XGBoost risk scoring node
- Add Claude synthesis node
- Add human approval gate
- Add approved Slack delivery node
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.workflows.release_risk_graph import (
    WorkflowStateInput,
    WorkflowStateUpdate,
)
from app.workflows.release_risk_nodes import (
    complete_release_risk_workflow,
    start_release_risk_workflow,
)
from app.workflows.release_risk_service_nodes import (
    ReleaseRiskCollectionService,
    create_collect_release_risks_node,
)
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStatus,
)


_ROUTE_COMPLETE = "complete"
_ROUTE_END = "end"


def _validate_state_input(state: WorkflowStateInput) -> ReleaseRiskState:
    """Convert raw LangGraph state into the validated workflow state model."""
    if isinstance(state, ReleaseRiskState):
        return state

    return ReleaseRiskState.model_validate(state)


def _dump_state_update(state: ReleaseRiskState) -> WorkflowStateUpdate:
    """Convert validated workflow state into a LangGraph-compatible update."""
    return state.model_dump(mode="python")


def _start_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
    """Run the workflow start node and return a LangGraph state update."""
    validated_state = _validate_state_input(state)
    updated_state = start_release_risk_workflow(validated_state)

    return _dump_state_update(updated_state)


def _complete_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
    """Run the workflow completion node and return a LangGraph state update."""
    validated_state = _validate_state_input(state)
    updated_state = complete_release_risk_workflow(validated_state)

    return _dump_state_update(updated_state)


def _route_after_collection(state: WorkflowStateInput) -> str:
    """Route after release-risk collection based on workflow status.

    If collection failed, the graph stops immediately.
    If collection succeeded or remained running, the graph completes normally.
    """
    validated_state = _validate_state_input(state)

    if validated_state.status == ReleaseRiskWorkflowStatus.FAILED:
        return _ROUTE_END

    return _ROUTE_COMPLETE


def build_release_risk_service_graph(
    service: ReleaseRiskCollectionService,
) -> Any:
    """Build and compile the service-backed release-risk workflow graph.

    Args:
        service: Existing application service that collects release risks.

    Returns:
        A compiled LangGraph workflow.

    The graph intentionally depends on the service protocol instead of the
    concrete ReleaseRunService class. This keeps the graph testable with fake
    services and production-ready with the real service.
    """
    graph = StateGraph(ReleaseRiskState)

    graph.add_node("start", _start_node)
    graph.add_node(
        "collect_release_risks",
        create_collect_release_risks_node(service),
    )
    graph.add_node("complete", _complete_node)

    graph.add_edge(START, "start")
    graph.add_edge("start", "collect_release_risks")
    graph.add_conditional_edges(
        "collect_release_risks",
        _route_after_collection,
        {
            _ROUTE_COMPLETE: "complete",
            _ROUTE_END: END,
        },
    )
    graph.add_edge("complete", END)

    return graph.compile()