"""LangGraph assembly for AgentFlow AI release-risk workflows.

This module builds the initial release-risk graph.

Current scope:
- Create a deterministic LangGraph workflow
- Wire existing state transition nodes
- Prove the graph can execute end-to-end in memory

Future scope:
- Replace preparation nodes with real async GitHub/Jira collection nodes
- Add hybrid RAG retrieval
- Add XGBoost risk scoring
- Add Claude synthesis
- Add human-in-the-loop approval
- Add approved Slack delivery
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.workflows.release_risk_nodes import (
    complete_release_risk_workflow,
    prepare_github_risk_collection,
    prepare_jira_risk_collection,
    prepare_release_summary,
    start_release_risk_workflow,
)
from app.workflows.release_risk_state import ReleaseRiskState


WorkflowStateInput = ReleaseRiskState | dict[str, Any]
WorkflowStateUpdate = dict[str, Any]


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


def _prepare_github_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
    """Run the GitHub preparation node and return a LangGraph state update."""
    validated_state = _validate_state_input(state)
    updated_state = prepare_github_risk_collection(validated_state)

    return _dump_state_update(updated_state)


def _prepare_jira_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
    """Run the Jira preparation node and return a LangGraph state update."""
    validated_state = _validate_state_input(state)
    updated_state = prepare_jira_risk_collection(validated_state)

    return _dump_state_update(updated_state)


def _prepare_summary_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
    """Run the release summary preparation node and return a state update."""
    validated_state = _validate_state_input(state)
    updated_state = prepare_release_summary(validated_state)

    return _dump_state_update(updated_state)


def _complete_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
    """Run the workflow completion node and return a LangGraph state update."""
    validated_state = _validate_state_input(state)
    updated_state = complete_release_risk_workflow(validated_state)

    return _dump_state_update(updated_state)


def build_release_risk_graph() -> Any:
    """Build and compile the initial release-risk LangGraph workflow.

    Returns:
        A compiled LangGraph workflow that can be invoked with ReleaseRiskState
        compatible input.

    The return type is Any because LangGraph's compiled graph type can change
    between versions, and we do not want our application code tightly coupled
    to an internal third-party class name.
    """
    graph = StateGraph(ReleaseRiskState)

    graph.add_node("start", _start_node)
    graph.add_node("prepare_github", _prepare_github_node)
    graph.add_node("prepare_jira", _prepare_jira_node)
    graph.add_node("prepare_summary", _prepare_summary_node)
    graph.add_node("complete", _complete_node)

    graph.add_edge(START, "start")
    graph.add_edge("start", "prepare_github")
    graph.add_edge("prepare_github", "prepare_jira")
    graph.add_edge("prepare_jira", "prepare_summary")
    graph.add_edge("prepare_summary", "complete")
    graph.add_edge("complete", END)

    return graph.compile()