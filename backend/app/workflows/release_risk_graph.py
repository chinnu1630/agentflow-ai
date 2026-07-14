"""LangGraph assembly for AgentFlow AI release-risk workflows.

This module builds the initial release-risk graph.

Current scope:
- Create a deterministic LangGraph workflow
- Wire state transition nodes
- Allow sync and async node-set injection

Future scope:
- Replace preparation nodes with real async GitHub/Jira collection nodes
- Add hybrid RAG retrieval
- Add XGBoost risk scoring
- Add Claude synthesis
- Add human-in-the-loop approval
- Add approved Slack delivery
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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
WorkflowNodeResult = WorkflowStateUpdate | Awaitable[WorkflowStateUpdate]
WorkflowNode = Callable[[WorkflowStateInput], WorkflowNodeResult]


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
    """Run the release summary preparation node and return a LangGraph state update."""
    validated_state = _validate_state_input(state)
    updated_state = prepare_release_summary(validated_state)

    return _dump_state_update(updated_state)


def _complete_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
    """Run the workflow completion node and return a LangGraph state update."""
    validated_state = _validate_state_input(state)
    updated_state = complete_release_risk_workflow(validated_state)

    return _dump_state_update(updated_state)


@dataclass(frozen=True, slots=True)
class ReleaseRiskGraphNodeSet:
    """Node functions used by the release-risk graph.

    The default node set uses safe placeholder state-transition nodes.

    Later, we will inject real async nodes that call GitHub, Jira, RAG, ML,
    Claude, HITL, and Slack while keeping the graph structure stable.
    """

    start: WorkflowNode = _start_node
    prepare_github: WorkflowNode = _prepare_github_node
    prepare_jira: WorkflowNode = _prepare_jira_node
    prepare_summary: WorkflowNode = _prepare_summary_node
    complete: WorkflowNode = _complete_node


def build_release_risk_graph(
    nodes: ReleaseRiskGraphNodeSet | None = None,
) -> CompiledStateGraph[
    ReleaseRiskState,
    None,
    ReleaseRiskState,
    ReleaseRiskState,
]:
    """Build and compile the initial release-risk LangGraph workflow.

    Args:
        nodes: Optional node set. Defaults to safe placeholder workflow nodes.

    Returns:
        A compiled LangGraph workflow that can be invoked with a
        ReleaseRiskState-compatible dictionary.

    The return type is intentionally Any because LangGraph's compiled graph
    class is a third-party implementation detail. Application code should rely
    on graph behavior, not a private concrete class name.
    """
    graph_nodes = nodes or ReleaseRiskGraphNodeSet()
    graph = StateGraph(ReleaseRiskState)

    graph.add_node("start", graph_nodes.start)
    graph.add_node("prepare_github", graph_nodes.prepare_github)
    graph.add_node("prepare_jira", graph_nodes.prepare_jira)
    graph.add_node("prepare_summary", graph_nodes.prepare_summary)
    graph.add_node("complete", graph_nodes.complete)

    graph.add_edge(START, "start")
    graph.add_edge("start", "prepare_github")
    graph.add_edge("prepare_github", "prepare_jira")
    graph.add_edge("prepare_jira", "prepare_summary")
    graph.add_edge("prepare_summary", "complete")
    graph.add_edge("complete", END)

    return graph.compile()