"""Service-backed LangGraph workflow for AgentFlow AI release-risk collection.

This graph connects LangGraph orchestration to existing application services.

Current scope:
- Start workflow
- Call ReleaseRunService.collect_release_risks()
- Optionally retrieve internal engineering knowledge context
- Route to completion when collection succeeds
- Stop as failed when release run is missing

Future scope:
- Split GitHub/Jira collection into separate parallel nodes
- Add hybrid RAG with pgvector + BM25 + reranker
- Add XGBoost risk scoring node
- Add Claude synthesis node
- Add human approval gate
- Add approved Slack delivery node
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.workflows.release_risk_graph import (
    WorkflowStateInput,
    WorkflowStateUpdate,
)
from app.workflows.release_risk_nodes import (
    complete_release_risk_workflow,
    start_release_risk_workflow,
)
from app.workflows.release_risk_service_nodes import (
    KnowledgeRetrievalService,
    ReleaseRiskCollectionService,
    create_collect_release_risks_node,
    create_determine_approval_requirement_node,
    create_retrieve_knowledge_context_node,
    create_score_release_risk_node,
)
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStatus,
)

_ROUTE_COMPLETE = "complete"
_ROUTE_KNOWLEDGE = "knowledge"
_ROUTE_SCORE = "score"
_ROUTE_APPROVAL = "approval"
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


def _route_after_collection_without_knowledge(state: WorkflowStateInput) -> str:
    """Route after collection when no Knowledge Agent service is configured."""
    validated_state = _validate_state_input(state)

    if validated_state.status == ReleaseRiskWorkflowStatus.FAILED:
        return _ROUTE_END

    return _ROUTE_SCORE


def _route_after_collection_with_knowledge(state: WorkflowStateInput) -> str:
    """Route after collection when Knowledge Agent retrieval is configured."""
    validated_state = _validate_state_input(state)

    if validated_state.status == ReleaseRiskWorkflowStatus.FAILED:
        return _ROUTE_END

    return _ROUTE_KNOWLEDGE



def _route_after_scoring(state: WorkflowStateInput) -> str:
    """Route after release-risk scoring."""
    validated_state = _validate_state_input(state)

    if validated_state.status == ReleaseRiskWorkflowStatus.FAILED:
        return _ROUTE_END

    return _ROUTE_COMPLETE


def _route_after_approval_decision(state: WorkflowStateInput) -> str:
    """Route after HITL approval requirement decision."""
    validated_state = _validate_state_input(state)

    if validated_state.status == ReleaseRiskWorkflowStatus.FAILED:
        return _ROUTE_END

    return _ROUTE_COMPLETE

def build_release_risk_service_graph(
    service: ReleaseRiskCollectionService,
    *,
    knowledge_service: KnowledgeRetrievalService | None = None,
) -> CompiledStateGraph[
    ReleaseRiskState,
    None,
    ReleaseRiskState,
    ReleaseRiskState,
]:
    """Build and compile the service-backed release-risk workflow graph.

    Args:
        service: Existing application service that collects release risks.
        knowledge_service: Optional service that retrieves engineering docs.

    Returns:
        A compiled LangGraph workflow.

    The graph intentionally depends on service protocols instead of concrete
    service classes. This keeps the graph testable with fake services and
    production-ready with real services.
    """
    graph = StateGraph(ReleaseRiskState)

    graph.add_node("start", _start_node)
    graph.add_node(
        "collect_release_risks",
        create_collect_release_risks_node(service),
    )
    graph.add_node("complete", _complete_node)
    graph.add_node("score_release_risk", create_score_release_risk_node())
    graph.add_node("determine_approval_requirement", create_determine_approval_requirement_node())

    graph.add_edge(START, "start")
    graph.add_edge("start", "collect_release_risks")

    if knowledge_service is None:
        graph.add_conditional_edges(
            "collect_release_risks",
            _route_after_collection_without_knowledge,
            {
                _ROUTE_SCORE: "score_release_risk",
                _ROUTE_END: END,
            },
        )
    else:
        graph.add_node(
            "retrieve_knowledge_context",
            create_retrieve_knowledge_context_node(knowledge_service),
        )
        graph.add_conditional_edges(
            "collect_release_risks",
            _route_after_collection_with_knowledge,
            {
                _ROUTE_KNOWLEDGE: "retrieve_knowledge_context",
                _ROUTE_END: END,
            },
        )
        graph.add_edge("retrieve_knowledge_context", "score_release_risk")

    graph.add_conditional_edges(
        "score_release_risk",
        _route_after_scoring,
        {
            _ROUTE_COMPLETE: "determine_approval_requirement",
            _ROUTE_END: END,
        },
    )
    graph.add_conditional_edges(
        "determine_approval_requirement",
        _route_after_approval_decision,
        {
            _ROUTE_COMPLETE: "complete",
            _ROUTE_END: END,
        },
    )
    graph.add_edge("complete", END)

    return graph.compile()
