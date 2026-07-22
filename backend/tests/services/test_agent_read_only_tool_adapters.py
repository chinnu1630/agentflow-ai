"""Tests for AgentFlow read-only runtime tool adapters."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.models.engineering_document import (
    EngineeringDocumentSourceType,
)
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryContext,
    AgentQueryPlan,
    AgentQueryRequest,
    ResponseDepth,
)
from app.schemas.agent_tool import (
    AgentToolExecutionStatus,
    AgentToolInvocation,
    AgentToolName,
)
from app.services.agent_read_only_tool_adapters import (
    AgentToolExecutionContextProvider,
    LoadCurrentRiskSnapshotAdapter,
    LookupGitHubPullRequestAdapter,
    LookupJiraIssueAdapter,
    SearchEngineeringKnowledgeAdapter,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalResponse,
    EngineeringDocumentRetrievalResult,
)
from tests.services.test_agent_github_pr_resolver import (
    build_release_risk_response as build_github_release_risk,
)
from tests.services.test_agent_jira_ticket_resolver import (
    build_release_risk_response as build_jira_release_risk,
)


class FakeContextResolver:
    """Return configured trusted context and record load count."""

    def __init__(self, context: AgentQueryContext) -> None:
        self.context = context
        self.call_count = 0

    async def resolve(
        self,
        request: AgentQueryRequest,
        plan: AgentQueryPlan,
    ) -> AgentQueryContext:
        """Return the configured context."""
        del request, plan
        self.call_count += 1
        return self.context


class FakeKnowledgeRetrievalService:
    """Return a configured retrieval response."""

    def __init__(
        self,
        response: EngineeringDocumentRetrievalResponse,
    ) -> None:
        self.response = response
        self.requested_query: str | None = None
        self.requested_top_k: int | None = None
        self.run_id: str | None = None

    async def retrieve_relevant_chunks(
        self,
        retrieval_request: object,
        *,
        run_id: str | None = None,
    ) -> EngineeringDocumentRetrievalResponse:
        """Return the configured response."""
        self.requested_query = retrieval_request.query
        self.requested_top_k = retrieval_request.top_k
        self.run_id = run_id
        return self.response


def _build_plan() -> AgentQueryPlan:
    """Create a trusted deterministic query plan."""
    return AgentQueryPlan(
        intent=AgentIntent.EXPLAIN_RISK_SCORE,
        response_depth=ResponseDepth.DEEP,
        confidence=1.0,
        requires_current_snapshot=True,
        routing_reason_code="test_dynamic_execution",
    )


def _build_context(
    *,
    release_risk: object,
) -> AgentQueryContext:
    """Create trusted persisted execution context."""
    validated_release_risk = release_risk
    return AgentQueryContext(
        release_run_id=validated_release_risk.release_run.id,
        snapshot_id=uuid4(),
        snapshot_version=3,
        release_risk=validated_release_risk,
    )


def _build_provider(
    *,
    release_risk: object,
) -> tuple[AgentToolExecutionContextProvider, FakeContextResolver]:
    """Create a provider backed by one fake context resolver."""
    context = _build_context(release_risk=release_risk)
    resolver = FakeContextResolver(context)
    provider = AgentToolExecutionContextProvider(
        request=AgentQueryRequest(
            query="Explain this release.",
            release_run_id=context.release_run_id,
        ),
        query_plan=_build_plan().model_copy(
            update={"release_run_id": context.release_run_id}
        ),
        context_resolver=resolver,
    )
    return provider, resolver


@pytest.mark.asyncio
async def test_context_provider_loads_snapshot_once() -> None:
    """Concurrent adapters share one trusted snapshot load."""
    provider, resolver = _build_provider(
        release_risk=build_github_release_risk()
    )

    first_context = await provider.get_context()
    second_context = await provider.get_context()

    assert first_context is second_context
    assert resolver.call_count == 1


@pytest.mark.asyncio
async def test_snapshot_adapter_returns_bounded_summary() -> None:
    """Snapshot adapter excludes the full nested risk payload."""
    provider, _ = _build_provider(
        release_risk=build_github_release_risk()
    )
    adapter = LoadCurrentRiskSnapshotAdapter(
        context_provider=provider
    )

    result = await adapter.execute(
        invocation=AgentToolInvocation(
            step_id="snapshot",
            tool_name=AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
            timeout_seconds=10,
        ),
        dependency_results={},
    )

    assert result.status is AgentToolExecutionStatus.SUCCESS
    assert result.output["snapshot_version"] == 3
    assert result.output["risk_score"] == 0.78
    assert "github" not in result.output
    assert result.evidence[0].source_type == "release_risk_snapshot"


@pytest.mark.asyncio
async def test_github_adapter_resolves_persisted_pr() -> None:
    """GitHub adapter reuses the existing trusted PR resolver."""
    provider, _ = _build_provider(
        release_risk=build_github_release_risk()
    )
    adapter = LookupGitHubPullRequestAdapter(
        context_provider=provider,
        request_id="request-123",
    )

    result = await adapter.execute(
        invocation=AgentToolInvocation(
            step_id="github",
            tool_name=AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
            arguments={"pull_request_number": 42},
            timeout_seconds=10,
        ),
        dependency_results={},
    )

    assert result.output["pull_request_number"] == 42
    assert result.output["total_score"] == 0.85
    assert result.evidence[0].source_id == "PR-42"


@pytest.mark.asyncio
async def test_jira_adapter_resolves_persisted_issue() -> None:
    """Jira adapter reuses the existing trusted issue resolver."""
    provider, _ = _build_provider(
        release_risk=build_jira_release_risk()
    )
    adapter = LookupJiraIssueAdapter(
        context_provider=provider,
        request_id="request-123",
    )

    result = await adapter.execute(
        invocation=AgentToolInvocation(
            step_id="jira",
            tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
            arguments={"issue_key": "PAY-102"},
            timeout_seconds=10,
        ),
        dependency_results={},
    )

    assert result.output["issue_key"] == "PAY-102"
    assert result.output["title"] == "Payment release blocker"
    assert result.evidence[0].source_id == "PAY-102"


@pytest.mark.asyncio
async def test_knowledge_adapter_returns_bounded_cited_chunks() -> None:
    """Knowledge adapter returns bounded content and trusted evidence."""
    document_id = UUID("11111111-1111-1111-1111-111111111111")
    chunk_id = UUID("22222222-2222-2222-2222-222222222222")
    retrieval = EngineeringDocumentRetrievalResponse(
        query="payment rollback",
        total_candidates=4,
        results=[
            EngineeringDocumentRetrievalResult(
                document_id=document_id,
                chunk_id=chunk_id,
                title="Payment Service Runbook",
                source_type=EngineeringDocumentSourceType.RUNBOOK,
                source_uri="internal://payment-runbook",
                chunk_index=2,
                score=0.91,
                content="Rollback the payment service using the prior image.",
                token_count=10,
                metadata_json={"team": "payments"},
            )
        ],
    )
    retrieval_service = FakeKnowledgeRetrievalService(retrieval)
    adapter = SearchEngineeringKnowledgeAdapter(
        retrieval_service=retrieval_service,
        request_id="request-123",
    )

    result = await adapter.execute(
        invocation=AgentToolInvocation(
            step_id="knowledge",
            tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
            arguments={
                "query": "payment rollback",
                "top_k": 3,
            },
            timeout_seconds=30,
        ),
        dependency_results={},
    )

    assert result.status is AgentToolExecutionStatus.SUCCESS
    assert result.output["result_count"] == 1
    assert result.output["results"][0]["chunk_id"] == str(chunk_id)
    assert result.evidence[0].source_id == str(chunk_id)
    assert retrieval_service.requested_top_k == 3
    assert retrieval_service.run_id == "request-123"
