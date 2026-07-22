"""Read-only runtime adapters for bounded AgentFlow tool execution."""

from __future__ import annotations

import asyncio
from typing import Protocol, cast

from pydantic import JsonValue

from app.schemas.agent_query import (
    AgentEntityReferences,
    AgentQueryContext,
    AgentQueryPlan,
    AgentQueryRequest,
)
from app.schemas.agent_tool import (
    AgentToolEvidence,
    AgentToolExecutionStatus,
    AgentToolInvocation,
    AgentToolName,
    AgentToolResult,
)
from app.schemas.agent_tool_arguments import (
    LookupGitHubPullRequestArguments,
    LookupJiraIssueArguments,
    SearchEngineeringKnowledgeArguments,
)
from app.services.agent_dynamic_execution_service import (
    AgentToolAdapterError,
)
from app.services.agent_github_pr_resolver import (
    AgentGitHubPRResolver,
    AgentGitHubPRResolverError,
)
from app.services.agent_jira_ticket_resolver import (
    AgentJiraTicketResolver,
    AgentJiraTicketResolverError,
)
from app.services.agent_query_context_resolver import (
    AgentQueryContextResolver,
    AgentQueryContextResolverError,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalRequest,
    EngineeringDocumentRetrievalResponse,
)


class EngineeringKnowledgeRetrievalProtocol(Protocol):
    """Knowledge retrieval operation required by the adapter."""

    async def retrieve_relevant_chunks(
        self,
        retrieval_request: EngineeringDocumentRetrievalRequest,
        *,
        run_id: str | None = None,
    ) -> EngineeringDocumentRetrievalResponse:
        """Return bounded ranked engineering-document chunks."""
        ...


class AgentToolExecutionContextProvider:
    """Load trusted release context once and share it across tool adapters."""

    def __init__(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
        context_resolver: AgentQueryContextResolver,
    ) -> None:
        """Initialize the cached execution-context provider.

        Args:
            request: Original validated manager query.
            query_plan: Trusted deterministic routing plan.
            context_resolver: Resolver for persisted release-risk snapshots.
        """
        self._request = request
        self._query_plan = query_plan
        self._context_resolver = context_resolver
        self._context: AgentQueryContext | None = None
        self._lock = asyncio.Lock()

    @property
    def query_plan(self) -> AgentQueryPlan:
        """Return the trusted deterministic query plan."""
        return self._query_plan

    async def get_context(self) -> AgentQueryContext:
        """Return cached trusted context, loading it at most once."""
        if self._context is not None:
            return self._context

        async with self._lock:
            if self._context is None:
                try:
                    self._context = await self._context_resolver.resolve(
                        self._request,
                        self._query_plan,
                    )
                except AgentQueryContextResolverError as exc:
                    raise AgentToolAdapterError(
                        "Trusted release context could not be resolved."
                    ) from exc

        return self._context


class LoadCurrentRiskSnapshotAdapter:
    """Expose a bounded summary of the trusted persisted risk snapshot."""

    def __init__(
        self,
        *,
        context_provider: AgentToolExecutionContextProvider,
    ) -> None:
        """Initialize the snapshot adapter."""
        self._context_provider = context_provider

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: dict[str, AgentToolResult],
    ) -> AgentToolResult:
        """Load trusted release context and return selected safe fields."""
        del dependency_results
        context = await self._context_provider.get_context()
        release_risk = context.release_risk
        risk_score = release_risk.risk_score

        output = cast(
            dict[str, JsonValue],
            {
                "release_run_id": str(context.release_run_id),
                "snapshot_id": str(context.snapshot_id),
                "snapshot_version": context.snapshot_version,
                "run_id": release_risk.release_run.run_id,
                "release_status": release_risk.release_run.status,
                "overall_severity": (
                    release_risk.release_summary.overall_severity.value
                ),
                "recommended_action": (
                    release_risk.release_summary.recommended_action.value
                ),
                "risk_score": risk_score.score if risk_score else None,
                "risk_level": (
                    risk_score.risk_level.value if risk_score else None
                ),
                "approval_required": release_risk.approval_required,
                "approval_status": release_risk.approval_status,
                "top_risk_count": len(
                    release_risk.release_summary.top_risks
                ),
            },
        )

        return AgentToolResult(
            step_id=invocation.step_id,
            tool_name=AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
            status=AgentToolExecutionStatus.SUCCESS,
            output=output,
            evidence=[
                AgentToolEvidence(
                    source_type="release_risk_snapshot",
                    source_id=str(context.snapshot_id),
                    title=(
                        "Release risk snapshot "
                        f"v{context.snapshot_version}"
                    ),
                )
            ],
            duration_ms=0,
        )


class LookupGitHubPullRequestAdapter:
    """Resolve one GitHub PR from trusted persisted release context."""

    def __init__(
        self,
        *,
        context_provider: AgentToolExecutionContextProvider,
        request_id: str,
    ) -> None:
        """Initialize the GitHub PR adapter."""
        self._context_provider = context_provider
        self._resolver = AgentGitHubPRResolver(request_id=request_id)

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: dict[str, AgentToolResult],
    ) -> AgentToolResult:
        """Resolve and normalize one persisted GitHub PR."""
        del dependency_results
        arguments = LookupGitHubPullRequestArguments.model_validate(
            invocation.arguments
        )
        context = await self._context_provider.get_context()
        plan = self._context_provider.query_plan.model_copy(
            update={
                "entity_references": AgentEntityReferences(
                    pull_request_numbers=[
                        arguments.pull_request_number
                    ]
                )
            }
        )

        try:
            pull_request = self._resolver.resolve(
                plan=plan,
                release_risk=context.release_risk,
            )
        except AgentGitHubPRResolverError as exc:
            raise AgentToolAdapterError(
                "The requested GitHub pull request was not found."
            ) from exc

        return AgentToolResult(
            step_id=invocation.step_id,
            tool_name=AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
            status=AgentToolExecutionStatus.SUCCESS,
            output=cast(
                dict[str, JsonValue],
                pull_request.model_dump(mode="json"),
            ),
            evidence=[
                AgentToolEvidence(
                    source_type=pull_request.source_type,
                    source_id=pull_request.source_id,
                    title=(
                        f"GitHub pull request "
                        f"#{pull_request.pull_request_number}"
                    ),
                    source_url=pull_request.source_url,
                )
            ],
            duration_ms=0,
        )


class LookupJiraIssueAdapter:
    """Resolve one Jira issue from trusted persisted release context."""

    def __init__(
        self,
        *,
        context_provider: AgentToolExecutionContextProvider,
        request_id: str,
    ) -> None:
        """Initialize the Jira adapter."""
        self._context_provider = context_provider
        self._resolver = AgentJiraTicketResolver(request_id=request_id)

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: dict[str, AgentToolResult],
    ) -> AgentToolResult:
        """Resolve and normalize one persisted Jira issue."""
        del dependency_results
        arguments = LookupJiraIssueArguments.model_validate(
            invocation.arguments
        )
        context = await self._context_provider.get_context()
        plan = self._context_provider.query_plan.model_copy(
            update={
                "entity_references": AgentEntityReferences(
                    jira_issue_keys=[arguments.issue_key]
                )
            }
        )

        try:
            jira_issue = self._resolver.resolve(
                plan=plan,
                release_risk=context.release_risk,
            )
        except AgentJiraTicketResolverError as exc:
            raise AgentToolAdapterError(
                "The requested Jira issue was not found."
            ) from exc

        return AgentToolResult(
            step_id=invocation.step_id,
            tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
            status=AgentToolExecutionStatus.SUCCESS,
            output=cast(
                dict[str, JsonValue],
                jira_issue.model_dump(mode="json"),
            ),
            evidence=[
                AgentToolEvidence(
                    source_type="jira_issue",
                    source_id=jira_issue.issue_key,
                    title=jira_issue.title,
                    source_url=jira_issue.issue_url,
                )
            ],
            duration_ms=0,
        )


class SearchEngineeringKnowledgeAdapter:
    """Search trusted engineering documents with bounded hybrid retrieval."""

    def __init__(
        self,
        *,
        retrieval_service: EngineeringKnowledgeRetrievalProtocol,
        request_id: str,
    ) -> None:
        """Initialize the engineering-knowledge adapter."""
        self._retrieval_service = retrieval_service
        self._request_id = request_id

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: dict[str, AgentToolResult],
    ) -> AgentToolResult:
        """Retrieve bounded document chunks and trusted citations."""
        del dependency_results
        arguments = SearchEngineeringKnowledgeArguments.model_validate(
            invocation.arguments
        )

        try:
            retrieval = (
                await self._retrieval_service.retrieve_relevant_chunks(
                    EngineeringDocumentRetrievalRequest(
                        query=arguments.query,
                        top_k=arguments.top_k,
                    ),
                    run_id=self._request_id,
                )
            )
        except Exception as exc:
            raise AgentToolAdapterError(
                "Engineering knowledge retrieval failed."
            ) from exc

        results = [
            {
                "document_id": str(result.document_id),
                "chunk_id": str(result.chunk_id),
                "title": result.title,
                "source_type": result.source_type.value,
                "source_uri": result.source_uri,
                "chunk_index": result.chunk_index,
                "score": result.score,
                "content": result.content[:2_000],
            }
            for result in retrieval.results
        ]
        evidence = [
            AgentToolEvidence(
                source_type="engineering_document_chunk",
                source_id=str(result.chunk_id),
                title=result.title,
                source_url=result.source_uri,
            )
            for result in retrieval.results
        ]

        return AgentToolResult(
            step_id=invocation.step_id,
            tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
            status=AgentToolExecutionStatus.SUCCESS,
            output=cast(
                dict[str, JsonValue],
                {
                    "query": retrieval.query,
                    "total_candidates": retrieval.total_candidates,
                    "result_count": len(results),
                    "results": results,
                },
            ),
            evidence=evidence,
            duration_ms=0,
        )


def build_read_only_tool_adapters(
    *,
    request: AgentQueryRequest,
    query_plan: AgentQueryPlan,
    context_resolver: AgentQueryContextResolver,
    knowledge_retrieval_service: EngineeringKnowledgeRetrievalProtocol,
    request_id: str,
) -> dict[AgentToolName, object]:
    """Build the first approved read-only runtime adapter set.

    Args:
        request: Original validated manager query.
        query_plan: Trusted deterministic routing plan.
        context_resolver: Persisted risk-context resolver.
        knowledge_retrieval_service: Trusted hybrid retrieval service.
        request_id: Correlation identifier for logs and retrieval.

    Returns:
        Adapter mapping suitable for the dynamic execution service.
    """
    context_provider = AgentToolExecutionContextProvider(
        request=request,
        query_plan=query_plan,
        context_resolver=context_resolver,
    )

    return {
        AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT: (
            LoadCurrentRiskSnapshotAdapter(
                context_provider=context_provider
            )
        ),
        AgentToolName.LOOKUP_GITHUB_PULL_REQUEST: (
            LookupGitHubPullRequestAdapter(
                context_provider=context_provider,
                request_id=request_id,
            )
        ),
        AgentToolName.LOOKUP_JIRA_ISSUE: LookupJiraIssueAdapter(
            context_provider=context_provider,
            request_id=request_id,
        ),
        AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE: (
            SearchEngineeringKnowledgeAdapter(
                retrieval_service=knowledge_retrieval_service,
                request_id=request_id,
            )
        ),
    }
