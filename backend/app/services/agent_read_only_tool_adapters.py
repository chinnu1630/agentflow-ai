"""Read-only runtime adapters for bounded AgentFlow tool execution."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, cast
from uuid import UUID

from pydantic import JsonValue

from app.repositories.release_run_approval_repository import (
    ReleaseRunApprovalRepositoryError,
)
from app.repositories.release_run_slack_alert_repository import (
    ReleaseRunSlackAlertRepositoryError,
)
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
    LookupApprovalStatusArguments,
    LookupGitHubPullRequestArguments,
    LookupJiraIssueArguments,
    LookupReleaseHistoryArguments,
    LookupSimilarReleaseArguments,
    LookupSlackStatusArguments,
    SearchEngineeringKnowledgeArguments,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.agent_dynamic_execution_service import (
    AgentToolAdapter,
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
from app.services.agent_similar_release_matcher import (
    AgentSimilarReleaseMatcher,
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


class ApprovalRecordProtocol(Protocol):
    """Durable approval fields required by the status adapter."""

    id: UUID
    release_run_id: UUID
    approval_status: str
    approval_reason: str
    approval_policy_version: str
    requested_by: str | None
    decided_by: str | None
    decision_note: str | None
    created_at: datetime
    decided_at: datetime | None


class ApprovalRepositoryProtocol(Protocol):
    """Approval repository operation required by the status adapter."""

    async def get_latest_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> ApprovalRecordProtocol | None:
        """Return the latest durable approval for a release run."""
        ...


class SlackAlertRecordProtocol(Protocol):
    """Durable Slack delivery fields required by the status adapter."""

    id: UUID
    release_run_id: UUID
    approval_request_id: UUID | None
    snapshot_id: UUID | None
    snapshot_version: int | None
    delivery_status: str
    slack_channel: str
    slack_timestamp: str
    risk_level: str
    risk_score: float
    recommended_action: str
    created_at: datetime


class SlackAlertRepositoryProtocol(Protocol):
    """Slack repository operation required by the status adapter."""

    async def get_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> SlackAlertRecordProtocol | None:
        """Return the persisted Slack delivery record."""
        ...


def _datetime_to_json(value: datetime | None) -> str | None:
    """Convert a timestamp into a JSON-safe ISO-8601 string."""

    return value.isoformat() if value is not None else None


def _release_risk_summary_output(
    release_risk: ReleaseRunRiskResponse,
) -> dict[str, JsonValue]:
    """Return a bounded JSON-safe summary of one validated release risk."""

    risk_score = release_risk.risk_score

    return cast(
        dict[str, JsonValue],
        {
            "release_run_id": str(release_risk.release_run.id),
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
        dependency_results: Mapping[str, AgentToolResult],
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
        dependency_results: Mapping[str, AgentToolResult],
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
        dependency_results: Mapping[str, AgentToolResult],
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
        dependency_results: Mapping[str, AgentToolResult],
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


class LookupReleaseHistoryAdapter:
    """Load bounded summaries of validated historical release snapshots."""

    def __init__(
        self,
        *,
        context_provider: AgentToolExecutionContextProvider,
        context_resolver: AgentQueryContextResolver,
    ) -> None:
        """Initialize the release-history adapter."""

        self._context_provider = context_provider
        self._context_resolver = context_resolver

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: Mapping[str, AgentToolResult],
    ) -> AgentToolResult:
        """Load validated historical release-risk summaries."""

        del dependency_results

        arguments = LookupReleaseHistoryArguments.model_validate(
            invocation.arguments
        )
        context = await self._context_provider.get_context()

        try:
            historical_risks = (
                await self._context_resolver
                .resolve_historical_release_risks(
                    exclude_release_run_id=context.release_run_id,
                    limit=arguments.limit,
                )
            )
        except AgentQueryContextResolverError as exc:
            raise AgentToolAdapterError(
                "Historical release-risk context could not be resolved."
            ) from exc

        releases = [
            _release_risk_summary_output(release_risk)
            for release_risk in historical_risks
        ]

        return AgentToolResult(
            step_id=invocation.step_id,
            tool_name=AgentToolName.LOOKUP_RELEASE_HISTORY,
            status=AgentToolExecutionStatus.SUCCESS,
            output=cast(
                dict[str, JsonValue],
                {
                    "release_count": len(releases),
                    "releases": releases,
                },
            ),
            evidence=[
                AgentToolEvidence(
                    source_type="historical_release_risk",
                    source_id=str(release_risk.release_run.id),
                    title=release_risk.release_run.run_id,
                )
                for release_risk in historical_risks
            ],
            duration_ms=0,
        )


class LookupSimilarReleaseAdapter:
    """Find the closest historical release using deterministic features."""

    def __init__(
        self,
        *,
        context_provider: AgentToolExecutionContextProvider,
        context_resolver: AgentQueryContextResolver,
        request_id: str,
    ) -> None:
        """Initialize the similar-release adapter."""

        self._context_provider = context_provider
        self._context_resolver = context_resolver
        self._matcher = AgentSimilarReleaseMatcher(
            request_id=request_id
        )

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: Mapping[str, AgentToolResult],
    ) -> AgentToolResult:
        """Return the deterministic closest historical release."""

        del dependency_results

        arguments = LookupSimilarReleaseArguments.model_validate(
            invocation.arguments
        )
        context = await self._context_provider.get_context()

        try:
            historical_risks = (
                await self._context_resolver
                .resolve_historical_release_risks(
                    exclude_release_run_id=context.release_run_id,
                    limit=arguments.limit,
                )
            )
        except AgentQueryContextResolverError as exc:
            raise AgentToolAdapterError(
                "Historical release-risk context could not be resolved."
            ) from exc

        match = self._matcher.match(
            current_release_risk=context.release_risk,
            historical_release_risks=historical_risks,
        )

        if match is None:
            return AgentToolResult(
                step_id=invocation.step_id,
                tool_name=AgentToolName.LOOKUP_SIMILAR_RELEASE,
                status=AgentToolExecutionStatus.SUCCESS,
                output={"found": False},
                evidence=[],
                duration_ms=0,
            )

        matched_risk = match.release_risk

        return AgentToolResult(
            step_id=invocation.step_id,
            tool_name=AgentToolName.LOOKUP_SIMILAR_RELEASE,
            status=AgentToolExecutionStatus.SUCCESS,
            output=cast(
                dict[str, JsonValue],
                {
                    "found": True,
                    "similarity_score": match.similarity_score,
                    "release": _release_risk_summary_output(
                        matched_risk
                    ),
                },
            ),
            evidence=[
                AgentToolEvidence(
                    source_type="historical_release_risk",
                    source_id=str(matched_risk.release_run.id),
                    title=matched_risk.release_run.run_id,
                )
            ],
            duration_ms=0,
        )


class LookupApprovalStatusAdapter:
    """Load the latest durable human-approval state."""

    def __init__(
        self,
        *,
        context_provider: AgentToolExecutionContextProvider,
        approval_repository: ApprovalRepositoryProtocol,
    ) -> None:
        """Initialize the approval-status adapter."""

        self._context_provider = context_provider
        self._approval_repository = approval_repository

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: Mapping[str, AgentToolResult],
    ) -> AgentToolResult:
        """Return the latest persisted approval record."""

        del dependency_results

        LookupApprovalStatusArguments.model_validate(
            invocation.arguments
        )
        context = await self._context_provider.get_context()

        try:
            approval = (
                await self._approval_repository
                .get_latest_by_release_run_id(context.release_run_id)
            )
        except ReleaseRunApprovalRepositoryError as exc:
            raise AgentToolAdapterError(
                "Durable approval status could not be loaded."
            ) from exc

        if approval is None:
            return AgentToolResult(
                step_id=invocation.step_id,
                tool_name=AgentToolName.LOOKUP_APPROVAL_STATUS,
                status=AgentToolExecutionStatus.SUCCESS,
                output={"found": False},
                evidence=[],
                duration_ms=0,
            )

        return AgentToolResult(
            step_id=invocation.step_id,
            tool_name=AgentToolName.LOOKUP_APPROVAL_STATUS,
            status=AgentToolExecutionStatus.SUCCESS,
            output=cast(
                dict[str, JsonValue],
                {
                    "found": True,
                    "approval_id": str(approval.id),
                    "release_run_id": str(approval.release_run_id),
                    "approval_status": approval.approval_status,
                    "approval_reason": approval.approval_reason,
                    "approval_policy_version": (
                        approval.approval_policy_version
                    ),
                    "requested_by": approval.requested_by,
                    "decided_by": approval.decided_by,
                    "decision_note": approval.decision_note,
                    "created_at": _datetime_to_json(
                        approval.created_at
                    ),
                    "decided_at": _datetime_to_json(
                        approval.decided_at
                    ),
                },
            ),
            evidence=[
                AgentToolEvidence(
                    source_type="release_run_approval",
                    source_id=str(approval.id),
                    title=(
                        f"Release approval {approval.approval_status}"
                    ),
                )
            ],
            duration_ms=0,
        )


class LookupSlackStatusAdapter:
    """Load persisted Slack delivery state without sending a message."""

    def __init__(
        self,
        *,
        context_provider: AgentToolExecutionContextProvider,
        slack_alert_repository: SlackAlertRepositoryProtocol,
    ) -> None:
        """Initialize the Slack-status adapter."""

        self._context_provider = context_provider
        self._slack_alert_repository = slack_alert_repository

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: Mapping[str, AgentToolResult],
    ) -> AgentToolResult:
        """Return the persisted Slack alert record."""

        del dependency_results

        LookupSlackStatusArguments.model_validate(
            invocation.arguments
        )
        context = await self._context_provider.get_context()

        try:
            alert = (
                await self._slack_alert_repository
                .get_by_release_run_id(context.release_run_id)
            )
        except ReleaseRunSlackAlertRepositoryError as exc:
            raise AgentToolAdapterError(
                "Persisted Slack delivery status could not be loaded."
            ) from exc

        if alert is None:
            return AgentToolResult(
                step_id=invocation.step_id,
                tool_name=AgentToolName.LOOKUP_SLACK_STATUS,
                status=AgentToolExecutionStatus.SUCCESS,
                output={"found": False},
                evidence=[],
                duration_ms=0,
            )

        return AgentToolResult(
            step_id=invocation.step_id,
            tool_name=AgentToolName.LOOKUP_SLACK_STATUS,
            status=AgentToolExecutionStatus.SUCCESS,
            output=cast(
                dict[str, JsonValue],
                {
                    "found": True,
                    "slack_alert_id": str(alert.id),
                    "release_run_id": str(alert.release_run_id),
                    "approval_request_id": (
                        str(alert.approval_request_id)
                        if alert.approval_request_id is not None
                        else None
                    ),
                    "snapshot_id": (
                        str(alert.snapshot_id)
                        if alert.snapshot_id is not None
                        else None
                    ),
                    "snapshot_version": alert.snapshot_version,
                    "delivery_status": alert.delivery_status,
                    "slack_channel": alert.slack_channel,
                    "slack_timestamp": alert.slack_timestamp,
                    "risk_level": alert.risk_level,
                    "risk_score": alert.risk_score,
                    "recommended_action": alert.recommended_action,
                    "created_at": _datetime_to_json(
                        alert.created_at
                    ),
                },
            ),
            evidence=[
                AgentToolEvidence(
                    source_type="release_run_slack_alert",
                    source_id=str(alert.id),
                    title=f"Slack delivery {alert.delivery_status}",
                )
            ],
            duration_ms=0,
        )


def build_read_only_tool_adapters(
    *,
    request: AgentQueryRequest,
    query_plan: AgentQueryPlan,
    context_resolver: AgentQueryContextResolver,
    knowledge_retrieval_service: EngineeringKnowledgeRetrievalProtocol,
    approval_repository: ApprovalRepositoryProtocol,
    slack_alert_repository: SlackAlertRepositoryProtocol,
    request_id: str,
) -> dict[AgentToolName, AgentToolAdapter]:
    """Build all approved read-only runtime tool adapters."""

    context_provider = AgentToolExecutionContextProvider(
        request=request,
        query_plan=query_plan,
        context_resolver=context_resolver,
    )

    return {
        AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT: (
            LoadCurrentRiskSnapshotAdapter(
                context_provider=context_provider,
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
        AgentToolName.LOOKUP_RELEASE_HISTORY: (
            LookupReleaseHistoryAdapter(
                context_provider=context_provider,
                context_resolver=context_resolver,
            )
        ),
        AgentToolName.LOOKUP_SIMILAR_RELEASE: (
            LookupSimilarReleaseAdapter(
                context_provider=context_provider,
                context_resolver=context_resolver,
                request_id=request_id,
            )
        ),
        AgentToolName.LOOKUP_APPROVAL_STATUS: (
            LookupApprovalStatusAdapter(
                context_provider=context_provider,
                approval_repository=approval_repository,
            )
        ),
        AgentToolName.LOOKUP_SLACK_STATUS: LookupSlackStatusAdapter(
            context_provider=context_provider,
            slack_alert_repository=slack_alert_repository,
        ),
    }
