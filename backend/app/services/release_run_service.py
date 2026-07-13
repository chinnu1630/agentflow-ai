"""Release run business service for AgentFlow AI.

This module owns release-run workflow orchestration at the service layer.
It coordinates persistence, deterministic GitHub risk collection, GitHub
summary generation, Jira risk collection, Jira summary generation, combined
release-risk summary generation, and release-run audit events.

Architecture position:
FastAPI route -> ReleaseRunService -> Repository + Audit Events + GitHub Collector + Jira Collector
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.models.release_run import ReleaseRun
from app.observability.tracing import start_business_span
from app.repositories.release_run_event_repository import (
    CreateReleaseRunEventCommand,
    ReleaseRunEventRepository,
    ReleaseRunEventRepositoryError,
)
from app.repositories.release_run_repository import (
    ReleaseRunRepository,
    ReleaseRunRepositoryError,
)
from app.schemas.risk import (
    JiraIssueRiskResponse,
    JiraRiskCollectionResponse,
    RiskCollectionStatusResponse,
    RiskSignalResponse,
)
from app.services.github_risk_collector import GitHubRiskCollectionResult
from app.services.github_risk_summary import GitHubRiskSummary, RiskSummaryGenerator
from app.services.jira_risk_collector import (
    JiraRiskCollectionResult,
    JiraRiskCollectionStatus,
)
from app.services.jira_risk_summary import JiraRiskSummary, JiraRiskSummaryGenerator
from app.services.release_risk_summary import (
    ReleaseRiskSummary,
    ReleaseRiskSummaryGenerator,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.workflows.release_risk_service_nodes import KnowledgeRetrievalService
    from app.workflows.release_risk_state import ReleaseRiskState


class StartReleaseRunCommand(BaseModel):
    """Validated input for starting a release-risk workflow."""

    query: str = Field(
        min_length=5,
        max_length=500,
        description="Manager's release-risk question.",
    )
    requested_by: str = Field(
        min_length=3,
        max_length=255,
        description="User or manager who requested the workflow.",
    )


class ReleaseRunResult(BaseModel):
    """Service response returned after release-run operations."""

    id: UUID
    run_id: str
    query: str
    requested_by: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None


class ReleaseRunRiskResult(BaseModel):
    """Service response returned after collecting release-run risks."""

    release_run: ReleaseRunResult
    github: GitHubRiskCollectionResult
    github_summary: GitHubRiskSummary
    jira: JiraRiskCollectionResponse
    jira_summary: JiraRiskSummary
    release_summary: ReleaseRiskSummary


class ReleaseRunServiceError(RuntimeError):
    """Raised when release-run service operations fail."""


class ReleaseRunRepositoryProtocol(Protocol):
    """Repository contract required by ReleaseRunService."""

    async def create(self, release_run: ReleaseRun) -> ReleaseRun:
        """Create a release run."""

        ...

    async def get_by_id(self, release_run_id: UUID) -> ReleaseRun | None:
        """Fetch a release run by ID."""

        ...

    async def update_status(
        self,
        release_run_id: UUID,
        status: str,
    ) -> ReleaseRun | None:
        """Update release run status."""

        ...


class GitHubRiskCollectorProtocol(Protocol):
    """Collector contract required for GitHub release-risk collection."""

    async def collect_github_risks(
        self,
        *,
        run_id: str,
    ) -> GitHubRiskCollectionResult:
        """Collect GitHub risk results for a workflow run."""

        ...


class JiraRiskCollectorProtocol(Protocol):
    """Collector contract required for Jira release-risk collection."""

    async def collect(
        self,
        *,
        run_id: str,
    ) -> JiraRiskCollectionResult:
        """Collect Jira risk results for a workflow run."""

        ...


class ReleaseRunService:
    """Business service for managing release-risk workflow runs."""

    def __init__(
        self,
        repository: ReleaseRunRepository | ReleaseRunRepositoryProtocol,
        request_id: str,
        risk_collector: GitHubRiskCollectorProtocol | None = None,
        jira_risk_collector: JiraRiskCollectorProtocol | None = None,
        risk_summary_generator: RiskSummaryGenerator | None = None,
        jira_risk_summary_generator: JiraRiskSummaryGenerator | None = None,
        release_risk_summary_generator: ReleaseRiskSummaryGenerator | None = None,
        event_repository: ReleaseRunEventRepository | None = None,
        knowledge_service: KnowledgeRetrievalService | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            repository: Repository used for release-run persistence.
            request_id: Request-level ID for structured logs.
            risk_collector: Optional collector used to collect GitHub risks.
            jira_risk_collector: Optional collector used to collect Jira risks.
            risk_summary_generator: Optional generator used to summarize GitHub risks.
            jira_risk_summary_generator: Optional generator used to summarize Jira risks.
            release_risk_summary_generator:
                Optional generator used to summarize combined release risks.
            event_repository: Optional repository used to persist audit events.
            knowledge_service: Optional service used to retrieve engineering document evidence.
        """

        self._repository = repository
        self._request_id = request_id
        self._risk_collector = risk_collector
        self._jira_risk_collector = jira_risk_collector
        self._risk_summary_generator = risk_summary_generator or RiskSummaryGenerator()
        self._jira_risk_summary_generator = (
            jira_risk_summary_generator or JiraRiskSummaryGenerator()
        )
        self._release_risk_summary_generator = (
            release_risk_summary_generator or ReleaseRiskSummaryGenerator()
        )
        self._event_repository = event_repository
        self._knowledge_service = knowledge_service

    async def start_release_run(
        self,
        command: StartReleaseRunCommand,
    ) -> ReleaseRunResult:
        """Start a new release-risk workflow.

        Args:
            command: Validated manager query and requester information.

        Returns:
            ReleaseRunResult for the created workflow run.

        Raises:
            ReleaseRunServiceError: If the workflow cannot be started.
        """

        workflow_run_id = f"release-run-{uuid4().hex[:12]}"

        release_run = ReleaseRun(
            run_id=workflow_run_id,
            query=command.query,
            requested_by=command.requested_by,
            status="created",
        )

        try:
            created_release_run = await self._repository.create(release_run)

            await self._record_event(
                release_run_id=created_release_run.id,
                event_type="release_run_started",
                event_status="success",
                message="Release-risk workflow run was created.",
                metadata_json={
                    "workflow_run_id": created_release_run.run_id,
                    "requested_by": created_release_run.requested_by,
                },
            )

            logger.info(
                "release_run_service_started",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(created_release_run.id),
                    "workflow_run_id": created_release_run.run_id,
                    "requested_by": created_release_run.requested_by,
                },
            )

            return self._to_result(created_release_run)

        except (ReleaseRunRepositoryError, ReleaseRunEventRepositoryError) as exc:
            logger.exception(
                "release_run_service_start_failed",
                extra={
                    "request_id": self._request_id,
                    "workflow_run_id": workflow_run_id,
                },
            )
            raise ReleaseRunServiceError(
                "Failed to start release-risk workflow."
            ) from exc

    async def get_release_run(self, release_run_id: UUID) -> ReleaseRunResult | None:
        """Fetch a release run by ID.

        Args:
            release_run_id: Release run database UUID.

        Returns:
            ReleaseRunResult if found, otherwise None.

        Raises:
            ReleaseRunServiceError: If the lookup fails.
        """

        try:
            release_run = await self._repository.get_by_id(release_run_id)

            if release_run is None:
                return None

            return self._to_result(release_run)

        except ReleaseRunRepositoryError as exc:
            logger.exception(
                "release_run_service_fetch_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                },
            )
            raise ReleaseRunServiceError(
                "Failed to fetch release run."
            ) from exc

    async def collect_release_risks(
        self,
        release_run_id: UUID,
    ) -> ReleaseRunRiskResult | None:
        """Collect release risks across configured engineering sources.

        Args:
            release_run_id: Release run database UUID.

        Returns:
            ReleaseRunRiskResult if release run exists, otherwise None.

        Raises:
            ReleaseRunServiceError: If risk collection cannot be completed.
        """

        if self._risk_collector is None:
            logger.error(
                "release_run_service_risk_collector_missing",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                },
            )
            raise ReleaseRunServiceError(
                "Risk collector is not configured for this service."
            )

        try:
            release_run = await self._repository.get_by_id(release_run_id)

            if release_run is None:
                return None

            logger.info(
                "release_run_service_risk_collection_started",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run.id),
                    "workflow_run_id": release_run.run_id,
                },
            )

            await self._repository.update_status(
                release_run_id=release_run_id,
                status="running",
            )

            await self._record_event(
                release_run_id=release_run.id,
                event_type="risk_collection_started",
                event_status="started",
                message="Release risk collection started.",
                metadata_json={
                    "workflow_run_id": release_run.run_id,
                },
            )

            github_result = await self._risk_collector.collect_github_risks(
                run_id=release_run.run_id,
            )

            github_summary = self._risk_summary_generator.summarize_github_risks(
                github_result,
                run_id=release_run.run_id,
            )

            await self._record_event(
                release_run_id=release_run.id,
                event_type="github_collection_completed",
                event_status=github_result.status.value,
                message="GitHub risk collection completed.",
                metadata_json={
                    "pull_request_count": github_result.pull_request_count,
                    "risk_result_count": github_result.risk_result_count,
                    "total_signal_count": github_result.total_signal_count,
                    "high_risk_count": github_result.high_risk_count,
                    "overall_severity": github_summary.overall_severity.value,
                    "recommended_action": github_summary.recommended_action.value,
                },
            )

            jira_result = await self._collect_jira_risk_result(
                run_id=release_run.run_id,
            )

            jira_response = self._to_jira_response(jira_result)

            jira_summary = self._jira_risk_summary_generator.summarize_jira_risks(
                jira_result,
                run_id=release_run.run_id,
            )

            await self._record_event(
                release_run_id=release_run.id,
                event_type="jira_collection_completed",
                event_status=jira_result.status.value,
                message="Jira risk collection completed.",
                metadata_json={
                    "total_issues_analyzed": jira_response.total_issues_analyzed,
                    "total_signals": jira_response.total_signals,
                    "overall_severity": jira_summary.overall_severity.value,
                    "recommended_action": jira_summary.recommended_action.value,
                },
            )

            release_summary = (
                self._release_risk_summary_generator.summarize_release_risks(
                    github_summary=github_summary,
                    jira_summary=jira_summary,
                    run_id=release_run.run_id,
                )
            )

            await self._record_event(
                release_run_id=release_run.id,
                event_type="release_summary_created",
                event_status="success",
                message="Release-level risk summary created.",
                metadata_json={
                    "overall_severity": release_summary.overall_severity.value,
                    "recommended_action": release_summary.recommended_action.value,
                },
            )

            completed_release_run = await self._repository.update_status(
                release_run_id=release_run_id,
                status="completed",
            )

            if completed_release_run is None:
                logger.error(
                    "release_run_service_completion_missing",
                    extra={
                        "request_id": self._request_id,
                        "release_run_id": str(release_run_id),
                    },
                )
                raise ReleaseRunServiceError(
                    "Release run disappeared while completing risk collection."
                )

            await self._record_event(
                release_run_id=completed_release_run.id,
                event_type="risk_collection_completed",
                event_status="success",
                message="Release risk collection completed.",
                metadata_json={
                    "workflow_run_id": completed_release_run.run_id,
                    "github_status": github_result.status.value,
                    "jira_status": jira_response.status.value,
                    "release_overall_severity": release_summary.overall_severity.value,
                    "release_recommended_action": release_summary.recommended_action.value,
                },
            )

            logger.info(
                "release_run_service_risk_collection_completed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(completed_release_run.id),
                    "workflow_run_id": completed_release_run.run_id,
                    "github_status": github_result.status.value,
                    "pull_request_count": github_result.pull_request_count,
                    "total_signal_count": github_result.total_signal_count,
                    "high_risk_count": github_result.high_risk_count,
                    "github_summary_overall_severity": github_summary.overall_severity.value,
                    "github_summary_recommended_action": github_summary.recommended_action.value,
                    "jira_status": jira_response.status.value,
                    "jira_total_issues_analyzed": jira_response.total_issues_analyzed,
                    "jira_total_signals": jira_response.total_signals,
                    "jira_summary_overall_severity": jira_summary.overall_severity.value,
                    "jira_summary_recommended_action": jira_summary.recommended_action.value,
                    "release_summary_overall_severity": release_summary.overall_severity.value,
                    "release_summary_recommended_action": release_summary.recommended_action.value,
                },
            )

            return ReleaseRunRiskResult(
                release_run=self._to_result(completed_release_run),
                github=github_result,
                github_summary=github_summary,
                jira=jira_response,
                jira_summary=jira_summary,
                release_summary=release_summary,
            )

        except (ReleaseRunRepositoryError, ReleaseRunEventRepositoryError) as exc:
            logger.exception(
                "release_run_service_risk_collection_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                },
            )
            raise ReleaseRunServiceError(
                "Failed to collect release risks."
            ) from exc

    async def collect_github_risks(
        self,
        release_run_id: UUID,
    ) -> ReleaseRunRiskResult | None:
        """Backward-compatible wrapper for the previous GitHub-only method name.

        The workflow originally collected GitHub risks only. It now collects
        GitHub risks, Jira risks, source summaries, and the combined release
        summary. This wrapper keeps older callers working while newer code uses
        collect_release_risks().
        """

        return await self.collect_release_risks(release_run_id)

    async def run_release_risk_workflow(
        self,
        release_run_id: UUID,
        *,
        manager_query: str = "What are the biggest release risks this week?",
        requested_by: str | None = None,
    ) -> ReleaseRiskState:
        """Run release-risk collection through the LangGraph workflow.

        This method is intentionally separate from collect_release_risks().
        The LangGraph service node calls collect_release_risks(), so putting
        graph execution inside collect_release_risks() would create recursion.

        Args:
            release_run_id: Release run database UUID.
            manager_query: Original manager question for workflow context.
            requested_by: Optional user or system actor that requested the workflow.

        Returns:
            Final validated workflow state from the LangGraph service runner.
        """

        with start_business_span(
            "release_risk.workflow",
            {
                "release_run_id": str(release_run_id),
                "run_id": self._request_id,
                "manager_query_present": manager_query != "",
                "requested_by_present": requested_by is not None,
            },
        ):
            from app.workflows.release_risk_service_runner import (
                ReleaseRiskServiceWorkflowRunner,
            )

            await self._record_event(
                release_run_id=release_run_id,
                event_type="workflow_started",
                event_status="started",
                message="LangGraph release-risk workflow started.",
                metadata_json={
                    "manager_query": manager_query,
                    "requested_by": requested_by,
                },
            )

            runner = ReleaseRiskServiceWorkflowRunner(
                self,
                knowledge_service=getattr(self, "_knowledge_service", None),
            )

            try:
                workflow_state = await runner.run(
                    release_run_id=release_run_id,
                    run_id=self._request_id,
                    manager_query=manager_query,
                    requested_by=requested_by,
                )

                await self._record_knowledge_retrieval_event(
                    release_run_id=release_run_id,
                    workflow_state=workflow_state,
                )

                await self._record_event(
                    release_run_id=release_run_id,
                    event_type="workflow_completed",
                    event_status="success",
                    message="LangGraph release-risk workflow completed.",
                    metadata_json={
                        "manager_query": manager_query,
                        "requested_by": requested_by,
                    },
                )

                return workflow_state

            except ReleaseRunServiceError:
                await self._record_event(
                    release_run_id=release_run_id,
                    event_type="workflow_failed",
                    event_status="failed",
                    message="LangGraph release-risk workflow failed.",
                    metadata_json={
                        "manager_query": manager_query,
                        "requested_by": requested_by,
                    },
                )
                raise

    async def mark_running(self, release_run_id: UUID) -> ReleaseRunResult | None:
        """Mark a release run as running."""

        return await self._update_status(
            release_run_id=release_run_id,
            status="running",
        )

    async def mark_completed(self, release_run_id: UUID) -> ReleaseRunResult | None:
        """Mark a release run as completed."""

        return await self._update_status(
            release_run_id=release_run_id,
            status="completed",
        )

    async def mark_failed(self, release_run_id: UUID) -> ReleaseRunResult | None:
        """Mark a release run as failed."""

        return await self._update_status(
            release_run_id=release_run_id,
            status="failed",
        )

    async def _update_status(
        self,
        release_run_id: UUID,
        status: str,
    ) -> ReleaseRunResult | None:
        """Update release run status through the repository."""

        try:
            updated_release_run = await self._repository.update_status(
                release_run_id=release_run_id,
                status=status,
            )

            if updated_release_run is None:
                return None

            logger.info(
                "release_run_service_status_updated",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "status": status,
                },
            )

            return self._to_result(updated_release_run)

        except ReleaseRunRepositoryError as exc:
            logger.exception(
                "release_run_service_status_update_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "status": status,
                },
            )
            raise ReleaseRunServiceError(
                "Failed to update release run status."
            ) from exc

    async def _record_knowledge_retrieval_event(
        self,
        *,
        release_run_id: UUID,
        workflow_state: object,
    ) -> None:
        """Record safe audit metadata for Knowledge Agent retrieval.

        The audit event intentionally stores only counts and status metadata.
        It does not store raw document chunks, retrieved content, or the full
        knowledge query because engineering documents may contain sensitive
        internal system details.
        """

        knowledge_status = self._get_workflow_state_value(
            workflow_state,
            "knowledge_status",
        )
        status_value = self._normalize_workflow_state_value(knowledge_status)

        if status_value in {None, "not_started"}:
            return

        knowledge_results = self._get_workflow_state_value(
            workflow_state,
            "knowledge_results",
        )
        knowledge_query = self._get_workflow_state_value(
            workflow_state,
            "knowledge_query",
        )

        result_count = (
            len(knowledge_results)
            if isinstance(knowledge_results, list)
            else 0
        )
        query_length = len(knowledge_query) if isinstance(knowledge_query, str) else 0

        if status_value in {"completed", "no_results"}:
            await self._record_event(
                release_run_id=release_run_id,
                event_type="knowledge_retrieval_completed",
                event_status="success",
                message="Knowledge Agent retrieval completed.",
                metadata_json={
                    "result_count": result_count,
                    "query_length": query_length,
                    "knowledge_status": status_value,
                    "error_present": False,
                },
            )
            return

        if status_value == "failed":
            knowledge_error = self._get_workflow_state_value(
                workflow_state,
                "knowledge_error",
            )
            await self._record_event(
                release_run_id=release_run_id,
                event_type="knowledge_retrieval_failed",
                event_status="failed",
                message="Knowledge Agent retrieval failed.",
                metadata_json={
                    "result_count": result_count,
                    "query_length": query_length,
                    "knowledge_status": status_value,
                    "error_present": bool(knowledge_error),
                },
            )

    @staticmethod
    def _get_workflow_state_value(
        workflow_state: object,
        key: str,
    ) -> object | None:
        """Read a value from dict-like or Pydantic workflow state."""

        if isinstance(workflow_state, dict):
            return workflow_state.get(key)

        if hasattr(workflow_state, key):
            return getattr(workflow_state, key)

        if hasattr(workflow_state, "model_dump"):
            dumped_state = workflow_state.model_dump(mode="python")
            if isinstance(dumped_state, dict):
                return dumped_state.get(key)

        return None

    @staticmethod
    def _normalize_workflow_state_value(value: object | None) -> str | None:
        """Normalize enum or string workflow state values for audit decisions."""

        if value is None:
            return None

        if isinstance(value, Enum):
            return str(value.value)

        return str(value)

    async def _record_event(
        self,
        *,
        release_run_id: UUID,
        event_type: str,
        event_status: str,
        message: str,
        metadata_json: dict[str, Any] | None = None,
    ) -> None:
        """Record an audit event for a release run when audit logging is configured.

        The event repository is optional to preserve compatibility with older
        tests and service construction paths. When configured, audit writes are
        treated as part of the workflow because enterprise release decisions
        require traceability.
        """

        event_repository = getattr(self, "_event_repository", None)

        if event_repository is None:
            return

        await event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type=event_type,
                event_status=event_status,
                message=message,
                metadata_json=metadata_json or {},
            )
        )

    async def _collect_jira_risk_result(
        self,
        *,
        run_id: str,
    ) -> JiraRiskCollectionResult:
        """Collect Jira risks or return an empty result when Jira is not configured.

        Args:
            run_id: Workflow run ID used for audit logs and risk traceability.

        Returns:
            Jira collector result used by both Jira summary and API response mapping.
        """

        if self._jira_risk_collector is None:
            logger.info(
                "release_run_service_jira_risk_collector_missing",
                extra={
                    "request_id": self._request_id,
                    "workflow_run_id": run_id,
                },
            )
            return self._build_empty_jira_result()

        return await self._jira_risk_collector.collect(run_id=run_id)

    @staticmethod
    def _to_jira_response(
        jira_result: JiraRiskCollectionResult,
    ) -> JiraRiskCollectionResponse:
        """Convert Jira collector result into API response schema."""

        issue_responses = [
            JiraIssueRiskResponse(
                issue_key=issue.issue_key,
                title=issue.title,
                issue_url=issue.issue_url,
                signals=[
                    RiskSignalResponse.model_validate(
                        signal,
                        from_attributes=True,
                    )
                    for signal in issue_result.signals
                ],
            )
            for issue, issue_result in zip(
                jira_result.issues,
                jira_result.issue_results,
                strict=True,
            )
        ]

        return JiraRiskCollectionResponse(
            status=RiskCollectionStatusResponse(jira_result.status.value),
            total_issues_analyzed=jira_result.total_issues_analyzed,
            total_signals=jira_result.total_signals,
            issues=issue_responses,
            signals=[
                RiskSignalResponse.model_validate(
                    signal,
                    from_attributes=True,
                )
                for signal in jira_result.signals
            ],
            error_message=jira_result.error_message,
            duration_ms=jira_result.duration_ms,
        )

    @staticmethod
    def _build_empty_jira_result() -> JiraRiskCollectionResult:
        """Build an empty Jira collector result for tests or local fallback paths."""

        return JiraRiskCollectionResult(
            status=JiraRiskCollectionStatus.SUCCESS,
            issues=[],
            issue_results=[],
            signals=[],
            error_message=None,
            duration_ms=0.0,
        )

    @staticmethod
    def _to_result(release_run: ReleaseRun) -> ReleaseRunResult:
        """Convert a ReleaseRun database model into a service result."""

        return ReleaseRunResult(
            id=release_run.id,
            run_id=release_run.run_id,
            query=release_run.query,
            requested_by=release_run.requested_by,
            status=release_run.status,
            created_at=release_run.created_at,
            completed_at=release_run.completed_at,
        )