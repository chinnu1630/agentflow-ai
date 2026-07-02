"""Release run business service for AgentFlow AI.

This module owns release-run workflow orchestration at the service layer.
It coordinates persistence, deterministic GitHub risk collection, GitHub
summary generation, and Jira risk collection.

Architecture position:
FastAPI route -> ReleaseRunService -> Repository + GitHub Collector + Jira Collector
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.models.release_run import ReleaseRun
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
from app.services.jira_risk_collector import JiraRiskCollectionResult
from app.services.github_risk_collector import GitHubRiskCollectionResult
from app.services.github_risk_summary import GitHubRiskSummary, RiskSummaryGenerator

logger = logging.getLogger(__name__)


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
    ) -> None:
        """Initialize the service.

        Args:
            repository: Repository used for release-run persistence.
            request_id: Request-level ID for structured logs.
            risk_collector: Optional collector used to collect GitHub risks.
            jira_risk_collector: Optional collector used to collect Jira risks.
            risk_summary_generator: Optional generator used to summarize GitHub risks.
        """

        self._repository = repository
        self._request_id = request_id
        self._risk_collector = risk_collector
        self._jira_risk_collector = jira_risk_collector
        self._risk_summary_generator = risk_summary_generator or RiskSummaryGenerator()

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

        except ReleaseRunRepositoryError as exc:
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

    async def collect_github_risks(
        self,
        release_run_id: UUID,
    ) -> ReleaseRunRiskResult | None:
        """Collect GitHub and Jira risks for an existing release run.

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

            github_result = await self._risk_collector.collect_github_risks(
                run_id=release_run.run_id,
            )

            github_summary = self._risk_summary_generator.summarize_github_risks(
                github_result,
                run_id=release_run.run_id,
            )

            jira_response = await self._collect_jira_risks(
                run_id=release_run.run_id,
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
                    "overall_severity": github_summary.overall_severity.value,
                    "recommended_action": github_summary.recommended_action.value,
                    "jira_status": jira_response.status.value,
                    "jira_total_issues_analyzed": jira_response.total_issues_analyzed,
                    "jira_total_signals": jira_response.total_signals,
                },
            )

            return ReleaseRunRiskResult(
                release_run=self._to_result(completed_release_run),
                github=github_result,
                github_summary=github_summary,
                jira=jira_response,
            )

        except ReleaseRunRepositoryError as exc:
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

    async def _collect_jira_risks(
        self,
        *,
        run_id: str,
    ) -> JiraRiskCollectionResponse:
        """Collect Jira risks or return an empty response when Jira is not configured.

        Args:
            run_id: Workflow run ID used for audit logs and risk traceability.

        Returns:
            API-safe Jira risk collection response.
        """

        if self._jira_risk_collector is None:
            logger.info(
                "release_run_service_jira_risk_collector_missing",
                extra={
                    "request_id": self._request_id,
                    "workflow_run_id": run_id,
                },
            )
            return self._build_empty_jira_response()

        jira_result = await self._jira_risk_collector.collect(run_id=run_id)
        return self._to_jira_response(jira_result)

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
    def _build_empty_jira_response() -> JiraRiskCollectionResponse:
        """Build an empty Jira risk response.

        This keeps the release-risk API contract stable when Jira collection is
        not configured in a test or local development path.
        """

        return JiraRiskCollectionResponse(
            status=RiskCollectionStatusResponse.SUCCESS,
            total_issues_analyzed=0,
            total_signals=0,
            issues=[],
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