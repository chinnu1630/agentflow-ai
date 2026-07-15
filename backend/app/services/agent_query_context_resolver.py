"""Resolve trusted persisted context for follow-up AgentFlow queries."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from app.repositories.release_run_risk_snapshot_repository import (
    ReleaseRunRiskSnapshotRepositoryError,
)
from app.schemas.agent_query import (
    AgentQueryContext,
    AgentQueryPlan,
    AgentQueryRequest,
)
from app.schemas.risk import ReleaseRunRiskResponse

logger = logging.getLogger(__name__)


class RiskSnapshotRecordProtocol(Protocol):
    """Snapshot fields required by the context resolver."""

    id: UUID
    release_run_id: UUID
    snapshot_version: int
    risk_payload_json: str


class RiskSnapshotRepositoryProtocol(Protocol):
    """Repository operations required by the context resolver."""

    async def get_latest_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> RiskSnapshotRecordProtocol | None:
        """Return the latest persisted snapshot for a release run."""

        ...

    async def list_latest_previous_release_snapshots(
        self,
        *,
        exclude_release_run_id: UUID,
        limit: int = 10,
    ) -> Sequence[RiskSnapshotRecordProtocol]:
        """Return the latest snapshot from each previous release run."""

        ...


class AgentQueryContextResolverError(RuntimeError):
    """Base error raised when query context cannot be resolved."""


class AgentQueryContextRequiredError(AgentQueryContextResolverError):
    """Raised when a follow-up query has no release-run context."""


class AgentQueryContextConflictError(AgentQueryContextResolverError):
    """Raised when request and plan context identifiers conflict."""


class AgentQuerySnapshotNotFoundError(AgentQueryContextResolverError):
    """Raised when no persisted snapshot exists for a release run."""


class AgentQuerySnapshotValidationError(AgentQueryContextResolverError):
    """Raised when persisted snapshot data is invalid or inconsistent."""


class AgentQueryContextResolver:
    """Resolve follow-up query context from trusted risk snapshots."""

    def __init__(
        self,
        snapshot_repository: RiskSnapshotRepositoryProtocol,
        request_id: str,
    ) -> None:
        """Initialize the context resolver.

        Args:
            snapshot_repository: Repository used to read trusted snapshots.
            request_id: Request identifier used for structured logging.
        """

        self._snapshot_repository = snapshot_repository
        self._request_id = request_id

    async def resolve(
        self,
        request: AgentQueryRequest,
        plan: AgentQueryPlan,
    ) -> AgentQueryContext:
        """Resolve and validate the latest release-risk snapshot.

        Args:
            request: Original validated natural-language query.
            plan: Validated query routing plan.

        Returns:
            Trusted persisted query context.

        Raises:
            AgentQueryContextRequiredError: When no release-run ID is present.
            AgentQueryContextConflictError: When context IDs conflict.
            AgentQuerySnapshotNotFoundError: When no snapshot exists.
            AgentQuerySnapshotValidationError: When snapshot data is invalid.
            AgentQueryContextResolverError: When repository access fails.
        """

        release_run_id = self._resolve_release_run_id(request, plan)

        try:
            snapshot = await self._snapshot_repository.get_latest_by_release_run_id(release_run_id)
        except ReleaseRunRiskSnapshotRepositoryError as exc:
            raise AgentQueryContextResolverError(
                "Failed to load persisted agent query context."
            ) from exc

        if snapshot is None:
            raise AgentQuerySnapshotNotFoundError("No persisted release-risk snapshot was found.")

        release_risk = self._validate_snapshot(
            snapshot,
            expected_release_run_id=release_run_id,
        )

        logger.info(
            "agent_query_context_resolved",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_run_id),
                "snapshot_id": str(snapshot.id),
                "snapshot_version": snapshot.snapshot_version,
                "intent": plan.intent.value,
            },
        )

        return AgentQueryContext(
            release_run_id=release_run_id,
            snapshot_id=snapshot.id,
            snapshot_version=snapshot.snapshot_version,
            release_risk=release_risk,
        )

    async def resolve_historical_release_risks(
        self,
        *,
        exclude_release_run_id: UUID,
        limit: int = 10,
    ) -> list[ReleaseRunRiskResponse]:
        """Load and validate previous persisted release-risk snapshots.

        Args:
            exclude_release_run_id: Current release run to exclude.
            limit: Maximum number of previous releases to load.

        Returns:
            Validated previous release-risk responses ordered newest first.

        Raises:
            AgentQueryContextResolverError: When repository access fails.
            AgentQuerySnapshotValidationError: When persisted data is invalid.
        """
        try:
            snapshots = (
                await self._snapshot_repository
                .list_latest_previous_release_snapshots(
                    exclude_release_run_id=exclude_release_run_id,
                    limit=limit,
                )
            )
        except ReleaseRunRiskSnapshotRepositoryError as exc:
            raise AgentQueryContextResolverError(
                "Failed to load persisted historical risk context."
            ) from exc

        historical_release_risks: list[ReleaseRunRiskResponse] = []

        for snapshot in snapshots:
            if snapshot.release_run_id == exclude_release_run_id:
                raise AgentQuerySnapshotValidationError(
                    "Historical snapshot unexpectedly belongs to the "
                    "current release run."
                )

            historical_release_risks.append(
                self._validate_snapshot(
                    snapshot,
                    expected_release_run_id=snapshot.release_run_id,
                )
            )

        logger.info(
            "agent_query_historical_context_resolved",
            extra={
                "run_id": self._request_id,
                "exclude_release_run_id": str(exclude_release_run_id),
                "limit": limit,
                "historical_release_count": len(
                    historical_release_risks
                ),
            },
        )

        return historical_release_risks

    @staticmethod
    def _validate_snapshot(
        snapshot: RiskSnapshotRecordProtocol,
        *,
        expected_release_run_id: UUID,
    ) -> ReleaseRunRiskResponse:
        """Validate one persisted snapshot against its database ownership."""
        if snapshot.release_run_id != expected_release_run_id:
            raise AgentQuerySnapshotValidationError(
                "Snapshot release-run context is inconsistent."
            )

        try:
            payload = json.loads(snapshot.risk_payload_json)
        except json.JSONDecodeError as exc:
            raise AgentQuerySnapshotValidationError(
                "Persisted release-risk snapshot contains invalid JSON."
            ) from exc

        try:
            release_risk = ReleaseRunRiskResponse.model_validate(payload)
        except ValidationError as exc:
            raise AgentQuerySnapshotValidationError(
                "Persisted release-risk snapshot failed validation."
            ) from exc

        if release_risk.release_run.id != expected_release_run_id:
            raise AgentQuerySnapshotValidationError(
                "Snapshot payload belongs to a different release run."
            )

        return release_risk

    @staticmethod
    def _resolve_release_run_id(
        request: AgentQueryRequest,
        plan: AgentQueryPlan,
    ) -> UUID:
        """Resolve one consistent release-run ID from request and plan."""

        if (
            request.release_run_id is not None
            and plan.release_run_id is not None
            and request.release_run_id != plan.release_run_id
        ):
            raise AgentQueryContextConflictError(
                "Request and query plan release-run IDs do not match."
            )

        release_run_id = plan.release_run_id or request.release_run_id

        if release_run_id is None:
            raise AgentQueryContextRequiredError(
                "A release-run ID is required for this follow-up query."
            )

        return release_run_id
