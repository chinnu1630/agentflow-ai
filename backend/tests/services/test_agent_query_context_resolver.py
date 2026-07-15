"""Unit tests for persisted AgentFlow query context resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from app.repositories.release_run_risk_snapshot_repository import (
    ReleaseRunRiskSnapshotRepositoryError,
)
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
    ResponseDepth,
)
from app.services.agent_query_context_resolver import (
    AgentQueryContextConflictError,
    AgentQueryContextRequiredError,
    AgentQueryContextResolver,
    AgentQueryContextResolverError,
    AgentQuerySnapshotNotFoundError,
    AgentQuerySnapshotValidationError,
)
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


@dataclass(frozen=True)
class FakeSnapshot:
    """Minimal persisted snapshot used by resolver tests."""

    id: UUID
    release_run_id: UUID
    snapshot_version: int
    risk_payload_json: str


class FakeSnapshotRepository:
    """Fake snapshot repository with configurable behavior."""

    def __init__(
        self,
        snapshot: FakeSnapshot | None = None,
        historical_snapshots: list[FakeSnapshot] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Initialize the fake repository."""

        self.snapshot = snapshot
        self.historical_snapshots = historical_snapshots or []
        self.error = error
        self.requested_release_run_id: UUID | None = None
        self.excluded_release_run_id: UUID | None = None
        self.historical_limit: int | None = None

    async def get_latest_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> FakeSnapshot | None:
        """Return the configured latest snapshot."""

        self.requested_release_run_id = release_run_id

        if self.error is not None:
            raise self.error

        return self.snapshot

    async def list_latest_previous_release_snapshots(
        self,
        *,
        exclude_release_run_id: UUID,
        limit: int = 10,
    ) -> list[FakeSnapshot]:
        """Return configured historical snapshots."""

        self.excluded_release_run_id = exclude_release_run_id
        self.historical_limit = limit

        if self.error is not None:
            raise self.error

        return self.historical_snapshots


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for async resolver tests."""

    return "asyncio"


def build_plan(
    release_run_id: UUID | None,
) -> AgentQueryPlan:
    """Build a follow-up query plan."""

    return AgentQueryPlan(
        intent=AgentIntent.EXPLAIN_RISK_SCORE,
        response_depth=ResponseDepth.DEEP,
        confidence=1.0,
        release_run_id=release_run_id,
        requires_current_snapshot=True,
        routing_reason_code="explain_risk_score",
    )


def build_snapshot(release_run_id: UUID) -> FakeSnapshot:
    """Build a valid persisted release-risk snapshot."""

    payload = build_snapshot_payload(
        release_run_id=release_run_id,
        approval_request_id=uuid4(),
    )

    return FakeSnapshot(
        id=uuid4(),
        release_run_id=release_run_id,
        snapshot_version=3,
        risk_payload_json=json.dumps(payload),
    )


@pytest.mark.anyio
async def test_resolves_latest_persisted_snapshot() -> None:
    """Resolver should return validated trusted context."""

    release_run_id = uuid4()
    snapshot = build_snapshot(release_run_id)
    repository = FakeSnapshotRepository(snapshot=snapshot)
    resolver = AgentQueryContextResolver(
        snapshot_repository=repository,
        request_id="request-123",
    )

    context = await resolver.resolve(
        AgentQueryRequest(
            query="Why is the risk score high?",
            release_run_id=release_run_id,
        ),
        build_plan(release_run_id),
    )

    assert repository.requested_release_run_id == release_run_id
    assert context.release_run_id == release_run_id
    assert context.snapshot_id == snapshot.id
    assert context.snapshot_version == 3
    assert context.release_risk.release_run.id == release_run_id
    assert context.release_risk.risk_score is not None


@pytest.mark.anyio
async def test_requires_release_run_context() -> None:
    """Follow-up queries must identify a release run."""

    repository = FakeSnapshotRepository()
    resolver = AgentQueryContextResolver(
        snapshot_repository=repository,
        request_id="request-123",
    )

    with pytest.raises(
        AgentQueryContextRequiredError,
        match="release-run ID is required",
    ):
        await resolver.resolve(
            AgentQueryRequest(query="Why is the risk score high?"),
            build_plan(None),
        )


@pytest.mark.anyio
async def test_rejects_conflicting_release_run_ids() -> None:
    """Request and plan context IDs must match."""

    resolver = AgentQueryContextResolver(
        snapshot_repository=FakeSnapshotRepository(),
        request_id="request-123",
    )

    with pytest.raises(
        AgentQueryContextConflictError,
        match="do not match",
    ):
        await resolver.resolve(
            AgentQueryRequest(
                query="Explain this risk.",
                release_run_id=uuid4(),
            ),
            build_plan(uuid4()),
        )


@pytest.mark.anyio
async def test_raises_when_snapshot_is_missing() -> None:
    """Resolver should report when no persisted snapshot exists."""

    release_run_id = uuid4()
    resolver = AgentQueryContextResolver(
        snapshot_repository=FakeSnapshotRepository(snapshot=None),
        request_id="request-123",
    )

    with pytest.raises(
        AgentQuerySnapshotNotFoundError,
        match="No persisted release-risk snapshot",
    ):
        await resolver.resolve(
            AgentQueryRequest(
                query="Explain this risk.",
                release_run_id=release_run_id,
            ),
            build_plan(release_run_id),
        )


@pytest.mark.anyio
async def test_rejects_invalid_snapshot_json() -> None:
    """Invalid persisted JSON must not enter the agent response."""

    release_run_id = uuid4()
    snapshot = FakeSnapshot(
        id=uuid4(),
        release_run_id=release_run_id,
        snapshot_version=1,
        risk_payload_json="{invalid-json",
    )
    resolver = AgentQueryContextResolver(
        snapshot_repository=FakeSnapshotRepository(snapshot=snapshot),
        request_id="request-123",
    )

    with pytest.raises(
        AgentQuerySnapshotValidationError,
        match="invalid JSON",
    ):
        await resolver.resolve(
            AgentQueryRequest(
                query="Explain this risk.",
                release_run_id=release_run_id,
            ),
            build_plan(release_run_id),
        )


@pytest.mark.anyio
async def test_rejects_invalid_snapshot_schema() -> None:
    """Snapshot JSON must satisfy the public risk-response contract."""

    release_run_id = uuid4()
    snapshot = FakeSnapshot(
        id=uuid4(),
        release_run_id=release_run_id,
        snapshot_version=1,
        risk_payload_json=json.dumps({"unexpected": "payload"}),
    )
    resolver = AgentQueryContextResolver(
        snapshot_repository=FakeSnapshotRepository(snapshot=snapshot),
        request_id="request-123",
    )

    with pytest.raises(
        AgentQuerySnapshotValidationError,
        match="failed validation",
    ):
        await resolver.resolve(
            AgentQueryRequest(
                query="Explain this risk.",
                release_run_id=release_run_id,
            ),
            build_plan(release_run_id),
        )


@pytest.mark.anyio
async def test_rejects_snapshot_record_context_mismatch() -> None:
    """Snapshot database ownership must match the requested release run."""

    requested_release_run_id = uuid4()
    snapshot = build_snapshot(uuid4())
    resolver = AgentQueryContextResolver(
        snapshot_repository=FakeSnapshotRepository(snapshot=snapshot),
        request_id="request-123",
    )

    with pytest.raises(
        AgentQuerySnapshotValidationError,
        match="context is inconsistent",
    ):
        await resolver.resolve(
            AgentQueryRequest(
                query="Explain this risk.",
                release_run_id=requested_release_run_id,
            ),
            build_plan(requested_release_run_id),
        )


@pytest.mark.anyio
async def test_rejects_snapshot_payload_context_mismatch() -> None:
    """Snapshot payload ownership must match its database record."""

    release_run_id = uuid4()
    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    snapshot = FakeSnapshot(
        id=uuid4(),
        release_run_id=release_run_id,
        snapshot_version=1,
        risk_payload_json=json.dumps(payload),
    )
    resolver = AgentQueryContextResolver(
        snapshot_repository=FakeSnapshotRepository(snapshot=snapshot),
        request_id="request-123",
    )

    with pytest.raises(
        AgentQuerySnapshotValidationError,
        match="different release run",
    ):
        await resolver.resolve(
            AgentQueryRequest(
                query="Explain this risk.",
                release_run_id=release_run_id,
            ),
            build_plan(release_run_id),
        )


@pytest.mark.anyio
async def test_wraps_snapshot_repository_failure() -> None:
    """Repository failures should become resolver-layer errors."""

    release_run_id = uuid4()
    resolver = AgentQueryContextResolver(
        snapshot_repository=FakeSnapshotRepository(
            error=ReleaseRunRiskSnapshotRepositoryError("database unavailable")
        ),
        request_id="request-123",
    )

    with pytest.raises(
        AgentQueryContextResolverError,
        match="Failed to load persisted agent query context",
    ):
        await resolver.resolve(
            AgentQueryRequest(
                query="Explain this risk.",
                release_run_id=release_run_id,
            ),
            build_plan(release_run_id),
        )

@pytest.mark.anyio
async def test_resolves_validated_historical_snapshots() -> None:
    """Resolver should validate previous persisted release snapshots."""

    current_release_run_id = uuid4()
    previous_release_run_id = uuid4()
    previous_snapshot = build_snapshot(previous_release_run_id)
    repository = FakeSnapshotRepository(
        historical_snapshots=[previous_snapshot],
    )
    resolver = AgentQueryContextResolver(
        snapshot_repository=repository,
        request_id="request-123",
    )

    historical_release_risks = await resolver.resolve_historical_release_risks(
        exclude_release_run_id=current_release_run_id,
        limit=5,
    )

    assert repository.excluded_release_run_id == current_release_run_id
    assert repository.historical_limit == 5
    assert len(historical_release_risks) == 1
    assert (
        historical_release_risks[0].release_run.id
        == previous_release_run_id
    )
