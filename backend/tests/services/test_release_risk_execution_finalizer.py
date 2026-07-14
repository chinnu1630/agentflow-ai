"""Unit tests for release-risk execution finalization."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.schemas.risk import ReleaseRunRiskResponse
from app.services.release_risk_execution_finalizer import (
    ReleaseRiskExecutionFinalizer,
)


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for async finalizer tests."""

    return "asyncio"


def build_finalizer() -> ReleaseRiskExecutionFinalizer:
    """Build a finalizer with mocked repositories."""

    return ReleaseRiskExecutionFinalizer(
        release_run_repository=Mock(),
        approval_repository=Mock(),
        event_repository=Mock(),
        risk_snapshot_repository=Mock(),
    )


@pytest.mark.anyio
async def test_finalize_runs_all_persistence_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalizer should audit, apply approval state, and persist a snapshot."""

    release_run_id = uuid4()
    response = cast(
        ReleaseRunRiskResponse,
        Mock(spec=ReleaseRunRiskResponse),
    )
    finalized_response = cast(
        ReleaseRunRiskResponse,
        Mock(spec=ReleaseRunRiskResponse),
    )

    finalizer = build_finalizer()

    record_scoring = AsyncMock()
    ensure_approval = AsyncMock(return_value=finalized_response)
    persist_snapshot = AsyncMock()

    monkeypatch.setattr(
        finalizer,
        "_record_scoring_audit_events",
        record_scoring,
    )
    monkeypatch.setattr(
        finalizer,
        "_ensure_pending_approval_request",
        ensure_approval,
    )
    monkeypatch.setattr(
        finalizer,
        "_persist_release_risk_snapshot",
        persist_snapshot,
    )

    result = await finalizer.finalize(
        release_run_id=release_run_id,
        response=response,
    )

    assert result is finalized_response

    record_scoring.assert_awaited_once_with(
        release_run_id=release_run_id,
        response=response,
    )
    ensure_approval.assert_awaited_once_with(
        release_run_id=release_run_id,
        response=response,
    )
    persist_snapshot.assert_awaited_once_with(
        release_run_id=release_run_id,
        response=finalized_response,
    )


@pytest.mark.anyio
async def test_finalize_stops_when_audit_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalizer should not continue after an audit persistence failure."""

    release_run_id = uuid4()
    response = cast(
        ReleaseRunRiskResponse,
        Mock(spec=ReleaseRunRiskResponse),
    )
    finalizer = build_finalizer()

    record_scoring = AsyncMock(side_effect=RuntimeError("audit persistence failed"))
    ensure_approval = AsyncMock()
    persist_snapshot = AsyncMock()

    monkeypatch.setattr(
        finalizer,
        "_record_scoring_audit_events",
        record_scoring,
    )
    monkeypatch.setattr(
        finalizer,
        "_ensure_pending_approval_request",
        ensure_approval,
    )
    monkeypatch.setattr(
        finalizer,
        "_persist_release_risk_snapshot",
        persist_snapshot,
    )

    with pytest.raises(
        RuntimeError,
        match="audit persistence failed",
    ):
        await finalizer.finalize(
            release_run_id=release_run_id,
            response=response,
        )

    ensure_approval.assert_not_awaited()
    persist_snapshot.assert_not_awaited()
