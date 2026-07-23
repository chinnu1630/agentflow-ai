from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.release_run import (
    ReleaseRunCreate,
    ReleaseRunRead,
    ReleaseRunStatus,
)


def test_release_run_create_accepts_valid_payload() -> None:
    """ReleaseRunCreate should accept a valid manager release-risk query."""
    payload = ReleaseRunCreate(
        query="What are the biggest release risks this week?",
    )

    assert payload.query == "What are the biggest release risks this week?"


def test_release_run_create_strips_whitespace() -> None:
    """ReleaseRunCreate should strip leading and trailing query whitespace."""
    payload = ReleaseRunCreate(
        query="   What are the biggest release risks this week?   ",
    )

    assert payload.query == "What are the biggest release risks this week?"


def test_release_run_create_rejects_spoofed_requester() -> None:
    """Authenticated identity must not be accepted from request JSON."""
    with pytest.raises(ValidationError):
        ReleaseRunCreate(
            query="What are the biggest release risks this week?",
            requested_by="attacker@example.com",
        )


def test_release_run_create_rejects_short_query() -> None:
    """ReleaseRunCreate should reject queries that are too short to be useful."""
    with pytest.raises(ValidationError):
        ReleaseRunCreate(query="Risk")


def test_release_run_status_rejects_invalid_value() -> None:
    """ReleaseRunStatus should only allow known lifecycle states."""
    with pytest.raises(ValueError):
        ReleaseRunStatus("done")


def test_release_run_read_accepts_valid_response_data() -> None:
    """ReleaseRunRead should accept valid release run response data."""
    release_run_id = uuid4()
    created_at = datetime.now(UTC)

    response = ReleaseRunRead(
        id=release_run_id,
        run_id="run_20260630_001",
        query="What are the biggest release risks this week?",
        requested_by="engineering.manager@company.com",
        status=ReleaseRunStatus.CREATED,
        created_at=created_at,
        completed_at=None,
    )

    assert response.id == release_run_id
    assert response.run_id == "run_20260630_001"
    assert response.status == ReleaseRunStatus.CREATED
    assert response.completed_at is None