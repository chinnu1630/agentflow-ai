from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReleaseRunStatus(StrEnum):
    """Allowed lifecycle statuses for a release risk analysis run."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReleaseRunCreate(BaseModel):
    """Request schema for creating a new release risk analysis run."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Manager question for release risk analysis.",
        examples=["What are the biggest release risks this week?"],
    )

    requested_by: str = Field(
        ...,
        min_length=3,
        max_length=255,
        description="User or service account that requested the analysis.",
        examples=["engineering.manager@company.com"],
    )


class ReleaseRunRead(BaseModel):
    """Response schema returned after creating or reading a release run."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: str
    query: str
    requested_by: str
    status: ReleaseRunStatus
    created_at: datetime
    completed_at: datetime | None = None
    