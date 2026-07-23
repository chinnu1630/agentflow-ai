"""API schemas for release-run HITL approvals."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReleaseRunApprovalDecisionStatus(StrEnum):
    """Allowed terminal approval decisions from the API."""

    APPROVED = "approved"
    REJECTED = "rejected"


class ReleaseRunApprovalResponse(BaseModel):
    """API response model for one release-run approval request."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    release_run_id: UUID
    approval_status: str
    approval_reason: str
    approval_policy_version: str
    requested_by: str | None = None
    decided_by: str | None = None
    decision_note: str | None = None
    created_at: datetime
    decided_at: datetime | None = None


class ReleaseRunApprovalListResponse(BaseModel):
    """API response model for release-run approval history."""

    release_run_id: UUID
    approvals: list[ReleaseRunApprovalResponse]


class ReleaseRunApprovalDecisionRequest(BaseModel):
    """API request model for approving or rejecting a release approval."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    approval_status: ReleaseRunApprovalDecisionStatus
    decision_note: str | None = Field(default=None, max_length=1_000)


class PendingReleaseRunApprovalListResponse(BaseModel):
    """API response model for manager pending approval queue."""

    approval_status: str
    approvals: list[ReleaseRunApprovalResponse]
