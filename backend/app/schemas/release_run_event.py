"""API schemas for release-run audit events."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ReleaseRunEventResponse(BaseModel):
    """API response model for one release-run audit event."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    release_run_id: UUID
    event_type: str
    event_status: str
    message: str
    metadata_json: dict[str, Any]
    created_at: datetime


class ReleaseRunEventListResponse(BaseModel):
    """API response model for a release-run audit event timeline."""

    release_run_id: UUID
    events: list[ReleaseRunEventResponse]
