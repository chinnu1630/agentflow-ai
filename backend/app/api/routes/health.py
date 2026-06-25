from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response returned by the health check endpoint."""

    status: str = Field(description="Current service status")
    service: str = Field(description="Name of the running service")
    timestamp: datetime = Field(description="Current server timestamp")


router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return backend health status."""
    return HealthResponse(
        status="ok",
        service="AgentFlow AI Backend",
        timestamp=datetime.now(UTC),
    )