from datetime import UTC, datetime

from fastapi import APIRouter

from app.schemas.health import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return the current health status of the AgentFlow backend."""
    return HealthResponse(
        status="ok",
        service="AgentFlow AI Backend",
        timestamp=datetime.now(UTC),
    )