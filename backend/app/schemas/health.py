from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = Field(description="Current API health status")
    service: str = Field(description="Service name")
    timestamp: datetime = Field(description="Current UTC timestamp")