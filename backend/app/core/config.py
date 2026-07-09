from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = Field(
        default="AgentFlow AI Backend",
        description="Name of the FastAPI application.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Current application version.",
    )
    environment: str = Field(
        default="local",
        description="Application runtime environment.",
    )
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="Base prefix for version 1 API routes.",
    )
    otel_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing instrumentation.",
    )
    otel_service_name: str = Field(
        default="agentflow-ai-backend",
        description="Service name shown in distributed traces.",
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description="Optional OTLP HTTP endpoint for exporting traces.",
    )
    otel_sample_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling ratio between 0.0 and 1.0.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()