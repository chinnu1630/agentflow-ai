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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()