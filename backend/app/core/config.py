from functools import lru_cache

from pydantic import Field, SecretStr
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
    database_url: str | None = Field(
        default=None,
        description="Async SQLAlchemy database connection URL.",
    )
    github_repository_owner: str | None = None
    github_repository_name: str | None = None
    github_default_branch: str = "main"
    github_token: SecretStr | None = None
    jira_base_url: str | None = None
    jira_email: str | None = None
    jira_api_token: SecretStr | None = None
    jira_project_key: str | None = None
    slack_bot_token: SecretStr | None = None
    slack_channel_id: str | None = None
    anthropic_enabled: bool = Field(
        default=False,
        description="Enable Claude-based structured release-risk synthesis.",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key loaded only from environment variables.",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-5",
        min_length=1,
        description="Claude model used for structured release-risk synthesis.",
    )
    anthropic_max_tokens: int = Field(
        default=4_096,
        ge=256,
        le=8_192,
        description="Maximum Claude output-token budget for one synthesis call.",
    )
    anthropic_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Timeout for one Claude API request.",
    )
    anthropic_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retries for transient Claude API failures.",
    )
    knowledge_embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
        description="Local Sentence Transformer model used for semantic retrieval.",
    )
    knowledge_embedding_dimension: int = Field(
        default=384,
        ge=384,
        le=384,
        description=(
            "Vector dimension fixed by the engineering-document vector(384) schema."
        ),
    )
    knowledge_reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        min_length=1,
        description="Local cross-encoder model used to rerank hybrid candidates.",
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
