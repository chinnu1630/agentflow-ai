from decimal import Decimal
from functools import lru_cache
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
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
    cors_allowed_origins: tuple[str, ...] = Field(
        default=(),
        description=(
            "Explicit browser origins permitted to call the AgentFlow API."
        ),
    )
    trusted_hosts: tuple[str, ...] = Field(
        default=("localhost", "127.0.0.1", "testserver"),
        min_length=1,
        description="Host headers accepted by the AgentFlow API.",
    )
    max_request_body_bytes: int = Field(
        default=1_048_576,
        ge=1,
        le=104_857_600,
        description="Maximum accepted HTTP request-body size in bytes.",
    )
    rate_limit_enabled: bool = Field(
        default=False,
        description=(
            "Enable distributed Redis-backed API rate limiting."
        ),
    )
    redis_url: SecretStr | None = Field(
        default=None,
        description=(
            "Redis connection URL used only for distributed rate limiting."
        ),
    )
    redis_max_connections: int = Field(
        default=20,
        ge=1,
        le=1_000,
        description="Maximum Redis connections available to one API process.",
    )
    redis_connect_timeout_seconds: float = Field(
        default=1.0,
        gt=0.0,
        le=30.0,
        description="Maximum time allowed to establish a Redis connection.",
    )
    redis_socket_timeout_seconds: float = Field(
        default=1.0,
        gt=0.0,
        le=30.0,
        description="Maximum time allowed for one Redis operation.",
    )
    rate_limit_key_hmac_secret: SecretStr | None = Field(
        default=None,
        description=(
            "Secret used to derive pseudonymous Redis identity keys."
        ),
    )
    rate_limit_standard_capacity: int = Field(
        default=60,
        ge=1,
        le=100_000,
        description="Maximum tokens in the standard API token bucket.",
    )
    rate_limit_standard_refill_rate_per_second: float = Field(
        default=1.0,
        gt=0.0,
        le=10_000.0,
        description="Tokens added per second to the standard API bucket.",
    )
    rate_limit_expensive_capacity: int = Field(
        default=5,
        ge=1,
        le=10_000,
        description="Maximum tokens in the expensive workflow bucket.",
    )
    rate_limit_expensive_refill_rate_per_second: float = Field(
        default=0.1,
        gt=0.0,
        le=1_000.0,
        description="Tokens added per second to the expensive workflow bucket.",
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
    auth_enabled: bool = Field(
        default=False,
        description=(
            "Enable verification of externally issued JWT access tokens."
        ),
    )
    auth_jwt_algorithm: Literal["RS256"] = Field(
        default="RS256",
        description=(
            "Asymmetric JWT signature algorithm accepted by AgentFlow."
        ),
    )
    auth_jwt_issuer: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
        description="Trusted identity-provider JWT issuer.",
    )
    auth_jwt_audience: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Expected AgentFlow API audience claim.",
    )
    auth_jwt_public_key: SecretStr | None = Field(
        default=None,
        description=(
            "PEM public key used only to verify identity-provider JWTs."
        ),
    )
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
    agent_dynamic_planning_enabled: bool = Field(
        default=False,
        description=(
            "Enable bounded Claude execution planning for agent queries."
        ),
    )
    agent_dynamic_planner_model: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional lower-cost Claude model used only for dynamic planning."
        ),
    )
    agent_dynamic_planner_max_tokens: int = Field(
        default=1_024,
        ge=256,
        le=4_096,
        description="Maximum output-token budget for dynamic planning.",
    )
    agent_dynamic_synthesis_model: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional Claude model used only for dynamic answer synthesis."
        ),
    )
    agent_dynamic_synthesis_max_tokens: int = Field(
        default=2_048,
        ge=256,
        le=8_192,
        description="Maximum output-token budget for dynamic synthesis.",
    )
    agent_dynamic_planner_input_cost_per_million_usd: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Configured planner input-token price per million in USD.",
    )
    agent_dynamic_planner_output_cost_per_million_usd: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Configured planner output-token price per million in USD.",
    )
    agent_dynamic_synthesis_input_cost_per_million_usd: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Configured synthesis input-token price per million in USD.",
    )
    agent_dynamic_synthesis_output_cost_per_million_usd: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Configured synthesis output-token price per million in USD.",
    )
    agent_dynamic_max_estimated_cost_usd: Decimal | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional fail-closed maximum estimated USD cost for one "
            "dynamic query."
        ),
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
    otel_metrics_enabled: bool = Field(
        default=False,
        description="Enable periodic OpenTelemetry metrics export.",
    )
    otel_metrics_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description=(
            "Explicit OTLP HTTP endpoint for exporting metrics."
        ),
    )
    otel_metrics_export_interval_milliseconds: int = Field(
        default=60_000,
        ge=1_000,
        le=300_000,
        description=(
            "Interval in milliseconds between periodic metric exports."
        ),
    )

    @field_validator(
        "cors_allowed_origins",
        "trusted_hosts",
    )
    @classmethod
    def validate_http_boundary_entries(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize HTTP boundary entries and reject unsafe wildcards."""
        normalized_values = tuple(
            value.strip() for value in values if value.strip()
        )

        if "*" in normalized_values:
            raise ValueError(
                "Wildcard HTTP boundary entries are not permitted."
            )

        if len(normalized_values) != len(values):
            raise ValueError(
                "HTTP boundary entries must not be blank."
            )

        return normalized_values

    @field_validator(
        "otel_metrics_exporter_otlp_endpoint",
    )
    @classmethod
    def validate_otel_metrics_endpoint(
        cls,
        value: str | None,
    ) -> str | None:
        """Accept only explicit HTTP(S) OTLP metrics endpoints."""
        if value is None:
            return None

        normalized_value = value.strip()

        if not normalized_value:
            return None

        parsed_url = urlsplit(normalized_value)

        if (
            parsed_url.scheme not in {"http", "https"}
            or parsed_url.hostname is None
        ):
            raise ValueError(
                "OTEL_METRICS_EXPORTER_OTLP_ENDPOINT must use "
                "http:// or https:// with a hostname."
            )

        return normalized_value

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        """Accept only network Redis URLs with an explicit hostname."""
        if value is None:
            return None

        raw_url = value.get_secret_value().strip()
        parsed_url = urlsplit(raw_url)

        if (
            parsed_url.scheme not in {"redis", "rediss"}
            or parsed_url.hostname is None
        ):
            raise ValueError(
                "REDIS_URL must use redis:// or rediss:// with a hostname."
            )

        return SecretStr(raw_url)

    @field_validator("rate_limit_key_hmac_secret")
    @classmethod
    def validate_rate_limit_hmac_secret(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        """Reject blank or weak rate-limit identity-key secrets."""
        if value is None:
            return None

        secret_value = value.get_secret_value().strip()

        if len(secret_value) < 16:
            raise ValueError(
                "RATE_LIMIT_KEY_HMAC_SECRET must contain at least 16 characters."
            )

        return SecretStr(secret_value)

    @model_validator(mode="after")
    def validate_authentication_configuration(self) -> Self:
        """Enforce fail-closed authentication configuration.

        Local and test environments may explicitly run without authentication
        for developer productivity. Every deployed environment must enable
        authentication, and enabled authentication must include all values
        needed for deterministic JWT verification.
        """
        environment = self.environment.strip().lower()
        local_environments = {
            "local",
            "test",
            "testing",
            "development",
            "dev",
        }
        local_only_trusted_hosts = {
            "localhost",
            "127.0.0.1",
            "testserver",
        }

        if (
            environment not in local_environments
            and set(self.trusted_hosts).issubset(local_only_trusted_hosts)
        ):
            raise ValueError(
                "Deployed environments require explicit trusted hosts."
            )

        if not self.auth_enabled:
            if environment not in local_environments:
                raise ValueError(
                    "Authentication must be enabled outside local and test "
                    "environments."
                )
            return self

        missing_fields: list[str] = []

        if not self.auth_jwt_issuer or not self.auth_jwt_issuer.strip():
            missing_fields.append("AUTH_JWT_ISSUER")

        if not self.auth_jwt_audience or not self.auth_jwt_audience.strip():
            missing_fields.append("AUTH_JWT_AUDIENCE")

        if (
            self.auth_jwt_public_key is None
            or not self.auth_jwt_public_key.get_secret_value().strip()
        ):
            missing_fields.append("AUTH_JWT_PUBLIC_KEY")

        if missing_fields:
            missing_names = ", ".join(missing_fields)
            raise ValueError(
                "Enabled authentication requires: "
                f"{missing_names}."
            )

        return self

    @model_validator(mode="after")
    def validate_otel_metrics_configuration(self) -> Self:
        """Require an explicit exporter endpoint when metrics are enabled."""
        if (
            self.otel_metrics_enabled
            and self.otel_metrics_exporter_otlp_endpoint is None
        ):
            raise ValueError(
                "Enabled OpenTelemetry metrics require: "
                "OTEL_METRICS_EXPORTER_OTLP_ENDPOINT."
            )

        return self

    @model_validator(mode="after")
    def validate_rate_limit_configuration(self) -> Self:
        """Enforce distributed rate limiting in deployed environments."""
        environment = self.environment.strip().lower()
        local_environments = {
            "local",
            "test",
            "testing",
            "development",
            "dev",
        }

        if not self.rate_limit_enabled:
            if environment not in local_environments:
                raise ValueError(
                    "Rate limiting must be enabled outside local and test "
                    "environments."
                )
            return self

        missing_fields: list[str] = []

        if self.redis_url is None:
            missing_fields.append("REDIS_URL")

        if self.rate_limit_key_hmac_secret is None:
            missing_fields.append("RATE_LIMIT_KEY_HMAC_SECRET")

        if missing_fields:
            raise ValueError(
                "Enabled rate limiting requires: "
                f"{', '.join(missing_fields)}."
            )

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
