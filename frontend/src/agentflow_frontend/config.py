"""Environment-based configuration for the AgentFlow manager frontend."""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """Validated runtime settings for the Streamlit frontend.

    The backend URL must be supplied through the environment so local,
    container, staging, and production deployments can use different hosts
    without changing application code.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTFLOW_FRONTEND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend_base_url: AnyHttpUrl
    auth_required: bool = Field(
        default=True,
        description=(
            "Require a signed bearer token for frontend API requests. "
            "Disable only when using the backend's validated local "
            "development authentication mode."
        ),
    )
    demo_access_token: SecretStr | None = Field(
        default=None,
        description=(
            "Optional server-side read-only JWT for the public portfolio "
            "demo. This secret must never be rendered in the browser."
        ),
    )
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)

    @field_validator("demo_access_token", mode="before")
    @classmethod
    def normalize_blank_demo_access_token(cls, value: object) -> object:
        """Treat an empty environment value as an unconfigured demo token."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def demo_mode(self) -> bool:
        """Return whether server-side read-only portfolio access is enabled."""
        return self.demo_access_token is not None

    def build_api_url(self, path: str) -> str:
        """Build a backend API URL from a trusted relative path.

        Args:
            path: Absolute API path beginning with one slash.

        Returns:
            Fully qualified backend API URL.

        Raises:
            ValueError: If the path is empty, malformed, or an external URL.
        """
        parsed_path = urlsplit(path)

        if (
            not path.startswith("/")
            or path.startswith("//")
            or parsed_path.scheme
            or parsed_path.netloc
        ):
            raise ValueError("API path must be an absolute path without a hostname.")

        return f"{str(self.backend_base_url).rstrip('/')}{path}"


@lru_cache(maxsize=1)
def get_frontend_settings() -> FrontendSettings:
    """Load and cache validated frontend runtime settings."""
    return FrontendSettings()
