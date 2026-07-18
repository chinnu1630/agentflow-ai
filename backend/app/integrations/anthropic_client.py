"""Async Anthropic client for structured release-risk synthesis."""

from __future__ import annotations

import time

import anthropic
import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.schemas.llm_risk_synthesis import ClaudeReleaseRiskReport

logger = structlog.get_logger(__name__)


class AnthropicClientConfig(BaseModel):
    """Validated configuration for Claude risk synthesis."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    api_key: SecretStr
    model: str = Field(min_length=1, max_length=255)
    max_tokens: int = Field(ge=256, le=8_192)
    timeout_seconds: float = Field(ge=1.0, le=120.0)
    max_retries: int = Field(ge=0, le=5)


class ClaudeSynthesisResult(BaseModel):
    """Safe result and usage metadata returned from Claude synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report: ClaudeReleaseRiskReport
    message_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    stop_reason: str | None = None
    duration_ms: float = Field(ge=0.0)
    prompt_version: str = Field(min_length=1, max_length=100)


class AnthropicClientError(RuntimeError):
    """Base error raised when Claude synthesis cannot be completed."""


class AnthropicClientTimeoutError(AnthropicClientError):
    """Raised when Claude synthesis exceeds its configured timeout."""


class AnthropicClientRateLimitError(AnthropicClientError):
    """Raised when Anthropic rejects a request because of rate limits."""


class AnthropicClientUnavailableError(AnthropicClientError):
    """Raised when the Anthropic API is temporarily unavailable."""


class AnthropicClientResponseError(AnthropicClientError):
    """Raised when Claude returns an unusable structured response."""


class AnthropicRiskSynthesisClient:
    """Small async Claude client with structured-output enforcement."""

    def __init__(
        self,
        *,
        config: AnthropicClientConfig,
        run_id: str,
        client: AsyncAnthropic | None = None,
    ) -> None:
        """Initialize the Claude synthesis client.

        Args:
            config: Validated Anthropic connection and request limits.
            run_id: AgentFlow workflow correlation identifier.
            client: Optional injected Anthropic client for testing.
        """
        self._config = config
        self._run_id = run_id
        self._owns_client = client is None
        self._client = client or AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    async def __aenter__(self) -> AnthropicRiskSynthesisClient:
        """Enter the async client context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the internally created Anthropic client."""
        if self._owns_client:
            await self._client.close()

    async def synthesize_release_risk(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeSynthesisResult:
        """Generate one validated, evidence-grounded release-risk report.

        Prompt contents are deliberately excluded from logs because they may
        contain confidential GitHub, Jira, or engineering-document evidence.

        Args:
            system_prompt: Trusted AgentFlow synthesis instructions.
            user_prompt: Sanitized release evidence for Claude.
            prompt_version: Version identifier for LLMOps traceability.

        Returns:
            Validated Claude report with safe usage metadata.

        Raises:
            AnthropicClientError: If the API call or response validation fails.
        """
        self._validate_prompt(
            prompt=system_prompt,
            prompt_name="system_prompt",
        )
        self._validate_prompt(
            prompt=user_prompt,
            prompt_name="user_prompt",
        )
        self._validate_prompt(
            prompt=prompt_version,
            prompt_name="prompt_version",
        )

        started_at = time.perf_counter()

        logger.info(
            "claude_risk_synthesis_started",
            run_id=self._run_id,
            model=self._config.model,
            prompt_version=prompt_version,
            system_prompt_length=len(system_prompt),
            user_prompt_length=len(user_prompt),
            max_tokens=self._config.max_tokens,
        )

        try:
            response = await self._client.messages.parse(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                output_format=ClaudeReleaseRiskReport,
            )
        except anthropic.APITimeoutError as exc:
            self._log_failure(
                event_name="claude_risk_synthesis_timeout",
                error=exc,
                prompt_version=prompt_version,
            )
            raise AnthropicClientTimeoutError(
                "Claude risk synthesis timed out."
            ) from exc
        except anthropic.RateLimitError as exc:
            self._log_failure(
                event_name="claude_risk_synthesis_rate_limited",
                error=exc,
                prompt_version=prompt_version,
            )
            raise AnthropicClientRateLimitError(
                "Claude risk synthesis was rate limited."
            ) from exc
        except anthropic.APIConnectionError as exc:
            self._log_failure(
                event_name="claude_risk_synthesis_connection_failed",
                error=exc,
                prompt_version=prompt_version,
            )
            raise AnthropicClientUnavailableError(
                "Claude risk synthesis service is unavailable."
            ) from exc
        except anthropic.APIStatusError as exc:
            self._log_failure(
                event_name="claude_risk_synthesis_api_failed",
                error=exc,
                prompt_version=prompt_version,
            )
            raise AnthropicClientUnavailableError(
                "Claude risk synthesis API request failed."
            ) from exc
        except (anthropic.APIError, ValidationError) as exc:
            self._log_failure(
                event_name="claude_risk_synthesis_response_failed",
                error=exc,
                prompt_version=prompt_version,
            )
            raise AnthropicClientResponseError(
                "Claude risk synthesis response validation failed."
            ) from exc

        if response.stop_reason == "max_tokens":
            raise AnthropicClientResponseError(
                "Claude risk synthesis exceeded the output-token limit."
            )

        report = response.parsed_output

        if report is None:
            raise AnthropicClientResponseError(
                "Claude did not return a parsed release-risk report."
            )

        duration_ms = round(
            (time.perf_counter() - started_at) * 1_000,
            2,
        )

        result = ClaudeSynthesisResult(
            report=report,
            message_id=response.id,
            model=str(response.model),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            duration_ms=duration_ms,
            prompt_version=prompt_version,
        )

        logger.info(
            "claude_risk_synthesis_completed",
            run_id=self._run_id,
            model=result.model,
            prompt_version=prompt_version,
            message_id=result.message_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            stop_reason=result.stop_reason,
            duration_ms=result.duration_ms,
            risk_count=len(result.report.risks),
            recommendation=result.report.recommendation.value,
        )

        return result

    @staticmethod
    def _validate_prompt(
        *,
        prompt: str,
        prompt_name: str,
    ) -> None:
        """Reject blank prompt values before making an external API call."""
        if not prompt.strip():
            raise AnthropicClientResponseError(
                f"{prompt_name} must not be blank."
            )

    def _log_failure(
        self,
        *,
        event_name: str,
        error: Exception,
        prompt_version: str,
    ) -> None:
        """Log safe failure metadata without prompts, evidence, or secrets."""
        status_code: int | None = None

        if isinstance(error, anthropic.APIStatusError):
            status_code = error.status_code

        logger.warning(
            event_name,
            run_id=self._run_id,
            model=self._config.model,
            prompt_version=prompt_version,
            error_type=type(error).__name__,
            status_code=status_code,
        )
