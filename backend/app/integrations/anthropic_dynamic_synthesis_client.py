"""Async Anthropic client for structured dynamic-agent synthesis."""

from __future__ import annotations

import time

import anthropic
import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.integrations.anthropic_client import (
    AnthropicClientConfig,
    AnthropicClientRateLimitError,
    AnthropicClientResponseError,
    AnthropicClientTimeoutError,
    AnthropicClientUnavailableError,
)
from app.schemas.agent_dynamic_synthesis import AgentDynamicAnswer

logger = structlog.get_logger(__name__)


class ClaudeDynamicSynthesisResult(BaseModel):
    """Validated dynamic answer and safe Claude usage metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: AgentDynamicAnswer
    message_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    stop_reason: str | None = None
    duration_ms: float = Field(ge=0.0)
    prompt_version: str = Field(min_length=1, max_length=100)


class AnthropicDynamicSynthesisClient:
    """Async Claude client with strict dynamic-answer validation."""

    def __init__(
        self,
        *,
        config: AnthropicClientConfig,
        run_id: str,
        client: AsyncAnthropic | None = None,
    ) -> None:
        """Initialize the dynamic-answer synthesis client.

        Args:
            config: Validated Anthropic connection and request limits.
            run_id: AgentFlow request correlation identifier.
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

    async def __aenter__(self) -> AnthropicDynamicSynthesisClient:
        """Enter the async synthesis client context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the internally created Anthropic client."""
        if self._owns_client:
            await self._client.close()

    async def synthesize_dynamic_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeDynamicSynthesisResult:
        """Generate one validated evidence-grounded dynamic answer.

        Prompt contents are excluded from logs because they may contain
        confidential GitHub, Jira, approval, or engineering-document data.

        Args:
            system_prompt: Trusted AgentFlow synthesis policy.
            user_prompt: Bounded tool outputs and trusted evidence metadata.
            prompt_version: Version identifier for LLMOps traceability.

        Returns:
            Validated dynamic answer with safe usage metadata.

        Raises:
            AnthropicClientTimeoutError: When the request times out.
            AnthropicClientRateLimitError: When Anthropic rate-limits the call.
            AnthropicClientUnavailableError: For connection or API failures.
            AnthropicClientResponseError: For invalid structured output.
        """
        self._validate_prompt(system_prompt, "system_prompt")
        self._validate_prompt(user_prompt, "user_prompt")
        self._validate_prompt(prompt_version, "prompt_version")

        started_at = time.perf_counter()

        logger.info(
            "claude_dynamic_synthesis_started",
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
                output_format=AgentDynamicAnswer,
            )
        except anthropic.APITimeoutError as exc:
            self._log_failure(
                "claude_dynamic_synthesis_timeout",
                exc,
                prompt_version,
            )
            raise AnthropicClientTimeoutError(
                "Claude dynamic synthesis timed out."
            ) from exc
        except anthropic.RateLimitError as exc:
            self._log_failure(
                "claude_dynamic_synthesis_rate_limited",
                exc,
                prompt_version,
            )
            raise AnthropicClientRateLimitError(
                "Claude dynamic synthesis was rate limited."
            ) from exc
        except anthropic.APIConnectionError as exc:
            self._log_failure(
                "claude_dynamic_synthesis_connection_failed",
                exc,
                prompt_version,
            )
            raise AnthropicClientUnavailableError(
                "Claude dynamic synthesis service is unavailable."
            ) from exc
        except anthropic.APIStatusError as exc:
            self._log_failure(
                "claude_dynamic_synthesis_api_failed",
                exc,
                prompt_version,
            )
            raise AnthropicClientUnavailableError(
                "Claude dynamic synthesis API request failed."
            ) from exc
        except (anthropic.APIError, ValidationError) as exc:
            self._log_failure(
                "claude_dynamic_synthesis_response_failed",
                exc,
                prompt_version,
            )
            raise AnthropicClientResponseError(
                "Claude dynamic-answer validation failed."
            ) from exc

        if response.stop_reason == "max_tokens":
            raise AnthropicClientResponseError(
                "Claude dynamic synthesis exceeded the output-token limit."
            )

        answer = response.parsed_output

        if answer is None:
            raise AnthropicClientResponseError(
                "Claude did not return a parsed dynamic answer."
            )

        duration_ms = round(
            (time.perf_counter() - started_at) * 1_000,
            2,
        )

        result = ClaudeDynamicSynthesisResult(
            answer=answer,
            message_id=response.id,
            model=str(response.model),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            duration_ms=duration_ms,
            prompt_version=prompt_version,
        )

        logger.info(
            "claude_dynamic_synthesis_completed",
            run_id=self._run_id,
            model=result.model,
            prompt_version=result.prompt_version,
            message_id=result.message_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=result.duration_ms,
            citation_count=len(result.answer.citations),
            requires_human_review=result.answer.requires_human_review,
        )

        return result

    @staticmethod
    def _validate_prompt(prompt: str, prompt_name: str) -> None:
        """Reject blank prompts before making an external API call."""
        if not prompt.strip():
            raise AnthropicClientResponseError(
                f"{prompt_name} must not be blank."
            )

    def _log_failure(
        self,
        event_name: str,
        error: Exception,
        prompt_version: str,
    ) -> None:
        """Log safe failure metadata without prompts or secrets."""
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
