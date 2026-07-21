"""Async Anthropic client for bounded AgentFlow execution planning."""

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
from app.schemas.agent_execution_plan import AgentExecutionPlan

logger = structlog.get_logger(__name__)


class ClaudeExecutionPlanResult(BaseModel):
    """Validated execution plan and safe Claude usage metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: AgentExecutionPlan
    message_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    stop_reason: str | None = None
    duration_ms: float = Field(ge=0.0)
    prompt_version: str = Field(min_length=1, max_length=100)


class AnthropicExecutionPlannerClient:
    """Async Claude client with strict execution-plan output validation."""

    def __init__(
        self,
        *,
        config: AnthropicClientConfig,
        run_id: str,
        client: AsyncAnthropic | None = None,
    ) -> None:
        """Initialize the bounded Claude execution planner.

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

    async def __aenter__(self) -> AnthropicExecutionPlannerClient:
        """Enter the async planner client context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the internally created Anthropic client."""
        if self._owns_client:
            await self._client.close()

    async def create_execution_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeExecutionPlanResult:
        """Generate one validated bounded AgentFlow execution plan.

        Prompt contents are excluded from logs because the manager query and
        extracted entities may contain confidential engineering information.

        Args:
            system_prompt: Trusted AgentFlow planning policy.
            user_prompt: Bounded routing and approved-tool data.
            prompt_version: Version identifier for LLMOps traceability.

        Returns:
            Validated execution plan with safe usage metadata.

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
            "claude_execution_planning_started",
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
                output_format=AgentExecutionPlan,
            )
        except anthropic.APITimeoutError as exc:
            self._log_failure(
                "claude_execution_planning_timeout",
                exc,
                prompt_version,
            )
            raise AnthropicClientTimeoutError(
                "Claude execution planning timed out."
            ) from exc
        except anthropic.RateLimitError as exc:
            self._log_failure(
                "claude_execution_planning_rate_limited",
                exc,
                prompt_version,
            )
            raise AnthropicClientRateLimitError(
                "Claude execution planning was rate limited."
            ) from exc
        except anthropic.APIConnectionError as exc:
            self._log_failure(
                "claude_execution_planning_connection_failed",
                exc,
                prompt_version,
            )
            raise AnthropicClientUnavailableError(
                "Claude execution planning service is unavailable."
            ) from exc
        except anthropic.APIStatusError as exc:
            self._log_failure(
                "claude_execution_planning_api_failed",
                exc,
                prompt_version,
            )
            raise AnthropicClientUnavailableError(
                "Claude execution planning API request failed."
            ) from exc
        except (anthropic.APIError, ValidationError) as exc:
            self._log_failure(
                "claude_execution_planning_response_failed",
                exc,
                prompt_version,
            )
            raise AnthropicClientResponseError(
                "Claude execution-plan validation failed."
            ) from exc

        if response.stop_reason == "max_tokens":
            raise AnthropicClientResponseError(
                "Claude execution planning exceeded the output-token limit."
            )

        plan = response.parsed_output

        if plan is None:
            raise AnthropicClientResponseError(
                "Claude did not return a parsed execution plan."
            )

        duration_ms = round(
            (time.perf_counter() - started_at) * 1_000,
            2,
        )

        result = ClaudeExecutionPlanResult(
            plan=plan,
            message_id=response.id,
            model=str(response.model),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            duration_ms=duration_ms,
            prompt_version=prompt_version,
        )

        logger.info(
            "claude_execution_planning_completed",
            run_id=self._run_id,
            model=result.model,
            prompt_version=result.prompt_version,
            message_id=result.message_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=result.duration_ms,
            intent=result.plan.intent.value,
            step_count=len(result.plan.steps),
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
