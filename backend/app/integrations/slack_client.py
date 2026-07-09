"""Async Slack Web API client for AgentFlow AI notifications.

This client is responsible only for sending already-built Slack payloads.
Payload formatting belongs in SlackAlertPayloadService. This separation keeps
message generation deterministic and Slack delivery isolated as an external
side effect.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.services.slack_alert_payload_service import SlackReleaseRiskAlertPayload

logger = structlog.get_logger(__name__)


class SlackClientConfig(BaseModel):
    """Configuration for Slack Web API message delivery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bot_token: SecretStr
    channel_id: str = Field(min_length=1, max_length=255)
    api_base_url: str = Field(
        default="https://slack.com/api",
        min_length=1,
        max_length=500,
    )
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=60.0)
    max_retries: int = Field(default=3, ge=0, le=5)
    retry_base_delay_seconds: float = Field(default=0.25, ge=0.0, le=10.0)

    @field_validator("channel_id", "api_base_url")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """Normalize required text configuration values."""
        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("value must not be blank")

        return stripped_value.rstrip("/") if stripped_value.startswith("http") else stripped_value


class SlackPostMessageResult(BaseModel):
    """Result returned after a successful Slack chat.postMessage call."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    channel: str
    timestamp: str
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class SlackClientError(RuntimeError):
    """Raised when Slack message delivery fails."""


class SlackClient:
    """Small async Slack Web API client with retry handling."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        config: SlackClientConfig,
        request_id: str,
    ) -> None:
        """Initialize the Slack client.

        Args:
            http_client: Shared async HTTP client.
            config: Slack delivery configuration.
            request_id: Request or workflow ID for safe structured logs.
        """
        self._http_client = http_client
        self._config = config
        self._request_id = request_id

    async def send_release_risk_alert(
        self,
        payload: SlackReleaseRiskAlertPayload,
    ) -> SlackPostMessageResult:
        """Send a release-risk alert payload to Slack.

        Args:
            payload: Slack-compatible payload built by SlackAlertPayloadService.

        Returns:
            Successful Slack postMessage result.

        Raises:
            SlackClientError: If Slack rejects the request or delivery fails.
        """
        started_at = time.perf_counter()
        endpoint = f"{self._config.api_base_url}/chat.postMessage"

        request_body = {
            "channel": self._config.channel_id,
            "text": payload.text,
            "blocks": payload.blocks,
            "metadata": {
                "event_type": "agentflow_release_risk_alert",
                "event_payload": payload.metadata,
            },
        }

        headers = {
            "Authorization": (
                f"Bearer {self._config.bot_token.get_secret_value()}"
            ),
            "Content-Type": "application/json",
        }

        last_error: str | None = None

        for attempt_index in range(self._config.max_retries + 1):
            try:
                response = await self._http_client.post(
                    endpoint,
                    json=request_body,
                    headers=headers,
                    timeout=self._config.timeout_seconds,
                )

                if self._should_retry_http_status(response.status_code):
                    last_error = f"retryable_http_{response.status_code}"

                    if attempt_index < self._config.max_retries:
                        await self._sleep_before_retry(
                            response=response,
                            attempt_index=attempt_index,
                        )
                        continue

                    raise SlackClientError(
                        "Slack message delivery failed after retryable HTTP response."
                    )

                if response.status_code >= 400:
                    raise SlackClientError(
                        "Slack message delivery failed with non-retryable HTTP response."
                    )

                response_data = self._parse_response_json(response)

                if response_data.get("ok") is not True:
                    slack_error = str(response_data.get("error", "unknown_error"))
                    last_error = slack_error

                    if self._is_retryable_slack_error(slack_error) and (
                        attempt_index < self._config.max_retries
                    ):
                        await self._sleep_before_retry(
                            response=response,
                            attempt_index=attempt_index,
                        )
                        continue

                    raise SlackClientError(
                        f"Slack API rejected message: {slack_error}."
                    )

                result = SlackPostMessageResult(
                    ok=True,
                    channel=str(response_data.get("channel", "")),
                    timestamp=str(response_data.get("ts", "")),
                    response_metadata=self._safe_response_metadata(response_data),
                )

                logger.info(
                    "slack_release_risk_alert_sent",
                    request_id=self._request_id,
                    channel_id=self._config.channel_id,
                    slack_channel=result.channel,
                    attempt_count=attempt_index + 1,
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
                    payload_top_risk_count=payload.metadata.get("top_risk_count"),
                )

                return result

            except httpx.TransportError as exc:
                last_error = exc.__class__.__name__

                if attempt_index < self._config.max_retries:
                    await self._sleep_before_retry(
                        response=None,
                        attempt_index=attempt_index,
                    )
                    continue

                raise SlackClientError(
                    "Slack message delivery failed due to transport error."
                ) from exc

        raise SlackClientError(
            f"Slack message delivery failed after retries: {last_error}."
        )

    @staticmethod
    def _should_retry_http_status(status_code: int) -> bool:
        """Return whether an HTTP status is retryable."""
        return status_code == 429 or 500 <= status_code <= 599

    @staticmethod
    def _is_retryable_slack_error(error_code: str) -> bool:
        """Return whether a Slack API error code is retryable."""
        return error_code in {
            "ratelimited",
            "rate_limited",
            "internal_error",
            "fatal_error",
        }

    @staticmethod
    def _parse_response_json(response: httpx.Response) -> Mapping[str, Any]:
        """Parse Slack JSON response safely."""
        try:
            response_data = response.json()
        except ValueError as exc:
            raise SlackClientError("Slack returned invalid JSON response.") from exc

        if not isinstance(response_data, Mapping):
            raise SlackClientError("Slack returned unexpected response shape.")

        return response_data

    @staticmethod
    def _safe_response_metadata(
        response_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return safe response metadata without copying full Slack response."""
        metadata: dict[str, Any] = {}

        if "response_metadata" in response_data and isinstance(
            response_data["response_metadata"],
            Mapping,
        ):
            metadata["has_response_metadata"] = True

        return metadata

    async def _sleep_before_retry(
        self,
        *,
        response: httpx.Response | None,
        attempt_index: int,
    ) -> None:
        """Sleep before retry using Retry-After or exponential backoff."""
        retry_after = self._retry_after_seconds(response)

        delay_seconds = (
            retry_after
            if retry_after is not None
            else self._config.retry_base_delay_seconds * (2**attempt_index)
        )

        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        else:
            await asyncio.sleep(0)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response | None) -> float | None:
        """Read Retry-After header when Slack provides one."""
        if response is None:
            return None

        retry_after_value = response.headers.get("Retry-After")

        if retry_after_value is None:
            return None

        try:
            retry_after_seconds = float(retry_after_value)
        except ValueError:
            return None

        if retry_after_seconds < 0:
            return None

        return retry_after_seconds
