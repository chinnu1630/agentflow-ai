"""Async HTTP client for the AgentFlow FastAPI backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import httpx
import structlog
from pydantic import SecretStr, ValidationError

from agentflow_frontend.api_models import AgentQueryRequest, AgentQueryResponse
from agentflow_frontend.config import FrontendSettings

logger = structlog.get_logger(__name__)


class AgentFlowAPIError(RuntimeError):
    """Base error raised when an AgentFlow backend request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a safe API error."""
        super().__init__(message)
        self.status_code = status_code
        self.run_id = run_id


class AgentFlowAuthenticationError(AgentFlowAPIError):
    """Raised when the backend rejects the JWT."""


class AgentFlowAuthorizationError(AgentFlowAPIError):
    """Raised when the principal lacks a required scope."""


class AgentFlowRateLimitError(AgentFlowAPIError):
    """Raised when Redis-backed API rate limiting rejects a request."""


class AgentFlowServiceUnavailableError(AgentFlowAPIError):
    """Raised when the backend or one of its required services is unavailable."""


class AgentFlowResponseValidationError(AgentFlowAPIError):
    """Raised when the backend returns an invalid response contract."""


@dataclass(frozen=True, slots=True)
class AgentQueryCallResult:
    """Validated agent response plus its request correlation identifier."""

    response: AgentQueryResponse
    run_id: str


class AgentFlowAPIClient:
    """Typed asynchronous client for AgentFlow manager operations."""

    def __init__(
        self,
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            settings: Validated frontend runtime settings.
            bearer_token: Signed JWT used for backend authorization.
            http_client: Optional injected client used by tests or shared callers.
        """
        token_value = bearer_token.get_secret_value().strip()

        if not token_value:
            raise ValueError("Bearer token must not be empty.")

        self._settings = settings
        self._bearer_token = bearer_token
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=settings.request_timeout_seconds,
                connect=settings.connect_timeout_seconds,
            ),
            follow_redirects=False,
        )

    async def __aenter__(self) -> AgentFlowAPIClient:
        """Enter the async client context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Close internally owned HTTP resources."""
        await self.close()

    async def close(self) -> None:
        """Close the internally created HTTP client."""
        if self._owns_http_client:
            await self._http_client.aclose()

    async def execute_agent_query(
        self,
        request: AgentQueryRequest,
    ) -> AgentQueryCallResult:
        """Execute a manager question through the existing backend agent API.

        Args:
            request: Validated natural-language query and optional context.

        Returns:
            Validated agent response with the frontend-generated run ID.

        Raises:
            AgentFlowAuthenticationError: When the JWT is invalid or expired.
            AgentFlowAuthorizationError: When the JWT lacks ``agent:query``.
            AgentFlowRateLimitError: When the request exceeds its Redis limit.
            AgentFlowServiceUnavailableError: On timeout or service outage.
            AgentFlowResponseValidationError: On an invalid backend response.
            AgentFlowAPIError: For other unsuccessful HTTP responses.
        """
        run_id = str(uuid4())
        url = self._settings.build_api_url("/api/v1/agent/query")

        try:
            response = await self._http_client.post(
                url,
                headers={
                    "Authorization": (
                        f"Bearer {self._bearer_token.get_secret_value()}"
                    ),
                    "Content-Type": "application/json",
                    "X-Run-ID": run_id,
                },
                json=request.model_dump(mode="json", exclude_none=True),
            )
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            logger.warning(
                "agentflow_frontend_agent_query_timeout",
                run_id=run_id,
                error_type=type(exc).__name__,
            )
            raise AgentFlowServiceUnavailableError(
                "The AgentFlow backend timed out.",
                run_id=run_id,
            ) from exc
        except httpx.NetworkError as exc:
            logger.warning(
                "agentflow_frontend_agent_query_network_error",
                run_id=run_id,
                error_type=type(exc).__name__,
            )
            raise AgentFlowServiceUnavailableError(
                "The AgentFlow backend is unavailable.",
                run_id=run_id,
            ) from exc

        response_run_id = response.headers.get("X-Run-ID", run_id)

        if response.is_error:
            self._raise_for_error_response(
                response=response,
                run_id=response_run_id,
            )

        try:
            payload = cast(dict[str, Any], response.json())
            parsed_response = AgentQueryResponse.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            logger.error(
                "agentflow_frontend_agent_query_response_invalid",
                run_id=response_run_id,
                status_code=response.status_code,
                error_type=type(exc).__name__,
            )
            raise AgentFlowResponseValidationError(
                "The AgentFlow backend returned an invalid response.",
                status_code=response.status_code,
                run_id=response_run_id,
            ) from exc

        logger.info(
            "agentflow_frontend_agent_query_completed",
            run_id=response_run_id,
            status_code=response.status_code,
            approval_required=parsed_response.approval_required,
            citation_count=len(parsed_response.citations),
        )

        return AgentQueryCallResult(
            response=parsed_response,
            run_id=response_run_id,
        )

    @staticmethod
    def _raise_for_error_response(
        *,
        response: httpx.Response,
        run_id: str,
    ) -> None:
        """Convert backend HTTP failures into safe frontend exceptions."""
        status_code = response.status_code

        if status_code == 401:
            raise AgentFlowAuthenticationError(
                "Authentication failed. The access token may be invalid or expired.",
                status_code=status_code,
                run_id=run_id,
            )

        if status_code == 403:
            raise AgentFlowAuthorizationError(
                "You are not authorized to run AgentFlow queries.",
                status_code=status_code,
                run_id=run_id,
            )

        if status_code == 429:
            raise AgentFlowRateLimitError(
                "The AgentFlow query rate limit was exceeded.",
                status_code=status_code,
                run_id=run_id,
            )

        if status_code in {502, 503, 504}:
            raise AgentFlowServiceUnavailableError(
                "The AgentFlow service is temporarily unavailable.",
                status_code=status_code,
                run_id=run_id,
            )

        raise AgentFlowAPIError(
            "The AgentFlow backend rejected the request.",
            status_code=status_code,
            run_id=run_id,
        )
