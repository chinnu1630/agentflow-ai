"""Async HTTP client for the AgentFlow FastAPI backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import uuid4

import httpx
import structlog
from pydantic import BaseModel, SecretStr, ValidationError

from agentflow_frontend.api_models import (
    AgentQueryRequest,
    AgentQueryResponse,
    PendingReleaseRunApprovalList,
    ReleaseRunApproval,
    ReleaseRunApprovalDecisionRequest,
    ReleaseRunEventList,
    ReleaseRunStatus,
    SlackReleaseAlertResult,
)
from agentflow_frontend.config import FrontendSettings

logger = structlog.get_logger(__name__)

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


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


@dataclass(frozen=True, slots=True)
class PendingApprovalsCallResult:
    """Validated pending approvals plus the request correlation identifier."""

    response: PendingReleaseRunApprovalList
    run_id: str



@dataclass(frozen=True, slots=True)
class ApprovalDecisionCallResult:
    """Validated approval decision plus the request correlation identifier."""

    response: ReleaseRunApproval
    run_id: str



@dataclass(frozen=True, slots=True)
class ReleaseRunStatusCallResult:
    """Validated workflow status plus the request correlation identifier."""

    response: ReleaseRunStatus
    run_id: str


@dataclass(frozen=True, slots=True)
class ReleaseRunEventsCallResult:
    """Validated audit timeline plus the request correlation identifier."""

    response: ReleaseRunEventList
    run_id: str



@dataclass(frozen=True, slots=True)
class SlackAlertCallResult:
    """Validated Slack delivery result plus request correlation identifier."""

    response: SlackReleaseAlertResult
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

        if settings.auth_required and not token_value:
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
            Validated agent response with its request correlation identifier.
        """
        parsed_response, response_run_id = await self._request_model(
            method="POST",
            path="/api/v1/agent/query",
            response_model=AgentQueryResponse,
            operation_name="agent_query",
            json_body=request.model_dump(mode="json", exclude_none=True),
        )

        logger.info(
            "agentflow_frontend_agent_query_completed",
            run_id=response_run_id,
            approval_required=parsed_response.approval_required,
            citation_count=len(parsed_response.citations),
        )

        return AgentQueryCallResult(
            response=parsed_response,
            run_id=response_run_id,
        )

    async def list_pending_approvals(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> PendingApprovalsCallResult:
        """Load pending HITL approvals for the manager dashboard.

        Args:
            limit: Maximum approvals to request, from 1 through 500.
            offset: Number of matching approvals to skip.

        Returns:
            Validated pending approval queue with request correlation ID.

        Raises:
            ValueError: If pagination values violate the backend contract.
        """
        if not 1 <= limit <= 500:
            raise ValueError("Approval limit must be between 1 and 500.")

        if offset < 0:
            raise ValueError("Approval offset must be zero or greater.")

        parsed_response, response_run_id = await self._request_model(
            method="GET",
            path="/api/v1/release-runs/approvals/pending",
            response_model=PendingReleaseRunApprovalList,
            operation_name="pending_approvals",
            query_params={
                "limit": limit,
                "offset": offset,
            },
        )

        logger.info(
            "agentflow_frontend_pending_approvals_loaded",
            run_id=response_run_id,
            approval_count=len(parsed_response.approvals),
        )

        return PendingApprovalsCallResult(
            response=parsed_response,
            run_id=response_run_id,
        )

    async def decide_release_run_approval(
        self,
        *,
        release_run_id: str,
        approval_id: str,
        decision: ReleaseRunApprovalDecisionRequest,
    ) -> ApprovalDecisionCallResult:
        """Approve or reject one pending release approval.

        Args:
            release_run_id: Backend release-run UUID.
            approval_id: Pending approval UUID belonging to the release run.
            decision: Validated approved or rejected decision payload.

        Returns:
            Persisted approval result with request correlation ID.
        """
        parsed_response, response_run_id = await self._request_model(
            method="POST",
            path=(
                f"/api/v1/release-runs/{release_run_id}"
                f"/approvals/{approval_id}/decision"
            ),
            response_model=ReleaseRunApproval,
            operation_name="approval_decision",
            json_body=decision.model_dump(mode="json", exclude_none=True),
        )

        logger.info(
            "agentflow_frontend_approval_decided",
            run_id=response_run_id,
            release_run_id=release_run_id,
            approval_id=approval_id,
            approval_status=parsed_response.approval_status,
        )

        return ApprovalDecisionCallResult(
            response=parsed_response,
            run_id=response_run_id,
        )

    async def get_release_run_status(
        self,
        *,
        release_run_id: str,
    ) -> ReleaseRunStatusCallResult:
        """Load the current persisted workflow status for one release run."""
        parsed_response, response_run_id = await self._request_model(
            method="GET",
            path=f"/api/v1/release-runs/{release_run_id}",
            response_model=ReleaseRunStatus,
            operation_name="release_run_status",
        )

        logger.info(
            "agentflow_frontend_release_run_status_loaded",
            run_id=response_run_id,
            release_run_id=release_run_id,
            release_run_status=parsed_response.status,
        )

        return ReleaseRunStatusCallResult(
            response=parsed_response,
            run_id=response_run_id,
        )

    async def list_release_run_events(
        self,
        *,
        release_run_id: str,
    ) -> ReleaseRunEventsCallResult:
        """Load the append-only audit timeline for one release run."""
        parsed_response, response_run_id = await self._request_model(
            method="GET",
            path=f"/api/v1/release-runs/{release_run_id}/events",
            response_model=ReleaseRunEventList,
            operation_name="release_run_events",
        )

        logger.info(
            "agentflow_frontend_release_run_events_loaded",
            run_id=response_run_id,
            release_run_id=release_run_id,
            event_count=len(parsed_response.events),
        )

        return ReleaseRunEventsCallResult(
            response=parsed_response,
            run_id=response_run_id,
        )

    async def send_release_run_slack_alert(
        self,
        *,
        release_run_id: str,
    ) -> SlackAlertCallResult:
        """Send an approval-gated release alert through the backend.

        The frontend does not determine whether sending is allowed. FastAPI
        verifies the signed principal's ``release:notify`` scope, durable
        approval state, trusted risk snapshot, and duplicate-send protection.

        Args:
            release_run_id: Backend release-run UUID.

        Returns:
            Validated Slack delivery result with request correlation ID.
        """
        parsed_response, response_run_id = await self._request_model(
            method="POST",
            path=f"/api/v1/release-runs/{release_run_id}/slack-alert",
            response_model=SlackReleaseAlertResult,
            operation_name="slack_alert",
        )

        logger.info(
            "agentflow_frontend_slack_alert_sent",
            run_id=response_run_id,
            release_run_id=release_run_id,
            sent=parsed_response.sent,
            risk_level=parsed_response.risk_level,
        )

        return SlackAlertCallResult(
            response=parsed_response,
            run_id=response_run_id,
        )

    async def _request_model(
        self,
        *,
        method: str,
        path: str,
        response_model: type[ResponseModelT],
        operation_name: str,
        json_body: dict[str, Any] | None = None,
        query_params: dict[str, str | int] | None = None,
    ) -> tuple[ResponseModelT, str]:
        """Execute one backend request and validate its response model."""
        run_id = str(uuid4())
        url = self._settings.build_api_url(path)

        try:
            response = await self._http_client.request(
                method=method,
                url=url,
                headers=self._build_headers(run_id),
                json=json_body,
                params=query_params,
            )
        except httpx.TimeoutException as exc:
            logger.warning(
                "agentflow_frontend_request_timeout",
                run_id=run_id,
                operation=operation_name,
                error_type=type(exc).__name__,
            )
            raise AgentFlowServiceUnavailableError(
                "The AgentFlow backend timed out.",
                run_id=run_id,
            ) from exc
        except httpx.NetworkError as exc:
            logger.warning(
                "agentflow_frontend_request_network_error",
                run_id=run_id,
                operation=operation_name,
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
            payload = response.json()
            parsed_response = response_model.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            logger.error(
                "agentflow_frontend_response_invalid",
                run_id=response_run_id,
                operation=operation_name,
                status_code=response.status_code,
                error_type=type(exc).__name__,
            )
            raise AgentFlowResponseValidationError(
                "The AgentFlow backend returned an invalid response.",
                status_code=response.status_code,
                run_id=response_run_id,
            ) from exc

        return parsed_response, response_run_id

    def _build_headers(self, run_id: str) -> dict[str, str]:
        """Build safe request headers without exposing authentication data."""
        headers = {
            "Content-Type": "application/json",
            "X-Run-ID": run_id,
        }

        if self._settings.auth_required:
            headers["Authorization"] = (
                f"Bearer {self._bearer_token.get_secret_value()}"
            )

        return headers

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
                "You are not authorized to perform this AgentFlow operation.",
                status_code=status_code,
                run_id=run_id,
            )

        if status_code == 429:
            raise AgentFlowRateLimitError(
                "The AgentFlow API rate limit was exceeded.",
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
