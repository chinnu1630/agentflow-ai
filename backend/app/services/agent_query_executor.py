"""Execute validated natural-language AgentFlow query plans."""

from __future__ import annotations

import logging
from typing import Protocol
from uuid import UUID

from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.release_risk_response_mapper import (
    extract_risk_result_from_workflow_state,
    to_release_run_risk_response,
)
from app.services.release_run_service import (
    ReleaseRunResult,
    StartReleaseRunCommand,
)

logger = logging.getLogger(__name__)


class ReleaseRunWorkflowServiceProtocol(Protocol):
    """Release-run operations required by the agent query executor."""

    async def start_release_run(
        self,
        command: StartReleaseRunCommand,
    ) -> ReleaseRunResult:
        """Create a release run."""

        ...

    async def run_release_risk_workflow(
        self,
        release_run_id: UUID,
        *,
        manager_query: str,
        requested_by: str | None,
    ) -> object:
        """Execute the release-risk workflow."""

        ...


class AgentQueryExecutionError(RuntimeError):
    """Raised when an agent query plan cannot be executed."""


class UnsupportedAgentQueryIntentError(AgentQueryExecutionError):
    """Raised when the executor does not support the routed intent."""


class AgentQueryContextMismatchError(AgentQueryExecutionError):
    """Raised when request and plan context identifiers conflict."""


class AgentQueryResultError(AgentQueryExecutionError):
    """Raised when workflow execution returns no usable result."""


class AgentQueryExecutor:
    """Execute validated AgentFlow query plans using existing services."""

    def __init__(
        self,
        release_run_service: ReleaseRunWorkflowServiceProtocol,
        request_id: str,
    ) -> None:
        """Initialize the executor.

        Args:
            release_run_service: Existing release-run workflow service.
            request_id: Request identifier used for structured logging.
        """

        self._release_run_service = release_run_service
        self._request_id = request_id

    async def execute(
        self,
        request: AgentQueryRequest,
        plan: AgentQueryPlan,
        *,
        requested_by: str,
    ) -> ReleaseRunRiskResponse:
        """Execute a supported query plan and return release-risk results.

        Args:
            request: Original validated natural-language query.
            plan: Validated routing plan.
            requested_by: Authenticated actor requesting execution.

        Returns:
            Public release-risk response.

        Raises:
            UnsupportedAgentQueryIntentError: If the intent is unsupported.
            AgentQueryContextMismatchError: If request and plan IDs conflict.
            AgentQueryResultError: If the workflow returns no usable result.
        """

        if plan.intent is not AgentIntent.RELEASE_RISK_SUMMARY:
            raise UnsupportedAgentQueryIntentError(
                f"Unsupported agent query intent: {plan.intent.value}"
            )

        release_run_id = self._resolve_release_run_id(request, plan)

        if release_run_id is None:
            release_run = await self._release_run_service.start_release_run(
                StartReleaseRunCommand(
                    query=request.query,
                    requested_by=requested_by,
                )
            )
            release_run_id = release_run.id

        workflow_state = await self._release_run_service.run_release_risk_workflow(
            release_run_id,
            manager_query=request.query,
            requested_by=requested_by,
        )

        result = extract_risk_result_from_workflow_state(workflow_state)

        if result is None:
            raise AgentQueryResultError("Release-risk workflow returned no usable result.")

        response = to_release_run_risk_response(result)

        logger.info(
            "agent_query_executed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_run_id),
                "intent": plan.intent.value,
                "requested_by": requested_by,
                "approval_required": response.approval_required,
            },
        )

        return response

    @staticmethod
    def _resolve_release_run_id(
        request: AgentQueryRequest,
        plan: AgentQueryPlan,
    ) -> UUID | None:
        """Resolve and validate the release-run context."""

        if (
            request.release_run_id is not None
            and plan.release_run_id is not None
            and request.release_run_id != plan.release_run_id
        ):
            raise AgentQueryContextMismatchError(
                "Request and query plan release-run IDs do not match."
            )

        return plan.release_run_id or request.release_run_id
