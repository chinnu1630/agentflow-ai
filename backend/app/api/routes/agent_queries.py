"""FastAPI routes for natural-language AgentFlow query planning.

This module exposes the deterministic AgentQueryRouter through an HTTP API.
The route performs classification only and does not execute release workflows,
retrieve internal data, approve releases, or send Slack alerts.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.schemas.agent_query import AgentQueryPlan, AgentQueryRequest
from app.services.agent_query_router import AgentQueryRouter

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


def get_agent_query_router() -> AgentQueryRouter:
    """Return the AgentQueryRouter dependency.

    Returns:
        A deterministic query-router instance.
    """

    return AgentQueryRouter()


AgentQueryRouterDependency = Annotated[
    AgentQueryRouter,
    Depends(get_agent_query_router),
]


@router.post(
    "/query-plan",
    response_model=AgentQueryPlan,
    status_code=status.HTTP_200_OK,
)
async def create_agent_query_plan(
    payload: AgentQueryRequest,
    request: Request,
    query_router: AgentQueryRouterDependency,
) -> AgentQueryPlan:
    """Convert a natural-language question into a safe workflow plan.

    Args:
        payload: Validated natural-language query and optional context IDs.
        request: Current FastAPI request containing request context.
        query_router: Injected natural-language query router.

    Returns:
        A validated query plan describing the intended AgentFlow operation.
    """

    plan = await query_router.create_plan(payload)

    logger.info(
        "agent_query_plan_created",
        extra={
            "run_id": getattr(request.state, "run_id", None),
            "intent": plan.intent.value,
            "response_depth": plan.response_depth.value,
            "confidence": plan.confidence,
            "requires_current_snapshot": (plan.requires_current_snapshot),
            "requires_historical_lookup": (plan.requires_historical_lookup),
            "requires_human_approval": (plan.requires_human_approval),
            "may_execute_side_effect": plan.may_execute_side_effect,
            "query_character_count": len(payload.query),
        },
    )

    return plan
