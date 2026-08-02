"""Unit tests for AgentFlow fresh-or-reuse query execution policy."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
    ResponseDepth,
)
from app.services.agent_query_execution_policy import (
    ANALYTICAL_RELEASE_INTENTS,
    PERSISTED_ONLY_RELEASE_INTENTS,
    has_explicit_release_run_context,
    should_resolve_persisted_context,
)


def build_plan(
    *,
    intent: AgentIntent,
    release_run_id: object | None = None,
) -> AgentQueryPlan:
    """Build a validated query plan for policy tests."""

    return AgentQueryPlan(
        intent=intent,
        response_depth=ResponseDepth.STANDARD,
        confidence=1.0,
        release_run_id=release_run_id,
        requires_current_snapshot=True,
        routing_reason_code="test_execution_policy",
    )


def test_intent_groups_do_not_overlap() -> None:
    """Analytical and persisted-only intents must remain mutually exclusive."""

    assert ANALYTICAL_RELEASE_INTENTS.isdisjoint(
        PERSISTED_ONLY_RELEASE_INTENTS
    )


def test_detects_explicit_request_release_run_context() -> None:
    """A request release-run ID should count as explicit context."""

    release_run_id = uuid4()
    request = AgentQueryRequest(
        query="Show GitHub risks.",
        release_run_id=release_run_id,
    )
    plan = build_plan(intent=AgentIntent.FILTER_RISKS)

    assert has_explicit_release_run_context(request, plan) is True


def test_detects_explicit_plan_release_run_context() -> None:
    """A validated plan release-run ID should count as explicit context."""

    release_run_id = uuid4()
    request = AgentQueryRequest(query="Show GitHub risks.")
    plan = build_plan(
        intent=AgentIntent.FILTER_RISKS,
        release_run_id=release_run_id,
    )

    assert has_explicit_release_run_context(request, plan) is True


@pytest.mark.parametrize(
    "intent",
    sorted(
        ANALYTICAL_RELEASE_INTENTS,
        key=lambda candidate: candidate.value,
    ),
)
def test_standalone_analytical_query_uses_fresh_assessment(
    intent: AgentIntent,
) -> None:
    """Analytical queries without an ID must not reuse persisted context."""

    request = AgentQueryRequest(query="Analyze current release risk.")
    plan = build_plan(intent=intent)

    assert should_resolve_persisted_context(request, plan) is False


@pytest.mark.parametrize(
    "intent",
    sorted(
        ANALYTICAL_RELEASE_INTENTS,
        key=lambda candidate: candidate.value,
    ),
)
def test_analytical_query_with_id_reuses_persisted_snapshot(
    intent: AgentIntent,
) -> None:
    """Analytical queries with an explicit ID should reuse that snapshot."""

    release_run_id = uuid4()
    request = AgentQueryRequest(
        query="Analyze this release run.",
        release_run_id=release_run_id,
    )
    plan = build_plan(
        intent=intent,
        release_run_id=release_run_id,
    )

    assert should_resolve_persisted_context(request, plan) is True


@pytest.mark.parametrize(
    "intent",
    sorted(
        PERSISTED_ONLY_RELEASE_INTENTS,
        key=lambda candidate: candidate.value,
    ),
)
def test_operational_query_enters_strict_persisted_resolver(
    intent: AgentIntent,
) -> None:
    """Operational queries must enter the resolver even when the ID is absent."""

    request = AgentQueryRequest(query="Check durable release state.")
    plan = build_plan(intent=intent)

    assert should_resolve_persisted_context(request, plan) is True


def test_unrelated_intent_does_not_resolve_release_context() -> None:
    """Out-of-scope requests must not enter either release execution path."""

    request = AgentQueryRequest(query="Write a recipe.")
    plan = build_plan(intent=AgentIntent.OUT_OF_SCOPE)

    assert should_resolve_persisted_context(request, plan) is False
