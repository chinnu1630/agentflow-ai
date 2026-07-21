"""Tests for the Anthropic structured execution-planner client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from anthropic import AsyncAnthropic
from pydantic import SecretStr

from app.integrations.anthropic_client import (
    AnthropicClientConfig,
    AnthropicClientResponseError,
)
from app.integrations.anthropic_execution_planner_client import (
    AnthropicExecutionPlannerClient,
)
from app.schemas.agent_execution_plan import (
    AgentExecutionPlan,
    AgentExecutionStep,
)
from app.schemas.agent_query import AgentIntent, ResponseDepth
from app.schemas.agent_tool import (
    AgentToolInvocation,
    AgentToolName,
)


def _build_config() -> AnthropicClientConfig:
    """Build reusable Anthropic planner configuration."""
    return AnthropicClientConfig(
        api_key=SecretStr("test-anthropic-key"),
        model="test-claude-model",
        max_tokens=2_048,
        timeout_seconds=10.0,
        max_retries=2,
    )


def _build_plan() -> AgentExecutionPlan:
    """Build one valid bounded execution plan."""
    step_id = "search_knowledge"

    return AgentExecutionPlan(
        objective="Find rollback guidance for the payment service.",
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        steps=[
            AgentExecutionStep(
                step_id=step_id,
                invocation=AgentToolInvocation(
                    step_id=step_id,
                    tool_name=(
                        AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
                    ),
                    arguments={
                        "query": "payment service rollback",
                    },
                    timeout_seconds=30,
                ),
            )
        ],
        plan_reason_code="search_engineering_knowledge",
    )


def _build_fake_client(
    response: object,
) -> tuple[AsyncAnthropic, AsyncMock]:
    """Build an injected fake Anthropic client."""
    parse_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(parse=parse_mock),
        close=AsyncMock(),
    )

    return cast(AsyncAnthropic, fake_client), parse_mock


@pytest.mark.anyio
async def test_create_execution_plan_returns_validated_plan() -> None:
    """Client should return structured plan and safe usage metadata."""
    response = SimpleNamespace(
        id="msg_plan_123",
        model="test-claude-model",
        parsed_output=_build_plan(),
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=300,
            output_tokens=120,
        ),
    )
    fake_client, parse_mock = _build_fake_client(response)

    client = AnthropicExecutionPlannerClient(
        config=_build_config(),
        run_id="run-plan-001",
        client=fake_client,
    )

    result = await client.create_execution_plan(
        system_prompt="Create a bounded execution plan.",
        user_prompt="Use approved read-only tools.",
        prompt_version="agent-execution-planner-v1",
    )

    assert result.plan.intent is AgentIntent.KNOWLEDGE_DOC_QUESTION
    assert result.plan.steps[0].invocation.tool_name is (
        AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
    )
    assert result.message_id == "msg_plan_123"
    assert result.input_tokens == 300
    assert result.output_tokens == 120

    assert parse_mock.await_args is not None
    request_arguments = dict(parse_mock.await_args.kwargs)

    assert request_arguments["model"] == "test-claude-model"
    assert request_arguments["max_tokens"] == 2_048
    assert request_arguments["temperature"] == 0.0
    assert request_arguments["output_format"] is AgentExecutionPlan


@pytest.mark.anyio
async def test_create_execution_plan_rejects_blank_prompt() -> None:
    """Blank prompts must fail before an external API request."""
    fake_client, parse_mock = _build_fake_client(object())

    client = AnthropicExecutionPlannerClient(
        config=_build_config(),
        run_id="run-plan-002",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="system_prompt must not be blank",
    ):
        await client.create_execution_plan(
            system_prompt=" ",
            user_prompt="Use approved tools.",
            prompt_version="agent-execution-planner-v1",
        )

    parse_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_create_execution_plan_rejects_missing_output() -> None:
    """Responses without parsed plans must fail closed."""
    response = SimpleNamespace(
        id="msg_plan_456",
        model="test-claude-model",
        parsed_output=None,
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
        ),
    )
    fake_client, _ = _build_fake_client(response)

    client = AnthropicExecutionPlannerClient(
        config=_build_config(),
        run_id="run-plan-003",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="did not return a parsed execution plan",
    ):
        await client.create_execution_plan(
            system_prompt="Create a bounded execution plan.",
            user_prompt="Use approved tools.",
            prompt_version="agent-execution-planner-v1",
        )


@pytest.mark.anyio
async def test_create_execution_plan_rejects_truncation() -> None:
    """Token-truncated plans must never be executed."""
    response = SimpleNamespace(
        id="msg_plan_789",
        model="test-claude-model",
        parsed_output=_build_plan(),
        stop_reason="max_tokens",
        usage=SimpleNamespace(
            input_tokens=500,
            output_tokens=2_048,
        ),
    )
    fake_client, _ = _build_fake_client(response)

    client = AnthropicExecutionPlannerClient(
        config=_build_config(),
        run_id="run-plan-004",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="exceeded the output-token limit",
    ):
        await client.create_execution_plan(
            system_prompt="Create a bounded execution plan.",
            user_prompt="Use approved tools.",
            prompt_version="agent-execution-planner-v1",
        )
