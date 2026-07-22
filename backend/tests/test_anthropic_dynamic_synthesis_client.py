"""Tests for the Anthropic dynamic-answer synthesis client."""

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
from app.integrations.anthropic_dynamic_synthesis_client import (
    AnthropicDynamicSynthesisClient,
)
from app.schemas.agent_dynamic_synthesis import (
    AgentDynamicAnswer,
    AgentDynamicAnswerCitation,
)


def _build_config() -> AnthropicClientConfig:
    """Build reusable Anthropic synthesis configuration."""
    return AnthropicClientConfig(
        api_key=SecretStr("test-anthropic-key"),
        model="test-claude-model",
        max_tokens=2_048,
        timeout_seconds=10.0,
        max_retries=2,
    )


def _build_answer() -> AgentDynamicAnswer:
    """Build one valid evidence-grounded dynamic answer."""
    return AgentDynamicAnswer(
        answer="Follow the documented payment rollback procedure.",
        confidence=0.95,
        citations=[
            AgentDynamicAnswerCitation(
                source_type="engineering_document_chunk",
                source_id="chunk-123",
                title="Payment Service Runbook",
                supporting_fact="The runbook defines the rollback steps.",
            )
        ],
        requires_human_review=False,
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
async def test_synthesize_dynamic_answer_returns_validated_answer() -> None:
    """Client should return structured answer and safe usage metadata."""
    response = SimpleNamespace(
        id="msg_dynamic_123",
        model="test-claude-model",
        parsed_output=_build_answer(),
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=350,
            output_tokens=140,
        ),
    )
    fake_client, parse_mock = _build_fake_client(response)

    client = AnthropicDynamicSynthesisClient(
        config=_build_config(),
        run_id="run-dynamic-001",
        client=fake_client,
    )

    result = await client.synthesize_dynamic_answer(
        system_prompt="Synthesize only from trusted evidence.",
        user_prompt="Use the validated tool results.",
        prompt_version="agent-dynamic-synthesis-v1",
    )

    assert result.answer.confidence == 0.95
    assert result.message_id == "msg_dynamic_123"
    assert result.input_tokens == 350
    assert result.output_tokens == 140

    assert parse_mock.await_args is not None
    request_arguments = dict(parse_mock.await_args.kwargs)

    assert request_arguments["model"] == "test-claude-model"
    assert request_arguments["max_tokens"] == 2_048
    assert request_arguments["temperature"] == 0.0
    assert request_arguments["output_format"] is AgentDynamicAnswer


@pytest.mark.anyio
async def test_synthesize_dynamic_answer_rejects_blank_prompt() -> None:
    """Blank prompts must fail before an external API request."""
    fake_client, parse_mock = _build_fake_client(object())

    client = AnthropicDynamicSynthesisClient(
        config=_build_config(),
        run_id="run-dynamic-002",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="system_prompt must not be blank",
    ):
        await client.synthesize_dynamic_answer(
            system_prompt=" ",
            user_prompt="Use trusted evidence.",
            prompt_version="agent-dynamic-synthesis-v1",
        )

    parse_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_synthesize_dynamic_answer_rejects_missing_output() -> None:
    """Responses without parsed answers must fail closed."""
    response = SimpleNamespace(
        id="msg_dynamic_456",
        model="test-claude-model",
        parsed_output=None,
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
        ),
    )
    fake_client, _ = _build_fake_client(response)

    client = AnthropicDynamicSynthesisClient(
        config=_build_config(),
        run_id="run-dynamic-003",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="did not return a parsed dynamic answer",
    ):
        await client.synthesize_dynamic_answer(
            system_prompt="Synthesize a grounded answer.",
            user_prompt="Use trusted evidence.",
            prompt_version="agent-dynamic-synthesis-v1",
        )


@pytest.mark.anyio
async def test_synthesize_dynamic_answer_rejects_truncation() -> None:
    """Token-truncated dynamic answers must never be returned."""
    response = SimpleNamespace(
        id="msg_dynamic_789",
        model="test-claude-model",
        parsed_output=_build_answer(),
        stop_reason="max_tokens",
        usage=SimpleNamespace(
            input_tokens=500,
            output_tokens=2_048,
        ),
    )
    fake_client, _ = _build_fake_client(response)

    client = AnthropicDynamicSynthesisClient(
        config=_build_config(),
        run_id="run-dynamic-004",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="exceeded the output-token limit",
    ):
        await client.synthesize_dynamic_answer(
            system_prompt="Synthesize a grounded answer.",
            user_prompt="Use trusted evidence.",
            prompt_version="agent-dynamic-synthesis-v1",
        )
