"""Tests for the async Anthropic structured-synthesis client."""

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
    AnthropicRiskSynthesisClient,
)
from app.schemas.llm_risk_synthesis import (
    ClaudeReleaseRiskReport,
    SynthesisEvidenceCitation,
    SynthesisEvidenceSource,
    SynthesizedReleaseRisk,
)
from app.schemas.risk import (
    RiskSeverityResponse,
    RiskSummaryActionResponse,
)


def build_config() -> AnthropicClientConfig:
    """Build reusable Anthropic client configuration for tests."""
    return AnthropicClientConfig(
        api_key=SecretStr("test-anthropic-key"),
        model="test-claude-model",
        max_tokens=2_048,
        timeout_seconds=10.0,
        max_retries=2,
    )


def build_report() -> ClaudeReleaseRiskReport:
    """Build one valid structured Claude risk report."""
    return ClaudeReleaseRiskReport(
        recommendation=RiskSummaryActionResponse.BLOCK_RELEASE,
        confidence=0.95,
        executive_summary="The payment release should be blocked.",
        risks=[
            SynthesizedReleaseRisk(
                rank=1,
                title="Payment release blocker",
                severity=RiskSeverityResponse.CRITICAL,
                confidence=0.96,
                explanation="PAY-102 remains unresolved.",
                evidence=[
                    SynthesisEvidenceCitation(
                        source=SynthesisEvidenceSource.JIRA_ISSUE,
                        source_id="PAY-102",
                        title="Payment release blocker",
                        source_url="https://jira.test/browse/PAY-102",
                        supporting_fact=(
                            "PAY-102 is an unresolved release-blocking issue."
                        ),
                    )
                ],
                mitigations=["Resolve PAY-102 before deployment."],
            )
        ],
        requires_human_review=True,
    )


def build_fake_client(response: object) -> tuple[AsyncAnthropic, AsyncMock]:
    """Build an injected fake Anthropic client with an async parse method."""
    parse_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(parse=parse_mock),
        close=AsyncMock(),
    )

    return cast(AsyncAnthropic, fake_client), parse_mock


@pytest.mark.anyio
async def test_synthesize_release_risk_returns_validated_report() -> None:
    """Client should return parsed output and safe usage metadata."""
    response = SimpleNamespace(
        id="msg_test_123",
        model="test-claude-model",
        parsed_output=build_report(),
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=450,
            output_tokens=220,
        ),
    )
    fake_client, parse_mock = build_fake_client(response)

    client = AnthropicRiskSynthesisClient(
        config=build_config(),
        run_id="run-test-001",
        client=fake_client,
    )

    result = await client.synthesize_release_risk(
        system_prompt="Return a grounded release-risk report.",
        user_prompt="Trusted Jira evidence: PAY-102 is unresolved.",
        prompt_version="release-risk-synthesis-v1",
    )

    assert result.report.recommendation is RiskSummaryActionResponse.BLOCK_RELEASE
    assert result.message_id == "msg_test_123"
    assert result.input_tokens == 450
    assert result.output_tokens == 220
    assert result.prompt_version == "release-risk-synthesis-v1"

    assert parse_mock.await_args is not None
    request_arguments = dict(parse_mock.await_args.kwargs)

    assert request_arguments["model"] == "test-claude-model"
    assert request_arguments["max_tokens"] == 2_048
    assert request_arguments["temperature"] == 0.0
    assert request_arguments["output_format"] is ClaudeReleaseRiskReport


@pytest.mark.anyio
async def test_synthesize_release_risk_rejects_blank_prompt() -> None:
    """Blank prompts must fail before making an external API request."""
    fake_client, parse_mock = build_fake_client(object())

    client = AnthropicRiskSynthesisClient(
        config=build_config(),
        run_id="run-test-002",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="system_prompt must not be blank",
    ):
        await client.synthesize_release_risk(
            system_prompt=" ",
            user_prompt="Trusted release evidence.",
            prompt_version="release-risk-synthesis-v1",
        )

    parse_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_synthesize_release_risk_rejects_missing_parsed_output() -> None:
    """Client should reject responses without validated structured output."""
    response = SimpleNamespace(
        id="msg_test_456",
        model="test-claude-model",
        parsed_output=None,
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
        ),
    )
    fake_client, _ = build_fake_client(response)

    client = AnthropicRiskSynthesisClient(
        config=build_config(),
        run_id="run-test-003",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="did not return a parsed release-risk report",
    ):
        await client.synthesize_release_risk(
            system_prompt="Return structured output.",
            user_prompt="Trusted release evidence.",
            prompt_version="release-risk-synthesis-v1",
        )


@pytest.mark.anyio
async def test_synthesize_release_risk_rejects_truncated_response() -> None:
    """Output-token truncation must never produce a trusted report."""
    response = SimpleNamespace(
        id="msg_test_789",
        model="test-claude-model",
        parsed_output=build_report(),
        stop_reason="max_tokens",
        usage=SimpleNamespace(
            input_tokens=300,
            output_tokens=2_048,
        ),
    )
    fake_client, _ = build_fake_client(response)

    client = AnthropicRiskSynthesisClient(
        config=build_config(),
        run_id="run-test-004",
        client=fake_client,
    )

    with pytest.raises(
        AnthropicClientResponseError,
        match="exceeded the output-token limit",
    ):
        await client.synthesize_release_risk(
            system_prompt="Return structured output.",
            user_prompt="Trusted release evidence.",
            prompt_version="release-risk-synthesis-v1",
        )
