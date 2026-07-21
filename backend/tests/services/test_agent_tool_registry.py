"""Tests for the deterministic AgentFlow tool registry."""

from typing import cast

import pytest

from app.schemas.agent_tool import (
    AgentToolEffect,
    AgentToolName,
)
from app.services.agent_tool_registry import (
    AgentToolNotRegisteredError,
    AgentToolRegistry,
)


def test_returns_registered_tool_definition() -> None:
    """A registered tool should return trusted immutable metadata."""
    registry = AgentToolRegistry()

    definition = registry.get_definition(
        AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
    )

    assert definition.name is AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
    assert definition.effect is AgentToolEffect.READ_ONLY
    assert definition.default_timeout_seconds == 30


def test_default_listing_excludes_side_effect_tools() -> None:
    """Automatic planners must see read-only tools by default."""
    registry = AgentToolRegistry()

    definitions = registry.list_definitions()

    assert definitions
    assert all(
        definition.effect is AgentToolEffect.READ_ONLY
        for definition in definitions
    )
    assert AgentToolName.SEND_APPROVED_SLACK_ALERT not in {
        definition.name for definition in definitions
    }


def test_explicit_listing_can_include_policy_gated_side_effects() -> None:
    """Policy-aware callers may inspect side-effecting tool metadata."""
    registry = AgentToolRegistry()

    definitions = registry.list_definitions(include_side_effects=True)

    definition_by_name = {
        definition.name: definition
        for definition in definitions
    }

    slack_definition = definition_by_name[
        AgentToolName.SEND_APPROVED_SLACK_ALERT
    ]

    assert len(definitions) == len(AgentToolName)
    assert slack_definition.effect is AgentToolEffect.SIDE_EFFECT
    assert slack_definition.requires_human_approval is True


def test_registry_listing_is_deterministically_sorted() -> None:
    """Stable ordering keeps prompts and evaluation fixtures reproducible."""
    registry = AgentToolRegistry()

    definitions = registry.list_definitions(include_side_effects=True)
    tool_names = [definition.name.value for definition in definitions]

    assert tool_names == sorted(tool_names)


def test_raises_for_unregistered_tool() -> None:
    """Unknown planner-generated tool names must fail closed."""
    registry = AgentToolRegistry()
    unknown_tool = cast(AgentToolName, "unknown_tool")

    with pytest.raises(
        AgentToolNotRegisteredError,
        match="Agent tool is not registered",
    ):
        registry.get_definition(unknown_tool)
