"""Tests for centralized AgentFlow prompt governance."""

from app.services.llm_prompt_registry import (
    AGENT_DYNAMIC_SYNTHESIS_PROMPT_VERSION,
    AGENT_EXECUTION_PLANNER_PROMPT_VERSION,
    RELEASE_RISK_SYNTHESIS_PROMPT_VERSION,
    LLMPromptName,
    LLMPromptRegistry,
)


def test_registry_returns_governed_prompt_versions() -> None:
    """Every production prompt should have one stable registered version."""
    registry = LLMPromptRegistry()

    assert (
        registry.get_definition(
            LLMPromptName.AGENT_EXECUTION_PLANNER
        ).version
        == AGENT_EXECUTION_PLANNER_PROMPT_VERSION
    )
    assert (
        registry.get_definition(
            LLMPromptName.AGENT_DYNAMIC_SYNTHESIS
        ).version
        == AGENT_DYNAMIC_SYNTHESIS_PROMPT_VERSION
    )
    assert (
        registry.get_definition(
            LLMPromptName.RELEASE_RISK_SYNTHESIS
        ).version
        == RELEASE_RISK_SYNTHESIS_PROMPT_VERSION
    )


def test_registry_lists_unique_prompts_deterministically() -> None:
    """Prompt inventory should be unique and deterministically ordered."""
    definitions = LLMPromptRegistry().list_definitions()

    names = [definition.name for definition in definitions]
    versions = [definition.version for definition in definitions]

    assert names == sorted(names, key=str)
    assert len(names) == len(set(names))
    assert len(versions) == len(set(versions))
    assert len(definitions) == 3
