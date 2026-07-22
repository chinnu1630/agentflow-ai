"""Central registry for versioned AgentFlow LLM prompts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class LLMPromptName(StrEnum):
    """Stable identifiers for production LLM prompts."""

    AGENT_EXECUTION_PLANNER = "agent_execution_planner"
    AGENT_DYNAMIC_SYNTHESIS = "agent_dynamic_synthesis"
    RELEASE_RISK_SYNTHESIS = "release_risk_synthesis"


class LLMPromptDefinition(BaseModel):
    """Immutable metadata for one governed LLM prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: LLMPromptName
    version: str = Field(min_length=1, max_length=100)
    purpose: str = Field(min_length=1, max_length=500)


AGENT_EXECUTION_PLANNER_PROMPT_VERSION: Final[str] = (
    "agent-execution-planner-v1"
)
AGENT_DYNAMIC_SYNTHESIS_PROMPT_VERSION: Final[str] = (
    "agent-dynamic-synthesis-v1"
)
RELEASE_RISK_SYNTHESIS_PROMPT_VERSION: Final[str] = (
    "release-risk-synthesis-v1"
)


class LLMPromptRegistry:
    """Provide deterministic, immutable prompt-version metadata."""

    _DEFINITIONS: Final[Mapping[LLMPromptName, LLMPromptDefinition]] = (
        MappingProxyType(
            {
                LLMPromptName.AGENT_EXECUTION_PLANNER: LLMPromptDefinition(
                    name=LLMPromptName.AGENT_EXECUTION_PLANNER,
                    version=AGENT_EXECUTION_PLANNER_PROMPT_VERSION,
                    purpose=(
                        "Create bounded read-only execution plans from "
                        "deterministic routing output."
                    ),
                ),
                LLMPromptName.AGENT_DYNAMIC_SYNTHESIS: LLMPromptDefinition(
                    name=LLMPromptName.AGENT_DYNAMIC_SYNTHESIS,
                    version=AGENT_DYNAMIC_SYNTHESIS_PROMPT_VERSION,
                    purpose=(
                        "Produce evidence-grounded manager answers from "
                        "validated dynamic tool results."
                    ),
                ),
                LLMPromptName.RELEASE_RISK_SYNTHESIS: LLMPromptDefinition(
                    name=LLMPromptName.RELEASE_RISK_SYNTHESIS,
                    version=RELEASE_RISK_SYNTHESIS_PROMPT_VERSION,
                    purpose=(
                        "Produce evidence-grounded release-risk "
                        "recommendations from deterministic signals."
                    ),
                ),
            }
        )
    )

    def get_definition(
        self,
        prompt_name: LLMPromptName,
    ) -> LLMPromptDefinition:
        """Return governed metadata for one prompt."""
        return self._DEFINITIONS[prompt_name]

    def list_definitions(self) -> tuple[LLMPromptDefinition, ...]:
        """Return all prompt definitions in deterministic name order."""
        return tuple(
            sorted(
                self._DEFINITIONS.values(),
                key=lambda definition: definition.name.value,
            )
        )
