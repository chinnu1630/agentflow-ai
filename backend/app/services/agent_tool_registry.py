"""Deterministic registry of planner-selectable AgentFlow tools."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from app.schemas.agent_tool import (
    AgentToolDefinition,
    AgentToolEffect,
    AgentToolName,
)


class AgentToolNotRegisteredError(LookupError):
    """Raised when a requested tool is absent from the trusted registry."""


class AgentToolRegistry:
    """Provide immutable metadata for approved AgentFlow capabilities."""

    _DEFINITIONS: Final[Mapping[AgentToolName, AgentToolDefinition]] = (
        MappingProxyType(
            {
                AgentToolName.RUN_FRESH_RELEASE_RISK_ANALYSIS: (
                    AgentToolDefinition(
                        name=AgentToolName.RUN_FRESH_RELEASE_RISK_ANALYSIS,
                        description=(
                            "Collect current GitHub, Jira, and engineering "
                            "knowledge evidence and run release-risk analysis."
                        ),
                        effect=AgentToolEffect.READ_ONLY,
                        requires_release_run_context=False,
                        requires_human_approval=False,
                        default_timeout_seconds=120,
                    )
                ),
                AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT: AgentToolDefinition(
                    name=AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
                    description=(
                        "Load the latest persisted release-risk snapshot."
                    ),
                    effect=AgentToolEffect.READ_ONLY,
                    requires_release_run_context=True,
                    requires_human_approval=False,
                    default_timeout_seconds=10,
                ),
                AgentToolName.LOOKUP_GITHUB_PULL_REQUEST: AgentToolDefinition(
                    name=AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
                    description=(
                        "Resolve a GitHub pull request from trusted "
                        "release-risk context."
                    ),
                    effect=AgentToolEffect.READ_ONLY,
                    requires_release_run_context=True,
                    requires_human_approval=False,
                    default_timeout_seconds=10,
                ),
                AgentToolName.LOOKUP_JIRA_ISSUE: AgentToolDefinition(
                    name=AgentToolName.LOOKUP_JIRA_ISSUE,
                    description=(
                        "Resolve a Jira issue from trusted release-risk context."
                    ),
                    effect=AgentToolEffect.READ_ONLY,
                    requires_release_run_context=True,
                    requires_human_approval=False,
                    default_timeout_seconds=10,
                ),
                AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE: (
                    AgentToolDefinition(
                        name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
                        description=(
                            "Search trusted engineering documents using "
                            "bounded hybrid retrieval."
                        ),
                        effect=AgentToolEffect.READ_ONLY,
                        requires_release_run_context=False,
                        requires_human_approval=False,
                        default_timeout_seconds=30,
                    )
                ),
                AgentToolName.LOOKUP_RELEASE_HISTORY: AgentToolDefinition(
                    name=AgentToolName.LOOKUP_RELEASE_HISTORY,
                    description=(
                        "Load validated historical release-risk snapshots."
                    ),
                    effect=AgentToolEffect.READ_ONLY,
                    requires_release_run_context=True,
                    requires_human_approval=False,
                    default_timeout_seconds=20,
                ),
                AgentToolName.LOOKUP_SIMILAR_RELEASE: AgentToolDefinition(
                    name=AgentToolName.LOOKUP_SIMILAR_RELEASE,
                    description=(
                        "Find the most similar validated historical release."
                    ),
                    effect=AgentToolEffect.READ_ONLY,
                    requires_release_run_context=True,
                    requires_human_approval=False,
                    default_timeout_seconds=20,
                ),
                AgentToolName.LOOKUP_APPROVAL_STATUS: AgentToolDefinition(
                    name=AgentToolName.LOOKUP_APPROVAL_STATUS,
                    description=(
                        "Load the latest durable human approval decision."
                    ),
                    effect=AgentToolEffect.READ_ONLY,
                    requires_release_run_context=True,
                    requires_human_approval=False,
                    default_timeout_seconds=10,
                ),
                AgentToolName.LOOKUP_SLACK_STATUS: AgentToolDefinition(
                    name=AgentToolName.LOOKUP_SLACK_STATUS,
                    description=(
                        "Load the persisted Slack alert delivery status."
                    ),
                    effect=AgentToolEffect.READ_ONLY,
                    requires_release_run_context=True,
                    requires_human_approval=False,
                    default_timeout_seconds=10,
                ),
                AgentToolName.SEND_APPROVED_SLACK_ALERT: AgentToolDefinition(
                    name=AgentToolName.SEND_APPROVED_SLACK_ALERT,
                    description=(
                        "Send a Slack release alert only after durable "
                        "human approval."
                    ),
                    effect=AgentToolEffect.SIDE_EFFECT,
                    requires_release_run_context=True,
                    requires_human_approval=True,
                    default_timeout_seconds=30,
                ),
            }
        )
    )

    def get_definition(
        self,
        tool_name: AgentToolName,
    ) -> AgentToolDefinition:
        """Return trusted metadata for one registered tool.

        Args:
            tool_name: Typed AgentFlow tool identifier.

        Returns:
            Immutable validated tool metadata.

        Raises:
            AgentToolNotRegisteredError: When the tool is not registered.
        """
        try:
            return self._DEFINITIONS[tool_name]
        except KeyError as exc:
            raise AgentToolNotRegisteredError(
                f"Agent tool is not registered: {tool_name}"
            ) from exc

    def list_definitions(
        self,
        *,
        include_side_effects: bool = False,
    ) -> tuple[AgentToolDefinition, ...]:
        """List tools visible to a bounded execution planner.

        Args:
            include_side_effects: Whether explicitly policy-gated tools should
                be included.

        Returns:
            Registered tool definitions in deterministic name order.
        """
        definitions = (
            definition
            for definition in self._DEFINITIONS.values()
            if (
                include_side_effects
                or definition.effect is AgentToolEffect.READ_ONLY
            )
        )

        return tuple(
            sorted(
                definitions,
                key=lambda definition: definition.name.value,
            )
        )
