"""Build citation-specific trusted evidence for dynamic agent synthesis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from pydantic import JsonValue

from app.schemas.agent_execution_result import AgentExecutionResult
from app.schemas.agent_tool import AgentToolName, AgentToolResult

AgentDynamicCitationKey: TypeAlias = tuple[str, str]


class AgentDynamicSynthesisEvidenceIndex:
    """Index executed tool evidence by exact dynamic citation identity."""

    def build(
        self,
        execution_result: AgentExecutionResult,
    ) -> dict[AgentDynamicCitationKey, str]:
        """Return citation keys mapped to matching trusted tool-output text.

        Collection-returning tools are matched by source ID so one citation
        cannot borrow facts from another result in the same tool response.

        Complexity:
            Time: O(t + e + j), where t is tool results, e is citations, and
            j is traversed bounded JSON output.
            Space: O(e + j) for citation-specific evidence text.
        """
        text_by_key: defaultdict[
            AgentDynamicCitationKey,
            list[str],
        ] = defaultdict(list)

        for result in execution_result.tool_results:
            for evidence in result.evidence:
                key = (evidence.source_type, evidence.source_id)
                matched_output = self._matching_output(
                    result=result,
                    source_id=evidence.source_id,
                )

                self._append(
                    text_by_key,
                    key,
                    evidence.title,
                    evidence.source_url,
                    self._flatten_json(matched_output),
                )

        return {
            key: " ".join(parts)
            for key, parts in text_by_key.items()
        }

    def _matching_output(
        self,
        *,
        result: AgentToolResult,
        source_id: str,
    ) -> JsonValue:
        """Return only the output portion belonging to one citation."""
        if (
            result.tool_name
            is AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
        ):
            return self._find_mapping_in_collection(
                result.output.get("results"),
                identifier_key="chunk_id",
                source_id=source_id,
            )

        if result.tool_name is AgentToolName.LOOKUP_RELEASE_HISTORY:
            return self._find_mapping_in_collection(
                result.output.get("releases"),
                identifier_key="release_run_id",
                source_id=source_id,
            )

        if result.tool_name is AgentToolName.LOOKUP_SIMILAR_RELEASE:
            release = result.output.get("release")

            if (
                isinstance(release, dict)
                and str(release.get("release_run_id", "")) == source_id
            ):
                return release

            return {}

        if len(result.evidence) == 1:
            return result.output

        return {}

    @staticmethod
    def _find_mapping_in_collection(
        value: JsonValue | None,
        *,
        identifier_key: str,
        source_id: str,
    ) -> JsonValue:
        """Return the collection item whose trusted identifier matches."""
        if not isinstance(value, list):
            return {}

        for item in value:
            if (
                isinstance(item, dict)
                and str(item.get(identifier_key, "")) == source_id
            ):
                return item

        return {}

    @classmethod
    def _flatten_json(
        cls,
        value: JsonValue,
        *,
        depth: int = 0,
    ) -> str:
        """Flatten bounded JSON into deterministic lexical evidence text."""
        if depth >= 8:
            return ""

        if value is None:
            return ""

        if isinstance(value, bool):
            return str(value).lower()

        if isinstance(value, (str, int, float)):
            return str(value)

        if isinstance(value, Mapping):
            parts: list[str] = []

            for key, item in value.items():
                parts.append(str(key))
                parts.append(
                    cls._flatten_json(item, depth=depth + 1)
                )

            return " ".join(part for part in parts if part)

        if isinstance(value, Sequence):
            return " ".join(
                part
                for item in value
                if (
                    part := cls._flatten_json(
                        item,
                        depth=depth + 1,
                    )
                )
            )

        return ""

    @staticmethod
    def _append(
        index: defaultdict[AgentDynamicCitationKey, list[str]],
        key: AgentDynamicCitationKey,
        *values: object,
    ) -> None:
        """Append normalized non-empty values without duplication."""
        for value in values:
            if value is None:
                continue

            normalized = " ".join(str(value).split())

            if normalized and normalized not in index[key]:
                index[key].append(normalized)
