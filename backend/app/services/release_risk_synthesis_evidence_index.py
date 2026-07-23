"""Build citation-specific trusted evidence for release-risk synthesis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TypeAlias

from app.schemas.llm_risk_synthesis import SynthesisEvidenceSource
from app.schemas.risk import ReleaseRunRiskResponse

SynthesisCitationKey: TypeAlias = tuple[SynthesisEvidenceSource, str]


class ReleaseRiskSynthesisEvidenceIndex:
    """Index trusted workflow evidence by synthesis citation identity."""

    def build(
        self,
        release_risk: ReleaseRunRiskResponse,
    ) -> dict[SynthesisCitationKey, str]:
        """Return citation keys mapped to their available trusted evidence text.

        Complexity:
            Time: O(e), where e is the amount of trusted workflow evidence.
            Space: O(e) for citation-specific normalized evidence text.
        """
        text_by_key: defaultdict[SynthesisCitationKey, list[str]] = defaultdict(list)

        for risk in release_risk.release_summary.top_risks:
            source = (
                SynthesisEvidenceSource.GITHUB_PULL_REQUEST
                if risk.source_type == "github_pull_request"
                else SynthesisEvidenceSource.JIRA_ISSUE
            )
            self._append(
                text_by_key,
                (source, risk.source_id),
                risk.title,
                risk.reason,
                *self._evidence_values(risk.evidence),
            )

        for pull_request in release_risk.github.risk_results:
            pull_request_key = (
                SynthesisEvidenceSource.GITHUB_PULL_REQUEST,
                pull_request.source_id,
            )

            for signal in pull_request.signals:
                self._append(
                    text_by_key,
                    pull_request_key,
                    signal.title,
                    signal.description,
                    *self._evidence_values(signal.evidence),
                )
                self._append(
                    text_by_key,
                    (
                        SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE,
                        signal.rule_id,
                    ),
                    signal.title,
                    signal.description,
                    *self._evidence_values(signal.evidence),
                )

        for issue in release_risk.jira.issues:
            issue_key = (
                SynthesisEvidenceSource.JIRA_ISSUE,
                issue.issue_key,
            )
            self._append(text_by_key, issue_key, issue.title)

            for signal in issue.signals:
                self._append(
                    text_by_key,
                    issue_key,
                    signal.title,
                    signal.description,
                    *self._evidence_values(signal.evidence),
                )
                self._append(
                    text_by_key,
                    (
                        SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE,
                        signal.rule_id,
                    ),
                    signal.title,
                    signal.description,
                    *self._evidence_values(signal.evidence),
                )

        for signal in release_risk.jira.signals:
            self._append(
                text_by_key,
                (
                    SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE,
                    signal.rule_id,
                ),
                signal.title,
                signal.description,
                *self._evidence_values(signal.evidence),
            )

        for result in release_risk.knowledge_results:
            evidence_text = (
                result.title,
                result.content,
                result.source_type,
                *self._evidence_values(result.metadata),
            )

            if result.chunk_id is not None:
                self._append(
                    text_by_key,
                    (
                        SynthesisEvidenceSource.ENGINEERING_DOCUMENT,
                        str(result.chunk_id),
                    ),
                    *evidence_text,
                )

            if result.document_id is not None:
                self._append(
                    text_by_key,
                    (
                        SynthesisEvidenceSource.ENGINEERING_DOCUMENT,
                        str(result.document_id),
                    ),
                    *evidence_text,
                )

        return {
            key: " ".join(parts)
            for key, parts in text_by_key.items()
        }

    @staticmethod
    def _append(
        index: defaultdict[SynthesisCitationKey, list[str]],
        key: SynthesisCitationKey,
        *values: object,
    ) -> None:
        """Append non-empty evidence values to one citation record."""
        for value in values:
            if value is None:
                continue

            normalized = " ".join(str(value).split())

            if normalized and normalized not in index[key]:
                index[key].append(normalized)

    @staticmethod
    def _evidence_values(
        evidence: Mapping[str, object],
    ) -> Iterable[object]:
        """Yield safe scalar evidence keys and values for lexical evaluation."""
        for key, value in evidence.items():
            yield key

            if isinstance(value, (str, bool, int, float)):
                yield value
