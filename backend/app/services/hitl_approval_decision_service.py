"""Human approval decision service for AgentFlow AI release-risk workflow.

This module determines whether a release-risk result requires human approval.
It is intentionally deterministic and does not call LLMs, databases, or
external services.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger(__name__)


class HITLApprovalDecision(BaseModel):
    """Deterministic human-in-the-loop approval decision."""

    model_config = ConfigDict(frozen=True)

    approval_policy_version: Literal["hitl_policy_v1"] = "hitl_policy_v1"
    approval_required: bool
    approval_reason: str | None = None


class HITLApprovalDecisionService:
    """Determine whether a release requires human approval."""

    def determine_approval(
        self,
        risk_score: Mapping[str, Any] | object | None,
        *,
        run_id: str | None = None,
    ) -> HITLApprovalDecision:
        """Determine HITL approval requirement from a risk score.

        Args:
            risk_score: Rule-based risk score as a mapping or Pydantic model.
            run_id: Optional workflow run ID used only for safe logs.

        Returns:
            Deterministic HITL approval decision.
        """
        started_at = time.perf_counter()
        score_data = self._to_mapping(risk_score)

        risk_level = self._normalize_text(score_data.get("risk_level"))
        recommended_action = self._normalize_text(
            score_data.get("recommended_action")
        )

        decision = self._build_decision(
            risk_level=risk_level,
            recommended_action=recommended_action,
        )

        logger.info(
            "hitl_approval_decision_created",
            run_id=run_id,
            approval_policy_version=decision.approval_policy_version,
            approval_required=decision.approval_required,
            risk_level=risk_level or "unknown",
            recommended_action=recommended_action or "unknown",
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )

        return decision

    def _build_decision(
        self,
        *,
        risk_level: str,
        recommended_action: str,
    ) -> HITLApprovalDecision:
        """Build deterministic approval decision from normalized score fields."""
        if recommended_action == "block_release":
            return HITLApprovalDecision(
                approval_required=True,
                approval_reason="Release is blocked by deterministic risk scoring.",
            )

        if recommended_action == "partial_data_review":
            return HITLApprovalDecision(
                approval_required=True,
                approval_reason="Release analysis used degraded or partial data.",
            )

        if risk_level in {"critical", "high"}:
            return HITLApprovalDecision(
                approval_required=True,
                approval_reason=f"{risk_level.title()} release risk requires manager approval.",
            )

        if not risk_level or not recommended_action:
            return HITLApprovalDecision(
                approval_required=True,
                approval_reason="Release risk score is unavailable or incomplete.",
            )

        return HITLApprovalDecision(
            approval_required=False,
            approval_reason=None,
        )

    @staticmethod
    def _to_mapping(value: Mapping[str, Any] | object | None) -> dict[str, Any]:
        """Convert mapping-like or Pydantic objects into a dictionary."""
        if value is None:
            return {}

        if isinstance(value, Mapping):
            return dict(value)

        if hasattr(value, "model_dump"):
            dumped = value.model_dump(mode="python")
            if isinstance(dumped, dict):
                return dumped

        return {}

    @staticmethod
    def _normalize_text(value: object) -> str:
        """Normalize string or enum-like values into lowercase text."""
        if value is None:
            return ""

        enum_value = getattr(value, "value", None)
        raw_value = enum_value if enum_value is not None else value

        return str(raw_value).strip().lower()
