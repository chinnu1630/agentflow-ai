"""Finalize release-risk workflow execution with persistence and audit events."""

from __future__ import annotations

from uuid import UUID

from app.observability.tracing import start_business_span
from app.repositories.release_run_approval_repository import (
    CreateReleaseRunApprovalCommand,
    ReleaseRunApprovalRepository,
    ReleaseRunApprovalStatus,
)
from app.repositories.release_run_event_repository import (
    CreateReleaseRunEventCommand,
    ReleaseRunEventRepository,
)
from app.repositories.release_run_repository import ReleaseRunRepository
from app.repositories.release_run_risk_snapshot_repository import (
    CreateReleaseRunRiskSnapshotCommand,
    ReleaseRunRiskSnapshotRepository,
)
from app.schemas.risk import ReleaseRunRiskResponse


class ReleaseRiskExecutionFinalizer:
    """Persist and audit the result of a release-risk workflow."""

    def __init__(
        self,
        *,
        release_run_repository: ReleaseRunRepository,
        approval_repository: ReleaseRunApprovalRepository,
        event_repository: ReleaseRunEventRepository,
        risk_snapshot_repository: ReleaseRunRiskSnapshotRepository,
    ) -> None:
        """Initialize the finalizer.

        Args:
            release_run_repository: Repository for release-run status updates.
            approval_repository: Repository for HITL approval requests.
            event_repository: Repository for release-run audit events.
            risk_snapshot_repository: Repository for trusted risk snapshots.
        """

        self._release_run_repository = release_run_repository
        self._approval_repository = approval_repository
        self._event_repository = event_repository
        self._risk_snapshot_repository = risk_snapshot_repository

    async def finalize(
        self,
        *,
        release_run_id: UUID,
        response: ReleaseRunRiskResponse,
    ) -> ReleaseRunRiskResponse:
        """Persist audit events, approval state, and the final risk snapshot.

        Args:
            release_run_id: Release run being finalized.
            response: Validated release-risk workflow response.

        Returns:
            Response enriched with approval request details when required.
        """

        await self._record_scoring_audit_events(
            release_run_id=release_run_id,
            response=response,
        )

        finalized_response = await self._ensure_pending_approval_request(
            release_run_id=release_run_id,
            response=response,
        )

        await self._persist_release_risk_snapshot(
            release_run_id=release_run_id,
            response=finalized_response,
        )

        return finalized_response

    async def _persist_release_risk_snapshot(
        self,
        *,
        release_run_id: UUID,
        response: ReleaseRunRiskResponse,
    ) -> None:
        """Persist the trusted backend-generated release-risk snapshot."""

        with start_business_span(
            "snapshot.persist",
            {
                "release_run_id": str(release_run_id),
                "approval_required": response.approval_required is True,
                "overall_severity": _safe_enum_value(response.release_summary.overall_severity),
            },
        ):
            approval_required = response.approval_required is True
            approval_status_at_snapshot = response.approval_status

            if approval_status_at_snapshot is None:
                approval_status_at_snapshot = (
                    ReleaseRunApprovalStatus.PENDING.value if approval_required else "not_required"
                )

            snapshot = await self._risk_snapshot_repository.create_snapshot(
                CreateReleaseRunRiskSnapshotCommand(
                    release_run_id=release_run_id,
                    risk_payload=response.model_dump(mode="json"),
                    overall_severity=_safe_enum_value(response.release_summary.overall_severity),
                    approval_required=approval_required,
                    approval_status_at_snapshot=approval_status_at_snapshot,
                )
            )

            await self._event_repository.create(
                CreateReleaseRunEventCommand(
                    release_run_id=release_run_id,
                    event_type="release_risk_snapshot_created",
                    event_status="success",
                    message=("Trusted release-risk report snapshot was persisted."),
                    metadata_json={
                        "snapshot_id": str(snapshot.id),
                        "snapshot_version": snapshot.snapshot_version,
                        "overall_severity": snapshot.overall_severity,
                        "approval_required": snapshot.approval_required,
                        "approval_status_at_snapshot": (snapshot.approval_status_at_snapshot),
                    },
                )
            )

    async def _ensure_pending_approval_request(
        self,
        *,
        release_run_id: UUID,
        response: ReleaseRunRiskResponse,
    ) -> ReleaseRunRiskResponse:
        """Create or reuse a pending HITL approval request when required."""

        with start_business_span(
            "approval.ensure_pending",
            {
                "release_run_id": str(release_run_id),
                "approval_required": response.approval_required is True,
            },
        ):
            if response.approval_required is not True:
                return response

            latest_approval = await self._approval_repository.get_latest_by_release_run_id(
                release_run_id
            )

            if (
                latest_approval is not None
                and latest_approval.approval_status == ReleaseRunApprovalStatus.PENDING.value
            ):
                await self._mark_release_run_waiting_for_approval(
                    release_run_id=release_run_id,
                    approval_request_id=latest_approval.id,
                    approval_status=latest_approval.approval_status,
                    approval_policy_version=(latest_approval.approval_policy_version),
                )

                return response.model_copy(
                    update={
                        "approval_request_id": latest_approval.id,
                        "approval_status": latest_approval.approval_status,
                    }
                )

            approval = await self._approval_repository.create_pending(
                CreateReleaseRunApprovalCommand(
                    release_run_id=release_run_id,
                    approval_reason=(
                        response.approval_reason
                        or ("Release requires human approval before proceeding.")
                    ),
                    approval_policy_version=(response.approval_policy_version or "hitl_policy_v1"),
                    requested_by=response.release_run.requested_by,
                )
            )

            await self._event_repository.create(
                CreateReleaseRunEventCommand(
                    release_run_id=release_run_id,
                    event_type="approval_request_created",
                    event_status="success",
                    message="Pending release approval request was created.",
                    metadata_json={
                        "approval_request_id": str(approval.id),
                        "approval_status": approval.approval_status,
                        "approval_policy_version": (approval.approval_policy_version),
                        "approval_reason_present": bool(approval.approval_reason),
                    },
                )
            )

            await self._mark_release_run_waiting_for_approval(
                release_run_id=release_run_id,
                approval_request_id=approval.id,
                approval_status=approval.approval_status,
                approval_policy_version=approval.approval_policy_version,
            )

            return response.model_copy(
                update={
                    "approval_request_id": approval.id,
                    "approval_status": approval.approval_status,
                    "release_run": response.release_run.model_copy(
                        update={"status": "waiting_for_approval"}
                    ),
                }
            )

    async def _mark_release_run_waiting_for_approval(
        self,
        *,
        release_run_id: UUID,
        approval_request_id: UUID,
        approval_status: str,
        approval_policy_version: str,
    ) -> None:
        """Mark a release run as waiting for human approval."""

        await self._release_run_repository.update_status(
            release_run_id=release_run_id,
            status="waiting_for_approval",
        )

        await self._event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="release_run_waiting_for_approval",
                event_status="success",
                message="Release run is waiting for human approval.",
                metadata_json={
                    "approval_request_id": str(approval_request_id),
                    "approval_status": approval_status,
                    "approval_policy_version": approval_policy_version,
                },
            )
        )

    async def _record_scoring_audit_events(
        self,
        *,
        release_run_id: UUID,
        response: ReleaseRunRiskResponse,
    ) -> None:
        """Persist safe feature, scoring, and approval audit events."""

        github_risk_count = _count_collection_risks(response.github)
        jira_risk_count = _count_collection_risks(response.jira)

        with start_business_span(
            "risk.scoring_audit",
            {
                "release_run_id": str(release_run_id),
                "github_risk_count": github_risk_count,
                "jira_risk_count": jira_risk_count,
                "total_risk_count": (github_risk_count + jira_risk_count),
                "approval_required": response.approval_required is True,
            },
        ):
            if response.risk_features is None or response.risk_score is None:
                return

            risk_features = response.risk_features
            risk_score = response.risk_score

            await self._event_repository.create(
                CreateReleaseRunEventCommand(
                    release_run_id=release_run_id,
                    event_type="risk_features_extracted",
                    event_status="success",
                    message=("Release-risk scoring features were extracted."),
                    metadata_json={
                        "feature_version": risk_features.feature_version,
                        "total_risk_count": (risk_features.total_risk_count),
                        "github_risk_count": (risk_features.github_risk_count),
                        "jira_risk_count": risk_features.jira_risk_count,
                        "critical_risk_count": (risk_features.critical_risk_count),
                        "high_risk_count": risk_features.high_risk_count,
                        "knowledge_result_count": (risk_features.knowledge_result_count),
                        "knowledge_no_results": (risk_features.knowledge_no_results),
                        "knowledge_failed": (risk_features.knowledge_failed),
                    },
                )
            )

            await self._event_repository.create(
                CreateReleaseRunEventCommand(
                    release_run_id=release_run_id,
                    event_type="release_risk_scored",
                    event_status="success",
                    message=("Release risk was scored using deterministic rule-based scoring."),
                    metadata_json={
                        "scoring_version": risk_score.scoring_version,
                        "feature_version": risk_score.feature_version,
                        "score": risk_score.score,
                        "risk_level": _safe_enum_value(risk_score.risk_level),
                        "recommended_action": _safe_enum_value(risk_score.recommended_action),
                        "reason_count": len(risk_score.reasons),
                        "component_score_count": len(risk_score.component_scores),
                    },
                )
            )

            if response.approval_policy_version is not None:
                await self._event_repository.create(
                    CreateReleaseRunEventCommand(
                        release_run_id=release_run_id,
                        event_type="approval_requirement_determined",
                        event_status="success",
                        message=("HITL approval requirement was determined."),
                        metadata_json={
                            "approval_policy_version": (response.approval_policy_version),
                            "approval_required": (response.approval_required),
                            "approval_reason_present": (response.approval_reason is not None),
                            "risk_level": _safe_enum_value(risk_score.risk_level),
                            "recommended_action": _safe_enum_value(risk_score.recommended_action),
                        },
                    )
                )


def _count_collection_risks(collection: object) -> int:
    """Return a safe risk count from a collection response."""

    for attribute_name in ("risks", "risk_signals", "signals"):
        value = getattr(collection, attribute_name, None)

        if isinstance(value, list):
            return len(value)

    return 0


def _safe_enum_value(value: object) -> str:
    """Return a safe string value for enum-like audit metadata."""

    enum_value = getattr(value, "value", None)

    if enum_value is not None:
        return str(enum_value)

    return str(value)
