"""Map release-risk workflow results into the public API response."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from app.schemas.risk import ReleaseRunRiskResponse
from app.services.hitl_approval_decision_service import (
    HITLApprovalDecisionService,
)
from app.services.risk_feature_extraction_service import (
    RiskFeatureExtractionService,
)
from app.services.rule_based_risk_scoring_service import (
    RuleBasedRiskScoringService,
)


def merge_workflow_knowledge_context(
    result: object,
    workflow_state: Mapping[str, object],
) -> object:
    """Merge top-level workflow Knowledge Agent fields into result data."""

    knowledge_keys = (
        "knowledge_query",
        "knowledge_results",
        "knowledge_status",
        "knowledge_error",
        "risk_score",
        "approval_policy_version",
        "approval_reason",
        "approval_required",
        "risk_features",
    )

    knowledge_fields: dict[str, object] = {}

    for key in knowledge_keys:
        if key not in workflow_state:
            continue

        value = workflow_state[key]

        if hasattr(value, "value"):
            value = value.value

        knowledge_fields[key] = value

    if not knowledge_fields:
        return result

    if hasattr(result, "model_dump"):
        result_data = result.model_dump()
    elif isinstance(result, Mapping):
        result_data = dict(result)
    else:
        return result

    result_data.update(knowledge_fields)
    return result_data


def extract_risk_result_from_workflow_state(
    workflow_state: object,
) -> object | None:
    """Extract the release-risk result from LangGraph workflow state."""

    result_keys = (
        "risk_result",
        "release_risk_result",
        "release_run_risk_result",
        "release_risk_response",
        "final_result",
        "result",
        "response",
    )

    response_shape_keys = {
        "release_run",
        "github",
        "github_summary",
        "jira",
        "jira_summary",
        "release_summary",
    }

    if isinstance(workflow_state, Mapping):
        for key in result_keys:
            result = workflow_state.get(key)

            if result is not None:
                return merge_workflow_knowledge_context(
                    result=result,
                    workflow_state=workflow_state,
                )

        if response_shape_keys.issubset(workflow_state.keys()):
            return merge_workflow_knowledge_context(
                result=workflow_state,
                workflow_state=workflow_state,
            )

        return None

    for key in result_keys:
        if hasattr(workflow_state, key):
            result = cast(object | None, getattr(workflow_state, key))

            if result is not None:
                return result

    if hasattr(workflow_state, "model_dump"):
        dumped_state = workflow_state.model_dump()

        for key in result_keys:
            result = dumped_state.get(key)

            if result is not None:
                return merge_workflow_knowledge_context(
                    result=result,
                    workflow_state=dumped_state,
                )

        if response_shape_keys.issubset(dumped_state.keys()):
            return merge_workflow_knowledge_context(
                result=dumped_state,
                workflow_state=dumped_state,
            )

    return None


def to_release_run_risk_response(result: object) -> ReleaseRunRiskResponse:
    """Convert a workflow result into the public release-risk response."""

    if hasattr(result, "model_dump"):
        result_data = result.model_dump(mode="python")
    elif isinstance(result, Mapping):
        result_data = dict(result)
    else:
        return ReleaseRunRiskResponse.model_validate(result)

    has_existing_scoring = (
        result_data.get("risk_features") is not None and result_data.get("risk_score") is not None
    )

    if has_existing_scoring:
        if result_data.get("approval_policy_version") is None:
            approval_decision = HITLApprovalDecisionService().determine_approval(
                result_data.get("risk_score"),
                run_id=extract_scoring_run_id(result_data),
            )
            result_data.update(
                {
                    "approval_required": (approval_decision.approval_required),
                    "approval_reason": approval_decision.approval_reason,
                    "approval_policy_version": (approval_decision.approval_policy_version),
                }
            )

        return ReleaseRunRiskResponse.model_validate(result_data)

    run_id = extract_scoring_run_id(result_data)

    risk_features = RiskFeatureExtractionService().extract_from_payload(
        result_data,
        run_id=run_id,
    )
    risk_score = RuleBasedRiskScoringService().score_release(
        risk_features,
        run_id=run_id,
    )

    enriched_result = {
        **result_data,
        "risk_features": risk_features.model_dump(mode="python"),
        "risk_score": risk_score.model_dump(mode="python"),
    }

    return ReleaseRunRiskResponse.model_validate(enriched_result)


def extract_scoring_run_id(
    result_data: Mapping[str, object],
) -> str | None:
    """Extract a safe workflow run ID for scoring logs."""

    release_run = result_data.get("release_run")

    if hasattr(release_run, "model_dump"):
        release_run = release_run.model_dump(mode="python")

    if not isinstance(release_run, Mapping):
        return None

    run_id = release_run.get("run_id")

    if isinstance(run_id, str) and run_id.strip():
        return run_id

    return None
