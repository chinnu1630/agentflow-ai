"""Unit tests for release-risk workflow response mapping."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.agent_query import ResponseDepth
from app.services.release_risk_response_mapper import (
    extract_risk_result_from_workflow_state,
    extract_scoring_run_id,
    merge_workflow_knowledge_context,
)


class _ReleaseRunPayload(BaseModel):
    """Minimal release-run payload used by mapper tests."""

    run_id: str


def test_merges_workflow_knowledge_context_into_mapping() -> None:
    """Top-level workflow fields should be merged into result data."""

    original_result = {
        "release_run": {
            "run_id": "release-run-123",
        }
    }
    workflow_state = {
        "knowledge_query": "payment rollback procedure",
        "knowledge_status": ResponseDepth.DEEP,
        "approval_required": True,
    }

    merged_result = merge_workflow_knowledge_context(
        result=original_result,
        workflow_state=workflow_state,
    )

    assert isinstance(merged_result, dict)
    assert merged_result["knowledge_query"] == "payment rollback procedure"
    assert merged_result["knowledge_status"] == ResponseDepth.DEEP.value
    assert merged_result["approval_required"] is True
    assert "knowledge_query" not in original_result


def test_returns_original_result_when_no_context_exists() -> None:
    """Mapper should return the original object when nothing must be merged."""

    result = {"release_run": {"run_id": "release-run-123"}}

    merged_result = merge_workflow_knowledge_context(
        result=result,
        workflow_state={"unrelated_field": "ignored"},
    )

    assert merged_result is result


def test_extracts_nested_workflow_result_and_merges_context() -> None:
    """Known result keys should be extracted from workflow state."""

    workflow_state = {
        "risk_result": {
            "release_run": {
                "run_id": "release-run-123",
            }
        },
        "knowledge_query": "payment service risks",
        "approval_required": False,
    }

    result = extract_risk_result_from_workflow_state(workflow_state)

    assert isinstance(result, dict)
    assert result["release_run"]["run_id"] == "release-run-123"
    assert result["knowledge_query"] == "payment service risks"
    assert result["approval_required"] is False


def test_extracts_public_response_shape_from_full_state() -> None:
    """A workflow state already matching the public shape should be returned."""

    workflow_state = {
        "release_run": {},
        "github": {},
        "github_summary": {},
        "jira": {},
        "jira_summary": {},
        "release_summary": {},
        "knowledge_status": "success",
    }

    result = extract_risk_result_from_workflow_state(workflow_state)

    assert isinstance(result, dict)
    assert result["knowledge_status"] == "success"
    assert result["release_summary"] == {}


def test_returns_none_when_workflow_contains_no_result() -> None:
    """Unknown workflow state shapes should not produce a response."""

    result = extract_risk_result_from_workflow_state(
        {
            "status": "failed",
            "errors": ["Release run not found."],
        }
    )

    assert result is None


def test_extracts_scoring_run_id_from_mapping() -> None:
    """Scoring mapper should extract a non-empty workflow run ID."""

    run_id = extract_scoring_run_id(
        {
            "release_run": {
                "run_id": "release-run-123",
            }
        }
    )

    assert run_id == "release-run-123"


def test_extracts_scoring_run_id_from_pydantic_model() -> None:
    """Scoring mapper should support nested Pydantic release-run data."""

    run_id = extract_scoring_run_id(
        {
            "release_run": _ReleaseRunPayload(
                run_id="release-run-456",
            )
        }
    )

    assert run_id == "release-run-456"


def test_returns_none_for_invalid_scoring_run_id() -> None:
    """Missing or whitespace-only run IDs should not enter logs."""

    assert extract_scoring_run_id({}) is None
    assert extract_scoring_run_id({"release_run": {}}) is None
    assert extract_scoring_run_id({"release_run": {"run_id": "   "}}) is None
