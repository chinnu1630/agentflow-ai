"""CI quality gate for deterministic dynamic-agent contracts."""

import pytest

from app.services.agent_dynamic_evaluation_service import (
    DynamicAgentEvaluationReport,
    DynamicAgentEvaluationService,
)
from app.services.agent_query_router import AgentQueryRouter
from tests.fixtures.agent_dynamic_eval_cases import (
    build_dynamic_agent_eval_cases,
)
from tests.services.test_agent_dynamic_evaluation_service import GoldenPlanner

MIN_ROUTING_ACCURACY = 1.0
MIN_TOOL_ACCURACY = 1.0
MIN_SAFETY_ACCURACY = 1.0
MIN_OVERALL_ACCURACY = 1.0


def _format_safe_failure_summary(
    report: DynamicAgentEvaluationReport,
) -> str:
    """Build a CI-safe summary without raw manager queries."""
    failures = [
        {
            "case_name": failure.case_name,
            "reason": failure.reason,
            "expected_intent": failure.expected_intent.value,
            "actual_intent": (
                failure.actual_intent.value
                if failure.actual_intent is not None
                else None
            ),
            "expected_tool_name": (
                failure.expected_tool_name.value
                if failure.expected_tool_name is not None
                else None
            ),
            "actual_tool_names": [
                tool_name.value
                for tool_name in failure.actual_tool_names
            ],
        }
        for failure in report.failed_case_details
    ]

    return (
        "Dynamic-agent quality gate failed. "
        f"routing_accuracy={report.routing_accuracy}, "
        f"tool_accuracy={report.tool_accuracy}, "
        f"safety_accuracy={report.safety_accuracy}, "
        f"overall_accuracy={report.overall_accuracy}, "
        f"failures={failures}"
    )


@pytest.mark.anyio
async def test_dynamic_agent_golden_dataset_quality_gate() -> None:
    """Routing, tool selection, and safety must remain regression-free."""
    service = DynamicAgentEvaluationService(
        router=AgentQueryRouter(),
        planner=GoldenPlanner(),
    )

    report = await service.evaluate(build_dynamic_agent_eval_cases())
    failure_summary = _format_safe_failure_summary(report)

    assert report.routing_accuracy >= MIN_ROUTING_ACCURACY, failure_summary
    assert report.tool_accuracy >= MIN_TOOL_ACCURACY, failure_summary
    assert report.safety_accuracy >= MIN_SAFETY_ACCURACY, failure_summary
    assert report.overall_accuracy >= MIN_OVERALL_ACCURACY, failure_summary
    assert report.failed_cases == 0, failure_summary
