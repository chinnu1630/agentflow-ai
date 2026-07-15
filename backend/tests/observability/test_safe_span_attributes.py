"""Safety tests for AgentFlow OpenTelemetry business span metadata."""

from __future__ import annotations

from pathlib import Path

APP_SOURCE_ROOT = Path("app")

FORBIDDEN_SPAN_ATTRIBUTE_TERMS = {
    "slack_bot_token",
    "SLACK_BOT_TOKEN",
    "bot_token",
    "github_token",
    "GITHUB_TOKEN",
    "jira_token",
    "JIRA_TOKEN",
    "password",
    "secret",
    "raw_payload",
    "risk_payload",
    "message_body",
    "slack_message",
    "document_text",
    "chunk_content",
    "jira_description",
    "pr_description",
    "pull_request_body",
    '"manager_query": manager_query',
}


def _extract_business_span_blocks(source: str) -> list[str]:
    """Extract source blocks that start with start_business_span(...).

    The safety test should inspect only span metadata blocks, not the whole
    source file. Route files may legitimately read secrets from environment
    variables, but those values must never be attached to OpenTelemetry spans.
    """
    lines = source.splitlines()
    blocks: list[str] = []

    for index, line in enumerate(lines):
        if "start_business_span(" not in line:
            continue

        indent_width = len(line) - len(line.lstrip())
        block_lines = [line]

        for following_line in lines[index + 1 :]:
            stripped = following_line.strip()

            if not stripped:
                block_lines.append(following_line)
                continue

            following_indent_width = len(following_line) - len(
                following_line.lstrip()
            )

            if following_indent_width <= indent_width:
                break

            block_lines.append(following_line)

        blocks.append("\n".join(block_lines))

    return blocks


def test_business_span_blocks_do_not_reference_sensitive_terms() -> None:
    """Business span blocks should not attach secrets or raw enterprise content."""
    source_files = sorted(APP_SOURCE_ROOT.rglob("*.py"))
    violations: list[str] = []

    for source_file in source_files:
        source = source_file.read_text()

        for span_block in _extract_business_span_blocks(source):
            for forbidden_term in FORBIDDEN_SPAN_ATTRIBUTE_TERMS:
                if forbidden_term in span_block:
                    violations.append(f"{source_file}: {forbidden_term}")

    assert violations == []


def test_known_business_span_names_are_stable() -> None:
    """Business spans should use stable domain names for trace analysis."""
    combined_source = "\n".join(
        source_file.read_text()
        for source_file in sorted(APP_SOURCE_ROOT.rglob("*.py"))
    )

    expected_spans = {
        "release_run.risks_endpoint",
        "release_risk.workflow",
        "risk.scoring_audit",
        "approval.ensure_pending",
        "snapshot.persist",
        "approval.decision",
        "knowledge.retrieve",
        "slack.release_alert.route",
        "slack.release_alert.duplicate_check",
    }

    for span_name in expected_spans:
        assert f'"{span_name}"' in combined_source
