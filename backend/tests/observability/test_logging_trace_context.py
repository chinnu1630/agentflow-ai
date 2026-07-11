"""Tests for OpenTelemetry trace correlation in structured logs."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

from app.core import logging as app_logging
from app.core.logging import StructuredJsonFormatter


def test_structured_json_formatter_omits_trace_ids_without_active_span() -> None:
    """Logs should not require tracing to be enabled."""
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="agentflow.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="test_log",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "test_log"
    assert "trace_id" not in payload
    assert "span_id" not in payload


def test_structured_json_formatter_adds_trace_ids_from_trace_context() -> None:
    """Logs should include trace correlation IDs when the helper returns them."""
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="agentflow.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=20,
        msg="test_log_with_trace",
        args=(),
        exc_info=None,
    )

    with patch.object(
        app_logging,
        "_get_current_trace_context",
        return_value={
            "trace_id": "1234567890abcdef1234567890abcdef",
            "span_id": "1234567890abcdef",
        },
    ):
        payload = json.loads(formatter.format(record))

    assert payload["message"] == "test_log_with_trace"
    assert payload["trace_id"] == "1234567890abcdef1234567890abcdef"
    assert payload["span_id"] == "1234567890abcdef"


def test_get_current_trace_context_without_active_span_returns_empty_dict() -> None:
    """Trace context helper should be safe when tracing is disabled."""
    assert app_logging._get_current_trace_context() == {}
