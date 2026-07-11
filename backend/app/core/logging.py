import json
import logging
import sys
from typing import Any

from opentelemetry import trace


class StructuredJsonFormatter(logging.Formatter):
    """Format log records as structured JSON for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record as a JSON string."""
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        log_record.update(_get_current_trace_context())

        for key, value in record.__dict__.items():
            if key not in _reserved_log_record_fields():
                log_record[key] = value

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str)


def _get_current_trace_context() -> dict[str, str]:
    """Return active OpenTelemetry trace identifiers for log correlation.

    When no valid span is active, return an empty dictionary. This keeps local
    tests and development logs working even when tracing is disabled.
    """
    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()

    if not span_context.is_valid:
        return {}

    return {
        "trace_id": format(span_context.trace_id, "032x"),
        "span_id": format(span_context.span_id, "016x"),
    }


def _reserved_log_record_fields() -> set[str]:
    """Return standard logging fields that should not be duplicated."""
    return {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application-wide structured JSON logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured application logger."""
    return logging.getLogger(name)
