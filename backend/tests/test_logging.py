import json
import logging

from app.core.logging import StructuredJsonFormatter, get_logger, setup_logging


def test_structured_json_formatter_outputs_valid_json() -> None:
    """StructuredJsonFormatter should return a valid JSON log string."""
    formatter = StructuredJsonFormatter()

    record = logging.LogRecord(
        name="agentflow.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test log message",
        args=(),
        exc_info=None,
    )

    formatted_log = formatter.format(record)
    log_data = json.loads(formatted_log)

    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "agentflow.test"
    assert log_data["message"] == "Test log message"
    assert "timestamp" in log_data


def test_setup_logging_adds_handler() -> None:
    """setup_logging should configure at least one root logging handler."""
    setup_logging()

    root_logger = logging.getLogger()

    assert len(root_logger.handlers) == 1


def test_get_logger_returns_named_logger() -> None:
    """get_logger should return a logger with the requested name."""
    logger = get_logger("agentflow.health")

    assert logger.name == "agentflow.health"