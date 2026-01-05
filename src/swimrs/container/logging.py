"""
Structured logging configuration for SwimContainer.

Provides:
- ContainerLogger: Structured logger with context binding
- get_logger: Get a logger for a specific component
- configure_logging: Configure logging output format

The logging system is designed to work with or without structlog.
When structlog is available, it provides rich structured output.
Otherwise, it falls back to standard logging with JSON formatting.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Try to import structlog for rich structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class ContainerLogger:
    """
    Structured logger for container operations.

    Provides context binding and structured output suitable for
    monitoring and analysis.

    Example:
        log = ContainerLogger("ingestor")
        log = log.bind(source="gridmet", target="meteorology/gridmet/eto")

        log.info("starting_ingestion", n_files=100)
        log.debug("processing_file", file="met_001.parquet", progress="1/100")
        log.info("ingestion_complete", records=10000, duration=12.5)
    """

    def __init__(
        self,
        component: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the logger.

        Args:
            component: Component name (e.g., "ingestor", "calculator")
            context: Initial context bindings
        """
        self._component = component
        self._context = context or {}

        if STRUCTLOG_AVAILABLE:
            self._logger = structlog.get_logger(component)
            if context:
                self._logger = self._logger.bind(**context)
        else:
            self._logger = logging.getLogger(f"swimrs.{component}")

    def bind(self, **kwargs: Any) -> "ContainerLogger":
        """
        Create a new logger with additional context bindings.

        Args:
            **kwargs: Key-value pairs to bind to the logger

        Returns:
            New ContainerLogger with bound context
        """
        new_context = {**self._context, **kwargs}
        return ContainerLogger(self._component, new_context)

    def _format_message(self, event: str, **kwargs: Any) -> str:
        """Format message for standard logging."""
        data = {
            "event": event,
            "component": self._component,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **self._context,
            **kwargs,
        }
        return json.dumps(data)

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log debug message."""
        if STRUCTLOG_AVAILABLE:
            self._logger.debug(event, **kwargs)
        else:
            self._logger.debug(self._format_message(event, **kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        """Log info message."""
        if STRUCTLOG_AVAILABLE:
            self._logger.info(event, **kwargs)
        else:
            self._logger.info(self._format_message(event, **kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log warning message."""
        if STRUCTLOG_AVAILABLE:
            self._logger.warning(event, **kwargs)
        else:
            self._logger.warning(self._format_message(event, **kwargs))

    def error(self, event: str, **kwargs: Any) -> None:
        """Log error message."""
        if STRUCTLOG_AVAILABLE:
            self._logger.error(event, **kwargs)
        else:
            self._logger.error(self._format_message(event, **kwargs))

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        if STRUCTLOG_AVAILABLE:
            self._logger.exception(event, **kwargs)
        else:
            self._logger.exception(self._format_message(event, **kwargs))


def get_logger(component: str) -> ContainerLogger:
    """
    Get a logger for a specific component.

    Args:
        component: Component name (e.g., "ingestor", "calculator", "exporter")

    Returns:
        ContainerLogger instance

    Example:
        log = get_logger("ingestor")
        log.info("starting", source="gridmet")
    """
    return ContainerLogger(component)


def configure_logging(
    level: str = "INFO",
    format: str = "json",
    output: str = "stderr",
) -> None:
    """
    Configure logging output.

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
        format: Output format ("json", "console", "simple")
        output: Output destination ("stderr", "stdout", or file path)

    Example:
        # JSON output for production
        configure_logging(level="INFO", format="json")

        # Pretty console output for development
        configure_logging(level="DEBUG", format="console")

        # Log to file
        configure_logging(level="INFO", format="json", output="/var/log/swimrs.log")
    """
    level_num = getattr(logging, level.upper(), logging.INFO)

    # Configure output handler
    if output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    elif output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(output)

    # Configure formatter
    if format == "json":
        # JSON formatter for machine parsing
        handler.setFormatter(logging.Formatter("%(message)s"))
    elif format == "console":
        # Pretty console format for development
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    else:
        # Simple format
        handler.setFormatter(logging.Formatter("%(message)s"))

    # Configure root logger for swimrs
    root_logger = logging.getLogger("swimrs")
    root_logger.setLevel(level_num)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Configure structlog if available
    if STRUCTLOG_AVAILABLE:
        if format == "json":
            processors = [
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ]
        else:
            processors = [
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="%H:%M:%S"),
                structlog.dev.ConsoleRenderer(colors=True),
            ]

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


# Pre-configured loggers for common components
ingestor_logger = get_logger("ingestor")
calculator_logger = get_logger("calculator")
exporter_logger = get_logger("exporter")
query_logger = get_logger("query")
