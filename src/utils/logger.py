"""
Structured logging for the Personal Knowledge Assistant.

Provides a configured logger with consistent formatting across all modules.
Uses Python's built-in logging module with sensible defaults.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing document: %s", doc_name)
    logger.debug("Generated %d chunks", len(chunks))
"""

import logging
import sys
from typing import Optional


# Default format: timestamp - module - level - message
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: Optional[int] = None,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level. Defaults to INFO.
        fmt: Log message format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(level or logging.INFO)

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level or logging.INFO)

        formatter = logging.Formatter(
            fmt=fmt or DEFAULT_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def set_global_level(level: int) -> None:
    """
    Set the logging level for all PKA loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.WARNING).
    """
    # Update the root PKA logger and all children
    root = logging.getLogger("src")
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)
