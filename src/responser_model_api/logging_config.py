"""Logging setup for the model API.

Centralizes logging configuration so behaviour is consistent and tunable via the
environment, which is useful when debugging or iterating on how the model
responds (you can crank up the level to see full prompts and raw completions).

Environment variables:
  RESPONSER_LOG_LEVEL    Logging level (DEBUG, INFO, WARNING, ...). Default INFO.
  RESPONSER_LOG_FILE     Optional path to also write logs to a file.
  RESPONSER_LOG_PROMPTS  If "true", DEBUG logs include the full prompt messages
                         and raw model output. Off by default to avoid writing
                         private chat content to logs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

_LOGGER_NAME = "responser.model_api"

# Whether full prompts / raw completions may be logged at DEBUG level. Off by
# default: prompts contain private chat content and should not be logged unless
# explicitly opted in for debugging.
LOG_PROMPTS = os.environ.get("RESPONSER_LOG_PROMPTS", "false").lower() == "true"


def configure_logging() -> logging.Logger:
    """Configure and return the model API's root logger (idempotent)."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:  # already configured (e.g. reload); do not double-add
        return logger

    level_name = os.environ.get("RESPONSER_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file: Optional[str] = os.environ.get("RESPONSER_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the model API logger, configuring it on first use."""
    return configure_logging()
