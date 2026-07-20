"""Logging setup for the model API.

Centralizes logging configuration so behaviour is consistent and tunable via the
environment, which is useful when debugging or iterating on how the model
responds (you can crank up the level to see full prompts and raw completions).

Environment variables:
  RESPONSER_LOG_LEVEL    Logging level (DEBUG, INFO, WARNING, ...). Default INFO.
  RESPONSER_LOG_FILE     Path of the log file. Defaults to ``logs/model_api.log``.
                         Set to an empty string to disable file logging.
  RESPONSER_LOG_PROMPTS  If "true", DEBUG logs include the full prompt messages
                         and raw model output. Off by default to avoid writing
                         private chat content to logs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOGGER_NAME = "responser.model_api"

# Whether full prompts / raw completions may be logged at DEBUG level. Off by
# default: prompts contain private chat content and should not be logged unless
# explicitly opted in for debugging.
LOG_PROMPTS = os.environ.get("RESPONSER_LOG_PROMPTS", "false").lower() == "true"

# Default log file, relative to the project root (two levels up from this file).
# Logs are written here unless RESPONSER_LOG_FILE is set to something else, or to
# an empty string to disable file logging entirely.
_DEFAULT_LOG_FILE = str(Path(__file__).resolve().parents[2] / "logs" / "model_api.log")


def _resolve_log_file() -> str:
    """Return the log file path, or "" if file logging is disabled."""
    # `os.environ.get(..., default)` returns the default only when the var is
    # unset; an explicit empty string disables file logging.
    return os.environ.get("RESPONSER_LOG_FILE", _DEFAULT_LOG_FILE)


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

    # Logs are persisted to a file by default so past runs can be inspected.
    log_file = _resolve_log_file()
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the model API logger, configuring it on first use."""
    return configure_logging()
