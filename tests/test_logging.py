"""Tests for the logging configuration."""

from __future__ import annotations

import logging

from responser_model_api.logging_config import configure_logging, get_logger


def test_configure_logging_is_idempotent() -> None:
    logger = configure_logging()
    count = len(logger.handlers)
    # Calling again must not add duplicate handlers.
    again = configure_logging()
    assert again is logger
    assert len(again.handlers) == count
    assert count >= 1


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger()
    assert logger.name == "responser.model_api"
    assert isinstance(logger, logging.Logger)


def test_respects_log_level_env(monkeypatch) -> None:
    # A fresh logger (cleared handlers) should pick up the env level.
    monkeypatch.setenv("RESPONSER_LOG_LEVEL", "DEBUG")
    logger = logging.getLogger("responser.model_api")
    logger.handlers.clear()
    configured = configure_logging()
    assert configured.level == logging.DEBUG
