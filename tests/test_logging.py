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


def test_respects_log_level_env(monkeypatch, tmp_path) -> None:
    # A fresh logger (cleared handlers) should pick up the env level.
    monkeypatch.setenv("RESPONSER_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RESPONSER_LOG_FILE", str(tmp_path / "test.log"))
    logger = logging.getLogger("responser.model_api")
    logger.handlers.clear()
    configured = configure_logging()
    assert configured.level == logging.DEBUG


def test_logs_are_written_to_file(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "model_api.log"
    monkeypatch.setenv("RESPONSER_LOG_FILE", str(log_file))
    logger = logging.getLogger("responser.model_api")
    logger.handlers.clear()

    configured = configure_logging()
    configured.info("hello file log")
    for handler in configured.handlers:
        handler.flush()

    assert log_file.exists()
    assert "hello file log" in log_file.read_text(encoding="utf-8")


def test_file_logging_can_be_disabled(monkeypatch) -> None:
    # An explicit empty RESPONSER_LOG_FILE disables the file handler.
    monkeypatch.setenv("RESPONSER_LOG_FILE", "")
    logger = logging.getLogger("responser.model_api")
    logger.handlers.clear()

    configured = configure_logging()
    assert not any(isinstance(h, logging.FileHandler) for h in configured.handlers)

