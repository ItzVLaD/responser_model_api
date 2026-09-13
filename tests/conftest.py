"""Shared test fixtures for the model API test suite."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Application imports configure logging during collection, before autouse
# fixtures run. Disable file output now so offline tests never open live logs.
os.environ["RESPONSER_LOG_FILE"] = ""
os.environ["RESPONSER_CONTEXT_TRACE"] = "false"


@pytest.fixture(autouse=True)
def _isolate_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Keep tests from writing to the real ./logs directory.

    Points the log file at a temp path and clears any handlers left over from a
    previous test so each test configures logging fresh.
    """
    monkeypatch.setenv("RESPONSER_LOG_FILE", str(tmp_path / "test.log"))
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE", "false")
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE_DIR", str(tmp_path / "context-traces"))
    _close_handlers()
    yield
    _close_handlers()


def _close_handlers() -> None:
    """Release file descriptors rather than merely detaching their handlers."""
    logger = logging.getLogger("responser.model_api")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
