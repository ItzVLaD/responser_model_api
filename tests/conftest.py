"""Shared test fixtures for the model API test suite."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _isolate_logging(monkeypatch, tmp_path):
    """Keep tests from writing to the real ./logs directory.

    Points the log file at a temp path and clears any handlers left over from a
    previous test so each test configures logging fresh.
    """
    monkeypatch.setenv("RESPONSER_LOG_FILE", str(tmp_path / "test.log"))
    logging.getLogger("responser.model_api").handlers.clear()
    yield
    logging.getLogger("responser.model_api").handlers.clear()
