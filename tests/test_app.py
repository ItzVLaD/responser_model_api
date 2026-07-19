"""Smoke tests for the model API.

The Ollama generator is stubbed so the tests run without a live model.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from responser_model_api import app as app_module
from responser_model_api.schemas import GeneratedReply


def test_health() -> None:
    client = TestClient(app_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "personality" in body


def test_generate_reply(monkeypatch) -> None:
    def fake_generate(snapshot, config):
        return GeneratedReply(text="hello there", model_name=config.model_name)

    monkeypatch.setattr(app_module._generator, "generate", fake_generate)

    client = TestClient(app_module.app)
    payload = {
        "snapshot": {
            "chat": {"raw_id": "1", "title": "Test", "has_unread": True},
            "messages": [{"sender_type": "other", "text": "hi"}],
        },
        "dry_run": True,
    }
    resp = client.post("/generate_reply", json=payload)
    assert resp.status_code == 200
    assert resp.json()["text"] == "hello there"
