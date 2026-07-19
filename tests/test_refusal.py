"""Tests for anti-refusal handling in the reply generator."""

from __future__ import annotations

from responser_model_api.ollama_client import (
    OllamaReplyGenerator,
    _DEFLECTION_FALLBACK,
    _looks_like_refusal,
)
from responser_model_api.config import GenerationSettings
from responser_model_api.personality import Personality
from responser_model_api.schemas import ChatDescriptor, ChatSnapshot, Message


def _snapshot() -> ChatSnapshot:
    return ChatSnapshot(
        chat=ChatDescriptor(raw_id="1", title="Bob", has_unread=True),
        messages=[Message(sender_type="other", text="hey")],
    )


def _generator() -> OllamaReplyGenerator:
    gen = OllamaReplyGenerator.__new__(OllamaReplyGenerator)
    gen._personality = Personality(name="Alex")
    gen._settings = GenerationSettings()
    return gen


def test_looks_like_refusal_detects_boilerplate() -> None:
    assert _looks_like_refusal("I cannot create content that depicts...")
    assert _looks_like_refusal("Is there anything else I can help you with?")
    assert _looks_like_refusal("As an AI, I can't do that")
    assert not _looks_like_refusal("haha nah, you're funny")


def test_refusal_triggers_retry_that_succeeds(monkeypatch) -> None:
    gen = _generator()
    replies = iter(
        [
            {"message": {"content": "I cannot fulfill your request."}},
            {"message": {"content": "lol stop it, what else is going on?"}},
        ]
    )
    gen._client = type("C", (), {"chat": lambda self, **kw: next(replies)})()

    result = gen.generate(_snapshot())
    assert result.text == "lol stop it, what else is going on?"


def test_persistent_refusal_uses_deflection_fallback() -> None:
    gen = _generator()
    # Model refuses on both the first attempt and the retry.
    gen._client = type(
        "C", (), {"chat": lambda self, **kw: {"message": {"content": "I can't help with that."}}}
    )()

    result = gen.generate(_snapshot())
    assert result.text == _DEFLECTION_FALLBACK


def test_normal_reply_passes_through() -> None:
    gen = _generator()
    gen._client = type(
        "C", (), {"chat": lambda self, **kw: {"message": {"content": "doing great, you?"}}}
    )()

    result = gen.generate(_snapshot())
    assert result.text == "doing great, you?"
