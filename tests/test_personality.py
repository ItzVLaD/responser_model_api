"""Tests for personality loading and system-prompt construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from responser_model_api.config import RESPONSE_FORMAT_INSTRUCTIONS
from responser_model_api.ollama_client import _snapshot_to_messages, _system_prompt
from responser_model_api.personality import (
    Personality,
    PersonalityError,
    available_personalities,
    load_personality,
)
from responser_model_api.schemas import ChatDescriptor, ChatSnapshot, Message


def _write(dir_: Path, name: str, text: str) -> None:
    (dir_ / f"{name}.yaml").write_text(text, encoding="utf-8")


def test_load_personality(tmp_path) -> None:
    _write(
        tmp_path,
        "friendly",
        "name: Friendly\ntone: warm\nrules:\n  - Be kind.\n"
        "examples:\n  - incoming: hi\n    reply: hey!\n",
    )
    p = load_personality("friendly", tmp_path)
    assert p.name == "Friendly"
    assert p.tone == "warm"
    assert p.rules == ["Be kind."]
    assert p.examples[0].incoming == "hi"
    assert p.examples[0].reply == "hey!"


def test_available_personalities(tmp_path) -> None:
    _write(tmp_path, "friendly", "name: Friendly\n")
    _write(tmp_path, "professional", "name: Professional\n")
    assert available_personalities(tmp_path) == ["friendly", "professional"]


def test_missing_personality_raises(tmp_path) -> None:
    with pytest.raises(PersonalityError, match="not found"):
        load_personality("ghost", tmp_path)


def test_invalid_yaml_raises(tmp_path) -> None:
    (tmp_path / "broken.yaml").write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(PersonalityError):
        load_personality("broken", tmp_path)


def test_system_prompt_layers_format_over_persona() -> None:
    persona = Personality(name="Friendly", tone="warm", rules=["Be kind."])
    prompt = _system_prompt(persona)
    # Fixed contract stays present; persona is layered on top.
    assert RESPONSE_FORMAT_INSTRUCTIONS in prompt
    assert "Friendly" in prompt
    assert "Be kind." in prompt


def test_system_prompt_includes_rich_identity_fields() -> None:
    persona = Personality(
        name="Alex",
        identity="a 28-year-old game developer",
        background="Loves indie games.",
        speech_style="short lowercase sentences",
        language="English",
        emoji_usage="often 😄",
        signature_phrases=["haha", "for real"],
        interests=["games", "coffee"],
        avoid=["being formal"],
    )
    prompt = persona.to_system_prompt()
    assert "You are Alex, a 28-year-old game developer." in prompt
    assert "Loves indie games." in prompt
    assert "short lowercase sentences" in prompt
    assert "English" in prompt
    assert "😄" in prompt
    assert '"haha"' in prompt
    assert "games, coffee" in prompt
    assert "being formal" in prompt


def test_format_instructions_enforce_human_act() -> None:
    # The model must always deny being an AI and act human.
    lowered = RESPONSE_FORMAT_INSTRUCTIONS.lower()
    assert "human" in lowered
    assert "never an ai" in lowered or "deny being ai" in lowered


def test_examples_become_fewshot_turns() -> None:
    persona = Personality(
        name="Friendly",
        examples=[{"incoming": "hi", "reply": "hey!"}],
    )
    snapshot = ChatSnapshot(
        chat=ChatDescriptor(raw_id="1", title="", has_unread=True),
        messages=[Message(sender_type="other", text="how are you")],
    )
    messages = _snapshot_to_messages(snapshot, persona)

    # system, then the few-shot pair, then the real message, then the ask.
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hey!"}
    assert messages[3] == {"role": "user", "content": "how are you"}
    assert messages[-1]["content"] == "Write my next reply to this conversation."


def test_platform_and_account_context_injected() -> None:
    persona = Personality(name="Alex")
    snapshot = ChatSnapshot(
        chat=ChatDescriptor(raw_id="1", title="Bob", has_unread=True),
        messages=[Message(sender_type="other", text="hi")],
        platform="Telegram",
        account_name="Alex K",
    )
    messages = _snapshot_to_messages(snapshot, persona)

    # A second system message carries the platform/account/chat context.
    context = messages[1]
    assert context["role"] == "system"
    assert "Telegram" in context["content"]
    assert "Alex K" in context["content"]
    assert "Bob" in context["content"]


def test_no_context_message_when_absent() -> None:
    persona = Personality(name="Alex")
    snapshot = ChatSnapshot(
        chat=ChatDescriptor(raw_id="1", title="", has_unread=True),
        messages=[Message(sender_type="other", text="hi")],
    )
    messages = _snapshot_to_messages(snapshot, persona)
    # Only the persona system message; no extra context system message.
    assert sum(1 for m in messages if m["role"] == "system") == 1

