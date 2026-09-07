"""Tests for personality loading and system-prompt construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from responser_model_api.config import PERSONALITIES_DIR, RESPONSE_FORMAT_INSTRUCTIONS
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


def test_system_prompt_includes_gender() -> None:
    persona = Personality(name="Mia", gender="female")
    prompt = persona.to_system_prompt()
    assert "You are female" in prompt
    assert "never hedge your gender" in prompt
    # The agent must stay in its gender role, not just use the right grammar.
    assert "ALWAYS stay in that role" in prompt
    assert "react as a female person would" in prompt
    assert "another gender" in prompt


def test_format_instructions_mirror_message_length() -> None:
    lowered = RESPONSE_FORMAT_INSTRUCTIONS.lower()
    assert "mirror the other person's message length" in lowered
    # Short in -> short out, with an explicit exception for content that
    # genuinely needs detail (e.g. being asked about hobbies).
    assert "reply short" in lowered
    assert "hobbies" in lowered


def test_format_instructions_enforce_human_act() -> None:
    # The model must always deny being an AI and act human.
    lowered = RESPONSE_FORMAT_INSTRUCTIONS.lower()
    assert "human" in lowered
    assert "never an ai" in lowered or "deny being ai" in lowered


def test_system_prompt_has_adaptability_note() -> None:
    persona = Personality(name="Mia", speech_style="short and punchy")
    prompt = persona.to_system_prompt()
    lowered = prompt.lower()
    # Style is framed as a tendency and the model is told to adapt to context.
    assert "adapt" in lowered
    assert "tendency" in lowered


def test_bundled_personalities_load() -> None:
    directory = PERSONALITIES_DIR
    names = available_personalities(directory)
    assert {"friendly", "professional", "mia"}.issubset(set(names))
    for name in names:
        load_personality(name, directory)  # must parse without error


def test_examples_are_illustrations_not_conversation_turns() -> None:
    persona = Personality(
        name="Friendly",
        examples=[{"incoming": "hi", "reply": "hey!"}],
    )
    snapshot = ChatSnapshot(
        chat=ChatDescriptor(raw_id="1", title="", has_unread=True),
        messages=[Message(sender_type="other", text="how are you")],
    )
    messages = _snapshot_to_messages(snapshot, persona)

    # The example must NOT appear as a real user/assistant turn (that would
    # pollute the conversation history).
    non_system = [m for m in messages if m["role"] != "system"]
    assert {"role": "user", "content": "hi"} not in non_system
    assert {"role": "assistant", "content": "hey!"} not in non_system

    # The only real conversation content is the incoming message + the ask.
    assert non_system[0] == {"role": "user", "content": "how are you"}
    assert non_system[-1]["content"] == "Write my next reply to this conversation."

    # The example lives in the system prompt as an illustration instead.
    assert 'reply: "hey!"' in messages[0]["content"] or "hey!" in messages[0]["content"]
    assert "NOT part of the real conversation" in messages[0]["content"]


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
    # The other person must be framed as being addressed directly ("you"),
    # not as a third party to talk about.
    assert "talking directly to Bob" in context["content"]
    assert "third person" in context["content"]


def test_no_context_message_when_absent() -> None:
    persona = Personality(name="Alex")
    snapshot = ChatSnapshot(
        chat=ChatDescriptor(raw_id="1", title="", has_unread=True),
        messages=[Message(sender_type="other", text="hi")],
    )
    messages = _snapshot_to_messages(snapshot, persona)
    # Persona + relationship-stage system messages only; no platform context.
    system = [m["content"] for m in messages if m["role"] == "system"]
    assert len(system) == 2
    assert not any("chatting on" in s for s in system)


def _snapshot_with_n_messages(n: int) -> ChatSnapshot:
    msgs = [
        Message(sender_type="other" if i % 2 == 0 else "me", text=f"m{i}")
        for i in range(n)
    ]
    return ChatSnapshot(
        chat=ChatDescriptor(raw_id="1", title="Bob", has_unread=True),
        messages=msgs,
    )


def _relationship_system_note(snapshot: ChatSnapshot) -> str:
    messages = _snapshot_to_messages(snapshot, Personality(name="Mia"))
    notes = [
        m["content"]
        for m in messages
        if m["role"] == "system" and "Relationship stage" in m["content"]
    ]
    assert len(notes) == 1
    return notes[0]


def test_new_contact_is_reserved() -> None:
    note = _relationship_system_note(_snapshot_with_n_messages(2))
    assert "NEW contact" in note
    assert "reserved" in note
    assert "not too open" in note


def test_acquaintance_warms_up_but_keeps_reserve() -> None:
    note = _relationship_system_note(_snapshot_with_n_messages(10))
    assert "acquaintance" in note
    assert "still keep some reserve" in note


def test_long_relationship_is_friendly_and_informal() -> None:
    # A full reader window (20 messages) must count as "talked a lot".
    note = _relationship_system_note(_snapshot_with_n_messages(20))
    assert "talked with a lot" in note
    assert "informal" in note
    # Informality is conditional on the situation, not unconditional.
    assert "where the conversation allows" in note

