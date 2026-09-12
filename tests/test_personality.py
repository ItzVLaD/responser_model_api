"""Tests for personality loading and system-prompt construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from responser_model_api.config import PERSONALITIES_DIR, RESPONSE_FORMAT_INSTRUCTIONS
from responser_model_api.ollama_client import (
    _length_budget,
    _snapshot_to_messages,
    _system_prompt,
)
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
    assert non_system[-1]["content"].startswith("Write my next reply to this conversation.")

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
    # Multi-word texts: a run of one-word messages would read as "curt" and
    # override the history-based stage, which is not what these tests probe.
    msgs = [
        Message(
            sender_type="other" if i % 2 == 0 else "me",
            text=f"this is message number {i} in the chat",
        )
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


def _snapshot(*turns: tuple[str, str]) -> ChatSnapshot:
    return ChatSnapshot(
        chat=ChatDescriptor(raw_id="1", title="Bob", has_unread=True),
        messages=[Message(sender_type=who, text=text) for who, text in turns],
    )


def test_curt_other_person_overrides_long_history() -> None:
    # Plenty of history, but they have gone cold ("bruh", "boring", "..."):
    # rapport is about current engagement, not raw message count.
    turns = [("other" if i % 2 == 0 else "me", f"a fairly normal message {i}") for i in range(16)]
    turns += [("other", "bruh"), ("me", "hey"), ("other", "boring"), ("other", "...")]
    note = _relationship_system_note(_snapshot(*turns))
    assert "curt" in note
    assert "do not push" in note


def test_single_short_opener_is_not_curt() -> None:
    # "hi" alone is a normal opener, not disengagement.
    note = _relationship_system_note(_snapshot(("other", "hi")))
    assert "NEW contact" in note
    assert "curt" not in note


def test_length_budget_mirrors_short_messages_with_token_cap() -> None:
    budget = _length_budget(
        _snapshot(("other", "hey"), ("me", "hi there"), ("other", "bruh"), ("other", "boring"))
    )
    # ~1 word each -> the floor of 6 words, backed by a tight generation cap
    # (English text: 2 tokens/word + margin).
    assert "about 1 words each" in budget.note
    assert "at most about 6 words" in budget.note
    assert budget.max_tokens == 6 * 2 + 16


def test_length_budget_scales_with_longer_messages() -> None:
    long = " ".join(["word"] * 15)
    budget = _length_budget(_snapshot(("other", long), ("other", long)))
    assert "at most about 30 words" in budget.note
    assert budget.max_tokens == 30 * 2 + 16


def test_length_budget_uses_fixed_english_token_estimate() -> None:
    budget = _length_budget(_snapshot(("other", "hello"), ("other", "you okay")))
    assert budget.max_tokens == 6 * 2 + 16


def test_length_budget_never_exceeds_mirror_ceiling() -> None:
    huge = " ".join(["word"] * 200)
    budget = _length_budget(_snapshot(("other", huge)))
    assert "at most about 40 words" in budget.note


@pytest.mark.parametrize(
    "incoming",
    [
        "tell me about your hobbies",
        "what do you like doing on weekends",
        "how was your day?",
        "describe yourself",
        "what are your interests",
    ],
)
def test_length_budget_opens_up_when_asked_to_elaborate(incoming: str) -> None:
    # A short request to tell/explain deserves a real answer: a few sentences,
    # still bounded so it cannot become an essay.
    budget = _length_budget(_snapshot(("other", "hey"), ("me", "hi"), ("other", incoming)))
    assert "tell or explain" in budget.note
    assert "60 words" in budget.note
    assert budget.max_tokens > 6 * 2 + 16


@pytest.mark.parametrize(
    "incoming",
    ["Hello sweetheart\nHow are you?", "Is it AI?", "Bruh\nAre you kidding?", "you?"],
)
def test_plain_questions_stay_in_mirror_mode(incoming: str) -> None:
    # Most chat messages end in '?'; that alone must NOT unlock long replies -
    # "how are you?" from a 3-word texter wants a 1-line answer. (Live: every
    # Test User turn hit open mode via '?' and replies ballooned to 100 tokens.)
    budget = _length_budget(_snapshot(("other", incoming)))
    assert "tell or explain" not in budget.note
    assert budget.max_tokens <= 10 * 2 + 16


def test_length_budget_only_looks_at_unanswered_messages() -> None:
    # An old request we already answered must not keep the budget open.
    budget = _length_budget(
        _snapshot(("other", "tell me about your day"), ("me", "it was fine"), ("other", "cool"))
    )
    assert "tell or explain" not in budget.note


def test_length_note_rides_on_final_user_ask_not_a_trailing_system_message() -> None:
    messages = _snapshot_to_messages(_snapshot(("other", "hey")), Personality(name="Mia"))
    last = messages[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("Write my next reply to this conversation.")
    assert "Length:" in last["content"]
    # A system message after the conversation turns gets echoed verbatim by
    # ChatML models (observed live) - no system role may follow a real turn.
    first_turn = next(i for i, m in enumerate(messages) if m["role"] != "system")
    assert all(m["role"] != "system" for m in messages[first_turn:])


def test_format_instructions_read_the_room() -> None:
    lowered = RESPONSE_FORMAT_INSTRUCTIONS.lower()
    assert "read the room" in lowered
    assert "dial the energy down" in lowered

