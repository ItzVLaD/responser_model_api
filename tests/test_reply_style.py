"""Natural-texting regressions using synthetic replies, never real transcripts."""

from __future__ import annotations

import pytest
from ollama import ChatResponse

from responser_model_api import ollama_client as reply_module
from responser_model_api.config import GenerationSettings, PERSONALITIES_DIR
from responser_model_api.ollama_client import OllamaReplyGenerator, _snapshot_to_messages
from responser_model_api.personality import Personality, load_personality
from responser_model_api.schemas import ChatDescriptor, ChatSnapshot, Message


def _snapshot(*past: str) -> ChatSnapshot:
    return ChatSnapshot(chat=ChatDescriptor(raw_id="test", title="Test"), messages=[
        *[Message(sender_type="me", text=text) for text in past],
        Message(sender_type="other", text="I spent the afternoon painting"),
    ])


def _generate(monkeypatch: pytest.MonkeyPatch, output: str, *past: str) -> tuple[str, int]:
    generator = OllamaReplyGenerator(load_personality("mia", PERSONALITIES_DIR), GenerationSettings())
    calls = 0

    def chat(messages: list[dict[str, str]], max_tokens: int) -> ChatResponse:
        nonlocal calls
        calls += 1
        return ChatResponse(message={"role": "assistant", "content": output}, done_reason="stop")

    monkeypatch.setattr(generator, "_chat", chat)
    return generator.generate(_snapshot(*past)).text, calls


@pytest.mark.parametrize("ending", [
    "Let me know what else you want.", "Let me know if you need anything else!",
    "Is there anything else I can help you with?", "Feel free to ask if you have any other questions.",
    "Let me know if there's anything else I can help with. 😈",
])
def test_service_closing_removed_without_extra_inference(monkeypatch: pytest.MonkeyPatch, ending: str) -> None:
    text, calls = _generate(monkeypatch, "That shade of blue is lovely. " + ending)
    assert text == "That shade of blue is lovely"
    assert calls == 1


@pytest.mark.parametrize("text", [
    "Let me know when you get home?", "Let me know which colour you choose",
    "Let's compare the two paintings", "I prefer the blue one. What did you paint?",
    'The title is "Let me know what else you want."',
    "The value is 3.14", "The version is 1.2.3", "Maybe...", "Really?!",
    "Read https://example.com/a.b", "Save notes.txt", "I live in the U.S.",
])
def test_cleanup_preserves_specific_questions_quotes_numbers_and_meaning(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    result, calls = _generate(monkeypatch, text)
    assert result == text and calls == 1


@pytest.mark.parametrize("text,expected", [
    ("I like blue.", "I like blue"),
    ("I like blue. It feels calm.", "I like blue. It feels calm"),
    ("That looks great. 🙂", "That looks great 🙂"),
    ("Try version 1.2.3.", "Try version 1.2.3"),
    ("At 3 p.m.", "At 3 p.m."),
])
def test_casual_final_period_only(monkeypatch: pytest.MonkeyPatch, text: str, expected: str) -> None:
    assert _generate(monkeypatch, text)[0] == expected


def test_recent_emoji_not_repeated_or_substituted_with_another(monkeypatch: pytest.MonkeyPatch) -> None:
    text, calls = _generate(monkeypatch, "Nice colours 😈", "That sounds fun 😈")
    assert text == "Nice colours" and calls == 1


def test_emoji_not_appended_to_consecutive_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _generate(monkeypatch, "Nice colours 🙂", "That sounds fun 😈")[0] == "Nice colours"
    assert _generate(monkeypatch, "Nice colours 🙂.", "That sounds fun 😈")[0] == "Nice colours"


def test_multi_codepoint_emojis_removed_as_whole_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _generate(monkeypatch, "Nice work 👩🏽‍🎨", "Painting again 👩🏽‍🎨")[0] == "Nice work"
    assert _generate(monkeypatch, "Good idea 👍🏿", "Nice 👍🏻", "What next?")[0] == "Good idea"


@pytest.mark.parametrize("text", ['He sent "😈"', "I meant 2️⃣", "I live in 🇮🇪", "Pick 🔴 rather than blue"])
def test_semantic_inline_emoji_flags_and_keycaps_are_not_erased(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    assert _generate(monkeypatch, text, "Nice 😈")[0] == text


def test_only_one_decorative_emoji_and_never_empty_emoji_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _generate(monkeypatch, "Nice work 🙂 🎨")[0] == "Nice work 🙂"
    assert _generate(monkeypatch, "🙂", "🙂")[0] == "🙂"


def test_recent_lets_gets_history_aware_guidance_without_echoing_private_text() -> None:
    snapshot = _snapshot("Let's talk about PRIVATE_TOPIC 😈")
    original = snapshot.model_dump_json()
    messages = _snapshot_to_messages(snapshot, load_personality("mia", PERSONALITIES_DIR))
    system = messages[0]["content"]
    assert "Do not use a 'let's' construction in this reply" in system
    assert "No emoji in this reply" in system
    assert "PRIVATE_TOPIC" not in system
    assert snapshot.model_dump_json() == original
    first_turn = next(i for i, m in enumerate(messages) if m["role"] != "system")
    assert all(m["role"] != "system" for m in messages[first_turn:])


def test_repeated_generic_lets_pivot_is_removed_but_specific_suggestion_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _generate(monkeypatch, "I understand. Let's change the subject.", "Let's talk about something else")
    assert result == "I understand" and calls == 1
    assert _generate(monkeypatch, "Let's compare the two paintings", "Let's talk about art")[0] == "Let's compare the two paintings"


def test_only_boilerplate_retries_once_then_returns_no_service_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _generate(monkeypatch, "Let me know what else you want.")
    assert calls == 2 and result.strip()
    assert "let me know" not in result.lower() and "let's" not in result.lower()


def test_formal_personality_keeps_terminal_period(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = OllamaReplyGenerator(Personality(name="Professional", tone="formal"), GenerationSettings())
    monkeypatch.setattr(generator, "_chat", lambda messages, max_tokens: ChatResponse(message={"role": "assistant", "content": "The report is ready."}))
    assert generator.generate(_snapshot()).text == "The report is ready."


def test_bundled_mia_examples_do_not_teach_emoji_on_every_turn() -> None:
    mia = load_personality("mia", PERSONALITIES_DIR)
    assert mia.casual_texting is True
    assert any(all(ord(char) < 0x2000 for char in example.reply) for example in mia.examples)
    assert "let's" not in reply_module._DEFLECTION_FALLBACK.lower()