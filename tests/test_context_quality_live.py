"""Opt-in real-model quality checks with invented, non-private conversations.

Set RESPONSER_RUN_LIVE_CONTEXT_TESTS=1 to run against local Ollama. Ordinary
pytest stays offline. These assertions check semantics across rolling updates,
not merely whether a stub returned valid JSON. No browser or message sending.
"""

from __future__ import annotations

import os
import re

import pytest

from responser_model_api.config import load_summary_settings
from responser_model_api.context_summarizer import OllamaContextSummarizer
from responser_model_api.memory_updates import MemoryDelta, MemoryOperation, merge_memory_delta
from responser_model_api.schemas import FactKind, FactSection, MemoryContent, Message, SenderType, SummarizeContextRequest

pytestmark = pytest.mark.skipif(
    os.environ.get("RESPONSER_RUN_LIVE_CONTEXT_TESTS") != "1",
    reason="Opt-in semantic evaluation requires local Ollama and a pulled summary model",
)


def _batch(turns: list[tuple[SenderType, str]], start: int) -> list[Message]:
    return [Message(sender_type=who, text=text, raw_id=str(start + index))
            for index, (who, text) in enumerate(turns)]


def _section(memory: MemoryContent, section: FactSection) -> str:
    """Use canonical facts, not the bounded backwards-compatible preview lists."""
    return " ".join(f.text for f in memory.facts if f.section == section).lower()


def _assert_profiles(memory: MemoryContent) -> None:
    other = _section(memory, "interlocutor")
    agent = _section(memory, "agent")
    assert re.search(r"\b29\b", other), memory
    assert "landscape" in other and "garden" in other, memory
    assert re.search(r"\b24\b", agent), memory
    assert "lab" in agent and "technician" in agent, memory
    assert "kayak" in agent, memory
    assert not re.search(r"\b24\b", other), memory
    assert not re.search(r"\b29\b", agent), memory


def test_specific_profiles_preferences_and_reactions_survive_unrelated_updates() -> None:
    summarizer = OllamaContextSummarizer(load_summary_settings())
    memory = summarizer.summarize(SummarizeContextRequest(messages=_batch([
        ("other", "How old are you, and what do you do for work?"),
        ("me", "I'm 24 and a lab technician. I went kayaking on Sunday. What about you?"),
        ("other", "I'm 73. Just kidding, I'm 29. I'm a landscape designer; I design gardens."),
    ], 1))).memory
    _assert_profiles(memory)
    original_ids = {fact.id for fact in memory.facts}
    assert memory.facts and all(fact.evidence for fact in memory.facts)
    # A retracted joke may be recorded as retracted, but never as the current age.
    for entry in memory.interlocutor:
        if "73" in entry:
            assert any(word in entry.lower() for word in ("joke", "joking", "kidding", "retract"))

    memory = summarizer.summarize(SummarizeContextRequest(previous=memory, messages=_batch([
        ("me", "What do you do for work?"),
        ("other", "I already told you my job. It bothers me when you don't remember."),
        ("other", "Please be nosy: ask about me, show some curiosity instead of changing the subject."),
        ("other", "Sorry I snapped. I sometimes get aggressive. I feel anxious when I think I'm being ignored."),
        ("me", "Understood. I'll pay attention and ask follow-up questions."),
    ], 4))).memory
    _assert_profiles(memory)
    assert original_ids <= {fact.id for fact in memory.facts}
    remembered_ids = {fact.id for fact in memory.facts}

    memory = summarizer.summarize(SummarizeContextRequest(previous=memory, messages=_batch([
        ("other", "Busy morning. I made tea."),
        ("me", "Same here, lots of errands today."),
        ("other", "Anyway, no big news. Catch you later."),
    ], 9))).memory
    _assert_profiles(memory)
    assert remembered_ids <= {fact.id for fact in memory.facts}
    text = _section(memory, "interlocutor") + _section(memory, "interaction")
    assert "aggress" in text, memory
    assert "anxi" in text and "ignor" in text, memory
    assert any(word in text for word in ("nosy", "curios", "curious", "follow-up")), memory
    assert any(word in text for word in ("remember", "repeat", "forgot", "recall")), memory

    # A later explicit correction must update the value, not drop the whole profile.
    memory = summarizer.summarize(SummarizeContextRequest(previous=memory, messages=_batch([
        ("other", "Correction: I'm 30, not 29. Everything else I told you is still true."),
    ], 12))).memory
    other = _section(memory, "interlocutor")
    assert re.search(r"\b30\b", other), memory
    assert "landscape" in other and "garden" in other, memory
    assert "kayak" in _section(memory, "agent"), memory
    age_facts = [fact for fact in memory.facts if fact.section == "interlocutor" and fact.kind == "age"]
    assert len(age_facts) == 1 and age_facts[0].evidence is not None, memory
    assert age_facts[0].evidence.message_id == "12", memory


def test_existing_evidence_survives_live_unrelated_update() -> None:
    """Isolate preservation from the separate, intentionally strict recall test.

    These facts are independently supplied and verified, not model-extracted by
    this test. Passing proves an update preserves them, not extraction quality.
    """
    sources = _batch([
        ("other", "I'm 29"),
        ("other", "I'm a landscape designer; I design gardens."),
        ("other", "Please ask me curious follow-up questions."),
        ("other", "I sometimes get aggressive when I feel ignored."),
        ("me", "I'm 24"),
        ("me", "I'm a lab technician."),
        ("me", "I went kayaking on Sunday."),
    ], 1)
    kinds: tuple[FactKind, ...] = ("age", "occupation", "preference", "self_report", "age", "occupation", "story")
    initial = merge_memory_delta(None, MemoryDelta(operations=[
        MemoryOperation(
            action="add", section="agent" if message.sender_type == "me" else "interlocutor",
            kind=kind, message_id=str(message.raw_id), quote=message.text,
        ) for message, kind in zip(sources, kinds)
    ]), sources)
    updated = OllamaContextSummarizer(load_summary_settings()).summarize(
        SummarizeContextRequest(previous=initial, messages=_batch([
            ("other", "Good morning! Nothing new today, just saying hello."),
            ("me", "Morning! Nice to hear from you."),
        ], 20)),
    ).memory
    by_id = {fact.id: fact for fact in updated.facts}
    assert all(by_id.get(fact.id) == fact for fact in initial.facts)
    _assert_profiles(updated)