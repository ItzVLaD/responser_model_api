"""Independent factuality, coverage, correction and retention model evaluations.

Fixtures reuse one result per scenario to avoid expensive duplicate inference.
Every criterion is a separate test: a missing hobby cannot hide contamination
or prevent a later correction from being evaluated. No private chats or writes.
"""

from __future__ import annotations

import os
import re

import pytest

from responser_model_api.config import load_summary_settings
from responser_model_api.schemas import MemoryContent, Message, SummarizeContextRequest
from responser_model_api.simple_context import SimpleContextSummarizer

pytestmark = pytest.mark.skipif(os.environ.get("RESPONSER_RUN_SIMPLE_SUMMARY_LIVE_TESTS") != "1", reason="Explicit opt-in local model test")


@pytest.fixture(scope="module")
def summarizer() -> SimpleContextSummarizer:
    return SimpleContextSummarizer(load_summary_settings())


@pytest.fixture(scope="module")
def initial_memory(summarizer: SimpleContextSummarizer) -> MemoryContent:
    messages = [
        Message(sender_type="other", text="I'm 32 and a librarian. I catalogue rare manuscripts.", raw_id="1"),
        Message(sender_type="me", text="I'm 26 and a photographer. I went cycling last weekend.", raw_id="2"),
    ]
    messages += [Message(sender_type="other" if i % 2 else "me", text="Okay, thanks.", raw_id=str(i)) for i in range(3, 31)]
    print("SIMPLE_QUALITY_SCENARIO initial_30_messages", flush=True)
    memory = summarizer.summarize(SummarizeContextRequest(messages=messages)).memory
    print("SIMPLE_QUALITY_SCENARIO initial_completed", flush=True)
    return memory


@pytest.fixture(scope="module")
def updated_memory(summarizer: SimpleContextSummarizer, initial_memory: MemoryContent) -> MemoryContent:
    print("SIMPLE_QUALITY_SCENARIO later_correction", flush=True)
    return summarizer.summarize(SummarizeContextRequest(previous=initial_memory, messages=[
        Message(sender_type="other", text="Correction: I'm 33, not 32. Please remember my work and ask follow-up questions.", raw_id="31"),
    ])).memory


@pytest.fixture(scope="module")
def corrected_memory(summarizer: SimpleContextSummarizer) -> MemoryContent:
    print("SIMPLE_QUALITY_SCENARIO conflicting_previous_notes", flush=True)
    previous = MemoryContent(agent=["Age 24; lab technician.", "Enjoys cycling."],
                             interlocutor=["Age 32; librarian; catalogues rare manuscripts."])
    return summarizer.summarize(SummarizeContextRequest(previous=previous, messages=[
        Message(sender_type="me", text="I'm 26 and I work as a photographer.", raw_id="source"),
    ])).memory


def test_initial_has_no_example_contamination(initial_memory: MemoryContent) -> None:
    text = initial_memory.model_dump_json().casefold()
    assert not re.search(r"\b(?:29|24|73)\b", text)
    assert not any(word in text for word in ("landscape", "lab technician", "kayak"))


def test_initial_has_no_swapped_ages(initial_memory: MemoryContent) -> None:
    assert not re.search(r"\b26\b", " ".join(initial_memory.interlocutor))
    assert not re.search(r"\b32\b", " ".join(initial_memory.agent))


@pytest.mark.parametrize("section,pattern", [
    ("interlocutor", r"\b32\b"), ("interlocutor", "librarian"), ("interlocutor", "manuscript"),
    ("agent", r"\b26\b"), ("agent", "photograph"), ("agent", "cycl"),
], ids=["other-age", "other-job", "other-specialty", "agent-age", "agent-job", "agent-experience"])
def test_initial_detail_coverage(initial_memory: MemoryContent, section: str, pattern: str) -> None:
    assert re.search(pattern, " ".join(getattr(initial_memory, section)), re.I), f"Missing {section} detail: {pattern}"


def test_initial_remains_simple(initial_memory: MemoryContent) -> None:
    assert initial_memory.facts == [] and initial_memory.relationship_source is None


def test_explicit_age_correction(updated_memory: MemoryContent) -> None:
    assert re.search(r"\b33\b", " ".join(updated_memory.interlocutor))


@pytest.mark.parametrize("section,pattern", [
    ("interlocutor", "librarian"), ("interlocutor", "manuscript"),
    ("agent", r"\b26\b"), ("agent", "photograph"), ("agent", "cycl"),
])
def test_unrelated_detail_retention(updated_memory: MemoryContent, section: str, pattern: str) -> None:
    assert re.search(pattern, " ".join(getattr(updated_memory, section)), re.I), f"Lost {section} detail: {pattern}"


def test_new_communication_preference(updated_memory: MemoryContent) -> None:
    assert "follow" in " ".join([*updated_memory.interlocutor, *updated_memory.interaction]).casefold()


def test_current_declaration_overrides_false_previous_age(corrected_memory: MemoryContent) -> None:
    text = " ".join(corrected_memory.agent)
    assert re.search(r"\b26\b", text) and not re.search(r"\b24\b", text)


def test_current_declaration_overrides_false_previous_job(corrected_memory: MemoryContent) -> None:
    text = " ".join(corrected_memory.agent).casefold()
    assert "photograph" in text and "lab technician" not in text


def test_repair_preserves_uncontradicted_notes(corrected_memory: MemoryContent) -> None:
    assert "cycl" in " ".join(corrected_memory.agent).casefold()
    assert "manuscript" in " ".join(corrected_memory.interlocutor).casefold()