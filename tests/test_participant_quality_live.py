"""Paired quality gate for a TEST-ONLY prototype, not ordinary offline pytest.

RESPONSER_RUN_PARTICIPANT_EXPERIMENT=1 enables real local inference on invented
conversations. Do not tune assertions after observing results. Failing the gate
means the prototype must not replace the production extractor.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import pytest

from participant_experiment import ParticipantExperiment
from responser_model_api.config import load_summary_settings
from responser_model_api.context_summarizer import OllamaContextSummarizer
from responser_model_api.memory_updates import MemoryDelta, MemoryOperation, merge_memory_delta
from responser_model_api.schemas import FactKind, FactSection, MemoryContent, Message, SenderType, SummarizeContextRequest

pytestmark = pytest.mark.skipif(
    os.environ.get("RESPONSER_RUN_PARTICIPANT_EXPERIMENT") != "1",
    reason="Opt-in paired comparison requires local Ollama; never uses private chats",
)


@dataclass(frozen=True)
class ExpectedFact:
    section: FactSection
    kind: FactKind
    pattern: str


def _initial(swapped: bool = False) -> tuple[SummarizeContextRequest, tuple[ExpectedFact, ...]]:
    turns: list[tuple[SenderType, str]] = [
        ("other", "How old are you, and what do you do for work?"),
        ("me", "I'm 24 and a lab technician. I went kayaking on Sunday. What about you?"),
        ("other", "I'm 73. Just kidding, I'm 29. I'm a landscape designer; I design gardens."),
    ]
    if swapped:
        turns = [("me" if speaker == "other" else "other", text) for speaker, text in turns]
    request = SummarizeContextRequest(messages=[Message(raw_id=f"m{index}", sender_type=speaker, text=text) for index, (speaker, text) in enumerate(turns)])
    first: FactSection = "interlocutor" if swapped else "agent"
    second: FactSection = "agent" if swapped else "interlocutor"
    return request, (
        ExpectedFact(first, "age", r"\b24\b"),
        ExpectedFact(first, "occupation", r"lab technician"),
        ExpectedFact(first, "story", r"kayak"),
        ExpectedFact(second, "age", r"\b29\b"),
        ExpectedFact(second, "occupation", r"landscape designer"),
        ExpectedFact(second, "specialty", r"design gardens"),
    )


def _score(memory: MemoryContent, expected: tuple[ExpectedFact, ...]) -> dict[str, object]:
    found = [any(fact.section == need.section and fact.kind == need.kind and re.search(need.pattern, fact.text, re.I)
                 for fact in memory.facts) for need in expected]
    unsupported = [fact for fact in memory.facts if not any(
        fact.section == need.section and fact.kind == need.kind and re.search(need.pattern, fact.text, re.I)
        for need in expected
    )]
    return {"covered": sum(found), "required": len(expected),
            "missing": [f"{need.section}:{need.kind}" for need, present in zip(expected, found) if not present],
            "unsupported_or_misclassified": len(unsupported),
            "retracted_age_present": any(re.search(r"\b73\b", fact.text) for fact in memory.facts)}


def _run_initial_pair(swapped: bool) -> None:
    request, expected = _initial(swapped)
    baseline = OllamaContextSummarizer(load_summary_settings())
    prototype = ParticipantExperiment(load_summary_settings())
    results: dict[str, dict[str, object]] = {}
    for name, summarizer in (("single_pass", baseline), ("participant_review", prototype)):
        started = time.monotonic()
        try:
            memory = summarizer.summarize(request).memory
            score = _score(memory, expected)
        except Exception as exc:
            score = {"covered": 0, "required": len(expected), "error_type": type(exc).__name__, "unsupported_or_misclassified": -1}
        results[name] = {**score, "elapsed_seconds": round(time.monotonic() - started, 1),
                         "calls": prototype.calls if name == "participant_review" else 1}
        print("QUALITY_COMPARISON", json.dumps({"swapped": swapped, "variant": name, **results[name]}), flush=True)
    candidate = results["participant_review"]
    assert candidate["covered"] == len(expected), "Prototype failed complete factual coverage; do not deploy"
    assert candidate["unsupported_or_misclassified"] == 0, "Prototype introduced unsupported or misclassified facts"
    assert not candidate["retracted_age_present"], "Prototype retained a retracted value"
    assert candidate["covered"] > results["single_pass"]["covered"], "No measurable coverage improvement over baseline"


@pytest.mark.parametrize("swapped", [False, True], ids=["original-roles", "swapped-roles"])
def test_paired_profile_coverage_gate(swapped: bool) -> None:
    _run_initial_pair(swapped)


def test_prototype_retains_seeded_facts_and_applies_only_targeted_correction() -> None:
    messages = [
        Message(raw_id="old-age", sender_type="other", text="I'm 29"),
        Message(raw_id="old-job", sender_type="other", text="I'm a designer."),
        Message(raw_id="old-story", sender_type="me", text="I went kayaking."),
    ]
    initial = merge_memory_delta(None, MemoryDelta(operations=[
        MemoryOperation(action="add", section="interlocutor" if index < 2 else "agent", kind=kind,
                        message_id=str(message.raw_id), quote=message.text)
        for index, (message, kind) in enumerate(zip(messages, ("age", "occupation", "story")))
    ]), messages)
    prototype = ParticipantExperiment(load_summary_settings())
    result = prototype.summarize(SummarizeContextRequest(previous=initial, messages=[
        Message(raw_id="hello", sender_type="me", text="Good morning. No news today."),
        Message(raw_id="correction", sender_type="other", text="Correction: I'm 30, not 29. My work is unchanged."),
    ])).memory
    assert all(old in result.facts for old in initial.facts[1:])
    assert initial.facts[0].id not in {fact.id for fact in result.facts}
    ages = [fact for fact in result.facts if fact.kind == "age" and fact.section == "interlocutor"]
    assert len(ages) == 1 and re.search(r"\b30\b", ages[0].text)