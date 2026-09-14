"""Offline safety and scoping checks for the test-only extraction hypothesis."""

from __future__ import annotations

import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from ollama import Client

from participant_experiment import ParticipantExperiment, focused_plan
from responser_model_api.config import SummarySettings
from responser_model_api.memory_updates import MemoryDelta, MemoryOperation, merge_memory_delta
from responser_model_api.schemas import MemoryContent, Message, SummarizeContextRequest
from responser_model_api.source_selection import SelectionDelta, SelectionError, build_selection_plan


def _request() -> SummarizeContextRequest:
    return SummarizeContextRequest(messages=[
        Message(raw_id="one", sender_type="other", text="I'm a designer."),
        Message(raw_id="two", sender_type="me", text="I'm a nurse."),
    ])


def test_focused_passes_restrict_evidence_and_scope_without_hiding_context() -> None:
    plan = build_selection_plan(_request())
    for focus, allowed in (("interlocutor", "s0"), ("agent", "s1")):
        scoped = focused_plan(plan, focus)
        assert scoped.payload == plan.payload and set(scoped.sources) == {allowed}
        valid = {"operations": [{"action": "add", "source_id": allowed, "scope": "profile", "kind": "occupation"}], "relationship": None}
        validator = Draft202012Validator(scoped.schema())
        validator.validate(valid)
        assert scoped.resolve(SelectionDelta.model_validate(valid)).operations[0].section == focus
        for invalid in (
            {**valid, "relationship": {"stage": "familiar", "source_id": allowed}},
            {"operations": [{**valid["operations"][0], "scope": "interaction"}], "relationship": None},
            {"operations": [{**valid["operations"][0], "source_id": "s1" if allowed == "s0" else "s0"}], "relationship": None},
        ):
            assert not validator.is_valid(invalid)
            with pytest.raises(SelectionError):
                scoped.resolve(SelectionDelta.model_validate(invalid))
    interaction = focused_plan(plan, "interaction")
    assert set(interaction.sources) == set(plan.sources)
    assert all(scope != "profile" for source in interaction.sources.values() for scope, _ in source.allowed_adds())


def test_each_target_is_owned_by_one_pass() -> None:
    request = _request()
    request.previous = MemoryContent(facts=[
        {"id": "a", "section": "agent", "kind": "occupation", "text": "Old job"},
        {"id": "b", "section": "interlocutor", "kind": "occupation", "text": "Old work"},
        {"id": "c", "section": "open_threads", "kind": "question", "text": "When?"},
    ])
    plan = build_selection_plan(request)
    sets = [set(focused_plan(plan, focus).targets) for focus in ("agent", "interlocutor", "interaction")]
    assert set.union(*sets) == set(plan.targets)
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_all_four_passes_must_succeed_and_coverage_sees_provisional_results() -> None:
    request = _request()
    before = request.model_dump_json()
    calls: list[httpx.Request] = []
    outputs = [
        {"operations": [{"action": "add", "source_id": "s0", "scope": "profile", "kind": "occupation"}]},
        {"operations": [{"action": "add", "source_id": "s1", "scope": "profile", "kind": "occupation"}]},
        {},
        {"operations": [{"action": "add", "source_id": "unknown", "scope": "profile", "kind": "occupation"}]},
    ]

    def handle(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(outputs[len(calls) - 1])}, "done": True})

    experiment = ParticipantExperiment(SummarySettings(), client=Client(host="http://test", transport=httpx.MockTransport(handle)))
    with pytest.raises(SelectionError):
        experiment.summarize(request)
    assert len(calls) == 4 and request.model_dump_json() == before
    review = json.loads(json.loads(calls[-1].content)["messages"][1]["content"])
    assert review["coverage_review"]["current_profiles"] == {
        "agent": [{"kind": "occupation", "text": "I'm a nurse."}],
        "interlocutor": [{"kind": "occupation", "text": "I'm a designer."}],
    }


def test_profile_pass_cannot_remove_other_participants_fact() -> None:
    request = _request()
    request.previous = merge_memory_delta(None, MemoryDelta(operations=[MemoryOperation(
        action="add", section="agent", kind="occupation", quote="I'm a nurse.", message_id="two",
    )]), request.messages)
    scoped = focused_plan(build_selection_plan(request), "interlocutor")
    assert not scoped.targets
    with pytest.raises(SelectionError):
        scoped.resolve(SelectionDelta.model_validate({"operations": [{"action": "replace", "target_id": "t0", "source_id": "s0", "kind": "occupation"}]}))


def test_quality_scorer_requires_exact_section_kind_and_both_participants() -> None:
    from test_participant_quality_live import _initial, _score

    _, expected = _initial()
    correct = MemoryContent(facts=[
        {"id": "a", "section": "agent", "kind": "age", "text": "I'm 24"},
        {"id": "b", "section": "agent", "kind": "occupation", "text": "I'm a lab technician."},
        {"id": "c", "section": "agent", "kind": "story", "text": "I went kayaking."},
        {"id": "d", "section": "interlocutor", "kind": "age", "text": "I'm 29"},
        {"id": "e", "section": "interlocutor", "kind": "occupation", "text": "I'm a landscape designer."},
        {"id": "f", "section": "interlocutor", "kind": "specialty", "text": "I design gardens."},
    ])
    assert _score(correct, expected)["covered"] == 6
    assert _score(correct, expected)["unsupported_or_misclassified"] == 0
    for field, value in (("section", "interaction"), ("section", "agent"), ("kind", "self_report")):
        wrong = correct.model_copy(deep=True)
        setattr(wrong.facts[-1], field, value)
        score = _score(wrong, expected)
        assert score["covered"] == 5 and score["unsupported_or_misclassified"] == 1
    other_only = MemoryContent(facts=correct.facts[3:])
    assert _score(other_only, expected)["covered"] == 3
    retracted = correct.model_copy(deep=True)
    retracted.facts[3].text = "I'm 73"
    assert _score(retracted, expected)["retracted_age_present"] is True


def test_late_pass_transport_failure_leaves_previous_memory_unchanged() -> None:
    previous = MemoryContent(agent=["Existing unrelated story"])
    request = _request()
    request.previous = previous
    before = request.model_dump_json()
    count = 0

    def handle(incoming: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 2:
            raise httpx.ReadTimeout("offline synthetic failure")
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps({
            "operations": [{"action": "add", "source_id": "s0", "scope": "profile", "kind": "occupation"}],
        })}, "done": True})

    prototype = ParticipantExperiment(SummarySettings(), client=Client(host="http://test", transport=httpx.MockTransport(handle)))
    with pytest.raises(httpx.ReadTimeout):
        prototype.summarize(request)
    assert request.model_dump_json() == before and count == 2