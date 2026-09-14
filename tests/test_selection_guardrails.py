"""Non-private regressions derived from the four recorded bad summary batches."""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from responser_model_api.memory_updates import MemoryDelta, MemoryOperation, merge_memory_delta
from responser_model_api.schemas import MemoryContent, Message, SummarizeContextRequest
from responser_model_api.source_selection import SelectionDelta, SelectionError, SelectionPlan, build_selection_plan


def _plan(text: str, previous: MemoryContent | None = None) -> SelectionPlan:
    return build_selection_plan(SummarizeContextRequest(previous=previous, messages=[
        Message(raw_id="new", sender_type="other", text=text),
    ]))


def _selection(source_id: str, kind: str, scope: str = "profile") -> dict[str, object]:
    return {"operations": [{"action": "add", "source_id": source_id, "kind": kind, "scope": scope}], "relationship": None}


def _reject(plan: SelectionPlan, output: dict[str, object]) -> None:
    assert not Draft202012Validator(plan.schema()).is_valid(output)
    with pytest.raises(SelectionError, match="selection_choice_invalid"):
        plan.resolve(SelectionDelta.model_validate(output))


@pytest.mark.parametrize("text", ["What do you do for work?", "Do you know the reason?", "What is your job"])
@pytest.mark.parametrize("kind,scope", [("occupation", "profile"), ("self_report", "interaction"), ("interest", "profile")])
def test_question_cannot_become_profile_fact_or_statement(text: str, kind: str, scope: str) -> None:
    plan = _plan(text)
    _reject(plan, _selection("s0", kind, scope))
    output = _selection("s0", "question", "open_threads")
    Draft202012Validator(plan.schema()).validate(output)
    assert plan.resolve(SelectionDelta.model_validate(output)).operations[0].quote == text


def test_question_cannot_replace_an_existing_occupation() -> None:
    previous = merge_memory_delta(None, MemoryDelta(operations=[MemoryOperation(
        action="add", section="interlocutor", kind="occupation", message_id="old", quote="I'm a designer.",
    )]), [Message(raw_id="old", sender_type="other", text="I'm a designer.")])
    original = previous.model_dump_json()
    plan = _plan("What do you do for work?", previous)
    _reject(plan, {"operations": [{"action": "replace", "source_id": "s0", "target_id": "t0", "kind": "occupation"}], "relationship": None})
    assert previous.model_dump_json() == original


@pytest.mark.parametrize("text", ["I wanted to ask about your work.", "I enjoy our conversations.", "I prefer tea."])
def test_newer_statement_is_not_itself_evidence_of_correction(text: str) -> None:
    previous = MemoryContent(facts=[{"id": "old", "section": "interlocutor", "kind": "occupation", "text": "I'm a designer."}])
    plan = _plan(text, previous)
    _reject(plan, {"operations": [{"action": "replace", "source_id": "s0", "target_id": "t0", "kind": "occupation"}], "relationship": None})


@pytest.mark.parametrize("text", ["I now work as a nurse.", "Correction: I work as a nurse."])
def test_explicit_correction_is_still_an_available_replacement(text: str) -> None:
    previous = MemoryContent(facts=[{"id": "old", "section": "interlocutor", "kind": "occupation", "text": "I'm a designer."}])
    plan = _plan(text, previous)
    source_id = next(key for key, source in plan.sources.items() if "work as a nurse" in source.evidence.quote)
    output = {"operations": [{"action": "replace", "source_id": source_id, "target_id": "t0", "kind": "occupation"}], "relationship": None}
    Draft202012Validator(plan.schema()).validate(output)
    plan.resolve(SelectionDelta.model_validate(output))


@pytest.mark.parametrize("kind,scope", [("occupation", "interaction"), ("self_report", "interaction"), ("interest", "profile"), ("occupation", "profile")])
def test_bare_age_cannot_escape_age_rules_via_another_kind(kind: str, scope: str) -> None:
    plan = _plan("I’m 29")
    _reject(plan, _selection("s0", kind, scope))
    Draft202012Validator(plan.schema()).validate(_selection("s0", "age"))


@pytest.mark.parametrize("kind", ["age", "name", "occupation", "specialty"])
def test_profile_value_kinds_never_use_interaction_scope(kind: str) -> None:
    text = "I'm 29 and enjoying life!" if kind == "age" else "I work as a designer."
    _reject(_plan(text), _selection("s0", kind, "interaction"))


@pytest.mark.parametrize("text", ["For sure", "I think", "Yes.", "Okay!", "Just kidding"])
def test_standalone_fillers_remain_context_without_fact_or_replacement_choices(text: str) -> None:
    plan = _plan(text)
    _reject(plan, _selection("s0", "self_report", "interaction"))
    _reject(plan, {"operations": [], "relationship": {"stage": "familiar", "source_id": "s0"}})
    assert text in plan.payload
    previous = MemoryContent(facts=[{"id": "old", "section": "interaction", "kind": "self_report", "text": "I'm 29"}])
    plan = _plan(text, previous)
    _reject(plan, {"operations": [{"action": "replace", "source_id": "s0", "target_id": "t0", "kind": "self_report"}], "relationship": None})


@pytest.mark.parametrize("text", [
    "I think, being listened to matters to me.",
    "I design gardens, especially roof gardens.",
    "I dislike being ignored, so please ask follow-up questions.",
])
def test_comma_does_not_detach_content_from_its_subject(text: str) -> None:
    plan = _plan(text)
    assert plan.sources["s0"].evidence.quote == text
    assert len(plan.sources) == 1


@pytest.mark.parametrize("same_message", [False, True])
def test_explicit_same_speaker_age_retraction_disables_only_retracted_age(same_message: bool) -> None:
    messages = [Message(raw_id=str(index), sender_type="other", text=text) for index, text in enumerate([
        "I'm 73.", "Just kidding,", "I'm 29.",
    ])]
    if same_message:
        messages = [Message(raw_id="one", sender_type="other", text="I'm 73. Just kidding, I'm 29.")]
    plan = build_selection_plan(SummarizeContextRequest(messages=messages))
    for key, source in plan.sources.items():
        if "73" in source.evidence.quote:
            _reject(plan, _selection(key, "age"))
            _reject(plan, _selection(key, "self_report", "interaction"))
    correct = next(key for key, source in plan.sources.items() if source.evidence.quote == "I'm 29.")
    result = merge_memory_delta(None, plan.resolve(SelectionDelta.model_validate(_selection(correct, "age"))), messages)
    assert result.interlocutor == ["I'm 29."]
    assert "73" in plan.payload and "kidding" in plan.payload
    rows = json.loads(plan.payload)["messages"]
    assert ["".join(part["text"] for part in row["parts"]) for row in rows] == [message.text for message in messages]


def test_other_speaker_retraction_never_invalidates_age() -> None:
    messages = [Message(raw_id="a", sender_type="other", text="I'm 73."),
                Message(raw_id="b", sender_type="me", text="Just kidding,"),
                Message(raw_id="c", sender_type="other", text="I'm 29.")]
    plan = build_selection_plan(SummarizeContextRequest(messages=messages))
    Draft202012Validator(plan.schema()).validate(_selection("s0", "age"))


def test_correction_marker_does_not_leak_across_messages() -> None:
    previous = MemoryContent(facts=[{"id": "old", "section": "interlocutor", "kind": "occupation", "text": "Designer"}])
    plan = build_selection_plan(SummarizeContextRequest(previous=previous, messages=[
        Message(raw_id="prefix", sender_type="me", text="Correction:"),
        Message(raw_id="claim", sender_type="other", text="I work as a nurse."),
    ]))
    source = next(key for key, value in plan.sources.items() if value.evidence.message_id == "claim")
    _reject(plan, {"operations": [{"action": "replace", "target_id": "t0", "source_id": source, "kind": "occupation"}], "relationship": None})


def test_schema_and_resolver_agree_for_all_add_source_categories() -> None:
    plan = build_selection_plan(SummarizeContextRequest(messages=[
        Message(raw_id=str(i), sender_type="other", text=text)
        for i, text in enumerate(["I'm 29", "What is your job?", "For sure", "I design gardens.", "I’m 30 and a nurse."])
    ]))
    validator = Draft202012Validator(plan.schema())
    for key, source in plan.sources.items():
        for scope in ("profile", "interaction", "open_threads"):
            for kind in ("age", "occupation", "specialty", "name", "self_report", "question", "commitment", "interest"):
                output = _selection(key, kind, scope)
                expected = (scope, kind) in source.allowed_adds()
                assert validator.is_valid(output) == expected
                if expected:
                    plan.resolve(SelectionDelta.model_validate(output))
                else:
                    with pytest.raises(SelectionError):
                        plan.resolve(SelectionDelta.model_validate(output))


@pytest.mark.parametrize("text", [
    "I'm an engineer.", "I develop software agents.", "I am 29 and a lab technician.",
    "Please stop asking about my income.", "I think about moving to a new city.",
])
def test_real_self_reports_are_not_removed_by_negative_guards(text: str) -> None:
    plan = _plan(text)
    kind = "occupation" if "engineer" in text or "technician" in text else "self_report"
    Draft202012Validator(plan.schema()).validate(_selection("s0", kind))