"""Offline source-proof and preservation tests, not semantic-quality guarantees."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from responser_model_api.memory_updates import (
    MemoryDelta,
    MemoryOperation,
    MemoryUpdateError,
    RelationshipChange,
    merge_memory_delta,
    migrate_legacy,
)
from responser_model_api.schemas import (
    FACT_SECTIONS,
    FactEvidence,
    FactKind,
    FactSection,
    MemoryContent,
    MemoryFact,
    Message,
    RelationshipState,
    SenderType,
)


def _message(text: str, raw_id: str | None = "new", speaker: SenderType = "other") -> Message:
    return Message(sender_type=speaker, text=text, raw_id=raw_id)


def _operation(
    quote: str, *, section: FactSection = "interlocutor", kind: FactKind = "preference",
    action: Literal["add", "replace", "remove"] = "add", target: str | None = None, message_id: str = "new",
) -> MemoryOperation:
    return MemoryOperation(action=action, section=section, kind=kind, target_id=target, message_id=message_id, quote=quote)


def _merge_one(previous: MemoryContent | None, operation: MemoryOperation, speaker: SenderType = "other") -> MemoryContent:
    return merge_memory_delta(previous, MemoryDelta(operations=[operation]), [_message(operation.quote, operation.message_id, speaker)])


def _profiles() -> MemoryContent:
    rows: list[tuple[FactSection, FactKind, SenderType, str]] = [
        ("interlocutor", "age", "other", "I'm 29"),
        ("interlocutor", "occupation", "other", "I'm a landscape designer."),
        ("interlocutor", "specialty", "other", "I design gardens."),
        ("interlocutor", "preference", "other", "Please ask attentive follow-up questions."),
        ("agent", "age", "me", "I am 24"),
        ("agent", "occupation", "me", "I'm a laboratory technician."),
        ("agent", "story", "me", "I went kayaking last Sunday."),
    ]
    operations = [_operation(text, section=section, kind=kind, message_id=str(index))
                  for index, (section, kind, _, text) in enumerate(rows)]
    messages = [_message(text, str(index), speaker) for index, (_, _, speaker, text) in enumerate(rows)]
    return merge_memory_delta(None, MemoryDelta(operations=operations), messages)


def test_legacy_migration_preserves_text_and_has_stable_ids_without_fake_evidence() -> None:
    memory = MemoryContent(
        interlocutor=["Age: 29", "Garden designer"], agent=["Old kayaking story"],
        interaction=["Asked for attentive questions"], open_threads=["Follow up next week"],
        relationship=RelationshipState(stage="familiar", evidence="Legacy rapport"),
    )
    before = memory.model_dump_json()
    migrated = migrate_legacy(memory)
    assert memory.model_dump_json() == before and memory.facts == []
    assert migrated.model_dump_json() == migrate_legacy(memory).model_dump_json()
    assert migrated.model_dump_json() == migrate_legacy(migrated).model_dump_json()
    assert migrated.relationship == memory.relationship and migrated.relationship_source is None
    for section in FACT_SECTIONS:
        assert [fact.text for fact in migrated.facts if fact.section == section] == getattr(memory, section)
    assert all(fact.kind == "other" and fact.evidence is None and len(fact.id) == 64 for fact in migrated.facts)
    assert "legacy-unverified" in json.dumps(migrated.prompt_view())


def test_canonical_facts_are_authoritative_and_migration_preserves_relationship_source() -> None:
    memory = _profiles()
    memory.agent = ["Stale projection must not become a fact"]
    proof = FactEvidence(message_id="rapport", sender_type="other", quote="Good talking to you")
    memory.relationship = RelationshipState(stage="familiar", evidence=proof.quote)
    memory.relationship_source = proof
    migrated = migrate_legacy(memory)
    assert migrated.model_dump_json() == memory.model_dump_json()
    assert migrated.relationship_source == proof
    assert all("Stale" not in fact.text for fact in migrated.facts)
    migrated.facts[0].text = "detached mutation"
    assert memory.facts[0].text == "I'm 29"


def test_noop_and_unrelated_add_preserve_exact_ages_jobs_preferences_and_stories() -> None:
    memory = _profiles()
    before = memory.model_dump_json()
    noop = merge_memory_delta(memory, MemoryDelta(), [])
    assert noop.model_dump_json() == before and noop is not memory
    updated = _merge_one(memory, _operation("I prefer green tea."))
    assert updated.facts[:-1] == memory.facts
    assert memory.model_dump_json() == before
    assert updated.relationship == memory.relationship


def test_age_replacement_changes_only_the_correct_speaker_target() -> None:
    memory = _profiles()
    target = memory.facts[0]
    quote = "I’m 30"
    updated = _merge_one(memory, _operation(quote, kind="age", action="replace", target=target.id))
    assert updated.facts[1:] == memory.facts[1:]
    fact = updated.facts[0]
    assert fact.id != target.id and fact.text == quote
    assert fact.evidence == FactEvidence(message_id="new", sender_type="other", quote=quote)
    assert "I'm 29" not in updated.interlocutor
    assert memory.facts[0] == target


@pytest.mark.parametrize("quote", ["I am 29", "I'm 29", "I’m 29", "29 years old", "age: 29", "I am 29 years old."])
def test_minimal_age_declarations_are_supported(quote: str) -> None:
    assert _merge_one(None, _operation(quote, kind="age")).facts[0].text == quote


@pytest.mark.parametrize("quote", [
    "Are you 29?", "I am 29?", "29", "He is 29 years old", "My sister is 29 years old",
    "I have 29 plants", "I am 29 plants tall", "I'm 73. Just kidding, I'm 29.",
])
def test_questions_nonself_reports_and_multiclause_age_quotes_fail(quote: str) -> None:
    with pytest.raises(MemoryUpdateError, match="age quote"):
        _merge_one(None, _operation(quote, kind="age"))


def test_corrected_age_clause_is_selected_verbatim_not_inferred_by_regex() -> None:
    quote = "I'm 29"
    updated = merge_memory_delta(None, MemoryDelta(operations=[_operation(quote, kind="age")]), [
        _message("I'm 73. Just kidding, I'm 29."),
    ])
    assert updated.facts[0].text == quote


@pytest.mark.parametrize("messages", [
    [_message("I like tea.", "wrong")], [_message("I like coffee.")],
    [_message("I like tea.", None)], [_message("I like tea.", speaker="me")],
    [_message("I like tea.", speaker="system")],
    [_message("I like tea.", speaker="other"), _message("I like tea.", speaker="me")],
])
def test_false_ids_quotes_missing_ids_wrong_speakers_and_system_sources_fail(messages: list[Message]) -> None:
    with pytest.raises(MemoryUpdateError):
        merge_memory_delta(None, MemoryDelta(operations=[_operation("I like tea.")]), messages)


def test_repeated_id_fragments_resolve_by_exact_quote_and_unambiguous_speaker() -> None:
    quote = "I like tea."
    messages = [_message("unrelated", speaker="me"), _message(quote), _message("prefix " + quote)]
    memory = merge_memory_delta(None, MemoryDelta(operations=[_operation(quote)]), messages)
    assert memory.facts[0].evidence == FactEvidence(message_id="new", sender_type="other", quote=quote)


@pytest.mark.parametrize("operation", [
    _operation("I like tea.", target="forbidden"),
    _operation("I like tea.", action="replace"),
    _operation("I like tea.", action="remove"),
    _operation("I like tea.", action="replace", target="unknown"),
    _operation("I like tea.", action="remove", target="unknown"),
])
def test_operation_target_requirements(operation: MemoryOperation) -> None:
    with pytest.raises(MemoryUpdateError, match="target ID"):
        _merge_one(_profiles(), operation)


def test_repeated_target_and_cross_section_kind_or_speaker_replacements_fail() -> None:
    memory = _profiles()
    target = memory.facts[0]
    operation = _operation("I'm 30", kind="age", action="replace", target=target.id)
    with pytest.raises(MemoryUpdateError, match="repeated target"):
        merge_memory_delta(memory, MemoryDelta(operations=[operation, operation]), [_message("I'm 30")])
    with pytest.raises(MemoryUpdateError, match="section mismatch"):
        _merge_one(memory, _operation("I'm 30", section="agent", kind="age", action="replace", target=target.id), "me")
    with pytest.raises(MemoryUpdateError, match="kind mismatch"):
        _merge_one(memory, _operation("I like tea.", action="replace", target=target.id))
    interaction = _merge_one(None, _operation("I will ask later", section="interaction", kind="commitment"))
    with pytest.raises(MemoryUpdateError, match="speaker mismatch"):
        _merge_one(interaction, _operation("I will ask tomorrow", section="interaction", kind="commitment",
                                        action="replace", target=interaction.facts[0].id), "me")


def test_legacy_other_kind_can_be_corrected_without_fabricating_prior_proof() -> None:
    memory = migrate_legacy(MemoryContent(interlocutor=["Age 73"], agent=["Old job"]))
    updated = _merge_one(memory, _operation("I'm 29", kind="age", action="replace", target=memory.facts[0].id))
    assert updated.facts[0].kind == "age" and updated.facts[0].evidence is not None
    assert updated.facts[1] == memory.facts[1] and updated.facts[1].evidence is None


def test_remove_only_open_questions_or_commitments_with_new_evidence() -> None:
    quote = "I will send the document."
    memory = _merge_one(_profiles(), _operation(quote, section="open_threads", kind="commitment", message_id="promise"))
    target = memory.facts[-1]
    updated = _merge_one(memory, _operation("I sent the document.", section="open_threads", kind="commitment",
                                         action="remove", target=target.id))
    assert updated.facts == memory.facts[:-1] and updated.open_threads == []
    assert target in memory.facts
    with pytest.raises(MemoryUpdateError, match="only open_threads"):
        _merge_one(memory, _operation("I prefer coffee now.", action="remove", target=memory.facts[3].id))
    with pytest.raises(MemoryUpdateError, match="new evidence"):
        _merge_one(memory, _operation(quote, section="open_threads", kind="commitment", action="remove", target=target.id))


def test_open_thread_non_question_non_commitment_cannot_be_removed() -> None:
    memory = _merge_one(None, _operation("A note", section="open_threads", kind="other", message_id="old"))
    with pytest.raises(MemoryUpdateError, match="only open_threads"):
        _merge_one(memory, _operation("Another note", section="open_threads", kind="other", action="remove", target=memory.facts[0].id))


@pytest.mark.parametrize("quote,message_id", [("I'm 29", "fresh"), ("I'm 30", "0"), ("I am 24", "fresh")])
def test_previous_source_cannot_replace_itself(quote: str, message_id: str) -> None:
    memory = _profiles()
    # The third quote comes from a different speaker, so it is fresh by source;
    # semantic attribution cannot be disproved when the new speaker says it too.
    operation = _operation(quote, kind="age", action="replace", target=memory.facts[0].id, message_id=message_id)
    if quote == "I am 24":
        assert _merge_one(memory, operation).facts[0].text == quote
    else:
        with pytest.raises(MemoryUpdateError, match="new evidence"):
            _merge_one(memory, operation)


def test_dedup_replays_preserve_original_provenance_and_generated_ids() -> None:
    memory = _merge_one(None, _operation("I like tea.", message_id="first"))
    fact = memory.facts[0]
    canonical = json.dumps(["interlocutor", "other", "i like tea."], ensure_ascii=False, separators=(",", ":"))
    assert fact.id == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    for quote in ("I like tea.", "I  LIKE\ttea.", "Ｉ like tea."):
        result = _merge_one(memory, _operation(quote, message_id="later"))
        assert result.model_dump_json() == memory.model_dump_json()
    twice = MemoryDelta(operations=[_operation("I like tea."), _operation("I like tea.")])
    assert len(merge_memory_delta(None, twice, [_message("I like tea.")]).facts) == 1
    legacy = migrate_legacy(MemoryContent(interlocutor=["I like tea."]))
    assert _merge_one(legacy, _operation("I like tea.")).facts == legacy.facts
    assert legacy.facts[0].evidence is None


def test_same_quote_from_different_speakers_or_sections_keeps_distinct_ids() -> None:
    quote = "I like tea."
    memory = _merge_one(None, _operation(quote, section="interaction", message_id="other"))
    memory = _merge_one(memory, _operation(quote, section="interaction", message_id="me"), "me")
    memory = _merge_one(memory, _operation(quote, section="agent", message_id="me"), "me")
    assert len({fact.id for fact in memory.facts}) == 3


def test_all_facts_reach_reply_prompt_but_previews_take_only_first_eight() -> None:
    memory: MemoryContent | None = None
    for index in range(12):
        memory = _merge_one(memory, _operation(f"I like trail {index}.", message_id=str(index)))
    assert memory is not None and len(memory.interlocutor) == 8 and len(memory.facts) == 12
    assert memory.interlocutor == [fact.text for fact in memory.facts[:8]]
    assert "trail 11" in json.dumps(memory.prompt_view())
    assert merge_memory_delta(memory, MemoryDelta(), []).facts == memory.facts


def test_capacity_overflow_aborts_without_eviction() -> None:
    memory = MemoryContent(facts=[MemoryFact(id=str(index), section="agent", kind="story", text=f"Story {index}") for index in range(64)])
    before = memory.model_dump_json()
    assert merge_memory_delta(memory, MemoryDelta(), []).model_dump_json() == before
    with pytest.raises(MemoryUpdateError, match="capacity"):
        _merge_one(memory, _operation("A new story", section="agent", kind="story"), "me")
    assert memory.model_dump_json() == before


def test_persisted_utf8_overflow_aborts_without_mutating_previous() -> None:
    # Sixty-four short-in-characters proofs exceed the independent storage byte
    # cap. Previews remain below 6000 characters, so this is not the legacy cap.
    operations = [_operation(f"{index}:" + "🙂" * 130, section="interaction", kind="story", message_id=str(index)) for index in range(64)]
    messages = [_message(operation.quote, operation.message_id) for operation in operations]
    memory = merge_memory_delta(None, MemoryDelta(operations=operations[:32]), messages)
    before = memory.model_dump_json()
    with pytest.raises(MemoryUpdateError, match="capacity") as error:
        merge_memory_delta(memory, MemoryDelta(operations=operations[32:]), messages)
    assert error.value.__cause__ is not None and "48000 UTF-8 bytes" in str(error.value.__cause__)
    assert memory.model_dump_json() == before


def test_late_bad_operation_is_all_or_none_for_memory_delta_and_messages() -> None:
    memory = _profiles()
    delta = MemoryDelta(operations=[_operation("I like tea."), _operation("Invented quote", message_id="absent")])
    messages = [_message("I like tea.")]
    before = (memory.model_dump_json(), delta.model_dump_json(), messages[0].model_dump_json())
    with pytest.raises(MemoryUpdateError):
        merge_memory_delta(memory, delta, messages)
    assert before == (memory.model_dump_json(), delta.model_dump_json(), messages[0].model_dump_json())


def test_relationship_changes_are_sourced_and_failure_discards_other_operations() -> None:
    memory = _profiles()
    quote = "I enjoy our chats."
    delta = MemoryDelta(relationship=RelationshipChange(stage="familiar", message_id="rapport", quote=quote))
    result = merge_memory_delta(memory, delta, [_message(quote, "rapport")])
    assert result.facts == memory.facts
    assert result.relationship == RelationshipState(stage="familiar", evidence=quote)
    assert result.relationship_source == FactEvidence(message_id="rapport", sender_type="other", quote=quote)
    assert merge_memory_delta(result, delta, [_message(quote, "rapport")]) == result
    assert merge_memory_delta(result, MemoryDelta(), []) == result
    before = memory.model_dump_json()
    delta.operations = [_operation("I like tea.")]
    with pytest.raises(MemoryUpdateError):
        merge_memory_delta(memory, delta, [_message("I like tea.")])
    assert memory.model_dump_json() == before
    with pytest.raises(MemoryUpdateError):
        merge_memory_delta(memory, delta, [_message(quote, "rapport", "system"), _message("I like tea.")])
    with pytest.raises(MemoryUpdateError, match="new evidence"):
        merge_memory_delta(result, MemoryDelta(relationship=RelationshipChange(stage="strained", message_id="rapport", quote=quote)),
                           [_message(quote, "rapport")])


def test_superseded_quote_bytes_are_replaced_not_copied_and_roundtrip_is_lossless() -> None:
    memory = _merge_one(None, _operation('I like "café".', message_id="old"))
    quote = 'I prefer "茶".\nReally.'
    result = _merge_one(memory, _operation(quote, action="replace", target=memory.facts[0].id))
    encoded = result.model_dump_json().encode("utf-8")
    assert "café".encode("utf-8") not in encoded
    restored = MemoryContent.model_validate_json(encoded)
    assert restored.facts[0].text.encode("utf-8") == quote.encode("utf-8")
    assert restored.facts[0].evidence is not None and restored.facts[0].evidence.quote == quote


@pytest.mark.parametrize("model,payload", [
    (MemoryOperation, {**_operation("quote").model_dump(), "text": "Model paraphrase"}),
    (MemoryOperation, {**_operation("quote").model_dump(), "id": "invented"}),
    (RelationshipChange, {"stage": "familiar", "message_id": "1", "quote": "quote", "evidence": "invented"}),
    (MemoryDelta, {"facts": []}),
    (MemoryDelta, {"operations": [_operation("quote").model_dump()] * 41}),
    (MemoryOperation, {**_operation("quote").model_dump(), "message_id": " "}),
    (MemoryOperation, {**_operation("quote").model_dump(), "quote": " "}),
    (MemoryOperation, {**_operation("quote").model_dump(), "quote": "x" * 201}),
])
def test_internal_contract_forbids_free_prose_extra_fields_and_out_of_bounds_values(
    model: type[BaseModel], payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_exact_source_does_not_prove_semantic_truth_or_extraction_completeness() -> None:
    # A misclassified quote stays a quote; this layer cannot infer an occupation
    # or promise that every useful detail was extracted from supplied messages.
    quote = "My friend is a pilot."
    memory = merge_memory_delta(None, MemoryDelta(operations=[_operation(quote, kind="occupation")]), [
        _message(quote + " I design gardens."),
    ])
    assert memory.facts[0].text == quote
    assert "They are a pilot" not in json.dumps(memory.prompt_view())
    assert len(memory.facts) == 1


def test_other_speaker_can_answer_an_open_question_and_last_fact_stays_removed() -> None:
    memory = _merge_one(None, _operation("What time does it start?", section="open_threads", kind="question", message_id="old"))
    result = _merge_one(memory, _operation("It starts at noon.", section="open_threads", kind="question",
                                        action="remove", target=memory.facts[0].id), "me")
    assert result.facts == [] and result.open_threads == []
    assert migrate_legacy(result).facts == []


def test_profile_removal_and_mutated_invalid_fact_fail_without_changes() -> None:
    memory = _profiles()
    with pytest.raises(MemoryUpdateError, match="only open_threads"):
        _merge_one(memory, _operation("I'm 30", kind="age", action="remove", target=memory.facts[0].id))
    memory.facts[0].text = "Paraphrase with a mismatched source"
    before = memory.model_dump_json()
    with pytest.raises(MemoryUpdateError):
        merge_memory_delta(memory, MemoryDelta(), [])
    assert memory.model_dump_json() == before


def test_preview_overflow_fails_instead_of_dropping_or_truncating_facts() -> None:
    facts = [MemoryFact(id=f"{section}-{index}", section=section, kind="other", text="x" * 200)
             for section in FACT_SECTIONS for index in range(8)]
    memory = MemoryContent(facts=facts)
    before = memory.model_dump_json()
    with pytest.raises(MemoryUpdateError) as error:
        _merge_one(memory, _operation("New note"))
    assert error.value.__cause__ is not None and "6000 characters" in str(error.value.__cause__)
    assert memory.model_dump_json() == before


def test_different_targets_cannot_be_replaced_by_one_colliding_fact() -> None:
    memory = _merge_one(None, _operation("I like tea.", message_id="tea"))
    memory = _merge_one(memory, _operation("I like coffee.", message_id="coffee"))
    delta = MemoryDelta(operations=[_operation("I like juice.", action="replace", target=fact.id) for fact in memory.facts])
    before = memory.model_dump_json()
    with pytest.raises(MemoryUpdateError, match="replacement conflicts"):
        merge_memory_delta(memory, delta, [_message("I like juice.")])
    assert memory.model_dump_json() == before


def test_add_cannot_resurrect_a_replaced_fact_in_same_delta() -> None:
    memory = _merge_one(None, _operation("I like tea.", message_id="old"))
    delta = MemoryDelta(operations=[
        _operation("I like coffee.", action="replace", target=memory.facts[0].id),
        _operation("I like tea.", message_id="replay"),
    ])
    before = memory.model_dump_json()
    with pytest.raises(MemoryUpdateError, match="add conflicts"):
        merge_memory_delta(memory, delta, [_message("I like coffee."), _message("I like tea.", "replay")])
    assert memory.model_dump_json() == before


def test_operation_and_relationship_defaults_are_independent() -> None:
    first, second = MemoryDelta(), MemoryDelta()
    first.operations.append(_operation("A quote"))
    assert second.operations == [] and second.relationship is None
    assert _operation("A quote").target_id is None