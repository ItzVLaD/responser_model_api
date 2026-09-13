"""Formatting-only citation recovery must preserve exact source evidence."""

from __future__ import annotations

import pytest

from responser_model_api.memory_updates import (
    MemoryDelta, MemoryOperation, MemoryUpdateError, RelationshipChange, merge_memory_delta,
)
from responser_model_api.schemas import FactEvidence, MemoryContent, Message
from responser_model_api.source_quotes import formatting_equivalent_spans


def _operation(quote: str, message_id: str = "source") -> MemoryOperation:
    return MemoryOperation(
        action="add", section="interlocutor", kind="preference", message_id=message_id, quote=quote,
    )


@pytest.mark.parametrize("source,proposed,expected", [
    ("I’m 23", "I'm 23", "I’m 23"),
    ("I prefer ‘quiet’ places.", "I prefer 'quiet' places.", "I prefer ‘quiet’ places."),
    ('I prefer “quiet” places.', 'I prefer "quiet" places.', 'I prefer “quiet” places.'),
    ("I design\n\ngardens.", "I design gardens.", "I design\n\ngardens."),
    ("I design\r\ngardens.", "I design gardens.", "I design\r\ngardens."),
    ("I prefer\u00a0tea.", "I prefer tea.", "I prefer\u00a0tea."),
    ("Before. I\tlike   tea. After.", "I like tea.", "I\tlike   tea."),
    ("I like tea.", "I\nlike\ttea.", "I like tea."),
    ("Before\t  \t  After.", "Before  After.", "Before\t  \t  After."),
    ("I\tlike tea. I\tlike tea.", "I like tea.", "I\tlike tea."),
])
def test_typography_recovery_stores_original_source_substring(
    source: str, proposed: str, expected: str,
) -> None:
    operation = _operation(proposed)
    delta = MemoryDelta(operations=[operation])
    message = Message(raw_id="source", sender_type="other", text=source)
    before = (delta.model_dump_json(), message.model_dump_json())
    memory = merge_memory_delta(None, delta, [message])
    fact = memory.facts[0]
    assert fact.text == expected
    assert fact.evidence == FactEvidence(message_id="source", sender_type="other", quote=expected)
    assert expected in source
    assert before == (delta.model_dump_json(), message.model_dump_json())


@pytest.mark.parametrize("source,proposed", [
    ("I'm 23.", "I'm 24."),
    ("I like tea.", "I dislike tea."),
    ("I don't like tea.", "I like tea."),
    ("I like Tea.", "I like tea."),
    ("I work as a designer.", "I'm a designer."),
    ("My name is Ann.", "My name is Ana."),
    ("I like tea, not coffee.", "I like tea coffee."),
    ("I prefer tea.", "I prefer tea" + "." * 3),
    ("I have 123\tplants.", "23 plants."),
    ("NotI\tlike tea.", "I like tea."),
    ("I\tlike teapots.", "I like tea"),
    ("I like café.", "I like cafe."),
    ("My code is A-B.", "My code is A—B."),
    ("I'm\t23.5 years old.", "I'm 23"),
    ("I own\t2,000 books.", "I own 2"),
    ("I'm\t23–24 years old.", "I'm 23"),
    ("I bought 2.5\tmetres.", "5 metres."),
])
def test_substantive_changes_are_not_repaired(source: str, proposed: str) -> None:
    with pytest.raises(MemoryUpdateError) as error:
        merge_memory_delta(None, MemoryDelta(operations=[_operation(proposed)]), [
            Message(raw_id="source", sender_type="other", text=source),
        ])
    assert error.value.reason == "citation_quote_mismatch"


def test_recovery_does_not_search_another_message() -> None:
    with pytest.raises(MemoryUpdateError) as error:
        merge_memory_delta(None, MemoryDelta(operations=[_operation("I'm 23")]), [
            Message(raw_id="source", sender_type="other", text="I like tea."),
            Message(raw_id="different", sender_type="other", text="I’m 23"),
        ])
    assert error.value.reason == "citation_quote_mismatch"


def test_distinct_matching_spans_are_ambiguous() -> None:
    # Both satisfy the normalized form, but there is no unique original span.
    with pytest.raises(MemoryUpdateError) as error:
        merge_memory_delta(None, MemoryDelta(operations=[_operation("I like tea.")]), [
            Message(raw_id="source", sender_type="other", text="I\tlike tea. I  like tea."),
        ])
    assert error.value.reason == "citation_quote_ambiguous"


def test_repeated_id_fragments_still_need_unambiguous_evidence() -> None:
    with pytest.raises(MemoryUpdateError) as error:
        merge_memory_delta(None, MemoryDelta(operations=[_operation("I like tea.")]), [
            Message(raw_id="source", sender_type="other", text="I\tlike tea."),
            Message(raw_id="source", sender_type="other", text="I  like tea."),
        ])
    assert error.value.reason == "citation_quote_ambiguous"


def test_cross_speaker_normalized_match_does_not_resolve_attribution() -> None:
    with pytest.raises(MemoryUpdateError) as error:
        merge_memory_delta(None, MemoryDelta(operations=[_operation("I'm 23")]), [
            Message(raw_id="source", sender_type="other", text="I’m 23"),
            Message(raw_id="source", sender_type="me", text="I‘m 23"),
        ])
    assert error.value.reason == "citation_speaker_ambiguous"


def test_recovered_relationship_and_fact_share_exact_original_evidence() -> None:
    source = "I’m glad we talked."
    proposed = "I'm glad we talked."
    memory = merge_memory_delta(None, MemoryDelta(
        operations=[_operation(proposed)],
        relationship=RelationshipChange(stage="acquaintance", message_id="source", quote=proposed),
    ), [Message(raw_id="source", sender_type="other", text=source)])
    assert memory.relationship.evidence == source
    assert memory.relationship_source == memory.facts[0].evidence


def test_recovered_then_verbatim_replay_keeps_original_provenance() -> None:
    source = Message(raw_id="source", sender_type="other", text="I’m 23")
    memory = merge_memory_delta(None, MemoryDelta(operations=[_operation("I'm 23")]), [source])
    replay = merge_memory_delta(memory, MemoryDelta(operations=[_operation(source.text)]), [source])
    assert replay == memory


def test_late_bad_quote_discards_earlier_recovered_quote() -> None:
    previous = MemoryContent(agent=["Existing story"])
    before = previous.model_dump_json()
    with pytest.raises(MemoryUpdateError):
        merge_memory_delta(previous, MemoryDelta(operations=[
            _operation("I'm 23"), _operation("Invented fact"),
        ]), [Message(raw_id="source", sender_type="other", text="I’m 23")])
    assert previous.model_dump_json() == before


def test_recovery_does_not_truncate_oversized_source_span() -> None:
    # Proposed collapsed whitespace fits 200 characters, its source span does not.
    with pytest.raises(MemoryUpdateError):
        merge_memory_delta(None, MemoryDelta(operations=[_operation("I like tea.")]), [
            Message(raw_id="source", sender_type="other", text="I" + " " * 210 + "like tea."),
        ])


@pytest.mark.parametrize("speaker,reason", [
    ("me", "profile_speaker_mismatch"), ("system", "citation_service_event"),
])
def test_recovery_keeps_speaker_restrictions(speaker: str, reason: str) -> None:
    message = Message.model_validate({"raw_id": "source", "sender_type": speaker, "text": "I’m 23"})
    with pytest.raises(MemoryUpdateError) as error:
        merge_memory_delta(None, MemoryDelta(operations=[_operation("I'm 23")]), [message])
    assert error.value.reason == reason


def test_recovered_age_is_checked_using_original_text() -> None:
    operation = _operation("I'm 23 and enjoying life!")
    operation.kind = "age"
    source = Message(raw_id="source", sender_type="other", text="I’m 23\nand enjoying life!")
    memory = merge_memory_delta(None, MemoryDelta(operations=[operation]), [source])
    assert memory.facts[0].text == source.text and memory.facts[0].kind == "age"


def test_recovered_replacement_still_requires_fresh_evidence() -> None:
    message = Message(raw_id="source", sender_type="other", text="I’m fond of tea.")
    previous = merge_memory_delta(None, MemoryDelta(operations=[_operation(message.text)]), [message])
    operation = _operation("I'm fond of tea.")
    operation.action = "replace"
    operation.target_id = previous.facts[0].id
    with pytest.raises(MemoryUpdateError) as error:
        merge_memory_delta(previous, MemoryDelta(operations=[operation]), [message])
    assert error.value.reason == "target_evidence_not_new"


def test_recovered_relationship_replay_preserves_prior_source() -> None:
    message = Message(raw_id="source", sender_type="other", text="I’m glad we talked.")
    change = RelationshipChange(stage="acquaintance", message_id="source", quote=message.text)
    memory = merge_memory_delta(None, MemoryDelta(relationship=change), [message])
    change.quote = "I'm glad we talked."
    change.message_id = "later"
    later = Message(raw_id="later", sender_type="other", text=message.text)
    assert merge_memory_delta(memory, MemoryDelta(relationship=change), [later]) == memory
    change.stage = "familiar"
    with pytest.raises(MemoryUpdateError) as error:
        merge_memory_delta(memory, MemoryDelta(relationship=change), [later])
    assert error.value.reason == "relationship_evidence_not_new"


@pytest.mark.parametrize("quote", ["", " ", "\n\t"])
def test_empty_normalized_quote_matches_nothing(quote: str) -> None:
    assert formatting_equivalent_spans("Original source", quote) == set()