"""Bounded memory validation and optional workspace mirror compatibility checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from responser_model_api import schemas
from responser_model_api.schemas import (
    ChatDescriptor,
    ChatSnapshot,
    ConversationContext,
    FactEvidence,
    MemoryContent,
    MemoryFact,
    Message,
    RelationshipState,
    SummarizeContextRequest,
    SummarizeContextResponse,
)


def _checkpoint() -> ConversationContext:
    return ConversationContext(
        memory=MemoryContent(interlocutor=["Enjoys hiking."], agent=["Agent said they enjoy chess."]),
        last_message_id="123", summarized_message_count=30,
        model_name="qwen2.5:7b", updated_at="2026-09-12T12:00:00Z",
    )


def test_memory_defaults_are_independent_and_greetings_can_be_empty() -> None:
    first, second = MemoryContent(), MemoryContent()
    first.agent.append("Agent said hello.")
    assert second.agent == []
    assert second.relationship == RelationshipState(stage="unknown", evidence="")
    snapshot = ChatSnapshot(chat=ChatDescriptor(raw_id="1", title="Test"))
    assert snapshot.context is None


@pytest.mark.parametrize("field", ["interlocutor", "agent", "interaction", "open_threads"])
@pytest.mark.parametrize("value", [[""], [" \n "], ["x" * 201], ["fact"] * 9, [1], None])
def test_memory_list_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MemoryContent.model_validate({field: value})


def test_memory_total_size_boundary_and_json_escaping() -> None:
    values = {name: ["x" * 180] * 8 for name in (
        "interlocutor", "agent", "interaction", "open_threads",
    )}
    base = MemoryContent.model_validate(values)
    excluded = {"facts", "relationship_source"}
    remaining = schemas.MAX_MEMORY_CHARACTERS - len(base.model_dump_json(exclude=excluded))
    assert 0 < remaining < 300
    payload = {**values, "relationship": {"evidence": "x" * remaining}}
    assert len(MemoryContent.model_validate(payload).model_dump_json(exclude=excluded)) == 6000
    payload["relationship"] = {"evidence": "x" * (remaining + 1)}
    with pytest.raises(ValidationError, match="6000 characters"):
        MemoryContent.model_validate(payload)
    # Escaped quotes consume two serialized characters, not one.
    with pytest.raises(ValidationError, match="6000 characters"):
        MemoryContent(interlocutor=['"' * 200] * 8, agent=['"' * 200] * 8)


@pytest.mark.parametrize("field", ["last_message_id", "model_name", "updated_at"])
@pytest.mark.parametrize("value", ["", "  ", None])
def test_checkpoint_strings_must_be_nonempty(field: str, value: object) -> None:
    payload = _checkpoint().model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        ConversationContext.model_validate(payload)


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_checkpoint_count_is_positive_integer(count: float) -> None:
    with pytest.raises(ValidationError):
        ConversationContext.model_validate({**_checkpoint().model_dump(), "summarized_message_count": count})


@pytest.mark.parametrize("model,payload", [
    (FactEvidence, {"message_id": "1", "sender_type": "other", "quote": "Hi", "extra": 1}),
    (MemoryFact, {"id": "1", "section": "agent", "kind": "other", "text": "Hi", "extra": 1}),
    (RelationshipState, {"extra": 1}),
    (MemoryContent, {"extra": 1}),
    (ConversationContext, {**_checkpoint().model_dump(), "extra": 1}),
    (SummarizeContextRequest, {"messages": [{"sender_type": "other", "text": "hi"}], "extra": 1}),
    (SummarizeContextResponse, {"memory": {}, "model_name": "local", "extra": 1}),
])
def test_new_models_forbid_extra_fields(model: type[BaseModel], payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        model.model_validate(payload)


@pytest.fixture
def mirror(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load only the mirrored file, never require/install the reader package."""
    path = Path(__file__).resolve().parents[2] / "responser_web_reader/src/responser_web_reader/schemas.py"
    if not path.is_file():
        pytest.skip("reader mirror is not present in this standalone checkout")
    spec = importlib.util.spec_from_file_location("_reader_schema_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [
    "Message", "ChatDescriptor", "ChatSnapshot", "GenerateReplyRequest", "GeneratedReply",
    "FactEvidence", "MemoryFact", "RelationshipState", "MemoryContent", "ConversationContext",
    "SummarizeContextRequest", "SummarizeContextResponse",
])
def test_json_schema_matches_reader_mirror(mirror: ModuleType, name: str) -> None:
    api_model = cast(type[BaseModel], getattr(schemas, name))
    reader_model = cast(type[BaseModel], getattr(mirror, name))
    assert api_model.model_json_schema() == reader_model.model_json_schema()


def test_mirrored_source_and_payloads_are_identical(mirror: ModuleType) -> None:
    assert schemas.__file__ is not None and mirror.__file__ is not None
    api_source = Path(schemas.__file__).read_text(encoding="utf-8")
    reader_source = Path(mirror.__file__).read_text(encoding="utf-8")
    assert api_source.split("from __future__", 1)[1] == reader_source.split("from __future__", 1)[1]
    snapshot = ChatSnapshot(chat=ChatDescriptor(raw_id="1", title="Test"), context=_checkpoint())
    reader_snapshot = mirror.ChatSnapshot.model_validate_json(snapshot.model_dump_json())
    assert reader_snapshot.model_dump_json() == snapshot.model_dump_json()
    request = {"previous": _checkpoint().memory.model_dump(), "messages": [{"sender_type": "other", "text": "hi"}]}
    api_request = SummarizeContextRequest.model_validate(request)
    assert mirror.SummarizeContextRequest.model_validate_json(api_request.model_dump_json()).model_dump_json() == api_request.model_dump_json()
    response = SummarizeContextResponse(memory=_checkpoint().memory, model_name="qwen2.5:7b")
    assert mirror.SummarizeContextResponse.model_validate_json(response.model_dump_json()).model_dump_json() == response.model_dump_json()
    assert json.loads(reader_snapshot.model_dump_json())["context"]["summarized_message_count"] == 30


def _fact(index: int = 0) -> MemoryFact:
    quote = f"I like trail {index}."
    return MemoryFact(
        id=f"fact-{index}", section="interlocutor", kind="preference", text=quote,
        evidence=FactEvidence(message_id=f"source-{index}", sender_type="other", quote=quote),
    )


@pytest.mark.parametrize("field,value", [
    ("message_id", ""), ("message_id", " \n"), ("message_id", "x" * 129),
    ("sender_type", "system"), ("quote", ""), ("quote", "  "), ("quote", "x" * 201),
])
def test_evidence_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        FactEvidence.model_validate({"message_id": "1", "sender_type": "other", "quote": "Hi", field: value})


@pytest.mark.parametrize("field,value", [
    ("id", ""), ("id", "  "), ("id", "x" * 129), ("section", "unknown"),
    ("kind", "diagnosis"), ("text", ""), ("text", " \n"), ("text", "x" * 201),
])
def test_fact_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MemoryFact.model_validate({**_fact().model_dump(), field: value})


def test_fact_and_relationship_evidence_must_match_exactly() -> None:
    fact = _fact()
    with pytest.raises(ValidationError, match="text must equal"):
        MemoryFact.model_validate({**fact.model_dump(), "text": "Inferred personality trait"})
    with pytest.raises(ValidationError, match="relationship_source.quote"):
        MemoryContent(relationship=RelationshipState(evidence="Paraphrase"), relationship_source=fact.evidence)
    with pytest.raises(ValidationError, match="duplicate fact IDs"):
        MemoryContent(facts=[fact, fact])
    with pytest.raises(ValidationError):
        MemoryContent(facts=[_fact(index) for index in range(65)])


def test_prompt_and_summary_views_never_duplicate_previews_or_proofs() -> None:
    legacy = MemoryContent(agent=["Unverified story"])
    expected = legacy.model_dump(mode="json", exclude={"facts", "relationship_source"})
    assert legacy.prompt_view() == legacy.summary_view() == expected
    facts = [_fact(index) for index in range(12)]
    facts.append(MemoryFact(id="unverified-record-id", section="agent", kind="other", text="Old story"))
    memory = MemoryContent(facts=facts, interlocutor=["STALE_PREVIEW"])
    prompt = memory.prompt_view()
    assert isinstance(prompt["interlocutor"], list) and len(prompt["interlocutor"]) == 12
    prompt_json = json.dumps(prompt)
    for fact in facts:
        assert fact.id not in prompt_json
    assert "other [preference]" in prompt_json and "legacy-unverified" in prompt_json
    assert "source-" not in prompt_json and "STALE_PREVIEW" not in prompt_json
    summary = memory.summary_view()
    assert set(summary) == {"facts", "relationship"}
    assert summary["facts"] == [
        {"id": fact.id, "section": fact.section, "kind": fact.kind, "text": fact.text}
        for fact in facts
    ]


def test_full_memory_utf8_boundary_is_separate_from_legacy_previews() -> None:
    facts = [MemoryFact(id=str(index), section="agent", kind="story", text="🙂" * 150) for index in range(64)]
    unchecked = MemoryContent.model_construct(facts=facts)
    remaining = schemas.MAX_PERSISTED_MEMORY_BYTES - len(unchecked.model_dump_json().encode("utf-8"))
    assert remaining > 0
    for fact in facts:
        count = min(200 - len(fact.text), remaining // 4)
        fact.text += "🙂" * count
        remaining -= count * 4
        if 0 < remaining < 4 and len(fact.text) + remaining <= 200:
            fact.text += "x" * remaining
            remaining = 0
    assert remaining == 0
    valid = MemoryContent(facts=facts)
    assert len(valid.model_dump_json().encode("utf-8")) == 48_000
    assert len(valid.model_dump_json(exclude={"facts", "relationship_source"})) < 6000
    next(fact for fact in facts if len(fact.text) < 200).text += "x"
    with pytest.raises(ValidationError, match="48000 UTF-8 bytes"):
        MemoryContent(facts=facts)


def test_inference_payload_uses_exact_compact_registry_and_absolute_labels() -> None:
    memory = MemoryContent(facts=[_fact()], interlocutor=["IGNORED_PREVIEW"])
    request = SummarizeContextRequest(previous=memory, messages=[
        Message(sender_type="me", text="hello", raw_id="1", timestamp="now"),
        Message(sender_type="other", text="reply", raw_id="2"),
        Message(sender_type="system", text="service"),
    ])
    payload = json.loads(request.inference_payload())
    assert payload == {"previous": memory.summary_view(), "messages": [
        {"speaker": "AGENT", "text": "hello", "message_id": "1", "timestamp": "now"},
        {"speaker": "INTERLOCUTOR", "text": "reply", "message_id": "2", "timestamp": None},
        {"speaker": "SERVICE_EVENT", "text": "service", "message_id": None, "timestamp": None},
    ]}
    assert "IGNORED_PREVIEW" not in request.inference_payload()


def test_projected_input_byte_boundary_and_separate_wire_cap() -> None:
    request = SummarizeContextRequest(messages=[Message(sender_type="other", text="")])
    overhead = len(request.inference_payload().encode("utf-8"))
    request.messages[0].text = "x" * (12_000 - overhead - 4) + "🙂"
    assert len(request.inference_payload().encode("utf-8")) == 12_000
    SummarizeContextRequest.model_validate_json(request.model_dump_json())
    request.messages[0].text += "x"
    with pytest.raises(ValidationError, match="12000 UTF-8 bytes"):
        SummarizeContextRequest.model_validate_json(request.model_dump_json())
    request.messages[0].text = "x" * 64_000
    with pytest.raises(ValidationError, match="64000 UTF-8 bytes"):
        SummarizeContextRequest.model_validate_json(request.model_dump_json())


def test_proofs_can_exceed_old_wire_budget_without_bloating_inference() -> None:
    facts = [MemoryFact(
        id=str(index), section="interaction", kind="story", text="🙂" * 100,
        evidence=FactEvidence(message_id=f"source-{index}", sender_type="other", quote="🙂" * 100),
    ) for index in range(18)]
    request = SummarizeContextRequest(previous=MemoryContent(facts=facts), messages=[Message(sender_type="me", text="hi")])
    assert 12_000 < len(request.model_dump_json().encode("utf-8")) < 64_000
    assert len(request.inference_payload().encode("utf-8")) < 12_000


def test_fact_wire_roundtrip_and_projections_match_mirror(mirror: ModuleType) -> None:
    fact = _fact()
    assert fact.evidence is not None
    memory = MemoryContent(
        facts=[fact], relationship=RelationshipState(stage="new", evidence=fact.text),
        relationship_source=fact.evidence,
    )
    reader_memory = mirror.MemoryContent.model_validate_json(memory.model_dump_json())
    assert reader_memory.model_dump_json() == memory.model_dump_json()
    assert reader_memory.prompt_view() == memory.prompt_view()
    assert reader_memory.summary_view() == memory.summary_view()
    request = SummarizeContextRequest(previous=memory, messages=[Message(sender_type="other", text="hi", raw_id="new")])
    reader_request = mirror.SummarizeContextRequest.model_validate_json(request.model_dump_json())
    assert reader_request.model_dump_json() == request.model_dump_json()
    assert reader_request.inference_payload() == request.inference_payload()
    for constant in ("MAX_MEMORY_CHARACTERS", "MAX_SUMMARY_INPUT_BYTES", "MAX_PERSISTED_MEMORY_BYTES", "MAX_SUMMARY_WIRE_BYTES"):
        assert getattr(schemas, constant) == getattr(mirror, constant)